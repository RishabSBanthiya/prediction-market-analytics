"""
Delta-Neutral Arbitrage Strategy - Backtest

Core logic: Buy BOTH UP and DOWN tokens when combined cost < $1.00.
Guaranteed profit at resolution since one outcome pays $1.

Only 3 parameters:
- min_edge_bps: minimum edge in basis points (30-100)
- order_size_pct: order size as % of capital (0.05-0.20)
- max_positions: maximum concurrent positions (3-10)

Run backtest:
    python -m polymarket.backtesting.strategies.arb_backtest --backtest

Run with custom params:
    python -m polymarket.backtesting.strategies.arb_backtest --backtest --min-edge 50
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from ...core.models import Market, Token, HistoricalPrice
from ...core.config import get_config
from ...core.api import PolymarketAPI
from ..results import BacktestResults, SimulatedTrade
from ..data.cached_fetcher import CachedDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class ArbParams:
    """Arb strategy parameters."""
    min_edge_bps: int = 50  # Minimum edge in basis points
    order_size_pct: float = 0.10  # Order size as % of capital
    max_positions: int = 5  # Max concurrent positions
    fee_bps: int = 100  # 1% fee estimate (taker fee)


@dataclass
class ArbPosition:
    """Tracks a delta-neutral position (both sides)."""
    market_id: str
    question: str
    up_token_id: str
    down_token_id: str
    up_shares: float
    down_shares: float
    up_price: float
    down_price: float
    total_cost: float
    entry_time: datetime
    edge_bps: int


class ArbBacktester:
    """
    Backtest for delta-neutral arbitrage strategy.

    Looks for opportunities where:
    - UP_price + DOWN_price < $1.00 (after fees)
    - Buying both sides guarantees profit at resolution
    """

    def __init__(
        self,
        params: ArbParams,
        initial_capital: float = 1000.0,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.fee_multiplier = 1 + (params.fee_bps / 10000)

        self.cash = initial_capital
        self.positions: List[ArbPosition] = []
        self._price_cache: Dict[str, List[HistoricalPrice]] = {}

    def set_price_cache(self, cache: Dict[str, List[Dict]]):
        """Set pre-fetched price history."""
        for token_id, prices in cache.items():
            if prices:
                converted = []
                for p in prices:
                    ts = p.get('t') or p.get('timestamp')
                    price = p.get('p') or p.get('price')
                    if ts and price:
                        converted.append(HistoricalPrice(
                            timestamp=int(ts),
                            price=float(price)
                        ))
                if converted:
                    # Sort by timestamp
                    converted.sort(key=lambda x: x.timestamp)
                    self._price_cache[token_id] = converted

    def run_sync(self, markets: List[Market]) -> BacktestResults:
        """Synchronous backtest with cached data."""
        return asyncio.run(self.run(markets))

    async def run(self, markets: List[Market]) -> BacktestResults:
        """Run backtest on given markets."""
        start_date = datetime.now(timezone.utc) - timedelta(days=30)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"Arb ({self.params.min_edge_bps}bps min)",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital
        self.positions = []

        opportunities_found = 0
        total_edge_captured = 0.0

        for market in markets:
            opp = self._check_market_for_arb(market)
            if opp:
                opportunities_found += 1
                position, trade_info = self._execute_arb(market, opp, results)
                if position:
                    total_edge_captured += position.edge_bps
                    # Immediately resolve (since we're backtesting closed markets)
                    self._resolve_position(position, market, results)

        # Log summary
        logger.info(f"Arb Backtest Complete:")
        logger.info(f"  Markets analyzed: {len(markets)}")
        logger.info(f"  Opportunities found: {opportunities_found}")
        logger.info(f"  Avg edge: {total_edge_captured/max(1,opportunities_found):.0f} bps")

        results.finalize()
        return results

    def _check_market_for_arb(
        self,
        market: Market,
    ) -> Optional[Dict]:
        """
        Check if a market had arbitrage opportunity.

        For each timestamp where we have prices for BOTH tokens,
        check if combined ask < $1.00.
        """
        if len(market.tokens) != 2:
            return None

        up_token = market.tokens[0]
        down_token = market.tokens[1]

        up_history = self._price_cache.get(up_token.token_id, [])
        down_history = self._price_cache.get(down_token.token_id, [])

        if len(up_history) < 10 or len(down_history) < 10:
            return None

        # Build timestamp-aligned price pairs
        # Use a dict for fast lookup
        down_prices = {p.timestamp: p.price for p in down_history}

        best_opportunity = None
        best_edge = 0

        for up_point in up_history:
            # Find closest down price within 60 seconds
            closest_down = None
            min_diff = float('inf')

            for ts, price in down_prices.items():
                diff = abs(ts - up_point.timestamp)
                if diff < min_diff and diff <= 60:
                    min_diff = diff
                    closest_down = price

            if closest_down is None:
                continue

            up_price = up_point.price
            down_price = closest_down

            # Calculate buy cost with fees
            buy_cost = (up_price + down_price) * self.fee_multiplier

            if buy_cost < 1.0:
                edge = 1.0 - buy_cost
                edge_bps = int(edge * 10000)

                if edge_bps >= self.params.min_edge_bps and edge_bps > best_edge:
                    best_edge = edge_bps
                    best_opportunity = {
                        'timestamp': up_point.timestamp,
                        'up_price': up_price,
                        'down_price': down_price,
                        'buy_cost': buy_cost,
                        'edge_bps': edge_bps,
                        'guaranteed_profit_pct': (1.0 / buy_cost - 1) * 100,
                    }

        return best_opportunity

    def _execute_arb(
        self,
        market: Market,
        opportunity: Dict,
        results: BacktestResults,
    ) -> Tuple[Optional[ArbPosition], Optional[Dict]]:
        """Execute an arbitrage trade (buy both sides)."""

        # Check position limits
        if len(self.positions) >= self.params.max_positions:
            return None, None

        # Calculate position size
        position_dollars = self.cash * self.params.order_size_pct

        if position_dollars < 5.0:  # Min trade size
            return None, None

        # Calculate shares (equal dollar amount on each side)
        up_price = opportunity['up_price']
        down_price = opportunity['down_price']
        buy_cost = opportunity['buy_cost']

        # Total cost per pair = up_price + down_price (+ fees)
        cost_per_pair = buy_cost
        num_pairs = position_dollars / cost_per_pair

        up_shares = num_pairs
        down_shares = num_pairs

        total_cost = cost_per_pair * num_pairs

        if total_cost > self.cash:
            return None, None

        # Execute
        self.cash -= total_cost

        entry_time = datetime.fromtimestamp(
            opportunity['timestamp'],
            tz=timezone.utc
        )

        position = ArbPosition(
            market_id=market.condition_id,
            question=market.question,
            up_token_id=market.tokens[0].token_id,
            down_token_id=market.tokens[1].token_id,
            up_shares=up_shares,
            down_shares=down_shares,
            up_price=up_price,
            down_price=down_price,
            total_cost=total_cost,
            entry_time=entry_time,
            edge_bps=opportunity['edge_bps'],
        )

        self.positions.append(position)

        logger.debug(
            f"ARB: {market.question[:40]}... | "
            f"UP={up_price:.3f} DOWN={down_price:.3f} | "
            f"cost=${total_cost:.2f} edge={opportunity['edge_bps']}bp"
        )

        return position, opportunity

    def _resolve_position(
        self,
        position: ArbPosition,
        market: Market,
        results: BacktestResults,
    ):
        """Resolve position at market end (guaranteed $1 per pair)."""

        # At resolution, we get $1 per share pair
        # (one side wins, pays $1, other side pays $0)
        proceeds = position.up_shares  # 1 share pair = $1 at resolution

        pnl = proceeds - position.total_cost
        pnl_pct = pnl / position.total_cost if position.total_cost > 0 else 0

        self.cash += proceeds

        # Record as a single combined trade
        trade = SimulatedTrade(
            market_question=f"[ARB] {position.question[:90]}",
            token_id=f"{position.up_token_id}+{position.down_token_id}",
            token_outcome="BOTH",
            entry_time=position.entry_time,
            entry_price=position.total_cost / position.up_shares,  # cost per pair
            exit_time=market.end_date,
            exit_price=1.0,  # Always resolves to $1 per pair
            shares=position.up_shares,
            cost=position.total_cost,
            proceeds=proceeds,
            pnl=pnl,
            pnl_percent=pnl_pct,
            resolved_to=1.0,
            held_to_resolution=True,
            reason=f"arb_resolved_{position.edge_bps}bp",
        )

        results.add_trade(trade)

        # Remove from positions
        if position in self.positions:
            self.positions.remove(position)


async def fetch_15min_markets_for_backtest(
    api: PolymarketAPI,
    days: int = 7,
) -> List[Market]:
    """
    Fetch closed 15-minute markets for backtesting.

    These have the predictable slug pattern and fee structure.
    """
    markets = []

    # Fetch recently closed markets
    raw_markets = await api.fetch_closed_markets(days=days)

    for raw in raw_markets:
        market = api.parse_market(raw)
        if not market:
            continue

        # Filter for 15-min crypto markets
        question = market.question.lower()
        is_15min = (
            'up or down' in question and
            any(crypto in question for crypto in ['bitcoin', 'ethereum', 'solana', 'xrp'])
        )

        if is_15min and market.fees_enabled and len(market.tokens) == 2:
            markets.append(market)

    return markets


async def run_backtest(params: ArbParams, days: int = 7, verbose: bool = False):
    """Run the arb backtest."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print("=" * 60)
        print("DELTA-NEUTRAL ARBITRAGE BACKTEST")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  Min Edge:      {params.min_edge_bps} bps ({params.min_edge_bps/100:.2f}%)")
        print(f"  Order Size:    {params.order_size_pct*100:.0f}% of capital")
        print(f"  Max Positions: {params.max_positions}")
        print(f"  Fee Estimate:  {params.fee_bps} bps")
        print(f"  Lookback:      {days} days")
        print("=" * 60)

        # Fetch markets
        print("\nFetching closed 15-min markets...")
        markets = await fetch_15min_markets_for_backtest(api, days=days)
        print(f"Found {len(markets)} closed 15-min markets")

        if not markets:
            print("No markets found for backtesting")
            return None

        # Fetch price history for all tokens
        print("\nFetching price history...")
        fetcher = CachedDataFetcher()
        await fetcher.initialize()

        price_cache = {}
        try:
            for i, market in enumerate(markets):
                for token in market.tokens:
                    if token.token_id not in price_cache:
                        try:
                            history = await fetcher.get_price_history(token.token_id)
                            if history:
                                price_cache[token.token_id] = history
                        except Exception as e:
                            logger.debug(f"Failed to fetch history for {token.token_id}: {e}")

                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(markets)} markets...")

            print(f"Cached prices for {len(price_cache)} tokens")
        finally:
            await fetcher.close()

        # Run backtest
        print("\nRunning backtest...")
        backtester = ArbBacktester(params)
        backtester.set_price_cache(price_cache)

        results = await backtester.run(markets)

        # Print results
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"  Total Trades:    {results.total_trades}")
        print(f"  Win Rate:        {results.win_rate*100:.1f}%")
        print(f"  Total P&L:       ${results.total_pnl:.2f}")
        print(f"  Return:          {results.return_pct*100:.2f}%")
        sharpe = results.sharpe_ratio
        print(f"  Sharpe Ratio:    {sharpe:.2f}" if sharpe else "  Sharpe Ratio:    N/A")
        max_dd = results.max_drawdown
        print(f"  Max Drawdown:    {max_dd*100:.2f}%" if max_dd else "  Max Drawdown:    N/A")
        print("=" * 60)

        if results.trades:
            print("\nTrades:")
            for trade in results.trades[:10]:
                edge = trade.reason.split('_')[-1] if trade.reason else 'N/A'
                print(f"  {trade.market_question[:60]}")
                print(f"    Cost: ${trade.cost:.2f} -> Return: ${trade.proceeds:.2f} | P&L: ${trade.pnl:.2f} ({trade.pnl_percent*100:.2f}%) | Edge: {edge}")

        return results

    finally:
        await api.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Delta-Neutral Arbitrage Backtest"
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest"
    )

    parser.add_argument(
        "--min-edge",
        type=int,
        default=50,
        help="Minimum edge in basis points (default: 50)"
    )

    parser.add_argument(
        "--size",
        type=float,
        default=0.10,
        help="Position size as fraction of capital (default: 0.10)"
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum concurrent positions (default: 5)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days of history to backtest (default: 7)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.backtest:
        params = ArbParams(
            min_edge_bps=args.min_edge,
            order_size_pct=args.size,
            max_positions=args.max_positions,
        )
        asyncio.run(run_backtest(params, days=args.days, verbose=args.verbose))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
