"""
Simplified Bond Strategy - Backtest + Optimization in one file.

Core logic: Buy tokens priced 92-97c near expiry, hold to resolution at $1.

Only 3 parameters:
- entry_price: minimum price to enter (0.92-0.97)
- max_spread_pct: maximum acceptable spread (0.01-0.06)
- max_position_pct: max position as % of capital (0.05-0.20)

Run backtest:
    python -m polymarket.backtesting.strategies.bond_simple --backtest

Run optimization:
    python -m polymarket.backtesting.strategies.bond_simple --optimize
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

from ...core.models import Market, Token, HistoricalPrice
from ...core.config import get_config
from ...core.api import PolymarketAPI
from ..results import BacktestResults, SimulatedTrade
from ..optimization import (
    OptimizationConfigV3,
    BayesianOptimizerV3,
    generate_optimization_summary_v3,
    save_optimization_report_v3,
    BOND_DEFAULTS,
)

logger = logging.getLogger(__name__)


@dataclass
class BondParams:
    """Simple bond parameters - 3 only."""
    entry_price: float = 0.95  # Min price to enter
    max_spread_pct: float = 0.03  # Max spread
    max_position_pct: float = 0.10  # Max position size


class SimpleBondBacktester:
    """
    Minimal bond strategy backtester.

    ~150 lines instead of 900. No hedging, no dynamic time windows,
    no complex exit strategies. Just the core logic.
    """

    def __init__(
        self,
        params: BondParams,
        initial_capital: float = 1000.0,
        slippage_pct: float = 0.005,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct

        self.cash = initial_capital
        self._price_cache: Dict[str, List[HistoricalPrice]] = {}

    def set_price_cache(self, cache: Dict[str, List[Dict]]):
        """Set pre-fetched price history to avoid API calls."""
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
                    self._price_cache[token_id] = converted

    def run_sync(self, markets: List[Market]) -> BacktestResults:
        """Synchronous version for optimization (no async needed with cached data)."""
        start_date = datetime.now(timezone.utc) - timedelta(days=60)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"Bond Simple ({self.params.entry_price:.0%}+)",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital

        for market in markets:
            for token in market.tokens:
                trade = self._process_token(market, token, results)
                if trade:
                    break  # One trade per market

        results.finalize()
        return results

    async def run(self, markets: List[Market]) -> BacktestResults:
        """Run backtest on given markets."""
        start_date = datetime.now(timezone.utc) - timedelta(days=60)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"Bond Simple ({self.params.entry_price:.0%}+)",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital

        for market in markets:
            for token in market.tokens:
                trade = self._process_token(market, token, results)
                if trade:
                    break  # One trade per market

        results.finalize()
        return results

    def _process_token(
        self,
        market: Market,
        token: Token,
        results: BacktestResults,
    ) -> Optional[SimulatedTrade]:
        """Process a single token for trade opportunities."""
        history = self._price_cache.get(token.token_id, [])

        if len(history) < 10:
            return None

        # Only look at last 20% of market life (near expiry)
        start_idx = int(len(history) * 0.8)

        for i in range(start_idx, len(history) - 1):
            point = history[i]

            # Check entry criteria
            if point.price < self.params.entry_price:
                continue
            if point.price > 0.99:
                continue  # Already resolved

            # Estimate spread from recent price volatility
            recent = history[max(0, i-5):i+1]
            if len(recent) >= 2:
                prices = [p.price for p in recent]
                spread = (max(prices) - min(prices)) / point.price
            else:
                spread = 0.02

            if spread > self.params.max_spread_pct:
                continue

            # Calculate position size
            position_dollars = self.cash * self.params.max_position_pct
            if position_dollars < 10:
                continue

            # Execute with slippage
            exec_price = point.price * (1 + self.slippage_pct)
            shares = position_dollars / exec_price
            cost = shares * exec_price

            if cost > self.cash:
                continue

            self.cash -= cost

            # Determine exit price (resolution)
            is_winner = (
                market.winning_outcome and
                token.outcome == market.winning_outcome
            )

            final_price = history[-1].price

            if is_winner or final_price > 0.99:
                exit_price = 1.0
                exit_reason = "Resolved YES"
            elif final_price < 0.01:
                exit_price = 0.0
                exit_reason = "Resolved NO"
            else:
                # Sell at current price with slippage
                exit_price = final_price * (1 - self.slippage_pct)
                exit_reason = "Market close"

            proceeds = shares * exit_price
            self.cash += proceeds
            pnl = proceeds - cost

            trade = SimulatedTrade(
                market_question=market.question[:80],
                token_id=token.token_id,
                token_outcome=token.outcome,
                entry_time=point.datetime,
                entry_price=exec_price,
                exit_time=history[-1].datetime,
                exit_price=exit_price,
                shares=shares,
                cost=cost,
                proceeds=proceeds,
                pnl=pnl,
                pnl_percent=pnl / cost if cost > 0 else 0,
                resolved_to=1.0 if exit_price > 0.99 else (0.0 if exit_price < 0.01 else None),
                held_to_resolution=exit_price > 0.99 or exit_price < 0.01,
                reason=f"Entry @ {point.price:.2%}, {exit_reason}"
            )

            results.add_trade(trade)
            return trade

        return None


async def run_backtest(
    params: Optional[BondParams] = None,
    capital: float = 1000.0,
    days: int = 60,
    verbose: bool = False,
) -> BacktestResults:
    """Run a single backtest with given parameters."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    params = params or BondParams()
    backtester = SimpleBondBacktester(params, capital)

    # Fetch markets
    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print(f"Fetching markets (last {days} days)...")
        raw_markets = await api.fetch_closed_markets(days=days)

        markets = []
        for raw in raw_markets:
            m = api.parse_market(raw)
            if m and m.winning_outcome:
                markets.append(m)

        print(f"Found {len(markets)} resolved markets")

        # Fetch price history
        print("Fetching price history...")
        price_cache = {}
        for market in markets[:200]:  # Limit for speed
            for token in market.tokens:
                history = await api.fetch_price_history(token.token_id)
                if history:
                    price_cache[token.token_id] = [
                        {'t': h.timestamp, 'p': h.price} for h in history
                    ]

        backtester.set_price_cache(price_cache)

        results = await backtester.run(markets[:200])
        results.print_report()

        return results

    finally:
        await api.close()


async def run_optimization(
    n_calls: int = 50,
    days: int = 180,
    capital: float = 1000.0,
) -> None:
    """Run optimization to find best parameters."""
    logging.basicConfig(level=logging.INFO)

    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print(f"Fetching markets (last {days} days)...")
        raw_markets = await api.fetch_closed_markets(days=days)

        # Convert to dicts for optimizer
        markets_data = []
        for raw in raw_markets:
            m = api.parse_market(raw)
            if m and m.winning_outcome:
                markets_data.append({
                    'condition_id': m.condition_id,
                    'question': m.question,
                    'tokens': [{'token_id': t.token_id, 'outcome': t.outcome} for t in m.tokens],
                    'end_date': m.end_date.isoformat() if m.end_date else None,
                    'winning_outcome': m.winning_outcome,
                })

        print(f"Found {len(markets_data)} resolved markets")

        # Fetch price history
        print("Fetching price history...")
        price_cache = {}
        for i, market in enumerate(markets_data[:300]):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{min(300, len(markets_data))}...")
            for token in market['tokens']:
                token_id = token['token_id']
                history = await api.fetch_price_history(token_id)
                if history:
                    price_cache[token_id] = [
                        {'t': h.timestamp, 'p': h.price} for h in history
                    ]

        # SYNC backtest function for optimizer (no async in optimizer loop)
        def backtest_fn(params: Dict, fold_markets: List[Dict]) -> BacktestResults:
            bond_params = BondParams(
                entry_price=params.get('entry_price', 0.95),
                max_spread_pct=params.get('max_spread_pct', 0.03),
                max_position_pct=params.get('max_position_pct', 0.10),
            )

            backtester = SimpleBondBacktester(bond_params, capital)

            # Filter price cache to fold markets
            fold_token_ids = set()
            for m in fold_markets:
                for t in m.get('tokens', []):
                    fold_token_ids.add(t['token_id'])

            fold_cache = {k: v for k, v in price_cache.items() if k in fold_token_ids}
            backtester.set_price_cache(fold_cache)

            # Convert dicts to Market objects
            market_objs = []
            for m in fold_markets:
                tokens = [Token(t['token_id'], t['outcome']) for t in m.get('tokens', [])]
                end_date_str = m.get('end_date')
                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        end_date = datetime.now(timezone.utc)
                else:
                    end_date = datetime.now(timezone.utc)
                market_objs.append(Market(
                    condition_id=m['condition_id'],
                    question=m['question'],
                    slug=m.get('slug', ''),
                    tokens=tokens,
                    end_date=end_date,
                    winning_outcome=m.get('winning_outcome'),
                ))

            return backtester.run_sync(market_objs)

        # Run optimizer (synchronous)
        opt_config = OptimizationConfigV3(
            total_days=days,
            n_calls=n_calls,
            initial_capital=capital,
        )

        optimizer = BayesianOptimizerV3(
            strategy_type="bond",
            config=opt_config,
            backtest_fn=backtest_fn,
            markets=markets_data[:300],
        )

        result = optimizer.optimize()  # Sync call

        # Print results
        summary = generate_optimization_summary_v3(result)
        print("\n" + summary)

        # Save report
        save_optimization_report_v3(result)

    finally:
        await api.close()


def main():
    parser = argparse.ArgumentParser(description="Simple Bond Strategy")
    parser.add_argument('--backtest', action='store_true', help='Run single backtest')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--entry-price', type=float, default=0.95, help='Entry price threshold')
    parser.add_argument('--max-spread', type=float, default=0.03, help='Max spread')
    parser.add_argument('--max-position', type=float, default=0.10, help='Max position pct')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.optimize:
        asyncio.run(run_optimization(
            n_calls=args.iterations,
            days=args.days,
            capital=args.capital,
        ))
    else:
        # Default to backtest
        params = BondParams(
            entry_price=args.entry_price,
            max_spread_pct=args.max_spread,
            max_position_pct=args.max_position,
        )
        asyncio.run(run_backtest(
            params=params,
            capital=args.capital,
            days=args.days,
            verbose=args.verbose,
        ))


if __name__ == "__main__":
    main()
