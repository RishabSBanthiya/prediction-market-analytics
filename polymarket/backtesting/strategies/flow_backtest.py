"""
Flow Strategy V6 - Optimized Parameters from Real Alert Backtest.

V6 OPTIMIZATIONS (based on analysis of 3,524 real flow alerts):
- Stop Loss: 20% (wider to avoid whipsaws)
- Take Profit: 50% (let winners run)
- Price Range: 50c-90c (mid-range has best win rate)
- Alert Type: SMART_MONEY_ACTIVITY has 75% win rate
- Severity: HIGH outperforms CRITICAL

IMPORTANT: This backtest uses price patterns as a PROXY for smart money flow.
The live flow strategy uses actual trade data (wallet addresses, trade sizes)
from the RTDS WebSocket to identify and follow smart money.

FOR ACCURATE BACKTESTING with real flow alerts:
Use flow_trade_backtest.py which uses recorded alerts from risk_state.db.

Run backtest:
    python -m polymarket.backtesting.strategies.flow_backtest --backtest

Run optimization:
    python -m polymarket.backtesting.strategies.flow_backtest --optimize
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
    FLOW_DEFAULTS,
)

logger = logging.getLogger(__name__)

# V5: Sports market keywords - skip these (efficiently priced)
SPORTS_KEYWORDS = [
    "nba", "nfl", "nhl", "mlb", "cfb", "cbb", "mls", "ufc", "pga",
    "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
    "champions league", "world cup", "euro 2024", "olympics",
    "wimbledon", "us open", "french open", "australian open",
    "super bowl", "world series", "stanley cup", "nba finals",
    "march madness", "college football", "college basketball",
    " vs ", " v ", "game ", "match ", "fight ", "bout ",
]

# V5: Category-specific parameters
CATEGORY_PARAMS = {
    "crypto": {
        "momentum_threshold": 0.03,  # 3% for volatile crypto
        "take_profit_mult": 1.2,     # Higher TP for crypto
        "stop_loss_mult": 1.0,
        "position_mult": 0.8,        # Smaller positions
    },
    "politics": {
        "momentum_threshold": 0.02,
        "take_profit_mult": 1.5,     # Higher TP, longer hold
        "stop_loss_mult": 1.2,
        "position_mult": 1.2,        # Can size up on politics
    },
    "finance": {
        "momentum_threshold": 0.025,
        "take_profit_mult": 1.0,
        "stop_loss_mult": 1.0,
        "position_mult": 1.0,
    },
    "other": {
        "momentum_threshold": 0.02,
        "take_profit_mult": 1.0,
        "stop_loss_mult": 1.0,
        "position_mult": 0.7,        # Conservative on unknown
    },
}

# Fixed parameters (not optimized) - V6
LOOKBACK_BARS = 10  # Bars to check for trend
SLIPPAGE_PCT = 0.01  # 1% slippage


def is_sports_market(question: str) -> bool:
    """Check if market is a sports betting market."""
    if not question:
        return False
    q_lower = question.lower()
    return any(keyword in q_lower for keyword in SPORTS_KEYWORDS)


def get_category(question: str) -> str:
    """Determine market category from question text."""
    if not question:
        return "other"
    q_lower = question.lower()

    # Crypto keywords
    crypto_kw = ["btc", "bitcoin", "eth", "ethereum", "crypto", "token",
                 "solana", "sol", "xrp", "doge", "bnb", "cardano", "ada"]
    if any(kw in q_lower for kw in crypto_kw):
        return "crypto"

    # Politics keywords
    politics_kw = ["president", "election", "congress", "senate", "vote",
                   "trump", "biden", "republican", "democrat", "governor"]
    if any(kw in q_lower for kw in politics_kw):
        return "politics"

    # Finance keywords
    finance_kw = ["fed", "interest rate", "inflation", "gdp", "stock",
                  "s&p", "nasdaq", "dow", "economy", "treasury"]
    if any(kw in q_lower for kw in finance_kw):
        return "finance"

    return "other"


@dataclass
class FlowParams:
    """V6 Flow parameters - optimized from real flow alert backtest."""
    # V6 OPTIMIZED: Entry price range 50c-90c has best win rate
    min_entry_price: float = 0.50  # Min price to buy (V6: 50c, was 0)
    max_entry_price: float = 0.90  # Max price to buy (V6: 90c)
    # V6 OPTIMIZED: Wider stops, higher targets
    stop_loss_pct: float = 0.20  # V6: 20% stop loss (was 10%)
    take_profit_pct: float = 0.50  # V6: 50% take profit (was 8%)
    # Position sizing
    max_position_pct: float = 0.10  # Max position size (V6: 10%)


class SimpleFlowBacktester:
    """
    V6 Flow Strategy Backtester - Optimized from Real Alert Analysis.

    V6 Optimizations:
    - Entry price range: 50c-90c (mid-range has best win rate)
    - Stop Loss: 20% (wider to avoid whipsaws)
    - Take Profit: 50% (let winners run)
    - Sports market filtering

    Based on analysis of 3,524 real SMART_MONEY_ACTIVITY alerts.
    """

    def __init__(
        self,
        params: FlowParams,
        initial_capital: float = 1000.0,
        filter_sports: bool = True,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.filter_sports = filter_sports
        self._price_cache: Dict[str, List[HistoricalPrice]] = {}

        # V5: Track filter stats
        self._filter_stats = {
            "sports_filtered": 0,
            "resolved_filtered": 0,
            "price_range_filtered": 0,
            "processed": 0,
            "traded": 0,
        }
        self._category_stats: Dict[str, int] = {"crypto": 0, "politics": 0, "finance": 0, "other": 0}

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
            strategy_name=f"Flow V6 (Price {self.params.min_entry_price:.0%}-{self.params.max_entry_price:.0%}, SL {self.params.stop_loss_pct:.0%}, TP {self.params.take_profit_pct:.0%})",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital

        # Reset filter stats
        for key in self._filter_stats:
            self._filter_stats[key] = 0
        for key in self._category_stats:
            self._category_stats[key] = 0

        for market in markets:
            self._filter_stats["processed"] += 1

            # V5: Filter sports markets
            if self.filter_sports and is_sports_market(market.question):
                self._filter_stats["sports_filtered"] += 1
                continue

            # Get category for this market
            category = get_category(market.question)
            self._category_stats[category] += 1

            for token in market.tokens:
                trade = self._process_token(market, token, results, category)
                if trade:
                    self._filter_stats["traded"] += 1
                    break  # One trade per market

        results.finalize()
        return results

    async def run(self, markets: List[Market]) -> BacktestResults:
        """Run backtest on given markets."""
        start_date = datetime.now(timezone.utc) - timedelta(days=60)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"Flow V6 (Price {self.params.min_entry_price:.0%}-{self.params.max_entry_price:.0%}, SL {self.params.stop_loss_pct:.0%}, TP {self.params.take_profit_pct:.0%})",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital

        # Reset filter stats
        for key in self._filter_stats:
            self._filter_stats[key] = 0
        for key in self._category_stats:
            self._category_stats[key] = 0

        for market in markets:
            self._filter_stats["processed"] += 1

            # V5: Filter sports markets
            if self.filter_sports and is_sports_market(market.question):
                self._filter_stats["sports_filtered"] += 1
                continue

            # Get category for this market
            category = get_category(market.question)
            self._category_stats[category] += 1

            for token in market.tokens:
                trade = self._process_token(market, token, results, category)
                if trade:
                    self._filter_stats["traded"] += 1
                    break  # One trade per market

        results.finalize()
        return results

    def get_filter_stats(self) -> Dict:
        """Get filter statistics."""
        return {
            "filter_stats": self._filter_stats.copy(),
            "category_stats": self._category_stats.copy(),
        }

    def _detect_trend(
        self,
        history: List[HistoricalPrice],
        idx: int,
        lookback: int = LOOKBACK_BARS,
    ) -> Tuple[float, str]:
        """
        Detect price trend direction and strength.

        Returns: (trend_strength, direction)
        - trend_strength: absolute price change over lookback
        - direction: "UP" or "DOWN" or "FLAT"
        """
        if idx < lookback:
            return 0.0, "FLAT"

        current_price = history[idx].price
        past_price = history[idx - lookback].price

        if past_price <= 0:
            return 0.0, "FLAT"

        change = (current_price - past_price) / past_price

        if change > 0.005:  # 0.5% threshold for direction
            return abs(change), "UP"
        elif change < -0.005:
            return abs(change), "DOWN"

        return abs(change), "FLAT"

    def _process_token(
        self,
        market: Market,
        token: Token,
        results: BacktestResults,
        category: str = "other",
    ) -> Optional[SimulatedTrade]:
        """
        V6: Process token with optimized entry price range and TP/SL exits.

        Entry criteria:
        1. Price in range [min_entry_price, max_entry_price] (50c-90c optimal)
        2. Not already resolved (price not near 0 or 1)

        Exit: Stop Loss (-20%) or Take Profit (+50%) or Resolution
        """
        history = self._price_cache.get(token.token_id, [])

        if len(history) < 20:
            return None

        # V6: Get category-specific position multiplier only
        cat_params = CATEGORY_PARAMS.get(category, CATEGORY_PARAMS["other"])
        position_mult = cat_params["position_mult"]

        # Get resolution price (last price in history)
        resolution_price = history[-1].price

        # Scan for entry opportunities
        # Start at 10% into history, end at 70% (need room for exits)
        start_idx = max(LOOKBACK_BARS + 1, len(history) // 10)
        end_idx = int(len(history) * 0.7)

        for i in range(start_idx, end_idx):
            current_price = history[i].price

            # V6: Entry price range filter (50c-90c optimal based on backtest)
            if current_price < self.params.min_entry_price:
                self._filter_stats["price_range_filtered"] += 1
                continue
            if current_price > self.params.max_entry_price:
                self._filter_stats["price_range_filtered"] += 1
                continue

            # Skip if already resolved (price near extremes)
            if current_price >= 0.95 or current_price <= 0.05:
                self._filter_stats["resolved_filtered"] += 1
                continue

            # Position sizing with category multiplier
            base_position = self.cash * self.params.max_position_pct
            position_dollars = base_position * position_mult
            if position_dollars < 10:
                continue

            # Execute entry with slippage
            entry_price = current_price * (1 + SLIPPAGE_PCT)
            shares = position_dollars / entry_price
            cost = shares * entry_price

            if cost > self.cash:
                continue

            self.cash -= cost

            # V6: Scan forward for TP/SL exit or hold to resolution
            stop_loss_price = entry_price * (1 - self.params.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.params.take_profit_pct)

            exit_idx = len(history) - 1
            exit_price = resolution_price
            exit_reason = "Resolution"

            # Scan for TP/SL triggers
            for j in range(i + 1, len(history)):
                price = history[j].price

                # Check stop loss
                if price <= stop_loss_price:
                    exit_idx = j
                    exit_price = stop_loss_price * (1 - SLIPPAGE_PCT)
                    exit_reason = f"Stop Loss @ -{self.params.stop_loss_pct:.0%}"
                    break

                # Check take profit
                if price >= take_profit_price:
                    exit_idx = j
                    exit_price = take_profit_price * (1 - SLIPPAGE_PCT)
                    exit_reason = f"Take Profit @ +{self.params.take_profit_pct:.0%}"
                    break

            # If no TP/SL, exit at resolution
            if exit_reason == "Resolution":
                exit_price = resolution_price * (1 - SLIPPAGE_PCT)
                if resolution_price >= 0.90:
                    exit_reason = "Resolved YES ($1)"
                elif resolution_price <= 0.10:
                    exit_reason = "Resolved NO ($0)"
                else:
                    exit_reason = f"Exit @ {resolution_price:.2f}"

            proceeds = shares * exit_price
            self.cash += proceeds
            pnl = proceeds - cost

            trade = SimulatedTrade(
                market_question=market.question[:80],
                token_id=token.token_id,
                token_outcome=token.outcome,
                entry_time=history[i].datetime,
                entry_price=entry_price,
                exit_time=history[exit_idx].datetime,
                exit_price=exit_price,
                shares=shares,
                cost=cost,
                proceeds=proceeds,
                pnl=pnl,
                pnl_percent=pnl / cost if cost > 0 else 0,
                resolved_to=resolution_price,
                held_to_resolution=(exit_reason.startswith("Resolved") or exit_reason.startswith("Exit")),
                reason=f"[{category}] Entry @ {entry_price:.2f}, {exit_reason}"
            )

            results.add_trade(trade)
            return trade  # One trade per token

        return None

async def run_backtest(
    params: Optional[FlowParams] = None,
    capital: float = 1000.0,
    days: int = 60,
    verbose: bool = False,
    filter_sports: bool = True,
) -> BacktestResults:
    """Run a single backtest with given parameters and V6 filters."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    params = params or FlowParams()
    backtester = SimpleFlowBacktester(params, capital, filter_sports=filter_sports)

    # Fetch markets
    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print(f"\n{'='*60}")
        print(f"FLOW STRATEGY V6 BACKTEST (Optimized)")
        print(f"{'='*60}")
        print(f"V6 Optimized Parameters:")
        print(f"  Entry Price Range: {params.min_entry_price:.0%} - {params.max_entry_price:.0%}")
        print(f"  Stop Loss:         -{params.stop_loss_pct:.0%}")
        print(f"  Take Profit:       +{params.take_profit_pct:.0%}")
        print(f"  Max Position:      {params.max_position_pct:.0%}")
        print(f"  Filter Sports:     {'ON' if filter_sports else 'OFF'}")
        print(f"{'='*60}\n")

        print(f"Fetching markets (last {days} days)...")
        raw_markets = await api.fetch_closed_markets(days=days)

        markets = []
        for raw in raw_markets:
            m = api.parse_market(raw)
            if m:
                markets.append(m)

        print(f"Found {len(markets)} markets")

        # Fetch price history
        print("Fetching price history...")
        price_cache = {}
        market_limit = 300  # Increased for better stats
        for i, market in enumerate(markets[:market_limit]):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{min(market_limit, len(markets))}...")
            for token in market.tokens:
                history = await api.fetch_price_history(token.token_id)
                if history:
                    price_cache[token.token_id] = [
                        {'t': h.timestamp, 'p': h.price} for h in history
                    ]

        backtester.set_price_cache(price_cache)

        results = await backtester.run(markets[:market_limit])

        # Print V6 filter statistics
        stats = backtester.get_filter_stats()
        print(f"\n{'='*60}")
        print(f"V6 FILTER STATISTICS")
        print(f"{'='*60}")
        print(f"  Markets Processed:    {stats['filter_stats']['processed']}")
        print(f"  Sports Filtered:      {stats['filter_stats']['sports_filtered']}")
        print(f"  Resolved Filtered:    {stats['filter_stats']['resolved_filtered']}")
        print(f"  Markets Traded:       {stats['filter_stats']['traded']}")
        print(f"\n  Category Breakdown:")
        for cat, count in stats['category_stats'].items():
            print(f"    {cat.capitalize():12} {count}")
        print(f"{'='*60}\n")

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
            if m:
                markets_data.append({
                    'condition_id': m.condition_id,
                    'question': m.question,
                    'tokens': [{'token_id': t.token_id, 'outcome': t.outcome} for t in m.tokens],
                    'end_date': m.end_date.isoformat() if m.end_date else None,
                })

        print(f"Found {len(markets_data)} markets")

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
            flow_params = FlowParams(
                min_entry_price=params.get('min_entry_price', 0.50),
                max_entry_price=params.get('max_entry_price', 0.90),
                stop_loss_pct=params.get('stop_loss_pct', 0.20),
                take_profit_pct=params.get('take_profit_pct', 0.50),
                max_position_pct=params.get('max_position_pct', 0.10),
            )

            backtester = SimpleFlowBacktester(flow_params, capital)

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
                ))

            return backtester.run_sync(market_objs)

        # Run optimizer (synchronous)
        opt_config = OptimizationConfigV3(
            total_days=days,
            n_calls=n_calls,
            initial_capital=capital,
        )

        optimizer = BayesianOptimizerV3(
            strategy_type="flow",
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
    parser = argparse.ArgumentParser(description="Flow Strategy V6 - Optimized from Real Alert Backtest")
    parser.add_argument('--backtest', action='store_true', help='Run single backtest')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='Optimization iterations')
    # V6 Optimized Parameters
    parser.add_argument('--min-entry-price', type=float, default=0.50, help='Min entry price (V6: 50c)')
    parser.add_argument('--max-entry-price', type=float, default=0.90, help='Max entry price (V6: 90c)')
    parser.add_argument('--stop-loss', type=float, default=0.20, help='Stop loss pct (V6: 20%%)')
    parser.add_argument('--take-profit', type=float, default=0.50, help='Take profit pct (V6: 50%%)')
    parser.add_argument('--max-position', type=float, default=0.10, help='Max position pct (V6: 10%%)')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.optimize:
        asyncio.run(run_optimization(
            n_calls=args.iterations,
            days=args.days,
            capital=args.capital,
        ))
    else:
        # Default to backtest with V6 optimized params
        params = FlowParams(
            min_entry_price=args.min_entry_price,
            max_entry_price=args.max_entry_price,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
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