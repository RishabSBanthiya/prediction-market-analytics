"""
Simplified Flow Strategy - Backtest + Optimization in one file.

Core logic: Detect price momentum, enter on breakout, exit on TP/SL.

Only 3 parameters:
- take_profit_pct: target profit to exit (0.03-0.15)
- stop_loss_pct: max loss before exit (0.04-0.20)
- max_position_pct: max position as % of capital (0.05-0.20)

Run backtest:
    python -m polymarket.backtesting.strategies.flow_simple --backtest

Run optimization:
    python -m polymarket.backtesting.strategies.flow_simple --optimize
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

# Fixed parameters (not optimized)
MOMENTUM_THRESHOLD = 0.02  # 2% price move triggers signal
MIN_PRICE = 0.20  # Avoid unlikely outcomes
MAX_PRICE = 0.80  # Avoid limited upside
MAX_HOLD_BARS = 30  # Time-based exit (30 price points)
SLIPPAGE_PCT = 0.005  # 0.5% slippage


@dataclass
class FlowParams:
    """Simple flow parameters - 3 only."""
    take_profit_pct: float = 0.06  # Exit at 6% profit
    stop_loss_pct: float = 0.08  # Exit at 8% loss
    max_position_pct: float = 0.10  # Max position size


class SimpleFlowBacktester:
    """
    Minimal flow strategy backtester.

    ~200 lines instead of 2000. No wallet profiling, no 10+ signal types,
    no complex hedging. Just momentum detection + TP/SL exits.
    """

    def __init__(
        self,
        params: FlowParams,
        initial_capital: float = 1000.0,
    ):
        self.params = params
        self.initial_capital = initial_capital
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
            strategy_name=f"Flow Simple (TP:{self.params.take_profit_pct:.0%}/SL:{self.params.stop_loss_pct:.0%})",
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
            strategy_name=f"Flow Simple (TP:{self.params.take_profit_pct:.0%}/SL:{self.params.stop_loss_pct:.0%})",
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

    def _detect_momentum(
        self,
        history: List[HistoricalPrice],
        idx: int,
        lookback: int = 5,
    ) -> Tuple[bool, str]:
        """
        Detect price momentum signal.

        Returns: (has_signal, direction)
        """
        if idx < lookback:
            return False, "NONE"

        current_price = history[idx].price
        past_price = history[idx - lookback].price

        if past_price <= 0:
            return False, "NONE"

        change = (current_price - past_price) / past_price

        if change >= MOMENTUM_THRESHOLD:
            return True, "BUY"
        elif change <= -MOMENTUM_THRESHOLD:
            return True, "SELL"

        return False, "NONE"

    def _process_token(
        self,
        market: Market,
        token: Token,
        results: BacktestResults,
    ) -> Optional[SimulatedTrade]:
        """Process a single token for trade opportunities."""
        history = self._price_cache.get(token.token_id, [])

        if len(history) < 20:
            return None

        # Scan for momentum signals
        for i in range(10, len(history) - MAX_HOLD_BARS - 1):
            current_price = history[i].price

            # Price filter
            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                continue

            has_signal, direction = self._detect_momentum(history, i)

            if not has_signal:
                continue

            # For SELL signals, we'd need to trade opposite token
            # For simplicity, only trade BUY signals on this token
            if direction != "BUY":
                continue

            # Position sizing
            position_dollars = self.cash * self.params.max_position_pct
            if position_dollars < 10:
                continue

            # Execute entry with slippage
            entry_price = current_price * (1 + SLIPPAGE_PCT)
            shares = position_dollars / entry_price
            cost = shares * entry_price

            if cost > self.cash:
                continue

            self.cash -= cost

            # Simulate position through remaining history
            exit_price, exit_idx, exit_reason = self._simulate_exit(
                history, i, entry_price
            )

            # Apply exit slippage
            exit_price_slipped = exit_price * (1 - SLIPPAGE_PCT)
            proceeds = shares * exit_price_slipped
            self.cash += proceeds
            pnl = proceeds - cost

            trade = SimulatedTrade(
                market_question=market.question[:80],
                token_id=token.token_id,
                token_outcome=token.outcome,
                entry_time=history[i].datetime,
                entry_price=entry_price,
                exit_time=history[exit_idx].datetime,
                exit_price=exit_price_slipped,
                shares=shares,
                cost=cost,
                proceeds=proceeds,
                pnl=pnl,
                pnl_percent=pnl / cost if cost > 0 else 0,
                resolved_to=None,
                held_to_resolution=False,
                reason=f"Momentum {direction}, {exit_reason}"
            )

            results.add_trade(trade)
            return trade  # One trade per token

        return None

    def _simulate_exit(
        self,
        history: List[HistoricalPrice],
        entry_idx: int,
        entry_price: float,
    ) -> Tuple[float, int, str]:
        """
        Simulate exit based on TP/SL/time.

        Returns: (exit_price, exit_idx, reason)
        """
        tp_price = entry_price * (1 + self.params.take_profit_pct)
        sl_price = entry_price * (1 - self.params.stop_loss_pct)

        for j in range(entry_idx + 1, min(entry_idx + MAX_HOLD_BARS, len(history))):
            current_price = history[j].price

            # Take profit
            if current_price >= tp_price:
                return current_price, j, f"TP @ {self.params.take_profit_pct:.0%}"

            # Stop loss
            if current_price <= sl_price:
                return current_price, j, f"SL @ {self.params.stop_loss_pct:.0%}"

        # Time-based exit
        exit_idx = min(entry_idx + MAX_HOLD_BARS, len(history) - 1)
        return history[exit_idx].price, exit_idx, "Time exit"


async def run_backtest(
    params: Optional[FlowParams] = None,
    capital: float = 1000.0,
    days: int = 60,
    verbose: bool = False,
) -> BacktestResults:
    """Run a single backtest with given parameters."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    params = params or FlowParams()
    backtester = SimpleFlowBacktester(params, capital)

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
            if m:
                markets.append(m)

        print(f"Found {len(markets)} markets")

        # Fetch price history
        print("Fetching price history...")
        price_cache = {}
        for market in markets[:200]:
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
                take_profit_pct=params.get('take_profit_pct', 0.06),
                stop_loss_pct=params.get('stop_loss_pct', 0.08),
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
    parser = argparse.ArgumentParser(description="Simple Flow Strategy")
    parser.add_argument('--backtest', action='store_true', help='Run single backtest')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--take-profit', type=float, default=0.06, help='Take profit pct')
    parser.add_argument('--stop-loss', type=float, default=0.08, help='Stop loss pct')
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
        params = FlowParams(
            take_profit_pct=args.take_profit,
            stop_loss_pct=args.stop_loss,
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
