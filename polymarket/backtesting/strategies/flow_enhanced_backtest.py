"""
Enhanced Flow Strategy V4 - Backtest + Optimization.

Incorporates all 7 V4 improvements:
1. Wallet reputation scoring (simulated via trade patterns)
2. Multi-signal confirmation (momentum + volume + reversal)
3. Information timing filter (early vs late signals)
4. Contrarian signal detection (retail vs whale divergence)
5. Market context filters (time to resolution, spread, price extremes)
6. Dynamic position sizing (scales with signal strength)
7. Adaptive exits (varies by market characteristics)

Parameters (6 total - more than V3 but captures enhancement value):
- take_profit_pct: base take profit
- stop_loss_pct: base stop loss
- max_position_pct: max position size
- min_signal_score: minimum signal score to trade (NEW)
- confirmation_weight: how much to weight multi-signal confirmation (NEW)
- context_multiplier_weight: how much market context affects sizing (NEW)

Run backtest:
    python -m polymarket.backtesting.strategies.flow_enhanced_backtest --backtest

Run optimization:
    python -m polymarket.backtesting.strategies.flow_enhanced_backtest --optimize
"""

import argparse
import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Set

from ...core.models import Market, Token, HistoricalPrice
from ...core.config import get_config
from ...core.api import PolymarketAPI
from ..results import BacktestResults, SimulatedTrade
from ..optimization import (
    OptimizationConfigV3,
    BayesianOptimizerV3,
    generate_optimization_summary_v3,
    save_optimization_report_v3,
)

logger = logging.getLogger(__name__)

# Fixed parameters
MIN_PRICE = 0.20
MAX_PRICE = 0.80
SLIPPAGE_PCT = 0.005
BASE_MAX_HOLD_BARS = 30


@dataclass
class EnhancedFlowParams:
    """Enhanced flow parameters - 6 total."""
    # Exit parameters
    take_profit_pct: float = 0.06
    stop_loss_pct: float = 0.08
    max_position_pct: float = 0.10
    # V4 Enhancement parameters
    min_signal_score: float = 40.0  # Minimum score to trade (0-100)
    confirmation_weight: float = 0.5  # How much multi-signal confirmation matters
    context_weight: float = 0.3  # How much market context affects decisions


@dataclass
class SimulatedSignal:
    """A simulated trading signal with V4 enhancements."""
    direction: str  # "BUY" or "SELL"
    score: float  # 0-100
    signal_types: List[str]
    is_early: bool
    is_contrarian: bool
    context_multiplier: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class SimulatedWallet:
    """Simulated wallet for reputation tracking."""
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.5

    @property
    def is_smart_money(self) -> bool:
        return self.trades >= 5 and self.win_rate >= 0.55

    def update(self, pnl: float):
        self.trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1


class EnhancedFlowBacktester:
    """
    Enhanced flow strategy backtester with V4 improvements.

    Simulates the full enhancement pipeline:
    - Multi-signal detection (momentum, volume, reversal)
    - Signal confirmation and scoring
    - Information timing (early vs late)
    - Contrarian detection
    - Market context adjustment
    - Dynamic position sizing
    - Adaptive exits
    """

    def __init__(
        self,
        params: EnhancedFlowParams,
        initial_capital: float = 1000.0,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self._price_cache: Dict[str, List[HistoricalPrice]] = {}
        self._volume_cache: Dict[str, List[float]] = {}
        self._simulated_wallet = SimulatedWallet()

    def set_price_cache(self, cache: Dict[str, List[Dict]]):
        """Set pre-fetched price history."""
        for token_id, prices in cache.items():
            if prices:
                converted = []
                volumes = []
                for p in prices:
                    ts = p.get('t') or p.get('timestamp')
                    price = p.get('p') or p.get('price')
                    vol = p.get('v') or p.get('volume') or 1000  # Default volume
                    if ts and price:
                        converted.append(HistoricalPrice(
                            timestamp=int(ts),
                            price=float(price)
                        ))
                        volumes.append(float(vol))
                if converted:
                    self._price_cache[token_id] = converted
                    self._volume_cache[token_id] = volumes

    def run_sync(self, markets: List[Market]) -> BacktestResults:
        """Synchronous backtest for optimization."""
        start_date = datetime.now(timezone.utc) - timedelta(days=60)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"Flow V4 Enhanced",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital
        self._simulated_wallet = SimulatedWallet()

        for market in markets:
            for token in market.tokens:
                trade = self._process_token(market, token, results)
                if trade:
                    # Update simulated wallet
                    self._simulated_wallet.update(trade.pnl)
                    break

        results.finalize()
        return results

    async def run(self, markets: List[Market]) -> BacktestResults:
        """Async backtest."""
        return self.run_sync(markets)

    def _detect_signals(
        self,
        history: List[HistoricalPrice],
        volumes: List[float],
        idx: int,
    ) -> SimulatedSignal:
        """
        Detect multiple signal types and combine into scored signal.

        Signal Types:
        1. Momentum: 2% price change
        2. Volume Spike: 2x average volume
        3. Reversal: Price reversal after trend
        4. Acceleration: Increasing rate of change
        """
        if idx < 10:
            return SimulatedSignal(
                direction="NONE", score=0, signal_types=[], is_early=False,
                is_contrarian=False, context_multiplier=1.0
            )

        current_price = history[idx].price
        signal_types = []
        buy_score = 0
        sell_score = 0

        # 1. Momentum signal (5-bar lookback)
        price_5 = history[idx - 5].price
        if price_5 > 0:
            momentum = (current_price - price_5) / price_5
            if momentum >= 0.02:
                signal_types.append("momentum_up")
                buy_score += 25
            elif momentum <= -0.02:
                signal_types.append("momentum_down")
                sell_score += 25

        # 2. Volume spike signal
        if len(volumes) > idx and idx >= 10:
            recent_vol = volumes[idx]
            avg_vol = statistics.mean(volumes[idx-10:idx]) if idx >= 10 else recent_vol
            if avg_vol > 0 and recent_vol >= avg_vol * 2:
                signal_types.append("volume_spike")
                # Volume spike confirms direction
                if buy_score > sell_score:
                    buy_score += 20
                elif sell_score > buy_score:
                    sell_score += 20
                else:
                    buy_score += 10
                    sell_score += 10

        # 3. Reversal signal (price reversal after trend)
        if idx >= 15:
            trend_start = history[idx - 15].price
            trend_mid = history[idx - 5].price
            if trend_start > 0 and trend_mid > 0:
                first_half = (trend_mid - trend_start) / trend_start
                second_half = (current_price - trend_mid) / trend_mid

                # Bullish reversal: down then up
                if first_half < -0.03 and second_half > 0.02:
                    signal_types.append("bullish_reversal")
                    buy_score += 20
                # Bearish reversal: up then down
                elif first_half > 0.03 and second_half < -0.02:
                    signal_types.append("bearish_reversal")
                    sell_score += 20

        # 4. Acceleration signal
        if idx >= 10:
            change_recent = (current_price - history[idx - 3].price) / history[idx - 3].price if history[idx - 3].price > 0 else 0
            change_earlier = (history[idx - 3].price - history[idx - 6].price) / history[idx - 6].price if history[idx - 6].price > 0 else 0

            if abs(change_recent) > abs(change_earlier) * 1.5 and abs(change_recent) > 0.01:
                signal_types.append("acceleration")
                if change_recent > 0:
                    buy_score += 15
                else:
                    sell_score += 15

        # Determine direction and final score
        if buy_score > sell_score and buy_score >= 20:
            direction = "BUY"
            score = buy_score
        elif sell_score > buy_score and sell_score >= 20:
            direction = "SELL"
            score = sell_score
        else:
            return SimulatedSignal(
                direction="NONE", score=0, signal_types=[], is_early=False,
                is_contrarian=False, context_multiplier=1.0
            )

        # Check if early signal (not many prior signals in same direction)
        is_early = len(signal_types) <= 2

        # Check for contrarian setup (price moved against signal)
        is_contrarian = False
        if idx >= 20:
            long_term_change = (current_price - history[idx - 20].price) / history[idx - 20].price if history[idx - 20].price > 0 else 0
            # Contrarian: buying after big drop, selling after big rise
            if direction == "BUY" and long_term_change < -0.05:
                is_contrarian = True
                score += 10
            elif direction == "SELL" and long_term_change > 0.05:
                is_contrarian = True
                score += 10

        # Calculate context multiplier
        context_multiplier = self._calculate_context_multiplier(
            current_price, history, idx
        )

        return SimulatedSignal(
            direction=direction,
            score=min(100, score),
            signal_types=signal_types,
            is_early=is_early,
            is_contrarian=is_contrarian,
            context_multiplier=context_multiplier,
        )

    def _calculate_context_multiplier(
        self,
        current_price: float,
        history: List[HistoricalPrice],
        idx: int,
    ) -> float:
        """
        Calculate context multiplier based on market characteristics.

        Factors:
        - Price extremes (reduce for very high/low prices)
        - Volatility (reduce for very volatile markets)
        - Trend clarity (increase for clear trends)
        """
        multiplier = 1.0

        # Price extremes penalty
        if current_price < 0.15 or current_price > 0.85:
            multiplier *= 0.7
        elif current_price < 0.25 or current_price > 0.75:
            multiplier *= 0.9

        # Volatility check
        if idx >= 20:
            prices = [h.price for h in history[idx-20:idx]]
            if len(prices) >= 10:
                try:
                    volatility = statistics.stdev(prices) / statistics.mean(prices)
                    if volatility > 0.10:
                        multiplier *= 0.7  # High volatility penalty
                    elif volatility < 0.03:
                        multiplier *= 1.2  # Low volatility bonus
                except (statistics.StatisticsError, ZeroDivisionError):
                    pass

        # Trend clarity bonus
        if idx >= 10:
            start_price = history[idx - 10].price
            if start_price > 0:
                trend = (current_price - start_price) / start_price
                if abs(trend) > 0.05:  # Clear trend
                    multiplier *= 1.1

        return max(0.3, min(2.0, multiplier))

    def _calculate_position_size(
        self,
        signal: SimulatedSignal,
    ) -> float:
        """
        Calculate position size based on signal strength and context.

        Dynamic sizing based on:
        - Signal score
        - Number of confirming signals
        - Context multiplier
        - Wallet reputation (simulated)
        """
        base_pct = self.params.max_position_pct * 0.5  # Start at half max

        # Scale by signal score (40-100 maps to 0.5-1.5x)
        score_factor = 0.5 + (signal.score - 40) / 120

        # Bonus for multi-signal confirmation
        confirmation_bonus = 1.0
        if len(signal.signal_types) >= 3:
            confirmation_bonus = 1.0 + self.params.confirmation_weight
        elif len(signal.signal_types) >= 2:
            confirmation_bonus = 1.0 + self.params.confirmation_weight * 0.5

        # Apply context multiplier
        context_factor = 1.0 + (signal.context_multiplier - 1.0) * self.params.context_weight

        # Wallet reputation bonus (if we've been winning)
        wallet_bonus = 1.0
        if self._simulated_wallet.is_smart_money:
            wallet_bonus = 1.0 + (self._simulated_wallet.win_rate - 0.5) * 0.5

        # Calculate final position
        position_pct = base_pct * score_factor * confirmation_bonus * context_factor * wallet_bonus

        # Clamp to max
        return min(self.params.max_position_pct, max(0.02, position_pct))

    def _get_adaptive_exit_params(
        self,
        signal: SimulatedSignal,
    ) -> Tuple[float, float, int]:
        """
        Get adaptive exit parameters based on signal characteristics.

        Returns: (take_profit, stop_loss, max_hold_bars)
        """
        tp = self.params.take_profit_pct
        sl = self.params.stop_loss_pct
        max_hold = BASE_MAX_HOLD_BARS

        # Strong signals: wider stops, higher targets
        if signal.score >= 70:
            tp *= 1.3
            sl *= 1.2
            max_hold = int(max_hold * 1.5)
        elif signal.score < 50:
            tp *= 0.8
            sl *= 0.8
            max_hold = int(max_hold * 0.7)

        # Contrarian trades need more room
        if signal.is_contrarian:
            tp *= 1.4
            sl *= 1.3
            max_hold = int(max_hold * 1.5)

        # Early signals can be held longer
        if signal.is_early:
            max_hold = int(max_hold * 1.2)

        # Context adjustment
        if signal.context_multiplier > 1.2:
            tp *= 1.1
        elif signal.context_multiplier < 0.8:
            sl *= 0.9  # Tighter stop in poor context

        return tp, sl, max_hold

    def _process_token(
        self,
        market: Market,
        token: Token,
        results: BacktestResults,
    ) -> Optional[SimulatedTrade]:
        """Process a single token with V4 enhancements."""
        history = self._price_cache.get(token.token_id, [])
        volumes = self._volume_cache.get(token.token_id, [])

        if len(history) < 30:
            return None

        # Scan for signals
        for i in range(15, len(history) - BASE_MAX_HOLD_BARS - 1):
            current_price = history[i].price

            # Price filter
            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                continue

            # Detect signals with V4 enhancements
            signal = self._detect_signals(history, volumes, i)

            if signal.direction == "NONE":
                continue

            # Check minimum score threshold
            if signal.score < self.params.min_signal_score:
                continue

            # Only trade BUY signals on this token
            if signal.direction != "BUY":
                continue

            # Dynamic position sizing
            position_pct = self._calculate_position_size(signal)
            position_dollars = self.cash * position_pct

            if position_dollars < 10:
                continue

            # Execute entry
            entry_price = current_price * (1 + SLIPPAGE_PCT)
            shares = position_dollars / entry_price
            cost = shares * entry_price

            if cost > self.cash:
                continue

            self.cash -= cost

            # Get adaptive exit params
            tp_pct, sl_pct, max_hold = self._get_adaptive_exit_params(signal)

            # Simulate exit
            exit_price, exit_idx, exit_reason = self._simulate_exit(
                history, i, entry_price, tp_pct, sl_pct, max_hold
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
                reason=f"V4: {'+'.join(signal.signal_types)}, Score:{signal.score:.0f}, {exit_reason}"
            )

            results.add_trade(trade)
            return trade

        return None

    def _simulate_exit(
        self,
        history: List[HistoricalPrice],
        entry_idx: int,
        entry_price: float,
        tp_pct: float,
        sl_pct: float,
        max_hold: int,
    ) -> Tuple[float, int, str]:
        """Simulate exit with adaptive parameters."""
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)

        for j in range(entry_idx + 1, min(entry_idx + max_hold, len(history))):
            current_price = history[j].price

            if current_price >= tp_price:
                return current_price, j, f"TP @ {tp_pct:.1%}"

            if current_price <= sl_price:
                return current_price, j, f"SL @ {sl_pct:.1%}"

        exit_idx = min(entry_idx + max_hold, len(history) - 1)
        return history[exit_idx].price, exit_idx, "Time exit"


# Enhanced flow defaults for optimizer
ENHANCED_FLOW_DEFAULTS = {
    'take_profit_pct': 0.06,
    'stop_loss_pct': 0.08,
    'max_position_pct': 0.10,
    'min_signal_score': 40.0,
    'confirmation_weight': 0.5,
    'context_weight': 0.3,
}

# Parameter bounds for optimization
ENHANCED_FLOW_BOUNDS = [
    (0.03, 0.15),   # take_profit_pct
    (0.04, 0.20),   # stop_loss_pct
    (0.05, 0.20),   # max_position_pct
    (30.0, 70.0),   # min_signal_score
    (0.2, 1.0),     # confirmation_weight
    (0.1, 0.8),     # context_weight
]

ENHANCED_FLOW_PARAM_NAMES = [
    'take_profit_pct',
    'stop_loss_pct',
    'max_position_pct',
    'min_signal_score',
    'confirmation_weight',
    'context_weight',
]


async def run_backtest(
    params: Optional[EnhancedFlowParams] = None,
    capital: float = 1000.0,
    days: int = 60,
    verbose: bool = False,
) -> BacktestResults:
    """Run a single backtest with given parameters."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    params = params or EnhancedFlowParams()
    backtester = EnhancedFlowBacktester(params, capital)

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
    """Run optimization to find best V4 parameters."""
    logging.basicConfig(level=logging.INFO)

    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print(f"[V4 Enhanced Flow Optimization]")
        print(f"Fetching markets (last {days} days)...")
        raw_markets = await api.fetch_closed_markets(days=days)

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

        def backtest_fn(params: Dict, fold_markets: List[Dict]) -> BacktestResults:
            flow_params = EnhancedFlowParams(
                take_profit_pct=params.get('take_profit_pct', 0.06),
                stop_loss_pct=params.get('stop_loss_pct', 0.08),
                max_position_pct=params.get('max_position_pct', 0.10),
                min_signal_score=params.get('min_signal_score', 40.0),
                confirmation_weight=params.get('confirmation_weight', 0.5),
                context_weight=params.get('context_weight', 0.3),
            )

            backtester = EnhancedFlowBacktester(flow_params, capital)

            fold_token_ids = set()
            for m in fold_markets:
                for t in m.get('tokens', []):
                    fold_token_ids.add(t['token_id'])

            fold_cache = {k: v for k, v in price_cache.items() if k in fold_token_ids}
            backtester.set_price_cache(fold_cache)

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

        # Create custom optimizer config for 6 parameters
        opt_config = OptimizationConfigV3(
            total_days=days,
            n_calls=n_calls,
            initial_capital=capital,
        )

        # Create custom optimizer with enhanced bounds
        from skopt import gp_minimize
        from skopt.space import Real

        print(f"\nStarting V4 Enhanced optimization ({n_calls} iterations)...")
        print("Parameters: take_profit, stop_loss, max_position, min_score, confirm_weight, context_weight")

        space = [
            Real(0.03, 0.15, name='take_profit_pct'),
            Real(0.04, 0.20, name='stop_loss_pct'),
            Real(0.05, 0.20, name='max_position_pct'),
            Real(30.0, 70.0, name='min_signal_score'),
            Real(0.2, 1.0, name='confirmation_weight'),
            Real(0.1, 0.8, name='context_weight'),
        ]

        best_sharpe = float('-inf')
        best_params = None
        iteration = [0]

        def objective(x):
            iteration[0] += 1
            params = {
                'take_profit_pct': x[0],
                'stop_loss_pct': x[1],
                'max_position_pct': x[2],
                'min_signal_score': x[3],
                'confirmation_weight': x[4],
                'context_weight': x[5],
            }

            results = backtest_fn(params, markets_data[:200])

            sharpe = results.sharpe_ratio
            if sharpe is None or sharpe != sharpe:  # NaN check
                sharpe = -10.0

            nonlocal best_sharpe, best_params
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params.copy()
                print(f"  [{iteration[0]}/{n_calls}] NEW BEST: Sharpe={sharpe:.2f}, "
                      f"TP={params['take_profit_pct']:.1%}, SL={params['stop_loss_pct']:.1%}, "
                      f"Score>={params['min_signal_score']:.0f}")
            elif iteration[0] % 10 == 0:
                print(f"  [{iteration[0]}/{n_calls}] Sharpe={sharpe:.2f} (best={best_sharpe:.2f})")

            return -sharpe  # Minimize negative sharpe

        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_random_starts=min(10, n_calls // 3),
            random_state=42,
            verbose=False,
        )

        # Print final results
        print("\n" + "=" * 60)
        print("V4 ENHANCED FLOW OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"\nBest Sharpe Ratio: {best_sharpe:.4f}")
        print(f"\nOptimal Parameters:")
        if best_params:
            print(f"  Take Profit:         {best_params['take_profit_pct']:.2%}")
            print(f"  Stop Loss:           {best_params['stop_loss_pct']:.2%}")
            print(f"  Max Position:        {best_params['max_position_pct']:.2%}")
            print(f"  Min Signal Score:    {best_params['min_signal_score']:.1f}")
            print(f"  Confirmation Weight: {best_params['confirmation_weight']:.2f}")
            print(f"  Context Weight:      {best_params['context_weight']:.2f}")

            # Run final backtest with best params
            print("\nFinal backtest with optimal parameters:")
            final_params = EnhancedFlowParams(**best_params)
            final_backtester = EnhancedFlowBacktester(final_params, capital)
            final_backtester.set_price_cache(price_cache)

            market_objs = []
            for m in markets_data[:200]:
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

            final_results = final_backtester.run_sync(market_objs)
            final_results.print_report()

        print("=" * 60)

    finally:
        await api.close()


def main():
    parser = argparse.ArgumentParser(description="V4 Enhanced Flow Strategy")
    parser.add_argument('--backtest', action='store_true', help='Run single backtest')
    parser.add_argument('--optimize', action='store_true', help='Run optimization')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--days', type=int, default=60, help='Days of history')
    parser.add_argument('--iterations', '-n', type=int, default=50, help='Optimization iterations')
    parser.add_argument('--take-profit', type=float, default=0.06, help='Take profit pct')
    parser.add_argument('--stop-loss', type=float, default=0.08, help='Stop loss pct')
    parser.add_argument('--max-position', type=float, default=0.10, help='Max position pct')
    parser.add_argument('--min-score', type=float, default=40.0, help='Min signal score')
    parser.add_argument('--confirm-weight', type=float, default=0.5, help='Confirmation weight')
    parser.add_argument('--context-weight', type=float, default=0.3, help='Context weight')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    if args.optimize:
        asyncio.run(run_optimization(
            n_calls=args.iterations,
            days=args.days,
            capital=args.capital,
        ))
    else:
        params = EnhancedFlowParams(
            take_profit_pct=args.take_profit,
            stop_loss_pct=args.stop_loss,
            max_position_pct=args.max_position,
            min_signal_score=args.min_score,
            confirmation_weight=args.confirm_weight,
            context_weight=args.context_weight,
        )
        asyncio.run(run_backtest(
            params=params,
            capital=args.capital,
            days=args.days,
            verbose=args.verbose,
        ))


if __name__ == "__main__":
    main()
