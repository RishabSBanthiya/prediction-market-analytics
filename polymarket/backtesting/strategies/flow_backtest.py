"""
Flow signal backtester.

Tests the predictive value of flow detection signals.
Includes parameter optimization to find best settings.

Uses authenticated py_clob_client for fetching trade data.
"""

import logging
import itertools
import statistics
import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from ...core.models import Market, Token, Trade, Side, HistoricalPrice
from ...core.api import PolymarketAPI
from ...core.config import get_config
from ...flow_detector import TradeFeedFlowDetector, FlowAlert, MarketState
from ...trading.components.hedge_monitor import HedgeConfig, HedgeAction
from ...trading.components.hedge_strategies import simulate_hedge_decision
from ..base import BaseBacktester
from ..results import BacktestResults, SimulatedTrade
from ..execution import RealisticExecution

logger = logging.getLogger(__name__)


# Default signal weights (same as live trading)
DEFAULT_SIGNAL_WEIGHTS = {
    "SMART_MONEY_ACTIVITY": 30,
    "OVERSIZED_BET": 25,
    "COORDINATED_WALLETS": 25,
    "VOLUME_SPIKE": 10,
    "PRICE_ACCELERATION": 10,
    "SUDDEN_PRICE_MOVEMENT": 8,
    "FRESH_WALLET_ACTIVITY": 5,
    "COLD_WALLET_ACTIVITY": 5,
}

# Severity multipliers
SEVERITY_MULTIPLIERS = {
    "LOW": 0.5,
    "MEDIUM": 1.0,
    "HIGH": 1.5,
    "CRITICAL": 2.0,
}


@dataclass
class ParameterSet:
    """A set of parameters to test"""
    min_score: float = 30.0
    min_trade_size: float = 100.0
    oversized_multiplier: float = 10.0
    volume_spike_multiplier: float = 3.0
    max_spread: float = 0.03
    max_price_drift: float = 0.10
    signal_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_SIGNAL_WEIGHTS.copy())
    
    def to_dict(self) -> dict:
        return {
            "min_score": self.min_score,
            "min_trade_size": self.min_trade_size,
            "oversized_multiplier": self.oversized_multiplier,
            "volume_spike_multiplier": self.volume_spike_multiplier,
            "max_spread": self.max_spread,
            "max_price_drift": self.max_price_drift,
        }


@dataclass
class SimulatedHedgeTrade:
    """A simulated hedge trade during backtesting"""
    action: HedgeAction
    token_id: str
    outcome: str
    trigger_time: datetime
    trigger_price: float
    entry_price: float  # Original position entry price
    shares: float
    cost_or_proceeds: float  # Positive for buys, negative for sells
    reason: str
    
    # For arbitrage
    opposite_token_id: Optional[str] = None
    opposite_price: Optional[float] = None
    arb_profit_locked: float = 0.0
    
    @property
    def is_exit(self) -> bool:
        return self.action in (HedgeAction.PARTIAL_EXIT, HedgeAction.STOP_LOSS)


@dataclass
class SimulatedPosition:
    """A position being tracked through price history"""
    token_id: str
    outcome: str
    entry_price: float
    entry_time: datetime
    entry_index: int  # Index in price history
    shares: float
    cost: float
    opposite_token_id: Optional[str] = None
    
    # Hedge state
    is_hedged: bool = False
    hedge_trades: List["SimulatedHedgeTrade"] = field(default_factory=list)
    partial_exit_executed: bool = False
    
    # Exit state
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    """Result for a single parameter set"""
    params: ParameterSet
    total_signals: int = 0
    signals_by_type: Dict[str, int] = field(default_factory=dict)
    predictive_rate: float = 0.0
    avg_return_1min: float = 0.0
    avg_return_5min: float = 0.0
    avg_return_15min: float = 0.0
    avg_return_30min: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    
    def score(self) -> float:
        """Combined score for ranking parameter sets"""
        # Weighted combination of metrics
        return (
            self.predictive_rate * 30 +
            self.win_rate * 20 +
            min(self.profit_factor, 5) * 15 +  # Cap at 5
            (self.avg_return_5min * 100) * 10 +  # 5min returns
            min(self.sharpe_ratio, 3) * 10 +  # Cap at 3
            (self.total_signals / 100) * 5  # More signals = more opportunities
        )


class FlowBacktester(BaseBacktester):
    """
    Backtester for flow detection signals.
    
    Tests how predictive various signal types are of future price moves.
    Includes parameter optimization to find optimal settings.
    
    Uses realistic execution with:
    - No fees (Polymarket has no fees)
    - Spread checks (max 3%)
    - Slippage based on liquidity
    - Position size limits based on market depth
    """
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        days: int = 3,
        min_trade_size: float = 100.0,
        evaluation_windows: List[int] = None,
        optimize_params: bool = False,
        max_spread_pct: float = 0.03,
        max_markets: int = 500,
        enable_hedging: bool = True,
        hedge_config: Optional[HedgeConfig] = None,
        **kwargs
    ):
        super().__init__(initial_capital, days, **kwargs)
        self.min_trade_size = min_trade_size
        self.evaluation_windows = evaluation_windows or [1, 5, 15, 30]  # minutes
        self.optimize_params = optimize_params
        self.max_spread_pct = max_spread_pct
        self.max_markets = max_markets
        
        # Hedge configuration
        self.enable_hedging = enable_hedging
        self.hedge_config = hedge_config or HedgeConfig()
        
        # Use realistic execution model (no fees, slippage/spread checks)
        self.execution = RealisticExecution(
            max_spread_pct=max_spread_pct,
            buy_slippage_pct=0.005,  # 0.5% base slippage
            sell_slippage_pct=0.005,
        )
        
        # Signal tracking
        self.signal_results: Dict[str, List[dict]] = defaultdict(list)
        
        # Current parameters being tested
        self.current_params: ParameterSet = ParameterSet(min_trade_size=min_trade_size)
        
        # Optimization results
        self.optimization_results: List[BacktestResult] = []
        
        # Hedge tracking
        self._all_hedge_trades: List[SimulatedHedgeTrade] = []
        self._hedges_triggered = 0
        self._hedge_pnl = 0.0
        self._loss_avoided_by_hedging = 0.0
        
        # Track liquidity estimates
        self._liquidity_cache: Dict[str, float] = {}
    
    @property
    def strategy_name(self) -> str:
        hedge_str = " +Hedge" if self.enable_hedging else ""
        return f"Flow Detection Signals{hedge_str}"
    
    async def run(self) -> BacktestResults:
        """Run backtest using historical price data from CLOB API"""
        logger.info(f"Starting backtest: {self.strategy_name}")
        logger.info(f"Capital: ${self.initial_capital:,.2f}, Days: {self.days}")
        
        # Initialize API for market data
        config = get_config()
        self.api = PolymarketAPI(config)
        await self.api.connect()
        
        # Initialize CLOB client for price history
        from py_clob_client.client import ClobClient
        
        self.clob_client = ClobClient(
            config.clob_host,
            key=config.private_key,
            chain_id=config.chain_id,
            signature_type=2,
            funder=config.proxy_address
        )
        self.clob_client.set_api_creds(self.clob_client.create_or_derive_api_creds())
        logger.info("CLOB client initialized")
        
        try:
            # Get all sampling markets (active, liquid markets) with pagination
            logger.info("Fetching active markets...")
            markets_data = []
            cursor = 'MA=='
            
            while cursor:
                try:
                    result = self.clob_client.get_sampling_simplified_markets(next_cursor=cursor)
                    data = result.get('data', [])
                    markets_data.extend(data)
                    cursor = result.get('next_cursor')
                    
                    if not data or not cursor:
                        break
                except Exception as e:
                    logger.debug(f"Pagination ended: {e}")
                    break
            
            logger.info(f"Found {len(markets_data)} sampling markets")
            
            # Fetch historical price data, spreads, and liquidity for each token
            import aiohttp
            self._price_history: Dict[str, List[dict]] = {}
            self._market_info: Dict[str, dict] = {}
            self._token_liquidity: Dict[str, dict] = {}  # Spread and liquidity data
            
            # Limit markets for reasonable backtest time
            num_markets = min(len(markets_data), self.max_markets)
            logger.info(f"Fetching historical price data for {num_markets} markets...")
            
            async with aiohttp.ClientSession() as session:
                for i, market in enumerate(markets_data[:num_markets]):
                    if (i + 1) % 50 == 0:
                        logger.info(f"  Fetching data for market {i+1}/{num_markets}...")
                    
                    condition_id = market.get('condition_id')
                    tokens = market.get('tokens', [])
                    rewards = market.get('rewards', {})
                    
                    # Extract liquidity info from rewards config
                    min_size = rewards.get('min_size', 50)
                    max_spread = rewards.get('max_spread', 5.0)  # Percentage
                    
                    self._market_info[condition_id] = {
                        "tokens": tokens,
                        "active": market.get('active'),
                        "min_size": min_size,
                        "max_spread": max_spread,
                    }
                    
                    for token in tokens:
                        token_id = token.get('token_id')
                        if not token_id:
                            continue
                        
                        # Store token liquidity data
                        token_price = token.get('price', 0.5)
                        self._token_liquidity[token_id] = {
                            "price": token_price,
                            "min_size": min_size,
                            "max_spread_pct": max_spread / 100,  # Convert to decimal
                            # Estimate liquidity from rewards config (markets with rewards are liquid)
                            "estimated_liquidity_usd": min_size * 20 if rewards else 100,
                        }
                        
                        # Fetch price history using the /prices-history endpoint
                        url = f"{config.clob_host}/prices-history?market={token_id}&interval=1h"
                        try:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    history = data.get('history', [])
                                    if history:
                                        self._price_history[token_id] = history
                        except Exception as e:
                            logger.debug(f"Error fetching price history for {token_id}: {e}")
            
            logger.info(f"Fetched price history for {len(self._price_history)} tokens")
            
            # Convert to market list format
            markets = []
            for market in markets_data[:num_markets]:
                tokens = market.get('tokens', [])
                if tokens:
                    condition_id = market.get('condition_id', '')
                    m = Market(
                        condition_id=condition_id,
                        question=f"Market {condition_id[:20]}...",
                        slug=f"market-{condition_id[:16]}",
                        tokens=[Token(
                            token_id=t.get('token_id'),
                            outcome=t.get('outcome', 'Unknown'),
                            price=t.get('price', 0.5)
                        ) for t in tokens],
                        end_date=datetime.now(timezone.utc) + timedelta(days=30),
                    )
                    markets.append(m)
            
            logger.info(f"Prepared {len(markets)} markets for backtesting")
            
            if self.optimize_params:
                return await self.run_parameter_optimization(markets)
            else:
                self.results = await self.run_strategy(markets)
                self.results.markets_analyzed = len(markets)
                return self.results
                
        finally:
            await self.api.close()
    
    async def run_parameter_optimization(self, markets: List[Market]) -> BacktestResults:
        """Run backtest with multiple parameter sets to find optimal"""
        print("\n" + "="*70)
        print("FLOW SIGNAL PARAMETER OPTIMIZATION")
        print("="*70)
        
        # Generate parameter sets to test
        param_sets = self._generate_param_sets()
        print(f"Testing {len(param_sets)} parameter combinations...\n")
        
        for i, params in enumerate(param_sets):
            self.current_params = params
            self.signal_results.clear()
            
            # Test this parameter set
            result = await self._test_param_set(params, markets)
            self.optimization_results.append(result)
            
            # Progress
            if (i + 1) % 5 == 0:
                print(f"  Tested {i+1}/{len(param_sets)} combinations...")
        
        # Find best parameters
        self._print_optimization_results()
        
        # Return results with best params
        best = max(self.optimization_results, key=lambda r: r.score())
        
        start_date = datetime.now(timezone.utc) - timedelta(days=self.days)
        end_date = datetime.now(timezone.utc)
        
        results = BacktestResults(
            strategy_name=self.strategy_name + " (Optimized)",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )
        results.markets_analyzed = len(markets)
        results.finalize()
        
        return results
    
    def _generate_param_sets(self) -> List[ParameterSet]:
        """Generate parameter combinations to test"""
        param_sets = []
        
        # Parameter ranges to test
        min_scores = [20, 30, 40, 50]
        min_trade_sizes = [50, 100, 200, 500]
        oversized_mults = [5, 10, 15, 20]
        volume_spike_mults = [2, 3, 5, 7]
        max_spreads = [0.02, 0.03, 0.05]
        max_drifts = [0.05, 0.10, 0.15]
        
        # Generate combinations (limit to reasonable number)
        for min_score, min_trade, oversized, volume, spread, drift in itertools.product(
            min_scores[:3], min_trade_sizes[:3], oversized_mults[:2], 
            volume_spike_mults[:2], max_spreads[:2], max_drifts[:2]
        ):
            param_sets.append(ParameterSet(
                min_score=min_score,
                min_trade_size=min_trade,
                oversized_multiplier=oversized,
                volume_spike_multiplier=volume,
                max_spread=spread,
                max_price_drift=drift,
            ))
        
        return param_sets
    
    async def _test_param_set(self, params: ParameterSet, markets: List[Market]) -> BacktestResult:
        """Test a single parameter set"""
        result = BacktestResult(params=params)
        
        all_evaluations = []
        
        for market in markets:
            for token in market.tokens:
                evaluations = await self._analyze_token_with_params(market, token, params)
                all_evaluations.extend(evaluations)
        
        if not all_evaluations:
            return result
        
        # Calculate metrics
        result.total_signals = len(all_evaluations)
        
        # Count by type
        for eval_data in all_evaluations:
            signal_type = eval_data.get("type", "UNKNOWN")
            result.signals_by_type[signal_type] = result.signals_by_type.get(signal_type, 0) + 1
        
        # Predictive rate
        predictive = [e for e in all_evaluations if e.get("was_predictive")]
        result.predictive_rate = len(predictive) / len(all_evaluations) if all_evaluations else 0
        
        # Average returns by window
        for window_name, window_min in [("1min", 1), ("5min", 5), ("15min", 15), ("30min", 30)]:
            returns = [e["returns"].get(window_min, 0) for e in all_evaluations if window_min in e.get("returns", {})]
            if returns:
                avg_return = sum(returns) / len(returns)
                setattr(result, f"avg_return_{window_name}", avg_return)
        
        # Win rate (based on direction)
        correct_direction = 0
        for e in all_evaluations:
            direction = e.get("direction", "NEUTRAL")
            if direction == "NEUTRAL":
                continue
            ret_5min = e.get("5min_return", 0)
            if (direction == "BUY" and ret_5min > 0) or (direction == "SELL" and ret_5min < 0):
                correct_direction += 1
        
        directional = [e for e in all_evaluations if e.get("direction") != "NEUTRAL"]
        result.win_rate = correct_direction / len(directional) if directional else 0
        
        # Profit factor
        gains = sum(e.get("5min_return", 0) for e in all_evaluations if e.get("5min_return", 0) > 0)
        losses = abs(sum(e.get("5min_return", 0) for e in all_evaluations if e.get("5min_return", 0) < 0))
        result.profit_factor = gains / losses if losses > 0 else gains if gains > 0 else 0
        
        # Sharpe-like ratio
        returns_5min = [e.get("5min_return", 0) for e in all_evaluations if "5min_return" in e]
        if len(returns_5min) > 1:
            mean_ret = statistics.mean(returns_5min)
            std_ret = statistics.stdev(returns_5min)
            result.sharpe_ratio = mean_ret / std_ret if std_ret > 0 else 0
        
        return result
    
    async def _analyze_token_with_params(
        self,
        market: Market,
        token: Token,
        params: ParameterSet
    ) -> List[dict]:
        """Analyze a token with specific parameters"""
        evaluations = []
        
        # Get cached price history
        history = self._price_history.get(token.token_id, [])
        if len(history) < 20:
            return evaluations
        
        # Set current params and detect signals
        self.current_params = params
        signals = self._detect_signals_from_price_history(
            token.token_id,
            market.condition_id,
            market.question,
            history
        )
        
        # Evaluate each signal
        for signal in signals:
            evaluation = self._evaluate_signal_from_history(signal, history)
            if evaluation:
                evaluations.append(evaluation)
        
        return evaluations
    
    def _detect_signals_with_params(
        self,
        trades: List[Trade],
        history: List,
        params: ParameterSet
    ) -> List[dict]:
        """Detect signals using specific parameters"""
        signals = []
        
        if not trades:
            return signals
        
        avg_size = sum(t.value_usd for t in trades) / len(trades)
        window_trades: Dict[str, List[Trade]] = defaultdict(list)
        
        for i, trade in enumerate(trades):
            # Oversized bet detection with param
            if trade.value_usd >= params.min_trade_size:
                if trade.value_usd >= avg_size * params.oversized_multiplier:
                    signals.append({
                        "type": "OVERSIZED_BET",
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "value": trade.value_usd,
                        "direction": "BUY" if trade.side.value == "BUY" else "SELL",
                    })
            
            # Volume spike detection with param
            window_key = trade.timestamp.strftime("%Y-%m-%d-%H-%M")
            window_trades[window_key].append(trade)
            
            if len(window_trades[window_key]) >= 5:
                window_value = sum(t.value_usd for t in window_trades[window_key])
                if window_value >= avg_size * params.volume_spike_multiplier * 5:
                    signals.append({
                        "type": "VOLUME_SPIKE",
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "value": window_value,
                        "direction": "NEUTRAL",
                    })
            
            # Price acceleration detection
            if i >= 5:
                recent_prices = [trades[j].price for j in range(i-5, i+1)]
                if len(recent_prices) >= 5:
                    early_change = abs(recent_prices[2] - recent_prices[0])
                    late_change = abs(recent_prices[-1] - recent_prices[2])
                    
                    if early_change > 0 and late_change > early_change * 2:
                        direction = "BUY" if recent_prices[-1] > recent_prices[0] else "SELL"
                        signals.append({
                            "type": "PRICE_ACCELERATION",
                            "timestamp": trade.timestamp,
                            "price": trade.price,
                            "value": late_change,
                            "direction": direction,
                        })
        
        return signals
    
    async def _simulate_position_with_hedges(
        self,
        position: SimulatedPosition,
        history: List[dict],
        opposite_history: Optional[List[dict]],
        liquidity_estimate: float,
    ) -> Tuple[float, float, str]:
        """
        Simulate a position through price history, checking for hedge triggers.
        
        Returns: (exit_price, exit_shares, exit_reason)
        """
        remaining_shares = position.shares
        total_hedge_cost = 0.0
        total_hedge_proceeds = 0.0
        
        # Track what would have happened without hedging for comparison
        unhedged_exit_price = history[-1].get('p', 0) if history else 0
        
        # Walk through price history after entry
        for j in range(position.entry_index + 1, len(history)):
            current_price = history[j].get('p', 0)
            current_time = datetime.fromtimestamp(history[j].get('t', 0), tz=timezone.utc)
            
            if current_price <= 0:
                continue
            
            # Get opposite price at same time (approximate by index ratio)
            opposite_price = None
            if opposite_history and position.opposite_token_id:
                opp_index = int(j * len(opposite_history) / len(history))
                opp_index = min(opp_index, len(opposite_history) - 1)
                opposite_price = opposite_history[opp_index].get('p', 0)
            
            # Check for hedge trigger if hedging enabled
            if self.enable_hedging and not position.is_hedged:
                hedge_action = simulate_hedge_decision(
                    current_price=current_price,
                    entry_price=position.entry_price,
                    opposite_price=opposite_price,
                    config=self.hedge_config,
                )
                
                if hedge_action:
                    self._hedges_triggered += 1
                    
                    # Simulate hedge execution
                    hedge_trade = self._execute_simulated_hedge(
                        position=position,
                        action=hedge_action,
                        current_price=current_price,
                        current_time=current_time,
                        opposite_price=opposite_price,
                        remaining_shares=remaining_shares,
                        liquidity_estimate=liquidity_estimate,
                    )
                    
                    if hedge_trade:
                        position.hedge_trades.append(hedge_trade)
                        self._all_hedge_trades.append(hedge_trade)
                        
                        # Update state based on hedge action
                        if hedge_action == HedgeAction.ARBITRAGE:
                            position.is_hedged = True
                            total_hedge_cost += hedge_trade.cost_or_proceeds
                            self.cash -= hedge_trade.cost_or_proceeds
                            
                        elif hedge_action == HedgeAction.PROTECTIVE_HEDGE:
                            position.is_hedged = True
                            total_hedge_cost += hedge_trade.cost_or_proceeds
                            self.cash -= hedge_trade.cost_or_proceeds
                            
                        elif hedge_action == HedgeAction.PARTIAL_EXIT:
                            position.partial_exit_executed = True
                            remaining_shares -= hedge_trade.shares
                            total_hedge_proceeds += abs(hedge_trade.cost_or_proceeds)
                            self.cash += abs(hedge_trade.cost_or_proceeds)
                            
                        elif hedge_action == HedgeAction.STOP_LOSS:
                            position.is_hedged = True
                            remaining_shares = 0
                            total_hedge_proceeds += abs(hedge_trade.cost_or_proceeds)
                            self.cash += abs(hedge_trade.cost_or_proceeds)
                            
                            # Position is closed, return immediately
                            return hedge_trade.trigger_price, 0, "Stop-loss"
        
        # Final resolution
        exit_price = history[-1].get('p', 0) if history else 0
        
        # Exit simulation (resolved markets have no slippage at $1.00)
        if exit_price > 0.99:
            actual_exit_price = 1.0
            exit_shares = remaining_shares
            exit_reason = "Resolved YES"
        elif exit_price < 0.01:
            actual_exit_price = 0.0
            exit_shares = remaining_shares
            exit_reason = "Resolved NO"
        else:
            actual_exit_price, exit_shares, _ = self.execution.execute_sell(
                exit_price,
                remaining_shares,
                None,
                liquidity_usd=liquidity_estimate
            )
            exit_reason = "Market close"
        
        # Calculate hedge impact
        if position.hedge_trades:
            # For arbitrage, we get guaranteed $1 at resolution regardless of outcome
            for ht in position.hedge_trades:
                if ht.action == HedgeAction.ARBITRAGE and ht.arb_profit_locked > 0:
                    self._hedge_pnl += ht.arb_profit_locked
                    # Also account for the NO shares payout at resolution
                    self.cash += ht.shares  # NO pays $1 at resolution
            
            # Calculate what loss would have been without hedging
            unhedged_pnl = (unhedged_exit_price - position.entry_price) * position.shares
            actual_pnl = (actual_exit_price - position.entry_price) * exit_shares
            if unhedged_pnl < actual_pnl:
                self._loss_avoided_by_hedging += (actual_pnl - unhedged_pnl)
        
        return actual_exit_price, exit_shares, exit_reason
    
    def _execute_simulated_hedge(
        self,
        position: SimulatedPosition,
        action: HedgeAction,
        current_price: float,
        current_time: datetime,
        opposite_price: Optional[float],
        remaining_shares: float,
        liquidity_estimate: float,
    ) -> Optional[SimulatedHedgeTrade]:
        """Execute a simulated hedge trade"""
        
        if action == HedgeAction.ARBITRAGE:
            if opposite_price is None:
                return None
            
            # Buy opposite to lock in arbitrage
            arb_profit = 1.0 - (current_price + opposite_price)
            
            # Execute NO buy
            exec_price, filled_shares, _ = self.execution.execute_buy(
                opposite_price,
                remaining_shares,  # Match position size
                None,
                liquidity_usd=liquidity_estimate
            )
            
            if filled_shares <= 0:
                return None
            
            cost = filled_shares * exec_price
            
            return SimulatedHedgeTrade(
                action=action,
                token_id=position.opposite_token_id or "",
                outcome="NO",
                trigger_time=current_time,
                trigger_price=current_price,
                entry_price=position.entry_price,
                shares=filled_shares,
                cost_or_proceeds=cost,
                reason=f"Arb: YES({current_price:.3f})+NO({opposite_price:.3f})=${current_price + opposite_price:.3f}",
                opposite_token_id=position.token_id,
                opposite_price=current_price,
                arb_profit_locked=arb_profit * filled_shares,
            )
        
        elif action == HedgeAction.PROTECTIVE_HEDGE:
            if opposite_price is None:
                return None
            
            # Buy some opposite to reduce downside
            hedge_shares = remaining_shares * self.hedge_config.hedge_cost_max_pct
            
            exec_price, filled_shares, _ = self.execution.execute_buy(
                opposite_price,
                hedge_shares,
                None,
                liquidity_usd=liquidity_estimate
            )
            
            if filled_shares <= 0:
                return None
            
            cost = filled_shares * exec_price
            
            return SimulatedHedgeTrade(
                action=action,
                token_id=position.opposite_token_id or "",
                outcome="NO",
                trigger_time=current_time,
                trigger_price=current_price,
                entry_price=position.entry_price,
                shares=filled_shares,
                cost_or_proceeds=cost,
                reason=f"Protective hedge at {(position.entry_price - current_price) / position.entry_price:.1%} loss",
                opposite_token_id=position.token_id,
                opposite_price=current_price,
            )
        
        elif action == HedgeAction.PARTIAL_EXIT:
            # Sell portion of position
            exit_shares = remaining_shares * self.hedge_config.partial_exit_pct
            
            exec_price, filled_shares, _ = self.execution.execute_sell(
                current_price,
                exit_shares,
                None,
                liquidity_usd=liquidity_estimate
            )
            
            if filled_shares <= 0:
                return None
            
            proceeds = filled_shares * exec_price
            
            return SimulatedHedgeTrade(
                action=action,
                token_id=position.token_id,
                outcome=position.outcome,
                trigger_time=current_time,
                trigger_price=current_price,
                entry_price=position.entry_price,
                shares=filled_shares,
                cost_or_proceeds=-proceeds,  # Negative = received
                reason=f"Partial exit ({self.hedge_config.partial_exit_pct:.0%}) at {(position.entry_price - current_price) / position.entry_price:.1%} loss",
            )
        
        elif action == HedgeAction.STOP_LOSS:
            # Exit entire position
            exec_price, filled_shares, _ = self.execution.execute_sell(
                current_price,
                remaining_shares,
                None,
                liquidity_usd=liquidity_estimate
            )
            
            if filled_shares <= 0:
                return None
            
            proceeds = filled_shares * exec_price
            loss_pct = (position.entry_price - exec_price) / position.entry_price
            
            return SimulatedHedgeTrade(
                action=action,
                token_id=position.token_id,
                outcome=position.outcome,
                trigger_time=current_time,
                trigger_price=current_price,
                entry_price=position.entry_price,
                shares=filled_shares,
                cost_or_proceeds=-proceeds,  # Negative = received
                reason=f"Stop-loss at {loss_pct:.1%} loss (threshold: {self.hedge_config.stop_loss_pct:.0%})",
            )
        
        return None
    
    def _print_optimization_results(self):
        """Print optimization results and recommendations"""
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS")
        print("="*70)
        
        # Sort by score
        sorted_results = sorted(self.optimization_results, key=lambda r: r.score(), reverse=True)
        
        print("\n--- TOP 5 PARAMETER SETS ---\n")
        
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"#{i} Score: {result.score():.1f}")
            print(f"   min_score: {result.params.min_score}")
            print(f"   min_trade_size: ${result.params.min_trade_size}")
            print(f"   oversized_multiplier: {result.params.oversized_multiplier}x")
            print(f"   volume_spike_multiplier: {result.params.volume_spike_multiplier}x")
            print(f"   max_spread: {result.params.max_spread:.0%}")
            print(f"   max_price_drift: {result.params.max_price_drift:.0%}")
            print(f"   ---")
            print(f"   Signals: {result.total_signals}")
            print(f"   Predictive Rate: {result.predictive_rate:.1%}")
            print(f"   Win Rate: {result.win_rate:.1%}")
            print(f"   5min Avg Return: {result.avg_return_5min:.2%}")
            print(f"   Profit Factor: {result.profit_factor:.2f}")
            print(f"   Sharpe: {result.sharpe_ratio:.2f}")
            print()
        
        # Best params
        best = sorted_results[0]
        print("="*70)
        print("RECOMMENDED PARAMETERS:")
        print("="*70)
        print(f"""
# Update flow_strategy.py with these values:
MIN_SCORE = {best.params.min_score}
MIN_TRADE_SIZE = {best.params.min_trade_size}
OVERSIZED_BET_MULTIPLIER = {best.params.oversized_multiplier}
VOLUME_SPIKE_MULTIPLIER = {best.params.volume_spike_multiplier}
MAX_SPREAD = {best.params.max_spread}
MAX_PRICE_DRIFT = {best.params.max_price_drift}

# Expected performance:
# - Signal count: ~{best.total_signals} per analysis period
# - Predictive rate: {best.predictive_rate:.1%}
# - Win rate: {best.win_rate:.1%}
# - Avg 5min return: {best.avg_return_5min:.2%}
# - Profit factor: {best.profit_factor:.2f}
""")
        print("="*70)
    
    async def run_strategy(self, markets: List[Market]) -> BacktestResults:
        """Run the flow signal backtest"""
        start_date = datetime.now(timezone.utc) - timedelta(days=self.days)
        end_date = datetime.now(timezone.utc)
        
        results = BacktestResults(
            strategy_name=self.strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )
        
        # Reset hedge tracking
        self._all_hedge_trades = []
        self._hedges_triggered = 0
        self._hedge_pnl = 0.0
        self._loss_avoided_by_hedging = 0.0
        
        markets_traded = 0
        
        for market in markets:
            if self.verbose:
                logger.info(f"Analyzing: {market.question[:50]}...")
            
            for token in market.tokens:
                # Only analyze tokens with sufficient price history
                history = self._price_history.get(token.token_id, [])
                if len(history) < 20:
                    continue
                
                await self._analyze_token_signals(market, token, results)
                
                # Check if any trades were executed for this market
                if results.trades:
                    recent_trades = [t for t in results.trades if t.token_id == token.token_id]
                    if recent_trades:
                        markets_traded += 1
        
        results.markets_traded = markets_traded
        
        # Add hedge metrics to results
        results.hedge_trades = self._all_hedge_trades
        results.hedges_triggered = self._hedges_triggered
        results.hedge_pnl = self._hedge_pnl
        results.loss_avoided_by_hedging = self._loss_avoided_by_hedging
        
        results.finalize()
        
        # Print signal analysis
        self._print_signal_analysis()
        
        return results
    
    async def _analyze_token_signals(
        self,
        market: Market,
        token: Token,
        results: BacktestResults
    ):
        """Analyze signals for a token and execute trades when signals are found"""
        # Get cached price history
        history = self._price_history.get(token.token_id, [])
        
        if len(history) < 20:
            return
        
        # Detect signals from price movements
        signals = self._detect_signals_from_price_history(
            token.token_id, 
            market.condition_id,
            market.question,
            history
        )
        
        # Find opposite token for hedging
        opposite_token_id = None
        opposite_history = None
        for other_token in market.tokens:
            if other_token.token_id != token.token_id:
                opposite_token_id = other_token.token_id
                opposite_history = self._price_history.get(opposite_token_id, [])
                break
        
        # Execute trades for each signal
        for signal in signals:
            evaluation = self._evaluate_signal_from_history(signal, history)
            if evaluation:
                self.signal_results[signal["type"]].append(evaluation)
                
                # Check if signal is tradeable (meets minimum score, spread acceptable)
                if evaluation.get("spread_acceptable") and evaluation.get("win"):
                    # Try to execute trade
                    trade = await self._find_trade_opportunity(
                        market, token, signal, history, results,
                        opposite_token_id, opposite_history, evaluation
                    )
                    if trade:
                        # Trade executed successfully
                        pass
    
    async def _find_trade_opportunity(
        self,
        market: Market,
        token: Token,
        signal: dict,
        history: List[dict],
        results: BacktestResults,
        opposite_token_id: Optional[str],
        opposite_history: Optional[List[dict]],
        evaluation: dict
    ) -> Optional[object]:
        """Find and execute a trade opportunity from a signal"""
        signal_idx = signal.get("index", 0)
        signal_price = signal.get("price", 0)
        signal_ts = signal.get("timestamp")
        direction = signal.get("direction", "NEUTRAL")
        
        if signal_price <= 0 or signal_idx >= len(history) - 5:
            return None
        
        # Only trade BUY signals for now (can extend to SELL later)
        if direction != "BUY":
            return None
        
        # Get liquidity info
        liquidity_info = self._token_liquidity.get(token.token_id, {})
        estimated_liquidity = liquidity_info.get("estimated_liquidity_usd", 500)
        max_spread_pct = liquidity_info.get("max_spread_pct", 0.05)
        
        # Check spread acceptability
        if max_spread_pct > self.max_spread_pct:
            return None
        
        # Check if we have capital
        if self.cash < self.config.risk.min_trade_value_usd:
            return None
        
        # Calculate position size based on signal strength
        # Use 2% of capital as base, scale up to 3x for high-confidence signals
        base_position_pct = 0.02
        signal_score = evaluation.get("5min_return", 0) * 100  # Convert to percentage
        score_multiplier = min(3.0, 1.0 + abs(signal_score) / 10.0)  # Scale up to 3x
        position_dollars = self.cash * base_position_pct * score_multiplier
        
        # Cap by estimated liquidity (max 10% of available)
        max_position = estimated_liquidity * 0.10
        position_dollars = min(position_dollars, max_position)
        
        if position_dollars < self.config.risk.min_trade_value_usd:
            return None
        
        # Simulate execution with liquidity-based slippage
        exec_price, filled_shares, fee = self.execution.execute_buy(
            signal_price,
            position_dollars / signal_price,
            None,
            liquidity_usd=estimated_liquidity
        )
        
        if filled_shares <= 0:
            return None
        
        cost = filled_shares * exec_price  # No fee
        
        if cost > self.cash:
            return None
        
        # Execute trade
        self.cash -= cost
        entry_time = signal_ts
        
        # Create simulated position for hedge tracking
        position = SimulatedPosition(
            token_id=token.token_id,
            outcome=token.outcome,
            entry_price=exec_price,
            entry_time=entry_time,
            entry_index=signal_idx,
            shares=filled_shares,
            cost=cost,
            opposite_token_id=opposite_token_id,
        )
        
        # Simulate position through remaining price history with hedge checks
        final_exit_price, final_exit_shares, exit_reason = await self._simulate_position_with_hedges(
            position, history, opposite_history, estimated_liquidity
        )
        
        exit_time = datetime.fromtimestamp(history[-1].get('t', 0), tz=timezone.utc) if history else entry_time
        proceeds = final_exit_shares * final_exit_price
        self.cash += proceeds
        
        # Record main trade
        reason_suffix = ""
        if position.hedge_trades:
            reason_suffix = f" [Hedged: {len(position.hedge_trades)} actions]"
        
        self.record_trade(
            results=results,
            market=market,
            token=token,
            entry_time=entry_time,
            entry_price=exec_price,
            shares=filled_shares,
            cost=cost,
            exit_time=exit_time,
            exit_price=final_exit_price,
            reason=f"Flow signal: {signal.get('type', 'UNKNOWN')} @ {signal_price:.4f}{reason_suffix}"
        )
        
        if self.verbose:
            pnl = proceeds - cost
            logger.info(
                f"  Trade: {token.outcome} {filled_shares:.2f} @ ${exec_price:.4f} -> "
                f"${final_exit_price:.4f} P&L: ${pnl:.2f} (signal: {signal.get('type', 'UNKNOWN')})"
            )
            if position.hedge_trades:
                logger.info(f"    Hedge actions: {[h.action.value for h in position.hedge_trades]}")
        
        return results.trades[-1] if results.trades else None
    
    def _detect_signals_from_price_history(
        self,
        token_id: str,
        market_id: str,
        question: str,
        history: List[dict]
    ) -> List[dict]:
        """Detect flow signals from historical price data"""
        signals = []
        
        if len(history) < 20:
            return signals
        
        # Get parameters
        params = getattr(self, 'current_params', None)
        if params:
            price_threshold = 0.05  # Base 5% threshold
            volume_mult = params.volume_spike_multiplier
        else:
            price_threshold = 0.05
            volume_mult = 3.0
        
        # Track price history for volatility calculations
        price_window = []
        
        for i, point in enumerate(history):
            ts = point.get('t', 0)  # Unix timestamp
            price = point.get('p', 0)  # Price
            
            if price <= 0:
                continue
            
            timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            price_window.append((ts, price))
            
            # Keep window to last hour (60 points at 1min interval or so)
            while price_window and ts - price_window[0][0] > 3600:
                price_window.pop(0)
            
            if len(price_window) < 5:
                continue
            
            # Calculate price changes
            prices = [p for _, p in price_window]
            
            # Sudden price movement detection
            if len(prices) >= 5:
                recent_change = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                
                if abs(recent_change) >= price_threshold:
                    signals.append({
                        "type": "SUDDEN_PRICE_MOVEMENT",
                        "timestamp": timestamp,
                        "price": price,
                        "value": recent_change,
                        "direction": "BUY" if recent_change > 0 else "SELL",
                        "index": i,
                        "token_id": token_id,
                    })
            
            # Price acceleration detection
            if len(prices) >= 10:
                early_change = abs(prices[4] - prices[0]) / prices[0] if prices[0] > 0 else 0
                late_change = abs(prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                
                if early_change > 0.01 and late_change > early_change * 2:
                    signals.append({
                        "type": "PRICE_ACCELERATION",
                        "timestamp": timestamp,
                        "price": price,
                        "value": late_change / early_change if early_change > 0 else 0,
                        "direction": "BUY" if prices[-1] > prices[0] else "SELL",
                        "index": i,
                        "token_id": token_id,
                    })
            
            # Volatility spike detection (simulate volume spike using price volatility)
            if len(prices) >= 10:
                returns = [(prices[j] - prices[j-1]) / prices[j-1] 
                          for j in range(1, len(prices)) if prices[j-1] > 0]
                if len(returns) >= 5:
                    recent_volatility = sum(abs(r) for r in returns[-5:]) / 5
                    baseline_volatility = sum(abs(r) for r in returns) / len(returns) if returns else 0.01
                    
                    if recent_volatility > baseline_volatility * volume_mult:
                        signals.append({
                            "type": "VOLUME_SPIKE",  # Approximated via volatility
                            "timestamp": timestamp,
                            "price": price,
                            "value": recent_volatility / baseline_volatility if baseline_volatility > 0 else 0,
                            "direction": "NEUTRAL",
                            "index": i,
                            "token_id": token_id,
                        })
        
        return signals
    
    def _evaluate_signal_from_history(self, signal: dict, history: List[dict]) -> Optional[dict]:
        """Evaluate signal outcome with realistic execution assumptions"""
        signal_idx = signal.get("index", 0)
        signal_price = signal["price"]
        signal_ts = signal["timestamp"]
        token_id = signal.get("token_id", "")
        
        if signal_price <= 0 or signal_idx >= len(history) - 5:
            return None
        
        # Get liquidity info for this token
        liquidity_info = self._token_liquidity.get(token_id, {})
        # max_spread from rewards config - if present, market is liquid enough
        # Default to 5% max spread if no data
        max_spread_pct = liquidity_info.get("max_spread_pct", 0.05)
        estimated_liquidity = liquidity_info.get("estimated_liquidity_usd", 500)
        has_rewards = estimated_liquidity > 100  # Markets with rewards are typically liquid
        
        # Estimate slippage based on trade size ($100 default) and liquidity
        trade_size = 100.0
        slippage_pct = self.execution.estimate_slippage(trade_size, estimated_liquidity)
        
        # Calculate returns at different horizons with slippage
        outcomes = {}
        horizons = [(5, "5min"), (15, "15min"), (30, "30min"), (60, "60min")]
        
        direction = signal.get("direction", "NEUTRAL")
        
        for idx_offset, label in horizons:
            target_idx = min(signal_idx + idx_offset, len(history) - 1)
            if target_idx > signal_idx:
                future_price = history[target_idx].get('p', 0)
                if future_price > 0:
                    # Calculate raw price change
                    raw_pct_change = (future_price - signal_price) / signal_price
                    
                    # Apply slippage to get realistic return
                    # Entry slippage: buy at higher price, sell at lower price
                    if direction == "BUY":
                        entry_price = signal_price * (1 + slippage_pct)
                        exit_price = future_price * (1 - slippage_pct)  # Assume exit with slippage too
                    elif direction == "SELL":
                        entry_price = signal_price * (1 - slippage_pct)
                        exit_price = future_price * (1 + slippage_pct)
                    else:
                        entry_price = signal_price
                        exit_price = future_price
                    
                    realistic_pct_change = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    outcomes[f"{label}_return"] = realistic_pct_change
                    outcomes[f"{label}_raw_return"] = raw_pct_change
        
        if not outcomes:
            return None
        
        # Check if spread would have been acceptable
        # Markets with rewards are actively market-made and typically liquid
        # Use the max_spread from rewards config as the threshold
        spread_acceptable = has_rewards or max_spread_pct <= self.max_spread_pct
        
        # Determine if signal was predictive and profitable after costs
        five_min_return = outcomes.get("5min_return", 0)
        five_min_raw = outcomes.get("5min_raw_return", 0)
        
        if direction == "BUY":
            predictive = five_min_raw > 0.005  # Raw return >0.5%
            profitable = five_min_return > 0    # Profitable after slippage
            win = five_min_return > 0 and spread_acceptable
        elif direction == "SELL":
            predictive = five_min_raw < -0.005
            profitable = five_min_return > 0  # For sells, we profit if price went down (but return is calculated as gain)
            win = five_min_return > 0 and spread_acceptable
        else:
            predictive = abs(five_min_raw) > 0.01
            profitable = abs(five_min_return) > 0.005
            win = profitable and spread_acceptable
        
        return {
            "type": signal["type"],
            "timestamp": signal_ts,
            "price": signal_price,
            "direction": direction,
            "predictive": predictive,
            "profitable": profitable,
            "win": win,
            "spread_acceptable": spread_acceptable,
            "slippage_pct": slippage_pct,
            "estimated_liquidity": estimated_liquidity,
            **outcomes
        }
    
    def _detect_signals_from_trades(
        self,
        trades: List[Trade],
        history: List
    ) -> List[dict]:
        """Detect flow signals from trade history (legacy method)"""
        signals = []
        
        # Calculate baseline metrics
        if not trades:
            return signals
        
        avg_size = sum(t.value_usd for t in trades) / len(trades)
        
        # Track recent trades for pattern detection
        window_trades: Dict[str, List[Trade]] = defaultdict(list)
        
        for i, trade in enumerate(trades):
            # Oversized bet detection
            if trade.value_usd >= self.min_trade_size * 10:
                if trade.value_usd >= avg_size * 10:
                    signals.append({
                        "type": "OVERSIZED_BET",
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "value": trade.value_usd,
                        "direction": "BUY" if trade.side.value == "BUY" else "SELL",
                    })
            
            # Volume spike detection (simple version)
            window_key = trade.timestamp.strftime("%Y-%m-%d-%H-%M")
            window_trades[window_key].append(trade)
            
            if len(window_trades[window_key]) >= 5:
                window_value = sum(t.value_usd for t in window_trades[window_key])
                if window_value >= avg_size * 20:
                    signals.append({
                        "type": "VOLUME_SPIKE",
                        "timestamp": trade.timestamp,
                        "price": trade.price,
                        "value": window_value,
                        "direction": "NEUTRAL",
                    })
            
            # Price acceleration detection
            if i >= 5:
                recent_prices = [trades[j].price for j in range(i-5, i+1)]
                if len(recent_prices) >= 5:
                    early_change = abs(recent_prices[2] - recent_prices[0])
                    late_change = abs(recent_prices[-1] - recent_prices[2])
                    
                    if early_change > 0 and late_change > early_change * 2:
                        direction = "BUY" if recent_prices[-1] > recent_prices[0] else "SELL"
                        signals.append({
                            "type": "PRICE_ACCELERATION",
                            "timestamp": trade.timestamp,
                            "price": trade.price,
                            "value": late_change,
                            "direction": direction,
                        })
        
        return signals
    
    def _evaluate_signal(self, signal: dict, history: List) -> Optional[dict]:
        """Evaluate a signal's predictive value"""
        if not history:
            return None
        
        # Find price at signal time
        signal_ts = signal["timestamp"].timestamp()
        signal_price = signal["price"]
        
        # Find future prices
        future_prices = {}
        for window in self.evaluation_windows:
            target_ts = signal_ts + (window * 60)
            
            # Find closest price point
            closest = None
            closest_diff = float('inf')
            
            for point in history:
                diff = abs(point.timestamp - target_ts)
                if diff < closest_diff:
                    closest_diff = diff
                    closest = point.price
            
            if closest and closest_diff < 300:  # Within 5 min of target
                future_prices[window] = closest
        
        if not future_prices:
            return None
        
        # Calculate returns
        returns = {}
        for window, price in future_prices.items():
            if signal_price > 0:
                returns[window] = (price - signal_price) / signal_price
        
        # Determine if predictive
        direction = signal.get("direction", "NEUTRAL")
        was_predictive = False
        
        if direction == "BUY":
            was_predictive = any(r > 0.01 for r in returns.values())
        elif direction == "SELL":
            was_predictive = any(r < -0.01 for r in returns.values())
        
        return {
            "type": signal["type"],
            "direction": direction,
            "price_at_signal": signal_price,
            "returns": returns,
            "was_predictive": was_predictive,
        }
    
    def _print_signal_analysis(self):
        """Print analysis of signal effectiveness"""
        print("\n" + "="*60)
        print("FLOW SIGNAL ANALYSIS")
        print("="*60)
        
        for signal_type, evaluations in self.signal_results.items():
            if not evaluations:
                continue
            
            print(f"\n--- {signal_type} ---")
            print(f"Total signals: {len(evaluations)}")
            
            predictive = [e for e in evaluations if e.get("predictive", False)]
            profitable = [e for e in evaluations if e.get("profitable", False)]
            wins = [e for e in evaluations if e.get("win", False)]
            tradeable = [e for e in evaluations if e.get("spread_acceptable", True)]
            
            print(f"Predictive (raw): {len(predictive)} ({len(predictive)/len(evaluations):.1%})")
            print(f"Profitable (after slippage): {len(profitable)} ({len(profitable)/len(evaluations):.1%})")
            print(f"Win rate (tradeable): {len(wins)/len(evaluations):.1%}")
            print(f"Tradeable (spread OK): {len(tradeable)} ({len(tradeable)/len(evaluations):.1%})")
            
            # Average slippage
            slippages = [e.get("slippage_pct", 0) for e in evaluations]
            if slippages:
                print(f"Avg slippage: {sum(slippages)/len(slippages):.2%}")
            
            # Average returns by window (after slippage)
            for label in ["5min", "15min", "30min", "60min"]:
                key = f"{label}_return"
                raw_key = f"{label}_raw_return"
                returns = [e.get(key, 0) for e in evaluations if key in e]
                raw_returns = [e.get(raw_key, 0) for e in evaluations if raw_key in e]
                if returns:
                    avg_return = sum(returns) / len(returns)
                    avg_raw = sum(raw_returns) / len(raw_returns) if raw_returns else 0
                    print(f"  {label}: raw {avg_raw:.4%} -> after slippage {avg_return:.4%}")
        
        print("\n" + "="*60)


async def run_flow_backtest(
    initial_capital: float = 1000.0,
    days: int = 3,
    verbose: bool = False,
    enable_hedging: bool = True,
    hedge_config: Optional[HedgeConfig] = None,
) -> BacktestResults:
    """Run flow signal backtest"""
    backtester = FlowBacktester(
        initial_capital=initial_capital,
        days=days,
        verbose=verbose,
        enable_hedging=enable_hedging,
        hedge_config=hedge_config,
    )
    
    results = await backtester.run()
    results.print_report()
    
    return results


if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description="Flow Signal Backtester")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    asyncio.run(run_flow_backtest(
        initial_capital=args.capital,
        days=args.days,
        verbose=args.verbose,
    ))

