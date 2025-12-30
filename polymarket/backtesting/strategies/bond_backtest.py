"""
Bond strategy backtester.

Tests the expiring market strategy on historical data.

Uses realistic execution with:
- No fees (Polymarket has no fees)
- Spread checks
- Slippage based on liquidity
- Position size limits
- Hedge simulation (arbitrage, protective hedge, partial exit, stop-loss)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

from ...core.models import Market, Token, HistoricalPrice
from ...trading.components.hedge_monitor import HedgeConfig, HedgeAction
from ...trading.components.hedge_strategies import simulate_hedge_decision
from ..base import BaseBacktester
from ..results import BacktestResults
from ..execution import RealisticExecution

logger = logging.getLogger(__name__)


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
    hedge_trades: List[SimulatedHedgeTrade] = field(default_factory=list)
    partial_exit_executed: bool = False
    
    # Exit state
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None


class BondBacktester(BaseBacktester):
    """
    Backtester for the bond (expiring market) strategy.
    
    Uses realistic execution:
    - No transaction fees
    - Slippage modeling based on liquidity
    - Spread checks before trading
    - Hedge simulation (arbitrage, protective hedge, partial exit, stop-loss)
    - Dynamic time windows based on volatility/volume
    """
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        days: int = 3,
        min_price: float = 0.95,
        max_price: float = 0.98,
        min_seconds_left: int = 60,
        max_seconds_left: int = 1800,
        max_spread_pct: float = 0.02,  # 2% max spread for bonds
        enable_hedging: bool = True,
        hedge_config: Optional[HedgeConfig] = None,
        # Dynamic time window parameters
        use_dynamic_time: bool = False,
        vol_multiplier: float = 1.0,  # Multiplier for volatility adjustment
        vol_threshold: float = 0.05,  # Baseline volatility threshold
        volume_weight: float = 0.5,   # Weight for volume in dynamic calc
        **kwargs
    ):
        super().__init__(initial_capital, days, **kwargs)
        self.min_price = min_price
        self.max_price = max_price
        self.min_seconds_left = min_seconds_left
        self.max_seconds_left = max_seconds_left
        self.max_spread_pct = max_spread_pct
        
        # Dynamic time window settings
        self.use_dynamic_time = use_dynamic_time
        self.vol_multiplier = vol_multiplier
        self.vol_threshold = vol_threshold
        self.volume_weight = volume_weight
        
        # Hedge configuration
        self.enable_hedging = enable_hedging
        self.hedge_config = hedge_config or HedgeConfig()
        
        # Use realistic execution (no fees, slippage/spread checks)
        self.execution = RealisticExecution(
            max_spread_pct=max_spread_pct,
            buy_slippage_pct=0.003,  # Lower slippage for near-expiry markets
            sell_slippage_pct=0.003,
        )
        
        # Track liquidity estimates
        self._liquidity_cache: Dict[str, float] = {}
        
        # Hedge tracking
        self._all_hedge_trades: List[SimulatedHedgeTrade] = []
        self._hedges_triggered = 0
        self._hedge_pnl = 0.0
        self._loss_avoided_by_hedging = 0.0
    
    @property
    def strategy_name(self) -> str:
        hedge_str = " +Hedge" if self.enable_hedging else ""
        dynamic_str = " +Dynamic" if self.use_dynamic_time else ""
        return f"Bond Strategy ({self.min_price:.0%}-{self.max_price:.0%}){hedge_str}{dynamic_str}"
    
    def calculate_volatility(self, history: List[HistoricalPrice]) -> float:
        """
        Calculate price volatility from history.
        
        Returns standard deviation of returns.
        """
        if len(history) < 3:
            return 0.05  # Default volatility
        
        prices = [p.price for p in history]
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.05
        
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def estimate_volume_proxy(self, history: List[HistoricalPrice]) -> float:
        """
        Estimate relative volume from trade frequency in price history.
        
        More frequent price updates = higher volume.
        Returns a score 0-1 where 1 is high volume.
        """
        if len(history) < 2:
            return 0.5  # Default medium volume
        
        # Calculate average time between price updates
        time_gaps = []
        for i in range(1, len(history)):
            gap = history[i].timestamp - history[i-1].timestamp
            time_gaps.append(gap)
        
        if not time_gaps:
            return 0.5
        
        avg_gap = sum(time_gaps) / len(time_gaps)
        
        # Normalize: <60s gaps = high volume, >3600s gaps = low volume
        # Score inversely proportional to gap
        if avg_gap <= 60:
            return 1.0
        elif avg_gap >= 3600:
            return 0.1
        else:
            # Linear interpolation
            return 1.0 - (avg_gap - 60) / (3600 - 60) * 0.9
    
    def calculate_dynamic_time_window(
        self,
        history: List[HistoricalPrice],
    ) -> Tuple[int, int]:
        """
        Calculate dynamic time window based on volatility and volume.
        
        High volatility + high volume = shorter window (faster resolution expected)
        Low volatility + low volume = longer window (more patience needed)
        
        Returns: (min_seconds, max_seconds)
        """
        if not self.use_dynamic_time:
            return self.min_seconds_left, self.max_seconds_left
        
        volatility = self.calculate_volatility(history)
        volume_score = self.estimate_volume_proxy(history)
        
        # Calculate adjustment factor
        # Higher volatility than threshold = shorter window
        vol_adjustment = 1.0 - self.vol_multiplier * (volatility - self.vol_threshold)
        vol_adjustment = max(0.5, min(2.0, vol_adjustment))  # Clamp to 0.5x-2x
        
        # Higher volume = shorter window (can exit faster)
        volume_adjustment = 1.0 - self.volume_weight * (volume_score - 0.5)
        volume_adjustment = max(0.7, min(1.5, volume_adjustment))
        
        # Combined adjustment
        total_adjustment = (vol_adjustment + volume_adjustment) / 2
        
        # Apply to base windows
        dynamic_min = int(self.min_seconds_left * total_adjustment)
        dynamic_max = int(self.max_seconds_left * total_adjustment)
        
        # Ensure minimum bounds
        dynamic_min = max(30, dynamic_min)
        dynamic_max = max(dynamic_min + 60, dynamic_max)
        
        return dynamic_min, dynamic_max
    
    async def run_strategy(self, markets: List[Market]) -> BacktestResults:
        """Run the bond strategy backtest"""
        start_date = datetime.now(timezone.utc) - timedelta(days=self.days)
        end_date = datetime.now(timezone.utc)
        
        results = BacktestResults(
            strategy_name=self.strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )
        
        # Track equity over time
        equity_history = [(start_date, self.initial_capital)]
        
        markets_traded = 0
        
        # Reset hedge tracking
        self._all_hedge_trades = []
        self._hedges_triggered = 0
        self._hedge_pnl = 0.0
        self._loss_avoided_by_hedging = 0.0
        
        for market in markets:
            if self.verbose:
                logger.info(f"Analyzing: {market.question[:50]}...")
            
            # Get all token histories for this market (needed for hedge simulation)
            token_histories: Dict[str, List[HistoricalPrice]] = {}
            for token in market.tokens:
                history = await self.fetch_price_history(token.token_id)
                if history:
                    token_histories[token.token_id] = history
            
            for token in market.tokens:
                history = token_histories.get(token.token_id)
                if not history:
                    continue
                
                # Find opposite token for hedging
                opposite_token_id = None
                opposite_history = None
                for other_token in market.tokens:
                    if other_token.token_id != token.token_id:
                        opposite_token_id = other_token.token_id
                        opposite_history = token_histories.get(opposite_token_id)
                        break
                
                # Find entry opportunities with hedge simulation
                trade = await self._find_trade_opportunity_with_hedge(
                    market, token, history, results,
                    opposite_token_id, opposite_history
                )
                
                if trade:
                    markets_traded += 1
                    equity_history.append((trade.entry_time, self.cash))
        
        results.markets_traded = markets_traded
        results.equity_curve = equity_history
        
        # Add hedge metrics to results
        results.hedge_trades = self._all_hedge_trades
        results.hedges_triggered = self._hedges_triggered
        results.hedge_pnl = self._hedge_pnl
        results.loss_avoided_by_hedging = self._loss_avoided_by_hedging
        
        results.finalize()
        
        return results
    
    async def _find_trade_opportunity_with_hedge(
        self,
        market: Market,
        token: Token,
        history: List[HistoricalPrice],
        results: BacktestResults,
        opposite_token_id: Optional[str],
        opposite_history: Optional[List[HistoricalPrice]],
    ) -> Optional[object]:
        """Find and simulate a trade opportunity with hedge simulation"""
        if len(history) < 10:
            return None
        
        # Estimate liquidity from price stability
        liquidity_estimate = self.estimate_liquidity(history, self.min_price)
        self._liquidity_cache[token.token_id] = liquidity_estimate
        
        # Calculate dynamic time window if enabled
        min_seconds, max_seconds = self.calculate_dynamic_time_window(history)
        
        # Position cutoff determines what % of market life we trade in
        # Default: trade in last 20% of market life (position_ratio >= 0.8)
        # With dynamic time: adjust based on volatility/volume
        if self.use_dynamic_time:
            volatility = self.calculate_volatility(history)
            volume_score = self.estimate_volume_proxy(history)
            
            # High volatility or high volume = can trade earlier (more activity)
            # Lower the cutoff to look at more of the history
            vol_adj = max(0.0, 0.1 * (volatility / self.vol_threshold - 1.0))
            vol_adj = min(0.2, vol_adj)  # Cap at 0.2 adjustment
            
            volume_adj = 0.1 * (volume_score - 0.5)  # -0.05 to +0.05
            
            position_cutoff = 0.8 - vol_adj - volume_adj
            position_cutoff = max(0.5, min(0.9, position_cutoff))  # Keep between 0.5-0.9
        else:
            position_cutoff = 0.8  # Default: last 20% of market life
        
        # Look for entry points
        for i, point in enumerate(history):
            # Check price range
            if not (self.min_price <= point.price <= self.max_price):
                continue
            
            # Estimate time to expiry at this point (position in history)
            position_ratio = i / len(history)
            if position_ratio < position_cutoff:
                continue
            
            # Estimate spread from price volatility
            recent_prices = [p.price for p in history[max(0, i-10):i+1]]
            if len(recent_prices) >= 2:
                price_range = max(recent_prices) - min(recent_prices)
                estimated_spread = price_range / point.price if point.price > 0 else 0.10
            else:
                estimated_spread = 0.02  # Default 2%
            
            # Check spread acceptability
            if estimated_spread > self.max_spread_pct:
                if self.verbose:
                    logger.debug(f"  Skipped: spread {estimated_spread:.1%} > {self.max_spread_pct:.1%}")
                continue
            
            # Check if we have capital
            if self.cash < self.config.risk.min_trade_value_usd:
                break
            
            # Calculate position size
            position_dollars, kelly = self.calculate_position_size(point.price, self.cash)
            
            if position_dollars <= 0:
                continue
            
            # Cap position by estimated liquidity (max 10% of available)
            max_position = liquidity_estimate * 0.10
            position_dollars = min(position_dollars, max_position)
            
            if position_dollars < self.config.risk.min_trade_value_usd:
                continue
            
            # Simulate execution with liquidity-based slippage
            exec_price, filled_shares, fee = self.execution.execute_buy(
                point.price,
                position_dollars / point.price,
                None,
                liquidity_usd=liquidity_estimate
            )
            
            if filled_shares <= 0:
                continue
            
            cost = filled_shares * exec_price  # No fee
            
            if cost > self.cash:
                continue
            
            # Execute trade
            self.cash -= cost
            entry_time = point.datetime
            
            # Create simulated position for hedge tracking
            position = SimulatedPosition(
                token_id=token.token_id,
                outcome=token.outcome,
                entry_price=exec_price,
                entry_time=entry_time,
                entry_index=i,
                shares=filled_shares,
                cost=cost,
                opposite_token_id=opposite_token_id,
            )
            
            # Simulate position through remaining price history with hedge checks
            final_exit_price, final_exit_shares, exit_reason = await self._simulate_position_with_hedges(
                position, history, opposite_history, liquidity_estimate
            )
            
            exit_time = history[-1].datetime
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
                reason=f"Bond @ {point.price:.2%} (spread: {estimated_spread:.1%}){reason_suffix}"
            )
            
            if self.verbose:
                pnl = proceeds - cost
                logger.info(
                    f"  Trade: {token.outcome} {filled_shares:.2f} @ ${exec_price:.4f} -> "
                    f"${final_exit_price:.4f} P&L: ${pnl:.2f} (liq: ${liquidity_estimate:.0f})"
                )
                if position.hedge_trades:
                    logger.info(f"    Hedge actions: {[h.action.value for h in position.hedge_trades]}")
            
            return results.trades[-1]
        
        return None
    
    async def _simulate_position_with_hedges(
        self,
        position: SimulatedPosition,
        history: List[HistoricalPrice],
        opposite_history: Optional[List[HistoricalPrice]],
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
        unhedged_exit_price = history[-1].price
        
        # Walk through price history after entry
        for j in range(position.entry_index + 1, len(history)):
            current_price = history[j].price
            current_time = history[j].datetime
            
            # Get opposite price at same time (approximate by index ratio)
            opposite_price = None
            if opposite_history and position.opposite_token_id:
                opp_index = int(j * len(opposite_history) / len(history))
                opp_index = min(opp_index, len(opposite_history) - 1)
                opposite_price = opposite_history[opp_index].price
            
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
        exit_price = history[-1].price
        
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


async def run_bond_backtest(
    initial_capital: float = 1000.0,
    days: int = 3,
    verbose: bool = False,
    enable_hedging: bool = True,
    hedge_config: Optional[HedgeConfig] = None,
    # Price range
    min_price: float = 0.95,
    max_price: float = 0.98,
    # Time window
    min_seconds_left: int = 60,
    max_seconds_left: int = 1800,
    # Dynamic time settings
    use_dynamic_time: bool = False,
    vol_multiplier: float = 1.0,
    vol_threshold: float = 0.05,
    volume_weight: float = 0.5,
) -> BacktestResults:
    """
    Run bond strategy backtest with hedge simulation.
    
    Args:
        initial_capital: Starting capital
        days: Number of days to backtest
        verbose: Enable verbose logging
        enable_hedging: Enable hedge simulation
        hedge_config: Hedge configuration (uses defaults if not provided)
        min_price: Minimum entry price (0.80-0.99)
        max_price: Maximum entry price (0.80-0.99)
        min_seconds_left: Minimum time to expiry
        max_seconds_left: Maximum time to expiry
        use_dynamic_time: Enable dynamic time window
        vol_multiplier: Volatility adjustment multiplier
        vol_threshold: Baseline volatility threshold
        volume_weight: Weight for volume in dynamic calculation
    """
    backtester = BondBacktester(
        initial_capital=initial_capital,
        days=days,
        verbose=verbose,
        enable_hedging=enable_hedging,
        hedge_config=hedge_config,
        min_price=min_price,
        max_price=max_price,
        min_seconds_left=min_seconds_left,
        max_seconds_left=max_seconds_left,
        use_dynamic_time=use_dynamic_time,
        vol_multiplier=vol_multiplier,
        vol_threshold=vol_threshold,
        volume_weight=volume_weight,
    )
    
    results = await backtester.run()
    results.print_report()
    
    return results


if __name__ == "__main__":
    import asyncio
    import argparse
    
    parser = argparse.ArgumentParser(description="Bond Strategy Backtester")
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    # Price range
    parser.add_argument("--min-price", type=float, default=0.95, help="Minimum entry price")
    parser.add_argument("--max-price", type=float, default=0.98, help="Maximum entry price")
    # Time window
    parser.add_argument("--min-seconds", type=int, default=60, help="Minimum seconds to expiry")
    parser.add_argument("--max-seconds", type=int, default=1800, help="Maximum seconds to expiry")
    # Dynamic time
    parser.add_argument("--dynamic-time", action="store_true", help="Enable dynamic time window")
    parser.add_argument("--vol-multiplier", type=float, default=1.0, help="Volatility multiplier")
    # Hedging
    parser.add_argument("--no-hedge", action="store_true", help="Disable hedge simulation")
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop-loss threshold (default 15%%)")
    parser.add_argument("--hedge-trigger", type=float, default=0.05, help="Hedge trigger threshold (default 5%%)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    
    # Build hedge config from args
    hedge_config = HedgeConfig(
        stop_loss_pct=args.stop_loss,
        price_drop_trigger_pct=args.hedge_trigger,
    )
    
    asyncio.run(run_bond_backtest(
        initial_capital=args.capital,
        days=args.days,
        verbose=args.verbose,
        enable_hedging=not args.no_hedge,
        hedge_config=hedge_config,
        min_price=args.min_price,
        max_price=args.max_price,
        min_seconds_left=args.min_seconds,
        max_seconds_left=args.max_seconds,
        use_dynamic_time=args.dynamic_time,
        vol_multiplier=args.vol_multiplier,
    ))

