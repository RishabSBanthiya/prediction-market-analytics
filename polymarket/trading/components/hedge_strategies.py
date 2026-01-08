"""
Hedge Strategies - Implementation of hedge execution logic.

Implements the cascading hedge strategies:
1. Arbitrage (YES + NO < 1)
2. Protective hedge (buy opposite outcome)  
3. Partial exit (sell portion of position)
4. Stop-loss (full exit)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from ...core.models import Side, ExecutionResult, Position
from .hedge_monitor import (
    HedgeAction,
    HedgeConfig,
    HedgeRecommendation,
    MonitoredPosition,
)

if TYPE_CHECKING:
    from py_clob_client.client import ClobClient
    from ...core.api import PolymarketAPI
    from .executors import ExecutionEngine

logger = logging.getLogger(__name__)


@dataclass
class HedgeResult:
    """Result of a hedge execution"""
    success: bool
    action: HedgeAction
    position_token_id: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    # Trade details
    trades_executed: int = 0
    total_cost: float = 0.0
    shares_bought: float = 0.0
    shares_sold: float = 0.0
    
    # For arbitrage
    arb_profit_locked: float = 0.0
    
    # For hedge
    hedge_coverage_pct: float = 0.0


class HedgeExecutor:
    """
    Executes hedge strategies based on recommendations.
    
    Uses the same execution engine as the main trading bot.
    """
    
    def __init__(
        self,
        api: "PolymarketAPI",
        executor: "ExecutionEngine",
        client: Optional["ClobClient"] = None,
        config: Optional[HedgeConfig] = None,
    ):
        self.api = api
        self.executor = executor
        self.client = client
        self.config = config or HedgeConfig()
    
    async def execute_hedge(
        self,
        recommendation: HedgeRecommendation,
        available_capital: float,
    ) -> HedgeResult:
        """
        Execute a hedge based on the recommendation.
        
        Args:
            recommendation: The hedge recommendation to execute
            available_capital: Capital available for hedge trades
        
        Returns:
            HedgeResult with execution details
        """
        action = recommendation.action
        monitored = recommendation.position
        
        logger.info(f"{'='*60}")
        logger.info(f"🛡️ EXECUTING HEDGE: {action.value.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"  Position: {monitored.position.outcome}")
        logger.info(f"  Reason: {recommendation.reason}")
        logger.info(f"  Urgency: {recommendation.urgency:.0%}")
        
        try:
            if action == HedgeAction.ARBITRAGE:
                return await self._execute_arbitrage(monitored, available_capital)
            elif action == HedgeAction.PROTECTIVE_HEDGE:
                return await self._execute_protective_hedge(monitored, available_capital)
            elif action == HedgeAction.PARTIAL_EXIT:
                return await self._execute_partial_exit(monitored)
            elif action == HedgeAction.STOP_LOSS:
                return await self._execute_stop_loss(monitored)
            else:
                return HedgeResult(
                    success=False,
                    action=action,
                    position_token_id=monitored.position.token_id,
                    error=f"Unknown action: {action}",
                )
        except Exception as e:
            logger.error(f"Hedge execution failed: {e}")
            return HedgeResult(
                success=False,
                action=action,
                position_token_id=monitored.position.token_id,
                error=str(e),
            )
    
    async def _execute_arbitrage(
        self,
        monitored: MonitoredPosition,
        available_capital: float,
    ) -> HedgeResult:
        """
        Execute arbitrage by buying both YES and NO.
        
        When YES + NO < $1.00, we can buy both and guarantee profit at resolution.
        """
        if monitored.opposite_price is None or monitored.opposite_token_id is None:
            return HedgeResult(
                success=False,
                action=HedgeAction.ARBITRAGE,
                position_token_id=monitored.position.token_id,
                error="No opposite token available for arbitrage",
            )
        
        yes_price = monitored.current_price
        no_price = monitored.opposite_price
        total_cost_per_pair = yes_price + no_price
        
        if total_cost_per_pair >= 1.0:
            return HedgeResult(
                success=False,
                action=HedgeAction.ARBITRAGE,
                position_token_id=monitored.position.token_id,
                error=f"No arbitrage: YES+NO = ${total_cost_per_pair:.4f} >= $1.00",
            )
        
        profit_per_pair = 1.0 - total_cost_per_pair
        
        # Calculate how many pairs we can buy
        # We already have YES shares, so we need to buy NO to match
        shares_to_hedge = monitored.position.shares
        no_cost = shares_to_hedge * no_price
        
        if no_cost > available_capital:
            # Scale down to available capital
            shares_to_hedge = available_capital / no_price
            no_cost = available_capital
        
        logger.info(f"  Arbitrage: Buying {shares_to_hedge:.4f} NO @ ${no_price:.4f}")
        logger.info(f"  Cost: ${no_cost:.2f}, Locked profit: ${shares_to_hedge * profit_per_pair:.2f}")
        
        # Execute the NO buy
        result = await self.executor.execute(
            client=self.client,
            token_id=monitored.opposite_token_id,
            side=Side.BUY,
            size_usd=no_cost,
            price=no_price,
            orderbook=None,
        )
        
        if result.success and result.filled_shares > 0:
            locked_profit = result.filled_shares * profit_per_pair
            
            logger.info(f"{'='*60}")
            logger.info(f"✅ ARBITRAGE EXECUTED")
            logger.info(f"{'='*60}")
            logger.info(f"  NO Shares: {result.filled_shares:.4f}")
            logger.info(f"  Fill Price: ${result.filled_price:.4f}")
            logger.info(f"  Locked Profit: ${locked_profit:.2f}")
            
            return HedgeResult(
                success=True,
                action=HedgeAction.ARBITRAGE,
                position_token_id=monitored.position.token_id,
                trades_executed=1,
                total_cost=result.filled_shares * result.filled_price,
                shares_bought=result.filled_shares,
                arb_profit_locked=locked_profit,
                details={
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "no_filled_price": result.filled_price,
                    "profit_per_share": profit_per_pair,
                }
            )
        
        return HedgeResult(
            success=False,
            action=HedgeAction.ARBITRAGE,
            position_token_id=monitored.position.token_id,
            error=result.error_message or "Failed to fill NO order",
        )
    
    async def _execute_protective_hedge(
        self,
        monitored: MonitoredPosition,
        available_capital: float,
    ) -> HedgeResult:
        """
        Execute protective hedge by buying the opposite outcome.
        
        This limits max loss to the hedge cost regardless of resolution.
        """
        if monitored.opposite_price is None or monitored.opposite_token_id is None:
            return HedgeResult(
                success=False,
                action=HedgeAction.PROTECTIVE_HEDGE,
                position_token_id=monitored.position.token_id,
                error="No opposite token available for hedge",
            )
        
        no_price = monitored.opposite_price
        
        # Calculate hedge size: enough NO to cover potential YES loss
        # If YES resolves to $0, we lose (entry_price * shares)
        # If we hold NO, we gain (1.0 * no_shares)
        # To fully hedge: no_shares = yes_shares * entry_price
        
        potential_loss = monitored.position.shares * monitored.entry_price
        full_hedge_shares = potential_loss  # NO pays $1 at resolution
        
        # Cap hedge cost at configured maximum
        max_hedge_cost = potential_loss * self.config.hedge_cost_max_pct
        hedge_cost = min(full_hedge_shares * no_price, max_hedge_cost, available_capital)
        shares_to_buy = hedge_cost / no_price
        
        coverage_pct = (shares_to_buy / full_hedge_shares) if full_hedge_shares > 0 else 0
        
        logger.info(f"  Protective hedge: Buying {shares_to_buy:.4f} NO @ ${no_price:.4f}")
        logger.info(f"  Cost: ${hedge_cost:.2f}, Coverage: {coverage_pct:.0%}")
        
        result = await self.executor.execute(
            client=self.client,
            token_id=monitored.opposite_token_id,
            side=Side.BUY,
            size_usd=hedge_cost,
            price=no_price,
            orderbook=None,
        )
        
        if result.success and result.filled_shares > 0:
            actual_coverage = result.filled_shares / full_hedge_shares if full_hedge_shares > 0 else 0
            
            logger.info(f"{'='*60}")
            logger.info(f"✅ PROTECTIVE HEDGE EXECUTED")
            logger.info(f"{'='*60}")
            logger.info(f"  NO Shares: {result.filled_shares:.4f}")
            logger.info(f"  Fill Price: ${result.filled_price:.4f}")
            logger.info(f"  Coverage: {actual_coverage:.0%}")
            
            return HedgeResult(
                success=True,
                action=HedgeAction.PROTECTIVE_HEDGE,
                position_token_id=monitored.position.token_id,
                trades_executed=1,
                total_cost=result.filled_shares * result.filled_price,
                shares_bought=result.filled_shares,
                hedge_coverage_pct=actual_coverage,
                details={
                    "no_price": no_price,
                    "no_filled_price": result.filled_price,
                    "potential_loss": potential_loss,
                    "full_hedge_shares": full_hedge_shares,
                }
            )
        
        return HedgeResult(
            success=False,
            action=HedgeAction.PROTECTIVE_HEDGE,
            position_token_id=monitored.position.token_id,
            error=result.error_message or "Failed to fill hedge order",
        )
    
    async def _execute_partial_exit(
        self,
        monitored: MonitoredPosition,
    ) -> HedgeResult:
        """
        Execute partial exit by selling a portion of the position.
        """
        exit_pct = self.config.partial_exit_pct
        shares_to_sell = monitored.position.shares * exit_pct
        
        logger.info(f"  Partial exit: Selling {shares_to_sell:.4f} ({exit_pct:.0%}) @ ${monitored.current_price:.4f}")
        
        result = await self.executor.execute(
            client=self.client,
            token_id=monitored.position.token_id,
            side=Side.SELL,
            size_usd=shares_to_sell * monitored.current_price,
            price=monitored.current_price,
            orderbook=None,
        )
        
        if result.success and result.filled_shares > 0:
            proceeds = result.filled_shares * result.filled_price
            actual_exit_pct = result.filled_shares / monitored.position.shares
            
            logger.info(f"{'='*60}")
            logger.info(f"✅ PARTIAL EXIT EXECUTED")
            logger.info(f"{'='*60}")
            logger.info(f"  Shares Sold: {result.filled_shares:.4f}")
            logger.info(f"  Fill Price: ${result.filled_price:.4f}")
            logger.info(f"  Proceeds: ${proceeds:.2f}")
            logger.info(f"  Exit %: {actual_exit_pct:.0%}")
            
            return HedgeResult(
                success=True,
                action=HedgeAction.PARTIAL_EXIT,
                position_token_id=monitored.position.token_id,
                trades_executed=1,
                shares_sold=result.filled_shares,
                details={
                    "exit_pct": actual_exit_pct,
                    "fill_price": result.filled_price,
                    "proceeds": proceeds,
                    "loss_at_exit": (monitored.entry_price - result.filled_price) * result.filled_shares,
                }
            )
        
        return HedgeResult(
            success=False,
            action=HedgeAction.PARTIAL_EXIT,
            position_token_id=monitored.position.token_id,
            error=result.error_message or "Failed to fill sell order",
        )
    
    async def _execute_stop_loss(
        self,
        monitored: MonitoredPosition,
    ) -> HedgeResult:
        """
        Execute stop-loss by selling the entire position.
        """
        shares_to_sell = monitored.position.shares
        
        logger.info(f"  Stop-loss: Selling ALL {shares_to_sell:.4f} @ ${monitored.current_price:.4f}")
        
        result = await self.executor.execute(
            client=self.client,
            token_id=monitored.position.token_id,
            side=Side.SELL,
            size_usd=shares_to_sell * monitored.current_price,
            price=monitored.current_price,
            orderbook=None,
        )
        
        if result.success and result.filled_shares > 0:
            proceeds = result.filled_shares * result.filled_price
            loss = (monitored.entry_price - result.filled_price) * result.filled_shares
            loss_pct = (monitored.entry_price - result.filled_price) / monitored.entry_price
            
            logger.info(f"{'='*60}")
            logger.info(f"🛑 STOP-LOSS EXECUTED")
            logger.info(f"{'='*60}")
            logger.info(f"  Shares Sold: {result.filled_shares:.4f}")
            logger.info(f"  Fill Price: ${result.filled_price:.4f}")
            logger.info(f"  Proceeds: ${proceeds:.2f}")
            logger.info(f"  Loss: ${loss:.2f} ({loss_pct:.1%})")
            
            return HedgeResult(
                success=True,
                action=HedgeAction.STOP_LOSS,
                position_token_id=monitored.position.token_id,
                trades_executed=1,
                shares_sold=result.filled_shares,
                details={
                    "fill_price": result.filled_price,
                    "proceeds": proceeds,
                    "loss": loss,
                    "loss_pct": loss_pct,
                }
            )
        
        return HedgeResult(
            success=False,
            action=HedgeAction.STOP_LOSS,
            position_token_id=monitored.position.token_id,
            error=result.error_message or "Failed to fill stop-loss order",
        )


# Convenience function for backtesting
def simulate_hedge_decision(
    current_price: float,
    entry_price: float,
    opposite_price: Optional[float],
    config: Optional[HedgeConfig] = None,
) -> Optional[HedgeAction]:
    """
    Simulate what hedge decision would be made given prices.
    
    Useful for backtesting without actual execution.
    """
    config = config or HedgeConfig()
    
    # Calculate loss
    loss_pct = (entry_price - current_price) / entry_price if entry_price > 0 else 0
    
    # Below trigger threshold
    if loss_pct < config.price_drop_trigger_pct:
        return None
    
    # Check arbitrage
    if opposite_price is not None:
        total = current_price + opposite_price
        if total < (1.0 - config.min_arb_profit_pct):
            return HedgeAction.ARBITRAGE
    
    # Check protective hedge
    if opposite_price is not None and opposite_price <= config.no_price_attractive_threshold:
        return HedgeAction.PROTECTIVE_HEDGE
    
    # Check stop-loss
    if loss_pct >= config.stop_loss_pct:
        return HedgeAction.STOP_LOSS
    
    # Partial exit
    return HedgeAction.PARTIAL_EXIT



