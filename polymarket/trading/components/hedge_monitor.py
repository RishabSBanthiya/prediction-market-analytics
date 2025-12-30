"""
Hedge Monitor - Position monitoring for adverse price movements.

Monitors open positions and triggers hedge cascade when adverse
price movements exceed configurable thresholds.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Callable, Any, TYPE_CHECKING
from enum import Enum

from ...core.models import Position, Token

if TYPE_CHECKING:
    from ...core.api import PolymarketAPI

logger = logging.getLogger(__name__)


class HedgeAction(Enum):
    """Type of hedge action taken"""
    NONE = "none"
    ARBITRAGE = "arbitrage"           # YES + NO < 1, buy both
    PROTECTIVE_HEDGE = "protective"   # Buy opposite outcome
    PARTIAL_EXIT = "partial_exit"     # Exit 50% of position
    STOP_LOSS = "stop_loss"           # Full exit


@dataclass
class HedgeConfig:
    """Configuration for hedge monitoring and execution"""
    # Trigger thresholds
    price_drop_trigger_pct: float = 0.05      # 5% drop triggers hedge check
    volatility_spike_trigger: float = 3.0     # 3x normal volatility
    
    # Arbitrage settings
    min_arb_profit_pct: float = 0.02          # Min 2% arb to execute
    
    # Protective hedge settings
    hedge_cost_max_pct: float = 0.10          # Max 10% of position for hedge
    no_price_attractive_threshold: float = 0.15  # NO price must be < 15% for hedge
    
    # Partial exit settings
    partial_exit_pct: float = 0.50            # Exit 50% of position
    
    # Stop-loss settings
    stop_loss_pct: float = 0.15               # Exit at 15% loss from entry
    
    # Monitoring
    check_interval_seconds: float = 5.0
    price_history_window: int = 20            # Data points for volatility calc
    
    # Cooldowns to prevent over-trading
    hedge_cooldown_seconds: float = 60.0      # Min time between hedge actions


@dataclass
class MonitoredPosition:
    """A position being monitored for adverse movements"""
    position: Position
    entry_price: float
    current_price: float
    opposite_token_id: Optional[str] = None
    opposite_price: Optional[float] = None
    price_history: List[float] = field(default_factory=list)
    last_hedge_time: Optional[datetime] = None
    hedge_actions_taken: List[HedgeAction] = field(default_factory=list)
    is_hedged: bool = False
    
    @property
    def price_change_pct(self) -> float:
        """Calculate price change from entry"""
        if self.entry_price > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        return 0.0
    
    @property
    def unrealized_loss_pct(self) -> float:
        """Calculate unrealized loss percentage (negative if profit)"""
        return -self.price_change_pct
    
    @property
    def volatility(self) -> float:
        """Calculate recent price volatility (std dev)"""
        if len(self.price_history) < 2:
            return 0.0
        
        mean = sum(self.price_history) / len(self.price_history)
        variance = sum((p - mean) ** 2 for p in self.price_history) / len(self.price_history)
        return variance ** 0.5
    
    @property
    def arb_opportunity(self) -> float:
        """
        Calculate arbitrage opportunity.
        Returns profit % if YES + NO < 1, else 0.
        """
        if self.opposite_price is None:
            return 0.0
        
        total = self.current_price + self.opposite_price
        if total < 1.0:
            return 1.0 - total  # Guaranteed profit at resolution
        return 0.0


@dataclass
class HedgeRecommendation:
    """Recommendation from the hedge monitor"""
    action: HedgeAction
    position: MonitoredPosition
    reason: str
    urgency: float  # 0-1, higher = more urgent
    details: Dict[str, Any] = field(default_factory=dict)


class HedgeMonitor:
    """
    Monitors positions for adverse price movements and recommends hedges.
    
    Cascade order:
    1. Arbitrage (YES + NO < 1)
    2. Protective hedge (buy opposite outcome)
    3. Partial exit (sell 50%)
    4. Stop-loss (full exit)
    """
    
    def __init__(
        self,
        api: "PolymarketAPI",
        config: Optional[HedgeConfig] = None,
        on_hedge_triggered: Optional[Callable[[HedgeRecommendation], None]] = None,
    ):
        self.api = api
        self.config = config or HedgeConfig()
        self.on_hedge_triggered = on_hedge_triggered
        
        # Tracked positions by token_id
        self._positions: Dict[str, MonitoredPosition] = {}
        
        # Token ID mappings: token_id -> opposite_token_id
        self._opposite_tokens: Dict[str, str] = {}
        
        # Market mappings: market_id -> list of token_ids
        self._market_tokens: Dict[str, List[str]] = {}
        
        # Baseline volatility per token for spike detection
        self._baseline_volatility: Dict[str, float] = {}
        
        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def add_position(
        self,
        position: Position,
        opposite_token_id: Optional[str] = None,
        market_tokens: Optional[List[Token]] = None,
    ):
        """
        Add a position to monitor.
        
        Args:
            position: The position to monitor
            opposite_token_id: Token ID of the opposite outcome (for hedging)
            market_tokens: All tokens in the market (to find opposite)
        """
        # Try to find opposite token if not provided
        if not opposite_token_id and market_tokens:
            for token in market_tokens:
                if token.token_id != position.token_id:
                    opposite_token_id = token.token_id
                    break
        
        monitored = MonitoredPosition(
            position=position,
            entry_price=position.entry_price,
            current_price=position.current_price or position.entry_price,
            opposite_token_id=opposite_token_id,
        )
        
        self._positions[position.token_id] = monitored
        
        if opposite_token_id:
            self._opposite_tokens[position.token_id] = opposite_token_id
        
        if position.market_id:
            if position.market_id not in self._market_tokens:
                self._market_tokens[position.market_id] = []
            if position.token_id not in self._market_tokens[position.market_id]:
                self._market_tokens[position.market_id].append(position.token_id)
        
        logger.info(
            f"📊 Monitoring position: {position.outcome} @ ${position.entry_price:.4f} "
            f"(opposite: {opposite_token_id[:8] if opposite_token_id else 'N/A'}...)"
        )
    
    def remove_position(self, token_id: str):
        """Remove a position from monitoring"""
        if token_id in self._positions:
            del self._positions[token_id]
            logger.info(f"🗑️ Removed position from monitoring: {token_id[:16]}...")
    
    def get_monitored_positions(self) -> List[MonitoredPosition]:
        """Get all monitored positions"""
        return list(self._positions.values())
    
    async def update_prices(self):
        """Update current prices for all monitored positions"""
        for token_id, monitored in self._positions.items():
            try:
                # Get current price from orderbook
                bid, ask, _ = await self.api.get_spread(token_id)
                
                if bid is not None and ask is not None:
                    # Use midpoint as current price
                    current_price = (bid + ask) / 2
                elif bid is not None:
                    current_price = bid
                elif ask is not None:
                    current_price = ask
                else:
                    continue
                
                monitored.current_price = current_price
                
                # Update price history
                monitored.price_history.append(current_price)
                if len(monitored.price_history) > self.config.price_history_window:
                    monitored.price_history.pop(0)
                
                # Update opposite price if available
                if monitored.opposite_token_id:
                    opp_bid, opp_ask, _ = await self.api.get_spread(monitored.opposite_token_id)
                    if opp_bid is not None and opp_ask is not None:
                        monitored.opposite_price = (opp_bid + opp_ask) / 2
                    elif opp_ask is not None:
                        monitored.opposite_price = opp_ask
                
            except Exception as e:
                logger.warning(f"Error updating price for {token_id[:16]}...: {e}")
    
    def check_position(self, monitored: MonitoredPosition) -> Optional[HedgeRecommendation]:
        """
        Check a single position for hedge triggers.
        
        Returns a HedgeRecommendation if action is needed, None otherwise.
        """
        # Check cooldown
        if monitored.last_hedge_time:
            elapsed = (datetime.now(timezone.utc) - monitored.last_hedge_time).total_seconds()
            if elapsed < self.config.hedge_cooldown_seconds:
                return None
        
        # Already fully hedged
        if monitored.is_hedged:
            return None
        
        loss_pct = monitored.unrealized_loss_pct
        
        # Check if loss exceeds trigger threshold
        if loss_pct < self.config.price_drop_trigger_pct:
            # Also check volatility spike
            if monitored.volatility > 0:
                baseline = self._baseline_volatility.get(
                    monitored.position.token_id, 
                    monitored.volatility
                )
                if monitored.volatility < baseline * self.config.volatility_spike_trigger:
                    return None
            else:
                return None
        
        logger.warning(
            f"⚠️ Adverse movement detected: {monitored.position.outcome} "
            f"down {loss_pct:.1%} from ${monitored.entry_price:.4f}"
        )
        
        # CASCADE 1: Check for arbitrage opportunity
        arb_profit = monitored.arb_opportunity
        if arb_profit >= self.config.min_arb_profit_pct:
            return HedgeRecommendation(
                action=HedgeAction.ARBITRAGE,
                position=monitored,
                reason=f"Arbitrage: YES+NO={monitored.current_price + (monitored.opposite_price or 0):.3f} < $1.00",
                urgency=min(1.0, arb_profit / 0.10),  # Higher profit = more urgent
                details={
                    "yes_price": monitored.current_price,
                    "no_price": monitored.opposite_price,
                    "profit_pct": arb_profit,
                }
            )
        
        # CASCADE 2: Check for protective hedge (buy opposite)
        if monitored.opposite_price is not None:
            # NO should be cheap enough to hedge
            if monitored.opposite_price <= self.config.no_price_attractive_threshold:
                # Calculate hedge cost vs potential loss
                potential_loss = monitored.position.shares * (monitored.entry_price - monitored.current_price)
                hedge_cost = monitored.position.shares * monitored.opposite_price
                
                if hedge_cost <= potential_loss * 2:  # Hedge if cost < 2x current loss
                    return HedgeRecommendation(
                        action=HedgeAction.PROTECTIVE_HEDGE,
                        position=monitored,
                        reason=f"Protective hedge: NO @ ${monitored.opposite_price:.4f}",
                        urgency=min(1.0, loss_pct / self.config.stop_loss_pct),
                        details={
                            "opposite_price": monitored.opposite_price,
                            "hedge_cost": hedge_cost,
                            "potential_loss": potential_loss,
                        }
                    )
        
        # CASCADE 3: Check for partial exit
        if loss_pct < self.config.stop_loss_pct:
            # Not at stop-loss yet, partial exit to reduce exposure
            if HedgeAction.PARTIAL_EXIT not in monitored.hedge_actions_taken:
                return HedgeRecommendation(
                    action=HedgeAction.PARTIAL_EXIT,
                    position=monitored,
                    reason=f"Partial exit: reduce exposure at {loss_pct:.1%} loss",
                    urgency=loss_pct / self.config.stop_loss_pct,
                    details={
                        "exit_pct": self.config.partial_exit_pct,
                        "current_loss_pct": loss_pct,
                    }
                )
        
        # CASCADE 4: Stop-loss
        if loss_pct >= self.config.stop_loss_pct:
            return HedgeRecommendation(
                action=HedgeAction.STOP_LOSS,
                position=monitored,
                reason=f"Stop-loss triggered at {loss_pct:.1%} loss",
                urgency=1.0,
                details={
                    "loss_pct": loss_pct,
                    "stop_loss_threshold": self.config.stop_loss_pct,
                }
            )
        
        return None
    
    async def check_all_positions(self) -> List[HedgeRecommendation]:
        """Check all positions and return recommendations"""
        await self.update_prices()
        
        recommendations = []
        for monitored in self._positions.values():
            rec = self.check_position(monitored)
            if rec:
                recommendations.append(rec)
                
                # Notify callback if set
                if self.on_hedge_triggered:
                    try:
                        self.on_hedge_triggered(rec)
                    except Exception as e:
                        logger.error(f"Error in hedge callback: {e}")
        
        return recommendations
    
    def mark_hedge_executed(self, token_id: str, action: HedgeAction):
        """Mark that a hedge action was executed for a position"""
        if token_id in self._positions:
            monitored = self._positions[token_id]
            monitored.last_hedge_time = datetime.now(timezone.utc)
            monitored.hedge_actions_taken.append(action)
            
            if action in (HedgeAction.ARBITRAGE, HedgeAction.STOP_LOSS):
                monitored.is_hedged = True
            
            logger.info(f"✅ Hedge executed: {action.value} for {token_id[:16]}...")
    
    async def start(self):
        """Start the monitoring loop"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🔍 Hedge monitor started")
    
    async def stop(self):
        """Stop the monitoring loop"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Hedge monitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                if self._positions:
                    recommendations = await self.check_all_positions()
                    
                    if recommendations:
                        logger.info(f"📋 {len(recommendations)} hedge recommendations pending")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            await asyncio.sleep(self.config.check_interval_seconds)
    
    def update_baseline_volatility(self, token_id: str, volatility: float):
        """Update baseline volatility for a token"""
        self._baseline_volatility[token_id] = volatility

