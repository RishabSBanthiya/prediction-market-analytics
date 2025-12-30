"""
Exit Strategy Module.

Provides configurable exit strategies for position management:
- Take-profit: Exit when target profit reached
- Trailing stop: Lock in profits with dynamic stop-loss
- Time-based: Force exit after maximum hold time
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exiting a position."""
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_LIMIT = "time_limit"
    STOP_LOSS = "stop_loss"
    MARKET_CLOSE = "market_close"
    HEDGE_EXIT = "hedge_exit"
    MANUAL = "manual"


@dataclass
class ExitConfig:
    """
    Configuration for exit strategies.
    
    All percentages are expressed as decimals (e.g., 0.03 = 3%).
    
    Default values optimized via Bayesian optimization:
    - 53.83% return, 72.5% win rate, 24x profit factor
    """
    # Take-profit settings
    take_profit_enabled: bool = True
    take_profit_pct: float = 0.05  # Exit at +5% profit (optimized)
    
    # Trailing stop settings
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.02  # Activate after +2% gain (optimized)
    trailing_stop_distance_pct: float = 0.01    # Trail by 1% from peak (optimized)
    
    # Time-based exit settings
    time_exit_enabled: bool = True
    max_hold_minutes: int = 75  # Force exit after 75 minutes (optimized)
    
    # Basic stop-loss (before trailing activates)
    stop_loss_enabled: bool = True
    stop_loss_pct: float = 0.25  # Exit at -25% loss (optimized - wider to let winners run)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.trailing_stop_distance_pct <= 0:
            raise ValueError("trailing_stop_distance_pct must be positive")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")


@dataclass
class PositionState:
    """
    Tracks state needed for exit monitoring.
    
    Used to track peak price for trailing stops, entry time for time-based exits, etc.
    """
    entry_price: float
    entry_time: datetime
    shares: float
    
    # Tracking
    peak_price: float = 0.0  # Highest price since entry
    trough_price: float = float('inf')  # Lowest price since entry
    trailing_stop_activated: bool = False
    trailing_stop_level: float = 0.0  # Current trailing stop price
    
    def __post_init__(self):
        """Initialize peak/trough to entry price."""
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.trough_price == float('inf'):
            self.trough_price = self.entry_price
    
    def update_price(self, current_price: float, config: ExitConfig) -> None:
        """
        Update tracked prices with new price point.
        
        Args:
            current_price: Current market price
            config: Exit configuration for trailing stop logic
        """
        # Update peak/trough
        self.peak_price = max(self.peak_price, current_price)
        self.trough_price = min(self.trough_price, current_price)
        
        # Check if trailing stop should activate
        if config.trailing_stop_enabled and not self.trailing_stop_activated:
            gain_pct = (current_price - self.entry_price) / self.entry_price
            if gain_pct >= config.trailing_stop_activation_pct:
                self.trailing_stop_activated = True
                self.trailing_stop_level = current_price * (1 - config.trailing_stop_distance_pct)
                logger.debug(
                    f"Trailing stop activated at {current_price:.4f}, "
                    f"stop level: {self.trailing_stop_level:.4f}"
                )
        
        # Update trailing stop level if activated
        if self.trailing_stop_activated:
            new_stop = current_price * (1 - config.trailing_stop_distance_pct)
            if new_stop > self.trailing_stop_level:
                self.trailing_stop_level = new_stop
    
    @property
    def current_return_pct(self) -> float:
        """Current return percentage based on peak price."""
        if self.entry_price <= 0:
            return 0.0
        return (self.peak_price - self.entry_price) / self.entry_price
    
    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown from peak."""
        if self.peak_price <= 0:
            return 0.0
        return (self.peak_price - self.trough_price) / self.peak_price


class ExitMonitor:
    """
    Monitors positions and determines when to exit.
    
    Checks multiple exit conditions in priority order:
    1. Take-profit (lock in gains)
    2. Trailing stop (protect profits)
    3. Stop-loss (limit losses)
    4. Time-based (maximum hold time)
    """
    
    def __init__(self, config: Optional[ExitConfig] = None):
        """
        Initialize exit monitor.
        
        Args:
            config: Exit strategy configuration. Uses defaults if not provided.
        """
        self.config = config or ExitConfig()
        self._position_states: dict = {}  # position_id -> PositionState
    
    def register_position(
        self,
        position_id: str,
        entry_price: float,
        entry_time: datetime,
        shares: float,
    ) -> PositionState:
        """
        Register a new position for monitoring.
        
        Args:
            position_id: Unique identifier for the position
            entry_price: Entry price
            entry_time: Entry timestamp
            shares: Number of shares
            
        Returns:
            PositionState object for tracking
        """
        state = PositionState(
            entry_price=entry_price,
            entry_time=entry_time,
            shares=shares,
        )
        self._position_states[position_id] = state
        return state
    
    def get_state(self, position_id: str) -> Optional[PositionState]:
        """Get state for a registered position."""
        return self._position_states.get(position_id)
    
    def remove_position(self, position_id: str) -> None:
        """Remove a position from monitoring."""
        self._position_states.pop(position_id, None)
    
    def check_exit_conditions(
        self,
        state: PositionState,
        current_price: float,
        current_time: datetime,
    ) -> Optional[Tuple[ExitReason, float, str]]:
        """
        Check all exit conditions for a position.
        
        Args:
            state: Position state being monitored
            current_price: Current market price
            current_time: Current timestamp
            
        Returns:
            Tuple of (ExitReason, exit_price, description) if exit triggered,
            None if position should continue
        """
        if state.entry_price <= 0:
            return None
        
        # Update state with current price
        state.update_price(current_price, self.config)
        
        # Calculate current P&L
        current_return = (current_price - state.entry_price) / state.entry_price
        
        # 1. Check take-profit
        if self.config.take_profit_enabled:
            if current_return >= self.config.take_profit_pct:
                return (
                    ExitReason.TAKE_PROFIT,
                    current_price,
                    f"Take-profit triggered at {current_return:.2%} gain"
                )
        
        # 2. Check trailing stop
        if self.config.trailing_stop_enabled and state.trailing_stop_activated:
            if current_price <= state.trailing_stop_level:
                locked_profit = (state.trailing_stop_level - state.entry_price) / state.entry_price
                return (
                    ExitReason.TRAILING_STOP,
                    current_price,
                    f"Trailing stop triggered at {current_price:.4f} "
                    f"(locked {locked_profit:.2%} profit)"
                )
        
        # 3. Check stop-loss (only if trailing not activated)
        if self.config.stop_loss_enabled and not state.trailing_stop_activated:
            if current_return <= -self.config.stop_loss_pct:
                return (
                    ExitReason.STOP_LOSS,
                    current_price,
                    f"Stop-loss triggered at {current_return:.2%} loss"
                )
        
        # 4. Check time-based exit
        if self.config.time_exit_enabled:
            hold_duration = current_time - state.entry_time
            max_hold = timedelta(minutes=self.config.max_hold_minutes)
            
            if hold_duration >= max_hold:
                return (
                    ExitReason.TIME_LIMIT,
                    current_price,
                    f"Time limit reached ({hold_duration.total_seconds() / 60:.1f} min)"
                )
        
        return None
    
    def check_exit_from_history(
        self,
        state: PositionState,
        price_history: List[dict],
        start_index: int = 0,
    ) -> Optional[Tuple[ExitReason, float, str, int, datetime]]:
        """
        Check exit conditions against price history.
        
        Used in backtesting to simulate exit triggers.
        
        Args:
            state: Position state
            price_history: List of {t: timestamp, p: price} dicts
            start_index: Index to start checking from
            
        Returns:
            Tuple of (ExitReason, exit_price, description, exit_index, exit_time)
            if exit triggered, None if no exit in history
        """
        for i in range(start_index, len(price_history)):
            point = price_history[i]
            
            price = point.get('p', 0)
            ts = point.get('t', 0)
            
            if price <= 0 or ts <= 0:
                continue
            
            current_time = datetime.fromtimestamp(ts)
            
            result = self.check_exit_conditions(state, price, current_time)
            
            if result:
                reason, exit_price, description = result
                return (reason, exit_price, description, i, current_time)
        
        return None


def create_default_exit_config() -> ExitConfig:
    """Create default exit configuration with optimized values.
    
    Optimized via Bayesian optimization:
    - 53.83% return, 72.5% win rate, 24x profit factor
    """
    return ExitConfig(
        take_profit_pct=0.05,  # 5% take-profit (optimized)
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=0.02,  # 2% activation (optimized)
        trailing_stop_distance_pct=0.01,  # 1% trail distance (optimized)
        max_hold_minutes=75,  # 75 min max hold (optimized)
        stop_loss_pct=0.25,  # 25% stop-loss (optimized)
    )


def create_aggressive_exit_config() -> ExitConfig:
    """Create aggressive exit configuration (quick profits, tight stops)."""
    return ExitConfig(
        take_profit_pct=0.02,  # +2%
        trailing_stop_activation_pct=0.01,  # Activate at +1%
        trailing_stop_distance_pct=0.005,   # 0.5% trail
        max_hold_minutes=30,
        stop_loss_pct=0.05,  # -5%
    )


def create_conservative_exit_config() -> ExitConfig:
    """Create conservative exit configuration (larger targets, wider stops)."""
    return ExitConfig(
        take_profit_pct=0.05,  # +5%
        trailing_stop_activation_pct=0.03,  # Activate at +3%
        trailing_stop_distance_pct=0.02,    # 2% trail
        max_hold_minutes=120,
        stop_loss_pct=0.15,  # -15%
    )

