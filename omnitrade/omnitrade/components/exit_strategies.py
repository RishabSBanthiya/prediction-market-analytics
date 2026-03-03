"""
Exit strategy module.

Configurable exit strategies for position management:
- Take-profit, trailing stop, stop-loss, time-based
- Near-resolution for binary markets
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Tuple

from ..core.enums import ExitReason
from ..core.models import PositionState

logger = logging.getLogger(__name__)


@dataclass
class ExitConfig:
    """Exit strategy configuration. Defaults optimized via Bayesian search."""

    # Near-resolution
    near_resolution_enabled: bool = True
    near_resolution_high: float = 0.99
    near_resolution_low: float = 0.01

    # Take-profit
    take_profit_enabled: bool = True
    take_profit_pct: float = 0.05

    # Trailing stop
    trailing_stop_enabled: bool = True
    trailing_stop_activation_pct: float = 0.02
    trailing_stop_distance_pct: float = 0.01

    # Time-based
    time_exit_enabled: bool = True
    max_hold_minutes: int = 75

    # Stop-loss
    stop_loss_enabled: bool = True
    stop_loss_pct: float = 0.25

    def __post_init__(self):
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")
        if self.trailing_stop_distance_pct <= 0:
            raise ValueError("trailing_stop_distance_pct must be positive")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")


class ExitMonitor:
    """
    Monitors positions and determines when to exit.

    Priority: near-resolution > take-profit > trailing stop > stop-loss > time limit
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        self.config = config or ExitConfig()
        self._states: dict[str, PositionState] = {}

    def register(self, position_id: str, state: PositionState) -> None:
        self._states[position_id] = state

    def unregister(self, position_id: str) -> None:
        self._states.pop(position_id, None)

    def get_state(self, position_id: str) -> Optional[PositionState]:
        return self._states.get(position_id)

    def check(
        self, state: PositionState, current_price: float, current_time: datetime,
    ) -> Optional[Tuple[ExitReason, float, str]]:
        """
        Check exit conditions.
        Returns (reason, exit_price, description) or None.
        """
        if state.entry_price <= 0:
            return None

        # Update tracking
        self._update_state(state, current_price)

        ret = (current_price - state.entry_price) / state.entry_price

        # 1. Near-resolution
        if self.config.near_resolution_enabled:
            if current_price >= self.config.near_resolution_high:
                return (ExitReason.NEAR_RESOLUTION, current_price,
                        f"Price {current_price:.4f} >= {self.config.near_resolution_high}")
            if current_price <= self.config.near_resolution_low:
                return (ExitReason.NEAR_RESOLUTION, current_price,
                        f"Price {current_price:.4f} <= {self.config.near_resolution_low}")

        # 2. Take-profit
        if self.config.take_profit_enabled and ret >= self.config.take_profit_pct:
            return (ExitReason.TAKE_PROFIT, current_price, f"Return {ret:.1%} >= {self.config.take_profit_pct:.1%}")

        # 3. Trailing stop
        if self.config.trailing_stop_enabled and state.trailing_stop_activated:
            if current_price <= state.trailing_stop_level:
                return (ExitReason.TRAILING_STOP, current_price,
                        f"Price {current_price:.4f} <= stop {state.trailing_stop_level:.4f}")

        # 4. Stop-loss
        if self.config.stop_loss_enabled and ret <= -self.config.stop_loss_pct:
            return (ExitReason.STOP_LOSS, current_price, f"Loss {ret:.1%} <= -{self.config.stop_loss_pct:.1%}")

        # 5. Time limit
        if self.config.time_exit_enabled:
            hold_time = (current_time - state.entry_time).total_seconds() / 60
            if hold_time >= self.config.max_hold_minutes:
                return (ExitReason.TIME_LIMIT, current_price, f"Held {hold_time:.0f} min >= {self.config.max_hold_minutes}")

        return None

    def _update_state(self, state: PositionState, price: float) -> None:
        """Update peak/trough and trailing stop."""
        state.peak_price = max(state.peak_price, price)
        state.trough_price = min(state.trough_price, price)

        if self.config.trailing_stop_enabled and not state.trailing_stop_activated:
            gain = (price - state.entry_price) / state.entry_price
            if gain >= self.config.trailing_stop_activation_pct:
                state.trailing_stop_activated = True
                state.trailing_stop_level = price * (1 - self.config.trailing_stop_distance_pct)

        if state.trailing_stop_activated:
            new_stop = price * (1 - self.config.trailing_stop_distance_pct)
            if new_stop > state.trailing_stop_level:
                state.trailing_stop_level = new_stop
