"""
Composable trading components.

These are pluggable components that can be mixed and matched
to create different trading strategies:

- SignalSource: Where trading signals come from
- PositionSizer: How to size positions
- ExecutionEngine: How to execute trades
"""

from .signals import SignalSource, ExpiringMarketSignals, FlowAlertSignals
from .sizers import PositionSizer, KellyPositionSizer, SignalScaledSizer, FixedSizer
from .executors import ExecutionEngine, AggressiveExecutor, LimitOrderExecutor
from .hedge_monitor import (
    HedgeMonitor,
    HedgeConfig,
    HedgeAction,
    HedgeRecommendation,
    MonitoredPosition,
)
from .hedge_strategies import HedgeExecutor, HedgeResult, simulate_hedge_decision

__all__ = [
    # Signal sources
    "SignalSource",
    "ExpiringMarketSignals",
    "FlowAlertSignals",
    # Position sizers
    "PositionSizer",
    "KellyPositionSizer",
    "SignalScaledSizer",
    "FixedSizer",
    # Execution engines
    "ExecutionEngine",
    "AggressiveExecutor",
    "LimitOrderExecutor",
    # Hedge components
    "HedgeMonitor",
    "HedgeConfig",
    "HedgeAction",
    "HedgeRecommendation",
    "MonitoredPosition",
    "HedgeExecutor",
    "HedgeResult",
    "simulate_hedge_decision",
]


