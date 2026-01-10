"""
Statistical and Cross-Market Arbitrage Strategy Module.

Supports:
- Pair trading (correlated market mean reversion)
- Multi-outcome sum arbitrage (sum != 100%)
- Conditional probability arbitrage (P(A&B) mispricings)
- Duplicate market arbitrage (same question, different prices)
"""

from .models import (
    ArbType,
    CorrelationType,
    MarketPair,
    MarketCluster,
    MultiOutcomeMarket,
    StatArbOpportunity,
    StatArbPosition,
    StatArbPositionStatus,
)
from .config import (
    CorrelationConfig,
    PairTradingConfig,
    MultiOutcomeConfig,
    DuplicateConfig,
    ConditionalConfig,
    StatArbConfig,
)
from .correlation_engine import CorrelationEngine
from .signals import StatArbSignals
from .position_manager import StatArbPositionManager
from .strategy import StatArbStrategy

__all__ = [
    # Models
    "ArbType",
    "CorrelationType",
    "MarketPair",
    "MarketCluster",
    "MultiOutcomeMarket",
    "StatArbOpportunity",
    "StatArbPosition",
    "StatArbPositionStatus",
    # Config
    "CorrelationConfig",
    "PairTradingConfig",
    "MultiOutcomeConfig",
    "DuplicateConfig",
    "ConditionalConfig",
    "StatArbConfig",
    # Components
    "CorrelationEngine",
    "StatArbSignals",
    "StatArbPositionManager",
    "StatArbStrategy",
]
