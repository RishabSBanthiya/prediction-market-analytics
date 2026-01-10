"""
Arbitrage opportunity scanners.

Each scanner specializes in detecting a specific type of arbitrage opportunity.
"""

from .multi_outcome import MultiOutcomeScanner
from .duplicate import DuplicateMarketScanner
from .pair_spread import PairSpreadScanner
from .conditional import ConditionalProbScanner

__all__ = [
    "MultiOutcomeScanner",
    "DuplicateMarketScanner",
    "PairSpreadScanner",
    "ConditionalProbScanner",
]
