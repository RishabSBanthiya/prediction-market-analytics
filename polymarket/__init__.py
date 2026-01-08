"""
Polymarket trading system module.

Main package containing all trading infrastructure, strategies, and utilities.
"""

from . import core
from . import trading
from . import strategies
from . import backtesting

__all__ = [
    "core",
    "trading",
    "strategies",
    "backtesting",
]



