"""
Backtesting package - Historical strategy testing infrastructure.

Contains:
- base: BaseBacktester class with common functionality
- results: BacktestResults and SimulatedTrade dataclasses
- execution: Simulated execution with fees and slippage
- optimization: Anti-overfitting Bayesian optimization (3 params, walk-forward CV)
- strategies: Bond and flow strategy backtesters
- data: Data caching and fetching utilities
"""

from .base import BaseBacktester
from .results import BacktestResults, SimulatedTrade
from .execution import SimulatedExecution

__all__ = [
    "BaseBacktester",
    "BacktestResults",
    "SimulatedTrade",
    "SimulatedExecution",
]

