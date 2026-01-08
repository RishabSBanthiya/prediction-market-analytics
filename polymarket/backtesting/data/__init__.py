"""
Data fetching modules for backtesting.

Provides access to:
- Trade history with wallet addresses
- On-chain wallet validation
- SQLite caching for fast re-runs
- Cached data fetcher
"""

from .trade_fetcher import TradeFetcher, TradeData
from .cache import BacktestDataCache, get_cache
from .cached_fetcher import CachedDataFetcher, warmup_cache

__all__ = [
    "TradeFetcher",
    "TradeData",
    "BacktestDataCache",
    "get_cache",
    "CachedDataFetcher",
    "warmup_cache",
]

