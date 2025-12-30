"""
Data fetching modules for backtesting.

Provides access to:
- Trade history with wallet addresses
- On-chain wallet validation
"""

from .trade_fetcher import TradeFetcher, TradeData

__all__ = ["TradeFetcher", "TradeData"]

