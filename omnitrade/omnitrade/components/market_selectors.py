"""
Market selectors for bots.

Filters and ranks instruments to determine which to trade.
"""

from abc import ABC, abstractmethod

from ..core.models import Instrument
from ..exchanges.base import ExchangeClient


class MarketSelector(ABC):
    """Abstract base for market selection."""

    @abstractmethod
    async def select(self, client: ExchangeClient) -> list[Instrument]:
        """Select instruments to trade. Called periodically."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ActiveMarketSelector(MarketSelector):
    """Select all active instruments matching filters."""

    def __init__(
        self, min_price: float = 0.05, max_price: float = 0.95,
        max_instruments: int = 20,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.max_instruments = max_instruments

    @property
    def name(self) -> str:
        return "active_markets"

    async def select(self, client: ExchangeClient) -> list[Instrument]:
        instruments = await client.get_instruments(active_only=True)
        filtered = [
            i for i in instruments
            if self.min_price <= i.price <= self.max_price
        ]
        # Sort by most liquid (tightest spread)
        filtered.sort(key=lambda i: abs(i.ask - i.bid) if i.ask > 0 and i.bid > 0 else float('inf'))
        return filtered[:self.max_instruments]
