"""
Quote engines for market making bots.

Generate two-sided quotes (bid + ask) for an instrument.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..core.models import Quote, OrderbookSnapshot
from ..exchanges.base import ExchangeClient


class QuoteEngine(ABC):
    """Abstract base for quote generation."""

    @abstractmethod
    async def generate_quote(
        self,
        client: ExchangeClient,
        instrument_id: str,
        inventory: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> Optional[Quote]:
        """Generate a two-sided quote. Returns None to skip."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class SimpleSpreadQuoter(QuoteEngine):
    """
    Simple fixed-spread market making.

    Places bids and asks around the midpoint with configurable spread.
    Skews quotes based on inventory to manage risk.
    """

    def __init__(
        self, half_spread: float = 0.02, size_usd: float = 25.0,
        inventory_skew: float = 0.5, max_inventory: float = 500.0,
    ):
        self.half_spread = half_spread
        self.size_usd = size_usd
        self.inventory_skew = inventory_skew
        self.max_inventory = max_inventory

    @property
    def name(self) -> str:
        return "simple_spread"

    async def generate_quote(
        self,
        client: ExchangeClient,
        instrument_id: str,
        inventory: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> Optional[Quote]:
        if orderbook is None:
            orderbook = await client.get_orderbook(instrument_id, depth=5)

        mid = orderbook.midpoint
        if mid is None or mid <= 0:
            return None

        # Inventory skew: shift quotes to reduce inventory
        skew = 0.0
        if abs(inventory) > 0 and self.max_inventory > 0:
            skew = (inventory / self.max_inventory) * self.inventory_skew * self.half_spread

        bid_price = mid - self.half_spread - skew
        ask_price = mid + self.half_spread - skew

        if bid_price <= 0 or ask_price <= 0:
            return None

        bid_size = self.size_usd / bid_price
        ask_size = self.size_usd / ask_price

        return Quote(
            instrument_id=instrument_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )
