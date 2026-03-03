"""
Inventory management for market making.

Tracks position inventory and provides risk metrics.
"""

import logging
from typing import Optional

from ..core.enums import Side
from ..exchanges.base import ExchangeClient

logger = logging.getLogger(__name__)


class InventoryManager:
    """
    Tracks inventory across instruments for a market making bot.

    Provides current inventory levels and whether the bot should
    skew quotes to reduce exposure.
    """

    def __init__(self, max_inventory_usd: float = 500.0):
        self.max_inventory_usd = max_inventory_usd
        self._inventory: dict[str, float] = {}  # instrument_id -> signed USD exposure

    def update_from_fill(self, instrument_id: str, side: Side, size_usd: float) -> None:
        """Update inventory after a fill."""
        current = self._inventory.get(instrument_id, 0.0)
        if side == Side.BUY:
            self._inventory[instrument_id] = current + size_usd
        else:
            self._inventory[instrument_id] = current - size_usd

    def get_inventory(self, instrument_id: str) -> float:
        """Get current inventory in USD (positive = long, negative = short)."""
        return self._inventory.get(instrument_id, 0.0)

    def get_inventory_ratio(self, instrument_id: str) -> float:
        """Inventory as fraction of max (-1 to 1)."""
        inv = self.get_inventory(instrument_id)
        if self.max_inventory_usd <= 0:
            return 0.0
        return max(-1.0, min(1.0, inv / self.max_inventory_usd))

    @property
    def total_exposure(self) -> float:
        """Total absolute exposure across all instruments."""
        return sum(abs(v) for v in self._inventory.values())

    def is_at_limit(self, instrument_id: str) -> bool:
        """Check if inventory is at max."""
        return abs(self.get_inventory(instrument_id)) >= self.max_inventory_usd

    async def sync_from_exchange(self, client: ExchangeClient) -> None:
        """Sync inventory from exchange positions."""
        positions = await client.get_positions()
        self._inventory.clear()
        for pos in positions:
            value = pos.size * pos.current_price if pos.current_price else pos.size * pos.entry_price
            if pos.side == Side.SELL:
                value = -value
            self._inventory[pos.instrument_id] = value
        logger.info(f"Synced inventory: {len(positions)} positions, ${self.total_exposure:.2f} total")
