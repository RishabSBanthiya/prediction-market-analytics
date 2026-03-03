"""
Exchange client abstract base class.

Every platform (Polymarket, Kalshi, Hyperliquid) implements this interface.
Bots only depend on ExchangeClient - they never import platform-specific code.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..core.enums import ExchangeId
from ..core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ..core.config import ExchangeConfig


class ExchangeAuth(ABC):
    """Authentication for an exchange."""

    @abstractmethod
    async def authenticate(self) -> None:
        """Perform authentication (load keys, sign, etc.)."""
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        pass


class ExchangeClient(ABC):
    """
    Unified exchange interface.

    Every platform implements this. Bots only depend on this.
    Prices are normalized:
    - Binary/event instruments: 0-1
    - Perpetuals: USD
    """

    def __init__(self, config: ExchangeConfig):
        self._config = config
        self._connected = False

    @property
    @abstractmethod
    def exchange_id(self) -> ExchangeId:
        """Which exchange this client connects to."""
        pass

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ==================== LIFECYCLE ====================

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection, authenticate, verify access."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    # ==================== MARKET DATA ====================

    @abstractmethod
    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        """Get available instruments. Filters are exchange-specific."""
        pass

    @abstractmethod
    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        """Get a single instrument by ID."""
        pass

    @abstractmethod
    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        """Get current orderbook."""
        pass

    async def get_midpoint(self, instrument_id: str) -> Optional[float]:
        """Get current midpoint price. Default implementation uses orderbook."""
        book = await self.get_orderbook(instrument_id, depth=1)
        return book.midpoint

    # ==================== TRADING ====================

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        """Cancel an order. Returns True if successfully cancelled."""
        pass

    @abstractmethod
    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        """Cancel all open orders. Returns count cancelled."""
        pass

    @abstractmethod
    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        """Get open orders."""
        pass

    # ==================== ACCOUNT ====================

    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        pass

    @abstractmethod
    async def get_positions(self) -> list[ExchangePosition]:
        """Get all positions."""
        pass

    async def get_position(self, instrument_id: str) -> Optional[ExchangePosition]:
        """Get position for a specific instrument. Default filters get_positions."""
        positions = await self.get_positions()
        for pos in positions:
            if pos.instrument_id == instrument_id:
                return pos
        return None
