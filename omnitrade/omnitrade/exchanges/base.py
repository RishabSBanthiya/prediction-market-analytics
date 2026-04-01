"""
Exchange client abstract base class.

Every platform (Polymarket, Kalshi, Hyperliquid) implements this interface.
Bots only depend on ExchangeClient - they never import platform-specific code.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..core.enums import ExchangeId, OrderStatus, Side
from ..core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ..core.config import ExchangeConfig

logger = logging.getLogger(__name__)


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

    async def cancel_orders(self, order_ids: list[str]) -> int:
        """Cancel a batch of orders by ID. Returns count cancelled. Default: sequential."""
        cancelled = 0
        for oid in order_ids:
            if await self.cancel_order(oid):
                cancelled += 1
        return cancelled

    @abstractmethod
    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        """Cancel all open orders. Returns count cancelled."""
        pass

    @abstractmethod
    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        """Get open orders."""
        pass

    async def get_order_status(self, order_id: str, instrument_id: str = "") -> Optional[OrderResult]:
        """Poll the current state of an order by ID.

        Returns an OrderResult reflecting the latest fill state, or None if the
        order cannot be found.  Subclasses should override this when the
        exchange provides an order-status endpoint.  The default implementation
        returns None (not supported).
        """
        return None

    async def amend_order(
        self,
        order_id: str,
        instrument_id: str = "",
        new_price: Optional[float] = None,
        new_size: Optional[float] = None,
    ) -> OrderResult:
        """Amend an existing order's price and/or size.

        Not all exchanges support amendment.  The default implementation
        cancels the old order and places a new one, which is **not** atomic
        and may result in both orders being filled in a race.  Subclasses
        should override this when the exchange provides a native amend
        endpoint.

        Returns the OrderResult for the replacement order.
        """
        cancelled = await self.cancel_order(order_id, instrument_id)
        if not cancelled:
            return OrderResult(
                success=False,
                error_message=f"Failed to cancel order {order_id} for amendment",
            )
        # We need at least one of price or size to place a new order.
        if new_price is None and new_size is None:
            return OrderResult(
                success=False,
                error_message="amend_order requires at least new_price or new_size",
            )
        # Build a minimal replacement request.  Callers using the default
        # cancel-replace path should prefer the dedicated amend helpers on
        # OrderTracker which carry forward the full original request.
        request = OrderRequest(
            instrument_id=instrument_id,
            side=Side.BUY,  # Will be overridden by subclass or tracker
            size=new_size or 0.0,
            price=new_price or 0.0,
        )
        return await self.place_order(request)

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


class PaperClient(ExchangeClient):
    """
    Paper trading wrapper around any ExchangeClient.

    Simulates order fills locally using orderbook prices + configurable slippage,
    without placing real orders. Delegates all read operations (market data, balance,
    positions) to the underlying client.

    Usage:
        client = PaperClient(real_client)
        # All bots use client.place_order() as normal — fills are simulated.
    """

    def __init__(self, client: ExchangeClient, slippage_pct: float = 0.001):
        # Skip ExchangeClient.__init__ — we delegate everything to the wrapped client
        self._client = client
        self.slippage_pct = slippage_pct
        self._order_counter = 0

    @property
    def exchange_id(self) -> ExchangeId:
        return self._client.exchange_id

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    # ==================== LIFECYCLE ====================

    async def connect(self) -> None:
        await self._client.connect()

    async def close(self) -> None:
        await self._client.close()

    # ==================== MARKET DATA (delegate) ====================

    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        return await self._client.get_instruments(active_only=active_only, **filters)

    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        return await self._client.get_instrument(instrument_id)

    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        return await self._client.get_orderbook(instrument_id, depth=depth)

    async def get_midpoint(self, instrument_id: str) -> Optional[float]:
        return await self._client.get_midpoint(instrument_id)

    # ==================== TRADING (simulated) ====================

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Simulate a fill using orderbook prices + slippage."""
        orderbook = await self._client.get_orderbook(request.instrument_id, depth=1)

        if request.side == Side.BUY:
            exec_price = orderbook.best_ask or request.price
            exec_price *= (1 + self.slippage_pct)
        else:
            exec_price = orderbook.best_bid or request.price
            exec_price *= (1 - self.slippage_pct)

        if exec_price <= 0:
            return OrderResult(success=False, error_message="No price available")

        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        logger.info(
            "[PAPER] %s %.4f @ $%.4f (requested %.4f @ $%.4f) instrument=%s",
            request.side.value, request.size, exec_price,
            request.size, request.price, request.instrument_id,
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_size=request.size,
            filled_price=exec_price,
            requested_size=request.size,
            requested_price=request.price,
        )

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        return True

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        return 0

    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        return []

    async def get_order_status(self, order_id: str, instrument_id: str = "") -> Optional[OrderResult]:
        """Paper orders are always immediately filled, so status returns None."""
        return None

    async def amend_order(
        self,
        order_id: str,
        instrument_id: str = "",
        new_price: Optional[float] = None,
        new_size: Optional[float] = None,
    ) -> OrderResult:
        """Paper amendment: simulate a new fill at amended parameters."""
        if new_price is None and new_size is None:
            return OrderResult(
                success=False,
                error_message="amend_order requires at least new_price or new_size",
            )
        request = OrderRequest(
            instrument_id=instrument_id,
            side=Side.BUY,
            size=new_size or 0.0,
            price=new_price or 0.0,
        )
        return await self.place_order(request)

    # ==================== ACCOUNT (delegate) ====================

    async def get_balance(self) -> AccountBalance:
        return await self._client.get_balance()

    async def get_positions(self) -> list[ExchangePosition]:
        return await self._client.get_positions()
