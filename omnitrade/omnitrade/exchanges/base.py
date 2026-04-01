"""
Exchange client abstract base class.

Every platform (Polymarket, Kalshi, Hyperliquid) implements this interface.
Bots only depend on ExchangeClient - they never import platform-specific code.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, Optional

from ..core.enums import ExchangeId, OrderStatus, Side
from ..core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ..core.config import ExchangeConfig

logger = logging.getLogger(__name__)


class MarketDataUpdate:
    """A real-time market data update from a WebSocket stream.

    Wraps an OrderbookSnapshot with metadata about how it was delivered.
    """

    __slots__ = ("snapshot", "source")

    def __init__(self, snapshot: OrderbookSnapshot, source: str = "ws"):
        self.snapshot = snapshot
        self.source = source  # "ws" or "rest"

    def __repr__(self) -> str:
        return (
            f"MarketDataUpdate(instrument={self.snapshot.instrument_id}, "
            f"mid={self.snapshot.midpoint}, source={self.source})"
        )


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
        self._poll_tasks: dict[str, asyncio.Task] = {}  # REST fallback tasks

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
        await self.unsubscribe_all()
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

    # ==================== STREAMING ====================

    @property
    def supports_streaming(self) -> bool:
        """Whether this client supports WebSocket streaming.

        Override to return True in subclasses that implement subscribe/unsubscribe.
        """
        return False

    async def subscribe_orderbook(
        self,
        instrument_id: str,
        callback: Callable[[MarketDataUpdate], None],
    ) -> None:
        """Subscribe to real-time orderbook updates for an instrument.

        Args:
            instrument_id: Instrument to subscribe to.
            callback: Called with each MarketDataUpdate. Must be non-blocking.

        Default implementation starts a REST polling loop as a fallback.
        Subclasses with WebSocket support should override this.
        """
        logger.info(
            "No WebSocket support for %s — falling back to REST polling for %s",
            self.exchange_id.value, instrument_id,
        )
        self._start_rest_poll(instrument_id, callback)

    async def unsubscribe_orderbook(self, instrument_id: str) -> None:
        """Stop receiving updates for an instrument.

        Default implementation cancels the REST polling task.
        """
        task = self._poll_tasks.pop(instrument_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.debug("Unsubscribed from %s", instrument_id)

    async def unsubscribe_all(self) -> None:
        """Stop all active subscriptions."""
        instrument_ids = list(self._poll_tasks.keys())
        for iid in instrument_ids:
            await self.unsubscribe_orderbook(iid)

    @property
    def active_subscriptions(self) -> set[str]:
        """Return the set of instrument IDs with active subscriptions."""
        return {
            iid for iid, task in self._poll_tasks.items()
            if not task.done()
        }

    # -- REST polling fallback internals --

    def _start_rest_poll(
        self,
        instrument_id: str,
        callback: Callable[[MarketDataUpdate], None],
        interval: float = 5.0,
    ) -> None:
        """Launch a background task that polls get_orderbook and invokes callback."""
        if instrument_id in self._poll_tasks and not self._poll_tasks[instrument_id].done():
            logger.debug("Already polling %s, skipping duplicate", instrument_id)
            return

        async def _poll() -> None:
            while True:
                try:
                    snapshot = await self.get_orderbook(instrument_id, depth=10)
                    callback(MarketDataUpdate(snapshot, source="rest"))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("REST poll error for %s", instrument_id)
                await asyncio.sleep(interval)

        self._poll_tasks[instrument_id] = asyncio.create_task(_poll())

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

    # ==================== STREAMING (delegate) ====================

    @property
    def supports_streaming(self) -> bool:
        return self._client.supports_streaming

    async def subscribe_orderbook(
        self,
        instrument_id: str,
        callback: Callable[[MarketDataUpdate], None],
    ) -> None:
        await self._client.subscribe_orderbook(instrument_id, callback)

    async def unsubscribe_orderbook(self, instrument_id: str) -> None:
        await self._client.unsubscribe_orderbook(instrument_id)

    async def unsubscribe_all(self) -> None:
        await self._client.unsubscribe_all()

    @property
    def active_subscriptions(self) -> set[str]:
        return self._client.active_subscriptions

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

    # ==================== ACCOUNT (delegate) ====================

    async def get_balance(self) -> AccountBalance:
        return await self._client.get_balance()

    async def get_positions(self) -> list[ExchangePosition]:
        return await self._client.get_positions()
