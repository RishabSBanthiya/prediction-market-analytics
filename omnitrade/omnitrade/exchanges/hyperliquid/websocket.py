"""
Hyperliquid WebSocket client for real-time L2 orderbook streaming.

Connects to wss://api.hyperliquid.xyz/ws and subscribes to l2Book channels.
Automatically reconnects on disconnect with exponential backoff.
"""

import asyncio
import json
import logging
from typing import Callable, Optional

from ...core.models import OrderbookSnapshot
from ..base import MarketDataUpdate
from .adapter import HyperliquidAdapter

logger = logging.getLogger(__name__)

WS_URL = "wss://api.hyperliquid.xyz/ws"
# Testnet: wss://api.hyperliquid-testnet.xyz/ws

# Reconnect parameters
_INITIAL_BACKOFF_S = 1.0
_MAX_BACKOFF_S = 30.0
_BACKOFF_FACTOR = 2.0


class HyperliquidWebSocket:
    """Manages a single WebSocket connection to Hyperliquid.

    Multiplexes multiple instrument subscriptions over one connection.
    Each subscription has a callback invoked on every L2 update.
    """

    def __init__(self, testnet: bool = False) -> None:
        self._url = (
            "wss://api.hyperliquid-testnet.xyz/ws" if testnet else WS_URL
        )
        self._ws: Optional[object] = None  # websockets connection
        self._subscriptions: dict[str, Callable[[MarketDataUpdate], None]] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._running

    @property
    def subscribed_instruments(self) -> set[str]:
        return set(self._subscriptions.keys())

    async def connect(self) -> None:
        """Open the WebSocket connection and start the receive loop."""
        if self._running:
            return

        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "websockets package is required for WebSocket streaming. "
                "Install with: pip install websockets"
            )

        self._running = True
        self._recv_task = asyncio.create_task(self._connection_loop())
        logger.info("Hyperliquid WebSocket started (url=%s)", self._url)

    async def close(self) -> None:
        """Close the connection and stop the receive loop."""
        self._running = False
        self._subscriptions.clear()

        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        await self._close_ws()
        logger.info("Hyperliquid WebSocket closed")

    async def subscribe(
        self,
        instrument_id: str,
        callback: Callable[[MarketDataUpdate], None],
    ) -> None:
        """Subscribe to L2 book updates for an instrument.

        Args:
            instrument_id: Coin symbol (e.g. "BTC", "ETH").
            callback: Invoked with each MarketDataUpdate.
        """
        self._subscriptions[instrument_id] = callback
        await self._send_subscription(instrument_id, subscribe=True)
        logger.info("Subscribed to %s L2 book via WebSocket", instrument_id)

    async def unsubscribe(self, instrument_id: str) -> None:
        """Unsubscribe from L2 book updates for an instrument."""
        self._subscriptions.pop(instrument_id, None)
        await self._send_subscription(instrument_id, subscribe=False)
        logger.debug("Unsubscribed from %s L2 book", instrument_id)

    # ==================== Internal ====================

    async def _connection_loop(self) -> None:
        """Maintain the WS connection with reconnect backoff."""
        backoff = _INITIAL_BACKOFF_S

        while self._running:
            try:
                import websockets

                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    backoff = _INITIAL_BACKOFF_S
                    logger.debug("WebSocket connected to %s", self._url)

                    # Resubscribe after reconnect
                    for instrument_id in list(self._subscriptions):
                        await self._send_subscription(instrument_id, subscribe=True)

                    await self._recv_loop(ws)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "WebSocket error, reconnecting in %.1fs", backoff
                )
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * _BACKOFF_FACTOR, _MAX_BACKOFF_S)

    async def _recv_loop(self, ws: object) -> None:
        """Read messages and dispatch to callbacks."""
        async for raw_msg in ws:  # type: ignore[attr-defined]
            try:
                msg = json.loads(raw_msg)
                self._handle_message(msg)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error handling WS message")

    def _handle_message(self, msg: dict) -> None:
        """Parse a WS message and invoke the appropriate callback."""
        channel = msg.get("channel")
        data = msg.get("data")

        if channel != "l2Book" or data is None:
            return

        coin = data.get("coin", "")
        callback = self._subscriptions.get(coin)
        if callback is None:
            return

        levels = data.get("levels", [])
        snapshot = HyperliquidAdapter.l2_to_snapshot(coin, {"levels": levels})
        callback(MarketDataUpdate(snapshot, source="ws"))

    async def _send_subscription(
        self, instrument_id: str, subscribe: bool = True
    ) -> None:
        """Send a subscribe/unsubscribe message if connected."""
        if self._ws is None:
            return

        msg = {
            "method": "subscribe" if subscribe else "unsubscribe",
            "subscription": {
                "type": "l2Book",
                "coin": instrument_id,
            },
        }
        try:
            await self._ws.send(json.dumps(msg))  # type: ignore[attr-defined]
        except Exception:
            logger.warning(
                "Failed to send %s for %s",
                "subscribe" if subscribe else "unsubscribe",
                instrument_id,
            )

    async def _close_ws(self) -> None:
        """Close the underlying WS connection."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()  # type: ignore[attr-defined]
            except Exception:
                pass
