"""
WebSocket client for CLOB orderbook streaming.

Connects to the Polymarket CLOB WebSocket and receives real-time
orderbook updates for subscribed tokens.

Features:
- Exponential backoff reconnection (1s-60s)
- Stale connection detection (configurable timeout)
- Callbacks for disconnect/reconnect events (for gap tracking)
- Subscription management for dynamic token lists
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Callable, List, Optional, Set, Dict, Any, Awaitable

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None
    ConnectionClosed = Exception

from ..core.models import OrderbookSnapshot

logger = logging.getLogger(__name__)

# CLOB WebSocket URL for orderbook data
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Default timeouts
DEFAULT_STALE_TIMEOUT_SECONDS = 60
DEFAULT_OPEN_TIMEOUT_SECONDS = 30
DEFAULT_PING_INTERVAL_SECONDS = 20
DEFAULT_PING_TIMEOUT_SECONDS = 10


class OrderbookWebSocketClient:
    """
    WebSocket client for CLOB orderbook streaming.

    Connects to wss://ws-subscriptions-clob.polymarket.com
    Subscribes to orderbook updates for specified tokens.

    Features:
    - Exponential backoff reconnection (1s initial, 60s max)
    - Stale connection detection with configurable timeout
    - Callbacks for connection state changes
    - Dynamic subscription management
    """

    def __init__(
        self,
        on_snapshot: Callable[[OrderbookSnapshot, str], Awaitable[None]],
        on_disconnect: Optional[Callable[[str, datetime], Awaitable[None]]] = None,
        on_reconnect: Optional[Callable[[datetime], Awaitable[None]]] = None,
        tokens: Optional[List[str]] = None,
        stale_timeout_seconds: int = DEFAULT_STALE_TIMEOUT_SECONDS,
        open_timeout_seconds: int = DEFAULT_OPEN_TIMEOUT_SECONDS,
        ping_interval_seconds: int = DEFAULT_PING_INTERVAL_SECONDS,
        ping_timeout_seconds: int = DEFAULT_PING_TIMEOUT_SECONDS,
    ):
        """
        Initialize the WebSocket client.

        Args:
            on_snapshot: Async callback for orderbook snapshots.
                        Receives (OrderbookSnapshot, token_id).
            on_disconnect: Optional async callback when connection drops.
                          Receives (reason, timestamp).
            on_reconnect: Optional async callback when connection resumes.
                         Receives (timestamp).
            tokens: Initial list of token IDs to subscribe to.
            stale_timeout_seconds: Seconds without data before forcing reconnect.
            open_timeout_seconds: Timeout for establishing connection.
            ping_interval_seconds: WebSocket ping interval.
            ping_timeout_seconds: WebSocket ping timeout.
        """
        if websockets is None:
            raise ImportError("websockets package required: pip install websockets")

        self._on_snapshot = on_snapshot
        self._on_disconnect = on_disconnect
        self._on_reconnect = on_reconnect

        self._tokens: Set[str] = set(tokens or [])
        self._pending_subscribes: Set[str] = set()
        self._pending_unsubscribes: Set[str] = set()

        self._stale_timeout = stale_timeout_seconds
        self._open_timeout = open_timeout_seconds
        self._ping_interval = ping_interval_seconds
        self._ping_timeout = ping_timeout_seconds

        # Connection state
        self._running = False
        self._connected = False
        self._ws: Optional[Any] = None
        self._last_message_time: Optional[datetime] = None

        # Reconnection settings
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60

        # Stats
        self._total_messages = 0
        self._total_reconnects = 0
        self._connection_start: Optional[datetime] = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self._connected and self._ws is not None

    @property
    def subscribed_tokens(self) -> Set[str]:
        """Get set of currently subscribed tokens."""
        return self._tokens.copy()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        uptime = None
        if self._connection_start and self._connected:
            uptime = (datetime.now(timezone.utc) - self._connection_start).total_seconds()

        return {
            'connected': self._connected,
            'subscribed_tokens': len(self._tokens),
            'total_messages': self._total_messages,
            'total_reconnects': self._total_reconnects,
            'uptime_seconds': uptime,
            'last_message': self._last_message_time.isoformat() if self._last_message_time else None,
        }

    def subscribe_token(self, token_id: str) -> None:
        """
        Add a token to the subscription list.

        If already connected, will send subscribe message.
        If not connected, will subscribe on next connection.

        Args:
            token_id: Token ID to subscribe to.
        """
        if token_id in self._tokens:
            return

        self._tokens.add(token_id)

        if self._connected and self._ws:
            self._pending_subscribes.add(token_id)

    def subscribe_tokens(self, token_ids: List[str]) -> None:
        """
        Add multiple tokens to the subscription list.

        Args:
            token_ids: List of token IDs to subscribe to.
        """
        for token_id in token_ids:
            self.subscribe_token(token_id)

    def unsubscribe_token(self, token_id: str) -> None:
        """
        Remove a token from the subscription list.

        Args:
            token_id: Token ID to unsubscribe from.
        """
        if token_id not in self._tokens:
            return

        self._tokens.discard(token_id)

        if self._connected and self._ws:
            self._pending_unsubscribes.add(token_id)

    def set_tokens(self, token_ids: List[str]) -> None:
        """
        Replace the entire subscription list.

        Args:
            token_ids: New list of token IDs.
        """
        new_tokens = set(token_ids)
        to_unsub = self._tokens - new_tokens
        to_sub = new_tokens - self._tokens

        for tid in to_unsub:
            self.unsubscribe_token(tid)
        for tid in to_sub:
            self.subscribe_token(tid)

    async def start(self) -> None:
        """Start the WebSocket connection loop."""
        if self._running:
            logger.warning("WebSocket client already running")
            return

        self._running = True
        logger.info(f"Starting orderbook WebSocket client with {len(self._tokens)} tokens")

        await self._connection_loop()

    async def stop(self) -> None:
        """Stop the WebSocket connection."""
        logger.info("Stopping orderbook WebSocket client...")
        self._running = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")

        self._connected = False

    async def _connection_loop(self) -> None:
        """Main connection loop with reconnection logic."""
        while self._running:
            disconnect_time: Optional[datetime] = None

            try:
                logger.info(f"Connecting to CLOB WebSocket: {CLOB_WS_URL}")

                async with websockets.connect(
                    CLOB_WS_URL,
                    open_timeout=self._open_timeout,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._connection_start = datetime.now(timezone.utc)
                    self._reconnect_delay = 1  # Reset on successful connection

                    logger.info("Connected to CLOB WebSocket")

                    # Notify reconnection if this isn't the first connection
                    if self._total_reconnects > 0 and self._on_reconnect:
                        try:
                            await self._on_reconnect(datetime.now(timezone.utc))
                        except Exception as e:
                            logger.error(f"Error in reconnect callback: {e}")

                    self._total_reconnects += 1

                    # Subscribe to all tokens
                    if self._tokens:
                        await self._send_subscribe(list(self._tokens))

                    # Message receive loop
                    await self._message_loop()

            except ConnectionClosed as e:
                disconnect_time = datetime.now(timezone.utc)
                logger.warning(f"WebSocket connection closed: {e}")
            except asyncio.TimeoutError:
                disconnect_time = datetime.now(timezone.utc)
                logger.warning("WebSocket connection timed out")
            except Exception as e:
                disconnect_time = datetime.now(timezone.utc)
                logger.error(f"WebSocket error: {e}")

            finally:
                self._connected = False
                self._ws = None

            # Notify disconnect
            if disconnect_time and self._on_disconnect and self._running:
                try:
                    await self._on_disconnect("connection_lost", disconnect_time)
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")

            # Reconnect with backoff
            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2,
                    self._max_reconnect_delay
                )

    async def _message_loop(self) -> None:
        """Process incoming messages with stale detection."""
        while self._running and self._connected:
            try:
                # Process any pending subscription changes
                await self._process_pending_subscriptions()

                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=self._stale_timeout
                )

                self._last_message_time = datetime.now(timezone.utc)
                self._total_messages += 1

                # Parse and process message
                try:
                    data = json.loads(message)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    logger.debug(f"Non-JSON message: {message[:100]}")
                except Exception as e:
                    logger.debug(f"Error processing message: {e}")

            except asyncio.TimeoutError:
                # No messages received within timeout - connection is stale
                logger.warning(
                    f"No messages received for {self._stale_timeout}s - "
                    "connection stale, reconnecting..."
                )
                if self._on_disconnect:
                    try:
                        await self._on_disconnect(
                            "stale_connection",
                            datetime.now(timezone.utc)
                        )
                    except Exception as e:
                        logger.error(f"Error in disconnect callback: {e}")
                break

            except ConnectionClosed:
                logger.warning("Connection closed during receive")
                break

    async def _process_pending_subscriptions(self) -> None:
        """Send any pending subscribe/unsubscribe messages."""
        if self._pending_subscribes:
            tokens = list(self._pending_subscribes)
            self._pending_subscribes.clear()
            await self._send_subscribe(tokens)

        if self._pending_unsubscribes:
            tokens = list(self._pending_unsubscribes)
            self._pending_unsubscribes.clear()
            await self._send_unsubscribe(tokens)

    async def _send_subscribe(self, token_ids: List[str]) -> None:
        """Send subscription message for tokens."""
        if not self._ws or not token_ids:
            return

        # CLOB WebSocket subscription format
        # Note: The exact format may need adjustment based on actual API
        for token_id in token_ids:
            try:
                msg = json.dumps({
                    "type": "subscribe",
                    "channel": "book",
                    "market": token_id,
                })
                await self._ws.send(msg)
                logger.debug(f"Subscribed to orderbook for {token_id[:16]}...")
            except Exception as e:
                logger.error(f"Error subscribing to {token_id}: {e}")

        logger.info(f"Sent subscription for {len(token_ids)} tokens")

    async def _send_unsubscribe(self, token_ids: List[str]) -> None:
        """Send unsubscription message for tokens."""
        if not self._ws or not token_ids:
            return

        for token_id in token_ids:
            try:
                msg = json.dumps({
                    "type": "unsubscribe",
                    "channel": "book",
                    "market": token_id,
                })
                await self._ws.send(msg)
                logger.debug(f"Unsubscribed from {token_id[:16]}...")
            except Exception as e:
                logger.error(f"Error unsubscribing from {token_id}: {e}")

    async def _process_message(self, data: Dict[str, Any]) -> None:
        """
        Process a WebSocket message.

        Expected message format (may vary based on actual CLOB API):
        {
            "channel": "book",
            "market": "token_id_here",
            "data": {
                "bids": [{"price": "0.55", "size": "1000"}, ...],
                "asks": [{"price": "0.56", "size": "800"}, ...],
                "timestamp": 1704067200000
            }
        }
        """
        # Handle different message types
        msg_type = data.get("type") or data.get("channel")

        if msg_type == "book" or msg_type == "orderbook":
            await self._handle_orderbook_update(data)
        elif msg_type == "error":
            logger.warning(f"WebSocket error message: {data}")
        elif msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {data.get('market', 'unknown')}")
        elif msg_type == "ping" or msg_type == "pong":
            pass  # Heartbeat
        else:
            # Log unknown message types for debugging
            if self._total_messages <= 5:
                logger.debug(f"Unknown message type: {msg_type}, data: {str(data)[:200]}")

    async def _handle_orderbook_update(self, data: Dict[str, Any]) -> None:
        """Parse orderbook update and call callback."""
        try:
            # Extract token ID
            token_id = data.get("market") or data.get("asset_id") or data.get("token_id")
            if not token_id:
                return

            # Extract orderbook data
            book_data = data.get("data", data)

            # Parse bids and asks
            raw_bids = book_data.get("bids", [])
            raw_asks = book_data.get("asks", [])

            bid_depth = []
            for b in raw_bids:
                if isinstance(b, dict):
                    price = float(b.get("price", 0))
                    size = float(b.get("size", 0))
                elif isinstance(b, (list, tuple)) and len(b) >= 2:
                    price = float(b[0])
                    size = float(b[1])
                else:
                    continue
                if price > 0 and size > 0:
                    bid_depth.append((price, size))

            ask_depth = []
            for a in raw_asks:
                if isinstance(a, dict):
                    price = float(a.get("price", 0))
                    size = float(a.get("size", 0))
                elif isinstance(a, (list, tuple)) and len(a) >= 2:
                    price = float(a[0])
                    size = float(a[1])
                else:
                    continue
                if price > 0 and size > 0:
                    ask_depth.append((price, size))

            # Sort depth levels
            bid_depth.sort(key=lambda x: x[0], reverse=True)  # Highest bid first
            ask_depth.sort(key=lambda x: x[0])  # Lowest ask first

            # Extract timestamp
            ts = book_data.get("timestamp")
            if ts:
                if isinstance(ts, (int, float)):
                    # Unix timestamp in ms or seconds
                    if ts > 1e12:
                        ts = ts / 1000  # Convert ms to seconds
                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Build snapshot
            best_bid = bid_depth[0][0] if bid_depth else None
            best_ask = ask_depth[0][0] if ask_depth else None
            bid_size = bid_depth[0][1] if bid_depth else 0.0
            ask_size = ask_depth[0][1] if ask_depth else 0.0

            snapshot = OrderbookSnapshot(
                token_id=token_id,
                timestamp=timestamp,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
            )

            # Call callback
            await self._on_snapshot(snapshot, token_id)

        except Exception as e:
            logger.debug(f"Error parsing orderbook update: {e}")
