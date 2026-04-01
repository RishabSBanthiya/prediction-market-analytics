"""
Hyperliquid exchange client.

Wraps hyperliquid-python-sdk (synchronous) with asyncio.to_thread.
Handles perpetual-specific features: leverage, funding, margin.
Supports WebSocket streaming for real-time L2 orderbook data.
"""

import asyncio
import logging
from typing import Callable, Optional

from ...core.enums import ExchangeId, Side, OrderType, OrderStatus
from ...core.config import ExchangeConfig
from ...core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ...core.errors import ExchangeError
from ...utils.rate_limiter import RateLimiter
from ..base import ExchangeClient, MarketDataUpdate
from ..registry import register_exchange
from ..auth_retry import with_auth_retry
from .auth import HyperliquidAuth
from .adapter import HyperliquidAdapter
from .websocket import HyperliquidWebSocket

logger = logging.getLogger(__name__)


@register_exchange(ExchangeId.HYPERLIQUID)
class HyperliquidClient(ExchangeClient):
    """
    Hyperliquid perpetual futures client.

    Wraps the sync SDK with asyncio.to_thread for async support.
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self._auth = HyperliquidAuth(config)
        self._limiter = RateLimiter(
            max_requests=config.rate_limit_per_window,
            window_seconds=config.rate_limit_window_seconds,
        )
        self._instruments_cache: list[Instrument] = []
        self._ws: Optional[HyperliquidWebSocket] = None

    @property
    def exchange_id(self) -> ExchangeId:
        return ExchangeId.HYPERLIQUID

    async def connect(self) -> None:
        await self._auth.authenticate()
        self._connected = True
        logger.info("Hyperliquid client connected")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._connected = False

    @with_auth_retry
    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        await self._limiter.wait_and_acquire()

        info = self._auth.info
        meta = await asyncio.to_thread(info.meta)
        asset_ctxs = await asyncio.to_thread(info.meta_and_asset_ctxs)

        # meta_and_asset_ctxs returns [meta, [ctx1, ctx2, ...]]
        ctxs = asset_ctxs[1] if isinstance(asset_ctxs, list) and len(asset_ctxs) > 1 else []
        actual_meta = asset_ctxs[0] if isinstance(asset_ctxs, list) else meta

        self._instruments_cache = HyperliquidAdapter.meta_to_instruments(actual_meta, ctxs)
        return self._instruments_cache

    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        if not self._instruments_cache:
            await self.get_instruments()
        for inst in self._instruments_cache:
            if inst.instrument_id == instrument_id:
                return inst
        return None

    @with_auth_retry
    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        await self._limiter.wait_and_acquire()
        info = self._auth.info
        l2 = await asyncio.to_thread(info.l2_snapshot, instrument_id)
        return HyperliquidAdapter.l2_to_snapshot(instrument_id, {"levels": l2})

    @with_auth_retry
    async def place_order(self, request: OrderRequest) -> OrderResult:
        await self._limiter.wait_and_acquire()

        exchange = self._auth.exchange
        is_buy = request.side == Side.BUY

        try:
            result = await asyncio.to_thread(
                exchange.order,
                request.instrument_id,
                is_buy,
                request.size,
                request.price,
                {"limit": {"tif": "Gtc"}},
            )
            return HyperliquidAdapter.order_response_to_result(result, request.size, request.price)
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e),
                requested_size=request.size,
                requested_price=request.price,
            )

    @with_auth_retry
    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        await self._limiter.wait_and_acquire()
        try:
            exchange = self._auth.exchange
            result = await asyncio.to_thread(
                exchange.cancel, instrument_id, int(order_id)
            )
            return result.get("status") == "ok"
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        # Get open orders first, then cancel each
        orders = await self.get_open_orders(instrument_id)
        cancelled = 0
        for order in orders:
            if await self.cancel_order(order.order_id, order.instrument_id):
                cancelled += 1
        return cancelled

    @with_auth_retry
    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        await self._limiter.wait_and_acquire()
        try:
            info = self._auth.info
            address = self._auth.address
            orders = await asyncio.to_thread(info.open_orders, address)

            result = []
            for o in orders:
                coin = o.get("coin", "")
                if instrument_id and coin != instrument_id:
                    continue
                result.append(OpenOrder(
                    order_id=str(o.get("oid", "")),
                    instrument_id=coin,
                    side=Side.BUY if o.get("side", "").upper() == "B" else Side.SELL,
                    size=float(o.get("sz", 0)),
                    filled_size=0.0,
                    price=float(o.get("limitPx", 0)),
                    order_type=OrderType.LIMIT,
                    status=OrderStatus.OPEN,
                ))
            return result
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")
            return []

    @with_auth_retry
    async def get_balance(self) -> AccountBalance:
        await self._limiter.wait_and_acquire()
        try:
            info = self._auth.info
            state = await asyncio.to_thread(info.user_state, self._auth.address)
            return HyperliquidAdapter.user_state_to_balance(state)
        except Exception as e:
            logger.warning(f"Failed to get balance: {e}")
            return AccountBalance(exchange=ExchangeId.HYPERLIQUID)

    @with_auth_retry
    async def get_positions(self) -> list[ExchangePosition]:
        await self._limiter.wait_and_acquire()
        try:
            info = self._auth.info
            state = await asyncio.to_thread(info.user_state, self._auth.address)
            return HyperliquidAdapter.user_state_to_positions(state)
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return []

    # ==================== STREAMING ====================

    @property
    def supports_streaming(self) -> bool:
        return True

    async def _ensure_ws(self) -> HyperliquidWebSocket:
        """Lazily create and connect the WebSocket."""
        if self._ws is None:
            testnet = "testnet" in self._config.api_base.lower() if self._config.api_base else False
            self._ws = HyperliquidWebSocket(testnet=testnet)
            await self._ws.connect()
        return self._ws

    async def subscribe_orderbook(
        self,
        instrument_id: str,
        callback: Callable[[MarketDataUpdate], None],
    ) -> None:
        """Subscribe to real-time L2 orderbook updates via WebSocket."""
        ws = await self._ensure_ws()
        await ws.subscribe(instrument_id, callback)

    async def unsubscribe_orderbook(self, instrument_id: str) -> None:
        """Unsubscribe from L2 orderbook updates."""
        if self._ws is not None:
            await self._ws.unsubscribe(instrument_id)

    async def unsubscribe_all(self) -> None:
        """Close the WebSocket and all subscriptions."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    @property
    def active_subscriptions(self) -> set[str]:
        if self._ws is not None:
            return self._ws.subscribed_instruments
        return set()

    @with_auth_retry
    async def set_leverage(self, instrument_id: str, leverage: int, is_cross: bool = True) -> bool:
        """Set leverage for an instrument. Hyperliquid-specific."""
        await self._limiter.wait_and_acquire()
        try:
            exchange = self._auth.exchange
            result = await asyncio.to_thread(
                exchange.update_leverage, leverage, instrument_id, is_cross
            )
            return result.get("status") == "ok"
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")
            return False
