"""
Polymarket exchange client.

Wraps py-clob-client (synchronous) with asyncio.to_thread for async support.
Maps all responses through PolymarketAdapter to unified models.
"""

import asyncio
import logging
from typing import Optional

import aiohttp

from ...core.enums import ExchangeId, Side, OrderType, OrderStatus
from ...core.config import ExchangeConfig
from ...core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ...core.errors import ExchangeError, InstrumentNotFoundError
from ...utils.rate_limiter import RateLimiter
from ..base import ExchangeClient
from ..registry import register_exchange
from .auth import PolymarketAuth
from .adapter import PolymarketAdapter

logger = logging.getLogger(__name__)


@register_exchange(ExchangeId.POLYMARKET)
class PolymarketClient(ExchangeClient):
    """
    Polymarket exchange client.

    Uses py-clob-client for CLOB operations and aiohttp for Gamma API.
    All sync SDK calls are wrapped with asyncio.to_thread.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"
    DATA_API = "https://data-api.polymarket.com"

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self._auth = PolymarketAuth(config)
        self._limiter = RateLimiter(
            max_requests=config.rate_limit_per_window,
            window_seconds=config.rate_limit_window_seconds,
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._paper_balance = 10_000.0

    @property
    def exchange_id(self) -> ExchangeId:
        return ExchangeId.POLYMARKET

    async def connect(self) -> None:
        await self._auth.authenticate()
        self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("Polymarket client connected")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    @property
    def _clob(self):
        """Get the underlying ClobClient."""
        return self._auth.client

    async def _gamma_get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """GET request to Gamma API."""
        await self._limiter.wait_and_acquire()
        if self._session is None:
            raise ExchangeError("polymarket", "Not connected")
        url = f"{self.GAMMA_API}{endpoint}"
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        params = {"active": "true"} if active_only else {}
        if "limit" not in filters:
            params["limit"] = "100"
        params.update({k: str(v) for k, v in filters.items()})

        data = await self._gamma_get("/markets", params)

        instruments = []
        markets = data if isinstance(data, list) else [data]
        for market in markets:
            instruments.extend(PolymarketAdapter.market_to_instruments(market))
        return instruments

    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        """Get instrument by token_id. Searches via Gamma API."""
        try:
            data = await self._gamma_get("/markets", {"token_id": instrument_id})
            markets = data if isinstance(data, list) else [data]
            for market in markets:
                for inst in PolymarketAdapter.market_to_instruments(market):
                    if inst.instrument_id == instrument_id:
                        return inst
        except Exception as e:
            logger.warning(f"Failed to get instrument {instrument_id}: {e}")
        return None

    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        await self._limiter.wait_and_acquire()
        book = await asyncio.to_thread(self._clob.get_order_book, instrument_id)
        return PolymarketAdapter.orderbook_to_snapshot(instrument_id, book)

    async def get_midpoint(self, instrument_id: str) -> Optional[float]:
        await self._limiter.wait_and_acquire()
        try:
            mid = await asyncio.to_thread(self._clob.get_midpoint, instrument_id)
            return float(mid) if mid else None
        except Exception:
            book = await self.get_orderbook(instrument_id, depth=1)
            return book.midpoint

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Place an order and update tracked balance on fill."""
        await self._limiter.wait_and_acquire()

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            clob_side = BUY if request.side == Side.BUY else SELL

            order_args = OrderArgs(
                price=request.price,
                size=request.size,
                side=clob_side,
                token_id=request.instrument_id,
            )

            signed = await asyncio.to_thread(self._clob.create_order, order_args)
            response = await asyncio.to_thread(self._clob.post_order, signed, ClobOrderType.GTC)

            result = PolymarketAdapter.order_response_to_result(
                response, request.size, request.price
            )
            self._apply_balance_delta(result, request.side)
            return result
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e),
                requested_size=request.size,
                requested_price=request.price,
            )

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        await self._limiter.wait_and_acquire()
        try:
            await asyncio.to_thread(self._clob.cancel, order_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        await self._limiter.wait_and_acquire()
        try:
            result = await asyncio.to_thread(self._clob.cancel_all)
            if isinstance(result, dict):
                return result.get("canceled", 0)
            return 0
        except Exception as e:
            logger.warning(f"Failed to cancel all orders: {e}")
            return 0

    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        await self._limiter.wait_and_acquire()
        try:
            orders = await asyncio.to_thread(self._clob.get_orders)
            result = []
            for o in (orders if isinstance(orders, list) else []):
                oid = o.get("id", "")
                asset = o.get("asset", "")
                if instrument_id and asset != instrument_id:
                    continue
                result.append(OpenOrder(
                    order_id=oid,
                    instrument_id=asset,
                    side=Side.BUY if o.get("side", "").upper() == "BUY" else Side.SELL,
                    size=float(o.get("original_size", 0)),
                    filled_size=float(o.get("size_matched", 0)),
                    price=float(o.get("price", 0)),
                    order_type=OrderType.GTC,
                    status=OrderStatus.OPEN,
                ))
            return result
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")
            return []

    def _apply_balance_delta(self, result: OrderResult, side: Side) -> None:
        """Update tracked balance based on a filled order result.

        For binary outcome markets, cost = size * price (USDC per share).
        Buys deduct from balance, sells add to balance.

        Args:
            result: The order execution result.
            side: The order side (BUY or SELL).
        """
        if not result.success or result.filled_size <= 0:
            return

        cost = result.filled_size * result.filled_price
        if result.filled_price <= 0:
            return

        if side == Side.BUY:
            self._paper_balance -= cost
        else:
            self._paper_balance += cost

        # The adapter does not populate fees on OrderResult, so estimate
        # using Polymarket's ~2% taker fee when no fee data is present.
        fees = result.fees if result.fees > 0 else cost * 0.02
        self._paper_balance -= fees

        if self._paper_balance < 0:
            logger.warning(
                "Paper balance went negative: %.4f (side=%s cost=%.4f fees=%.4f)",
                self._paper_balance, side.value, cost, fees,
            )

        logger.debug(
            "Balance updated: side=%s cost=%.4f fees=%.4f new_balance=%.4f",
            side.value, cost, fees, self._paper_balance,
        )

    async def get_balance(self) -> AccountBalance:
        """Get account balance.

        In live mode (authenticated with API credentials), attempts to fetch
        the on-chain USDC allowance via the CLOB client. Falls back to the
        locally tracked balance if the API call fails.

        The tracked balance starts at the configured initial amount and is
        updated on every successful order fill.
        """
        if self._auth.is_authenticated():
            try:
                allowance = await asyncio.to_thread(
                    self._clob.get_bal_allowance
                )
                if isinstance(allowance, dict):
                    bal = float(allowance.get("balance", 0)) / 1e6  # USDC has 6 decimals
                    return AccountBalance(
                        exchange=ExchangeId.POLYMARKET,
                        total_equity=bal,
                        available_balance=bal,
                        currency="USDC",
                    )
            except Exception as e:
                logger.debug("Could not fetch on-chain balance, using tracked: %s", e)

        return AccountBalance(
            exchange=ExchangeId.POLYMARKET,
            total_equity=self._paper_balance,
            available_balance=self._paper_balance,
            currency="USDC",
        )

    async def get_positions(self) -> list[ExchangePosition]:
        await self._limiter.wait_and_acquire()
        try:
            # Positions live on the Data API, not Gamma
            if self._session is None:
                raise ExchangeError("polymarket", "Not connected")
            url = f"{self.DATA_API}/positions"
            async with self._session.get(url, params={
                "user": self._config.proxy_address
            }) as resp:
                resp.raise_for_status()
                data = await resp.json()
            positions = data if isinstance(data, list) else []
            return [PolymarketAdapter.position_to_exchange_position(p) for p in positions]
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return []
