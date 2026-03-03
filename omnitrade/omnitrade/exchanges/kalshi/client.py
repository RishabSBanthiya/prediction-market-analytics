"""
Kalshi exchange client.

Native async via aiohttp. Prices converted between cents (Kalshi) and 0-1 (omnitrade).
"""

import logging
from typing import Optional

import aiohttp

from ...core.enums import ExchangeId, Side, OrderStatus, OrderType
from ...core.config import ExchangeConfig
from ...core.models import (
    Instrument, OrderbookSnapshot, OrderRequest, OrderResult,
    OpenOrder, AccountBalance, ExchangePosition,
)
from ...core.errors import ExchangeError
from ...utils.rate_limiter import RateLimiter
from ..base import ExchangeClient
from ..registry import register_exchange
from .auth import KalshiAuth
from .adapter import KalshiAdapter, normalized_to_cents, cents_to_normalized

logger = logging.getLogger(__name__)


@register_exchange(ExchangeId.KALSHI)
class KalshiClient(ExchangeClient):
    """
    Kalshi exchange client.

    Native async via aiohttp. Handles cents <-> normalized price conversion.
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__(config)
        self._auth = KalshiAuth(config)
        self._limiter = RateLimiter(
            max_requests=config.rate_limit_per_window,
            window_seconds=config.rate_limit_window_seconds,
        )
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def exchange_id(self) -> ExchangeId:
        return ExchangeId.KALSHI

    async def connect(self) -> None:
        await self._auth.authenticate()
        self._session = aiohttp.ClientSession()
        self._connected = True
        logger.info("Kalshi client connected")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def _request(self, method: str, path: str, json: Optional[dict] = None) -> dict:
        """Make authenticated request to Kalshi API."""
        await self._limiter.wait_and_acquire()
        if self._session is None:
            raise ExchangeError("kalshi", "Not connected")

        url = f"{self._config.api_base}{path}"
        # Signature must cover the full URL path (e.g. /trade-api/v2/portfolio/balance)
        from urllib.parse import urlparse
        full_path = urlparse(url).path
        headers = self._auth.sign_request(method, full_path)
        headers["Content-Type"] = "application/json"

        async with self._session.request(method, url, headers=headers, json=json) as resp:
            if resp.status == 429:
                from ...core.errors import RateLimitError
                retry = resp.headers.get("Retry-After")
                raise RateLimitError("kalshi", float(retry) if retry else None)
            resp.raise_for_status()
            return await resp.json()

    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        params = "?status=open" if active_only else ""
        limit = filters.get("limit", 100)
        params += f"&limit={limit}"

        data = await self._request("GET", f"/markets{params}")
        instruments = []
        for market in data.get("markets", []):
            instruments.extend(KalshiAdapter.event_to_instruments(market))
        return instruments

    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        # instrument_id is like "TICKER-YES"
        ticker = instrument_id.rsplit("-", 1)[0] if "-" in instrument_id else instrument_id
        try:
            data = await self._request("GET", f"/markets/{ticker}")
            market = data.get("market", data)
            for inst in KalshiAdapter.event_to_instruments(market):
                if inst.instrument_id == instrument_id:
                    return inst
        except Exception as e:
            logger.warning(f"Failed to get instrument {instrument_id}: {e}")
        return None

    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        ticker = instrument_id.rsplit("-", 1)[0] if "-" in instrument_id else instrument_id
        data = await self._request("GET", f"/markets/{ticker}/orderbook?depth={depth}")
        return KalshiAdapter.orderbook_to_snapshot(instrument_id, data.get("orderbook", data))

    async def place_order(self, request: OrderRequest) -> OrderResult:
        ticker = request.instrument_id.rsplit("-", 1)[0]
        side_suffix = request.instrument_id.rsplit("-", 1)[-1] if "-" in request.instrument_id else "YES"

        # Convert normalized price to cents
        price_cents = normalized_to_cents(request.price)

        order_data = {
            "ticker": ticker,
            "action": "buy" if request.side == Side.BUY else "sell",
            "side": side_suffix.lower(),  # "yes" or "no"
            "type": "limit",
            "count": max(1, int(request.size)),
            "yes_price": price_cents if side_suffix == "YES" else None,
            "no_price": price_cents if side_suffix == "NO" else None,
        }
        # Remove None values
        order_data = {k: v for k, v in order_data.items() if v is not None}

        try:
            data = await self._request("POST", "/portfolio/orders", json=order_data)
            return KalshiAdapter.order_response_to_result(data, request.size, request.price)
        except Exception as e:
            return OrderResult(
                success=False,
                error_message=str(e),
                requested_size=request.size,
                requested_price=request.price,
            )

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        try:
            await self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        try:
            data: dict = {}
            if instrument_id:
                ticker = instrument_id.rsplit("-", 1)[0]
                data = {"ticker": ticker}
            result = await self._request("DELETE", "/portfolio/orders", json=data)
            return result.get("reduced_count", 0)
        except Exception as e:
            logger.warning(f"Failed to cancel all: {e}")
            return 0

    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        try:
            data = await self._request("GET", "/portfolio/orders?status=resting")
            orders = []
            for o in data.get("orders", []):
                oid = o.get("order_id", "")
                ticker = o.get("ticker", "")
                kalshi_side = o.get("side", "yes")
                inst_id = f"{ticker}-{kalshi_side.upper()}"
                if instrument_id and inst_id != instrument_id:
                    continue
                total = int(o.get("count", 0))
                remaining = int(o.get("remaining_count", 0))
                price_cents = float(o.get("yes_price", 0) or o.get("no_price", 0))
                orders.append(OpenOrder(
                    order_id=oid,
                    instrument_id=inst_id,
                    side=Side.BUY if o.get("action") == "buy" else Side.SELL,
                    size=float(total),
                    filled_size=float(total - remaining),
                    price=cents_to_normalized(price_cents),
                    order_type=OrderType.LIMIT,
                    status=OrderStatus.OPEN,
                ))
            return orders
        except Exception as e:
            logger.warning(f"Failed to get open orders: {e}")
            return []

    async def get_balance(self) -> AccountBalance:
        try:
            data = await self._request("GET", "/portfolio/balance")
            balance_cents = float(data.get("balance", 0))
            return AccountBalance(
                exchange=ExchangeId.KALSHI,
                total_equity=balance_cents / 100,
                available_balance=balance_cents / 100,
                currency="USD",
            )
        except Exception as e:
            logger.warning(f"Failed to get balance: {e}")
            return AccountBalance(exchange=ExchangeId.KALSHI)

    async def get_positions(self) -> list[ExchangePosition]:
        try:
            data = await self._request("GET", "/portfolio/positions")
            positions = []
            for p in data.get("market_positions", []):
                ticker = p.get("ticker", "")
                yes_count = int(p.get("position", 0))
                if yes_count > 0:
                    positions.append(ExchangePosition(
                        instrument_id=f"{ticker}-YES",
                        exchange=ExchangeId.KALSHI,
                        side=Side.BUY,
                        size=float(yes_count),
                        entry_price=cents_to_normalized(float(p.get("avg_cost", 50))),
                    ))
                elif yes_count < 0:
                    positions.append(ExchangePosition(
                        instrument_id=f"{ticker}-NO",
                        exchange=ExchangeId.KALSHI,
                        side=Side.BUY,
                        size=float(abs(yes_count)),
                        entry_price=cents_to_normalized(float(p.get("avg_cost", 50))),
                    ))
            return positions
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            return []
