"""
Execution engines for order placement.

Executors handle the mechanics of turning a sized signal into a filled order.
They work with any ExchangeClient through the unified interface.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import Side, SignalDirection, OrderStatus, OrderType
from ..core.models import Signal, OrderRequest, OrderResult, OrderbookSnapshot
from ..exchanges.base import ExchangeClient

logger = logging.getLogger(__name__)


class Executor(ABC):
    """Abstract base for execution strategies."""

    @abstractmethod
    async def execute(
        self,
        client: ExchangeClient,
        instrument_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> OrderResult:
        """Execute a trade. Returns OrderResult."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @staticmethod
    def direction_to_side(direction: SignalDirection) -> Side:
        """Map signal direction to order side."""
        if direction == SignalDirection.LONG:
            return Side.BUY
        elif direction == SignalDirection.SHORT:
            return Side.SELL
        raise ValueError(f"Cannot convert {direction} to Side")


class DryRunExecutor(Executor):
    """
    Paper trading executor.

    Simulates fills at current market price without placing real orders.
    Used for paper testing environment.
    """

    def __init__(self, slippage_pct: float = 0.001):
        self.slippage_pct = slippage_pct
        self._order_counter = 0

    @property
    def name(self) -> str:
        return "dry_run"

    async def execute(
        self,
        client: ExchangeClient,
        instrument_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> OrderResult:
        # Get current market price for realistic simulation
        if orderbook is None:
            orderbook = await client.get_orderbook(instrument_id, depth=1)

        if side == Side.BUY:
            exec_price = orderbook.best_ask or price
            exec_price *= (1 + self.slippage_pct)  # Simulate slippage
        else:
            exec_price = orderbook.best_bid or price
            exec_price *= (1 - self.slippage_pct)

        if exec_price <= 0:
            return OrderResult(success=False, error_message="No price available")

        shares = size_usd / exec_price
        self._order_counter += 1
        order_id = f"DRY-{self._order_counter:06d}"

        logger.info(
            f"[DRY RUN] {side.value} {shares:.4f} @ ${exec_price:.4f} "
            f"(${size_usd:.2f}) instrument={instrument_id}"
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_size=shares,
            filled_price=exec_price,
            requested_size=shares,
            requested_price=price,
        )


class AggressiveExecutor(Executor):
    """
    Aggressive execution - takes best available price.

    Includes safety checks:
    - Spread check: reject if bid-ask too wide
    - Slippage check: reject if execution price drifts from target
    """

    def __init__(self, max_slippage: float = 0.02, max_spread: float = 0.03):
        self.max_slippage = max_slippage
        self.max_spread = max_spread

    @property
    def name(self) -> str:
        return "aggressive"

    async def execute(
        self,
        client: ExchangeClient,
        instrument_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> OrderResult:
        if orderbook is None:
            orderbook = await client.get_orderbook(instrument_id, depth=5)

        # Spread check (only for buys - sells should always be allowed to exit)
        if side == Side.BUY and orderbook.spread is not None:
            if orderbook.spread > self.max_spread:
                return OrderResult(
                    success=False,
                    error_message=f"Spread too wide: {orderbook.spread:.1%} > {self.max_spread:.1%}",
                    is_rejection=True,
                )

        # Determine execution price
        if side == Side.BUY:
            exec_price = orderbook.best_ask
        else:
            exec_price = orderbook.best_bid

        if exec_price is None or exec_price <= 0:
            return OrderResult(success=False, error_message=f"No {'ask' if side == Side.BUY else 'bid'} available")

        # Slippage check
        if price > 0:
            slippage = abs(exec_price - price) / price
            if slippage > self.max_slippage:
                return OrderResult(
                    success=False,
                    requested_price=price,
                    error_message=f"Slippage {slippage:.1%} > {self.max_slippage:.1%}",
                    is_rejection=True,
                )

        shares = size_usd / exec_price
        if shares <= 0:
            return OrderResult(success=False, error_message="Invalid share calculation")

        # Place via exchange client
        request = OrderRequest(
            instrument_id=instrument_id,
            side=side,
            size=shares,
            price=exec_price,
            order_type=OrderType.GTC,
        )

        result = await client.place_order(request)
        return result


class LimitExecutor(Executor):
    """Place limit orders at specified price."""

    def __init__(self, price_offset: float = 0.0):
        self.price_offset = price_offset  # Positive = more aggressive

    @property
    def name(self) -> str:
        return "limit"

    async def execute(
        self,
        client: ExchangeClient,
        instrument_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> OrderResult:
        # Adjust price
        if side == Side.BUY:
            limit_price = price + self.price_offset
        else:
            limit_price = price - self.price_offset

        if limit_price <= 0:
            return OrderResult(success=False, error_message="Invalid limit price")

        shares = size_usd / limit_price
        request = OrderRequest(
            instrument_id=instrument_id,
            side=side,
            size=shares,
            price=limit_price,
            order_type=OrderType.LIMIT,
        )

        return await client.place_order(request)
