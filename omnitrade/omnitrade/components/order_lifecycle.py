"""
Order lifecycle tracking for partial fills and open order management.

Provides ``OrderTracker`` which monitors working orders, detects partial
fills, and allows bots to react to fill progress without polling the
exchange on every iteration.

Usage::

    tracker = OrderTracker(client)
    tracker.track(order_result)          # start tracking after place_order
    updates = await tracker.poll_all()   # check all tracked orders
    for update in updates:
        if update.is_filled:
            ...
        elif update.is_partial:
            ...
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import OrderStatus, Side
from ..core.models import OrderRequest, OrderResult
from ..exchanges.base import ExchangeClient

logger = logging.getLogger(__name__)


@dataclass
class TrackedOrder:
    """Internal state for a tracked working order.

    Attributes:
        order_id: Exchange-assigned order ID.
        instrument_id: The instrument this order is for.
        side: Buy or sell.
        requested_size: Original requested quantity.
        requested_price: Original requested price.
        filled_size: Cumulative quantity filled so far.
        filled_price: Volume-weighted average fill price.
        status: Current order status.
        created_at: When the order was first placed.
        last_polled_at: When we last checked the exchange for updates.
    """
    order_id: str
    instrument_id: str
    side: Side
    requested_size: float
    requested_price: float
    filled_size: float = 0.0
    filled_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_polled_at: Optional[datetime] = None

    @property
    def remaining_size(self) -> float:
        """Quantity still unfilled."""
        return max(0.0, self.requested_size - self.filled_size)

    @property
    def fill_pct(self) -> float:
        """Fill ratio as a fraction in [0, 1]."""
        if self.requested_size <= 0:
            return 1.0 if self.filled_size > 0 else 0.0
        return min(1.0, self.filled_size / self.requested_size)

    @property
    def is_terminal(self) -> bool:
        """True when the order has reached a final state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    def to_order_result(self) -> OrderResult:
        """Convert the tracked state to an OrderResult snapshot."""
        return OrderResult(
            success=self.status not in (OrderStatus.REJECTED, OrderStatus.EXPIRED),
            order_id=self.order_id,
            status=self.status,
            filled_size=self.filled_size,
            filled_price=self.filled_price,
            requested_size=self.requested_size,
            requested_price=self.requested_price,
        )


@dataclass
class OrderUpdate:
    """Describes what changed when an order was polled.

    Attributes:
        tracked: The updated tracked order state.
        prev_filled_size: Filled size before this poll (for detecting new fills).
        prev_status: Status before this poll.
        new_fill_qty: Incremental fill quantity since last poll.
    """
    tracked: TrackedOrder
    prev_filled_size: float
    prev_status: OrderStatus
    new_fill_qty: float = 0.0

    @property
    def had_new_fill(self) -> bool:
        """True if additional quantity was filled since last poll."""
        return self.new_fill_qty > 0

    @property
    def status_changed(self) -> bool:
        """True if the order status changed since last poll."""
        return self.tracked.status != self.prev_status

    @property
    def is_filled(self) -> bool:
        """True if the order just became fully filled."""
        return self.tracked.status == OrderStatus.FILLED

    @property
    def is_partial(self) -> bool:
        """True if the order is partially filled (not yet complete)."""
        return self.tracked.status == OrderStatus.PARTIALLY_FILLED


class OrderTracker:
    """Tracks working orders and polls for fill updates.

    Designed to be used by bots that need to handle partial fills
    and open order management. Thread-safe for single-asyncio-task use.

    Args:
        client: The exchange client to poll order status from.
    """

    def __init__(self, client: ExchangeClient):
        self._client = client
        self._orders: dict[str, TrackedOrder] = {}

    @property
    def tracked_count(self) -> int:
        """Number of orders currently being tracked."""
        return len(self._orders)

    @property
    def open_orders(self) -> list[TrackedOrder]:
        """All tracked orders that are still working."""
        return [o for o in self._orders.values() if not o.is_terminal]

    @property
    def all_orders(self) -> list[TrackedOrder]:
        """All tracked orders including terminal ones."""
        return list(self._orders.values())

    def track(self, result: OrderResult, side: Side = Side.BUY, instrument_id: str = "") -> Optional[TrackedOrder]:
        """Start tracking an order from its initial OrderResult.

        Only tracks orders that were successfully submitted (success=True) and
        have an order_id. Returns None if the order cannot be tracked (e.g.,
        rejected or missing order_id).

        Args:
            result: The OrderResult from place_order().
            side: Order side (needed because OrderResult doesn't carry it).
            instrument_id: Instrument ID (for lookup during polling).

        Returns:
            The TrackedOrder if tracking started, None otherwise.
        """
        if not result.success or not result.order_id:
            return None

        # If already fully filled, track but mark terminal immediately
        tracked = TrackedOrder(
            order_id=result.order_id,
            instrument_id=instrument_id,
            side=side,
            requested_size=result.requested_size,
            requested_price=result.requested_price,
            filled_size=result.filled_size,
            filled_price=result.filled_price,
            status=result.status,
        )
        self._orders[result.order_id] = tracked
        logger.debug(
            "Tracking order %s: %s %.4f @ %.4f (status=%s, filled=%.4f)",
            result.order_id, side.value, result.requested_size,
            result.requested_price, result.status.value, result.filled_size,
        )
        return tracked

    def untrack(self, order_id: str) -> Optional[TrackedOrder]:
        """Stop tracking an order. Returns the removed TrackedOrder or None."""
        return self._orders.pop(order_id, None)

    def get(self, order_id: str) -> Optional[TrackedOrder]:
        """Get a tracked order by ID."""
        return self._orders.get(order_id)

    def purge_terminal(self) -> int:
        """Remove all orders in terminal states. Returns count removed."""
        terminal_ids = [
            oid for oid, o in self._orders.items() if o.is_terminal
        ]
        for oid in terminal_ids:
            del self._orders[oid]
        return len(terminal_ids)

    async def poll(self, order_id: str) -> Optional[OrderUpdate]:
        """Poll a single order for updates.

        Returns an OrderUpdate if the order is tracked, None otherwise.
        If the exchange's get_order_status returns None (not supported),
        falls back to scanning open orders.
        """
        tracked = self._orders.get(order_id)
        if tracked is None:
            return None

        if tracked.is_terminal:
            return OrderUpdate(
                tracked=tracked,
                prev_filled_size=tracked.filled_size,
                prev_status=tracked.status,
            )

        prev_filled = tracked.filled_size
        prev_status = tracked.status
        now = datetime.now(timezone.utc)

        # Try direct order status query first
        result = await self._client.get_order_status(
            order_id, tracked.instrument_id
        )

        if result is not None:
            self._apply_update(tracked, result)
        else:
            # Fallback: scan open orders for a match
            await self._poll_via_open_orders(tracked)

        tracked.last_polled_at = now
        new_fill = max(0.0, tracked.filled_size - prev_filled)

        if new_fill > 0:
            logger.info(
                "Order %s fill update: +%.4f (total %.4f/%.4f, status=%s)",
                order_id, new_fill, tracked.filled_size,
                tracked.requested_size, tracked.status.value,
            )

        return OrderUpdate(
            tracked=tracked,
            prev_filled_size=prev_filled,
            prev_status=prev_status,
            new_fill_qty=new_fill,
        )

    async def poll_all(self) -> list[OrderUpdate]:
        """Poll all non-terminal tracked orders for updates.

        Returns a list of OrderUpdate for each tracked order that was polled.
        """
        updates: list[OrderUpdate] = []
        for order_id in list(self._orders.keys()):
            tracked = self._orders[order_id]
            if tracked.is_terminal:
                continue
            update = await self.poll(order_id)
            if update is not None:
                updates.append(update)
        return updates

    def _apply_update(self, tracked: TrackedOrder, result: OrderResult) -> None:
        """Apply an OrderResult to a TrackedOrder."""
        # Only move forward (never reduce filled_size)
        if result.filled_size > tracked.filled_size:
            # Update VWAP if we have new fill price data
            if result.filled_price > 0:
                tracked.filled_price = result.filled_price
            tracked.filled_size = result.filled_size

        tracked.status = result.status

        # Auto-detect partial fill status if exchange didn't set it
        if (tracked.status == OrderStatus.OPEN
                and tracked.filled_size > 0
                and tracked.filled_size < tracked.requested_size):
            tracked.status = OrderStatus.PARTIALLY_FILLED

    async def _poll_via_open_orders(self, tracked: TrackedOrder) -> None:
        """Fallback: check if order is still in the open orders list."""
        try:
            open_orders = await self._client.get_open_orders(tracked.instrument_id)
        except Exception as e:
            logger.debug("Failed to poll open orders for %s: %s", tracked.order_id, e)
            return

        found = False
        for oo in open_orders:
            if oo.order_id == tracked.order_id:
                found = True
                if oo.filled_size > tracked.filled_size:
                    tracked.filled_size = oo.filled_size
                    # Propagate fill price from the open order's limit price
                    # as a best-effort average price when the exchange doesn't
                    # provide a dedicated avg_price field.
                    if oo.price > 0:
                        tracked.filled_price = oo.price
                tracked.status = oo.status
                # Auto-detect partial fill status
                if (tracked.status == OrderStatus.OPEN
                        and tracked.filled_size > 0
                        and tracked.filled_size < tracked.requested_size):
                    tracked.status = OrderStatus.PARTIALLY_FILLED
                break

        if not found and not tracked.is_terminal:
            # Order disappeared from open orders -- likely fully filled or cancelled.
            # If we have partial fills, assume filled; otherwise assume cancelled.
            if tracked.filled_size >= tracked.requested_size:
                tracked.status = OrderStatus.FILLED
            elif tracked.filled_size > 0:
                # Disappeared with partial fill -- could be fully filled between
                # polls or cancelled with a partial remainder.
                logger.warning(
                    "Order %s disappeared from open orders with partial fill "
                    "(%.4f/%.4f). May have been fully filled between polls or "
                    "cancelled with partial remainder.",
                    tracked.order_id, tracked.filled_size, tracked.requested_size,
                )
                tracked.status = OrderStatus.CANCELLED
            else:
                # No fills at all -- cancelled or expired
                tracked.status = OrderStatus.CANCELLED
