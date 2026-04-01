"""Tests for order lifecycle tracking: partial fills, order state, and OrderTracker."""

import pytest
from datetime import datetime, timezone

from omnitrade.core.enums import OrderStatus, Side, OrderType, ExchangeId
from omnitrade.core.models import OrderResult, OpenOrder
from omnitrade.components.order_lifecycle import OrderTracker, TrackedOrder, OrderUpdate


# ========================= OrderResult property tests =========================


class TestOrderResultProperties:
    """Test the new convenience properties on OrderResult."""

    def test_is_filled_when_status_filled(self):
        r = OrderResult(success=True, status=OrderStatus.FILLED, filled_size=10.0, requested_size=10.0)
        assert r.is_filled is True
        assert r.is_partial is False

    def test_is_partial_when_status_partial(self):
        r = OrderResult(
            success=True, status=OrderStatus.PARTIALLY_FILLED,
            filled_size=5.0, requested_size=10.0,
        )
        assert r.is_partial is True
        assert r.is_filled is False

    def test_is_partial_detected_from_sizes(self):
        """Partial fill detected even when status is still OPEN."""
        r = OrderResult(
            success=True, status=OrderStatus.OPEN,
            filled_size=3.0, requested_size=10.0,
        )
        assert r.is_partial is True

    def test_not_partial_when_fully_filled_sizes(self):
        """Not partial when filled_size equals requested_size."""
        r = OrderResult(
            success=True, status=OrderStatus.OPEN,
            filled_size=10.0, requested_size=10.0,
        )
        assert r.is_partial is False

    def test_not_partial_when_zero_fill(self):
        r = OrderResult(success=True, status=OrderStatus.OPEN, filled_size=0.0, requested_size=10.0)
        assert r.is_partial is False

    def test_is_open_states(self):
        for status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
            r = OrderResult(success=True, status=status)
            assert r.is_open is True, f"Expected is_open=True for {status}"

    def test_is_not_open_for_terminal(self):
        for status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            r = OrderResult(success=True, status=status)
            assert r.is_open is False, f"Expected is_open=False for {status}"

    def test_is_terminal(self):
        for status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
            r = OrderResult(success=True, status=status)
            assert r.is_terminal is True, f"Expected is_terminal=True for {status}"

    def test_not_terminal_for_working_states(self):
        for status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED):
            r = OrderResult(success=True, status=status)
            assert r.is_terminal is False, f"Expected is_terminal=False for {status}"

    def test_remaining_size(self):
        r = OrderResult(success=True, filled_size=3.0, requested_size=10.0)
        assert r.remaining_size == pytest.approx(7.0)

    def test_remaining_size_zero_when_filled(self):
        r = OrderResult(success=True, filled_size=10.0, requested_size=10.0)
        assert r.remaining_size == pytest.approx(0.0)

    def test_remaining_size_zero_when_no_request(self):
        r = OrderResult(success=True, filled_size=5.0, requested_size=0.0)
        assert r.remaining_size == pytest.approx(0.0)

    def test_fill_pct(self):
        r = OrderResult(success=True, filled_size=5.0, requested_size=10.0)
        assert r.fill_pct == pytest.approx(0.5)

    def test_fill_pct_full(self):
        r = OrderResult(success=True, filled_size=10.0, requested_size=10.0)
        assert r.fill_pct == pytest.approx(1.0)

    def test_fill_pct_zero(self):
        r = OrderResult(success=True, filled_size=0.0, requested_size=10.0)
        assert r.fill_pct == pytest.approx(0.0)

    def test_fill_pct_no_requested_size(self):
        """When requested_size is 0 but there are fills, return 1.0."""
        r = OrderResult(success=True, filled_size=5.0, requested_size=0.0)
        assert r.fill_pct == pytest.approx(1.0)

    def test_fill_pct_no_fills_no_request(self):
        r = OrderResult(success=True, filled_size=0.0, requested_size=0.0)
        assert r.fill_pct == pytest.approx(0.0)

    def test_backward_compatible_success_check(self):
        """Existing bot pattern: `if result.success and result.filled_size > 0` still works."""
        r = OrderResult(
            success=True, order_id="ORD-1", status=OrderStatus.FILLED,
            filled_size=100.0, filled_price=0.52, requested_size=100.0, requested_price=0.52,
        )
        assert r.success and r.filled_size > 0

    def test_backward_compatible_failure_check(self):
        """Existing bot pattern: `if not result.success` still works."""
        r = OrderResult(success=False, error_message="Spread too wide", is_rejection=True)
        assert not r.success
        assert r.is_rejection


# ========================= TrackedOrder tests =========================


class TestTrackedOrder:
    def test_remaining_size(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5, filled_size=3.0,
        )
        assert t.remaining_size == pytest.approx(7.0)

    def test_fill_pct(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5, filled_size=5.0,
        )
        assert t.fill_pct == pytest.approx(0.5)

    def test_is_terminal(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            status=OrderStatus.FILLED,
        )
        assert t.is_terminal is True

    def test_not_terminal(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            status=OrderStatus.OPEN,
        )
        assert t.is_terminal is False

    def test_to_order_result(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            filled_size=7.0, filled_price=0.51,
            status=OrderStatus.PARTIALLY_FILLED,
        )
        r = t.to_order_result()
        assert r.order_id == "O1"
        assert r.filled_size == pytest.approx(7.0)
        assert r.is_partial is True
        assert r.success is True


# ========================= OrderTracker tests =========================


class TestOrderTracker:
    async def test_track_successful_order(self, mock_client):
        tracker = OrderTracker(mock_client)
        result = OrderResult(
            success=True, order_id="ORD-1", status=OrderStatus.OPEN,
            requested_size=10.0, requested_price=0.5,
        )
        tracked = tracker.track(result, side=Side.BUY, instrument_id="test-token")
        assert tracked is not None
        assert tracker.tracked_count == 1
        assert tracked.order_id == "ORD-1"

    async def test_track_rejected_order_returns_none(self, mock_client):
        tracker = OrderTracker(mock_client)
        result = OrderResult(success=False, error_message="Rejected")
        tracked = tracker.track(result, side=Side.BUY)
        assert tracked is None
        assert tracker.tracked_count == 0

    async def test_track_no_order_id_returns_none(self, mock_client):
        tracker = OrderTracker(mock_client)
        result = OrderResult(success=True, order_id="")
        tracked = tracker.track(result, side=Side.BUY)
        assert tracked is None

    async def test_untrack(self, mock_client):
        tracker = OrderTracker(mock_client)
        result = OrderResult(
            success=True, order_id="ORD-1", status=OrderStatus.OPEN,
            requested_size=10.0, requested_price=0.5,
        )
        tracker.track(result, side=Side.BUY, instrument_id="test-token")
        removed = tracker.untrack("ORD-1")
        assert removed is not None
        assert tracker.tracked_count == 0

    async def test_untrack_nonexistent(self, mock_client):
        tracker = OrderTracker(mock_client)
        removed = tracker.untrack("nonexistent")
        assert removed is None

    async def test_get(self, mock_client):
        tracker = OrderTracker(mock_client)
        result = OrderResult(
            success=True, order_id="ORD-1", status=OrderStatus.OPEN,
            requested_size=10.0, requested_price=0.5,
        )
        tracker.track(result, side=Side.BUY, instrument_id="test-token")
        assert tracker.get("ORD-1") is not None
        assert tracker.get("nonexistent") is None

    async def test_open_orders_filters_terminal(self, mock_client):
        tracker = OrderTracker(mock_client)
        # Track an open and a filled order
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.OPEN,
                        requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="t1",
        )
        tracker.track(
            OrderResult(success=True, order_id="O-2", status=OrderStatus.FILLED,
                        filled_size=10.0, requested_size=10.0, requested_price=0.5),
            side=Side.SELL, instrument_id="t2",
        )
        assert len(tracker.open_orders) == 1
        assert tracker.open_orders[0].order_id == "O-1"

    async def test_purge_terminal(self, mock_client):
        tracker = OrderTracker(mock_client)
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.OPEN,
                        requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="t1",
        )
        tracker.track(
            OrderResult(success=True, order_id="O-2", status=OrderStatus.FILLED,
                        filled_size=10.0, requested_size=10.0, requested_price=0.5),
            side=Side.SELL, instrument_id="t2",
        )
        purged = tracker.purge_terminal()
        assert purged == 1
        assert tracker.tracked_count == 1
        assert tracker.get("O-1") is not None

    async def test_poll_terminal_order_no_exchange_call(self, mock_client):
        """Polling a terminal order should not hit the exchange."""
        tracker = OrderTracker(mock_client)
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.FILLED,
                        filled_size=10.0, requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="t1",
        )
        update = await tracker.poll("O-1")
        assert update is not None
        assert update.new_fill_qty == 0.0

    async def test_poll_nonexistent_returns_none(self, mock_client):
        tracker = OrderTracker(mock_client)
        update = await tracker.poll("nonexistent")
        assert update is None

    async def test_poll_falls_back_to_open_orders(self, mock_client):
        """When get_order_status returns None, fallback to open orders scan."""
        # Mock client's get_open_orders returns empty -> order disappeared
        tracker = OrderTracker(mock_client)
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.OPEN,
                        requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="test-token",
        )
        # get_open_orders returns [] by default, so order is "gone"
        update = await tracker.poll("O-1")
        assert update is not None
        # Order disappeared with no fills -> assumed cancelled
        assert update.tracked.status == OrderStatus.CANCELLED

    async def test_poll_detects_fill_via_open_orders(self, mock_client):
        """Detect partial fill from open orders list."""
        tracker = OrderTracker(mock_client)
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.OPEN,
                        requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="test-token",
        )

        # Simulate exchange returning the order with partial fill
        partial_open = OpenOrder(
            order_id="O-1", instrument_id="test-token",
            side=Side.BUY, size=10.0, filled_size=4.0,
            price=0.5, order_type=OrderType.LIMIT,
            status=OrderStatus.OPEN,
        )

        async def mock_get_open_orders(instrument_id=None):
            return [partial_open]

        mock_client.get_open_orders = mock_get_open_orders

        update = await tracker.poll("O-1")
        assert update is not None
        assert update.had_new_fill is True
        assert update.new_fill_qty == pytest.approx(4.0)
        assert update.tracked.filled_size == pytest.approx(4.0)
        # Auto-detected partial fill status
        assert update.tracked.status == OrderStatus.PARTIALLY_FILLED

    async def test_poll_all_skips_terminal(self, mock_client):
        tracker = OrderTracker(mock_client)
        tracker.track(
            OrderResult(success=True, order_id="O-1", status=OrderStatus.OPEN,
                        requested_size=10.0, requested_price=0.5),
            side=Side.BUY, instrument_id="test-token",
        )
        tracker.track(
            OrderResult(success=True, order_id="O-2", status=OrderStatus.FILLED,
                        filled_size=5.0, requested_size=5.0, requested_price=0.3),
            side=Side.SELL, instrument_id="test-token",
        )
        updates = await tracker.poll_all()
        # Only O-1 should be polled
        assert len(updates) == 1
        assert updates[0].tracked.order_id == "O-1"


# ========================= OrderUpdate tests =========================


class TestOrderUpdate:
    def test_had_new_fill(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            filled_size=5.0, status=OrderStatus.PARTIALLY_FILLED,
        )
        update = OrderUpdate(tracked=t, prev_filled_size=2.0, prev_status=OrderStatus.OPEN, new_fill_qty=3.0)
        assert update.had_new_fill is True
        assert update.status_changed is True
        assert update.is_partial is True
        assert update.is_filled is False

    def test_no_new_fill(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            status=OrderStatus.OPEN,
        )
        update = OrderUpdate(tracked=t, prev_filled_size=0.0, prev_status=OrderStatus.OPEN)
        assert update.had_new_fill is False
        assert update.status_changed is False

    def test_is_filled(self):
        t = TrackedOrder(
            order_id="O1", instrument_id="X", side=Side.BUY,
            requested_size=10.0, requested_price=0.5,
            filled_size=10.0, status=OrderStatus.FILLED,
        )
        update = OrderUpdate(
            tracked=t, prev_filled_size=7.0,
            prev_status=OrderStatus.PARTIALLY_FILLED, new_fill_qty=3.0,
        )
        assert update.is_filled is True
        assert update.is_partial is False
        assert update.had_new_fill is True
