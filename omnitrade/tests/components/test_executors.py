"""Tests for executors."""

import pytest
from omnitrade.core.enums import Side, OrderStatus, ExchangeId, InstrumentType, SignalDirection
from omnitrade.core.models import OrderbookSnapshot, OrderbookLevel, Instrument
from omnitrade.components.executors import DryRunExecutor, AggressiveExecutor, LimitExecutor, Executor


@pytest.fixture
def orderbook():
    return OrderbookSnapshot(
        instrument_id="test-token",
        bids=[OrderbookLevel(price=0.50, size=100)],
        asks=[OrderbookLevel(price=0.52, size=100)],
    )


class TestDryRunExecutor:
    async def test_buy(self, mock_client, orderbook):
        executor = DryRunExecutor()
        result = await executor.execute(
            mock_client, "test-token", Side.BUY, 50.0, 0.52, orderbook
        )
        assert result.success
        assert result.order_id.startswith("DRY-")
        assert result.filled_size > 0

    async def test_sell(self, mock_client, orderbook):
        executor = DryRunExecutor()
        result = await executor.execute(
            mock_client, "test-token", Side.SELL, 50.0, 0.50, orderbook
        )
        assert result.success
        assert result.filled_price > 0

    async def test_buy_uses_ask_with_slippage(self, mock_client, orderbook):
        executor = DryRunExecutor(slippage_pct=0.01)
        result = await executor.execute(
            mock_client, "test-token", Side.BUY, 100.0, 0.52, orderbook
        )
        # Should execute at ask (0.52) * (1 + 0.01) = 0.5252
        assert result.success
        assert abs(result.filled_price - 0.5252) < 0.001

    async def test_sell_uses_bid_with_slippage(self, mock_client, orderbook):
        executor = DryRunExecutor(slippage_pct=0.01)
        result = await executor.execute(
            mock_client, "test-token", Side.SELL, 100.0, 0.50, orderbook
        )
        # Should execute at bid (0.50) * (1 - 0.01) = 0.495
        assert result.success
        assert abs(result.filled_price - 0.495) < 0.001

    async def test_name(self):
        assert DryRunExecutor().name == "dry_run"

    async def test_sequential_order_ids(self, mock_client, orderbook):
        executor = DryRunExecutor()
        r1 = await executor.execute(mock_client, "test", Side.BUY, 10.0, 0.50, orderbook)
        r2 = await executor.execute(mock_client, "test", Side.BUY, 10.0, 0.50, orderbook)
        assert r1.order_id != r2.order_id

    async def test_no_orderbook_fetches_from_client(self, mock_client):
        executor = DryRunExecutor()
        result = await executor.execute(
            mock_client, "test-token", Side.BUY, 50.0, 0.52
        )
        assert result.success

    async def test_zero_price_orderbook(self, mock_client):
        zero_book = OrderbookSnapshot(
            instrument_id="test",
            bids=[],
            asks=[],
        )
        executor = DryRunExecutor()
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.0, zero_book
        )
        assert not result.success


class TestAggressiveExecutor:
    async def test_spread_too_wide(self, mock_client):
        wide_book = OrderbookSnapshot(
            instrument_id="test",
            bids=[OrderbookLevel(price=0.30, size=100)],
            asks=[OrderbookLevel(price=0.70, size=100)],
        )
        executor = AggressiveExecutor(max_spread=0.03)
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.50, wide_book
        )
        assert not result.success
        assert result.is_rejection
        assert "Spread" in result.error_message

    async def test_spread_check_only_on_buy(self, mock_client):
        """Spread check should not block sells (exit should always be allowed)."""
        wide_book = OrderbookSnapshot(
            instrument_id="test",
            bids=[OrderbookLevel(price=0.30, size=100)],
            asks=[OrderbookLevel(price=0.70, size=100)],
        )
        executor = AggressiveExecutor(max_spread=0.03, max_slippage=1.0)
        result = await executor.execute(
            mock_client, "test", Side.SELL, 50.0, 0.30, wide_book
        )
        # Sell should pass spread check even with wide spread
        assert result.success

    async def test_slippage_rejected(self, mock_client, orderbook):
        executor = AggressiveExecutor(max_slippage=0.001)
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.40, orderbook
        )
        assert not result.success
        assert result.is_rejection

    async def test_successful_buy(self, mock_client, orderbook):
        executor = AggressiveExecutor(max_spread=0.10, max_slippage=0.10)
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.52, orderbook
        )
        assert result.success

    async def test_name(self):
        assert AggressiveExecutor().name == "aggressive"

    async def test_no_ask_available(self, mock_client):
        empty_ask_book = OrderbookSnapshot(
            instrument_id="test",
            bids=[OrderbookLevel(price=0.50, size=100)],
            asks=[],
        )
        executor = AggressiveExecutor(max_spread=0.10, max_slippage=0.10)
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.50, empty_ask_book
        )
        assert not result.success

    async def test_no_bid_available(self, mock_client):
        empty_bid_book = OrderbookSnapshot(
            instrument_id="test",
            bids=[],
            asks=[OrderbookLevel(price=0.50, size=100)],
        )
        executor = AggressiveExecutor(max_spread=0.10, max_slippage=0.10)
        result = await executor.execute(
            mock_client, "test", Side.SELL, 50.0, 0.50, empty_bid_book
        )
        assert not result.success


class TestLimitExecutor:
    async def test_buy_with_offset(self, mock_client, orderbook):
        executor = LimitExecutor(price_offset=0.01)
        result = await executor.execute(
            mock_client, "test", Side.BUY, 50.0, 0.50, orderbook
        )
        assert result.success

    async def test_sell_with_offset(self, mock_client, orderbook):
        executor = LimitExecutor(price_offset=0.01)
        result = await executor.execute(
            mock_client, "test", Side.SELL, 50.0, 0.50, orderbook
        )
        assert result.success

    async def test_name(self):
        assert LimitExecutor().name == "limit"

    async def test_invalid_limit_price(self, mock_client, orderbook):
        executor = LimitExecutor(price_offset=0.0)
        result = await executor.execute(
            mock_client, "test", Side.SELL, 50.0, 0.0, orderbook
        )
        assert not result.success


class TestExecutorStatic:
    def test_direction_to_side_long(self):
        assert Executor.direction_to_side(SignalDirection.LONG) == Side.BUY

    def test_direction_to_side_short(self):
        assert Executor.direction_to_side(SignalDirection.SHORT) == Side.SELL

    def test_direction_to_side_neutral_raises(self):
        with pytest.raises(ValueError):
            Executor.direction_to_side(SignalDirection.NEUTRAL)
