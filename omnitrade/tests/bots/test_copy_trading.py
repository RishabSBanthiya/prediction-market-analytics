"""Tests for copy trading bot."""

import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from omnitrade.core.enums import Side
from omnitrade.core.models import OrderResult, OrderbookSnapshot, OrderbookLevel
from omnitrade.exchanges.base import PaperClient
from omnitrade.bots.copy_trading import (
    CopyTradingBot,
    CopyConfig,
    TargetTracker,
    TargetAccount,
    TargetPosition,
    PositionDelta,
)


# ==================== TargetTracker Tests ====================


class TestTargetTracker:
    def test_is_first_poll_initially_true(self):
        tracker = TargetTracker()
        assert tracker.is_first_poll is True

    async def test_first_poll_creates_snapshot(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC", label="whale")

        # Mock the fetch to return positions
        async def mock_fetch(address):
            return {
                "token-yes": TargetPosition(
                    instrument_id="token-yes", side=Side.BUY,
                    size=100.0, price=0.65, market_name="Test YES",
                ),
            }
        tracker._fetch_positions = mock_fetch

        deltas = await tracker.poll_and_diff(target)
        # First poll: all positions are "new"
        assert len(deltas) == 1
        assert deltas[0].instrument_id == "token-yes"
        assert deltas[0].size_delta == 100.0
        assert deltas[0].is_new_position
        assert tracker.is_first_poll is False

    async def test_no_change_produces_no_deltas(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC")

        positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        tracker._fetch_positions = AsyncMock(return_value=positions)

        # First poll
        await tracker.poll_and_diff(target)
        # Second poll with same data
        deltas = await tracker.poll_and_diff(target)
        assert len(deltas) == 0

    async def test_increased_position_detected(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC")

        initial = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        tracker._fetch_positions = AsyncMock(return_value=initial)
        await tracker.poll_and_diff(target)

        # Now increase
        increased = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=200.0, price=0.70,
            ),
        }
        tracker._fetch_positions = AsyncMock(return_value=increased)
        deltas = await tracker.poll_and_diff(target)

        assert len(deltas) == 1
        assert deltas[0].size_delta == 100.0  # +100 shares
        assert deltas[0].is_new_position

    async def test_closed_position_detected(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC")

        initial = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        tracker._fetch_positions = AsyncMock(return_value=initial)
        await tracker.poll_and_diff(target)

        # Now close
        tracker._fetch_positions = AsyncMock(return_value={})
        deltas = await tracker.poll_and_diff(target)

        assert len(deltas) == 1
        assert deltas[0].size_delta == -100.0
        assert deltas[0].is_close

    async def test_new_instrument_detected(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC")

        initial = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        tracker._fetch_positions = AsyncMock(return_value=initial)
        await tracker.poll_and_diff(target)

        # Add a new position
        updated = dict(initial)
        updated["token-no"] = TargetPosition(
            instrument_id="token-no", side=Side.BUY,
            size=50.0, price=0.35,
        )
        tracker._fetch_positions = AsyncMock(return_value=updated)
        deltas = await tracker.poll_and_diff(target)

        assert len(deltas) == 1
        assert deltas[0].instrument_id == "token-no"
        assert deltas[0].size_delta == 50.0


# ==================== CopyConfig Tests ====================


class TestCopyConfig:
    def test_defaults(self):
        config = CopyConfig()
        assert config.size_multiplier == 1.0
        assert config.copy_exits is True
        assert config.min_price == 0.05
        assert config.max_price == 0.95

    def test_custom_config(self):
        config = CopyConfig(size_multiplier=0.5, copy_exits=False)
        assert config.size_multiplier == 0.5
        assert config.copy_exits is False


# ==================== CopyTradingBot Tests ====================


def _make_copy_bot(mock_client, risk_coordinator, targets=None, config=None):
    """Helper to create a CopyTradingBot with validation pre-mocked."""
    paper_client = PaperClient(mock_client)
    tracker = TargetTracker()
    # Pre-mock _fetch_positions so validation succeeds during start()
    tracker._fetch_positions = AsyncMock(return_value={})
    targets = targets or [TargetAccount(address="0xABC", label="whale")]
    return CopyTradingBot(
        agent_id="test-copy-bot",
        client=paper_client,
        tracker=tracker,
        targets=targets,
        risk=risk_coordinator,
        config=config or CopyConfig(min_trade_usd=5.0, cooldown_seconds=0),
    )


class TestCopyTradingBot:
    async def test_start_stop(self, mock_client, risk_coordinator):
        bot = _make_copy_bot(mock_client, risk_coordinator)
        await bot.start()
        assert mock_client.is_connected
        await bot.stop()

    async def test_first_iteration_snapshots_only(self, mock_client, risk_coordinator):
        """First iteration should snapshot existing positions, not copy them."""
        bot = _make_copy_bot(mock_client, risk_coordinator)
        await bot.start()

        # Mock tracker to return positions on first poll
        async def mock_fetch(address):
            return {
                "token-yes": TargetPosition(
                    instrument_id="token-yes", side=Side.BUY,
                    size=100.0, price=0.65,
                ),
            }
        bot.tracker._fetch_positions = mock_fetch

        await bot._iteration()

        # Should NOT have opened any positions (first poll = snapshot only)
        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_copies_new_position_on_second_poll(self, mock_client, risk_coordinator):
        """After initial snapshot, new positions should be copied."""
        bot = _make_copy_bot(mock_client, risk_coordinator)
        await bot.start()

        # First poll: empty
        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # Second poll: new position appears
        new_positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.51,  # Close to mock midpoint of 0.51
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=new_positions)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 1
        assert positions[0]["instrument_id"] == "token-yes"
        assert positions[0]["side"] == "BUY"
        await bot.stop()

    async def test_skips_when_price_out_of_bounds(self, mock_client, risk_coordinator):
        """Positions with price outside min/max should be skipped."""
        config = CopyConfig(min_price=0.10, max_price=0.90, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, config=config)
        await bot.start()

        # Set midpoint to extreme
        mock_client._orderbook = OrderbookSnapshot(
            instrument_id="test-token",
            bids=[OrderbookLevel(price=0.02, size=100)],
            asks=[OrderbookLevel(price=0.04, size=100)],
        )

        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # New position at extreme price
        new_positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.03,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=new_positions)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_skips_close_when_copy_exits_disabled(self, mock_client, risk_coordinator):
        """When copy_exits=False, position closes should not be copied."""
        config = CopyConfig(copy_exits=False, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, config=config)
        await bot.start()

        # Initial snapshot with a position
        initial = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=initial)
        await bot._iteration()

        # Target closes position
        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # No crash, no copy trade for the close
        # (We didn't mirror it, so nothing to check beyond no errors)
        await bot.stop()

    async def test_size_multiplier_applied(self, mock_client, risk_coordinator):
        """Size multiplier should scale the copy trade."""
        config = CopyConfig(size_multiplier=0.5, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, config=config)
        await bot.start()

        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # New position with known size
        new_positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.51,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=new_positions)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 1
        # Target value = 100 * 0.51 = $51, with 0.5x multiplier = $25.50
        # Shares = 25.50 / 0.51 = 50
        assert positions[0]["size"] == pytest.approx(50.0, rel=0.1)
        await bot.stop()

    async def test_target_weight_applied(self, mock_client, risk_coordinator):
        """Target weight should scale the copy trade."""
        targets = [TargetAccount(address="0xABC", label="small-whale", weight=0.25)]
        config = CopyConfig(size_multiplier=1.0, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, targets=targets, config=config)
        await bot.start()

        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        new_positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.51,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=new_positions)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 1
        # Target value = 100 * 0.51 = $51, with 0.25 weight = $12.75
        # Shares = 12.75 / 0.51 = 25
        assert positions[0]["size"] == pytest.approx(25.0, rel=0.1)
        await bot.stop()

    async def test_skips_price_deviation_too_high(self, mock_client, risk_coordinator):
        """Should skip when current price deviates too much from target's entry."""
        config = CopyConfig(max_price_deviation_pct=0.03, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, config=config)
        await bot.start()

        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # Target traded at 0.65 but our midpoint is 0.51 — big deviation
        new_positions = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=new_positions)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_multiple_targets(self, mock_client, risk_coordinator):
        """Multiple targets should all be tracked independently."""
        targets = [
            TargetAccount(address="0xAAA", label="trader-a"),
            TargetAccount(address="0xBBB", label="trader-b"),
        ]
        config = CopyConfig(cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, targets=targets, config=config)
        await bot.start()

        call_count = 0
        async def mock_fetch(address):
            nonlocal call_count
            call_count += 1
            return {}
        bot.tracker._fetch_positions = mock_fetch
        await bot._iteration()

        # Both targets should have been polled
        assert call_count == 2
        await bot.stop()

    async def test_copies_exit_when_configured(self, mock_client, risk_coordinator):
        """When copy_exits=True, target closing should close our mirrored position."""
        config = CopyConfig(copy_exits=True, cooldown_seconds=0, min_trade_usd=5.0)
        bot = _make_copy_bot(mock_client, risk_coordinator, config=config)
        await bot.start()

        # Snapshot empty
        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        # Target opens position
        open_pos = {
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.51,
            ),
        }
        bot.tracker._fetch_positions = AsyncMock(return_value=open_pos)
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 1

        # Target closes position
        bot.tracker._fetch_positions = AsyncMock(return_value={})
        await bot._iteration()

        positions = risk_coordinator.storage.get_agent_positions("test-copy-bot", "open")
        assert len(positions) == 0
        await bot.stop()


# ==================== PositionDelta Tests ====================


class TestPositionDelta:
    def test_is_new_position(self):
        delta = PositionDelta(
            target=TargetAccount(address="0x"),
            instrument_id="test",
            side=Side.BUY,
            size_delta=10.0,
            current_price=0.5,
        )
        assert delta.is_new_position
        assert not delta.is_close

    def test_is_close(self):
        delta = PositionDelta(
            target=TargetAccount(address="0x"),
            instrument_id="test",
            side=Side.BUY,
            size_delta=-10.0,
            current_price=0.5,
        )
        assert delta.is_close
        assert not delta.is_new_position


# ==================== TargetAccount Tests ====================


class TestTargetAccount:
    def test_defaults(self):
        t = TargetAccount(address="0xABC")
        assert t.label == ""
        assert t.weight == 1.0

    def test_custom(self):
        t = TargetAccount(address="0xABC", label="whale", weight=0.5)
        assert t.label == "whale"
        assert t.weight == 0.5


# ==================== Validation Tests ====================


class TestTargetValidation:
    async def test_validate_target_success(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC", label="whale")

        tracker._fetch_positions = AsyncMock(return_value={
            "token-yes": TargetPosition(
                instrument_id="token-yes", side=Side.BUY,
                size=100.0, price=0.65,
            ),
        })

        is_valid, message, count = await tracker.validate_target(target)
        assert is_valid
        assert count == 1
        assert "OK" in message

    async def test_validate_target_empty_positions_still_valid(self):
        """An address with 0 positions is still valid (just no current trades)."""
        tracker = TargetTracker()
        target = TargetAccount(address="0xABC", label="inactive")

        tracker._fetch_positions = AsyncMock(return_value={})

        is_valid, message, count = await tracker.validate_target(target)
        assert is_valid
        assert count == 0

    async def test_validate_target_fetch_error(self):
        tracker = TargetTracker()
        target = TargetAccount(address="0xBAD", label="invalid")

        tracker._fetch_positions = AsyncMock(side_effect=Exception("404 Not Found"))

        is_valid, message, count = await tracker.validate_target(target)
        assert not is_valid
        assert "failed to fetch" in message
        assert count == 0

    async def test_bot_fails_when_all_targets_invalid(self, mock_client, risk_coordinator):
        """Bot should raise when no targets pass validation."""
        from omnitrade.core.errors import OmniTradeError

        paper_client = PaperClient(mock_client)
        tracker = TargetTracker()
        tracker._fetch_positions = AsyncMock(side_effect=Exception("unreachable"))

        bot = CopyTradingBot(
            agent_id="test-copy-bot",
            client=paper_client,
            tracker=tracker,
            targets=[TargetAccount(address="0xBAD", label="bad")],
            risk=risk_coordinator,
        )

        with pytest.raises(OmniTradeError, match="No valid targets"):
            await bot.start()

    async def test_bot_drops_invalid_keeps_valid(self, mock_client, risk_coordinator):
        """Bot should drop invalid targets but keep valid ones."""
        paper_client = PaperClient(mock_client)
        tracker = TargetTracker()

        call_count = 0
        async def mock_fetch(address):
            nonlocal call_count
            call_count += 1
            if address == "0xGOOD":
                return {}
            raise Exception("not found")

        tracker._fetch_positions = mock_fetch

        targets = [
            TargetAccount(address="0xGOOD", label="good"),
            TargetAccount(address="0xBAD", label="bad"),
        ]
        bot = CopyTradingBot(
            agent_id="test-copy-bot",
            client=paper_client,
            tracker=tracker,
            targets=targets,
            risk=risk_coordinator,
        )

        await bot.start()
        assert len(bot.targets) == 1
        assert bot.targets[0].address == "0xGOOD"
        await bot.stop()
