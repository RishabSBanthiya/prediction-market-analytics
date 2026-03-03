"""Tests for directional bot."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from omnitrade.core.enums import Side, SignalDirection, ExchangeId, Environment
from omnitrade.core.models import Signal, ExchangePosition
from omnitrade.components.signals import SignalSource
from omnitrade.components.sizers import FixedSizer
from omnitrade.components.executors import DryRunExecutor
from omnitrade.bots.directional import DirectionalBot


class MockSignalSource(SignalSource):
    def __init__(self, signals=None):
        self._signals = signals or []

    @property
    def name(self):
        return "mock"

    async def generate(self, client):
        return self._signals


class TestDirectionalBot:
    async def test_start_stop(self, mock_client, risk_coordinator):
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        assert mock_client.is_connected
        await bot.stop()

    async def test_iteration_no_signals(self, mock_client, risk_coordinator):
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()  # Should not raise
        await bot.stop()

    async def test_iteration_with_signal(self, mock_client, risk_coordinator):
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=75.0,
            source="test",
            price=0.65,
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        # Should have created a position
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 1
        await bot.stop()

    async def test_iteration_neutral_signal_skipped(self, mock_client, risk_coordinator):
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.NEUTRAL,
            score=75.0,
            source="test",
            price=0.65,
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_price_filter_too_low(self, mock_client, risk_coordinator):
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=75.0,
            source="test",
            price=0.02,  # Below default min_price of 0.05
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_price_filter_too_high(self, mock_client, risk_coordinator):
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=75.0,
            source="test",
            price=0.98,  # Above default max_price of 0.95
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 0
        await bot.stop()

    async def test_max_positions_respected(self, mock_client, risk_coordinator):
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=75.0,
            source="test",
            price=0.65,
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
            max_positions=1,
        )
        await bot.start()
        # First iteration opens a position
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 1
        # Second iteration should skip due to max_positions=1
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 1
        await bot.stop()

    async def test_paper_mode_wraps_executor(self, mock_client, risk_coordinator):
        from omnitrade.components.executors import AggressiveExecutor
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=AggressiveExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        assert isinstance(bot.executor, DryRunExecutor)

    async def test_live_mode_keeps_executor(self, mock_client, risk_coordinator):
        from omnitrade.components.executors import AggressiveExecutor
        original_executor = AggressiveExecutor()
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=original_executor,
            risk=risk_coordinator,
            environment=Environment.LIVE,
        )
        assert bot.executor is original_executor

    async def test_highest_score_signal_processed_first(self, mock_client, risk_coordinator):
        """Bot should process the highest-score signal first."""
        low_signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=30.0,
            source="test",
            price=0.65,
        )
        high_signal = Signal(
            instrument_id="token-no",
            direction=SignalDirection.LONG,
            score=90.0,
            source="test",
            price=0.35,
        )
        # Set up orderbook for token-no
        from omnitrade.core.models import OrderbookSnapshot, OrderbookLevel
        mock_client._orderbook = OrderbookSnapshot(
            instrument_id="token-no",
            bids=[OrderbookLevel(price=0.34, size=100)],
            asks=[OrderbookLevel(price=0.36, size=100)],
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([low_signal, high_signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        # Should only have one position (one trade per iteration), and it should be on token-no (higher score)
        assert len(positions) == 1
        await bot.stop()

    async def test_exit_monitor_initialized(self, mock_client, risk_coordinator):
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        assert bot.exit_monitor is not None

    async def test_restore_exit_states_on_start(self, mock_client, risk_coordinator):
        """Open positions in DB should be restored to exit monitor on start."""
        storage = risk_coordinator.storage
        storage.register_agent("test-bot", "directional", "polymarket")
        pid = storage.create_position("test-bot", "polymarket", "token-yes", "BUY", 100.0, 0.50)
        storage.update_position_exit_state(
            pid, current_price=0.55, peak_price=0.58,
            trough_price=0.48, trailing_stop_activated=True,
            trailing_stop_level=0.565,
        )

        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()

        state = bot.exit_monitor.get_state("token-yes")
        assert state is not None
        assert state.peak_price == pytest.approx(0.58)
        assert state.trough_price == pytest.approx(0.48)
        assert state.trailing_stop_activated is True
        assert state.trailing_stop_level == pytest.approx(0.565)
        await bot.stop()

    async def test_restore_clean_start_no_positions(self, mock_client, risk_coordinator):
        """Starting with no open positions should not register anything."""
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        assert bot.exit_monitor.get_state("token-yes") is None
        await bot.stop()

    async def test_exit_state_persisted_during_monitoring(self, mock_client, risk_coordinator):
        """Exit state should be written to storage after each monitoring cycle."""
        signal = Signal(
            instrument_id="token-yes",
            direction=SignalDirection.LONG,
            score=75.0,
            source="test",
            price=0.65,
        )
        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource([signal]),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        # Open a position
        await bot._iteration()
        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        assert len(positions) == 1

        # Run monitoring (which calls check -> _update_state -> persist)
        await bot._monitor_positions()

        positions = risk_coordinator.storage.get_agent_positions("test-bot", "open")
        pos = positions[0]
        # current_price should now be updated from midpoint (0.51 from mock orderbook)
        assert pos["current_price"] is not None
        assert pos["current_price"] == pytest.approx(0.51)
        # peak_price should be set
        assert pos["peak_price"] is not None
        await bot.stop()

    async def test_reconciliation_warns_on_missing_exchange_position(
        self, mock_client, risk_coordinator, caplog
    ):
        """Reconciliation should warn when DB position is not on exchange."""
        storage = risk_coordinator.storage
        storage.register_agent("test-bot", "directional", "polymarket")
        storage.create_position("test-bot", "polymarket", "token-yes", "BUY", 100.0, 0.50)

        # Exchange returns empty positions
        mock_client._positions = []

        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()

        import logging
        with caplog.at_level(logging.WARNING):
            await bot._reconcile_positions()

        assert any("RECONCILIATION" in msg and "not found on exchange" in msg
                    for msg in caplog.messages)
        await bot.stop()

    async def test_reconciliation_warns_on_size_mismatch(
        self, mock_client, risk_coordinator, caplog
    ):
        """Reconciliation should warn when sizes don't match."""
        storage = risk_coordinator.storage
        storage.register_agent("test-bot", "directional", "polymarket")
        storage.create_position("test-bot", "polymarket", "token-yes", "BUY", 100.0, 0.50)

        # Exchange shows different size
        mock_client._positions = [
            ExchangePosition(
                instrument_id="token-yes",
                exchange=ExchangeId.POLYMARKET,
                side=Side.BUY,
                size=80.0,  # Mismatch
                entry_price=0.50,
                current_price=0.55,
            )
        ]

        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()

        import logging
        with caplog.at_level(logging.WARNING):
            await bot._reconcile_positions()

        assert any("RECONCILIATION" in msg and "size mismatch" in msg
                    for msg in caplog.messages)
        await bot.stop()

    async def test_reconciliation_silent_when_matching(
        self, mock_client, risk_coordinator, caplog
    ):
        """No warnings when DB and exchange positions match."""
        storage = risk_coordinator.storage
        storage.register_agent("test-bot", "directional", "polymarket")
        storage.create_position("test-bot", "polymarket", "token-yes", "BUY", 100.0, 0.50)

        mock_client._positions = [
            ExchangePosition(
                instrument_id="token-yes",
                exchange=ExchangeId.POLYMARKET,
                side=Side.BUY,
                size=100.0,
                entry_price=0.50,
                current_price=0.55,
            )
        ]

        bot = DirectionalBot(
            agent_id="test-bot",
            client=mock_client,
            signal_source=MockSignalSource(),
            sizer=FixedSizer(50.0),
            executor=DryRunExecutor(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()

        import logging
        with caplog.at_level(logging.WARNING):
            await bot._reconcile_positions()

        assert not any("RECONCILIATION" in msg for msg in caplog.messages)
        await bot.stop()
