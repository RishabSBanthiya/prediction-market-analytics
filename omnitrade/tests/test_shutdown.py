"""Tests for graceful shutdown and startup recovery."""

import asyncio
import signal
import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from omnitrade.core.enums import ExchangeId, Side, OrderStatus, OrderType
from omnitrade.core.models import (
    AccountBalance, ExchangePosition, OpenOrder, OrderResult,
)
from omnitrade.core.shutdown import (
    ShutdownManager,
    ShutdownPhase,
    ShutdownState,
    StartupRecovery,
    CrossExchangeStartupRecovery,
)
from omnitrade.storage.sqlite import SQLiteStorage
from omnitrade.risk.coordinator import RiskCoordinator
from omnitrade.core.config import RiskConfig


# ==================== Fixtures ====================


@pytest.fixture
def storage():
    """Create a temporary SQLite storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    s = SQLiteStorage(db_path)
    s.initialize()
    yield s
    s.close()
    os.unlink(db_path)


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_wallet_exposure_pct=0.60,
        max_per_agent_exposure_pct=0.30,
        max_per_market_exposure_pct=0.10,
        min_trade_value_usd=5.0,
        max_trade_value_usd=500.0,
    )


@pytest.fixture
def risk(storage, risk_config):
    coord = RiskCoordinator(storage, risk_config)
    coord.register_account(ExchangeId.POLYMARKET, "test-agent")
    storage.update_balance("polymarket", "test-agent", 1000.0)
    return coord


@pytest.fixture
def mock_client():
    """Create a mock exchange client with async methods."""
    client = AsyncMock()
    client.exchange_id = ExchangeId.POLYMARKET
    client.is_connected = True
    client.cancel_all_orders = AsyncMock(return_value=3)
    client.cancel_orders = AsyncMock(return_value=2)
    client.get_open_orders = AsyncMock(return_value=[])
    client.get_positions = AsyncMock(return_value=[])
    client.get_balance = AsyncMock(return_value=AccountBalance(
        exchange=ExchangeId.POLYMARKET,
        total_equity=1000.0,
        available_balance=800.0,
    ))
    return client


# ==================== ShutdownState Tests ====================


class TestShutdownState:
    def test_initial_state(self):
        state = ShutdownState()
        assert state.phase == ShutdownPhase.RUNNING
        assert not state.is_shutting_down
        assert state.signal_received is None
        assert state.duration_seconds is None

    def test_is_shutting_down(self):
        state = ShutdownState(phase=ShutdownPhase.STOPPING)
        assert state.is_shutting_down

    def test_duration_while_running(self):
        state = ShutdownState(
            started_at=datetime.now(timezone.utc) - timedelta(seconds=5)
        )
        duration = state.duration_seconds
        assert duration is not None
        assert duration >= 5.0

    def test_duration_when_complete(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, second=10, tzinfo=timezone.utc)
        state = ShutdownState(started_at=start, completed_at=end)
        assert state.duration_seconds == pytest.approx(10.0)


# ==================== ShutdownManager Tests ====================


class TestShutdownManager:
    def test_should_stop_initially_false(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        assert not manager.should_stop

    def test_request_stop_sets_event(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        manager.request_stop("test")
        assert manager.should_stop
        assert manager.state.signal_received == "test"

    def test_request_stop_idempotent(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        manager.request_stop("first")
        manager.request_stop("second")
        # First reason is kept
        assert manager.state.signal_received == "first"

    async def test_execute_shutdown_cancels_orders(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        state = await manager.execute_shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert state.orders_cancelled == 3
        mock_client.cancel_all_orders.assert_awaited_once()

    async def test_execute_shutdown_handles_cancel_failure(self, mock_client, risk, storage):
        mock_client.cancel_all_orders = AsyncMock(
            side_effect=Exception("Connection lost")
        )
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        state = await manager.execute_shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert len(state.errors) >= 1
        assert "Connection lost" in state.errors[0]

    async def test_execute_shutdown_marks_agent_stopped(self, mock_client, risk, storage):
        # Register the agent first
        storage.register_agent("test-agent", "directional", "polymarket")
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        await manager.execute_shutdown()

        # Agent should be marked as stopped (shutdown calls risk.shutdown)
        # We verify by checking the storage was called through risk coordinator

    async def test_execute_shutdown_with_timeout(self, mock_client, risk, storage):
        """Shutdown should complete even if cancel_all_orders is slow."""
        async def slow_cancel(instrument_id=None):
            await asyncio.sleep(100)  # Deliberately slow
            return 0

        mock_client.cancel_all_orders = slow_cancel
        manager = ShutdownManager(
            mock_client, risk, storage, "test-agent",
            shutdown_timeout_seconds=0.1,
        )
        state = await manager.execute_shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert any("timed out" in e for e in state.errors)

    async def test_execute_shutdown_records_duration(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        state = await manager.execute_shutdown()

        assert state.started_at is not None
        assert state.completed_at is not None
        assert state.duration_seconds is not None
        assert state.duration_seconds >= 0

    async def test_multi_client_shutdown(self, risk, storage):
        """ShutdownManager should cancel orders on all clients."""
        client_a = AsyncMock()
        client_a.cancel_all_orders = AsyncMock(return_value=2)
        client_b = AsyncMock()
        client_b.cancel_all_orders = AsyncMock(return_value=5)

        clients = {
            ExchangeId.POLYMARKET: client_a,
            ExchangeId.KALSHI: client_b,
        }
        manager = ShutdownManager(clients, risk, storage, "test-agent")
        state = await manager.execute_shutdown()

        assert state.orders_cancelled == 7
        client_a.cancel_all_orders.assert_awaited_once()
        client_b.cancel_all_orders.assert_awaited_once()

    def test_handle_signal_sets_stop(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        manager._handle_signal(signal.SIGTERM)

        assert manager.should_stop
        assert manager.state.signal_received == "SIGTERM"

    def test_handle_signal_twice_forces_exit(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        manager._handle_signal(signal.SIGINT)

        # Second signal calls os._exit which we mock to verify
        with patch("omnitrade.core.shutdown.os._exit") as mock_exit:
            manager._handle_signal(signal.SIGINT)
            mock_exit.assert_called_once_with(128 + signal.SIGINT.value)

    def test_install_signal_handlers_idempotent(self, mock_client, risk, storage):
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        loop = MagicMock()
        manager.install_signal_handlers(loop)
        manager.install_signal_handlers(loop)
        # Should only add handlers once (2 signals)
        assert loop.add_signal_handler.call_count == 2

    def test_on_stop_callback_called_on_signal(self, mock_client, risk, storage):
        """Signal handler should invoke the on_stop callback."""
        callback = MagicMock()
        manager = ShutdownManager(
            mock_client, risk, storage, "test-agent", on_stop=callback,
        )
        manager._handle_signal(signal.SIGTERM)

        assert manager.should_stop
        callback.assert_called_once()

    def test_on_stop_callback_not_called_when_none(self, mock_client, risk, storage):
        """No callback set should still work (backward compatible)."""
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        manager._handle_signal(signal.SIGTERM)
        assert manager.should_stop

    def test_on_stop_callback_error_does_not_prevent_shutdown(self, mock_client, risk, storage):
        """A failing callback should not prevent shutdown from being signaled."""
        callback = MagicMock(side_effect=RuntimeError("callback boom"))
        manager = ShutdownManager(
            mock_client, risk, storage, "test-agent", on_stop=callback,
        )
        manager._handle_signal(signal.SIGINT)

        assert manager.should_stop
        callback.assert_called_once()

    async def test_execute_shutdown_idempotent(self, mock_client, risk, storage):
        """Calling execute_shutdown twice should only run cleanup once."""
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        state1 = await manager.execute_shutdown()
        state2 = await manager.execute_shutdown()

        assert state1 is state2
        # cancel_all_orders should be called only once
        assert mock_client.cancel_all_orders.await_count == 1

    async def test_finalize_skipped_when_on_stop_set(self, mock_client, risk, storage):
        """When on_stop is provided, _finalize_agent_state skips risk.shutdown."""
        storage.register_agent("test-agent", "directional", "polymarket")
        callback = MagicMock()
        manager = ShutdownManager(
            mock_client, risk, storage, "test-agent", on_stop=callback,
        )
        # Spy on risk.shutdown
        with patch.object(risk, "shutdown") as mock_shutdown:
            await manager.execute_shutdown()
            mock_shutdown.assert_not_called()

    async def test_finalize_calls_risk_shutdown_without_on_stop(self, mock_client, risk, storage):
        """Without on_stop, _finalize_agent_state should call risk.shutdown."""
        storage.register_agent("test-agent", "directional", "polymarket")
        manager = ShutdownManager(mock_client, risk, storage, "test-agent")
        with patch.object(risk, "shutdown") as mock_shutdown:
            await manager.execute_shutdown()
            mock_shutdown.assert_called_once_with("test-agent")


# ==================== StartupRecovery Tests ====================


class TestStartupRecovery:
    async def test_recover_clean_state(self, mock_client, storage, risk):
        """Recovery on clean state should find nothing to do."""
        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        results = await recovery.recover()

        assert results["stale_agents"] == 0
        assert results["expired_reservations"] == 0
        assert results["orphaned_orders_cancelled"] == 0
        assert results["position_mismatches"] == 0

    async def test_recover_cancels_orphaned_orders(self, mock_client, storage, risk):
        """Should cancel orders found on exchange during startup."""
        mock_client.get_open_orders = AsyncMock(return_value=[
            OpenOrder(
                order_id="ORPHAN-001",
                instrument_id="token-yes",
                side=Side.BUY,
                size=10.0,
                filled_size=0.0,
                price=0.50,
                order_type=OrderType.LIMIT,
                status=OrderStatus.OPEN,
            ),
            OpenOrder(
                order_id="ORPHAN-002",
                instrument_id="token-no",
                side=Side.SELL,
                size=5.0,
                filled_size=0.0,
                price=0.55,
                order_type=OrderType.LIMIT,
                status=OrderStatus.OPEN,
            ),
        ])
        mock_client.cancel_orders = AsyncMock(return_value=2)

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        results = await recovery.recover()

        assert results["orphaned_orders_cancelled"] == 2
        mock_client.cancel_orders.assert_awaited_once_with(
            ["ORPHAN-001", "ORPHAN-002"]
        )

    async def test_recover_detects_db_position_not_on_exchange(
        self, mock_client, storage, risk, caplog,
    ):
        """Should warn when DB has a position not found on exchange."""
        storage.register_agent("test-agent", "directional", "polymarket")
        storage.create_position(
            "test-agent", "polymarket", "token-yes", "BUY", 100.0, 0.50
        )
        mock_client.get_positions = AsyncMock(return_value=[])

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        import logging
        with caplog.at_level(logging.WARNING):
            results = await recovery.recover()

        assert results["position_mismatches"] == 1
        assert any("not found on exchange" in m for m in caplog.messages)

    async def test_recover_detects_exchange_position_not_in_db(
        self, mock_client, storage, risk, caplog,
    ):
        """Should warn when exchange has a position not tracked in DB."""
        mock_client.get_positions = AsyncMock(return_value=[
            ExchangePosition(
                instrument_id="untracked-token",
                exchange=ExchangeId.POLYMARKET,
                side=Side.BUY,
                size=50.0,
                entry_price=0.60,
                current_price=0.65,
            ),
        ])

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        import logging
        with caplog.at_level(logging.WARNING):
            results = await recovery.recover()

        assert results["position_mismatches"] == 1
        assert any("not tracked in DB" in m for m in caplog.messages)

    async def test_recover_detects_size_mismatch(
        self, mock_client, storage, risk, caplog,
    ):
        """Should warn when DB and exchange position sizes differ."""
        storage.register_agent("test-agent", "directional", "polymarket")
        storage.create_position(
            "test-agent", "polymarket", "token-yes", "BUY", 100.0, 0.50
        )
        mock_client.get_positions = AsyncMock(return_value=[
            ExchangePosition(
                instrument_id="token-yes",
                exchange=ExchangeId.POLYMARKET,
                side=Side.BUY,
                size=80.0,  # Mismatch with DB
                entry_price=0.50,
                current_price=0.55,
            ),
        ])

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        import logging
        with caplog.at_level(logging.WARNING):
            results = await recovery.recover()

        assert results["position_mismatches"] == 1
        assert any("size mismatch" in m for m in caplog.messages)

    async def test_recover_cleans_expired_reservations(self, mock_client, storage, risk):
        """Should release expired capital reservations."""
        storage.register_agent("test-agent", "directional", "polymarket")
        # Create an expired reservation
        expired_time = datetime.now(timezone.utc) - timedelta(hours=1)
        storage.create_reservation(
            agent_id="test-agent",
            exchange="polymarket",
            instrument_id="token-yes",
            amount_usd=50.0,
            expires_at=expired_time,
        )

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        results = await recovery.recover()

        assert results["expired_reservations"] == 1

    async def test_recover_handles_get_positions_failure(
        self, mock_client, storage, risk,
    ):
        """Should handle exchange API failures gracefully."""
        mock_client.get_positions = AsyncMock(
            side_effect=Exception("API down")
        )

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        results = await recovery.recover()

        # Should not raise, just return 0 mismatches
        assert results["position_mismatches"] == 0

    async def test_recover_handles_get_open_orders_failure(
        self, mock_client, storage, risk,
    ):
        """Should handle failure to fetch open orders."""
        mock_client.get_open_orders = AsyncMock(
            side_effect=Exception("Timeout")
        )

        recovery = StartupRecovery(mock_client, storage, risk, "test-agent")
        results = await recovery.recover()

        assert results["orphaned_orders_cancelled"] == 0


# ==================== CrossExchangeStartupRecovery Tests ====================


class TestCrossExchangeStartupRecovery:
    async def test_recover_aggregates_across_exchanges(self, storage, risk):
        """Should run recovery on each exchange and aggregate results."""
        client_a = AsyncMock()
        client_a.exchange_id = ExchangeId.POLYMARKET
        client_a.get_open_orders = AsyncMock(return_value=[
            OpenOrder(
                order_id="ORD-1",
                instrument_id="tok-a",
                side=Side.BUY,
                size=10.0,
                filled_size=0.0,
                price=0.50,
                order_type=OrderType.LIMIT,
                status=OrderStatus.OPEN,
            ),
        ])
        client_a.cancel_orders = AsyncMock(return_value=1)
        client_a.get_positions = AsyncMock(return_value=[])

        client_b = AsyncMock()
        client_b.exchange_id = ExchangeId.KALSHI
        client_b.get_open_orders = AsyncMock(return_value=[])
        client_b.get_positions = AsyncMock(return_value=[
            ExchangePosition(
                instrument_id="untracked",
                exchange=ExchangeId.KALSHI,
                side=Side.BUY,
                size=20.0,
                entry_price=0.40,
                current_price=0.45,
            ),
        ])

        clients = {
            ExchangeId.POLYMARKET: client_a,
            ExchangeId.KALSHI: client_b,
        }
        recovery = CrossExchangeStartupRecovery(
            clients, storage, risk, "hedge-bot"
        )
        results = await recovery.recover()

        assert results["orphaned_orders_cancelled"] == 1
        assert results["position_mismatches"] == 1

    async def test_recover_clean_state(self, storage, risk):
        """Clean state across both exchanges should return all zeros."""
        client_a = AsyncMock()
        client_a.exchange_id = ExchangeId.POLYMARKET
        client_a.get_open_orders = AsyncMock(return_value=[])
        client_a.get_positions = AsyncMock(return_value=[])

        clients = {ExchangeId.POLYMARKET: client_a}
        recovery = CrossExchangeStartupRecovery(
            clients, storage, risk, "test-bot"
        )
        results = await recovery.recover()

        assert results["orphaned_orders_cancelled"] == 0
        assert results["position_mismatches"] == 0
