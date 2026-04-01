"""Tests for SQLite storage."""

import pytest
from datetime import datetime, timezone, timedelta


class TestSQLiteStorage:
    def test_balance(self, tmp_db):
        tmp_db.update_balance("polymarket", "wallet-1", 1000.0)
        assert tmp_db.get_balance("polymarket", "wallet-1") == 1000.0

    def test_balance_missing(self, tmp_db):
        assert tmp_db.get_balance("polymarket", "nonexistent") == 0.0

    def test_balance_update(self, tmp_db):
        tmp_db.update_balance("polymarket", "wallet-1", 1000.0)
        tmp_db.update_balance("polymarket", "wallet-1", 1500.0)
        assert tmp_db.get_balance("polymarket", "wallet-1") == 1500.0

    def test_multiple_exchange_balances(self, tmp_db):
        tmp_db.update_balance("polymarket", "wallet-1", 1000.0)
        tmp_db.update_balance("kalshi", "acct-1", 2000.0)
        assert tmp_db.get_balance("polymarket", "wallet-1") == 1000.0
        assert tmp_db.get_balance("kalshi", "acct-1") == 2000.0

    def test_agent_registration(self, tmp_db):
        assert tmp_db.register_agent("bot-1", "directional", "polymarket")
        # Second registration of active agent should fail
        assert not tmp_db.register_agent("bot-1", "directional", "polymarket")

    def test_agent_reregistration_after_stop(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.set_agent_status("bot-1", "stopped")
        # Should be able to re-register after stopping
        assert tmp_db.register_agent("bot-1", "directional", "polymarket")

    def test_agent_heartbeat(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.update_heartbeat("bot-1")  # Should not raise

    def test_agent_set_status(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.set_agent_status("bot-1", "crashed")
        # After crash, can re-register
        assert tmp_db.register_agent("bot-1", "directional", "polymarket")

    def test_reservation_lifecycle(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)

        rid = tmp_db.create_reservation("bot-1", "polymarket", "token-1", 50.0, expires)
        assert rid

        reserved = tmp_db.get_reserved_amount("polymarket", "wallet-1")
        assert reserved == 50.0

        tmp_db.release_reservation(rid)
        reserved = tmp_db.get_reserved_amount("polymarket", "wallet-1")
        assert reserved == 0.0

    def test_reservation_executed(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        rid = tmp_db.create_reservation("bot-1", "polymarket", "token-1", 50.0, expires)

        tmp_db.mark_reservation_executed(rid, 48.0)

        # Executed reservations should not count as pending
        reserved = tmp_db.get_reserved_amount("polymarket", "wallet-1")
        assert reserved == 0.0

    def test_multiple_reservations(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)

        tmp_db.create_reservation("bot-1", "polymarket", "t1", 50.0, expires)
        tmp_db.create_reservation("bot-1", "polymarket", "t2", 30.0, expires)

        reserved = tmp_db.get_reserved_amount("polymarket", "wallet-1")
        assert reserved == 80.0

    def test_cleanup_expired_reservations(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        # Create an already-expired reservation
        expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        tmp_db.create_reservation("bot-1", "polymarket", "t1", 50.0, expired_time)

        cleaned = tmp_db.cleanup_expired_reservations()
        assert cleaned == 1

    def test_cleanup_expired_reservations_scoped_by_agent(self, tmp_db):
        """cleanup_expired_reservations with agent_id only expires that agent's reservations."""
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.register_agent("bot-2", "directional", "polymarket")

        expired_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        tmp_db.create_reservation("bot-1", "polymarket", "t1", 50.0, expired_time)
        tmp_db.create_reservation("bot-2", "polymarket", "t2", 30.0, expired_time)

        # Only clean bot-1's reservations
        cleaned = tmp_db.cleanup_expired_reservations(agent_id="bot-1")
        assert cleaned == 1

        # bot-2's reservation is still pending (expired but not cleaned yet)
        cleaned_all = tmp_db.cleanup_expired_reservations()
        assert cleaned_all == 1  # bot-2's reservation now cleaned

    def test_position_lifecycle(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")

        pid = tmp_db.create_position("bot-1", "polymarket", "token-1", "BUY", 100.0, 0.50)
        assert pid > 0

        positions = tmp_db.get_agent_positions("bot-1", "open")
        assert len(positions) == 1
        assert positions[0]["instrument_id"] == "token-1"

        tmp_db.close_position(pid, 0.60, "take_profit")
        positions = tmp_db.get_agent_positions("bot-1", "open")
        assert len(positions) == 0

    def test_position_pnl(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.close_position(pid, 0.60, "take_profit")

        closed = tmp_db.get_agent_positions("bot-1", "closed")
        assert len(closed) == 1
        assert closed[0]["pnl"] == pytest.approx(10.0)  # (0.60 - 0.50) * 100

    def test_position_pnl_sell_side(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "SELL", 100.0, 0.50)
        tmp_db.close_position(pid, 0.40, "take_profit")

        closed = tmp_db.get_agent_positions("bot-1", "closed")
        assert closed[0]["pnl"] == pytest.approx(10.0)  # (0.50 - 0.40) * 100

    def test_position_pnl_loss(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.close_position(pid, 0.40, "stop_loss")

        closed = tmp_db.get_agent_positions("bot-1", "closed")
        assert closed[0]["pnl"] == pytest.approx(-10.0)

    def test_exposure(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)

        exposure = tmp_db.get_total_exposure("polymarket", "wallet-1")
        assert exposure == 50.0  # 100 * 0.50

    def test_agent_exposure(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.create_position("bot-1", "polymarket", "t2", "BUY", 200.0, 0.30)

        exposure = tmp_db.get_agent_exposure("bot-1")
        assert exposure == pytest.approx(110.0)  # 100*0.50 + 200*0.30

    def test_instrument_exposure(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)

        exposure = tmp_db.get_instrument_exposure("polymarket", "t1")
        assert exposure == 50.0

    def test_instrument_exposure_zero(self, tmp_db):
        exposure = tmp_db.get_instrument_exposure("polymarket", "nonexistent")
        assert exposure == 0.0

    def test_update_position_price(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.update_position_price(pid, 0.60)

        positions = tmp_db.get_agent_positions("bot-1", "open")
        assert positions[0]["current_price"] == 0.60

    def test_get_exchange_positions(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.register_agent("bot-2", "directional", "polymarket")
        tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.create_position("bot-2", "polymarket", "t2", "BUY", 50.0, 0.60)

        positions = tmp_db.get_exchange_positions("polymarket", "open")
        assert len(positions) == 2

    def test_execution_log(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.log_execution("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50, "order-1")

        execs = tmp_db.get_executions(agent_id="bot-1")
        assert len(execs) == 1
        assert execs[0]["price"] == 0.50

    def test_execution_log_with_fees(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.log_execution("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50, "order-1", fees=0.25)

        execs = tmp_db.get_executions(agent_id="bot-1")
        assert execs[0]["fees"] == 0.25

    def test_execution_filter_by_exchange(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.log_execution("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50, "order-1")
        tmp_db.log_execution("bot-1", "kalshi", "t2", "BUY", 50.0, 0.60, "order-2")

        poly_execs = tmp_db.get_executions(exchange="polymarket")
        assert len(poly_execs) == 1
        assert poly_execs[0]["exchange"] == "polymarket"

    def test_execution_filter_by_since(self, tmp_db):
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        tmp_db.log_execution("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50, "order-1")

        # Query with future time should return nothing
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        execs = tmp_db.get_executions(since=future)
        assert len(execs) == 0

    def test_stale_agent_cleanup(self, tmp_db):
        tmp_db.register_agent("stale-bot", "directional", "polymarket")
        # Manually backdate heartbeat
        tmp_db._get_conn().execute(
            "UPDATE agents SET last_heartbeat=? WHERE agent_id=?",
            ((datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(), "stale-bot")
        )
        tmp_db._get_conn().commit()

        cleaned = tmp_db.cleanup_stale_agents(60)  # 60s threshold
        assert cleaned == 1

    def test_stale_agent_cleanup_no_stale(self, tmp_db):
        tmp_db.register_agent("fresh-bot", "directional", "polymarket")
        cleaned = tmp_db.cleanup_stale_agents(60)
        assert cleaned == 0

    def test_close_and_reopen(self, tmp_db):
        """Closing storage should not lose data on reconnect."""
        tmp_db.update_balance("polymarket", "w1", 500.0)
        db_path = tmp_db.db_path
        tmp_db.close()

        from omnitrade.storage.sqlite import SQLiteStorage
        reopened = SQLiteStorage(db_path)
        reopened.initialize()
        assert reopened.get_balance("polymarket", "w1") == 500.0
        reopened.close()

    def test_exit_state_roundtrip(self, tmp_db):
        """Exit state fields should survive write and read."""
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)

        tmp_db.update_position_exit_state(
            pid,
            current_price=0.55,
            peak_price=0.58,
            trough_price=0.48,
            trailing_stop_activated=True,
            trailing_stop_level=0.565,
        )

        positions = tmp_db.get_agent_positions("bot-1", "open")
        pos = positions[0]
        assert pos["current_price"] == pytest.approx(0.55)
        assert pos["peak_price"] == pytest.approx(0.58)
        assert pos["trough_price"] == pytest.approx(0.48)
        assert pos["trailing_stop_activated"] == 1
        assert pos["trailing_stop_level"] == pytest.approx(0.565)

    def test_exit_state_persists_across_close_reopen(self, tmp_db):
        """Exit state should survive DB close and reopen."""
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)
        tmp_db.update_position_exit_state(
            pid, current_price=0.55, peak_price=0.58,
            trough_price=0.48, trailing_stop_activated=True,
            trailing_stop_level=0.565,
        )
        db_path = tmp_db.db_path
        tmp_db.close()

        from omnitrade.storage.sqlite import SQLiteStorage
        reopened = SQLiteStorage(db_path)
        reopened.initialize()
        positions = reopened.get_agent_positions("bot-1", "open")
        pos = positions[0]
        assert pos["peak_price"] == pytest.approx(0.58)
        assert pos["trailing_stop_activated"] == 1
        reopened.close()

    def test_exit_state_null_for_legacy_positions(self, tmp_db):
        """Positions created before migration should have NULL exit state fields."""
        tmp_db.register_agent("bot-1", "directional", "polymarket")
        pid = tmp_db.create_position("bot-1", "polymarket", "t1", "BUY", 100.0, 0.50)

        positions = tmp_db.get_agent_positions("bot-1", "open")
        pos = positions[0]
        # New columns default to NULL (not set) for positions created without exit state
        assert pos["peak_price"] is None
        assert pos["trough_price"] is None
        assert pos["trailing_stop_activated"] == 0
        assert pos["trailing_stop_level"] == pytest.approx(0.0)

    def test_migration_duplicate_column_ignored(self, tmp_db):
        """Re-running initialize should not fail when columns already exist."""
        # initialize() was already called by the fixture; calling again should
        # silently skip the duplicate column additions.
        tmp_db.initialize()
        # Verify DB still works after double-initialize
        tmp_db.register_agent("bot-dup", "directional", "polymarket")
        pid = tmp_db.create_position("bot-dup", "polymarket", "t1", "BUY", 10.0, 0.50)
        tmp_db.update_position_exit_state(
            pid, current_price=0.55, peak_price=0.58,
            trough_price=0.48, trailing_stop_activated=False,
            trailing_stop_level=0.0,
        )
        positions = tmp_db.get_agent_positions("bot-dup", "open")
        assert positions[0]["peak_price"] == pytest.approx(0.58)

    def test_migration_reraises_non_duplicate_errors(self):
        """Migration should re-raise OperationalErrors that are not duplicate column."""
        import sqlite3
        import tempfile
        import os
        from unittest.mock import MagicMock
        from omnitrade.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            storage = SQLiteStorage(db_path)
            # Force connection creation and run schema setup first
            conn = storage._get_conn()
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS positions (
                    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL, exchange TEXT NOT NULL,
                    instrument_id TEXT NOT NULL, side TEXT NOT NULL,
                    size REAL NOT NULL, entry_price REAL NOT NULL,
                    current_price REAL, status TEXT NOT NULL DEFAULT 'open',
                    opened_at TEXT NOT NULL, closed_at TEXT,
                    exit_price REAL, exit_reason TEXT, pnl REAL
                );
            ''')
            conn.commit()

            # Wrap the real connection to intercept ALTER TABLE calls
            real_conn = storage._conn
            wrapper = MagicMock(wraps=real_conn)
            original_execute = real_conn.execute

            def intercept_execute(sql, *args, **kwargs):
                if isinstance(sql, str) and "ALTER TABLE" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return original_execute(sql, *args, **kwargs)

            wrapper.execute = intercept_execute
            storage._conn = wrapper

            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                storage.initialize()

            # Restore real connection for cleanup
            storage._conn = real_conn
            storage.close()
        finally:
            os.unlink(db_path)
