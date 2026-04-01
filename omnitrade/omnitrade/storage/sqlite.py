"""
SQLite storage backend with WAL mode.

Production-ready implementation with:
- WAL mode for concurrent read access
- All operations in transactions
- Multi-exchange support via exchange column
"""

import sqlite3
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path

from .base import StorageBackend

logger = logging.getLogger(__name__)


class SQLiteStorage(StorageBackend):
    """SQLite storage with WAL mode for concurrent access."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, timeout=30)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def initialize(self) -> None:
        conn = self._get_conn()
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS balances (
                exchange TEXT NOT NULL,
                account_id TEXT NOT NULL,
                balance REAL NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (exchange, account_id)
            );

            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                exchange TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                started_at TEXT NOT NULL,
                last_heartbeat TEXT NOT NULL,
                stopped_at TEXT
            );

            CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                instrument_id TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                filled_amount REAL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                instrument_id TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                status TEXT NOT NULL DEFAULT 'open',
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                instrument_id TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                price REAL NOT NULL,
                order_id TEXT,
                fees REAL DEFAULT 0,
                executed_at TEXT NOT NULL,
                FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
            );

            CREATE INDEX IF NOT EXISTS idx_positions_agent_status
                ON positions(agent_id, status);
            CREATE INDEX IF NOT EXISTS idx_positions_exchange_status
                ON positions(exchange, status);
            CREATE INDEX IF NOT EXISTS idx_reservations_status
                ON reservations(status);
            CREATE INDEX IF NOT EXISTS idx_reservations_agent
                ON reservations(agent_id, status);
            CREATE INDEX IF NOT EXISTS idx_executions_agent
                ON executions(agent_id, executed_at);
        ''')
        conn.commit()

        # Migration: add exit state columns to positions table
        for col, col_def in [
            ("peak_price", "REAL"),
            ("trough_price", "REAL"),
            ("trailing_stop_activated", "INTEGER DEFAULT 0"),
            ("trailing_stop_level", "REAL DEFAULT 0.0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {col_def}")
                conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    pass  # Column already exists
                else:
                    logger.error(f"Failed to add column {col}: {e}")
                    raise

        logger.info(f"SQLite storage initialized: {self.db_path}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ==================== BALANCE ====================

    def get_balance(self, exchange: str, account_id: str) -> float:
        row = self._get_conn().execute(
            "SELECT balance FROM balances WHERE exchange=? AND account_id=?",
            (exchange, account_id)
        ).fetchone()
        return row["balance"] if row else 0.0

    def update_balance(self, exchange: str, account_id: str, balance: float) -> None:
        self._get_conn().execute(
            """INSERT INTO balances (exchange, account_id, balance, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(exchange, account_id) DO UPDATE SET balance=?, updated_at=?""",
            (exchange, account_id, balance, self._now(), balance, self._now())
        )
        self._get_conn().commit()

    # ==================== AGENTS ====================

    def register_agent(self, agent_id: str, agent_type: str, exchange: str) -> bool:
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT status FROM agents WHERE agent_id=?", (agent_id,)
        ).fetchone()
        if existing and existing["status"] == "active":
            return False
        now = self._now()
        conn.execute(
            """INSERT INTO agents (agent_id, agent_type, exchange, status, started_at, last_heartbeat)
               VALUES (?, ?, ?, 'active', ?, ?)
               ON CONFLICT(agent_id) DO UPDATE SET
                   status='active', started_at=?, last_heartbeat=?, stopped_at=NULL""",
            (agent_id, agent_type, exchange, now, now, now, now)
        )
        conn.commit()
        return True

    def update_heartbeat(self, agent_id: str) -> None:
        self._get_conn().execute(
            "UPDATE agents SET last_heartbeat=? WHERE agent_id=?",
            (self._now(), agent_id)
        )
        self._get_conn().commit()

    def set_agent_status(self, agent_id: str, status: str) -> None:
        now = self._now()
        stopped = now if status in ("stopped", "crashed") else None
        self._get_conn().execute(
            "UPDATE agents SET status=?, stopped_at=? WHERE agent_id=?",
            (status, stopped, agent_id)
        )
        self._get_conn().commit()

    def cleanup_stale_agents(self, stale_threshold_seconds: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_threshold_seconds)).isoformat()
        cursor = self._get_conn().execute(
            """UPDATE agents SET status='crashed', stopped_at=?
               WHERE status='active' AND last_heartbeat < ?""",
            (self._now(), cutoff)
        )
        self._get_conn().commit()
        return cursor.rowcount

    # ==================== RESERVATIONS ====================

    def create_reservation(
        self, agent_id: str, exchange: str, instrument_id: str,
        amount_usd: float, expires_at: datetime,
    ) -> str:
        rid = str(uuid.uuid4())
        self._get_conn().execute(
            """INSERT INTO reservations
               (reservation_id, agent_id, exchange, instrument_id, amount_usd, status, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (rid, agent_id, exchange, instrument_id, amount_usd, self._now(), expires_at.isoformat())
        )
        self._get_conn().commit()
        return rid

    def mark_reservation_executed(self, reservation_id: str, filled_amount: float) -> None:
        self._get_conn().execute(
            "UPDATE reservations SET status='executed', filled_amount=? WHERE reservation_id=?",
            (filled_amount, reservation_id)
        )
        self._get_conn().commit()

    def release_reservation(self, reservation_id: str) -> None:
        self._get_conn().execute(
            "UPDATE reservations SET status='released' WHERE reservation_id=?",
            (reservation_id,)
        )
        self._get_conn().commit()

    def cleanup_expired_reservations(self, agent_id: str | None = None) -> int:
        now = self._now()
        if agent_id is not None:
            cursor = self._get_conn().execute(
                "UPDATE reservations SET status='expired' "
                "WHERE status='pending' AND expires_at < ? AND agent_id = ?",
                (now, agent_id),
            )
        else:
            cursor = self._get_conn().execute(
                "UPDATE reservations SET status='expired' "
                "WHERE status='pending' AND expires_at < ?",
                (now,),
            )
        self._get_conn().commit()
        return cursor.rowcount

    def get_reserved_amount(self, exchange: str, account_id: str) -> float:
        row = self._get_conn().execute(
            """SELECT COALESCE(SUM(amount_usd), 0) as total
               FROM reservations r
               JOIN agents a ON r.agent_id = a.agent_id
               WHERE r.exchange=? AND r.status='pending'""",
            (exchange,)
        ).fetchone()
        return row["total"] if row else 0.0

    # ==================== POSITIONS ====================

    def create_position(
        self, agent_id: str, exchange: str, instrument_id: str,
        side: str, size: float, entry_price: float,
    ) -> int:
        cursor = self._get_conn().execute(
            """INSERT INTO positions
               (agent_id, exchange, instrument_id, side, size, entry_price, current_price, status, opened_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
            (agent_id, exchange, instrument_id, side, size, entry_price, entry_price, self._now())
        )
        self._get_conn().commit()
        return cursor.lastrowid

    def get_agent_positions(self, agent_id: str, status: str = "open") -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM positions WHERE agent_id=? AND status=?",
            (agent_id, status)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_exchange_positions(self, exchange: str, status: str = "open") -> list[dict]:
        rows = self._get_conn().execute(
            "SELECT * FROM positions WHERE exchange=? AND status=?",
            (exchange, status)
        ).fetchall()
        return [dict(r) for r in rows]

    def close_position(self, position_id: int, exit_price: float, exit_reason: str) -> None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT entry_price, size, side FROM positions WHERE position_id=?",
            (position_id,)
        ).fetchone()
        pnl = 0.0
        if row:
            if row["side"] == "BUY":
                pnl = (exit_price - row["entry_price"]) * row["size"]
            else:
                pnl = (row["entry_price"] - exit_price) * row["size"]
        conn.execute(
            """UPDATE positions SET status='closed', closed_at=?, exit_price=?,
               exit_reason=?, pnl=?, current_price=? WHERE position_id=?""",
            (self._now(), exit_price, exit_reason, pnl, exit_price, position_id)
        )
        conn.commit()

    def update_position_price(self, position_id: int, current_price: float) -> None:
        self._get_conn().execute(
            "UPDATE positions SET current_price=? WHERE position_id=?",
            (current_price, position_id)
        )
        self._get_conn().commit()

    def update_position_exit_state(
        self,
        position_id: int,
        current_price: float,
        peak_price: float,
        trough_price: float,
        trailing_stop_activated: bool,
        trailing_stop_level: float,
    ) -> None:
        self._get_conn().execute(
            """UPDATE positions SET current_price=?, peak_price=?, trough_price=?,
               trailing_stop_activated=?, trailing_stop_level=?
               WHERE position_id=?""",
            (current_price, peak_price, trough_price,
             int(trailing_stop_activated), trailing_stop_level, position_id)
        )
        self._get_conn().commit()

    # ==================== EXPOSURE ====================

    def get_total_exposure(self, exchange: str, account_id: str) -> float:
        conn = self._get_conn()
        # Positions exposure
        pos_row = conn.execute(
            """SELECT COALESCE(SUM(size * entry_price), 0) as total
               FROM positions WHERE exchange=? AND status='open'""",
            (exchange,)
        ).fetchone()
        # Reservations exposure
        res_row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) as total
               FROM reservations WHERE exchange=? AND status='pending'""",
            (exchange,)
        ).fetchone()
        return (pos_row["total"] if pos_row else 0.0) + (res_row["total"] if res_row else 0.0)

    def get_agent_exposure(self, agent_id: str) -> float:
        conn = self._get_conn()
        pos_row = conn.execute(
            """SELECT COALESCE(SUM(size * entry_price), 0) as total
               FROM positions WHERE agent_id=? AND status='open'""",
            (agent_id,)
        ).fetchone()
        res_row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) as total
               FROM reservations WHERE agent_id=? AND status='pending'""",
            (agent_id,)
        ).fetchone()
        return (pos_row["total"] if pos_row else 0.0) + (res_row["total"] if res_row else 0.0)

    def get_instrument_exposure(self, exchange: str, instrument_id: str) -> float:
        conn = self._get_conn()
        pos_row = conn.execute(
            """SELECT COALESCE(SUM(size * entry_price), 0) as total
               FROM positions WHERE exchange=? AND instrument_id=? AND status='open'""",
            (exchange, instrument_id)
        ).fetchone()
        res_row = conn.execute(
            """SELECT COALESCE(SUM(amount_usd), 0) as total
               FROM reservations WHERE exchange=? AND instrument_id=? AND status='pending'""",
            (exchange, instrument_id)
        ).fetchone()
        return (pos_row["total"] if pos_row else 0.0) + (res_row["total"] if res_row else 0.0)

    # ==================== EXECUTIONS ====================

    def log_execution(
        self, agent_id: str, exchange: str, instrument_id: str,
        side: str, size: float, price: float, order_id: str, fees: float = 0.0,
    ) -> None:
        self._get_conn().execute(
            """INSERT INTO executions
               (agent_id, exchange, instrument_id, side, size, price, order_id, fees, executed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (agent_id, exchange, instrument_id, side, size, price, order_id, fees, self._now())
        )
        self._get_conn().commit()

    def get_executions(
        self, agent_id: Optional[str] = None, exchange: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        query = "SELECT * FROM executions WHERE 1=1"
        params: list = []
        if agent_id:
            query += " AND agent_id=?"
            params.append(agent_id)
        if exchange:
            query += " AND exchange=?"
            params.append(exchange)
        if since:
            query += " AND executed_at>=?"
            params.append(since.isoformat())
        query += " ORDER BY executed_at DESC"
        rows = self._get_conn().execute(query, params).fetchall()
        return [dict(r) for r in rows]
