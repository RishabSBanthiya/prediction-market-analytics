"""
SQLite storage for orderbook history.

Stores continuous orderbook snapshots for all active markets,
designed for use in backtesting with realistic liquidity data.

Uses a separate database from the main risk_state.db to avoid bloat.
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Generator, Dict, Any, Tuple

from ..core.models import OrderbookSnapshot

logger = logging.getLogger(__name__)


@dataclass
class StoredSnapshot:
    """Orderbook snapshot with market context."""
    id: int
    market_id: str
    token_id: str
    outcome: Optional[str]
    timestamp: datetime
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_size: float
    ask_size: float
    spread_pct: Optional[float]
    midpoint: Optional[float]
    bid_depth: List[Tuple[float, float]]
    ask_depth: List[Tuple[float, float]]
    total_bid_liquidity: float
    total_ask_liquidity: float


@dataclass
class MarketMetadata:
    """Metadata about a tracked market."""
    market_id: str
    question: str
    end_date: Optional[datetime]
    first_seen: datetime
    last_seen: datetime
    token_yes_id: Optional[str]
    token_no_id: Optional[str]


@dataclass
class RecordingGap:
    """A gap in orderbook recording that needs backfilling."""
    id: int
    token_id: str
    gap_start: datetime
    gap_end: Optional[datetime]
    gap_reason: str
    retry_count: int
    last_retry: Optional[datetime]
    resolved_at: Optional[datetime]
    created_at: datetime


@dataclass
class HighVolumeMarket:
    """Cached high-volume market for filtering."""
    token_id: str
    market_id: str
    question: str
    volume_24h: float
    liquidity_usd: float
    last_checked: datetime
    is_active: bool


class OrderbookStorage:
    """
    SQLite storage backend for orderbook snapshots.

    Uses WAL mode for better concurrent read/write performance.
    """

    DEFAULT_DB_PATH = "data/orderbook_history.db"

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize orderbook storage.

        Args:
            db_path: Path to SQLite database. Defaults to data/orderbook_history.db
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a new database connection with WAL mode."""
        conn = sqlite3.connect(self.db_path, isolation_level='IMMEDIATE')
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for atomic transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._transaction() as conn:
            # Main snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT NOT NULL,
                    token_id TEXT NOT NULL,
                    outcome TEXT,
                    timestamp TEXT NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    bid_size REAL DEFAULT 0,
                    ask_size REAL DEFAULT 0,
                    spread_pct REAL,
                    midpoint REAL,
                    bid_depth TEXT NOT NULL,
                    ask_depth TEXT NOT NULL,
                    total_bid_liquidity REAL DEFAULT 0,
                    total_ask_liquidity REAL DEFAULT 0,
                    UNIQUE(token_id, timestamp)
                )
            """)

            # Markets metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS markets_metadata (
                    market_id TEXT PRIMARY KEY,
                    question TEXT,
                    end_date TEXT,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    token_yes_id TEXT,
                    token_no_id TEXT
                )
            """)

            # Recording stats table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recording_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    markets_count INTEGER,
                    tokens_count INTEGER,
                    snapshots_saved INTEGER,
                    errors_count INTEGER,
                    cycle_duration_ms INTEGER
                )
            """)

            # Recording gaps table for tracking connection breaks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recording_gaps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT NOT NULL,
                    gap_start TEXT NOT NULL,
                    gap_end TEXT,
                    gap_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    last_retry TEXT,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # High volume markets cache for filtering
            conn.execute("""
                CREATE TABLE IF NOT EXISTS high_volume_markets (
                    token_id TEXT PRIMARY KEY,
                    market_id TEXT,
                    question TEXT,
                    volume_24h REAL,
                    liquidity_usd REAL,
                    last_checked TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1
                )
            """)

            # Create indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_token_time
                ON orderbook_snapshots(token_id, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_market_time
                ON orderbook_snapshots(market_id, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
                ON orderbook_snapshots(timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_end_date
                ON markets_metadata(end_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gaps_token
                ON recording_gaps(token_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gaps_unresolved
                ON recording_gaps(resolved_at) WHERE resolved_at IS NULL
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gaps_start
                ON recording_gaps(gap_start)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hvm_volume
                ON high_volume_markets(volume_24h DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hvm_active
                ON high_volume_markets(is_active)
            """)

            logger.info(f"Orderbook storage initialized at {self.db_path}")

    # ==================== WRITE OPERATIONS ====================

    def save_snapshot(
        self,
        snapshot: OrderbookSnapshot,
        market_id: str,
        outcome: Optional[str] = None
    ) -> int:
        """
        Save a single orderbook snapshot.

        Args:
            snapshot: OrderbookSnapshot from API
            market_id: Market condition_id
            outcome: Token outcome (Yes/No)

        Returns:
            Row ID of inserted snapshot
        """
        total_bid = sum(size for _, size in snapshot.bid_depth)
        total_ask = sum(size for _, size in snapshot.ask_depth)

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO orderbook_snapshots (
                    market_id, token_id, outcome, timestamp,
                    best_bid, best_ask, bid_size, ask_size,
                    spread_pct, midpoint, bid_depth, ask_depth,
                    total_bid_liquidity, total_ask_liquidity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_id,
                    snapshot.token_id,
                    outcome,
                    snapshot.timestamp.isoformat(),
                    snapshot.best_bid,
                    snapshot.best_ask,
                    snapshot.bid_size,
                    snapshot.ask_size,
                    snapshot.spread_pct,
                    snapshot.midpoint,
                    json.dumps(snapshot.bid_depth),
                    json.dumps(snapshot.ask_depth),
                    total_bid,
                    total_ask,
                )
            )
            return cursor.lastrowid

    def save_snapshots_batch(
        self,
        snapshots: List[Tuple[OrderbookSnapshot, str, Optional[str]]]
    ) -> int:
        """
        Save multiple snapshots in a single transaction for efficiency.

        Args:
            snapshots: List of (OrderbookSnapshot, market_id, outcome) tuples

        Returns:
            Number of snapshots saved
        """
        if not snapshots:
            return 0

        rows = []
        for snapshot, market_id, outcome in snapshots:
            total_bid = sum(size for _, size in snapshot.bid_depth)
            total_ask = sum(size for _, size in snapshot.ask_depth)
            rows.append((
                market_id,
                snapshot.token_id,
                outcome,
                snapshot.timestamp.isoformat(),
                snapshot.best_bid,
                snapshot.best_ask,
                snapshot.bid_size,
                snapshot.ask_size,
                snapshot.spread_pct,
                snapshot.midpoint,
                json.dumps(snapshot.bid_depth),
                json.dumps(snapshot.ask_depth),
                total_bid,
                total_ask,
            ))

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO orderbook_snapshots (
                    market_id, token_id, outcome, timestamp,
                    best_bid, best_ask, bid_size, ask_size,
                    spread_pct, midpoint, bid_depth, ask_depth,
                    total_bid_liquidity, total_ask_liquidity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows
            )
            return len(rows)

    def update_market_metadata(
        self,
        market_id: str,
        question: str,
        end_date: Optional[datetime],
        token_yes_id: Optional[str],
        token_no_id: Optional[str]
    ) -> None:
        """Update or insert market metadata."""
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO markets_metadata (
                    market_id, question, end_date, first_seen, last_seen,
                    token_yes_id, token_no_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(market_id) DO UPDATE SET
                    question = excluded.question,
                    end_date = excluded.end_date,
                    last_seen = excluded.last_seen,
                    token_yes_id = COALESCE(excluded.token_yes_id, markets_metadata.token_yes_id),
                    token_no_id = COALESCE(excluded.token_no_id, markets_metadata.token_no_id)
                """,
                (
                    market_id,
                    question,
                    end_date.isoformat() if end_date else None,
                    now,
                    now,
                    token_yes_id,
                    token_no_id,
                )
            )

    def save_recording_stats(
        self,
        markets_count: int,
        tokens_count: int,
        snapshots_saved: int,
        errors_count: int,
        cycle_duration_ms: int
    ) -> None:
        """Save stats from a recording cycle."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO recording_stats (
                    timestamp, markets_count, tokens_count,
                    snapshots_saved, errors_count, cycle_duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    markets_count,
                    tokens_count,
                    snapshots_saved,
                    errors_count,
                    cycle_duration_ms,
                )
            )

    # ==================== READ OPERATIONS ====================

    def get_snapshots(
        self,
        token_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[StoredSnapshot]:
        """
        Get orderbook snapshots for a token within a time range.

        Args:
            token_id: Token to query
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            limit: Max number of results

        Returns:
            List of StoredSnapshot ordered by timestamp ascending
        """
        query = "SELECT * FROM orderbook_snapshots WHERE token_id = ?"
        params: List[Any] = [token_id]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        query += " ORDER BY timestamp ASC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_snapshot(row) for row in rows]

    def get_snapshot_at(
        self,
        token_id: str,
        timestamp: datetime,
        tolerance_seconds: int = 60
    ) -> Optional[StoredSnapshot]:
        """
        Get the orderbook snapshot closest to a specific timestamp.

        Args:
            token_id: Token to query
            timestamp: Target timestamp
            tolerance_seconds: Max seconds difference allowed

        Returns:
            Closest snapshot within tolerance, or None
        """
        ts_iso = timestamp.isoformat()
        min_ts = (timestamp - timedelta(seconds=tolerance_seconds)).isoformat()
        max_ts = (timestamp + timedelta(seconds=tolerance_seconds)).isoformat()

        with self._transaction() as conn:
            # Find closest snapshot within tolerance
            row = conn.execute(
                """
                SELECT *, ABS(julianday(timestamp) - julianday(?)) as diff
                FROM orderbook_snapshots
                WHERE token_id = ?
                  AND timestamp BETWEEN ? AND ?
                ORDER BY diff ASC
                LIMIT 1
                """,
                (ts_iso, token_id, min_ts, max_ts)
            ).fetchone()

            if row:
                return self._row_to_snapshot(row)
            return None

    def get_spread_history(
        self,
        token_id: str,
        start: datetime,
        end: datetime
    ) -> List[Tuple[datetime, Optional[float]]]:
        """
        Get spread percentage time series for a token.

        Args:
            token_id: Token to query
            start: Start of time range
            end: End of time range

        Returns:
            List of (timestamp, spread_pct) tuples
        """
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, spread_pct
                FROM orderbook_snapshots
                WHERE token_id = ?
                  AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
                """,
                (token_id, start.isoformat(), end.isoformat())
            ).fetchall()

            return [
                (datetime.fromisoformat(row['timestamp']), row['spread_pct'])
                for row in rows
            ]

    def get_liquidity_at_distance(
        self,
        token_id: str,
        timestamp: datetime,
        distance_pct: float,
        tolerance_seconds: int = 60
    ) -> Optional[Tuple[float, float]]:
        """
        Get total liquidity within a percentage distance from midpoint.

        Args:
            token_id: Token to query
            timestamp: Target timestamp
            distance_pct: Distance from midpoint (e.g., 0.02 for 2%)
            tolerance_seconds: Max timestamp difference allowed

        Returns:
            Tuple of (bid_liquidity, ask_liquidity) within distance, or None
        """
        snapshot = self.get_snapshot_at(token_id, timestamp, tolerance_seconds)
        if not snapshot or not snapshot.midpoint:
            return None

        mid = snapshot.midpoint
        lower_bound = mid * (1 - distance_pct)
        upper_bound = mid * (1 + distance_pct)

        bid_liq = sum(
            size for price, size in snapshot.bid_depth
            if price >= lower_bound
        )
        ask_liq = sum(
            size for price, size in snapshot.ask_depth
            if price <= upper_bound
        )

        return (bid_liq, ask_liq)

    def get_available_markets(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[MarketMetadata]:
        """
        Get list of markets with recorded data in time range.

        Args:
            start: Start of time range (uses first_seen if None)
            end: End of time range (uses last_seen if None)

        Returns:
            List of MarketMetadata objects
        """
        query = "SELECT * FROM markets_metadata WHERE 1=1"
        params: List[Any] = []

        if start:
            query += " AND last_seen >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND first_seen <= ?"
            params.append(end.isoformat())

        query += " ORDER BY last_seen DESC"

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]

    def get_snapshot_count(
        self,
        token_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> int:
        """Get count of snapshots matching criteria."""
        query = "SELECT COUNT(*) as cnt FROM orderbook_snapshots WHERE 1=1"
        params: List[Any] = []

        if token_id:
            query += " AND token_id = ?"
            params.append(token_id)
        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())

        with self._transaction() as conn:
            row = conn.execute(query, params).fetchone()
            return row['cnt'] if row else 0

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._transaction() as conn:
            snapshot_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM orderbook_snapshots"
            ).fetchone()['cnt']

            market_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM markets_metadata"
            ).fetchone()['cnt']

            # Get time range
            time_range = conn.execute(
                """
                SELECT MIN(timestamp) as first, MAX(timestamp) as last
                FROM orderbook_snapshots
                """
            ).fetchone()

            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            return {
                'snapshot_count': snapshot_count,
                'market_count': market_count,
                'first_snapshot': time_range['first'],
                'last_snapshot': time_range['last'],
                'db_size_bytes': db_size,
                'db_size_mb': round(db_size / (1024 * 1024), 2),
            }

    # ==================== MAINTENANCE ====================

    def cleanup_old_data(self, days: int) -> int:
        """
        Delete snapshots older than specified days.

        Args:
            days: Delete data older than this many days

        Returns:
            Number of rows deleted
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM orderbook_snapshots WHERE timestamp < ?",
                (cutoff,)
            )
            deleted = cursor.rowcount

            # Vacuum to reclaim space
            conn.execute("VACUUM")

            logger.info(f"Cleaned up {deleted} snapshots older than {days} days")
            return deleted

    def vacuum(self) -> None:
        """Reclaim unused space in database."""
        with self._transaction() as conn:
            conn.execute("VACUUM")

    # ==================== GAP TRACKING ====================

    def record_gap_start(
        self,
        token_id: str,
        gap_start: datetime,
        reason: str
    ) -> int:
        """
        Record the start of a data gap.

        Args:
            token_id: Token that has a gap
            gap_start: When the gap started
            reason: Why the gap occurred (e.g., "websocket_disconnect", "stale_connection")

        Returns:
            Row ID of the created gap record
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recording_gaps (
                    token_id, gap_start, gap_reason, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (token_id, gap_start.isoformat(), reason, now)
            )
            return cursor.lastrowid

    def record_gap_start_batch(
        self,
        token_ids: List[str],
        gap_start: datetime,
        reason: str
    ) -> int:
        """
        Record gap start for multiple tokens at once.

        Args:
            token_ids: List of tokens with gaps
            gap_start: When the gap started
            reason: Why the gap occurred

        Returns:
            Number of gap records created
        """
        if not token_ids:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        gap_start_iso = gap_start.isoformat()

        rows = [(tid, gap_start_iso, reason, now) for tid in token_ids]

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO recording_gaps (
                    token_id, gap_start, gap_reason, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                rows
            )
            return len(rows)

    def record_gap_end(
        self,
        token_id: str,
        gap_end: datetime
    ) -> int:
        """
        Close all open gaps for a token.

        Args:
            token_id: Token whose gaps to close
            gap_end: When the gap ended

        Returns:
            Number of gaps closed
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE recording_gaps
                SET gap_end = ?
                WHERE token_id = ? AND gap_end IS NULL
                """,
                (gap_end.isoformat(), token_id)
            )
            return cursor.rowcount

    def record_gap_end_batch(
        self,
        token_ids: List[str],
        gap_end: datetime
    ) -> int:
        """
        Close all open gaps for multiple tokens.

        Args:
            token_ids: List of tokens whose gaps to close
            gap_end: When the gaps ended

        Returns:
            Number of gaps closed
        """
        if not token_ids:
            return 0

        gap_end_iso = gap_end.isoformat()
        placeholders = ",".join("?" * len(token_ids))

        with self._transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE recording_gaps
                SET gap_end = ?
                WHERE token_id IN ({placeholders}) AND gap_end IS NULL
                """,
                [gap_end_iso] + list(token_ids)
            )
            return cursor.rowcount

    def get_unresolved_gaps(
        self,
        token_id: Optional[str] = None,
        max_retry_count: int = 5,
        limit: int = 100
    ) -> List[RecordingGap]:
        """
        Get gaps that need backfilling.

        Args:
            token_id: Optional filter by specific token
            max_retry_count: Exclude gaps with more retries than this
            limit: Max gaps to return

        Returns:
            List of RecordingGap objects, ordered oldest first
        """
        query = """
            SELECT * FROM recording_gaps
            WHERE resolved_at IS NULL
              AND gap_end IS NOT NULL
              AND retry_count < ?
        """
        params: List[Any] = [max_retry_count]

        if token_id:
            query += " AND token_id = ?"
            params.append(token_id)

        query += " ORDER BY gap_start ASC LIMIT ?"
        params.append(limit)

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_gap(row) for row in rows]

    def get_open_gaps(self, token_id: Optional[str] = None) -> List[RecordingGap]:
        """
        Get gaps that are still open (no end time).

        Args:
            token_id: Optional filter by specific token

        Returns:
            List of open RecordingGap objects
        """
        query = "SELECT * FROM recording_gaps WHERE gap_end IS NULL"
        params: List[Any] = []

        if token_id:
            query += " AND token_id = ?"
            params.append(token_id)

        with self._transaction() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_gap(row) for row in rows]

    def mark_gap_resolved(self, gap_id: int) -> bool:
        """
        Mark a gap as successfully backfilled.

        Args:
            gap_id: ID of the gap to mark resolved

        Returns:
            True if gap was updated, False if not found
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE recording_gaps
                SET resolved_at = ?
                WHERE id = ? AND resolved_at IS NULL
                """,
                (now, gap_id)
            )
            return cursor.rowcount > 0

    def increment_gap_retry(self, gap_id: int, error: Optional[str] = None) -> bool:
        """
        Increment retry count after failed backfill attempt.

        Args:
            gap_id: ID of the gap
            error: Optional error message to log

        Returns:
            True if gap was updated, False if not found
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE recording_gaps
                SET retry_count = retry_count + 1,
                    last_retry = ?
                WHERE id = ?
                """,
                (now, gap_id)
            )
            return cursor.rowcount > 0

    def get_gap_stats(self) -> Dict[str, Any]:
        """Get statistics about recording gaps."""
        with self._transaction() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM recording_gaps"
            ).fetchone()['cnt']

            open_gaps = conn.execute(
                "SELECT COUNT(*) as cnt FROM recording_gaps WHERE gap_end IS NULL"
            ).fetchone()['cnt']

            unresolved = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM recording_gaps
                WHERE resolved_at IS NULL AND gap_end IS NOT NULL
                """
            ).fetchone()['cnt']

            resolved = conn.execute(
                "SELECT COUNT(*) as cnt FROM recording_gaps WHERE resolved_at IS NOT NULL"
            ).fetchone()['cnt']

            max_retries_exceeded = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM recording_gaps
                WHERE retry_count >= 5 AND resolved_at IS NULL
                """
            ).fetchone()['cnt']

            return {
                'total_gaps': total,
                'open_gaps': open_gaps,
                'unresolved_gaps': unresolved,
                'resolved_gaps': resolved,
                'max_retries_exceeded': max_retries_exceeded,
            }

    # ==================== VOLUME FILTERING ====================

    def update_market_volume(
        self,
        token_id: str,
        market_id: str,
        question: str,
        volume_24h: float,
        liquidity_usd: float,
        is_active: bool = True
    ) -> None:
        """
        Update or insert cached volume data for a market token.

        Args:
            token_id: Token identifier
            market_id: Market condition_id
            question: Market question
            volume_24h: 24-hour trading volume in USD
            liquidity_usd: Total orderbook liquidity in USD
            is_active: Whether market is currently active
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO high_volume_markets (
                    token_id, market_id, question, volume_24h,
                    liquidity_usd, last_checked, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET
                    market_id = excluded.market_id,
                    question = excluded.question,
                    volume_24h = excluded.volume_24h,
                    liquidity_usd = excluded.liquidity_usd,
                    last_checked = excluded.last_checked,
                    is_active = excluded.is_active
                """,
                (
                    token_id,
                    market_id,
                    question,
                    volume_24h,
                    liquidity_usd,
                    now,
                    1 if is_active else 0,
                )
            )

    def update_market_volumes_batch(
        self,
        markets: List[Dict[str, Any]]
    ) -> int:
        """
        Batch update volume data for multiple market tokens.

        Args:
            markets: List of dicts with keys: token_id, market_id, question,
                     volume_24h, liquidity_usd, is_active

        Returns:
            Number of markets updated
        """
        if not markets:
            return 0

        now = datetime.now(timezone.utc).isoformat()

        rows = [
            (
                m['token_id'],
                m['market_id'],
                m.get('question', ''),
                m.get('volume_24h', 0),
                m.get('liquidity_usd', 0),
                now,
                1 if m.get('is_active', True) else 0,
            )
            for m in markets
        ]

        with self._transaction() as conn:
            conn.executemany(
                """
                INSERT INTO high_volume_markets (
                    token_id, market_id, question, volume_24h,
                    liquidity_usd, last_checked, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(token_id) DO UPDATE SET
                    market_id = excluded.market_id,
                    question = excluded.question,
                    volume_24h = excluded.volume_24h,
                    liquidity_usd = excluded.liquidity_usd,
                    last_checked = excluded.last_checked,
                    is_active = excluded.is_active
                """,
                rows
            )
            return len(rows)

    def get_high_volume_tokens(
        self,
        min_volume_24h: float = 10000.0,
        min_liquidity: float = 5000.0,
        limit: int = 500
    ) -> List[str]:
        """
        Get token IDs meeting volume thresholds.

        Args:
            min_volume_24h: Minimum 24h volume in USD
            min_liquidity: Minimum orderbook liquidity in USD
            limit: Max tokens to return

        Returns:
            List of token IDs sorted by volume descending
        """
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT token_id FROM high_volume_markets
                WHERE is_active = 1
                  AND volume_24h >= ?
                  AND liquidity_usd >= ?
                ORDER BY volume_24h DESC
                LIMIT ?
                """,
                (min_volume_24h, min_liquidity, limit)
            ).fetchall()

            return [row['token_id'] for row in rows]

    def get_high_volume_markets(
        self,
        min_volume_24h: float = 10000.0,
        min_liquidity: float = 5000.0,
        limit: int = 500
    ) -> List[HighVolumeMarket]:
        """
        Get high-volume market details.

        Args:
            min_volume_24h: Minimum 24h volume in USD
            min_liquidity: Minimum orderbook liquidity in USD
            limit: Max markets to return

        Returns:
            List of HighVolumeMarket objects sorted by volume descending
        """
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM high_volume_markets
                WHERE is_active = 1
                  AND volume_24h >= ?
                  AND liquidity_usd >= ?
                ORDER BY volume_24h DESC
                LIMIT ?
                """,
                (min_volume_24h, min_liquidity, limit)
            ).fetchall()

            return [self._row_to_high_volume_market(row) for row in rows]

    def is_high_volume_market(
        self,
        token_id: str,
        min_volume_24h: float = 10000.0
    ) -> bool:
        """
        Check if a single token meets volume threshold.

        Args:
            token_id: Token to check
            min_volume_24h: Minimum 24h volume in USD

        Returns:
            True if token meets threshold, False otherwise
        """
        with self._transaction() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM high_volume_markets
                WHERE token_id = ?
                  AND is_active = 1
                  AND volume_24h >= ?
                """,
                (token_id, min_volume_24h)
            ).fetchone()

            return row is not None

    def mark_markets_inactive(self, token_ids: List[str]) -> int:
        """
        Mark markets as inactive (closed/resolved).

        Args:
            token_ids: List of tokens to mark inactive

        Returns:
            Number of markets updated
        """
        if not token_ids:
            return 0

        placeholders = ",".join("?" * len(token_ids))

        with self._transaction() as conn:
            cursor = conn.execute(
                f"""
                UPDATE high_volume_markets
                SET is_active = 0
                WHERE token_id IN ({placeholders})
                """,
                token_ids
            )
            return cursor.rowcount

    def get_volume_stats(self) -> Dict[str, Any]:
        """Get statistics about cached volume data."""
        with self._transaction() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as cnt FROM high_volume_markets"
            ).fetchone()['cnt']

            active = conn.execute(
                "SELECT COUNT(*) as cnt FROM high_volume_markets WHERE is_active = 1"
            ).fetchone()['cnt']

            high_vol = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM high_volume_markets
                WHERE is_active = 1 AND volume_24h >= 10000
                """
            ).fetchone()['cnt']

            total_volume = conn.execute(
                "SELECT SUM(volume_24h) as total FROM high_volume_markets WHERE is_active = 1"
            ).fetchone()['total'] or 0

            return {
                'total_markets': total,
                'active_markets': active,
                'high_volume_markets': high_vol,
                'total_volume_24h': total_volume,
            }

    # ==================== HELPERS ====================

    def _row_to_snapshot(self, row: sqlite3.Row) -> StoredSnapshot:
        """Convert database row to StoredSnapshot."""
        return StoredSnapshot(
            id=row['id'],
            market_id=row['market_id'],
            token_id=row['token_id'],
            outcome=row['outcome'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            best_bid=row['best_bid'],
            best_ask=row['best_ask'],
            bid_size=row['bid_size'] or 0,
            ask_size=row['ask_size'] or 0,
            spread_pct=row['spread_pct'],
            midpoint=row['midpoint'],
            bid_depth=json.loads(row['bid_depth']),
            ask_depth=json.loads(row['ask_depth']),
            total_bid_liquidity=row['total_bid_liquidity'] or 0,
            total_ask_liquidity=row['total_ask_liquidity'] or 0,
        )

    def _row_to_metadata(self, row: sqlite3.Row) -> MarketMetadata:
        """Convert database row to MarketMetadata."""
        return MarketMetadata(
            market_id=row['market_id'],
            question=row['question'],
            end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
            first_seen=datetime.fromisoformat(row['first_seen']),
            last_seen=datetime.fromisoformat(row['last_seen']),
            token_yes_id=row['token_yes_id'],
            token_no_id=row['token_no_id'],
        )

    def _row_to_gap(self, row: sqlite3.Row) -> RecordingGap:
        """Convert database row to RecordingGap."""
        return RecordingGap(
            id=row['id'],
            token_id=row['token_id'],
            gap_start=datetime.fromisoformat(row['gap_start']),
            gap_end=datetime.fromisoformat(row['gap_end']) if row['gap_end'] else None,
            gap_reason=row['gap_reason'] or '',
            retry_count=row['retry_count'] or 0,
            last_retry=datetime.fromisoformat(row['last_retry']) if row['last_retry'] else None,
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            created_at=datetime.fromisoformat(row['created_at']),
        )

    def _row_to_high_volume_market(self, row: sqlite3.Row) -> HighVolumeMarket:
        """Convert database row to HighVolumeMarket."""
        return HighVolumeMarket(
            token_id=row['token_id'],
            market_id=row['market_id'] or '',
            question=row['question'] or '',
            volume_24h=row['volume_24h'] or 0.0,
            liquidity_usd=row['liquidity_usd'] or 0.0,
            last_checked=datetime.fromisoformat(row['last_checked']),
            is_active=bool(row['is_active']),
        )
