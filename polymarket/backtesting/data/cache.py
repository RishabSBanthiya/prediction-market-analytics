"""
SQLite cache for historical backtesting data.

Provides persistent storage for:
- Price history (by token, with timestamp range)
- Trade data (by token and wallet)
- Orderbook snapshots (by token and timestamp)
- Market metadata

This allows fast re-runs of optimizations without re-fetching data.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Default cache location
DEFAULT_CACHE_PATH = Path("data/backtest_cache.db")


@dataclass
class CachedPricePoint:
    """A cached price point."""
    token_id: str
    timestamp: int  # Unix timestamp
    price: float
    interval: str  # '1h', '5m', '1m', 'tick'


@dataclass
class CachedTrade:
    """A cached trade."""
    token_id: str
    market_id: str
    trade_id: str
    timestamp: int
    price: float
    size: float
    side: str  # 'BUY' or 'SELL'
    maker: str
    taker: str
    value_usd: float


@dataclass
class CachedMarket:
    """Cached market metadata."""
    condition_id: str
    question: str
    slug: str
    tokens_json: str  # JSON string of tokens
    start_date: Optional[int]  # Unix timestamp
    end_date: Optional[int]  # Unix timestamp
    resolved: bool
    winning_outcome: Optional[str]
    fetched_at: int  # When this was cached


class BacktestDataCache:
    """
    SQLite-based cache for backtesting data.

    Usage:
        cache = BacktestDataCache()

        # Check if data exists
        if not cache.has_price_history(token_id, interval='5m', min_points=100):
            # Fetch from API and store
            prices = await fetch_prices(token_id)
            cache.store_price_history(token_id, prices, interval='5m')

        # Retrieve cached data
        prices = cache.get_price_history(token_id, interval='5m')
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cache.

        Args:
            db_path: Path to SQLite database. Uses default if not provided.
        """
        self.db_path = db_path or DEFAULT_CACHE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                -- Price history table
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    interval TEXT NOT NULL DEFAULT '1h',
                    UNIQUE(token_id, timestamp, interval)
                );
                CREATE INDEX IF NOT EXISTS idx_price_token_interval
                    ON price_history(token_id, interval);
                CREATE INDEX IF NOT EXISTS idx_price_timestamp
                    ON price_history(timestamp);

                -- Trades table
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT NOT NULL,
                    market_id TEXT,
                    trade_id TEXT UNIQUE,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    maker TEXT,
                    taker TEXT,
                    value_usd REAL
                );
                CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_id);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_maker ON trades(maker);
                CREATE INDEX IF NOT EXISTS idx_trades_taker ON trades(taker);

                -- Markets table
                CREATE TABLE IF NOT EXISTS markets (
                    condition_id TEXT PRIMARY KEY,
                    question TEXT,
                    slug TEXT,
                    tokens_json TEXT,
                    start_date INTEGER,
                    end_date INTEGER,
                    resolved INTEGER DEFAULT 0,
                    winning_outcome TEXT,
                    fetched_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_markets_resolved ON markets(resolved);
                CREATE INDEX IF NOT EXISTS idx_markets_end_date ON markets(end_date);

                -- Orderbook snapshots table
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    bid_depth_json TEXT,
                    ask_depth_json TEXT,
                    spread_pct REAL,
                    total_liquidity_usd REAL,
                    UNIQUE(token_id, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_orderbook_token ON orderbook_snapshots(token_id);

                -- Cache metadata table
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at INTEGER
                );
            """)

    @contextmanager
    def _get_conn(self):
        """Get database connection with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== Price History ====================

    def has_price_history(
        self,
        token_id: str,
        interval: str = '1h',
        min_points: int = 10,
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if sufficient price history exists in cache.

        Args:
            token_id: Token to check
            interval: Price interval ('1h', '5m', '1m', 'tick')
            min_points: Minimum data points required
            max_age_hours: Maximum age of newest data point

        Returns:
            True if sufficient fresh data exists
        """
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as count, MAX(timestamp) as newest
                FROM price_history
                WHERE token_id = ? AND interval = ?
            """, (token_id, interval)).fetchone()

            if row['count'] < min_points:
                return False

            if row['newest']:
                newest_time = datetime.fromtimestamp(row['newest'], tz=timezone.utc)
                age = datetime.now(timezone.utc) - newest_time
                if age > timedelta(hours=max_age_hours):
                    return False

            return True

    def store_price_history(
        self,
        token_id: str,
        prices: List[Dict],
        interval: str = '1h'
    ) -> int:
        """
        Store price history in cache.

        Args:
            token_id: Token ID
            prices: List of {'t': timestamp, 'p': price} dicts
            interval: Price interval

        Returns:
            Number of points stored
        """
        if not prices:
            return 0

        with self._get_conn() as conn:
            inserted = 0
            for p in prices:
                ts = p.get('t') or p.get('timestamp')
                price = p.get('p') or p.get('price')
                if ts and price:
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO price_history
                            (token_id, timestamp, price, interval)
                            VALUES (?, ?, ?, ?)
                        """, (token_id, int(ts), float(price), interval))
                        inserted += 1
                    except (sqlite3.Error, ValueError) as e:
                        logger.debug(f"Failed to insert price point: {e}")

            logger.debug(f"Stored {inserted} price points for {token_id[:16]}...")
            return inserted

    def get_price_history(
        self,
        token_id: str,
        interval: str = '1h',
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 10000
    ) -> List[Dict]:
        """
        Get price history from cache.

        Args:
            token_id: Token ID
            interval: Price interval
            start_ts: Start timestamp (optional)
            end_ts: End timestamp (optional)
            limit: Maximum points to return

        Returns:
            List of {'t': timestamp, 'p': price} dicts
        """
        with self._get_conn() as conn:
            query = """
                SELECT timestamp as t, price as p
                FROM price_history
                WHERE token_id = ? AND interval = ?
            """
            params = [token_id, interval]

            if start_ts:
                query += " AND timestamp >= ?"
                params.append(start_ts)
            if end_ts:
                query += " AND timestamp <= ?"
                params.append(end_ts)

            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [{'t': row['t'], 'p': row['p']} for row in rows]

    def get_all_cached_tokens(self, interval: str = '1h') -> List[str]:
        """Get list of all tokens with cached price history."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT token_id FROM price_history
                WHERE interval = ?
            """, (interval,)).fetchall()
            return [row['token_id'] for row in rows]

    # ==================== Trades ====================

    def has_trades(
        self,
        token_id: str,
        min_trades: int = 10,
        max_age_hours: int = 24
    ) -> bool:
        """Check if sufficient trade data exists."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT COUNT(*) as count, MAX(timestamp) as newest
                FROM trades
                WHERE token_id = ?
            """, (token_id,)).fetchone()

            if row['count'] < min_trades:
                return False

            if row['newest']:
                newest_time = datetime.fromtimestamp(row['newest'], tz=timezone.utc)
                age = datetime.now(timezone.utc) - newest_time
                if age > timedelta(hours=max_age_hours):
                    return False

            return True

    def store_trades(
        self,
        token_id: str,
        trades: List[Dict],
        market_id: Optional[str] = None
    ) -> int:
        """
        Store trade data in cache.

        Args:
            token_id: Token ID
            trades: List of trade dicts with keys:
                    id, timestamp, price, size, side, maker, taker
            market_id: Optional market/condition ID

        Returns:
            Number of trades stored
        """
        if not trades:
            return 0

        with self._get_conn() as conn:
            inserted = 0
            for t in trades:
                try:
                    # Handle various timestamp formats
                    ts = t.get('timestamp')
                    if isinstance(ts, datetime):
                        ts = int(ts.timestamp())
                    elif isinstance(ts, str):
                        ts = int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())
                    else:
                        ts = int(ts) if ts else 0

                    trade_id = t.get('id') or t.get('trade_id') or f"{token_id}_{ts}_{t.get('price', 0)}"

                    conn.execute("""
                        INSERT OR IGNORE INTO trades
                        (token_id, market_id, trade_id, timestamp, price, size, side, maker, taker, value_usd)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        token_id,
                        market_id or t.get('market_id'),
                        trade_id,
                        ts,
                        float(t.get('price', 0)),
                        float(t.get('size', 0)),
                        t.get('side', 'UNKNOWN'),
                        t.get('maker') or t.get('active_wallet'),
                        t.get('taker'),
                        float(t.get('value_usd', 0) or t.get('size', 0) * t.get('price', 0))
                    ))
                    inserted += 1
                except (sqlite3.Error, ValueError, TypeError) as e:
                    logger.debug(f"Failed to insert trade: {e}")

            logger.debug(f"Stored {inserted} trades for {token_id[:16]}...")
            return inserted

    def get_trades(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: int = 10000
    ) -> List[Dict]:
        """Get trades from cache."""
        with self._get_conn() as conn:
            query = """
                SELECT token_id, market_id, trade_id, timestamp, price,
                       size, side, maker, taker, value_usd
                FROM trades
                WHERE token_id = ?
            """
            params = [token_id]

            if start_ts:
                query += " AND timestamp >= ?"
                params.append(start_ts)
            if end_ts:
                query += " AND timestamp <= ?"
                params.append(end_ts)

            query += " ORDER BY timestamp ASC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_trades_by_wallet(
        self,
        wallet: str,
        limit: int = 1000
    ) -> List[Dict]:
        """Get all trades by a specific wallet."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT token_id, market_id, trade_id, timestamp, price,
                       size, side, maker, taker, value_usd
                FROM trades
                WHERE maker = ? OR taker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (wallet, wallet, limit)).fetchall()
            return [dict(row) for row in rows]

    def get_all_cached_trade_tokens(self) -> List[str]:
        """Get list of all tokens with cached trade data."""
        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT DISTINCT token_id FROM trades
            """).fetchall()
            return [row['token_id'] for row in rows]

    # ==================== Markets ====================

    def store_market(self, market: Dict) -> bool:
        """
        Store market metadata.

        Args:
            market: Market dict with condition_id, question, tokens, etc.

        Returns:
            True if stored successfully
        """
        try:
            with self._get_conn() as conn:
                # Parse dates
                start_date = None
                end_date = None

                for key in ['start_date', 'startDate', 'created_at']:
                    if market.get(key):
                        try:
                            val = market[key]
                            if isinstance(val, str):
                                start_date = int(datetime.fromisoformat(
                                    val.replace('Z', '+00:00')
                                ).timestamp())
                            elif isinstance(val, (int, float)):
                                start_date = int(val)
                            break
                        except (ValueError, TypeError):
                            pass

                for key in ['end_date', 'endDate', 'end_date_iso']:
                    if market.get(key):
                        try:
                            val = market[key]
                            if isinstance(val, str):
                                end_date = int(datetime.fromisoformat(
                                    val.replace('Z', '+00:00')
                                ).timestamp())
                            elif isinstance(val, (int, float)):
                                end_date = int(val)
                            break
                        except (ValueError, TypeError):
                            pass

                tokens = market.get('tokens', [])
                tokens_json = json.dumps(tokens) if tokens else '[]'

                conn.execute("""
                    INSERT OR REPLACE INTO markets
                    (condition_id, question, slug, tokens_json, start_date, end_date,
                     resolved, winning_outcome, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market.get('condition_id') or market.get('conditionId'),
                    market.get('question', ''),
                    market.get('slug', ''),
                    tokens_json,
                    start_date,
                    end_date,
                    1 if market.get('resolved') else 0,
                    market.get('winning_outcome') or market.get('winningOutcome'),
                    int(datetime.now(timezone.utc).timestamp())
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to store market: {e}")
            return False

    def get_market(self, condition_id: str) -> Optional[Dict]:
        """Get market by condition ID."""
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT * FROM markets WHERE condition_id = ?
            """, (condition_id,)).fetchone()

            if row:
                result = dict(row)
                result['tokens'] = json.loads(result.get('tokens_json', '[]'))
                result['resolved'] = bool(result.get('resolved'))
                return result
            return None

    def get_resolved_markets(
        self,
        min_end_date: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get all resolved markets."""
        with self._get_conn() as conn:
            query = "SELECT * FROM markets WHERE resolved = 1"
            params = []

            if min_end_date:
                query += " AND end_date >= ?"
                params.append(min_end_date)

            query += " ORDER BY end_date DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result['tokens'] = json.loads(result.get('tokens_json', '[]'))
                result['resolved'] = bool(result.get('resolved'))
                results.append(result)
            return results

    def get_active_markets(
        self,
        min_end_date: Optional[int] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get all active (non-resolved) markets."""
        with self._get_conn() as conn:
            now = int(datetime.now(timezone.utc).timestamp())
            query = "SELECT * FROM markets WHERE resolved = 0"
            params = []

            if min_end_date:
                query += " AND end_date >= ?"
                params.append(min_end_date)

            query += " ORDER BY end_date ASC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result['tokens'] = json.loads(result.get('tokens_json', '[]'))
                result['resolved'] = bool(result.get('resolved'))
                results.append(result)
            return results

    # ==================== Orderbook Snapshots ====================

    def store_orderbook_snapshot(
        self,
        token_id: str,
        timestamp: int,
        best_bid: Optional[float],
        best_ask: Optional[float],
        bid_depth: Optional[List] = None,
        ask_depth: Optional[List] = None
    ) -> bool:
        """Store orderbook snapshot."""
        try:
            spread_pct = None
            if best_bid and best_ask and best_bid > 0:
                spread_pct = (best_ask - best_bid) / best_bid

            total_liq = 0.0
            if bid_depth:
                total_liq += sum(p * s for p, s in bid_depth)
            if ask_depth:
                total_liq += sum(p * s for p, s in ask_depth)

            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO orderbook_snapshots
                    (token_id, timestamp, best_bid, best_ask, bid_depth_json,
                     ask_depth_json, spread_pct, total_liquidity_usd)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    token_id,
                    timestamp,
                    best_bid,
                    best_ask,
                    json.dumps(bid_depth) if bid_depth else None,
                    json.dumps(ask_depth) if ask_depth else None,
                    spread_pct,
                    total_liq
                ))
                return True
        except Exception as e:
            logger.error(f"Failed to store orderbook: {e}")
            return False

    def get_orderbook_snapshot(
        self,
        token_id: str,
        timestamp: int,
        tolerance_seconds: int = 300
    ) -> Optional[Dict]:
        """
        Get orderbook snapshot closest to timestamp.

        Args:
            token_id: Token ID
            timestamp: Target timestamp
            tolerance_seconds: Max time difference to accept

        Returns:
            Orderbook dict or None
        """
        with self._get_conn() as conn:
            row = conn.execute("""
                SELECT *, ABS(timestamp - ?) as time_diff
                FROM orderbook_snapshots
                WHERE token_id = ?
                AND ABS(timestamp - ?) <= ?
                ORDER BY time_diff ASC
                LIMIT 1
            """, (timestamp, token_id, timestamp, tolerance_seconds)).fetchone()

            if row:
                result = dict(row)
                if result.get('bid_depth_json'):
                    result['bid_depth'] = json.loads(result['bid_depth_json'])
                if result.get('ask_depth_json'):
                    result['ask_depth'] = json.loads(result['ask_depth_json'])
                return result
            return None

    # ==================== Cache Management ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_conn() as conn:
            stats = {}

            # Price history stats
            row = conn.execute("""
                SELECT COUNT(*) as count, COUNT(DISTINCT token_id) as tokens
                FROM price_history
            """).fetchone()
            stats['price_points'] = row['count']
            stats['price_tokens'] = row['tokens']

            # Trade stats
            row = conn.execute("""
                SELECT COUNT(*) as count, COUNT(DISTINCT token_id) as tokens,
                       COUNT(DISTINCT maker) as wallets
                FROM trades
            """).fetchone()
            stats['trades'] = row['count']
            stats['trade_tokens'] = row['tokens']
            stats['unique_wallets'] = row['wallets']

            # Market stats
            row = conn.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved
                FROM markets
            """).fetchone()
            stats['markets'] = row['total']
            stats['resolved_markets'] = row['resolved']

            # Orderbook stats
            row = conn.execute("""
                SELECT COUNT(*) as count, COUNT(DISTINCT token_id) as tokens
                FROM orderbook_snapshots
            """).fetchone()
            stats['orderbook_snapshots'] = row['count']
            stats['orderbook_tokens'] = row['tokens']

            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

            return stats

    def clear_old_data(self, max_age_days: int = 60):
        """Clear data older than specified days."""
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=max_age_days)).timestamp())

        with self._get_conn() as conn:
            conn.execute("DELETE FROM price_history WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff,))
            conn.execute("DELETE FROM orderbook_snapshots WHERE timestamp < ?", (cutoff,))
            conn.execute("VACUUM")

        logger.info(f"Cleared data older than {max_age_days} days")

    def clear_all(self):
        """Clear all cached data."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM price_history")
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM markets")
            conn.execute("DELETE FROM orderbook_snapshots")
            conn.execute("DELETE FROM cache_meta")
            conn.execute("VACUUM")

        logger.info("Cleared all cached data")


# Singleton instance
_cache_instance: Optional[BacktestDataCache] = None


def get_cache(db_path: Optional[Path] = None) -> BacktestDataCache:
    """Get or create cache singleton."""
    global _cache_instance
    if _cache_instance is None or (db_path and db_path != _cache_instance.db_path):
        _cache_instance = BacktestDataCache(db_path)
    return _cache_instance
