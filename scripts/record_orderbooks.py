#!/usr/bin/env python3
"""
Orderbook Recorder Daemon.

Continuously records orderbook snapshots for all active markets.
Data is stored in a separate SQLite database for use in backtesting.

Usage:
    # Run with default settings (30s interval)
    python scripts/record_orderbooks.py

    # Custom interval
    python scripts/record_orderbooks.py --interval 60

    # Dry run (fetch but don't save)
    python scripts/record_orderbooks.py --dry-run

    # Show stats for existing data
    python scripts/record_orderbooks.py stats

    # Cleanup old data
    python scripts/record_orderbooks.py cleanup --days 30
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from polymarket.core.api import PolymarketAPI
from polymarket.core.config import get_config
from polymarket.core.models import Market, Token, OrderbookSnapshot
from polymarket.data.orderbook_storage import OrderbookStorage


# Configure logging
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("orderbook_recorder")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class OrderbookRecorder:
    """
    Daemon that continuously records orderbook snapshots.

    Features:
    - Fetches all active markets periodically
    - Records orderbook snapshots for all tokens
    - Rate-limited to respect API limits
    - Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(
        self,
        interval_seconds: int = 30,
        market_refresh_seconds: int = 300,
        max_concurrent: int = 10,
        dry_run: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize recorder.

        Args:
            interval_seconds: Seconds between recording cycles
            market_refresh_seconds: Seconds between market list refreshes
            max_concurrent: Max concurrent orderbook fetches
            dry_run: If True, fetch but don't save
            db_path: Path to orderbook database
        """
        self.interval = interval_seconds
        self.market_refresh_interval = market_refresh_seconds
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run

        self.config = get_config()
        self.api: Optional[PolymarketAPI] = None
        self.storage = OrderbookStorage(db_path) if not dry_run else None

        self.logger = logging.getLogger("orderbook_recorder")
        self.running = False
        self.shutdown_event = asyncio.Event()

        # State
        self.markets: Dict[str, Market] = {}  # market_id -> Market
        self.token_to_market: Dict[str, Tuple[str, str]] = {}  # token_id -> (market_id, outcome)
        self.last_market_refresh: float = 0

        # Stats
        self.total_snapshots = 0
        self.total_errors = 0
        self.cycles_completed = 0

    async def start(self) -> None:
        """Start the recording daemon."""
        self.running = True
        self.logger.info(
            f"Starting orderbook recorder (interval={self.interval}s, "
            f"dry_run={self.dry_run})"
        )

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        self.api = PolymarketAPI(self.config)
        await self.api.connect()

        try:
            while self.running:
                cycle_start = time.time()

                # Refresh markets if needed
                if time.time() - self.last_market_refresh > self.market_refresh_interval:
                    await self._refresh_markets()

                # Record orderbooks for all tokens
                if self.token_to_market:
                    snapshots_saved, errors = await self._record_cycle()
                    self.total_snapshots += snapshots_saved
                    self.total_errors += errors
                    self.cycles_completed += 1

                    cycle_duration = time.time() - cycle_start
                    self._log_cycle_stats(snapshots_saved, errors, cycle_duration)

                    # Save recording stats
                    if self.storage and not self.dry_run:
                        self.storage.save_recording_stats(
                            markets_count=len(self.markets),
                            tokens_count=len(self.token_to_market),
                            snapshots_saved=snapshots_saved,
                            errors_count=errors,
                            cycle_duration_ms=int(cycle_duration * 1000),
                        )
                else:
                    self.logger.warning("No markets to record, waiting...")

                # Wait for next cycle or shutdown
                sleep_time = max(0, self.interval - (time.time() - cycle_start))
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=sleep_time
                    )
                    # Shutdown was triggered
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue to next cycle
                    pass

        finally:
            await self._cleanup()

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        self.logger.info("Shutdown signal received...")
        self.running = False
        self.shutdown_event.set()

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.api:
            await self.api.close()
        self.logger.info(
            f"Recorder stopped. Total: {self.total_snapshots} snapshots, "
            f"{self.total_errors} errors, {self.cycles_completed} cycles"
        )

    async def _refresh_markets(self) -> None:
        """Fetch and update active markets list."""
        self.logger.info("Refreshing active markets list...")

        try:
            raw_markets = await self.api.fetch_all_markets(
                closed=False,
                active=True,
                max_markets=5000,
                max_concurrent=10
            )

            if not raw_markets:
                self.logger.warning("Failed to fetch markets")
                return

            # Parse markets and build token mapping
            new_markets = {}
            new_token_map = {}

            for m in raw_markets:
                try:
                    market = self._parse_market(m)
                    if not market:
                        continue

                    new_markets[market.condition_id] = market

                    # Map tokens to market
                    for token in market.tokens:
                        new_token_map[token.token_id] = (
                            market.condition_id,
                            token.outcome
                        )

                    # Update market metadata in storage
                    if self.storage and not self.dry_run:
                        token_yes = next(
                            (t.token_id for t in market.tokens if t.outcome.lower() == "yes"),
                            None
                        )
                        token_no = next(
                            (t.token_id for t in market.tokens if t.outcome.lower() == "no"),
                            None
                        )
                        self.storage.update_market_metadata(
                            market_id=market.condition_id,
                            question=market.question,
                            end_date=market.end_date,
                            token_yes_id=token_yes,
                            token_no_id=token_no,
                        )

                except Exception as e:
                    self.logger.warning(f"Error parsing market: {e}")

            self.markets = new_markets
            self.token_to_market = new_token_map
            self.last_market_refresh = time.time()

            self.logger.info(
                f"Loaded {len(self.markets)} markets, {len(self.token_to_market)} tokens"
            )

        except Exception as e:
            self.logger.error(f"Error refreshing markets: {e}")

    def _parse_market(self, raw: dict) -> Optional[Market]:
        """Parse raw market data to Market object."""
        try:
            condition_id = raw.get("conditionId") or raw.get("condition_id")
            if not condition_id:
                return None

            # Parse end date
            end_date_str = raw.get("endDate") or raw.get("end_date")
            if end_date_str:
                if isinstance(end_date_str, str):
                    # Handle ISO format
                    end_date_str = end_date_str.replace("Z", "+00:00")
                    end_date = datetime.fromisoformat(end_date_str)
                else:
                    end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
            else:
                # Default to far future if no end date
                end_date = datetime.now(timezone.utc)

            # Parse tokens
            tokens = []
            raw_tokens = raw.get("tokens", [])
            for t in raw_tokens:
                token_id = t.get("token_id")
                if token_id:
                    tokens.append(Token(
                        token_id=token_id,
                        outcome=t.get("outcome", ""),
                        price=float(t.get("price", 0) or 0),
                    ))

            if not tokens:
                return None

            return Market(
                condition_id=condition_id,
                question=raw.get("question", ""),
                slug=raw.get("slug", ""),
                end_date=end_date,
                tokens=tokens,
                closed=raw.get("closed", False),
                resolved=raw.get("resolved", False),
            )

        except Exception as e:
            self.logger.debug(f"Failed to parse market: {e}")
            return None

    async def _record_cycle(self) -> Tuple[int, int]:
        """
        Record orderbook snapshots for all tokens.

        Returns:
            Tuple of (snapshots_saved, errors)
        """
        token_ids = list(self.token_to_market.keys())
        self.logger.debug(f"Recording {len(token_ids)} orderbooks...")

        # Use semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: List[Tuple[Optional[OrderbookSnapshot], str, Optional[str]]] = []

        async def fetch_one(token_id: str) -> Tuple[Optional[OrderbookSnapshot], str, Optional[str]]:
            async with semaphore:
                try:
                    snapshot = await self.api.fetch_orderbook(token_id)
                    market_id, outcome = self.token_to_market.get(token_id, ("", None))
                    return (snapshot, market_id, outcome)
                except Exception as e:
                    self.logger.debug(f"Error fetching {token_id}: {e}")
                    return (None, "", None)

        # Stagger requests to avoid rate limit spikes
        # Process in batches with small delays between batches
        batch_size = self.max_concurrent * 2
        snapshots_to_save = []
        errors = 0

        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i + batch_size]
            tasks = [fetch_one(tid) for tid in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    errors += 1
                    continue
                snapshot, market_id, outcome = result
                if snapshot:
                    snapshots_to_save.append((snapshot, market_id, outcome))
                else:
                    errors += 1

            # Small delay between batches to spread load
            if i + batch_size < len(token_ids):
                await asyncio.sleep(0.1)

        # Batch save to database
        saved = 0
        if snapshots_to_save and self.storage and not self.dry_run:
            try:
                saved = self.storage.save_snapshots_batch(snapshots_to_save)
            except Exception as e:
                self.logger.error(f"Error saving batch: {e}")
                errors += len(snapshots_to_save)
        elif self.dry_run:
            saved = len(snapshots_to_save)

        return saved, errors

    def _log_cycle_stats(self, saved: int, errors: int, duration: float) -> None:
        """Log stats for a recording cycle."""
        total_tokens = len(self.token_to_market)
        success_rate = (saved / total_tokens * 100) if total_tokens > 0 else 0

        self.logger.info(
            f"Cycle {self.cycles_completed}: "
            f"{saved}/{total_tokens} saved ({success_rate:.1f}%), "
            f"{errors} errors, "
            f"{duration:.2f}s"
        )


def cmd_run(args):
    """Run the recorder daemon."""
    logger = setup_logging("logs/orderbook_recorder.log")
    logger.info("=" * 60)
    logger.info("Orderbook Recorder Starting")
    logger.info("=" * 60)

    recorder = OrderbookRecorder(
        interval_seconds=args.interval,
        market_refresh_seconds=args.market_refresh,
        max_concurrent=args.concurrency,
        dry_run=args.dry_run,
        db_path=args.db,
    )

    asyncio.run(recorder.start())


def cmd_stats(args):
    """Show recording statistics."""
    storage = OrderbookStorage(args.db)
    stats = storage.get_storage_stats()

    print("\n=== Orderbook Recording Stats ===\n")
    print(f"Database: {storage.db_path}")
    print(f"Size: {stats['db_size_mb']} MB")
    print(f"Snapshots: {stats['snapshot_count']:,}")
    print(f"Markets: {stats['market_count']}")
    print(f"First: {stats['first_snapshot']}")
    print(f"Last: {stats['last_snapshot']}")


def cmd_cleanup(args):
    """Cleanup old data."""
    storage = OrderbookStorage(args.db)
    deleted = storage.cleanup_old_data(args.days)
    print(f"Deleted {deleted:,} snapshots older than {args.days} days")


def main():
    parser = argparse.ArgumentParser(
        description="Orderbook Recorder Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the recorder daemon")
    run_parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between recording cycles (default: 30)"
    )
    run_parser.add_argument(
        "--market-refresh",
        type=int,
        default=300,
        help="Seconds between market list refreshes (default: 300)"
    )
    run_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent API requests (default: 10)"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch but don't save to database"
    )
    run_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to orderbook database"
    )
    run_parser.set_defaults(func=cmd_run)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show recording statistics")
    stats_parser.add_argument("--db", type=str, default=None)
    stats_parser.set_defaults(func=cmd_stats)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Delete old data")
    cleanup_parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Delete data older than this many days"
    )
    cleanup_parser.add_argument("--db", type=str, default=None)
    cleanup_parser.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    # Default to run command if no subcommand specified
    if args.command is None:
        # Re-parse with run defaults
        args = parser.parse_args(["run"] + sys.argv[1:])

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
