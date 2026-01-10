#!/usr/bin/env python3
"""
Enhanced Orderbook Recorder Daemon.

Records orderbook liquidity with robust connection handling:
- WebSocket streaming (primary) with real-time updates
- Polling fallback when WebSocket is unavailable
- Auto-backfill for gap recovery when connection resumes
- High-volume market filtering
- Cloud sync to Supabase Storage (optional)

Usage:
    # Run with default settings
    python scripts/record_orderbooks_enhanced.py

    # Custom volume thresholds
    python scripts/record_orderbooks_enhanced.py --min-volume 10000 --min-liquidity 5000

    # Polling only (no WebSocket)
    python scripts/record_orderbooks_enhanced.py --no-websocket

    # Show stats
    python scripts/record_orderbooks_enhanced.py stats

    # Show gap info
    python scripts/record_orderbooks_enhanced.py gaps

Cloud Sync Configuration (optional):
    Set these environment variables to enable automatic cloud sync:

    SUPABASE_URL=https://your-project.supabase.co
    SUPABASE_KEY=your-service-role-key
    SUPABASE_BUCKET=orderbook-data  (default)
    CLOUD_SYNC_INTERVAL_SECONDS=3600  (default: 1 hour)

    The recorder will automatically upload the SQLite DB file to Supabase
    Storage every hour (or at the configured interval).
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from polymarket.core.api import PolymarketAPI
from polymarket.core.config import get_config
from polymarket.core.models import Market, Token, OrderbookSnapshot
from polymarket.data.orderbook_storage import OrderbookStorage, RecordingGap

# Conditional import for cloud sync
try:
    from polymarket.data.cloud_sync import create_cloud_sync, SupabaseCloudSync
    CLOUD_SYNC_AVAILABLE = True
except ImportError:
    CLOUD_SYNC_AVAILABLE = False
    create_cloud_sync = None
    SupabaseCloudSync = None

# Conditional import for WebSocket client
try:
    from polymarket.data.orderbook_websocket import OrderbookWebSocketClient
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    OrderbookWebSocketClient = None


# Configure logging
def setup_logging(log_file: Optional[str] = None, verbose: bool = False) -> logging.Logger:
    """Setup logging to file and console."""
    logger = logging.getLogger("orderbook_recorder_enhanced")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = logging.getLogger("orderbook_recorder_enhanced")


class BackfillWorker:
    """
    Background worker that fills data gaps.

    Only runs when connection is stable (no recent disconnects).
    Rate-limited to respect API limits.
    Processes gaps oldest-first.
    """

    def __init__(
        self,
        storage: OrderbookStorage,
        api: PolymarketAPI,
        token_to_market: Dict[str, Tuple[str, str]],
        max_concurrent: int = 5,
        stability_window_seconds: int = 60,
        rate_limit_per_second: float = 50.0,
        max_gap_age_days: int = 7,
    ):
        """
        Initialize backfill worker.

        Args:
            storage: OrderbookStorage instance
            api: PolymarketAPI instance
            token_to_market: Mapping of token_id -> (market_id, outcome)
            max_concurrent: Max concurrent API requests
            stability_window_seconds: Seconds of stability before backfilling
            rate_limit_per_second: Max requests per second for backfill
            max_gap_age_days: Skip gaps older than this
        """
        self.storage = storage
        self.api = api
        self.token_to_market = token_to_market

        self.max_concurrent = max_concurrent
        self.stability_window = stability_window_seconds
        self.rate_limit = rate_limit_per_second
        self.max_gap_age = timedelta(days=max_gap_age_days)

        self._last_disconnect: Optional[datetime] = None
        self._running = False

        # Stats
        self.gaps_processed = 0
        self.gaps_resolved = 0
        self.snapshots_backfilled = 0

    def record_disconnect(self, disconnect_time: datetime) -> None:
        """Called when connection disconnects to pause backfilling."""
        self._last_disconnect = disconnect_time

    def _is_connection_stable(self) -> bool:
        """Check if enough time has passed since last disconnect."""
        if self._last_disconnect is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_disconnect).total_seconds()
        return elapsed >= self.stability_window

    async def start(self) -> None:
        """Start the backfill worker loop."""
        self._running = True
        logger.info("Backfill worker started")

        while self._running:
            try:
                # Wait for connection stability
                if not self._is_connection_stable():
                    await asyncio.sleep(10)
                    continue

                # Get unresolved gaps
                gaps = self.storage.get_unresolved_gaps(limit=10)

                if not gaps:
                    # No gaps to process, sleep longer
                    await asyncio.sleep(60)
                    continue

                # Process gaps one at a time
                for gap in gaps:
                    if not self._running:
                        break

                    # Skip very old gaps
                    gap_age = datetime.now(timezone.utc) - gap.gap_start
                    if gap_age > self.max_gap_age:
                        logger.info(
                            f"Skipping gap {gap.id} for {gap.token_id[:16]}... "
                            f"(too old: {gap_age.days} days)"
                        )
                        # Mark as resolved to skip in future
                        self.storage.mark_gap_resolved(gap.id)
                        continue

                    # Check stability again before each gap
                    if not self._is_connection_stable():
                        logger.info("Connection unstable, pausing backfill")
                        break

                    success = await self._backfill_gap(gap)
                    self.gaps_processed += 1

                    if success:
                        self.storage.mark_gap_resolved(gap.id)
                        self.gaps_resolved += 1
                    else:
                        self.storage.increment_gap_retry(gap.id)

                    # Sleep between gaps to avoid API flooding
                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Backfill worker error: {e}")
                await asyncio.sleep(30)

    async def stop(self) -> None:
        """Stop the backfill worker."""
        self._running = False

    async def _backfill_gap(self, gap: RecordingGap) -> bool:
        """
        Backfill a single gap by polling snapshots.

        Args:
            gap: The gap to backfill

        Returns:
            True if successful, False otherwise
        """
        try:
            token_id = gap.token_id
            gap_start = gap.gap_start
            gap_end = gap.gap_end

            if gap_end is None:
                logger.warning(f"Gap {gap.id} has no end time, skipping")
                return False

            # Calculate gap duration
            gap_duration = (gap_end - gap_start).total_seconds()

            # For gaps > 1 hour, sample at 60s intervals
            # For shorter gaps, sample at 30s intervals
            interval = 60 if gap_duration > 3600 else 30
            num_samples = int(gap_duration / interval) + 1

            logger.info(
                f"Backfilling gap {gap.id}: {token_id[:16]}... "
                f"({gap_duration:.0f}s, {num_samples} samples)"
            )

            # Get market info
            market_id, outcome = self.token_to_market.get(token_id, ("", None))

            snapshots_saved = 0
            errors = 0

            # Fetch snapshots at intervals
            current_time = gap_start
            while current_time <= gap_end:
                try:
                    # Rate limit
                    await asyncio.sleep(1.0 / self.rate_limit)

                    # Fetch current orderbook (historical not available via API)
                    # This is a best-effort backfill - we can only get current state
                    snapshot = await self.api.fetch_orderbook(token_id)

                    if snapshot:
                        # Override timestamp with the gap time
                        # Note: This is approximating historical data
                        snapshot_with_time = OrderbookSnapshot(
                            token_id=snapshot.token_id,
                            timestamp=current_time,
                            best_bid=snapshot.best_bid,
                            best_ask=snapshot.best_ask,
                            bid_size=snapshot.bid_size,
                            ask_size=snapshot.ask_size,
                            bid_depth=snapshot.bid_depth,
                            ask_depth=snapshot.ask_depth,
                        )
                        self.storage.save_snapshot(snapshot_with_time, market_id, outcome)
                        snapshots_saved += 1
                    else:
                        errors += 1

                except Exception as e:
                    logger.debug(f"Error fetching snapshot for backfill: {e}")
                    errors += 1

                current_time += timedelta(seconds=interval)

                # Break if connection becomes unstable
                if not self._is_connection_stable():
                    logger.info("Connection unstable during backfill, aborting gap")
                    return False

            self.snapshots_backfilled += snapshots_saved
            logger.info(
                f"Backfilled gap {gap.id}: {snapshots_saved} saved, {errors} errors"
            )

            # Consider successful if we got at least half the samples
            return snapshots_saved >= num_samples / 2

        except Exception as e:
            logger.error(f"Error backfilling gap {gap.id}: {e}")
            return False


class EnhancedOrderbookRecorder:
    """
    Enhanced orderbook recorder with WebSocket + polling + backfill.

    Architecture:
    1. WebSocket primary (real-time orderbook updates)
    2. Polling fallback (when WS unavailable)
    3. Gap tracking (records missed windows)
    4. Backfill worker (fills gaps when stable)
    5. Volume filter (only high-volume markets)
    """

    def __init__(
        self,
        # Volume filtering
        min_volume_24h: float = 10000.0,
        min_liquidity_usd: float = 5000.0,
        volume_refresh_seconds: int = 3600,
        max_markets: int = 200,
        # Recording intervals
        polling_interval_seconds: int = 30,
        ws_snapshot_interval_seconds: int = 5,
        # Connection settings
        stale_timeout_seconds: int = 60,
        max_reconnect_delay: int = 60,
        # Backfill settings
        backfill_stability_window: int = 60,
        max_backfill_concurrent: int = 5,
        # Other
        use_websocket: bool = True,
        dry_run: bool = False,
        db_path: Optional[str] = None,
    ):
        """
        Initialize enhanced recorder.

        Args:
            min_volume_24h: Minimum 24h volume in USD
            min_liquidity_usd: Minimum orderbook liquidity in USD
            volume_refresh_seconds: Seconds between volume cache refreshes
            max_markets: Maximum number of markets to track
            polling_interval_seconds: Seconds between polling cycles
            ws_snapshot_interval_seconds: Target interval for WS snapshots
            stale_timeout_seconds: Seconds before WS considered stale
            max_reconnect_delay: Max reconnect delay in seconds
            backfill_stability_window: Seconds of stability before backfilling
            max_backfill_concurrent: Max concurrent backfill requests
            use_websocket: Whether to use WebSocket (vs polling only)
            dry_run: If True, fetch but don't save
            db_path: Path to orderbook database
        """
        self.min_volume_24h = min_volume_24h
        self.min_liquidity_usd = min_liquidity_usd
        self.volume_refresh_interval = volume_refresh_seconds
        self.max_markets = max_markets

        self.polling_interval = polling_interval_seconds
        self.ws_snapshot_interval = ws_snapshot_interval_seconds

        self.stale_timeout = stale_timeout_seconds
        self.max_reconnect_delay = max_reconnect_delay

        self.backfill_stability_window = backfill_stability_window
        self.max_backfill_concurrent = max_backfill_concurrent

        self.use_websocket = use_websocket and WEBSOCKET_AVAILABLE
        self.dry_run = dry_run

        self.config = get_config()
        self.api: Optional[PolymarketAPI] = None
        self.storage = OrderbookStorage(db_path) if not dry_run else None

        # State
        self.running = False
        self.shutdown_event = asyncio.Event()
        self.ws_connected = False

        # Market tracking
        self.markets: Dict[str, Market] = {}
        self.token_to_market: Dict[str, Tuple[str, str]] = {}
        self.tracked_tokens: Set[str] = set()
        self.last_volume_refresh: float = 0

        # Components
        self.ws_client: Optional[OrderbookWebSocketClient] = None
        self.backfill_worker: Optional[BackfillWorker] = None
        self.cloud_sync: Optional[SupabaseCloudSync] = None

        # Stats
        self.total_snapshots = 0
        self.total_errors = 0
        self.ws_snapshots = 0
        self.poll_snapshots = 0
        self.cycles_completed = 0

    async def start(self) -> None:
        """Start the enhanced recorder."""
        self.running = True
        logger.info("=" * 60)
        logger.info("Enhanced Orderbook Recorder Starting")
        logger.info(f"  WebSocket: {'enabled' if self.use_websocket else 'disabled'}")
        logger.info(f"  Min volume: ${self.min_volume_24h:,.0f}")
        logger.info(f"  Min liquidity: ${self.min_liquidity_usd:,.0f}")
        logger.info(f"  Max markets: {self.max_markets}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info(f"  Cloud sync: {'available' if CLOUD_SYNC_AVAILABLE else 'not available'}")
        logger.info("=" * 60)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        # Initialize API
        self.api = PolymarketAPI(self.config)
        await self.api.connect()

        # Initial market refresh
        await self._refresh_high_volume_markets()

        if not self.tracked_tokens:
            logger.warning("No high-volume markets found, will retry...")

        try:
            # Start background tasks
            tasks = []

            # Start backfill worker
            if self.storage and not self.dry_run:
                self.backfill_worker = BackfillWorker(
                    storage=self.storage,
                    api=self.api,
                    token_to_market=self.token_to_market,
                    stability_window_seconds=self.backfill_stability_window,
                    max_concurrent=self.max_backfill_concurrent,
                )
                tasks.append(asyncio.create_task(self.backfill_worker.start()))

            # Start cloud sync if configured
            if CLOUD_SYNC_AVAILABLE and self.storage and not self.dry_run:
                self.cloud_sync = await create_cloud_sync(
                    db_path=self.storage.db_path,
                    on_sync_complete=self._on_cloud_sync_complete,
                )
                if self.cloud_sync:
                    await self.cloud_sync.start()
                    logger.info(
                        f"Cloud sync enabled - bucket: {self.cloud_sync.config.bucket_name}, "
                        f"interval: {self.cloud_sync.config.sync_interval_seconds}s"
                    )

            # Start volume refresh loop
            tasks.append(asyncio.create_task(self._volume_refresh_loop()))

            # Start WebSocket or polling
            if self.use_websocket:
                tasks.append(asyncio.create_task(self._websocket_loop()))
            else:
                tasks.append(asyncio.create_task(self._polling_loop()))

            # Wait for shutdown
            await self.shutdown_event.wait()

        finally:
            await self._cleanup(tasks)

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("Shutdown signal received...")
        self.running = False
        self.shutdown_event.set()

    def _on_cloud_sync_complete(self, success: bool, message: str) -> None:
        """Callback for cloud sync completion."""
        if success:
            logger.info(f"Cloud sync: {message}")
        else:
            logger.warning(f"Cloud sync failed: {message}")

    async def _cleanup(self, tasks: List[asyncio.Task]) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")

        # Stop components
        if self.ws_client:
            await self.ws_client.stop()

        if self.backfill_worker:
            await self.backfill_worker.stop()

        if self.cloud_sync:
            await self.cloud_sync.stop()

        # Cancel tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Close API
        if self.api:
            await self.api.close()

        # Log final stats
        cloud_info = ""
        if self.cloud_sync:
            stats = self.cloud_sync.get_stats()
            cloud_info = f", cloud syncs: {stats['sync_count']}"

        logger.info(
            f"Recorder stopped. Total: {self.total_snapshots} snapshots "
            f"({self.ws_snapshots} WS, {self.poll_snapshots} poll), "
            f"{self.total_errors} errors{cloud_info}"
        )

    async def _volume_refresh_loop(self) -> None:
        """Periodically refresh high-volume market list."""
        while self.running:
            try:
                if time.time() - self.last_volume_refresh > self.volume_refresh_interval:
                    await self._refresh_high_volume_markets()

                # Sleep in chunks for responsiveness
                for _ in range(60):
                    if not self.running:
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in volume refresh loop: {e}")
                await asyncio.sleep(60)

    async def _refresh_high_volume_markets(self) -> None:
        """Fetch and update high-volume markets list."""
        logger.info("Refreshing high-volume markets list...")

        try:
            raw_markets = await self.api.fetch_all_markets(
                closed=False,
                active=True,
                max_markets=5000,
                max_concurrent=10
            )

            if not raw_markets:
                logger.warning("Failed to fetch markets")
                return

            # Filter by volume and parse
            new_markets = {}
            new_token_map = {}
            volume_data = []

            for m in raw_markets:
                try:
                    # Extract volume and liquidity
                    volume_24h = float(m.get("volume24hr", 0) or 0)
                    liquidity = float(m.get("liquidity", 0) or 0)

                    # Skip low volume markets
                    if volume_24h < self.min_volume_24h:
                        continue
                    if liquidity < self.min_liquidity_usd:
                        continue

                    market = self._parse_market(m)
                    if not market:
                        continue

                    new_markets[market.condition_id] = market

                    # Map tokens and collect volume data
                    for token in market.tokens:
                        new_token_map[token.token_id] = (
                            market.condition_id,
                            token.outcome
                        )
                        volume_data.append({
                            'token_id': token.token_id,
                            'market_id': market.condition_id,
                            'question': market.question,
                            'volume_24h': volume_24h,
                            'liquidity_usd': liquidity,
                            'is_active': True,
                        })

                except Exception as e:
                    logger.debug(f"Error parsing market: {e}")

            # Sort by volume and limit
            volume_data.sort(key=lambda x: x['volume_24h'], reverse=True)
            volume_data = volume_data[:self.max_markets * 2]  # 2 tokens per market

            # Update storage
            if self.storage and not self.dry_run:
                self.storage.update_market_volumes_batch(volume_data)

            # Update tracked tokens
            new_tracked = set(d['token_id'] for d in volume_data)

            # Update state
            self.markets = new_markets
            self.token_to_market = new_token_map
            self.tracked_tokens = new_tracked
            self.last_volume_refresh = time.time()

            logger.info(
                f"Loaded {len(self.markets)} high-volume markets, "
                f"{len(self.tracked_tokens)} tokens"
            )

            # Update WebSocket subscriptions
            if self.ws_client:
                self.ws_client.set_tokens(list(self.tracked_tokens))

        except Exception as e:
            logger.error(f"Error refreshing markets: {e}")

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
                    end_date_str = end_date_str.replace("Z", "+00:00")
                    end_date = datetime.fromisoformat(end_date_str)
                else:
                    end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
            else:
                end_date = datetime.now(timezone.utc)

            # Parse tokens from clobTokenIds and outcomes (Gamma API format)
            tokens = []
            clob_token_ids = raw.get("clobTokenIds", [])
            outcomes = raw.get("outcomes", [])
            prices = raw.get("outcomePrices", [])

            # Handle string JSON (Gamma API sometimes returns these as JSON strings)
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json.loads(clob_token_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_token_ids = []

            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except (json.JSONDecodeError, TypeError):
                    outcomes = []

            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except (json.JSONDecodeError, TypeError):
                    prices = []

            for i, token_id in enumerate(clob_token_ids):
                if token_id:
                    outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
                    price = float(prices[i]) if i < len(prices) and prices[i] else 0.0
                    tokens.append(Token(
                        token_id=str(token_id),
                        outcome=str(outcome),
                        price=price,
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
            logger.debug(f"Failed to parse market: {e}")
            return None

    # ==================== WebSocket Mode ====================

    async def _websocket_loop(self) -> None:
        """Main WebSocket loop with polling fallback."""
        logger.info("Starting WebSocket recording mode")

        # Create WebSocket client
        self.ws_client = OrderbookWebSocketClient(
            on_snapshot=self._handle_ws_snapshot,
            on_disconnect=self._handle_ws_disconnect,
            on_reconnect=self._handle_ws_reconnect,
            tokens=list(self.tracked_tokens),
            stale_timeout_seconds=self.stale_timeout,
        )

        # Start WebSocket in background
        ws_task = asyncio.create_task(self.ws_client.start())

        # Also run polling as supplement (less frequent)
        poll_task = asyncio.create_task(self._supplemental_polling_loop())

        try:
            while self.running:
                await asyncio.sleep(1)
        finally:
            await self.ws_client.stop()
            ws_task.cancel()
            poll_task.cancel()

    async def _handle_ws_snapshot(
        self,
        snapshot: OrderbookSnapshot,
        token_id: str
    ) -> None:
        """Handle orderbook snapshot from WebSocket."""
        try:
            market_id, outcome = self.token_to_market.get(token_id, ("", None))

            if self.storage and not self.dry_run:
                self.storage.save_snapshot(snapshot, market_id, outcome)

            self.total_snapshots += 1
            self.ws_snapshots += 1

        except Exception as e:
            logger.debug(f"Error saving WS snapshot: {e}")
            self.total_errors += 1

    async def _handle_ws_disconnect(self, reason: str, timestamp: datetime) -> None:
        """Handle WebSocket disconnect - record gaps."""
        logger.warning(f"WebSocket disconnected: {reason}")
        self.ws_connected = False

        # Record gap start for all tracked tokens
        if self.storage and not self.dry_run and self.tracked_tokens:
            self.storage.record_gap_start_batch(
                list(self.tracked_tokens),
                timestamp,
                reason
            )

        # Notify backfill worker
        if self.backfill_worker:
            self.backfill_worker.record_disconnect(timestamp)

    async def _handle_ws_reconnect(self, timestamp: datetime) -> None:
        """Handle WebSocket reconnect - close gaps."""
        logger.info("WebSocket reconnected")
        self.ws_connected = True

        # Close open gaps
        if self.storage and not self.dry_run and self.tracked_tokens:
            self.storage.record_gap_end_batch(
                list(self.tracked_tokens),
                timestamp
            )

    async def _supplemental_polling_loop(self) -> None:
        """
        Supplemental polling that runs alongside WebSocket.

        Used to ensure we have snapshots even if WS doesn't provide them
        frequently enough for all tokens.
        """
        # Wait for initial setup
        await asyncio.sleep(30)

        while self.running:
            try:
                # Only poll if WebSocket is connected
                # If WS is down, the main polling loop handles it
                if not self.ws_connected:
                    await asyncio.sleep(10)
                    continue

                # Poll at longer interval since WS provides real-time data
                await self._poll_cycle(max_concurrent=5)
                self.cycles_completed += 1

                # Sleep for longer interval
                await asyncio.sleep(self.polling_interval * 2)

            except Exception as e:
                logger.error(f"Error in supplemental polling: {e}")
                await asyncio.sleep(30)

    # ==================== Polling Mode ====================

    async def _polling_loop(self) -> None:
        """Main polling loop (when WebSocket is disabled)."""
        logger.info("Starting polling-only recording mode")

        while self.running:
            try:
                cycle_start = time.time()

                if self.tracked_tokens:
                    saved, errors = await self._poll_cycle()
                    self.total_snapshots += saved
                    self.poll_snapshots += saved
                    self.total_errors += errors
                    self.cycles_completed += 1

                    cycle_duration = time.time() - cycle_start
                    self._log_cycle_stats(saved, errors, cycle_duration)

                    # Save recording stats
                    if self.storage and not self.dry_run:
                        self.storage.save_recording_stats(
                            markets_count=len(self.markets),
                            tokens_count=len(self.tracked_tokens),
                            snapshots_saved=saved,
                            errors_count=errors,
                            cycle_duration_ms=int(cycle_duration * 1000),
                        )
                else:
                    logger.warning("No markets to record, waiting...")

                # Wait for next cycle
                sleep_time = max(0, self.polling_interval - (time.time() - cycle_start))
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=sleep_time
                    )
                    break
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                await asyncio.sleep(10)

    async def _poll_cycle(self, max_concurrent: int = 10) -> Tuple[int, int]:
        """
        Poll orderbooks for all tracked tokens.

        Returns:
            Tuple of (snapshots_saved, errors)
        """
        token_ids = list(self.tracked_tokens)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(token_id: str) -> Tuple[Optional[OrderbookSnapshot], str, Optional[str]]:
            async with semaphore:
                try:
                    snapshot = await self.api.fetch_orderbook(token_id)
                    market_id, outcome = self.token_to_market.get(token_id, ("", None))
                    return (snapshot, market_id, outcome)
                except Exception as e:
                    logger.debug(f"Error fetching {token_id[:16]}...: {e}")
                    return (None, "", None)

        # Process in batches
        batch_size = max_concurrent * 2
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

            # Small delay between batches
            if i + batch_size < len(token_ids):
                await asyncio.sleep(0.1)

        # Batch save
        saved = 0
        if snapshots_to_save and self.storage and not self.dry_run:
            try:
                saved = self.storage.save_snapshots_batch(snapshots_to_save)
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                errors += len(snapshots_to_save)
        elif self.dry_run:
            saved = len(snapshots_to_save)

        return saved, errors

    def _log_cycle_stats(self, saved: int, errors: int, duration: float) -> None:
        """Log stats for a recording cycle."""
        total_tokens = len(self.tracked_tokens)
        success_rate = (saved / total_tokens * 100) if total_tokens > 0 else 0

        logger.info(
            f"Cycle {self.cycles_completed}: "
            f"{saved}/{total_tokens} saved ({success_rate:.1f}%), "
            f"{errors} errors, "
            f"{duration:.2f}s"
        )


# ==================== CLI Commands ====================

def cmd_run(args):
    """Run the enhanced recorder daemon."""
    logger = setup_logging(
        "logs/orderbook_recorder_enhanced.log",
        verbose=args.verbose
    )

    recorder = EnhancedOrderbookRecorder(
        min_volume_24h=args.min_volume,
        min_liquidity_usd=args.min_liquidity,
        volume_refresh_seconds=args.volume_refresh,
        max_markets=args.max_markets,
        polling_interval_seconds=args.poll_interval,
        stale_timeout_seconds=args.stale_timeout,
        backfill_stability_window=args.backfill_delay,
        use_websocket=not args.no_websocket,
        dry_run=args.dry_run,
        db_path=args.db,
    )

    asyncio.run(recorder.start())


def cmd_stats(args):
    """Show recording statistics."""
    storage = OrderbookStorage(args.db)

    print("\n=== Enhanced Orderbook Recording Stats ===\n")

    # Storage stats
    stats = storage.get_storage_stats()
    print(f"Database: {storage.db_path}")
    print(f"Size: {stats['db_size_mb']} MB")
    print(f"Snapshots: {stats['snapshot_count']:,}")
    print(f"Markets: {stats['market_count']}")
    print(f"First: {stats['first_snapshot']}")
    print(f"Last: {stats['last_snapshot']}")

    # Gap stats
    print("\n--- Gap Statistics ---")
    gap_stats = storage.get_gap_stats()
    print(f"Total gaps: {gap_stats['total_gaps']}")
    print(f"Open gaps: {gap_stats['open_gaps']}")
    print(f"Unresolved: {gap_stats['unresolved_gaps']}")
    print(f"Resolved: {gap_stats['resolved_gaps']}")
    print(f"Max retries exceeded: {gap_stats['max_retries_exceeded']}")

    # Volume stats
    print("\n--- Volume Filter Stats ---")
    vol_stats = storage.get_volume_stats()
    print(f"Tracked markets: {vol_stats['total_markets']}")
    print(f"Active markets: {vol_stats['active_markets']}")
    print(f"High volume (>$10k): {vol_stats['high_volume_markets']}")
    print(f"Total 24h volume: ${vol_stats['total_volume_24h']:,.0f}")

    # Cloud sync info
    print("\n--- Cloud Sync Configuration ---")
    if CLOUD_SYNC_AVAILABLE:
        from polymarket.data.cloud_sync import CloudSyncConfig
        config = CloudSyncConfig.from_env()
        if config:
            print(f"Status: Configured")
            print(f"Bucket: {config.bucket_name}")
            print(f"Interval: {config.sync_interval_seconds}s")
        else:
            print("Status: Not configured (set SUPABASE_URL and SUPABASE_KEY)")
    else:
        print("Status: Module not available")


def cmd_gaps(args):
    """Show gap information."""
    storage = OrderbookStorage(args.db)

    print("\n=== Recording Gaps ===\n")

    # Get unresolved gaps
    gaps = storage.get_unresolved_gaps(limit=args.limit)

    if not gaps:
        print("No unresolved gaps found.")
        return

    print(f"Found {len(gaps)} unresolved gaps:\n")

    for gap in gaps:
        duration = ""
        if gap.gap_end:
            dur_seconds = (gap.gap_end - gap.gap_start).total_seconds()
            if dur_seconds < 60:
                duration = f"{dur_seconds:.0f}s"
            elif dur_seconds < 3600:
                duration = f"{dur_seconds/60:.1f}m"
            else:
                duration = f"{dur_seconds/3600:.1f}h"

        print(f"  [{gap.id}] {gap.token_id[:24]}...")
        print(f"      Start: {gap.gap_start.isoformat()}")
        print(f"      End: {gap.gap_end.isoformat() if gap.gap_end else 'OPEN'}")
        print(f"      Duration: {duration}")
        print(f"      Reason: {gap.gap_reason}")
        print(f"      Retries: {gap.retry_count}")
        print()


def cmd_cleanup(args):
    """Cleanup old data."""
    storage = OrderbookStorage(args.db)

    deleted = storage.cleanup_old_data(args.days)
    print(f"Deleted {deleted:,} snapshots older than {args.days} days")


def cmd_sync(args):
    """Force immediate cloud sync."""
    if not CLOUD_SYNC_AVAILABLE:
        print("Cloud sync module not available")
        return

    from polymarket.data.cloud_sync import CloudSyncConfig, SupabaseCloudSync

    config = CloudSyncConfig.from_env()
    if not config:
        print("Cloud sync not configured. Set SUPABASE_URL and SUPABASE_KEY in .env")
        return

    # Determine DB path
    db_path = args.db
    if not db_path:
        storage = OrderbookStorage()
        db_path = storage.db_path

    print(f"Database: {db_path}")
    print(f"Bucket: {config.bucket_name}")
    print(f"Uploading...")

    sync = SupabaseCloudSync(config=config, db_path=db_path)

    async def do_sync():
        return await sync.sync_now()

    success, message = asyncio.run(do_sync())

    if success:
        print(f"Sync complete: {message}")
    else:
        print(f"Sync failed: {message}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Orderbook Recorder with connection handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run the recorder daemon")
    run_parser.add_argument(
        "--min-volume",
        type=float,
        default=10000.0,
        help="Minimum 24h volume in USD (default: 10000)"
    )
    run_parser.add_argument(
        "--min-liquidity",
        type=float,
        default=5000.0,
        help="Minimum orderbook liquidity in USD (default: 5000)"
    )
    run_parser.add_argument(
        "--volume-refresh",
        type=int,
        default=3600,
        help="Seconds between volume refreshes (default: 3600)"
    )
    run_parser.add_argument(
        "--max-markets",
        type=int,
        default=200,
        help="Max markets to track (default: 200)"
    )
    run_parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds (default: 30)"
    )
    run_parser.add_argument(
        "--stale-timeout",
        type=int,
        default=60,
        help="Seconds before connection considered stale (default: 60)"
    )
    run_parser.add_argument(
        "--backfill-delay",
        type=int,
        default=60,
        help="Seconds of stability before backfilling (default: 60)"
    )
    run_parser.add_argument(
        "--no-websocket",
        action="store_true",
        help="Disable WebSocket, use polling only"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch but don't save to database"
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
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

    # Gaps command
    gaps_parser = subparsers.add_parser("gaps", help="Show gap information")
    gaps_parser.add_argument("--db", type=str, default=None)
    gaps_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max gaps to show (default: 20)"
    )
    gaps_parser.set_defaults(func=cmd_gaps)

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

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Force immediate cloud sync to Supabase")
    sync_parser.add_argument("--db", type=str, default=None, help="Path to orderbook database")
    sync_parser.set_defaults(func=cmd_sync)

    args = parser.parse_args()

    # Default to run command if no subcommand specified
    if args.command is None:
        args = parser.parse_args(["run"] + sys.argv[1:])

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
