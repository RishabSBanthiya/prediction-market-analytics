"""
Cloud sync module for uploading data to Supabase Storage.

Supports periodic uploads of SQLite database files to Supabase Storage buckets.
"""

import os
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CloudSyncConfig:
    """Configuration for cloud sync"""

    supabase_url: str
    supabase_key: str
    bucket_name: str = "orderbook-data"
    sync_interval_seconds: int = 3600  # 1 hour default

    @classmethod
    def from_env(cls) -> Optional["CloudSyncConfig"]:
        """Load from environment variables. Returns None if not configured."""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            return None

        return cls(
            supabase_url=url,
            supabase_key=key,
            bucket_name=os.getenv("SUPABASE_BUCKET", "orderbook-data"),
            sync_interval_seconds=int(os.getenv("CLOUD_SYNC_INTERVAL_SECONDS", "3600")),
        )


class SupabaseCloudSync:
    """Handles periodic uploads of database files to Supabase Storage."""

    def __init__(
        self,
        config: CloudSyncConfig,
        db_path: str,
        on_sync_complete: Optional[Callable[[bool, str], None]] = None,
    ):
        """
        Initialize cloud sync.

        Args:
            config: Supabase configuration
            db_path: Path to the SQLite database file to sync
            on_sync_complete: Optional callback(success, message) called after each sync
        """
        self.config = config
        self.db_path = Path(db_path)
        self.on_sync_complete = on_sync_complete
        self._client = None
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0

    def _get_client(self):
        """Lazy initialization of Supabase client."""
        if self._client is None:
            from supabase import create_client
            self._client = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
        return self._client

    async def _ensure_bucket_exists(self) -> bool:
        """Ensure the storage bucket exists, create if not."""
        try:
            client = self._get_client()
            buckets = client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]

            if self.config.bucket_name not in bucket_names:
                logger.info(f"Creating bucket: {self.config.bucket_name}")
                client.storage.create_bucket(
                    self.config.bucket_name,
                    options={"public": False}
                )
                logger.info(f"Bucket created: {self.config.bucket_name}")

            return True
        except Exception as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            return False

    async def sync_now(self) -> tuple[bool, str]:
        """
        Perform an immediate sync of the database file.

        Returns:
            Tuple of (success, message)
        """
        if not self.db_path.exists():
            msg = f"Database file not found: {self.db_path}"
            logger.error(msg)
            return False, msg

        try:
            # Ensure bucket exists
            if not await self._ensure_bucket_exists():
                return False, "Failed to ensure bucket exists"

            client = self._get_client()
            bucket = client.storage.from_(self.config.bucket_name)

            # Generate timestamped filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            remote_filename = f"orderbook_history_{timestamp}.db"

            # Also maintain a "latest" copy for easy access
            latest_filename = "orderbook_history_latest.db"

            # Read the database file
            file_size = self.db_path.stat().st_size
            logger.info(f"Uploading {self.db_path.name} ({file_size / 1024 / 1024:.2f} MB)")

            with open(self.db_path, "rb") as f:
                file_data = f.read()

            # Upload timestamped version
            bucket.upload(
                remote_filename,
                file_data,
                file_options={"content-type": "application/x-sqlite3"}
            )

            # Upload/update latest version with upsert
            bucket.upload(
                latest_filename,
                file_data,
                file_options={"content-type": "application/x-sqlite3", "upsert": "true"}
            )

            self._last_sync = datetime.utcnow()
            self._sync_count += 1

            msg = f"Synced {remote_filename} ({file_size / 1024 / 1024:.2f} MB)"
            logger.info(msg)

            if self.on_sync_complete:
                self.on_sync_complete(True, msg)

            return True, msg

        except Exception as e:
            self._error_count += 1
            msg = f"Sync failed: {e}"
            logger.error(msg)

            if self.on_sync_complete:
                self.on_sync_complete(False, msg)

            return False, msg

    async def _sync_loop(self):
        """Background sync loop."""
        logger.info(
            f"Cloud sync started - interval: {self.config.sync_interval_seconds}s, "
            f"bucket: {self.config.bucket_name}"
        )

        # Initial sync
        await self.sync_now()

        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                if self._running:
                    await self.sync_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                self._error_count += 1
                # Continue running, will retry next interval

    async def start(self):
        """Start the periodic sync background task."""
        if self._running:
            logger.warning("Cloud sync already running")
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info("Cloud sync background task started")

    async def stop(self):
        """Stop the periodic sync background task."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info(
            f"Cloud sync stopped - syncs: {self._sync_count}, errors: {self._error_count}"
        )

    def get_stats(self) -> dict:
        """Get sync statistics."""
        return {
            "running": self._running,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_count": self._sync_count,
            "error_count": self._error_count,
            "bucket": self.config.bucket_name,
            "interval_seconds": self.config.sync_interval_seconds,
        }


async def create_cloud_sync(
    db_path: str,
    on_sync_complete: Optional[Callable[[bool, str], None]] = None,
) -> Optional[SupabaseCloudSync]:
    """
    Factory function to create cloud sync if configured.

    Returns None if SUPABASE_URL and SUPABASE_KEY are not set.
    """
    config = CloudSyncConfig.from_env()

    if config is None:
        logger.info("Cloud sync not configured (SUPABASE_URL/SUPABASE_KEY not set)")
        return None

    return SupabaseCloudSync(
        config=config,
        db_path=db_path,
        on_sync_complete=on_sync_complete,
    )
