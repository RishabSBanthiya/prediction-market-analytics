"""
Sliding window rate limiter.

Each ExchangeClient owns its own limiter configured for that platform's limits.
Ported from polymarket-analytics with multi-exchange support.
"""

import asyncio
import time
import logging
from collections import deque
from typing import Deque

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Tracks requests in the last N seconds and blocks if over limit.
    Each exchange client creates its own instance with platform-specific limits.

    Examples:
        Polymarket CLOB: RateLimiter(9000, 10)   # 9000 req/10s
        Kalshi:          RateLimiter(100, 10)     # 100 req/10s
        Hyperliquid:     RateLimiter(1200, 60)    # 1200 req/min
    """

    def __init__(self, max_requests: int, window_seconds: int = 10):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Deque[float] = deque()
        self._lock = asyncio.Lock()

    def _cleanup(self):
        """Remove requests older than the window."""
        cutoff = time.time() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    async def acquire(self) -> bool:
        """
        Try to acquire a rate limit slot.
        Returns True if allowed, False if rate limit exceeded.
        """
        async with self._lock:
            self._cleanup()
            if len(self._requests) >= self.max_requests:
                logger.warning(
                    f"Rate limit exceeded: {len(self._requests)}/{self.max_requests} "
                    f"in last {self.window_seconds}s"
                )
                return False
            self._requests.append(time.time())
            return True

    async def wait_and_acquire(self, timeout: float = 30.0) -> bool:
        """
        Wait for a slot to become available.
        Returns True if acquired within timeout, False otherwise.
        """
        start = time.time()
        while time.time() - start < timeout:
            if await self.acquire():
                return True
            # Calculate wait time until oldest request expires
            async with self._lock:
                self._cleanup()
                if self._requests:
                    oldest = self._requests[0]
                    wait_time = (oldest + self.window_seconds) - time.time()
                    wait_time = max(0.05, min(wait_time, 1.0))
                else:
                    wait_time = 0.05
            await asyncio.sleep(wait_time)
        logger.error(f"Rate limit timeout after {timeout}s")
        return False

    @property
    def current_usage(self) -> int:
        self._cleanup()
        return len(self._requests)

    @property
    def available_slots(self) -> int:
        return max(0, self.max_requests - self.current_usage)


class EndpointRateLimiter:
    """
    Per-endpoint rate limiter.

    Different API endpoints may have different rate limits.
    Falls back to a default limiter if no endpoint-specific one exists.
    """

    def __init__(self, default_limit: int = 1000, window_seconds: int = 10):
        self._default = RateLimiter(default_limit, window_seconds)
        self._limiters: dict[str, RateLimiter] = {}
        self._window = window_seconds

    def add_endpoint(self, endpoint: str, max_requests: int) -> None:
        """Register a rate limit for a specific endpoint pattern."""
        self._limiters[endpoint] = RateLimiter(max_requests, self._window)

    def _get_limiter(self, endpoint: str) -> RateLimiter:
        """Get the most specific limiter for an endpoint."""
        # Check for exact match first
        if endpoint in self._limiters:
            return self._limiters[endpoint]
        # Check for prefix match
        for pattern, limiter in self._limiters.items():
            if endpoint.startswith(pattern):
                return limiter
        return self._default

    async def acquire(self, endpoint: str = "") -> bool:
        return await self._get_limiter(endpoint).acquire()

    async def wait_and_acquire(self, endpoint: str = "", timeout: float = 30.0) -> bool:
        return await self._get_limiter(endpoint).wait_and_acquire(timeout)
