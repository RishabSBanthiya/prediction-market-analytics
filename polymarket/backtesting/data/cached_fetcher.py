"""
Cached data fetcher for backtesting.

Wraps the trade fetcher and API calls with SQLite caching for fast re-runs.
Fetches data once, stores it, and serves from cache on subsequent runs.
"""

import asyncio
import aiohttp
import logging
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from .cache import BacktestDataCache, get_cache
from .trade_fetcher import TradeFetcher, TradeData
from ...core.models import Market, Token
from ...core.config import get_config

logger = logging.getLogger(__name__)


class CachedDataFetcher:
    """
    Cached data fetcher for backtesting.

    Fetches and caches:
    - Price history (hourly, 5-min, 1-min)
    - Trade data with wallet addresses
    - Market metadata
    - Orderbook snapshots

    Usage:
        fetcher = CachedDataFetcher()
        await fetcher.initialize()

        # First run: fetches from API and caches
        # Subsequent runs: serves from cache
        prices = await fetcher.get_price_history(token_id, interval='5m')
        trades = await fetcher.get_trades(token_id)

        await fetcher.close()
    """

    def __init__(
        self,
        cache: Optional[BacktestDataCache] = None,
        max_age_hours: int = 48,
        clob_client=None,
    ):
        """
        Initialize cached fetcher.

        Args:
            cache: Cache instance (uses default if not provided)
            max_age_hours: Max age before data is considered stale
            clob_client: Optional py_clob_client for authenticated requests
        """
        self.cache = cache or get_cache()
        self.max_age_hours = max_age_hours
        self._clob_client = clob_client
        self._session: Optional[aiohttp.ClientSession] = None
        self._trade_fetcher: Optional[TradeFetcher] = None
        self._config = get_config()

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0

    async def initialize(self):
        """Initialize HTTP session and trade fetcher."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

        if self._trade_fetcher is None:
            self._trade_fetcher = TradeFetcher(
                session=self._session,
                clob_client=self._clob_client
            )

    async def close(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # ==================== Price History ====================

    async def get_price_history(
        self,
        token_id: str,
        interval: str = '1h',
        min_points: int = 20,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Get price history, from cache or API.

        Args:
            token_id: Token ID
            interval: '1h', '5m', '1m', or 'tick'
            min_points: Minimum points required
            force_refresh: Force fetch from API

        Returns:
            List of {'t': timestamp, 'p': price} dicts
        """
        # Check cache first
        if not force_refresh and self.cache.has_price_history(
            token_id, interval, min_points, self.max_age_hours
        ):
            self.cache_hits += 1
            return self.cache.get_price_history(token_id, interval)

        # Fetch from API
        self.cache_misses += 1
        prices = await self._fetch_price_history_api(token_id, interval)

        if prices:
            self.cache.store_price_history(token_id, prices, interval)

        return prices

    async def _fetch_price_history_api(
        self,
        token_id: str,
        interval: str = '1h'
    ) -> List[Dict]:
        """Fetch price history from CLOB API."""
        self.api_calls += 1

        # Map interval to API parameter
        interval_map = {
            '1h': '1h',
            '5m': '5m',
            '1m': '1m',
            'tick': '1m',  # Use 1m as closest
        }
        api_interval = interval_map.get(interval, '1h')

        url = f"{self._config.clob_host}/prices-history"
        params = {
            "market": token_id,
            "interval": api_interval
        }

        try:
            async with self._session.get(url, params=params, timeout=30) as resp:
                if resp.status != 200:
                    logger.debug(f"Price history fetch failed: {resp.status}")
                    return []

                data = await resp.json()
                history = data.get('history', [])
                logger.debug(f"Fetched {len(history)} price points for {token_id[:16]}...")
                return history

        except Exception as e:
            logger.warning(f"Error fetching price history for {token_id[:16]}: {e}")
            return []

    async def get_price_history_batch(
        self,
        token_ids: List[str],
        interval: str = '1h',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get price history for multiple tokens.

        Uses cache where available, fetches missing data.
        """
        results = {}
        total = len(token_ids)
        fetched = 0

        for i, token_id in enumerate(token_ids):
            prices = await self.get_price_history(token_id, interval)
            if prices:
                results[token_id] = prices
                fetched += 1

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, total)

            # Small delay to avoid rate limiting
            if self.cache_misses > fetched * 0.5:  # More misses = more API calls
                await asyncio.sleep(0.05)

        if progress_callback:
            progress_callback(total, total)

        logger.info(
            f"Price history: {len(results)}/{total} tokens, "
            f"cache hits: {self.cache_hits}, misses: {self.cache_misses}"
        )
        return results

    # ==================== Trades ====================

    async def get_trades(
        self,
        token_id: str,
        min_trades: int = 10,
        limit: int = 500,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Get trades for a token, from cache or API.

        Args:
            token_id: Token ID
            min_trades: Minimum trades required
            limit: Max trades to fetch
            force_refresh: Force fetch from API

        Returns:
            List of trade dicts
        """
        # Check cache first
        if not force_refresh and self.cache.has_trades(
            token_id, min_trades, self.max_age_hours
        ):
            self.cache_hits += 1
            return self.cache.get_trades(token_id, limit=limit)

        # Fetch from API
        self.cache_misses += 1

        if not self._trade_fetcher:
            await self.initialize()

        trades = await self._trade_fetcher.fetch_trades_for_token(token_id, limit)

        if trades:
            # Convert TradeData to dicts for storage
            trade_dicts = []
            for t in trades:
                trade_dicts.append({
                    'id': t.trade_id,
                    'token_id': t.token_id,
                    'market_id': t.market_id,
                    'timestamp': t.timestamp,
                    'price': t.price,
                    'size': t.size,
                    'side': t.side,
                    'maker': t.maker_address,
                    'taker': t.taker_address,
                    'value_usd': t.value_usd,
                })

            self.cache.store_trades(token_id, trade_dicts)

        return self.cache.get_trades(token_id, limit=limit)

    async def get_trades_batch(
        self,
        token_ids: List[str],
        limit_per_token: int = 500,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get trades for multiple tokens.

        Uses cache where available, fetches missing data.
        """
        results = {}
        total = len(token_ids)

        for i, token_id in enumerate(token_ids):
            trades = await self.get_trades(token_id, limit=limit_per_token)
            if trades:
                results[token_id] = trades

            if progress_callback and (i + 1) % 50 == 0:
                progress_callback(i + 1, total)

            # Rate limiting for API calls
            await asyncio.sleep(0.05)

        if progress_callback:
            progress_callback(total, total)

        logger.info(
            f"Trades: {len(results)}/{total} tokens, "
            f"cache hits: {self.cache_hits}, misses: {self.cache_misses}"
        )
        return results

    # ==================== Markets ====================

    async def get_markets(
        self,
        include_resolved: bool = True,
        include_active: bool = True,
        days: int = 60,
        max_markets: int = 500,
        force_refresh: bool = False
    ) -> List[Dict]:
        """
        Get markets from cache or API.

        Args:
            include_resolved: Include resolved markets
            include_active: Include active markets
            days: How far back to look
            max_markets: Maximum markets to return
            force_refresh: Force fetch from API

        Returns:
            List of market dicts
        """
        markets = []
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

        # Check cache first
        if not force_refresh:
            if include_resolved:
                resolved = self.cache.get_resolved_markets(
                    min_end_date=cutoff, limit=max_markets
                )
                markets.extend(resolved)

            if include_active:
                active = self.cache.get_active_markets(
                    min_end_date=cutoff, limit=max_markets
                )
                markets.extend(active)

            if markets:
                self.cache_hits += 1
                logger.info(f"Loaded {len(markets)} markets from cache")
                return markets[:max_markets]

        # Fetch from API
        self.cache_misses += 1
        self.api_calls += 1

        markets = await self._fetch_markets_api(
            include_resolved=include_resolved,
            include_active=include_active,
            max_markets=max_markets
        )

        # Store in cache
        for market in markets:
            self.cache.store_market(market)

        logger.info(f"Fetched and cached {len(markets)} markets from API")
        return markets

    async def _fetch_markets_api(
        self,
        include_resolved: bool = True,
        include_active: bool = True,
        max_markets: int = 500
    ) -> List[Dict]:
        """Fetch markets from API."""
        markets = []

        # Use clob_client if available for authenticated access
        if self._clob_client:
            try:
                # Fetch sampling markets (active, liquid)
                cursor = 'MA=='
                while cursor and len(markets) < max_markets:
                    result = self._clob_client.get_sampling_simplified_markets(
                        next_cursor=cursor
                    )
                    data = result.get('data', [])
                    for m in data:
                        m['resolved'] = False  # Sampling markets are active
                    markets.extend(data)
                    cursor = result.get('next_cursor')

                    if not data:
                        break

                logger.info(f"Fetched {len(markets)} active markets via clob_client")

            except Exception as e:
                logger.warning(f"Error fetching markets via clob_client: {e}")

        # Also try to fetch closed/resolved markets
        if include_resolved:
            try:
                url = f"{self._config.clob_host}/markets"
                params = {
                    "closed": "true",
                    "limit": min(100, max_markets - len(markets))
                }

                async with self._session.get(url, params=params, timeout=30) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        resolved_markets = data if isinstance(data, list) else data.get('data', [])
                        for m in resolved_markets:
                            m['resolved'] = True
                        markets.extend(resolved_markets)
                        logger.info(f"Fetched {len(resolved_markets)} resolved markets")

            except Exception as e:
                logger.debug(f"Error fetching resolved markets: {e}")

        return markets[:max_markets]

    # ==================== Batch Operations ====================

    async def prefetch_all_data(
        self,
        token_ids: List[str],
        price_interval: str = '5m',
        fetch_trades: bool = True,
        progress_callback: Optional[callable] = None
    ):
        """
        Prefetch all data for a list of tokens.

        Useful for warming the cache before optimization.

        Args:
            token_ids: Tokens to prefetch
            price_interval: Price data interval
            fetch_trades: Whether to also fetch trade data
            progress_callback: Optional progress callback
        """
        total_steps = len(token_ids) * (2 if fetch_trades else 1)
        current = 0

        def update_progress():
            nonlocal current
            current += 1
            if progress_callback:
                progress_callback(current, total_steps)

        # Fetch prices
        logger.info(f"Prefetching price history for {len(token_ids)} tokens...")
        for token_id in token_ids:
            await self.get_price_history(token_id, price_interval)
            update_progress()

        # Fetch trades
        if fetch_trades:
            logger.info(f"Prefetching trades for {len(token_ids)} tokens...")
            for token_id in token_ids:
                await self.get_trades(token_id)
                update_progress()

        logger.info(
            f"Prefetch complete. Cache hits: {self.cache_hits}, "
            f"misses: {self.cache_misses}, API calls: {self.api_calls}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        cache_stats = self.cache.get_stats()
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0,
            **cache_stats
        }


async def warmup_cache(
    max_markets: int = 200,
    price_interval: str = '5m',
    fetch_trades: bool = True,
    clob_client=None
):
    """
    Warmup the cache with fresh data.

    Useful to run before optimization to ensure cache is populated.

    Args:
        max_markets: Max markets to fetch
        price_interval: Price data interval
        fetch_trades: Whether to fetch trade data
        clob_client: Optional authenticated clob client
    """
    logger.info("Starting cache warmup...")

    async with CachedDataFetcher(clob_client=clob_client) as fetcher:
        # Fetch markets
        markets = await fetcher.get_markets(
            include_resolved=True,
            include_active=True,
            max_markets=max_markets,
            force_refresh=True
        )

        # Extract token IDs
        token_ids = []
        for market in markets:
            tokens = market.get('tokens', [])
            for token in tokens:
                token_id = token.get('token_id')
                if token_id:
                    token_ids.append(token_id)

        logger.info(f"Found {len(token_ids)} tokens across {len(markets)} markets")

        # Prefetch all data
        await fetcher.prefetch_all_data(
            token_ids=token_ids,
            price_interval=price_interval,
            fetch_trades=fetch_trades,
            progress_callback=lambda c, t: logger.info(f"Progress: {c}/{t}")
            if c % 100 == 0 else None
        )

        stats = fetcher.get_stats()
        logger.info(f"Cache warmup complete: {stats}")

        return stats
