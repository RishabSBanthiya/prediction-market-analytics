"""
Async Polymarket API client.

Provides a unified interface for interacting with:
- Gamma API (market data)
- CLOB API (orderbook, price history)
- Data API (positions, activity)
- Blockchain (USDC balance)
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .models import (
    Market,
    Token,
    Position,
    Trade,
    HistoricalPrice,
    OrderbookSnapshot,
    PositionStatus,
    Side,
)
from .config import Config, get_config
from .rate_limiter import InMemoryRateLimiter

logger = logging.getLogger(__name__)


class PolymarketAPI:
    """
    Async Polymarket API client.
    
    Handles all API interactions with built-in rate limiting.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_window=self.config.risk.api_rate_limit_per_10s,
            window_seconds=self.config.risk.api_rate_limit_window_seconds
        )
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def connect(self):
        """Initialize HTTP session"""
        if self.session is None or self.session.closed:
            import ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _get(
        self, 
        url: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> Optional[Any]:
        """Make a rate-limited GET request"""
        # Ensure session is initialized
        if self.session is None or self.session.closed:
            await self.connect()
        
        if not await self.rate_limiter.wait_and_acquire("api", url, timeout=5.0):
            logger.error(f"Rate limit exceeded for {url}")
            return None
        
        try:
            async with self.session.get(
                url, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"API error {resp.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    # ==================== MARKET DATA ====================
    
    async def fetch_markets_batch(
        self, 
        offset: int = 0, 
        limit: int = 100,
        closed: bool = False,
        active: bool = True,
        retries: int = 3
    ) -> List[dict]:
        """Fetch a batch of markets from Gamma API with retry logic"""
        params = {
            "closed": str(closed).lower(),
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
        }
        
        # Use longer timeout for market fetching (can be slow with large datasets)
        timeout = 30.0
        
        for attempt in range(retries):
            data = await self._get(
                f"{self.config.gamma_api_base}/markets", 
                params,
                timeout=timeout
            )
            
            if data is not None:
                return data if isinstance(data, list) else []
            
            # If this wasn't the last attempt, wait before retrying
            if attempt < retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                logger.debug(f"Retrying market batch fetch (offset={offset}, attempt={attempt+1}/{retries}) after {wait_time}s")
                await asyncio.sleep(wait_time)
        
        logger.warning(f"Failed to fetch market batch after {retries} attempts (offset={offset})")
        return []
    
    async def fetch_all_markets(
        self, 
        closed: bool = False,
        active: bool = True,
        max_markets: int = 10000,
        max_concurrent: int = 10
    ) -> List[dict]:
        """
        Fetch all markets with controlled concurrency to avoid timeouts.
        
        Args:
            closed: Whether to fetch closed markets
            active: Whether to fetch active markets
            max_markets: Maximum number of markets to fetch
            max_concurrent: Maximum number of concurrent requests (default: 10)
        """
        # Fetch first batch to determine total count
        first_batch = await self.fetch_markets_batch(0, 100, closed, active)
        if not first_batch:
            logger.warning("Failed to fetch first batch of markets")
            return []
        
        all_markets = first_batch
        
        # If first batch is less than limit, we're done
        if len(first_batch) < 100:
            logger.info(f"Fetched {len(all_markets)} markets (single batch)")
            return all_markets
        
        # Calculate number of remaining batches
        remaining_batches = (max_markets - 100) // 100
        
        # Fetch remaining batches with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(offset: int):
            async with semaphore:
                return await self.fetch_markets_batch(offset, 100, closed, active)
        
        # Create tasks for remaining batches
        tasks = [
            fetch_with_semaphore(offset) 
            for offset in range(100, max_markets, 100)
        ]
        
        # Execute with controlled concurrency
        logger.info(f"Fetching {len(tasks)} additional batches (max {max_concurrent} concurrent)")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_batches = 0
        failed_batches = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch {i+1} failed: {result}")
                failed_batches += 1
                # Stop if we get too many failures in a row
                if failed_batches >= 3:
                    logger.warning(f"Too many consecutive failures, stopping at batch {i+1}")
                    break
            elif result:
                all_markets.extend(result)
                successful_batches += 1
                failed_batches = 0  # Reset failure counter on success
            else:
                # Empty result - likely reached end
                break
        
        logger.info(
            f"Market fetch complete: {len(all_markets)} total markets "
            f"({successful_batches} successful batches, {failed_batches} failed)"
        )

        return all_markets

    async def fetch_15min_markets(
        self,
        window_count: int = 12,  # 3 hours worth of 15-min windows
        include_past: int = 2,   # Include 2 past windows (30 min)
        cryptos: List[str] = None,  # Which cryptos to fetch
    ) -> List[dict]:
        """
        Fetch 15-minute crypto up/down markets.

        These markets are restricted and don't appear in standard market listings.
        They use a predictable slug pattern: {crypto}-updown-15m-{unix_timestamp}

        Args:
            window_count: Number of 15-min windows to fetch ahead
            include_past: Number of past windows to include (for recently expired)
            cryptos: List of crypto symbols to fetch (default: btc, eth, sol, xrp)

        Returns:
            List of raw market dicts
        """
        if cryptos is None:
            cryptos = ["btc", "eth", "sol", "xrp"]

        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())

        # Round to 15-minute boundary
        window_start = (current_ts // 900) * 900

        markets = []

        # Create tasks for parallel fetching
        async def fetch_window(crypto: str, window_ts: int) -> Optional[dict]:
            slug = f"{crypto}-updown-15m-{window_ts}"
            url = f"{self.config.gamma_api_base}/events"
            params = {"slug": slug}

            if not await self.rate_limiter.wait_and_acquire("api", url, timeout=5.0):
                return None

            try:
                async with self.session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10.0)
                ) as resp:
                    if resp.status == 200:
                        events = await resp.json()
                        if events and events[0].get("markets"):
                            return events[0]["markets"][0]
            except Exception as e:
                logger.debug(f"Failed to fetch 15-min market {slug}: {e}")

            return None

        # Calculate window timestamps
        window_timestamps = [
            window_start + (i * 900)
            for i in range(-include_past, window_count)
        ]

        # Fetch all cryptos and windows in parallel
        tasks = [
            fetch_window(crypto, ts)
            for crypto in cryptos
            for ts in window_timestamps
        ]
        results = await asyncio.gather(*tasks)

        # Collect valid results
        for result in results:
            if result:
                markets.append(result)

        if markets:
            logger.info(f"Fetched {len(markets)} 15-minute crypto markets ({', '.join(c.upper() for c in cryptos)})")

        return markets

    async def fetch_sports_games(
        self,
        days_ahead: int = 2,
        sports: List[str] = None,
    ) -> List[dict]:
        """
        Fetch restricted sports game markets.

        Sports games are restricted and use slug pattern: {sport}-{team1}-{team2}-{date}

        Args:
            days_ahead: Number of days to look ahead (default 2)
            sports: List of sports to fetch (default: ['nba'])

        Returns:
            List of raw market dicts from sports games
        """
        if sports is None:
            sports = ["nba"]

        now = datetime.now(timezone.utc)

        # Team codes by sport
        team_codes = {
            "nba": [
                "atl", "bos", "bkn", "cha", "chi", "cle", "dal", "den", "det", "gsw",
                "hou", "ind", "lac", "lal", "mem", "mia", "mil", "min", "nop", "nyk",
                "okc", "orl", "phi", "phx", "por", "sac", "sas", "tor", "uta", "was"
            ],
            "nfl": [
                "ari", "atl", "bal", "buf", "car", "chi", "cin", "cle", "dal", "den",
                "det", "gb", "hou", "ind", "jax", "kc", "lv", "lac", "lar", "mia",
                "min", "ne", "no", "nyg", "nyj", "phi", "pit", "sf", "sea", "tb", "ten", "was"
            ],
            "nhl": [
                "ana", "ari", "bos", "buf", "cgy", "car", "chi", "col", "cbj", "dal",
                "det", "edm", "fla", "la", "min", "mtl", "nsh", "nj", "nyi", "nyr",
                "ott", "phi", "pit", "sj", "sea", "stl", "tb", "tor", "van", "vgk", "wsh", "wpg"
            ],
        }

        markets = []
        found_slugs = set()

        # Generate dates to check
        dates = [(now + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(days_ahead + 1)]

        async def fetch_game(sport: str, team1: str, team2: str, date: str) -> List[dict]:
            """Try to fetch a game by slug."""
            results = []
            for slug in [f"{sport}-{team1}-{team2}-{date}", f"{sport}-{team2}-{team1}-{date}"]:
                if slug in found_slugs:
                    continue

                if not await self.rate_limiter.wait_and_acquire("api", slug, timeout=2.0):
                    continue

                try:
                    async with self.session.get(
                        f"{self.config.gamma_api_base}/events",
                        params={"slug": slug},
                        timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as resp:
                        if resp.status == 200:
                            events = await resp.json()
                            if events and events[0].get("markets"):
                                found_slugs.add(slug)
                                # Add all markets from this game
                                for m in events[0]["markets"]:
                                    results.append(m)
                                logger.debug(f"Found sports game: {slug} ({len(events[0]['markets'])} markets)")
                except Exception as e:
                    logger.debug(f"Failed to fetch sports game {slug}: {e}")

            return results

        # Fetch games for each sport
        for sport in sports:
            teams = team_codes.get(sport, [])
            if not teams:
                continue

            # Create tasks for all team combinations
            tasks = []
            for date in dates:
                for i, team1 in enumerate(teams):
                    for team2 in teams[i + 1:]:
                        tasks.append(fetch_game(sport, team1, team2, date))

            # Execute in batches to avoid overwhelming the API
            batch_size = 50
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                results = await asyncio.gather(*batch)
                for result in results:
                    markets.extend(result)

        if markets:
            logger.info(f"Fetched {len(markets)} sports game markets from {len(found_slugs)} games")

        return markets

    async def fetch_all_markets_including_restricted(
        self,
        closed: bool = False,
        active: bool = True,
        max_markets: int = 10000,
        include_15min: bool = True,
        include_sports: bool = True,
        sports: List[str] = None,
    ) -> List[dict]:
        """
        Fetch all markets including restricted markets (15-min crypto, sports games).

        Args:
            closed: Whether to fetch closed markets
            active: Whether to fetch active markets
            max_markets: Maximum number of standard markets to fetch
            include_15min: Whether to include restricted 15-min BTC markets
            include_sports: Whether to include restricted sports game markets
            sports: List of sports to include (default: ['nba'])

        Returns:
            Combined list of all markets
        """
        if sports is None:
            sports = ["nba"]

        # Fetch standard markets
        all_markets = await self.fetch_all_markets(
            closed=closed,
            active=active,
            max_markets=max_markets,
        )

        existing_ids = {m.get("conditionId") or m.get("condition_id") for m in all_markets}

        def add_markets(new_markets: List[dict]):
            """Add markets, deduplicating by condition_id."""
            for market in new_markets:
                market_id = market.get("conditionId") or market.get("condition_id")
                if market_id and market_id not in existing_ids:
                    all_markets.append(market)
                    existing_ids.add(market_id)

        # Fetch restricted 15-min crypto markets
        if include_15min and not closed:
            restricted_15min = await self.fetch_15min_markets()
            add_markets(restricted_15min)

        # Fetch restricted sports game markets
        if include_sports and not closed:
            sports_markets = await self.fetch_sports_games(sports=sports)
            add_markets(sports_markets)

        return all_markets

    async def fetch_closed_markets(self, days: int = 7, resolved_only: bool = True) -> List[dict]:
        """
        Fetch recently closed/resolved markets.
        
        Args:
            days: Number of days to look back
            resolved_only: If True, only fetch markets that are both closed AND resolved
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_markets = []
        offset = 0
        limit = 100
        consecutive_old_batches = 0  # Track consecutive batches with no valid markets
        max_consecutive_old = 3  # Stop after 3 consecutive batches with no valid markets
        
        logger.info(f"Fetching closed markets from last {days} days (cutoff: {cutoff.isoformat()}, resolved_only={resolved_only})")
        
        while True:
            params = {
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "order": "endDate",       # Sort by end date
                "ascending": "false",      # Most recent first
            }
            
            # Add resolved filter if requested
            if resolved_only:
                params["resolved"] = "true"
            
            data = await self._get(f"{self.config.gamma_api_base}/markets", params)
            if not data:
                logger.debug(f"No data returned at offset {offset}, stopping")
                break
            
            batch_valid_count = 0
            batch_old_count = 0
            batch_no_closed_time = 0
            
            # Filter by closed time
            for market in data:
                closed_time = None
                closed_time_str = market.get("closedTime")
                
                # Try to parse closedTime first
                if closed_time_str:
                    try:
                        # Handle different timestamp formats
                        closed_time_str = str(closed_time_str).replace("Z", "+00:00")
                        if " " in closed_time_str and "+" in closed_time_str:
                            # Format: "2025-12-18 14:14:02+00"
                            closed_time_str = closed_time_str.replace(" ", "T").replace("+00", "+00:00")
                        closed_time = datetime.fromisoformat(closed_time_str)
                    except ValueError as e:
                        logger.debug(f"Failed to parse closedTime: {closed_time_str}, {e}")
                        closed_time = None
                
                # Fallback to endDate if closedTime is not available
                if closed_time is None:
                    end_date_str = (
                        market.get("end_date_iso") or 
                        market.get("endDate") or 
                        market.get("end_date")
                    )
                    if end_date_str:
                        try:
                            end_date_str = str(end_date_str).replace("Z", "+00:00")
                            if "T" not in end_date_str:
                                end_date_str = end_date_str + "T00:00:00+00:00"
                            closed_time = datetime.fromisoformat(end_date_str)
                        except ValueError as e:
                            logger.debug(f"Failed to parse endDate: {end_date_str}, {e}")
                            closed_time = None
                
                # If we have a valid date, check against cutoff
                if closed_time is not None:
                    if closed_time >= cutoff:
                        all_markets.append(market)
                        batch_valid_count += 1
                    else:
                        batch_old_count += 1
                else:
                    # No date available - include if marked as closed/resolved
                    batch_no_closed_time += 1
                    if market.get("closed", False) or market.get("resolved", False):
                        all_markets.append(market)
                        batch_valid_count += 1
            
            if offset % 500 == 0:
                logger.info(
                    f"Offset {offset}: Found {batch_valid_count} valid, {batch_old_count} old, "
                    f"{batch_no_closed_time} no closedTime (total: {len(all_markets)})"
                )
            
            # If we found valid markets, reset the consecutive old counter
            if batch_valid_count > 0:
                consecutive_old_batches = 0
            else:
                consecutive_old_batches += 1
                # If we've had multiple consecutive batches with no valid markets,
                # and we're seeing old markets, we've likely passed the cutoff
                if consecutive_old_batches >= max_consecutive_old and batch_old_count > 0:
                    logger.info(f"Stopping: {consecutive_old_batches} consecutive batches with no valid markets")
                    break
            
            if len(data) < limit:
                logger.debug(f"Received {len(data)} markets (less than limit {limit}), stopping")
                break
            
            offset += limit
            
            # Increased safety limit to allow fetching more markets
            if offset >= 10000:
                logger.warning(f"Reached safety limit at offset {offset}, stopping")
                break
        
        logger.info(f"Fetched {len(all_markets)} closed markets total")
        return all_markets
    
    def parse_market(self, raw_market: dict) -> Optional[Market]:
        """Parse raw market data into Market object"""
        now = datetime.now(timezone.utc)

        # Parse end date - check multiple fields to find the best one
        # For active markets: closedTime is usually not set, use endDate
        # For closed markets: closedTime is when it actually closed
        end_date = None
        time_source = None

        # Get all available time fields for debugging
        closed_time_str = raw_market.get("closedTime")
        end_date_iso = raw_market.get("end_date_iso")  # Often just date, no time!
        end_date_field = raw_market.get("endDate") or raw_market.get("end_date")
        game_start_time = raw_market.get("gameStartTime")  # For sports markets

        # IMPORTANT: Prefer endDate over end_date_iso for short-term markets!
        # end_date_iso is often just "2026-01-08" without time
        # endDate has full timestamp like "2026-01-08T04:15:00Z"
        # For 15-minute markets, the time component is critical!

        # Try endDate first (has full timestamp with time)
        if end_date_field and "T" in str(end_date_field):
            try:
                end_str = str(end_date_field).replace("Z", "+00:00")
                end_date = datetime.fromisoformat(end_str)
                time_source = "endDate"
            except Exception as e:
                logger.debug(f"Failed to parse endDate '{end_date_field}': {e}")

        # Fallback to end_date_iso only if endDate wasn't usable
        if end_date is None and end_date_iso:
            try:
                end_str = str(end_date_iso).replace("Z", "+00:00")
                if "T" not in end_str:
                    # Date only - assume end of day in UTC
                    end_str = end_str + "T23:59:59+00:00"
                end_date = datetime.fromisoformat(end_str)
                time_source = "end_date_iso"
            except Exception as e:
                logger.debug(f"Failed to parse end_date_iso '{end_date_iso}': {e}")

        # Last fallback to endDate without T (rare)
        if end_date is None and end_date_field:
            try:
                end_str = str(end_date_field).replace("Z", "+00:00")
                if "T" not in end_str:
                    end_str = end_str + "T23:59:59+00:00"
                end_date = datetime.fromisoformat(end_str)
                time_source = "endDate_fallback"
            except Exception as e:
                logger.debug(f"Failed to parse endDate fallback '{end_date_field}': {e}")

        # If closedTime is set (market already closed), use it instead
        if closed_time_str:
            try:
                ct_str = str(closed_time_str).replace("Z", "+00:00")
                if " " in ct_str and "+" in ct_str:
                    # Format: "2025-12-18 14:14:02+00"
                    ct_str = ct_str.replace(" ", "T")
                    if ct_str.endswith("+00"):
                        ct_str = ct_str + ":00"
                closed_time = datetime.fromisoformat(ct_str)
                # Only use closedTime if it's in the future or recent past
                # (for active markets being checked)
                if closed_time > now - timedelta(hours=1):
                    end_date = closed_time
                    time_source = "closedTime"
            except Exception as e:
                logger.debug(f"Failed to parse closedTime '{closed_time_str}': {e}")

        if end_date is None:
            return None

        # Log short-term markets for debugging
        seconds_left = (end_date - now).total_seconds()
        if 0 < seconds_left < 3600:  # Less than 1 hour
            question = raw_market.get("question", "")[:60]
            logger.debug(
                f"Short-term market found: {question}... | "
                f"seconds_left={seconds_left:.0f} | source={time_source} | "
                f"closedTime={closed_time_str} | endDate={end_date_field}"
            )
        
        # Parse tokens
        tokens = []
        clob_token_ids = raw_market.get("clobTokenIds", [])
        outcomes = raw_market.get("outcomes", [])
        prices = raw_market.get("outcomePrices", [])
        
        # Handle string JSON
        if isinstance(clob_token_ids, str):
            import json
            try:
                clob_token_ids = json.loads(clob_token_ids)
            except:
                clob_token_ids = []
        
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except:
                outcomes = []
        
        if isinstance(prices, str):
            import json
            try:
                prices = json.loads(prices)
            except:
                prices = []
        
        for i, token_id in enumerate(clob_token_ids):
            outcome = outcomes[i] if i < len(outcomes) else f"Outcome {i}"
            price = float(prices[i]) if i < len(prices) and prices[i] else 0.0
            
            tokens.append(Token(
                token_id=token_id,
                outcome=str(outcome),
                price=price
            ))
        
        if not tokens:
            return None
        
        # Parse start date
        start_date = None
        start_str = raw_market.get("startDate") or raw_market.get("createdAt")
        if start_str:
            try:
                start_str = str(start_str).replace("Z", "+00:00")
                start_date = datetime.fromisoformat(start_str)
            except:
                pass
        
        # Determine winning outcome from outcomePrices
        # Winner has price = "1" or "1.0", loser has price = "0" or "0.0"
        # Note: The Gamma API may have a lag where `resolved=False` even when
        # prices are already set to 0/1. We detect resolution from prices directly.
        winning_outcome = None
        is_resolved = raw_market.get("resolved", False)
        is_closed = raw_market.get("closed", False)
        
        # First check if there's an explicit winningOutcome field
        winning_outcome = raw_market.get("winningOutcome") or raw_market.get("winning_outcome")
        
        # If not, determine from outcomePrices
        # Check if prices show clear resolution (exactly 0 and 1)
        if not winning_outcome and prices:
            parsed_prices = []
            for price_val in prices:
                try:
                    parsed_prices.append(float(price_val))
                except (ValueError, TypeError):
                    parsed_prices.append(None)
            
            # Check if this is a resolved market (one price is ~1.0, others are ~0.0)
            has_winner = any(p is not None and p >= 0.99 for p in parsed_prices)
            has_loser = any(p is not None and p <= 0.01 for p in parsed_prices)
            
            if has_winner and has_loser:
                # This is a resolved market - find the winner
                for i, price_float in enumerate(parsed_prices):
                    if price_float is not None and price_float >= 0.99 and i < len(tokens):
                        winning_outcome = tokens[i].outcome
                        is_resolved = True  # Override the API's resolved flag
                        break
        
        # Extract fee and order book fields for maker rebates
        fees_enabled = raw_market.get("feesEnabled") or raw_market.get("fees_enabled") or False
        enable_order_book = raw_market.get("enableOrderBook") or raw_market.get("enable_order_book") or False
        accepting_orders = raw_market.get("acceptingOrders") or raw_market.get("accepting_orders") or False
        taker_base_fee = raw_market.get("takerBaseFee") or raw_market.get("taker_base_fee")
        maker_base_fee = raw_market.get("makerBaseFee") or raw_market.get("maker_base_fee")

        return Market(
            condition_id=raw_market.get("conditionId") or raw_market.get("condition_id") or "",
            question=raw_market.get("question") or "",
            slug=raw_market.get("market_slug") or raw_market.get("slug") or "",
            end_date=end_date,
            start_date=start_date,
            category=raw_market.get("category"),
            tokens=tokens,
            closed=is_closed,
            resolved=is_resolved,
            winning_outcome=winning_outcome,
            fees_enabled=bool(fees_enabled),
            enable_order_book=bool(enable_order_book),
            accepting_orders=bool(accepting_orders),
            taker_base_fee=int(taker_base_fee) if taker_base_fee else None,
            maker_base_fee=int(maker_base_fee) if maker_base_fee else None,
        )
    
    # ==================== ORDERBOOK ====================
    
    async def fetch_orderbook(self, token_id: str) -> Optional[OrderbookSnapshot]:
        """Fetch current orderbook for a token"""
        data = await self._get(f"{self.config.clob_host}/book", {"token_id": token_id})
        
        if not data:
            return None
        
        try:
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            bid_depth = [(float(b.get("price", 0)), float(b.get("size", 0))) for b in bids]
            ask_depth = [(float(a.get("price", 0)), float(a.get("size", 0))) for a in asks]
            
            # Sort orderbook properly:
            # - Bids: descending by price (highest/best bid first)
            # - Asks: ascending by price (lowest/best ask first)
            # The Polymarket API may return them in the opposite order
            bid_depth = sorted(bid_depth, key=lambda x: x[0], reverse=True)
            ask_depth = sorted(ask_depth, key=lambda x: x[0], reverse=False)
            
            best_bid = bid_depth[0][0] if bid_depth else None
            best_ask = ask_depth[0][0] if ask_depth else None
            bid_size = bid_depth[0][1] if bid_depth else 0.0
            ask_size = ask_depth[0][1] if ask_depth else 0.0
            
            return OrderbookSnapshot(
                token_id=token_id,
                timestamp=datetime.now(timezone.utc),
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
            )
        except Exception as e:
            logger.error(f"Error parsing orderbook: {e}")
            return None
    
    async def get_spread(self, token_id: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get bid, ask, and spread for a token.
        
        Returns: (best_bid, best_ask, spread_pct)
        """
        orderbook = await self.fetch_orderbook(token_id)
        
        if not orderbook or not orderbook.best_bid or not orderbook.best_ask:
            return None, None, None
        
        spread_pct = orderbook.spread_pct or 0.0
        return orderbook.best_bid, orderbook.best_ask, spread_pct
    
    # ==================== PRICE HISTORY ====================
    
    async def fetch_price_history(
        self, 
        token_id: str, 
        interval: str = "max",
        fidelity: int = 1
    ) -> List[HistoricalPrice]:
        """Fetch historical price data for a token"""
        params = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }
        
        data = await self._get(f"{self.config.clob_host}/prices-history", params)
        
        if not data or not isinstance(data, dict):
            return []
        
        history = data.get("history", [])
        prices = []
        
        for point in history:
            try:
                ts = int(point.get("t", 0))
                price = float(point.get("p", 0))
                if ts > 0 and 0 <= price <= 1:
                    prices.append(HistoricalPrice(timestamp=ts, price=price))
            except (ValueError, TypeError):
                continue
        
        return sorted(prices, key=lambda x: x.timestamp)
    
    async def fetch_price_history_range(
        self, 
        token_id: str, 
        start_ts: int, 
        end_ts: int,
        fidelity: int = 1
    ) -> List[HistoricalPrice]:
        """Fetch historical price data for a specific time range"""
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }
        
        data = await self._get(f"{self.config.clob_host}/prices-history", params)
        
        if not data or not isinstance(data, dict):
            return []
        
        history = data.get("history", [])
        prices = []
        
        for point in history:
            try:
                ts = int(point.get("t", 0))
                price = float(point.get("p", 0))
                if ts > 0 and 0 <= price <= 1:
                    prices.append(HistoricalPrice(timestamp=ts, price=price))
            except (ValueError, TypeError):
                continue
        
        return sorted(prices, key=lambda x: x.timestamp)
    
    # ==================== POSITIONS ====================
    
    async def fetch_positions(self, wallet_address: str) -> List[Position]:
        """Fetch current positions for a wallet"""
        data = await self._get(
            f"{self.config.data_api_base}/positions",
            {"user": wallet_address}
        )
        
        if not data or not isinstance(data, list):
            return []
        
        positions = []
        for pos in data:
            try:
                token_id = pos.get("asset") or pos.get("tokenId") or pos.get("token_id", "")
                shares = float(pos.get("size", 0))

                if not token_id or shares <= 0:
                    continue

                # Skip resolved positions (redeemable with zero value)
                is_redeemable = pos.get("redeemable", False)
                current_value = float(pos.get("currentValue", 0))
                if is_redeemable and current_value == 0:
                    logger.debug(f"Skipping resolved position: {pos.get('title', token_id[:16])}")
                    continue

                current_price = None
                if pos.get("curPrice") is not None:
                    try:
                        current_price = float(pos.get("curPrice"))
                    except (ValueError, TypeError):
                        pass

                entry_price = None
                if pos.get("avgPrice") is not None:
                    try:
                        entry_price = float(pos.get("avgPrice"))
                    except (ValueError, TypeError):
                        pass

                positions.append(Position(
                    token_id=token_id,
                    market_id=pos.get("conditionId", ""),
                    outcome=pos.get("outcome", ""),
                    shares=shares,
                    entry_price=entry_price or current_price or 0.0,
                    current_price=current_price,
                    status=PositionStatus.OPEN,
                ))
            except Exception as e:
                logger.warning(f"Error parsing position: {e}")
                continue

        return positions
    
    # ==================== USDC BALANCE ====================
    
    async def fetch_usdc_balance(self, wallet_address: str) -> float:
        """
        Fetch USDC balance from blockchain.
        
        Note: This uses web3 which is synchronous, so it's wrapped in executor.
        """
        try:
            from web3 import Web3
        except ImportError:
            logger.warning("web3 not available, cannot query blockchain balance")
            return 0.0
        
        try:
            loop = asyncio.get_event_loop()
            balance = await loop.run_in_executor(
                None,
                self._get_usdc_balance_sync,
                wallet_address
            )
            return balance
        except Exception as e:
            logger.error(f"Error fetching USDC balance: {e}")
            return 0.0
    
    def _get_usdc_balance_sync(self, wallet_address: str) -> float:
        """Synchronous USDC balance fetch (called in executor)"""
        from web3 import Web3
        
        # Try multiple RPC endpoints
        rpc_urls = [
            self.config.polygon_rpc_url,
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
        ]
        
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if not w3.is_connected():
                    continue
                
                # USDC contract ABI (just balanceOf)
                abi = [{
                    "constant": True,
                    "inputs": [{"name": "_owner", "type": "address"}],
                    "name": "balanceOf",
                    "outputs": [{"name": "balance", "type": "uint256"}],
                    "type": "function"
                }]
                
                contract = w3.eth.contract(
                    address=Web3.to_checksum_address(self.config.usdc_contract_address),
                    abi=abi
                )
                
                balance_raw = contract.functions.balanceOf(
                    Web3.to_checksum_address(wallet_address)
                ).call()
                
                # USDC has 6 decimals
                return balance_raw / 1_000_000
                
            except Exception as e:
                logger.debug(f"RPC {rpc_url} failed: {e}")
                continue
        
        logger.error("All RPC endpoints failed")
        return 0.0
    
    # ==================== ACTIVITY FEED ====================
    
    async def fetch_activity(
        self,
        limit: int = 100,
        user: Optional[str] = None,
        offset: int = 0
    ) -> List[dict]:
        """Fetch activity feed, optionally filtered by user with pagination support"""
        params = {"limit": limit}
        if user:
            params["user"] = user
        if offset > 0:
            params["offset"] = offset
        # Use longer timeout for activity endpoint as it can be slow with large limits
        data = await self._get(
            f"{self.config.data_api_base}/activity",
            params,
            timeout=30.0  # Increased from default 10s
        )

        return data if isinstance(data, list) else []
    
    async def fetch_trades(
        self, 
        token_id: str, 
        limit: int = 100,
        market_id: Optional[str] = None
    ) -> List[Trade]:
        """
        Fetch recent trades for a token.
        
        Uses the public Data API activity endpoint (no auth required).
        The CLOB /data/trades endpoint requires L2 Header authentication,
        so we use the public activity feed for unauthenticated requests.
        
        For authenticated requests with py_clob_client, use get_trades() directly.
        See: https://docs.polymarket.com/developers/CLOB/trades/trades
        """
        trades = []
        
        # Use Data API activity endpoint (public, no auth required)
        # The CLOB /data/trades endpoint requires L2 authentication
        try:
            params = {"limit": limit}
            
            # The activity endpoint filters by condition_id (market), not asset_id directly
            # But we can filter the results after fetching
            data = await self._get(
                f"{self.config.data_api_base}/activity",
                params
            )
            
            if data and isinstance(data, list):
                for item in data:
                    try:
                        if item.get("type", "").upper() != "TRADE":
                            continue
                        
                        # Filter by token_id if provided
                        item_asset = str(item.get("asset", ""))
                        if token_id and item_asset != token_id:
                            continue
                        
                        # Filter by market_id if provided
                        item_market = item.get("conditionId", "")
                        if market_id and item_market != market_id:
                            continue
                        
                        # Parse timestamp (Unix timestamp from API)
                        timestamp = datetime.now(timezone.utc)
                        ts = item.get("timestamp")
                        if ts:
                            if isinstance(ts, (int, float)):
                                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                            else:
                                try:
                                    timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                                except:
                                    pass
                        
                        trades.append(Trade(
                            trade_id=item.get("transactionHash", ""),
                            market_id=item.get("conditionId", ""),
                            token_id=item_asset,
                            side=Side.BUY if item.get("side", "").upper() == "BUY" else Side.SELL,
                            shares=float(item.get("size", 0)),
                            price=float(item.get("price", 0)),
                            timestamp=timestamp,
                            maker_address=item.get("proxyWallet"),
                            taker_address=item.get("proxyWallet"),
                        ))
                    except Exception as e:
                        logger.debug(f"Error parsing activity trade: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Data API activity fetch failed: {e}")
        
        return trades
    
    async def fetch_user_trades(
        self,
        wallet_address: str,
        limit: int = 1000,
        offset: int = 0
    ) -> List[dict]:
        """
        Fetch trades for a user from Polymarket with pagination support.
        This includes both buy and sell trades.

        Note: The CLOB /trades endpoint requires L2 Header authentication,
        so we use the public Data API activity feed instead.
        For authenticated trade fetching, use py_clob_client directly.

        Args:
            wallet_address: Wallet address to fetch trades for
            limit: Maximum number of trades to fetch
            offset: Number of trades to skip (for pagination)
        """
        trades = []

        # Use Data API activity feed filtered by user
        try:
            activity = await self.fetch_activity(limit=limit, user=wallet_address, offset=offset)
            for item in activity:
                item_type = item.get("type", "").lower()
                if item_type in ["trade", "fill", "order", "fillorder"]:
                    try:
                        token_id = item.get("asset") or item.get("tokenId") or item.get("token_id") or item.get("asset_id")
                        if not token_id:
                            logger.debug(f"Skipping trade with no token_id: {item.get('id')}")
                            continue
                        
                        side = item.get("side", "").upper()
                        if not side:
                            # Determine side from maker/taker
                            maker = item.get("maker", "").lower() or item.get("maker_address", "").lower()
                            if maker == wallet_address.lower():
                                side = "SELL"
                            else:
                                side = "BUY"
                        
                        shares = float(item.get("size", 0) or item.get("shares", 0) or item.get("amount", 0))
                        price = float(item.get("price", 0) or item.get("fillPrice", 0) or item.get("fill_price", 0))
                        
                        if shares <= 0:
                            logger.debug(f"Skipping trade with shares <= 0: {shares}")
                            continue
                        if price <= 0:
                            logger.debug(f"Skipping trade with price <= 0: {price}")
                            continue
                        
                        timestamp = datetime.now(timezone.utc)
                        if item.get("createdAt"):
                            try:
                                timestamp = datetime.fromisoformat(
                                    item.get("createdAt").replace("Z", "+00:00")
                                )
                            except:
                                pass
                        elif item.get("timestamp"):
                            try:
                                ts = item.get("timestamp")
                                if isinstance(ts, (int, float)):
                                    # Unix timestamp
                                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                                else:
                                    # ISO format string
                                    timestamp = datetime.fromisoformat(
                                        str(ts).replace("Z", "+00:00")
                                    )
                            except:
                                pass
                        
                        trades.append({
                            "trade_id": item.get("id", ""),
                            "market_id": item.get("conditionId", "") or item.get("market", ""),
                            "token_id": token_id,
                            "side": side,
                            "shares": shares,
                            "price": price,
                            "timestamp": timestamp,
                            "maker_address": item.get("maker") or item.get("maker_address"),
                            "taker_address": item.get("taker") or item.get("taker_address"),
                            "type": "trade"
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing activity trade: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Activity feed failed: {e}")
        
        # Remove duplicates based on trade_id or (token_id, timestamp, side, shares, price)
        seen = set()
        unique_trades = []
        for trade in trades:
            trade_id = trade.get("trade_id")
            if trade_id:
                key = trade_id
            else:
                # Use a more specific key to avoid false duplicates
                key = (
                    trade["token_id"],
                    trade["timestamp"].isoformat(),
                    trade["side"],
                    round(trade["shares"], 4),
                    round(trade["price"], 6)
                )
            
            if key not in seen:
                seen.add(key)
                unique_trades.append(trade)
        
        # Sort by timestamp descending
        unique_trades.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return unique_trades
    
    async def fetch_user_transactions(
        self,
        wallet_address: str,
        limit: int = 1000
    ) -> List[dict]:
        """
        Fetch complete transaction history for a user including:
        - Trades (buys and sells)
        - Deposits
        - Withdrawals
        """
        transactions = []
        
        # Fetch trades (includes both buys and sells)
        try:
            trades = await self.fetch_user_trades(wallet_address, limit)
            transactions.extend(trades)
        except Exception as e:
            logger.warning(f"Error fetching user trades: {e}")
        
        # Try to fetch deposits/withdrawals from activity feed
        try:
            activity = await self.fetch_activity(limit=limit, user=wallet_address)
            for item in activity:
                item_type = item.get("type", "").lower()
                if item_type in ["deposit", "withdrawal", "transfer", "withdraw"]:
                    try:
                        amount = float(item.get("amount", 0) or item.get("value", 0) or item.get("quantity", 0))
                        if amount > 0:
                            timestamp = datetime.now(timezone.utc)
                            if item.get("createdAt"):
                                try:
                                    timestamp = datetime.fromisoformat(
                                        item.get("createdAt").replace("Z", "+00:00")
                                    )
                                except:
                                    pass
                            elif item.get("timestamp"):
                                try:
                                    ts = item.get("timestamp")
                                    if isinstance(ts, (int, float)):
                                        # Unix timestamp
                                        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                                    else:
                                        # ISO format string
                                        timestamp = datetime.fromisoformat(
                                            str(ts).replace("Z", "+00:00")
                                        )
                                except:
                                    pass
                            
                            side = "DEPOSIT"
                            if item_type in ["withdrawal", "withdraw"]:
                                side = "WITHDRAWAL"
                            elif item_type == "transfer":
                                # Determine direction from addresses
                                from_addr = item.get("from", "").lower()
                                if from_addr == wallet_address.lower():
                                    side = "WITHDRAWAL"
                                else:
                                    side = "DEPOSIT"
                            
                            transactions.append({
                                "trade_id": item.get("id", ""),
                                "market_id": "",
                                "token_id": "",
                                "side": side,
                                "shares": 0.0,
                                "price": amount,
                                "timestamp": timestamp,
                                "maker_address": None,
                                "taker_address": wallet_address,
                                "type": item_type
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing transaction: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Error fetching transactions from activity: {e}")
        
        # Sort by timestamp descending
        transactions.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return transactions
    
    # ==================== POLYGON EVENT QUERIES ====================
    
    async def fetch_ctf_transfer_events(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int,
        ctf_contract: Optional[str] = None
    ) -> List[dict]:
        """
        Fetch ERC-1155 TransferSingle/TransferBatch events from the CTF contract.
        
        These events represent:
        - Buys: Transfer from exchange to wallet
        - Sells: Transfer from wallet to exchange
        - Claims: Transfer from wallet to 0x0 (burn)
        
        Args:
            wallet_address: The wallet to fetch events for (involved as from OR to)
            from_block: Starting block number
            to_block: Ending block number
            ctf_contract: Optional CTF contract address (defaults to config)
        
        Returns:
            List of parsed transfer events
        """
        try:
            from web3 import Web3
        except ImportError:
            logger.warning("web3 not available, cannot query CTF events")
            return []
        
        ctf_address = ctf_contract or self.config.chain_sync.ctf_contract_address
        
        # ERC-1155 TransferSingle event signature
        # event TransferSingle(address indexed operator, address indexed from, address indexed to, uint256 id, uint256 value)
        TRANSFER_SINGLE_TOPIC = Web3.keccak(
            text="TransferSingle(address,address,address,uint256,uint256)"
        ).hex()
        
        # ERC-1155 TransferBatch event signature
        # event TransferBatch(address indexed operator, address indexed from, address indexed to, uint256[] ids, uint256[] values)
        TRANSFER_BATCH_TOPIC = Web3.keccak(
            text="TransferBatch(address,address,address,uint256[],uint256[])"
        ).hex()
        
        events = []
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._fetch_ctf_events_sync,
                wallet_address,
                from_block,
                to_block,
                ctf_address,
                TRANSFER_SINGLE_TOPIC,
                TRANSFER_BATCH_TOPIC
            )
            events = result
        except Exception as e:
            logger.error(f"Error fetching CTF events: {e}")
        
        return events
    
    def _fetch_ctf_events_sync(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int,
        ctf_address: str,
        transfer_single_topic: str,
        transfer_batch_topic: str
    ) -> List[dict]:
        """Synchronous CTF event fetching (called in executor)"""
        from web3 import Web3
        
        events = []
        wallet_lower = wallet_address.lower()
        wallet_topic = "0x" + wallet_lower[2:].zfill(64)
        
        # Try multiple RPC endpoints
        rpc_urls = [
            self.config.polygon_rpc_url,
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
        ]
        
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if not w3.is_connected():
                    continue
                
                # Query TransferSingle events where wallet is sender OR receiver
                # Note: We need two queries - one for 'from' and one for 'to'
                
                # Events where wallet is the sender (sells, claims)
                from_filter = {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": Web3.to_checksum_address(ctf_address),
                    "topics": [
                        transfer_single_topic,
                        None,  # operator (any)
                        wallet_topic,  # from (our wallet)
                    ]
                }
                
                from_logs = w3.eth.get_logs(from_filter)
                
                for log in from_logs:
                    parsed = self._parse_transfer_single_event(w3, log, wallet_lower)
                    if parsed:
                        events.append(parsed)
                
                # Events where wallet is the receiver (buys)
                to_filter = {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": Web3.to_checksum_address(ctf_address),
                    "topics": [
                        transfer_single_topic,
                        None,  # operator (any)
                        None,  # from (any)
                        wallet_topic,  # to (our wallet)
                    ]
                }
                
                to_logs = w3.eth.get_logs(to_filter)
                
                for log in to_logs:
                    parsed = self._parse_transfer_single_event(w3, log, wallet_lower)
                    if parsed:
                        # Avoid duplicates if wallet sent to itself
                        if not any(e["tx_hash"] == parsed["tx_hash"] and e["log_index"] == parsed["log_index"] for e in events):
                            events.append(parsed)
                
                # Successfully got events, no need to try other RPCs
                break
                
            except Exception as e:
                logger.debug(f"RPC {rpc_url} failed for CTF events: {e}")
                continue
        
        return events
    
    def _parse_transfer_single_event(
        self,
        w3,
        log: dict,
        wallet_address: str
    ) -> Optional[dict]:
        """Parse a TransferSingle event log"""
        try:
            # Topics: [event_sig, operator, from, to]
            # Data: [id (uint256), value (uint256)]
            
            topics = log.get("topics", [])
            if len(topics) < 4:
                return None
            
            from_addr = "0x" + topics[2].hex()[-40:]
            to_addr = "0x" + topics[3].hex()[-40:]
            
            # Decode data (id and value are each 32 bytes)
            data = log.get("data", b"")
            if isinstance(data, str):
                data = bytes.fromhex(data[2:] if data.startswith("0x") else data)
            
            if len(data) < 64:
                return None
            
            token_id = int.from_bytes(data[:32], "big")
            value = int.from_bytes(data[32:64], "big")
            
            # CTF tokens use 6 decimals (same as USDC collateral)
            # 1 share = 1 USDC potential payout
            shares = value / 1e6
            
            # Determine transaction type
            wallet_lower = wallet_address.lower()
            from_lower = from_addr.lower()
            to_lower = to_addr.lower()
            
            # Burn address
            ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
            
            if to_lower == ZERO_ADDRESS and from_lower == wallet_lower:
                tx_type = "claim"  # Burning tokens = claiming
            elif from_lower == wallet_lower:
                tx_type = "sell"  # Sending tokens = selling
            elif to_lower == wallet_lower:
                tx_type = "buy"  # Receiving tokens = buying
            else:
                return None  # Not relevant to this wallet
            
            # Get block for timestamp
            block_number = log.get("blockNumber", 0)
            
            return {
                "tx_hash": log.get("transactionHash", b"").hex() if isinstance(log.get("transactionHash"), bytes) else log.get("transactionHash", ""),
                "log_index": log.get("logIndex", 0),
                "block_number": block_number,
                "transaction_type": tx_type,
                "wallet_address": wallet_lower,
                "token_id": str(token_id),
                "from_address": from_lower,
                "to_address": to_lower,
                "shares": shares,
                "raw_log": {
                    "address": log.get("address", ""),
                    "topics": [t.hex() if isinstance(t, bytes) else t for t in topics],
                    "data": data.hex() if isinstance(data, bytes) else data,
                }
            }
        except Exception as e:
            logger.debug(f"Error parsing TransferSingle event: {e}")
            return None
    
    async def fetch_usdc_transfer_events(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int
    ) -> List[dict]:
        """
        Fetch USDC transfer events for deposits and withdrawals.
        
        Args:
            wallet_address: The wallet to fetch events for
            from_block: Starting block number
            to_block: Ending block number
        
        Returns:
            List of parsed USDC transfer events
        """
        try:
            from web3 import Web3
        except ImportError:
            logger.warning("web3 not available, cannot query USDC events")
            return []
        
        # ERC-20 Transfer event signature
        # event Transfer(address indexed from, address indexed to, uint256 value)
        TRANSFER_TOPIC = Web3.keccak(
            text="Transfer(address,address,uint256)"
        ).hex()
        
        events = []
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._fetch_usdc_events_sync,
                wallet_address,
                from_block,
                to_block,
                TRANSFER_TOPIC
            )
            events = result
        except Exception as e:
            logger.error(f"Error fetching USDC events: {e}")
        
        return events
    
    def _fetch_usdc_events_sync(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int,
        transfer_topic: str
    ) -> List[dict]:
        """Synchronous USDC event fetching (called in executor)"""
        from web3 import Web3
        
        events = []
        wallet_lower = wallet_address.lower()
        wallet_topic = "0x" + wallet_lower[2:].zfill(64)
        usdc_address = self.config.usdc_contract_address
        
        rpc_urls = [
            self.config.polygon_rpc_url,
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
        ]
        
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if not w3.is_connected():
                    continue
                
                # Deposits (transfers TO our wallet)
                deposit_filter = {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": Web3.to_checksum_address(usdc_address),
                    "topics": [
                        transfer_topic,
                        None,  # from (any)
                        wallet_topic,  # to (our wallet)
                    ]
                }
                
                deposit_logs = w3.eth.get_logs(deposit_filter)
                
                for log in deposit_logs:
                    parsed = self._parse_usdc_transfer_event(w3, log, wallet_lower, "deposit")
                    if parsed:
                        events.append(parsed)
                
                # Withdrawals (transfers FROM our wallet)
                withdrawal_filter = {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "address": Web3.to_checksum_address(usdc_address),
                    "topics": [
                        transfer_topic,
                        wallet_topic,  # from (our wallet)
                    ]
                }
                
                withdrawal_logs = w3.eth.get_logs(withdrawal_filter)
                
                for log in withdrawal_logs:
                    parsed = self._parse_usdc_transfer_event(w3, log, wallet_lower, "withdrawal")
                    if parsed:
                        events.append(parsed)
                
                break
                
            except Exception as e:
                logger.debug(f"RPC {rpc_url} failed for USDC events: {e}")
                continue
        
        return events
    
    def _parse_usdc_transfer_event(
        self,
        w3,
        log: dict,
        wallet_address: str,
        tx_type: str
    ) -> Optional[dict]:
        """Parse a USDC Transfer event log"""
        try:
            topics = log.get("topics", [])
            if len(topics) < 3:
                return None
            
            from_addr = "0x" + topics[1].hex()[-40:]
            to_addr = "0x" + topics[2].hex()[-40:]
            
            # Decode value (32 bytes)
            data = log.get("data", b"")
            if isinstance(data, str):
                data = bytes.fromhex(data[2:] if data.startswith("0x") else data)
            
            if len(data) < 32:
                return None
            
            value = int.from_bytes(data[:32], "big")
            # USDC has 6 decimals
            usdc_amount = value / 1e6
            
            return {
                "tx_hash": log.get("transactionHash", b"").hex() if isinstance(log.get("transactionHash"), bytes) else log.get("transactionHash", ""),
                "log_index": log.get("logIndex", 0),
                "block_number": log.get("blockNumber", 0),
                "transaction_type": tx_type,
                "wallet_address": wallet_address.lower(),
                "from_address": from_addr.lower(),
                "to_address": to_addr.lower(),
                "usdc_amount": usdc_amount,
                "raw_log": {
                    "address": log.get("address", ""),
                    "topics": [t.hex() if isinstance(t, bytes) else t for t in topics],
                    "data": data.hex() if isinstance(data, bytes) else data,
                }
            }
        except Exception as e:
            logger.debug(f"Error parsing USDC Transfer event: {e}")
            return None
    
    async def get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """Get the timestamp of a block"""
        try:
            from web3 import Web3
        except ImportError:
            return None
        
        try:
            loop = asyncio.get_event_loop()
            timestamp = await loop.run_in_executor(
                None,
                self._get_block_timestamp_sync,
                block_number
            )
            return timestamp
        except Exception as e:
            logger.debug(f"Error getting block timestamp: {e}")
            return None
    
    def _get_block_timestamp_sync(self, block_number: int) -> Optional[datetime]:
        """Synchronous block timestamp fetch"""
        from web3 import Web3
        
        rpc_urls = [
            self.config.polygon_rpc_url,
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
        ]
        
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if not w3.is_connected():
                    continue
                
                block = w3.eth.get_block(block_number)
                timestamp = block.get("timestamp", 0)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
                
            except Exception as e:
                logger.debug(f"RPC {rpc_url} failed for block timestamp: {e}")
                continue
        
        return None
    
    async def get_current_block(self) -> int:
        """Get the current block number"""
        try:
            from web3 import Web3
        except ImportError:
            return 0
        
        try:
            loop = asyncio.get_event_loop()
            block = await loop.run_in_executor(
                None,
                self._get_current_block_sync
            )
            return block
        except Exception as e:
            logger.debug(f"Error getting current block: {e}")
            return 0
    
    def _get_current_block_sync(self) -> int:
        """Synchronous current block fetch"""
        from web3 import Web3
        
        rpc_urls = [
            self.config.polygon_rpc_url,
            "https://rpc.ankr.com/polygon",
            "https://polygon.llamarpc.com",
        ]
        
        for rpc_url in rpc_urls:
            try:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                if not w3.is_connected():
                    continue
                
                return w3.eth.block_number
                
            except Exception as e:
                logger.debug(f"RPC {rpc_url} failed for current block: {e}")
                continue
        
        return 0


