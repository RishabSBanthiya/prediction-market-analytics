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
            self.session = aiohttp.ClientSession()
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
        active: bool = True
    ) -> List[dict]:
        """Fetch a batch of markets from Gamma API"""
        params = {
            "closed": str(closed).lower(),
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
        }
        
        data = await self._get(f"{self.config.gamma_api_base}/markets", params)
        return data if isinstance(data, list) else []
    
    async def fetch_all_markets(
        self, 
        closed: bool = False,
        active: bool = True,
        max_markets: int = 10000
    ) -> List[dict]:
        """Fetch all markets concurrently"""
        first_batch = await self.fetch_markets_batch(0, 100, closed, active)
        if not first_batch:
            return []
        
        # Fetch remaining in parallel
        tasks = []
        for offset in range(100, max_markets, 100):
            tasks.append(self.fetch_markets_batch(offset, 100, closed, active))
        
        results = await asyncio.gather(*tasks)
        
        all_markets = first_batch
        for batch in results:
            if batch:
                all_markets.extend(batch)
            else:
                break
        
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
        
        # Parse end date
        end_str = (
            raw_market.get("end_date_iso") or 
            raw_market.get("endDate") or 
            raw_market.get("end_date")
        )
        if not end_str:
            return None
        
        try:
            end_str = str(end_str).replace("Z", "+00:00")
            if "T" not in end_str:
                end_str = end_str + "T00:00:00+00:00"
            end_date = datetime.fromisoformat(end_str)
        except Exception:
            return None
        
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
    
    async def fetch_activity(self, limit: int = 100, user: Optional[str] = None) -> List[dict]:
        """Fetch activity feed, optionally filtered by user"""
        params = {"limit": limit}
        if user:
            params["user"] = user
        data = await self._get(
            f"{self.config.data_api_base}/activity",
            params
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
        limit: int = 1000
    ) -> List[dict]:
        """
        Fetch all trades for a user from Polymarket.
        This includes both buy and sell trades.
        Tries multiple endpoints to get complete trade history.
        """
        trades = []
        
        # Method 1: Try CLOB API trades endpoint with user filter
        try:
            data = await self._get(
                f"{self.config.clob_host}/trades",
                {"user": wallet_address, "limit": limit}
            )
            
            if data and isinstance(data, list):
                for t in data:
                    try:
                        # Check if this trade involves our wallet
                        maker = t.get("maker_address", "").lower()
                        taker = t.get("taker_address", "").lower()
                        wallet_lower = wallet_address.lower()
                        
                        if maker == wallet_lower or taker == wallet_lower:
                            trades.append({
                                "trade_id": t.get("id", ""),
                                "market_id": t.get("market", ""),
                                "token_id": t.get("asset_id", "") or t.get("asset", ""),
                                "side": "SELL" if maker == wallet_lower else "BUY",
                                "shares": float(t.get("size", 0)),
                                "price": float(t.get("price", 0)),
                                "timestamp": datetime.fromisoformat(
                                    t.get("created_at", "").replace("Z", "+00:00")
                                ) if t.get("created_at") else datetime.now(timezone.utc),
                                "maker_address": t.get("maker_address"),
                                "taker_address": t.get("taker_address"),
                                "type": "trade"
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing user trade: {e}")
                        continue
        except Exception as e:
            logger.debug(f"CLOB trades endpoint failed: {e}")
        
        # Method 2: Try Data API activity feed filtered by user
        try:
            activity = await self.fetch_activity(limit=limit, user=wallet_address)
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


