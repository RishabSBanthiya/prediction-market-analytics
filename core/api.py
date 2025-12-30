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
    
    async def fetch_closed_markets(self, days: int = 7) -> List[dict]:
        """Fetch recently closed/resolved markets"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_markets = []
        offset = 0
        limit = 100
        
        while True:
            params = {
                "closed": "true",
                "limit": limit,
                "offset": offset,
            }
            
            data = await self._get(f"{self.config.gamma_api_base}/markets", params)
            if not data:
                break
            
            # Filter by closed time
            for market in data:
                closed_time_str = market.get("closedTime")
                if closed_time_str:
                    try:
                        closed_time = datetime.fromisoformat(
                            closed_time_str.replace("Z", "+00:00")
                        )
                        if closed_time >= cutoff:
                            all_markets.append(market)
                    except ValueError:
                        continue
            
            if len(data) < limit:
                break
            
            offset += limit
            
            # Safety limit
            if offset >= 5000:
                break
        
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
        
        return Market(
            condition_id=raw_market.get("conditionId") or raw_market.get("condition_id") or "",
            question=raw_market.get("question") or "",
            slug=raw_market.get("market_slug") or raw_market.get("slug") or "",
            end_date=end_date,
            start_date=start_date,
            category=raw_market.get("category"),
            tokens=tokens,
            closed=raw_market.get("closed", False),
            resolved=raw_market.get("resolved", False),
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
    
    async def fetch_activity(self, limit: int = 100) -> List[dict]:
        """Fetch global activity feed"""
        data = await self._get(
            f"{self.config.data_api_base}/activity",
            {"limit": limit}
        )
        
        return data if isinstance(data, list) else []
    
    async def fetch_trades(
        self, 
        token_id: str, 
        limit: int = 100
    ) -> List[Trade]:
        """Fetch recent trades for a token"""
        data = await self._get(
            f"{self.config.clob_host}/trades",
            {"asset_id": token_id, "limit": limit}
        )
        
        if not data or not isinstance(data, list):
            return []
        
        trades = []
        for t in data:
            try:
                trades.append(Trade(
                    trade_id=t.get("id", ""),
                    market_id=t.get("market", ""),
                    token_id=token_id,
                    side=Side.BUY if t.get("side", "").upper() == "BUY" else Side.SELL,
                    shares=float(t.get("size", 0)),
                    price=float(t.get("price", 0)),
                    timestamp=datetime.fromisoformat(
                        t.get("created_at", "").replace("Z", "+00:00")
                    ) if t.get("created_at") else datetime.now(timezone.utc),
                    maker_address=t.get("maker_address"),
                    taker_address=t.get("taker_address"),
                ))
            except Exception as e:
                logger.debug(f"Error parsing trade: {e}")
                continue
        
        return trades


