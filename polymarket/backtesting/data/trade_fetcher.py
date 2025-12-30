"""
Trade History Fetcher for Backtesting.

Fetches historical trade data from the CLOB API including wallet addresses,
which enables smart money detection and trade-based signal analysis.
"""

import asyncio
import aiohttp
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)

# API Configuration
CLOB_HOST = "https://clob.polymarket.com"
POLYGONSCAN_API_BASE = "https://api.polygonscan.com/api"
POLYGONSCAN_API_KEY = os.getenv("POLYGONSCAN_API_KEY", "")

# Log if API key is loaded
if POLYGONSCAN_API_KEY:
    logger.debug(f"Polygonscan API key loaded: {POLYGONSCAN_API_KEY[:8]}...")
else:
    logger.warning("No POLYGONSCAN_API_KEY found in environment")

# Rate limiting
REQUEST_DELAY_MS = 100  # 100ms between requests


@dataclass
class TradeData:
    """Represents a single trade with wallet information."""
    trade_id: str
    token_id: str
    market_id: str
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    maker_address: str
    taker_address: str
    timestamp: datetime
    
    @property
    def value_usd(self) -> float:
        """Trade value in USD."""
        return self.price * self.size
    
    @property
    def active_wallet(self) -> str:
        """The wallet that initiated the trade (taker)."""
        return self.taker_address or self.maker_address


class TradeFetcher:
    """
    Fetches historical trade data from Polymarket CLOB API.
    
    Supports:
    - Fetching trades for specific tokens/markets (requires authenticated clob_client)
    - Batch fetching for multiple tokens
    - On-chain wallet validation via Polygonscan
    """
    
    def __init__(
        self, 
        session: Optional[aiohttp.ClientSession] = None,
        clob_client = None,  # py_clob_client.client.ClobClient
    ):
        self._session = session
        self._owns_session = session is None
        self._clob_client = clob_client
        self._request_count = 0
        
    async def __aenter__(self):
        if self._owns_session:
            self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session and self._session:
            await self._session.close()
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        await asyncio.sleep(REQUEST_DELAY_MS / 1000)
        self._request_count += 1
    
    async def fetch_trades_for_token(
        self,
        token_id: str,
        limit: int = 1000,
        before_ts: Optional[int] = None,
    ) -> List[TradeData]:
        """
        Fetch historical trades for a specific token.
        
        Args:
            token_id: The token/outcome ID
            limit: Maximum number of trades to fetch
            before_ts: Only fetch trades before this timestamp
            
        Returns:
            List of TradeData objects with wallet addresses
        """
        await self._rate_limit()
        
        # Use authenticated clob_client if available
        if self._clob_client:
            try:
                # Use py_clob_client to fetch trades (authenticated)
                trades_response = self._clob_client.get_trades(
                    asset_id=token_id,
                    limit=min(limit, 500),
                )
                trades_list = trades_response if isinstance(trades_response, list) else []
            except Exception as e:
                logger.debug(f"Error fetching trades via clob_client for {token_id}: {e}")
                return []
        else:
            # Fallback to unauthenticated request (will likely fail)
            url = f"{CLOB_HOST}/trades"
            params = {
                "asset_id": token_id,
                "limit": min(limit, 1000),
            }
            
            if before_ts:
                params["before"] = before_ts
            
            try:
                async with self._session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to fetch trades for {token_id}: {resp.status}")
                        return []
                    
                    data = await resp.json()
                    trades_list = data if isinstance(data, list) else data.get("data", [])
            except Exception as e:
                logger.warning(f"Error fetching trades for {token_id}: {e}")
                return []
        
        # Parse trades
        trades = []
        for t in trades_list:
            try:
                # Parse timestamp
                ts = t.get("match_time") or t.get("timestamp") or t.get("created_at")
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
                
                trade = TradeData(
                    trade_id=str(t.get("id", "")),
                    token_id=token_id,
                    market_id=t.get("market", "") or t.get("condition_id", ""),
                    price=float(t.get("price", 0)),
                    size=float(t.get("size", 0)),
                    side=t.get("side", "").upper() or ("BUY" if t.get("is_buy") else "SELL"),
                    maker_address=(t.get("maker_address") or t.get("maker") or "").lower(),
                    taker_address=(t.get("taker_address") or t.get("taker") or "").lower(),
                    timestamp=timestamp,
                )
                
                if trade.price > 0 and trade.size > 0:
                    trades.append(trade)
                    
            except Exception as e:
                logger.debug(f"Error parsing trade: {e}")
                continue
        
        return trades
    
    async def fetch_trades_for_tokens(
        self,
        token_ids: List[str],
        limit_per_token: int = 500,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, List[TradeData]]:
        """
        Fetch trades for multiple tokens.
        
        Args:
            token_ids: List of token IDs to fetch
            limit_per_token: Max trades per token
            progress_callback: Optional callback(current, total)
            
        Returns:
            Dict mapping token_id -> list of trades
        """
        results = {}
        total = len(token_ids)
        
        for i, token_id in enumerate(token_ids):
            if progress_callback and i % 50 == 0:
                progress_callback(i, total)
            
            trades = await self.fetch_trades_for_token(token_id, limit_per_token)
            if trades:
                results[token_id] = trades
        
        if progress_callback:
            progress_callback(total, total)
        
        logger.info(f"Fetched trades for {len(results)}/{total} tokens")
        return results
    
    async def fetch_wallet_chain_age(
        self,
        wallet: str
    ) -> Tuple[Optional[datetime], int, Optional[str]]:
        """
        Fetch when a wallet was first created on-chain.
        
        Args:
            wallet: Wallet address to check
            
        Returns:
            Tuple of (creation_date, total_tx_count, first_tx_hash)
        """
        if not POLYGONSCAN_API_KEY:
            logger.debug("No Polygonscan API key configured")
            return None, 0, None
        
        await self._rate_limit()
        
        try:
            # Get first transaction (oldest first)
            params = {
                "module": "account",
                "action": "txlist",
                "address": wallet,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",
                "page": 1,
                "offset": 1,
                "apikey": POLYGONSCAN_API_KEY,
            }
            
            async with self._session.get(
                POLYGONSCAN_API_BASE,
                params=params,
                timeout=15
            ) as resp:
                if resp.status != 200:
                    return None, 0, None
                
                data = await resp.json()
                
                if data.get("status") != "1" or not data.get("result"):
                    return None, 0, None
                
                first_tx = data["result"][0]
                
                # Parse timestamp
                try:
                    ts = int(first_tx.get("timeStamp", 0))
                    creation_date = datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, TypeError):
                    creation_date = None
                
                first_tx_hash = first_tx.get("hash", "")
            
            # Get total transaction count
            params_count = {
                "module": "proxy",
                "action": "eth_getTransactionCount",
                "address": wallet,
                "tag": "latest",
                "apikey": POLYGONSCAN_API_KEY,
            }
            
            total_tx_count = 0
            try:
                async with self._session.get(
                    POLYGONSCAN_API_BASE,
                    params=params_count,
                    timeout=10
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("result", "0x0")
                        if result and result.startswith("0x"):
                            total_tx_count = int(result, 16)
            except Exception:
                pass
            
            return creation_date, total_tx_count, first_tx_hash
            
        except Exception as e:
            logger.debug(f"Error fetching wallet chain age for {wallet[:10]}...: {e}")
            return None, 0, None
    
    async def validate_wallets_batch(
        self,
        wallets: List[str],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, dict]:
        """
        Validate multiple wallets on-chain.
        
        Args:
            wallets: List of wallet addresses
            progress_callback: Optional callback(current, total)
            
        Returns:
            Dict mapping wallet -> {creation_date, tx_count, age_days, is_fresh}
        """
        results = {}
        total = len(wallets)
        
        for i, wallet in enumerate(wallets):
            if progress_callback and i % 20 == 0:
                progress_callback(i, total)
            
            creation_date, tx_count, _ = await self.fetch_wallet_chain_age(wallet)
            
            age_days = None
            if creation_date:
                delta = datetime.now(timezone.utc) - creation_date
                age_days = delta.total_seconds() / 86400
            
            results[wallet.lower()] = {
                "creation_date": creation_date,
                "tx_count": tx_count,
                "age_days": age_days,
                "is_fresh": age_days is not None and age_days <= 30,
                "is_newly_created": age_days is not None and age_days <= 7,
                "is_low_activity": tx_count < 10,
            }
        
        if progress_callback:
            progress_callback(total, total)
        
        logger.info(f"Validated {len(results)} wallets on-chain")
        return results

