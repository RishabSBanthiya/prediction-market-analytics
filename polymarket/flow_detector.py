#!/usr/bin/env python3
"""
Polymarket Unusual Flow Detection Service - V2 (Activity Feed Approach)

This version monitors the GLOBAL ACTIVITY FEED from data-api.polymarket.com
The same data source that powers polymarket.com/activity

Key features:
1. Uses Data API /activity endpoint (same as website activity page)
2. Gets ALL trades across ALL markets in one API call
3. Creates market states on-demand when trades occur
4. Highly scalable - single request covers entire platform
5. No need to enumerate/poll individual markets

Advanced Features:
- Alert deduplication with cooldowns
- Volatility-adjusted dynamic thresholds
- Market category filtering (crypto/sports/politics)
- Price acceleration detection
- Cross-market correlation detection

Usage:
    python flow_detector_v2.py [--min-trade-size 100] [--verbose] [--category crypto]
"""

import asyncio
import aiohttp
import json
import logging
import os
import statistics
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..trading.storage.base import StorageBackend
from collections import deque
from enum import Enum
import argparse
import websockets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
# API endpoints
CLOB_HOST = "https://clob.polymarket.com"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"

# WebSocket for real-time trades (RTDS - Real-Time Data Socket)
RTDS_WEBSOCKET_URL = "wss://ws-live-data.polymarket.com"

# Blockchain API for on-chain transfer detection (Polygon)
POLYGONSCAN_API_BASE = "https://api.polygonscan.com/api"
POLYGONSCAN_API_KEY = os.getenv("POLYGONSCAN_API_KEY", "")  # Set via env for higher rate limits

# USDC contract on Polygon (main token for Polymarket)
USDC_CONTRACT_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
# USDC.e (bridged USDC) on Polygon
USDCE_CONTRACT_POLYGON = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"

# On-chain transfer detection settings
CHAIN_CHECK_INTERVAL_SECONDS = 300  # Check wallet's on-chain history every 5 minutes
CHAIN_LOOKBACK_BLOCKS = 50000  # Look back ~1 day of blocks on Polygon
MIN_TRANSFER_VALUE_USD = 100  # Minimum transfer value to track

# Detection thresholds (base values - adjusted by volatility)
# V5 OPTIMIZED: Tighter thresholds for better signal quality given latency constraints
PRICE_MOVEMENT_THRESHOLD = 0.05  # 5% - lower bar but with Z-score filtering
PRICE_MOVEMENT_WINDOW_SECONDS = 15  # Shorter window for faster signals
VOLUME_SPIKE_MULTIPLIER = 4.0  # 4x - slightly lower to catch more signals
OVERSIZED_BET_MULTIPLIER = 12.0  # 12x - balance between noise and signal
OVERSIZED_BET_MIN_USD = 10000  # $10k - lower to catch more meaningful flow
ORDERBOOK_IMBALANCE_RATIO = 4.0  # Slightly more sensitive
COORDINATED_WALLET_COUNT = 2  # Lower threshold - 2 wallets is still meaningful
COORDINATED_WALLET_WINDOW_SECONDS = 30  # Tighter window for coordination
MIN_TRADE_SIZE_FOR_TRACKING = 500  # $500 - HIGHER to reduce noise (was $250)

# Price acceleration thresholds
ACCELERATION_THRESHOLD = 2.0     # Recent change must be 2x the earlier change
MIN_ACCELERATION_CHANGE = 0.02   # Minimum 2% change to trigger acceleration alert

# Cross-market correlation
CORRELATION_WINDOW_SECONDS = 60  # Look for correlated moves in 60s window
MIN_CORRELATED_MARKETS = 3       # Minimum markets moving together
CORRELATION_MOVE_THRESHOLD = 0.03  # 3% move to be considered

# Alert deduplication
ALERT_COOLDOWN_SECONDS = 120     # 2 minute cooldown (was 5 min) - faster cycling

# Short-duration market handling
# Markets with lifetime < this threshold will skip price movement detection
# because they can never accumulate enough history for statistical analysis
MIN_MARKET_DURATION_HOURS = 1.0  # 1 hour minimum for price movement alerts

# Short-duration market MOMENTUM trading (alternative to skipping)
# For markets < 1 hour, we can trade momentum instead of statistical anomalies
SHORT_DURATION_MOMENTUM_THRESHOLD = 0.03  # 3% move triggers momentum signal
SHORT_DURATION_MIN_TRADES = 3  # Need at least 3 trades to confirm momentum
SHORT_DURATION_MOMENTUM_WINDOW_SECONDS = 30  # Look at last 30 seconds
SHORT_DURATION_MOMENTUM_ENABLED = True  # Enable momentum trading for short markets

# Trade feed polling
TRADE_FEED_POLL_INTERVAL = 0.5  # Poll every 0.5s (was 2s) - CRITICAL for latency
TRADE_FEED_LIMIT = 50  # Reduced - process faster with smaller batches

# Market state cleanup - more aggressive to prevent memory bloat
INACTIVE_MARKET_TIMEOUT_MINUTES = 15  # 15 min (was 30) - faster cleanup
STATE_CLEANUP_INTERVAL_SECONDS = 60   # 60s (was 300) - more frequent cleanup

# WebSocket stale connection detection
STALE_CONNECTION_TIMEOUT_SECONDS = 60  # Reconnect if no trades received for 60 seconds

# Historical data window - reduced for memory efficiency
MAX_PRICE_HISTORY_POINTS = 50   # 50 (was 100) - sufficient for Z-score calc
MAX_TRADES_PER_MARKET = 50      # 50 (was 100) - reduces memory per market

# Memory management
MAX_WALLET_PROFILES = 5000      # Cap wallet profiles to prevent unbounded growth
MAX_MARKET_STATES = 300         # Cap market states
MAX_PROCESSED_TRADE_IDS = 10000 # Cap processed trade IDs set


class MarketCategory(Enum):
    """Market category types"""
    CRYPTO = "crypto"
    SPORTS = "sports"
    POLITICS = "politics"
    ENTERTAINMENT = "entertainment"
    FINANCE = "finance"
    OTHER = "other"
    ALL = "all"


def categorize_market(question: str, tags: List[str] = None) -> MarketCategory:
    """Categorize a market based on its question and tags"""
    q = question.lower()
    
    # Check tags first if available
    if tags:
        tag_str = " ".join(tags).lower()
        if any(kw in tag_str for kw in ["crypto", "bitcoin", "ethereum"]):
            return MarketCategory.CRYPTO
        if any(kw in tag_str for kw in ["sports", "nfl", "nba"]):
            return MarketCategory.SPORTS
        if any(kw in tag_str for kw in ["politics", "election"]):
            return MarketCategory.POLITICS
    
    # Crypto keywords
    crypto_keywords = [
        "btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp", "ripple",
        "crypto", "token", "doge", "dogecoin", "bnb", "ada", "cardano", "avax",
        "matic", "polygon", "link", "chainlink", "uni", "uniswap", "aave"
    ]
    if any(kw in q for kw in crypto_keywords):
        return MarketCategory.CRYPTO
    
    # Sports keywords
    sports_keywords = [
        "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
        "baseball", "hockey", "tennis", "golf", "ufc", "mma", "boxing",
        "game", "match", "score", "win", "playoffs", "championship",
        "super bowl", "world series", "finals", "vs.", "versus"
    ]
    if any(kw in q for kw in sports_keywords):
        return MarketCategory.SPORTS
    
    # Politics keywords
    politics_keywords = [
        "president", "election", "vote", "congress", "senate", "house",
        "trump", "biden", "democrat", "republican", "governor", "mayor",
        "poll", "primary", "cabinet", "administration", "impeach"
    ]
    if any(kw in q for kw in politics_keywords):
        return MarketCategory.POLITICS
    
    # Entertainment keywords
    entertainment_keywords = [
        "oscar", "grammy", "emmy", "movie", "film", "music", "celebrity",
        "award", "album", "song", "actor", "actress", "director"
    ]
    if any(kw in q for kw in entertainment_keywords):
        return MarketCategory.ENTERTAINMENT
    
    # Finance keywords
    finance_keywords = [
        "fed", "interest rate", "inflation", "gdp", "stock", "s&p",
        "nasdaq", "dow", "economy", "recession", "treasury", "yield"
    ]
    if any(kw in q for kw in finance_keywords):
        return MarketCategory.FINANCE
    
    return MarketCategory.OTHER


@dataclass
class TradeEvent:
    """A single trade event"""
    trade_id: str
    token_id: str
    market_id: str
    side: str
    size: float
    price: float
    timestamp: datetime
    maker_address: Optional[str] = None
    taker_address: Optional[str] = None
    value_usd: float = 0.0
    
    def __post_init__(self):
        self.value_usd = self.size * self.price


@dataclass
class OnChainTransfer:
    """Represents an on-chain transfer between wallets"""
    from_address: str
    to_address: str
    value: float  # In USD or token amount
    token_symbol: str  # e.g., "USDC", "MATIC", "POLY"
    tx_hash: str
    timestamp: datetime
    block_number: int


@dataclass
class WalletProfile:
    """Profile of a wallet's trading history"""
    address: str
    first_seen: datetime  # First seen on Polymarket
    last_seen: datetime
    total_trades: int = 0
    total_volume_usd: float = 0.0
    markets_traded: Set[str] = field(default_factory=set)
    is_cold: bool = True  # No prior Polymarket trades
    # On-chain wallet age data
    chain_creation_date: Optional[datetime] = None  # When wallet was created on-chain
    chain_first_tx_hash: Optional[str] = None  # First ever transaction
    chain_total_tx_count: int = 0  # Total transactions on chain
    chain_age_checked: bool = False  # Whether we've checked chain age
    # Track on-chain transfers TO this wallet (funding sources)
    funded_by: Dict[str, List[OnChainTransfer]] = field(default_factory=dict)  # wallet -> transfers received
    # Track on-chain transfers FROM this wallet (funding destinations)
    funded_to: Dict[str, List[OnChainTransfer]] = field(default_factory=dict)  # wallet -> transfers sent
    # Cache for when we last checked on-chain transfers
    last_chain_check: Optional[datetime] = None
    recent_markets_with_timing: deque = field(default_factory=lambda: deque(maxlen=50))  # (market_id, timestamp, side)
    
    @property
    def is_smart_money(self) -> bool:
        return self.total_trades >= 10 and self.total_volume_usd > 50000
    
    @property
    def wallet_age_days(self) -> Optional[float]:
        """Get wallet age in days based on chain creation date"""
        if not self.chain_creation_date:
            return None
        delta = datetime.now(timezone.utc) - self.chain_creation_date
        return delta.total_seconds() / 86400  # Convert to days
    
    @property
    def is_newly_created_wallet(self) -> bool:
        """Check if wallet was created on-chain within the last 7 days"""
        if not self.chain_age_checked:
            return False  # Don't know yet
        if self.chain_creation_date is None:
            return False  # Couldn't determine
        age_days = self.wallet_age_days
        return age_days is not None and age_days <= 7
    
    @property
    def is_fresh_wallet(self) -> bool:
        """Check if wallet was created on-chain within the last 30 days"""
        if not self.chain_age_checked:
            return False
        if self.chain_creation_date is None:
            return False
        age_days = self.wallet_age_days
        return age_days is not None and age_days <= 30
    
    @property 
    def is_low_activity_wallet(self) -> bool:
        """Check if wallet has very few on-chain transactions (< 10)"""
        if not self.chain_age_checked:
            return False
        return self.chain_total_tx_count < 10
    
    def get_connected_wallets(self, min_transfers: int = 1) -> List[Tuple[str, int, float]]:
        """Get wallets connected via on-chain transfers
        
        Returns:
            List of (wallet_address, transfer_count, total_value)
        """
        connections = {}
        
        # Aggregate incoming transfers
        for wallet, transfers in self.funded_by.items():
            if wallet not in connections:
                connections[wallet] = {"count": 0, "value": 0.0}
            connections[wallet]["count"] += len(transfers)
            connections[wallet]["value"] += sum(t.value for t in transfers)
        
        # Aggregate outgoing transfers
        for wallet, transfers in self.funded_to.items():
            if wallet not in connections:
                connections[wallet] = {"count": 0, "value": 0.0}
            connections[wallet]["count"] += len(transfers)
            connections[wallet]["value"] += sum(t.value for t in transfers)
        
        # Filter and sort by transfer count
        result = [
            (wallet, data["count"], data["value"])
            for wallet, data in connections.items()
            if data["count"] >= min_transfers
        ]
        return sorted(result, key=lambda x: (-x[1], -x[2]))


@dataclass
class MarketState:
    """Dynamic market state (created on-demand)"""
    market_id: str
    token_id: str
    question: str = "Unknown"
    category: MarketCategory = MarketCategory.OTHER
    
    # Market timing (for slippage calculations)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Price tracking
    current_price: float = 0.0
    price_history: deque = field(default_factory=lambda: deque(maxlen=MAX_PRICE_HISTORY_POINTS))
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Volume tracking
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=MAX_TRADES_PER_MARKET))
    trade_sizes: deque = field(default_factory=lambda: deque(maxlen=50))
    avg_trade_size: float = 0.0
    
    # Wallet tracking
    recent_wallets: deque = field(default_factory=lambda: deque(maxlen=20))
    wallet_sides: Dict[str, str] = field(default_factory=dict)
    wallet_timestamps: Dict[str, datetime] = field(default_factory=dict)
    
    # Orderbook state
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    
    # Volatility tracking
    volatility: float = 0.0
    
    @property
    def lifetime_hours(self) -> Optional[float]:
        """Total market lifetime in hours (from start_date to end_date)."""
        if self.start_date is None or self.end_date is None:
            return None
        return (self.end_date - self.start_date).total_seconds() / 3600
    
    def get_timing_details(self) -> Dict[str, Any]:
        """Get market timing details for inclusion in FlowAlert details."""
        return {
            "market_start_date": self.start_date.isoformat() if self.start_date else None,
            "market_end_date": self.end_date.isoformat() if self.end_date else None,
            "market_lifetime_hours": round(self.lifetime_hours, 2) if self.lifetime_hours else None,
        }
    
    def add_trade(self, trade: TradeEvent):
        """Add trade and update statistics"""
        self.recent_trades.append(trade)
        self.trade_sizes.append(trade.value_usd)
        self.current_price = trade.price
        self.price_history.append((trade.timestamp, trade.price))
        self.last_update = trade.timestamp
        
        # Use MEDIAN instead of mean for baseline - much more robust to outliers
        # This prevents a single large trade from inflating the "average"
        if self.trade_sizes:
            sorted_sizes = sorted(self.trade_sizes)
            n = len(sorted_sizes)
            if n % 2 == 0:
                self.avg_trade_size = (sorted_sizes[n//2 - 1] + sorted_sizes[n//2]) / 2
            else:
                self.avg_trade_size = sorted_sizes[n//2]
        
        wallet = trade.taker_address or trade.maker_address
        if wallet:
            self.recent_wallets.append(wallet)
            self.wallet_sides[wallet] = trade.side
            self.wallet_timestamps[wallet] = trade.timestamp
    
    def get_price_change(self, window_seconds: int) -> Optional[float]:
        """Calculate price change over window"""
        if len(self.price_history) < 2:
            return None
        
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        old_price = None
        
        for timestamp, price in self.price_history:
            if timestamp >= cutoff:
                old_price = price
                break
        
        if old_price is None or old_price == 0:
            return None
        
        return (self.current_price - old_price) / old_price
    
    def get_price_change_window(self, start_seconds: int, end_seconds: int) -> Optional[float]:
        """Get price change between two time windows"""
        if len(self.price_history) < 2:
            return None
        
        now = datetime.now(timezone.utc)
        start_cutoff = now - timedelta(seconds=start_seconds)
        end_cutoff = now - timedelta(seconds=end_seconds)
        
        start_price = None
        end_price = None
        
        for timestamp, price in self.price_history:
            if timestamp <= start_cutoff:
                start_price = price
            if start_cutoff < timestamp <= end_cutoff:
                end_price = price
        
        if start_price is None or end_price is None or start_price == 0:
            return None
        
        return (end_price - start_price) / start_price
    
    def get_volume(self, window_minutes: int) -> float:
        """Calculate volume over window"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        return sum(t.value_usd for t in self.recent_trades if t.timestamp >= cutoff)
    
    def get_coordinated_wallets(self, window_seconds: int) -> Tuple[int, str]:
        """Count wallets trading same side"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        
        buy_wallets = set()
        sell_wallets = set()
        
        for wallet, timestamp in self.wallet_timestamps.items():
            if timestamp >= cutoff:
                if self.wallet_sides.get(wallet) == "BUY":
                    buy_wallets.add(wallet)
                elif self.wallet_sides.get(wallet) == "SELL":
                    sell_wallets.add(wallet)
        
        if len(buy_wallets) >= len(sell_wallets):
            return len(buy_wallets), "BUY"
        else:
            return len(sell_wallets), "SELL"
    
    def calculate_volatility(self) -> float:
        """Calculate rolling volatility from price history (standard deviation of returns)"""
        if len(self.price_history) < 10:
            return 0.0
        
        prices = [p for _, p in self.price_history]
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 5:
            return 0.0
        
        try:
            self.volatility = statistics.stdev(returns)
        except statistics.StatisticsError:
            self.volatility = 0.0
        
        return self.volatility
    
    def get_volatility_stats(self) -> Dict[str, float]:
        """Get comprehensive volatility statistics"""
        if len(self.price_history) < 10:
            return {"volatility": 0, "avg_move": 0, "max_move": 0, "samples": 0}
        
        prices = [p for _, p in self.price_history]
        abs_returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                abs_returns.append(abs((prices[i] - prices[i-1]) / prices[i-1]))
        
        if not abs_returns:
            return {"volatility": 0, "avg_move": 0, "max_move": 0, "samples": 0}
        
        return {
            "volatility": self.calculate_volatility(),
            "avg_move": statistics.mean(abs_returns),
            "max_move": max(abs_returns),
            "samples": len(abs_returns)
        }
    
    def get_z_score(self, current_change: float) -> Optional[float]:
        """Calculate Z-score: how many standard deviations is this move from normal?"""
        if len(self.price_history) < 15:
            return None
        
        prices = [p for _, p in self.price_history]
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
        
        if len(returns) < 10:
            return None
        
        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            
            if std_return == 0:
                return None
            
            return (current_change - mean_return) / std_return
        except statistics.StatisticsError:
            return None
    
    def is_move_unusual(self, change: float, z_threshold: float = 2.5) -> Tuple[bool, Optional[float], str]:
        """
        Determine if a price move is unusual for THIS market.
        
        Returns:
            - is_unusual: bool
            - z_score: how many std devs from normal
            - context: description of the market's typical behavior
        """
        z_score = self.get_z_score(change)
        vol_stats = self.get_volatility_stats()
        
        # Not enough data - fall back to absolute threshold
        if z_score is None or vol_stats["samples"] < 10:
            is_unusual = abs(change) >= 0.05  # Default 5% threshold
            return is_unusual, None, "insufficient history"
        
        is_unusual = abs(z_score) >= z_threshold
        
        # Build context string
        avg_move_pct = vol_stats["avg_move"] * 100
        max_move_pct = vol_stats["max_move"] * 100
        context = f"avg move: {avg_move_pct:.1f}%, max: {max_move_pct:.1f}%, samples: {vol_stats['samples']}"
        
        return is_unusual, z_score, context
    
    def get_dynamic_threshold(self, base_threshold: float) -> float:
        """Adjust threshold based on market's historical volatility"""
        vol_stats = self.get_volatility_stats()
        
        if vol_stats["samples"] < 10:
            return base_threshold
        
        # Use 2.5 standard deviations as the threshold
        # This means we alert on moves that are in the top ~1% of this market's moves
        dynamic_threshold = vol_stats["avg_move"] + (2.5 * vol_stats["volatility"])
        
        # Clamp between 0.5x and 3x the base threshold
        return max(base_threshold * 0.5, min(dynamic_threshold, base_threshold * 3.0))


@dataclass
class FlowAlert:
    """Unusual flow alert"""
    alert_type: str
    market_id: str
    token_id: str
    question: str
    timestamp: datetime
    severity: str
    reason: str
    details: Dict
    category: str = "other"
    
    def __str__(self):
        emoji = {"LOW": "info", "MEDIUM": "warning", "HIGH": "alert", "CRITICAL": "fire"}.get(self.severity, "?")
        return (f"\n[{emoji}] [{self.severity}] {self.alert_type}\n"
                f"   Market: {self.question[:60]}...\n"
                f"   Category: {self.category}\n"
                f"   Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"   Reason: {self.reason}\n"
                f"   Details: {json.dumps(self.details, indent=6)}")


class TradeFeedFlowDetector:
    """Flow detector using global trade feed with advanced features"""
    
    def __init__(self, min_trade_size: float = MIN_TRADE_SIZE_FOR_TRACKING, 
                 verbose: bool = False,
                 category_filter: MarketCategory = MarketCategory.ALL,
                 alert_callback: Optional[callable] = None,
                 storage: Optional["StorageBackend"] = None):
        """
        Initialize detector
        
        Args:
            min_trade_size: Minimum trade size (USD) to track
            verbose: Enable verbose logging (log every trade)
            category_filter: Filter by market category
            alert_callback: Optional callback function for alerts.
                           Called with FlowAlert as argument when alert fires.
                           Useful for integrating with trading bots.
            storage: Optional storage backend for persisting alerts
        """
        self.min_trade_size = min_trade_size
        self.verbose = verbose
        self.category_filter = category_filter
        self.alert_callback = alert_callback
        self.storage = storage
        
        # Dynamic state tracking
        self.market_states: Dict[str, MarketState] = {}  # token_id -> state
        self.wallet_profiles: Dict[str, WalletProfile] = {}
        self.market_info_cache: Dict[str, dict] = {}
        self.alerts: List[FlowAlert] = []
        
        # Alert deduplication: key = "token_id:alert_type" -> last alert time
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Trade tracking
        self.processed_trade_ids: Set[str] = set()
        self.last_trade_id: Optional[str] = None
        self._last_trade_time: datetime = datetime.now(timezone.utc)  # For stale connection detection
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        
        logger.info("Trade Feed Flow Detector initialized")
        logger.info(f"Min trade size: ${min_trade_size:,.2f}")
        logger.info(f"Category filter: {category_filter.value}")
        logger.info(f"Verbose mode: {'ON' if verbose else 'OFF'}")
        logger.info(f"Alert callback: {'configured' if alert_callback else 'none'}")
        logger.info(f"On-chain transfer detection: ENABLED")
        logger.info(f"Polygonscan API key: {'configured' if POLYGONSCAN_API_KEY else 'not set (rate limited)'}")
    
    def should_alert(self, token_id: str, alert_type: str) -> bool:
        """Check if alert should fire (respects cooldown)"""
        key = f"{token_id}:{alert_type}"
        last_alert = self.alert_cooldowns.get(key)
        
        if last_alert:
            elapsed = (datetime.now(timezone.utc) - last_alert).total_seconds()
            if elapsed < ALERT_COOLDOWN_SECONDS:
                return False
        
        self.alert_cooldowns[key] = datetime.now(timezone.utc)
        return True
    
    async def fetch_market_info(self, token_id: str) -> Optional[dict]:
        """Fetch market information (cached)"""
        if token_id in self.market_info_cache:
            return self.market_info_cache[token_id]
        
        try:
            async with self.session.get(
                f"{GAMMA_API_BASE}/markets",
                params={"active": "true", "limit": 1000},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    markets = await resp.json()
                    for market in markets:
                        token_ids = market.get("clobTokenIds", [])
                        if isinstance(token_ids, str):
                            token_ids = json.loads(token_ids)
                        
                        if token_id in token_ids:
                            self.market_info_cache[token_id] = market
                            return market
        except Exception as e:
            logger.debug(f"Error fetching market info: {e}")
        
        return None
    
    async def fetch_wallet_transfers(self, wallet: str) -> List[OnChainTransfer]:
        """Fetch on-chain token transfers for a wallet from Polygonscan
        
        This checks for USDC transfers to/from the wallet to identify
        wallets that are financially connected (funded by same source, 
        or transferring funds between each other).
        """
        transfers = []
        
        try:
            # Build API params
            params = {
                "module": "account",
                "action": "tokentx",  # Token transfers
                "address": wallet,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "desc",
                "page": 1,
                "offset": 100  # Last 100 transfers
            }
            
            if POLYGONSCAN_API_KEY:
                params["apikey"] = POLYGONSCAN_API_KEY
            
            async with self.session.get(
                POLYGONSCAN_API_BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"Polygonscan API error: {resp.status}")
                    return transfers
                
                data = await resp.json()
                
                if data.get("status") != "1" or not data.get("result"):
                    return transfers
                
                for tx in data["result"]:
                    # Filter for USDC/stablecoin transfers
                    token_symbol = tx.get("tokenSymbol", "").upper()
                    if token_symbol not in ["USDC", "USDC.E", "USDT", "DAI"]:
                        continue
                    
                    # Parse transfer value
                    try:
                        decimals = int(tx.get("tokenDecimal", 6))
                        raw_value = int(tx.get("value", 0))
                        value = raw_value / (10 ** decimals)
                    except (ValueError, TypeError):
                        continue
                    
                    # Skip small transfers
                    if value < MIN_TRANSFER_VALUE_USD:
                        continue
                    
                    # Parse timestamp
                    try:
                        ts = int(tx.get("timeStamp", 0))
                        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                    except (ValueError, TypeError):
                        timestamp = datetime.now(timezone.utc)
                    
                    transfer = OnChainTransfer(
                        from_address=tx.get("from", "").lower(),
                        to_address=tx.get("to", "").lower(),
                        value=value,
                        token_symbol=token_symbol,
                        tx_hash=tx.get("hash", ""),
                        timestamp=timestamp,
                        block_number=int(tx.get("blockNumber", 0))
                    )
                    transfers.append(transfer)
            
        except Exception as e:
            logger.debug(f"Error fetching wallet transfers for {wallet[:10]}...: {e}")
        
        return transfers
    
    async def fetch_wallet_chain_age(self, wallet: str) -> Tuple[Optional[datetime], int, Optional[str]]:
        """Fetch when a wallet was first created on-chain
        
        Returns:
            Tuple of (creation_date, total_tx_count, first_tx_hash)
            
        This uses Polygonscan to get the wallet's first ever transaction,
        which indicates when the wallet became active on-chain.
        """
        try:
            # Get normal transactions (oldest first)
            params = {
                "module": "account",
                "action": "txlist",
                "address": wallet,
                "startblock": 0,
                "endblock": 99999999,
                "sort": "asc",  # Oldest first
                "page": 1,
                "offset": 1  # Just need the first tx
            }
            
            if POLYGONSCAN_API_KEY:
                params["apikey"] = POLYGONSCAN_API_KEY
            
            async with self.session.get(
                POLYGONSCAN_API_BASE,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    return None, 0, None
                
                data = await resp.json()
                
                if data.get("status") != "1" or not data.get("result"):
                    # No transactions found - truly new wallet
                    return None, 0, None
                
                first_tx = data["result"][0]
                
                # Parse timestamp of first transaction
                try:
                    ts = int(first_tx.get("timeStamp", 0))
                    creation_date = datetime.fromtimestamp(ts, tz=timezone.utc)
                except (ValueError, TypeError):
                    creation_date = None
                
                first_tx_hash = first_tx.get("hash", "")
            
            # Now get total transaction count
            params_count = {
                "module": "proxy",
                "action": "eth_getTransactionCount",
                "address": wallet,
                "tag": "latest"
            }
            
            if POLYGONSCAN_API_KEY:
                params_count["apikey"] = POLYGONSCAN_API_KEY
            
            total_tx_count = 0
            try:
                async with self.session.get(
                    POLYGONSCAN_API_BASE,
                    params=params_count,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("result", "0x0")
                        if result and result.startswith("0x"):
                            total_tx_count = int(result, 16)
            except Exception:
                pass  # Non-critical, we still have creation date
            
            return creation_date, total_tx_count, first_tx_hash
            
        except Exception as e:
            logger.debug(f"Error fetching wallet chain age for {wallet[:10]}...: {e}")
            return None, 0, None
    
    async def update_wallet_chain_age(self, wallet: str):
        """Update a wallet's on-chain age information
        
        This is called once per wallet to determine how old it is on-chain.
        """
        if not wallet or wallet not in self.wallet_profiles:
            return
        
        profile = self.wallet_profiles[wallet]
        
        # Only check once per wallet
        if profile.chain_age_checked:
            return
        
        creation_date, tx_count, first_tx_hash = await self.fetch_wallet_chain_age(wallet)
        
        profile.chain_creation_date = creation_date
        profile.chain_total_tx_count = tx_count
        profile.chain_first_tx_hash = first_tx_hash
        profile.chain_age_checked = True
        
        if creation_date:
            age_days = profile.wallet_age_days
            if age_days is not None and age_days <= 7:
                logger.info(f"FRESH WALLET DETECTED: {wallet[:10]}... created {age_days:.1f} days ago, {tx_count} total txs")
    
    async def update_wallet_chain_connections(self, wallet: str):
        """Update a wallet's on-chain connections with other wallets
        
        This fetches recent token transfers and identifies wallets that
        have sent/received funds to/from this wallet.
        """
        if not wallet or wallet not in self.wallet_profiles:
            return
        
        profile = self.wallet_profiles[wallet]
        now = datetime.now(timezone.utc)
        
        # Check if we need to refresh (rate limit API calls)
        if profile.last_chain_check:
            elapsed = (now - profile.last_chain_check).total_seconds()
            if elapsed < CHAIN_CHECK_INTERVAL_SECONDS:
                return
        
        profile.last_chain_check = now
        wallet_lower = wallet.lower()
        
        # Fetch transfers
        transfers = await self.fetch_wallet_transfers(wallet)
        
        if not transfers:
            return
        
        # Process transfers and build connections
        for transfer in transfers:
            if transfer.from_address == wallet_lower:
                # Outgoing transfer - we funded another wallet
                other_wallet = transfer.to_address
                if other_wallet not in profile.funded_to:
                    profile.funded_to[other_wallet] = []
                profile.funded_to[other_wallet].append(transfer)
                
                # Also update the other wallet's profile if it exists
                if other_wallet in self.wallet_profiles:
                    other_profile = self.wallet_profiles[other_wallet]
                    if wallet_lower not in other_profile.funded_by:
                        other_profile.funded_by[wallet_lower] = []
                    other_profile.funded_by[wallet_lower].append(transfer)
                    
            elif transfer.to_address == wallet_lower:
                # Incoming transfer - we were funded by another wallet
                other_wallet = transfer.from_address
                if other_wallet not in profile.funded_by:
                    profile.funded_by[other_wallet] = []
                profile.funded_by[other_wallet].append(transfer)
                
                # Also update the other wallet's profile if it exists
                if other_wallet in self.wallet_profiles:
                    other_profile = self.wallet_profiles[other_wallet]
                    if wallet_lower not in other_profile.funded_to:
                        other_profile.funded_to[wallet_lower] = []
                    other_profile.funded_to[wallet_lower].append(transfer)
        
        if transfers:
            connected_count = len(profile.funded_by) + len(profile.funded_to)
            if connected_count > 0:
                logger.debug(f"Wallet {wallet[:10]}... has {connected_count} on-chain connections")
    
    def parse_rtds_message(self, data: Dict[str, Any]) -> Optional[TradeEvent]:
        """Parse a trade message from RTDS WebSocket"""
        try:
            payload = data.get("payload")
            if not payload:
                payload = data
            
            token_id = str(payload.get("asset") or payload.get("asset_id") or "")
            if not token_id:
                return None
            
            wallet = payload.get("proxyWallet") or payload.get("wallet") or ""
            timestamp_ms = data.get("timestamp") or payload.get("timestamp") or ""
            trade_id = f"{timestamp_ms}_{wallet}_{token_id}"
            
            if trade_id in self.processed_trade_ids:
                return None
            
            price = 0.0
            try:
                price = float(payload.get("price", 0))
            except:
                pass
            
            size = 0.0
            for field in ["size", "amount", "value", "shares"]:
                if field in payload:
                    try:
                        size = float(payload[field])
                        break
                    except:
                        pass
            
            if size == 0:
                size = 1
            
            timestamp = datetime.now(timezone.utc)
            ts_raw = data.get("timestamp") or payload.get("timestamp")
            if ts_raw:
                if isinstance(ts_raw, (int, float)):
                    if ts_raw > 1e12:
                        ts_raw = ts_raw / 1000
                    timestamp = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
                elif isinstance(ts_raw, str):
                    try:
                        timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except:
                        pass
            
            market_id = payload.get("conditionId") or payload.get("eventSlug") or token_id
            outcome = payload.get("outcome", "")
            event_slug = payload.get("eventSlug", "")
            
            side = "BUY"
            if outcome.lower() in ["no", "down"]:
                side = "SELL"
            
            trade = TradeEvent(
                trade_id=trade_id,
                token_id=token_id,
                market_id=market_id,
                side=side,
                size=size,
                price=price,
                timestamp=timestamp,
                maker_address=wallet,
                taker_address=wallet
            )
            
            if token_id:
                self.market_info_cache[token_id] = {
                    "question": event_slug.replace("-", " ").title() if event_slug else "Unknown",
                    "outcome": outcome,
                    "slug": event_slug,
                    "icon": payload.get("icon", ""),
                    "name": payload.get("name", "")
                }
            
            return trade
            
        except Exception as e:
            logger.debug(f"Error parsing RTDS message: {e}")
            return None
    
    async def connect_websocket(self):
        """Connect to RTDS WebSocket and stream trades"""
        reconnect_delay = 1
        max_reconnect_delay = 60
        
        while self.running:
            try:
                logger.info(f"Connecting to RTDS WebSocket: {RTDS_WEBSOCKET_URL}")
                
                async with websockets.connect(
                    RTDS_WEBSOCKET_URL,
                    open_timeout=30,
                    ping_interval=20,
                    ping_timeout=10
                ) as ws:
                    logger.info("Connected to RTDS WebSocket!")
                    reconnect_delay = 1
                    self._last_trade_time = datetime.now(timezone.utc)  # Reset on new connection
                    
                    subscribe_msg = json.dumps({
                        "action": "subscribe",
                        "subscriptions": [
                            {
                                "topic": "activity",
                                "type": "trades"
                            }
                        ]
                    })
                    await ws.send(subscribe_msg)
                    logger.info("Subscribed to activity/trades feed")
                    
                    # Use timeout-based receiving to detect stale connections
                    while self.running:
                        try:
                            message = await asyncio.wait_for(
                                ws.recv(), 
                                timeout=STALE_CONNECTION_TIMEOUT_SECONDS
                            )
                            self._last_trade_time = datetime.now(timezone.utc)
                            
                            try:
                                data = json.loads(message)
                                
                                if len(self.processed_trade_ids) < 5:
                                    logger.info(f"Raw message sample: {str(data)[:200]}...")
                                
                                if isinstance(data, list):
                                    for item in data:
                                        await self._process_ws_trade(item)
                                else:
                                    await self._process_ws_trade(data)
                                    
                            except json.JSONDecodeError:
                                logger.debug(f"Non-JSON message: {message[:100]}")
                            except Exception as e:
                                logger.debug(f"Error processing WS message: {e}")
                                
                        except asyncio.TimeoutError:
                            # No messages received within timeout - connection is stale
                            logger.warning(
                                f"⚠️ No trades received for {STALE_CONNECTION_TIMEOUT_SECONDS}s - "
                                f"connection stale, reconnecting..."
                            )
                            break  # Exit inner loop to trigger reconnection
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            if self.running:
                logger.info(f"Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
    
    async def _process_ws_trade(self, data: Dict[str, Any]):
        """Process a single trade from WebSocket"""
        trade = self.parse_rtds_message(data)
        if trade and trade.value_usd >= self.min_trade_size:
            self.processed_trade_ids.add(trade.trade_id)
            await self.process_trade(trade)
            
            # Always log significant trades (>= $500) for debugging stale connections
            if trade.value_usd >= 500:
                market_info = self.market_info_cache.get(trade.token_id, {})
                question = market_info.get("question", market_info.get("slug", "Unknown"))[:40]
                logger.info(
                    f"💰 TRADE: {trade.side} ${trade.value_usd:,.0f} @ {trade.price:.2f} | "
                    f"{question}..."
                )
            elif self.verbose:
                logger.info(f"   Trade: {trade.side} ${trade.value_usd:,.0f} @ {trade.price:.2f}")
    
    async def get_or_create_market_state(self, token_id: str) -> MarketState:
        """Get existing state or create new one on-demand"""
        if token_id in self.market_states:
            return self.market_states[token_id]
        
        # Fetch market info first to get the actual condition_id
        market_info = await self.fetch_market_info(token_id)
        
        # Use condition_id as market_id (same for both sides of a binary market)
        # This is critical for duplicate detection - both Lakers and Pistons
        # should have the same market_id
        condition_id = market_info.get("condition_id") if market_info else None
        
        state = MarketState(
            market_id=condition_id or token_id,  # Use condition_id, fallback to token_id
            token_id=token_id
        )
        if market_info:
            state.question = market_info.get("question", "Unknown")
            
            tags = market_info.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except:
                    tags = []
            
            state.category = categorize_market(state.question, tags)
            
            # Parse market timing for slippage calculations
            start_date_str = market_info.get("startDate") or market_info.get("start_date")
            end_date_str = market_info.get("endDate") or market_info.get("end_date")
            
            if start_date_str:
                try:
                    if isinstance(start_date_str, str):
                        state.start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
                    elif isinstance(start_date_str, (int, float)):
                        state.start_date = datetime.fromtimestamp(start_date_str, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass
            
            if end_date_str:
                try:
                    if isinstance(end_date_str, str):
                        state.end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    elif isinstance(end_date_str, (int, float)):
                        state.end_date = datetime.fromtimestamp(end_date_str, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass
        
        # Apply category filter
        if self.category_filter != MarketCategory.ALL:
            if state.category != self.category_filter:
                return None  # Don't track this market
        
        self.market_states[token_id] = state
        logger.info(f"NEW MARKET #{len(self.market_states)}: {state.question[:55]}...")
        duration_info = ""
        if state.lifetime_hours is not None:
            if state.lifetime_hours < MIN_MARKET_DURATION_HOURS:
                duration_info = f" | Duration: {state.lifetime_hours:.1f}h (short - skipping price alerts)"
            else:
                duration_info = f" | Duration: {state.lifetime_hours:.1f}h"
        logger.info(f"   Token ID: {token_id[:30]}... | Category: {state.category.value}{duration_info}")

        return state
    
    async def update_wallet_profile(self, wallet: str, trade: TradeEvent):
        """Update wallet profile and check for on-chain connections"""
        if not wallet:
            return
        
        now = datetime.now(timezone.utc)
        wallet_lower = wallet.lower()
        is_new_wallet = wallet_lower not in self.wallet_profiles
        
        if wallet_lower in self.wallet_profiles:
            profile = self.wallet_profiles[wallet_lower]
            profile.last_seen = now
            profile.total_trades += 1
            profile.total_volume_usd += trade.value_usd
            profile.markets_traded.add(trade.token_id)
            profile.is_cold = False
        else:
            profile = WalletProfile(
                address=wallet_lower,
                first_seen=now,
                last_seen=now,
                total_trades=1,
                total_volume_usd=trade.value_usd,
                markets_traded={trade.token_id},
                is_cold=True
            )
            self.wallet_profiles[wallet_lower] = profile
        
        # Track market/timing for coordination detection
        profile.recent_markets_with_timing.append((trade.token_id, now, trade.side))
        
        # For new wallets with significant trades, check their chain age
        # This tells us if it's a truly new wallet or just new to Polymarket
        if is_new_wallet and trade.value_usd >= 100:
            await self.update_wallet_chain_age(wallet_lower)
        
        # Check on-chain connections for this wallet (rate-limited internally)
        # Only check for significant trades to reduce API calls
        if trade.value_usd >= 500:
            await self.update_wallet_chain_connections(wallet_lower)
    
    # ==================== DETECTION ALGORITHMS ====================
    
    def detect_sudden_price_movement(self, state: MarketState) -> Optional[FlowAlert]:
        """Detect sudden price movements using RELATIVE volatility (Z-scores)

        A 5% move in a stable political market is HUGE
        A 5% move in a 15-min Bitcoin price market is NORMAL

        This uses Z-scores to measure how unusual a move is for THIS specific market.

        Note: Short-duration markets (< MIN_MARKET_DURATION_HOURS) are skipped
        because they can never accumulate enough history for statistical analysis.
        """
        # Skip short-duration markets - they can never build enough history
        # for meaningful statistical analysis. 15-minute markets will always
        # show "insufficient history" and generate noisy alerts.
        if state.lifetime_hours is not None and state.lifetime_hours < MIN_MARKET_DURATION_HOURS:
            return None

        change = state.get_price_change(PRICE_MOVEMENT_WINDOW_SECONDS)

        if change is None:
            return None

        abs_change = abs(change)

        # Use Z-score based detection - is this move unusual for THIS market?
        is_unusual, z_score, context = state.is_move_unusual(change, z_threshold=2.5)
        
        # Also check absolute threshold as a fallback
        abs_threshold = state.get_dynamic_threshold(PRICE_MOVEMENT_THRESHOLD)
        
        if is_unusual or abs_change >= abs_threshold:
            # Check cooldown - use market_id to prevent alerting on both sides
            if not self.should_alert(state.market_id, "SUDDEN_PRICE_MOVEMENT"):
                return None
            
            direction = "UP ⬆️" if change > 0 else "DOWN ⬇️"
            
            # Determine severity based on Z-score and absolute change
            if z_score is not None:
                if abs(z_score) >= 4.0 or abs_change >= 0.15:
                    severity = "CRITICAL"
                elif abs(z_score) >= 3.0 or abs_change >= 0.10:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"
            else:
                severity = "CRITICAL" if abs_change >= 0.10 else "HIGH"
            
            # Build reason with relative context
            if z_score is not None:
                reason = f"Price moved {direction} {abs_change*100:.1f}% ({abs(z_score):.1f}σ from normal)"
            else:
                reason = f"Price moved {direction} {abs_change*100:.1f}% in {PRICE_MOVEMENT_WINDOW_SECONDS}s"
            
            vol_stats = state.get_volatility_stats()
            
            return FlowAlert(
                alert_type="SUDDEN_PRICE_MOVEMENT",
                market_id=state.market_id,
                token_id=state.token_id,
                question=state.question,
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                reason=reason,
                details={
                    "price_change_pct": round(change * 100, 2),
                    "z_score": round(z_score, 2) if z_score else None,
                    "current_price": round(state.current_price, 4),
                    "market_avg_move_pct": round(vol_stats["avg_move"] * 100, 2),
                    "market_max_move_pct": round(vol_stats["max_move"] * 100, 2),
                    "market_volatility_pct": round(vol_stats["volatility"] * 100, 3),
                    "history_samples": vol_stats["samples"],
                    "relative_context": context,
                    **state.get_timing_details(),
                },
                category=state.category.value
            )
        
        return None
    
    def detect_oversized_bet(self, state: MarketState, trade: TradeEvent) -> Optional[FlowAlert]:
        """Detect oversized bets"""
        if state.avg_trade_size == 0:
            return None
        
        size_multiplier = trade.value_usd / state.avg_trade_size
        
        if size_multiplier >= OVERSIZED_BET_MULTIPLIER or trade.value_usd >= OVERSIZED_BET_MIN_USD:
            # Check cooldown - use market_id (condition_id) to prevent alerting on BOTH sides
            # of the same trade (BUY YES @ 0.57 = SELL NO @ 0.43 is ONE trade)
            if not self.should_alert(state.market_id, "OVERSIZED_BET"):
                return None
            
            severity = "CRITICAL" if size_multiplier >= 20 or trade.value_usd >= 50000 else "HIGH"
            
            return FlowAlert(
                alert_type="OVERSIZED_BET",
                market_id=state.market_id,
                token_id=state.token_id,
                question=state.question,
                timestamp=trade.timestamp,
                severity=severity,
                reason=f"Trade {size_multiplier:.1f}x larger than average (${trade.value_usd:,.2f})",
                details={
                    "trade_value_usd": trade.value_usd,
                    "avg_trade_size": state.avg_trade_size,
                    "size_multiplier": size_multiplier,
                    "side": trade.side,
                    "wallet": trade.taker_address or trade.maker_address,
                    "price": trade.price,
                    **state.get_timing_details(),
                },
                category=state.category.value
            )
        
        return None
    
    def detect_cold_wallet_activity(self, trade: TradeEvent, state: MarketState) -> Optional[FlowAlert]:
        """Detect suspicious wallet activity based on ON-CHAIN wallet age
        
        This looks for wallets that are:
        - Newly created on-chain (< 7 days old) placing significant trades
        - Fresh wallets (< 30 days old) with low transaction count
        - Wallets with very few prior on-chain transactions
        
        These are stronger signals than just "first Polymarket trade".
        """
        wallet = trade.taker_address or trade.maker_address
        if not wallet:
            return None
        
        wallet_lower = wallet.lower()
        profile = self.wallet_profiles.get(wallet_lower)
        
        if not profile or trade.value_usd < MIN_TRADE_SIZE_FOR_TRACKING:
            return None
        
        # Only alert on first Polymarket trade for this wallet
        if not profile.is_cold:
            return None
        
        # Check if we have on-chain age data
        if not profile.chain_age_checked:
            # Fall back to old behavior if we haven't checked chain age yet
            if not self.should_alert(wallet_lower, "COLD_WALLET_ACTIVITY"):
                return None
            
            return FlowAlert(
                alert_type="COLD_WALLET_ACTIVITY",
                market_id=trade.market_id,
                token_id=trade.token_id,
                question=state.question if state else "N/A",
                timestamp=trade.timestamp,
                severity="MEDIUM",
                reason=f"New Polymarket wallet placed ${trade.value_usd:,.2f} {trade.side} order (chain age pending)",
                details={
                    "wallet": wallet,
                    "trade_value_usd": trade.value_usd,
                    "side": trade.side,
                    "price": trade.price,
                    "chain_age_checked": False,
                    **(state.get_timing_details() if state else {}),
                },
                category=state.category.value if state else "other"
            )
        
        # Determine alert type and severity based on on-chain age
        wallet_age_days = profile.wallet_age_days
        tx_count = profile.chain_total_tx_count
        
        # Most suspicious: Newly created wallet (< 7 days)
        if profile.is_newly_created_wallet:
            if not self.should_alert(wallet_lower, "FRESH_WALLET_ACTIVITY"):
                return None
            
            # Severity based on trade size and wallet newness
            if trade.value_usd >= 25000 or (wallet_age_days and wallet_age_days <= 1):
                severity = "CRITICAL"
            elif trade.value_usd >= 5000 or (wallet_age_days and wallet_age_days <= 3):
                severity = "HIGH"
            else:
                severity = "MEDIUM"
            
            return FlowAlert(
                alert_type="FRESH_WALLET_ACTIVITY",
                market_id=trade.market_id,
                token_id=trade.token_id,
                question=state.question if state else "N/A",
                timestamp=trade.timestamp,
                severity=severity,
                reason=f"Newly created wallet ({wallet_age_days:.1f} days old) placed ${trade.value_usd:,.2f} {trade.side}",
                details={
                    "wallet": wallet,
                    "trade_value_usd": trade.value_usd,
                    "side": trade.side,
                    "price": trade.price,
                    "wallet_age_days": round(wallet_age_days, 1) if wallet_age_days else None,
                    "chain_tx_count": tx_count,
                    "first_tx_hash": profile.chain_first_tx_hash,
                    "chain_creation_date": profile.chain_creation_date.isoformat() if profile.chain_creation_date else None,
                    "detection_method": "blockchain_wallet_age",
                    **(state.get_timing_details() if state else {}),
                },
                category=state.category.value if state else "other"
            )
        
        # Suspicious: Fresh wallet (< 30 days) with low activity
        if profile.is_fresh_wallet and profile.is_low_activity_wallet:
            if not self.should_alert(wallet_lower, "COLD_WALLET_ACTIVITY"):
                return None
            
            if trade.value_usd >= 25000:
                severity = "HIGH"
            elif trade.value_usd >= 5000:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            return FlowAlert(
                alert_type="COLD_WALLET_ACTIVITY",
                market_id=trade.market_id,
                token_id=trade.token_id,
                question=state.question if state else "N/A",
                timestamp=trade.timestamp,
                severity=severity,
                reason=f"Low-activity wallet ({wallet_age_days:.0f} days old, {tx_count} txs) placed ${trade.value_usd:,.2f} {trade.side}",
                details={
                    "wallet": wallet,
                    "trade_value_usd": trade.value_usd,
                    "side": trade.side,
                    "price": trade.price,
                    "wallet_age_days": round(wallet_age_days, 1) if wallet_age_days else None,
                    "chain_tx_count": tx_count,
                    "chain_creation_date": profile.chain_creation_date.isoformat() if profile.chain_creation_date else None,
                    "detection_method": "blockchain_wallet_age",
                    **(state.get_timing_details() if state else {}),
                },
                category=state.category.value if state else "other"
            )
        
        # Very low activity wallet (< 10 txs) even if older
        if profile.is_low_activity_wallet and trade.value_usd >= 5000:
            if not self.should_alert(wallet_lower, "LOW_ACTIVITY_WALLET"):
                return None
            
            severity = "MEDIUM" if trade.value_usd >= 10000 else "LOW"
            
            return FlowAlert(
                alert_type="LOW_ACTIVITY_WALLET",
                market_id=trade.market_id,
                token_id=trade.token_id,
                question=state.question if state else "N/A",
                timestamp=trade.timestamp,
                severity=severity,
                reason=f"Low-activity wallet (only {tx_count} total txs) placed ${trade.value_usd:,.2f} {trade.side}",
                details={
                    "wallet": wallet,
                    "trade_value_usd": trade.value_usd,
                    "side": trade.side,
                    "price": trade.price,
                    "wallet_age_days": round(wallet_age_days, 1) if wallet_age_days else None,
                    "chain_tx_count": tx_count,
                    "detection_method": "blockchain_tx_count",
                    **(state.get_timing_details() if state else {}),
                },
                category=state.category.value if state else "other"
            )
        
        return None
    
    def detect_coordinated_wallets(self, state: MarketState, trade: TradeEvent) -> Optional[FlowAlert]:
        """Detect coordinated wallet activity based on ON-CHAIN TRANSFER patterns
        
        This detects wallets that are financially connected via blockchain transfers:
        - Wallets funded by the same source wallet
        - Wallets that have transferred funds to each other
        - Groups of wallets with shared funding chains
        
        This is a MUCH stronger signal than just trading at the same time,
        as it shows actual financial connections between the wallets.
        """
        wallet = trade.taker_address or trade.maker_address
        if not wallet:
            return None
        
        wallet_lower = wallet.lower()
        if wallet_lower not in self.wallet_profiles:
            return None
        
        profile = self.wallet_profiles[wallet_lower]
        
        # Get wallets that are connected via on-chain transfers
        connected_wallets = profile.get_connected_wallets(min_transfers=1)
        
        if not connected_wallets:
            return None
        
        # Check if any on-chain connected wallets also traded this market recently
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=COORDINATED_WALLET_WINDOW_SECONDS)
        related_wallets_trading = []
        total_transfer_value = 0.0
        
        for connected_wallet, transfer_count, transfer_value in connected_wallets:
            # Check both original case and lowercase in wallet_timestamps
            wallet_in_state = None
            for w in state.wallet_timestamps.keys():
                if w.lower() == connected_wallet.lower():
                    wallet_in_state = w
                    break
            
            if wallet_in_state and state.wallet_timestamps[wallet_in_state] >= cutoff:
                # Get transfer direction info
                is_funded_by = connected_wallet in profile.funded_by
                is_funded_to = connected_wallet in profile.funded_to
                
                if is_funded_by and is_funded_to:
                    relationship = "bidirectional"
                elif is_funded_by:
                    relationship = "funded_by"
                else:
                    relationship = "funded"
                
                related_wallets_trading.append({
                    "wallet": connected_wallet[:10] + "...",
                    "transfer_count": transfer_count,
                    "transfer_value_usd": round(transfer_value, 2),
                    "relationship": relationship,
                    "side": state.wallet_sides.get(wallet_in_state, "UNKNOWN")
                })
                total_transfer_value += transfer_value
        
        # Alert if 2+ wallets with on-chain connections are trading together
        if len(related_wallets_trading) >= 2:
            # Check cooldown - use market_id to prevent alerting on both sides
            if not self.should_alert(state.market_id, "COORDINATED_WALLETS"):
                return None
            
            total_transfers = sum(w["transfer_count"] for w in related_wallets_trading)
            
            # Severity based on transfer volume and count
            if total_transfer_value >= 50000 or len(related_wallets_trading) >= 4:
                severity = "CRITICAL"
            elif total_transfer_value >= 10000 or total_transfers >= 5:
                severity = "HIGH"
            else:
                severity = "MEDIUM"
            
            # Check if they're all on same side
            sides = [w["side"] for w in related_wallets_trading]
            same_side = len(set(sides)) == 1
            
            return FlowAlert(
                alert_type="COORDINATED_WALLETS",
                market_id=state.market_id,
                token_id=state.token_id,
                question=state.question,
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                reason=f"{len(related_wallets_trading)+1} on-chain connected wallets trading together (${total_transfer_value:,.0f} in prior transfers)",
                details={
                    "trigger_wallet": wallet[:10] + "...",
                    "connected_wallets": related_wallets_trading,
                    "total_on_chain_transfers": total_transfers,
                    "total_transfer_value_usd": round(total_transfer_value, 2),
                    "all_same_side": same_side,
                    "current_price": round(state.current_price, 4),
                    "detection_method": "blockchain_transfers",
                    **state.get_timing_details(),
                },
                category=state.category.value
            )
        
        return None
    
    def detect_volume_spike(self, state: MarketState) -> Optional[FlowAlert]:
        """Detect volume spikes with proper history requirements

        Requires sufficient trading history to avoid false positives on new markets.
        Short-duration markets are skipped as they lack baseline history.
        """
        # Skip short-duration markets - they can't build baseline volume history
        if state.lifetime_hours is not None and state.lifetime_hours < MIN_MARKET_DURATION_HOURS:
            return None

        # Get volumes for different time windows
        volume_1min = state.get_volume(1)
        volume_5min = state.get_volume(5)
        volume_30min = state.get_volume(30)
        
        # Require minimum history: at least 5 minutes of trading data
        # AND at least 5 trades in history to establish baseline
        if len(state.recent_trades) < 5:
            return None
        
        # Check if we have enough baseline volume (not a brand new market)
        # Need volume in the 5-30 min window to compare against
        baseline_volume = volume_30min - volume_5min  # Volume from 5-30 min ago
        
        if baseline_volume <= 0:
            # Not enough historical volume to compare - market too new
            return None
        
        # Calculate baseline average (volume per minute in 5-30 min window)
        baseline_avg_per_min = baseline_volume / 25  # 25 minutes of baseline
        
        # Minimum baseline threshold - need at least some activity to compare
        if baseline_avg_per_min < 10:  # Less than $10/min average
            return None
        
        # Calculate spike ratio against baseline (not including recent surge)
        spike_ratio = volume_1min / baseline_avg_per_min if baseline_avg_per_min > 0 else 0
        
        if spike_ratio >= VOLUME_SPIKE_MULTIPLIER:
            # Check cooldown - use market_id to prevent alerting on both sides
            if not self.should_alert(state.market_id, "VOLUME_SPIKE"):
                return None
            
            # Severity based on spike magnitude
            if spike_ratio >= 10:
                severity = "CRITICAL"
            elif spike_ratio >= 5:
                severity = "HIGH"
            else:
                severity = "MEDIUM"
            
            return FlowAlert(
                alert_type="VOLUME_SPIKE",
                market_id=state.market_id,
                token_id=state.token_id,
                question=state.question,
                timestamp=datetime.now(timezone.utc),
                severity=severity,
                reason=f"Volume {spike_ratio:.1f}x higher than baseline",
                details={
                    "volume_1min": round(volume_1min, 2),
                    "volume_5min": round(volume_5min, 2),
                    "baseline_avg_per_min": round(baseline_avg_per_min, 2),
                    "spike_ratio": round(spike_ratio, 1),
                    "trade_count": len(state.recent_trades),
                    **state.get_timing_details(),
                },
                category=state.category.value
            )
        
        return None
    
    def detect_smart_money_activity(self, trade: TradeEvent, state: MarketState) -> Optional[FlowAlert]:
        """Detect smart money activity"""
        wallet = trade.taker_address or trade.maker_address
        if not wallet:
            return None
        
        wallet_lower = wallet.lower()
        profile = self.wallet_profiles.get(wallet_lower)
        
        if profile and profile.is_smart_money:
            # Check cooldown
            if not self.should_alert(wallet, "SMART_MONEY_ACTIVITY"):
                return None
            
            return FlowAlert(
                alert_type="SMART_MONEY_ACTIVITY",
                market_id=trade.market_id,
                token_id=trade.token_id,
                question=state.question if state else "N/A",
                timestamp=trade.timestamp,
                severity="HIGH",
                reason=f"Smart money wallet ({profile.total_trades} trades, ${profile.total_volume_usd:,.0f} volume) placed ${trade.value_usd:,.2f} {trade.side}",
                details={
                    "wallet": wallet,
                    "total_trades": profile.total_trades,
                    "total_volume": profile.total_volume_usd,
                    "trade_value_usd": trade.value_usd,
                    "side": trade.side,
                    "price": trade.price,
                    **(state.get_timing_details() if state else {}),
                },
                category=state.category.value if state else "other"
            )
        
        return None
    
    def detect_short_duration_momentum(self, state: MarketState) -> Optional[FlowAlert]:
        """Detect momentum opportunities in short-duration markets (< 1 hour)

        For 15-minute crypto markets, we can't do statistical analysis, but we CAN
        trade momentum. This looks for:
        1. Strong directional price moves (3%+ in 30 seconds)
        2. Confirmed by multiple trades in same direction
        3. Trade in the direction of momentum

        Returns a signal with recommended direction (BUY/SELL).
        """
        if not SHORT_DURATION_MOMENTUM_ENABLED:
            return None

        # Only apply to short-duration markets
        if state.lifetime_hours is None or state.lifetime_hours >= MIN_MARKET_DURATION_HOURS:
            return None

        # Need minimum trades to confirm momentum
        if len(state.recent_trades) < SHORT_DURATION_MIN_TRADES:
            return None

        # Get recent price change
        change = state.get_price_change(SHORT_DURATION_MOMENTUM_WINDOW_SECONDS)
        if change is None:
            return None

        abs_change = abs(change)
        if abs_change < SHORT_DURATION_MOMENTUM_THRESHOLD:
            return None

        # Count trades in each direction within the window
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=SHORT_DURATION_MOMENTUM_WINDOW_SECONDS)
        buy_count = 0
        sell_count = 0
        buy_volume = 0.0
        sell_volume = 0.0

        for trade in state.recent_trades:
            if trade.timestamp >= cutoff:
                if trade.side == "BUY":
                    buy_count += 1
                    buy_volume += trade.value_usd
                else:
                    sell_count += 1
                    sell_volume += trade.value_usd

        total_trades = buy_count + sell_count
        if total_trades < SHORT_DURATION_MIN_TRADES:
            return None

        # Determine momentum direction
        if change > 0:
            momentum_direction = "BUY"
            dominant_volume = buy_volume
            momentum_ratio = buy_count / total_trades if total_trades > 0 else 0
        else:
            momentum_direction = "SELL"
            dominant_volume = sell_volume
            momentum_ratio = sell_count / total_trades if total_trades > 0 else 0

        # Require at least 60% of trades in momentum direction
        if momentum_ratio < 0.6:
            return None

        # Check cooldown - use market_id
        if not self.should_alert(state.market_id, "SHORT_DURATION_MOMENTUM"):
            return None

        # Calculate time remaining in market
        time_remaining_minutes = None
        if state.end_date:
            remaining = (state.end_date - datetime.now(timezone.utc)).total_seconds() / 60
            time_remaining_minutes = max(0, remaining)

        # Severity based on move size and volume
        if abs_change >= 0.05 and dominant_volume >= 5000:
            severity = "HIGH"
        elif abs_change >= 0.04 or dominant_volume >= 2000:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return FlowAlert(
            alert_type="SHORT_DURATION_MOMENTUM",
            market_id=state.market_id,
            token_id=state.token_id,
            question=state.question,
            timestamp=datetime.now(timezone.utc),
            severity=severity,
            reason=f"Short-duration momentum: {momentum_direction} ({abs_change*100:.1f}% in {SHORT_DURATION_MOMENTUM_WINDOW_SECONDS}s, {momentum_ratio*100:.0f}% trade flow)",
            details={
                "momentum_direction": momentum_direction,
                "price_change_pct": round(change * 100, 2),
                "current_price": round(state.current_price, 4),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_volume_usd": round(buy_volume, 2),
                "sell_volume_usd": round(sell_volume, 2),
                "momentum_ratio": round(momentum_ratio * 100, 1),
                "market_lifetime_hours": round(state.lifetime_hours, 2) if state.lifetime_hours else None,
                "time_remaining_minutes": round(time_remaining_minutes, 1) if time_remaining_minutes else None,
                "recommended_action": momentum_direction,
                **state.get_timing_details(),
            },
            category=state.category.value
        )

    def detect_price_acceleration(self, state: MarketState) -> Optional[FlowAlert]:
        """Detect when price movement is accelerating

        Short-duration markets are skipped as they lack sufficient price history.
        """
        # Skip short-duration markets - they can't accumulate 20+ price samples
        if state.lifetime_hours is not None and state.lifetime_hours < MIN_MARKET_DURATION_HOURS:
            return None

        if len(state.price_history) < 20:
            return None
        
        change_recent = state.get_price_change(15)
        change_earlier = state.get_price_change_window(45, 30)
        
        if change_recent is None or change_earlier is None:
            return None
        
        abs_recent = abs(change_recent)
        abs_earlier = abs(change_earlier)
        
        if abs_recent >= MIN_ACCELERATION_CHANGE and abs_earlier > 0:
            acceleration_ratio = abs_recent / abs_earlier
            
            if acceleration_ratio >= ACCELERATION_THRESHOLD:
                # Check cooldown - use market_id to prevent alerting on both sides
                if not self.should_alert(state.market_id, "PRICE_ACCELERATION"):
                    return None
                
                same_direction = (change_recent > 0) == (change_earlier > 0)
                
                if same_direction:
                    direction = "UP" if change_recent > 0 else "DOWN"
                    severity = "HIGH" if abs_recent >= 0.05 else "MEDIUM"
                    
                    return FlowAlert(
                        alert_type="PRICE_ACCELERATION",
                        market_id=state.market_id,
                        token_id=state.token_id,
                        question=state.question,
                        timestamp=datetime.now(timezone.utc),
                        severity=severity,
                        reason=f"Price {direction} movement accelerating: {abs_earlier*100:.1f}% -> {abs_recent*100:.1f}%",
                        details={
                            "recent_change_pct": round(change_recent * 100, 2),
                            "earlier_change_pct": round(change_earlier * 100, 2),
                            "acceleration_ratio": round(acceleration_ratio, 2),
                            "current_price": round(state.current_price, 4),
                            "direction": direction,
                            **state.get_timing_details(),
                        },
                        category=state.category.value
                    )
        
        return None
    
    def detect_correlated_movements(self) -> List[FlowAlert]:
        """Detect multiple markets moving together"""
        alerts = []
        now = datetime.now(timezone.utc)
        
        moving_markets = []
        
        for token_id, state in self.market_states.items():
            change = state.get_price_change(CORRELATION_WINDOW_SECONDS)
            if change is not None and abs(change) >= CORRELATION_MOVE_THRESHOLD:
                moving_markets.append({
                    "token_id": token_id,
                    "state": state,
                    "change": change,
                    "direction": "UP" if change > 0 else "DOWN"
                })
        
        up_markets = [m for m in moving_markets if m["direction"] == "UP"]
        down_markets = [m for m in moving_markets if m["direction"] == "DOWN"]
        
        if len(up_markets) >= MIN_CORRELATED_MARKETS:
            corr_key = "CORRELATION:UP:" + now.strftime("%Y%m%d%H%M")
            if self.should_alert(corr_key, "CORRELATED_MOVEMENT"):
                avg_change = sum(m["change"] for m in up_markets) / len(up_markets)
                
                alerts.append(FlowAlert(
                    alert_type="CORRELATED_MOVEMENT",
                    market_id="MULTIPLE",
                    token_id="MULTIPLE",
                    question=f"{len(up_markets)} markets moving UP together",
                    timestamp=now,
                    severity="HIGH" if len(up_markets) >= 5 else "MEDIUM",
                    reason=f"{len(up_markets)} markets moved UP {avg_change*100:.1f}% avg in {CORRELATION_WINDOW_SECONDS}s",
                    details={
                        "market_count": len(up_markets),
                        "avg_change_pct": round(avg_change * 100, 2),
                        "direction": "UP",
                        "markets": [
                            {
                                "question": m["state"].question[:40],
                                "change_pct": round(m["change"] * 100, 2),
                                "category": m["state"].category.value
                            }
                            for m in up_markets[:5]
                        ]
                    },
                    category="multiple"
                ))
        
        if len(down_markets) >= MIN_CORRELATED_MARKETS:
            corr_key = "CORRELATION:DOWN:" + now.strftime("%Y%m%d%H%M")
            if self.should_alert(corr_key, "CORRELATED_MOVEMENT"):
                avg_change = sum(m["change"] for m in down_markets) / len(down_markets)
                
                alerts.append(FlowAlert(
                    alert_type="CORRELATED_MOVEMENT",
                    market_id="MULTIPLE",
                    token_id="MULTIPLE",
                    question=f"{len(down_markets)} markets moving DOWN together",
                    timestamp=now,
                    severity="HIGH" if len(down_markets) >= 5 else "MEDIUM",
                    reason=f"{len(down_markets)} markets moved DOWN {abs(avg_change)*100:.1f}% avg in {CORRELATION_WINDOW_SECONDS}s",
                    details={
                        "market_count": len(down_markets),
                        "avg_change_pct": round(avg_change * 100, 2),
                        "direction": "DOWN",
                        "markets": [
                            {
                                "question": m["state"].question[:40],
                                "change_pct": round(m["change"] * 100, 2),
                                "category": m["state"].category.value
                            }
                            for m in down_markets[:5]
                        ]
                    },
                    category="multiple"
                ))
        
        return alerts
    
    # ==================== MAIN PROCESSING ====================
    
    async def process_trade(self, trade: TradeEvent):
        """Process a single trade"""
        # Get or create market state
        state = await self.get_or_create_market_state(trade.token_id)
        
        # If market was filtered out, skip processing
        if state is None:
            return
        
        # Verbose logging
        if self.verbose:
            logger.info(f"   Trade: ${trade.value_usd:,.2f} {trade.side} @ ${trade.price:.3f} | "
                       f"Wallet: {(trade.taker_address or 'N/A')[:12]}...")
        
        # Run detection algorithms BEFORE updating state
        # This ensures we compare against baseline that doesn't include this trade
        alerts = []
        
        # Per-trade detections (compare BEFORE adding to stats)
        alert = self.detect_oversized_bet(state, trade)
        if alert:
            alerts.append(alert)
        
        alert = self.detect_cold_wallet_activity(trade, state)
        if alert:
            alerts.append(alert)
        
        alert = self.detect_smart_money_activity(trade, state)
        if alert:
            alerts.append(alert)
        
        # Per-state detections
        alert = self.detect_sudden_price_movement(state)
        if alert:
            alerts.append(alert)
        
        alert = self.detect_volume_spike(state)
        if alert:
            alerts.append(alert)
        
        alert = self.detect_coordinated_wallets(state, trade)
        if alert:
            alerts.append(alert)
        
        alert = self.detect_price_acceleration(state)
        if alert:
            alerts.append(alert)
        
        # NOW update state after detection (so current trade doesn't affect its own baseline)
        state.add_trade(trade)

        # Short-duration momentum detection (runs AFTER adding trade to include it)
        # This is for 15-minute markets where we trade momentum instead of statistical anomalies
        alert = self.detect_short_duration_momentum(state)
        if alert:
            alerts.append(alert)

        # Update wallet profiles
        await self.update_wallet_profile(trade.taker_address, trade)
        await self.update_wallet_profile(trade.maker_address, trade)

        # Log alerts
        for alert in alerts:
            self.log_alert(alert)
    
    def log_alert(self, alert: FlowAlert):
        """Log alert and notify callback if configured"""
        self.alerts.append(alert)
        logger.warning(f"\n{alert}\n")
        
        # Save to storage if available
        if self.storage:
            try:
                with self.storage.transaction() as txn:
                    txn.save_alert(
                        alert_type=alert.alert_type,
                        market_id=alert.market_id,
                        token_id=alert.token_id,
                        question=alert.question,
                        timestamp=alert.timestamp,
                        severity=alert.severity,
                        reason=alert.reason,
                        details=alert.details,
                        category=alert.category,
                        score=alert.details.get("score")
                    )
            except Exception as e:
                logger.error(f"Failed to save alert to storage: {e}")
        
        # Call callback if configured
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def cleanup_inactive_markets(self):
        """Remove markets with no recent activity and enforce memory caps"""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=INACTIVE_MARKET_TIMEOUT_MINUTES)

        # Remove inactive markets
        inactive = [
            token_id for token_id, state in self.market_states.items()
            if state.last_update < cutoff
        ]

        for token_id in inactive:
            del self.market_states[token_id]

        # Enforce market state cap with LRU eviction
        if len(self.market_states) > MAX_MARKET_STATES:
            # Sort by last_update, remove oldest
            sorted_markets = sorted(
                self.market_states.items(),
                key=lambda x: x[1].last_update
            )
            to_remove = len(self.market_states) - MAX_MARKET_STATES
            for token_id, _ in sorted_markets[:to_remove]:
                del self.market_states[token_id]
            logger.info(f"LRU evicted {to_remove} markets (cap: {MAX_MARKET_STATES})")

        # Enforce wallet profile cap with LRU eviction
        if len(self.wallet_profiles) > MAX_WALLET_PROFILES:
            sorted_wallets = sorted(
                self.wallet_profiles.items(),
                key=lambda x: x[1].last_seen
            )
            to_remove = len(self.wallet_profiles) - MAX_WALLET_PROFILES
            for wallet, _ in sorted_wallets[:to_remove]:
                del self.wallet_profiles[wallet]
            logger.info(f"LRU evicted {to_remove} wallet profiles (cap: {MAX_WALLET_PROFILES})")

        # Cap processed trade IDs set
        if len(self.processed_trade_ids) > MAX_PROCESSED_TRADE_IDS:
            # Convert to list, keep most recent half
            trade_list = list(self.processed_trade_ids)
            self.processed_trade_ids = set(trade_list[-(MAX_PROCESSED_TRADE_IDS // 2):])
            logger.debug(f"Trimmed processed_trade_ids to {len(self.processed_trade_ids)}")

        # Clean up old alert cooldowns
        now = datetime.now(timezone.utc)
        old_cooldowns = [
            k for k, v in self.alert_cooldowns.items()
            if (now - v).total_seconds() > ALERT_COOLDOWN_SECONDS * 2
        ]
        for k in old_cooldowns:
            del self.alert_cooldowns[k]

        if inactive:
            logger.info(f"Cleaned up {len(inactive)} inactive markets. Active: {len(self.market_states)}")
    
    async def correlation_check_loop(self):
        """Periodic loop to check for cross-market correlations"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                correlation_alerts = self.detect_correlated_movements()
                for alert in correlation_alerts:
                    self.log_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error in correlation check loop: {e}")
    
    async def stats_loop(self):
        """Periodic stats and cleanup loop"""
        last_cleanup = datetime.now(timezone.utc)
        
        while self.running:
            try:
                await asyncio.sleep(30)
                
                now = datetime.now(timezone.utc)
                
                # Count active cooldowns
                active_cooldowns = sum(
                    1 for t in self.alert_cooldowns.values()
                    if (now - t).total_seconds() < ALERT_COOLDOWN_SECONDS
                )
                
                logger.info(f"STATUS [{now.strftime('%H:%M:%S')}] | "
                          f"Markets: {len(self.market_states)} | "
                          f"Wallets: {len(self.wallet_profiles)} | "
                          f"Alerts: {len(self.alerts)} | "
                          f"Cooldowns: {active_cooldowns} | "
                          f"Trades: {len(self.processed_trade_ids)}")
                
                if (now - last_cleanup).total_seconds() >= STATE_CLEANUP_INTERVAL_SECONDS:
                    self.cleanup_inactive_markets()
                    last_cleanup = now
                    
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop using WebSocket"""
        logger.info("=" * 60)
        logger.info("POLYMARKET FLOW DETECTOR V2 - Starting")
        logger.info("=" * 60)
        logger.info(f"Data source: RTDS WebSocket")
        logger.info(f"   {RTDS_WEBSOCKET_URL}")
        logger.info(f"Min trade size: ${self.min_trade_size:,.0f}")
        logger.info(f"Category filter: {self.category_filter.value}")
        logger.info(f"Alert cooldown: {ALERT_COOLDOWN_SECONDS}s")
        logger.info("Features: On-Chain Wallet Age, Blockchain Transfers, Dynamic Thresholds, Correlation")
        logger.info("=" * 60)
        
        # Start background loops
        stats_task = asyncio.create_task(self.stats_loop())
        correlation_task = asyncio.create_task(self.correlation_check_loop())
        
        try:
            await self.connect_websocket()
        finally:
            stats_task.cancel()
            correlation_task.cancel()
            try:
                await stats_task
                await correlation_task
            except asyncio.CancelledError:
                pass
    
    async def start(self):
        """Start detector"""
        self.running = True
        
        start_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.info("\n" + "="*70)
        logger.info("POLYMARKET FLOW DETECTOR V2 - STARTING")
        logger.info("="*70)
        logger.info(f"Start Time: {start_time}")
        logger.info(f"Min Trade Size: ${self.min_trade_size:,.2f}")
        logger.info(f"Category Filter: {self.category_filter.value}")
        logger.info(f"Poll Interval: {TRADE_FEED_POLL_INTERVAL}s")
        logger.info(f"Verbose Mode: {'ON' if self.verbose else 'OFF'}")
        logger.info(f"Detection Thresholds:")
        logger.info(f"  - Price Movement: {PRICE_MOVEMENT_THRESHOLD*100}% in {PRICE_MOVEMENT_WINDOW_SECONDS}s")
        logger.info(f"  - Oversized Bet: {OVERSIZED_BET_MULTIPLIER}x avg or >${OVERSIZED_BET_MIN_USD:,}")
        logger.info(f"  - Volume Spike: {VOLUME_SPIKE_MULTIPLIER}x avg")
        logger.info(f"  - Coordinated Wallets: On-chain transfer detection (2+ connected wallets in {COORDINATED_WALLET_WINDOW_SECONDS}s)")
        logger.info(f"  - Alert Cooldown: {ALERT_COOLDOWN_SECONDS}s")
        logger.info("="*70 + "\n")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            await self.monitor_loop()
    
    async def stop(self):
        """Stop detector"""
        self.running = False
        logger.info(f"Stopped. Total alerts: {len(self.alerts)}, Markets tracked: {len(self.market_states)}")


async def main():
    parser = argparse.ArgumentParser(description="Polymarket Flow Detector V2 (Trade Feed)")
    parser.add_argument(
        "--min-trade-size",
        type=float,
        default=MIN_TRADE_SIZE_FOR_TRACKING,
        help=f"Minimum trade size in USD to track (default: ${MIN_TRADE_SIZE_FOR_TRACKING})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (log every trade)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        choices=["all", "crypto", "sports", "politics", "entertainment", "finance", "other"],
        help="Filter by market category (default: all)"
    )
    
    args = parser.parse_args()
    
    # Convert category string to enum
    category_map = {
        "all": MarketCategory.ALL,
        "crypto": MarketCategory.CRYPTO,
        "sports": MarketCategory.SPORTS,
        "politics": MarketCategory.POLITICS,
        "entertainment": MarketCategory.ENTERTAINMENT,
        "finance": MarketCategory.FINANCE,
        "other": MarketCategory.OTHER
    }
    category_filter = category_map.get(args.category, MarketCategory.ALL)
    
    detector = TradeFeedFlowDetector(
        min_trade_size=args.min_trade_size,
        verbose=args.verbose,
        category_filter=category_filter
    )
    
    try:
        await detector.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await detector.stop()


if __name__ == "__main__":
    asyncio.run(main())
