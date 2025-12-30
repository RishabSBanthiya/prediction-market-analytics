"""
Wallet Profiler for Backtesting.

Builds wallet profiles from historical trade data to identify smart money,
coordinated wallets, and other wallet-based trading signals.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Set, Optional, Tuple

from .data.trade_fetcher import TradeData

logger = logging.getLogger(__name__)

# Smart money thresholds
SMART_MONEY_MIN_TRADES = 10
SMART_MONEY_MIN_VOLUME_USD = 50000
OVERSIZED_BET_MULTIPLIER = 10.0
OVERSIZED_BET_MIN_USD = 10000
COORDINATED_WALLET_WINDOW_SECONDS = 60
MIN_COORDINATED_WALLETS = 3


@dataclass
class WalletStats:
    """Statistics for a single wallet."""
    address: str
    first_seen: datetime
    last_seen: datetime
    total_trades: int = 0
    total_volume_usd: float = 0.0
    total_buys: int = 0
    total_sells: int = 0
    buy_volume_usd: float = 0.0
    sell_volume_usd: float = 0.0
    markets_traded: Set[str] = field(default_factory=set)
    avg_trade_size_usd: float = 0.0
    largest_trade_usd: float = 0.0
    
    # On-chain data (populated separately)
    chain_creation_date: Optional[datetime] = None
    chain_tx_count: int = 0
    chain_age_checked: bool = False
    
    # Win rate tracking (populated during backtest evaluation)
    profitable_trades: int = 0
    unprofitable_trades: int = 0
    
    @property
    def is_smart_money(self) -> bool:
        """Check if wallet qualifies as smart money based on activity."""
        return (
            self.total_trades >= SMART_MONEY_MIN_TRADES and 
            self.total_volume_usd >= SMART_MONEY_MIN_VOLUME_USD
        )
    
    @property
    def is_whale(self) -> bool:
        """Check if wallet is a whale (very high volume)."""
        return self.total_volume_usd >= 500000
    
    @property
    def is_active_trader(self) -> bool:
        """Check if wallet is actively trading."""
        return self.total_trades >= 5 and self.total_volume_usd >= 5000
    
    @property
    def buy_ratio(self) -> float:
        """Ratio of buys to total trades."""
        if self.total_trades == 0:
            return 0.5
        return self.total_buys / self.total_trades
    
    @property
    def wallet_age_days(self) -> Optional[float]:
        """Age of wallet on-chain in days."""
        if not self.chain_creation_date:
            return None
        delta = datetime.now(timezone.utc) - self.chain_creation_date
        return delta.total_seconds() / 86400
    
    @property
    def is_fresh_wallet(self) -> bool:
        """Wallet created on-chain within last 30 days."""
        age = self.wallet_age_days
        return age is not None and age <= 30
    
    @property
    def is_cold_wallet(self) -> bool:
        """Wallet with no prior Polymarket activity (first trade)."""
        return self.total_trades == 1
    
    @property
    def win_rate(self) -> float:
        """Win rate of trades (if tracked)."""
        total = self.profitable_trades + self.unprofitable_trades
        if total == 0:
            return 0.0
        return self.profitable_trades / total


class BacktestWalletProfiler:
    """
    Builds and manages wallet profiles from historical trade data.
    
    Used by the flow backtester to identify smart money and generate
    wallet-based trading signals.
    """
    
    def __init__(self):
        self.profiles: Dict[str, WalletStats] = {}
        self.market_avg_trade_size: Dict[str, float] = {}
        self._trades_by_market: Dict[str, List[TradeData]] = defaultdict(list)
        self._trades_by_time: List[TradeData] = []  # Sorted by timestamp
        
    def build_profiles_from_trades(self, trades: List[TradeData]) -> int:
        """
        Build wallet profiles from historical trade data.
        
        Args:
            trades: List of TradeData objects
            
        Returns:
            Number of unique wallets profiled
        """
        if not trades:
            return 0
        
        # Sort trades by timestamp for time-based analysis
        self._trades_by_time = sorted(trades, key=lambda t: t.timestamp)
        
        # Group by market for average calculations
        for trade in trades:
            self._trades_by_market[trade.market_id].append(trade)
        
        # Calculate market average trade sizes
        for market_id, market_trades in self._trades_by_market.items():
            if market_trades:
                avg = sum(t.value_usd for t in market_trades) / len(market_trades)
                self.market_avg_trade_size[market_id] = avg
        
        # Build wallet profiles
        for trade in trades:
            wallet = trade.active_wallet
            if not wallet:
                continue
            
            wallet_lower = wallet.lower()
            
            if wallet_lower not in self.profiles:
                self.profiles[wallet_lower] = WalletStats(
                    address=wallet_lower,
                    first_seen=trade.timestamp,
                    last_seen=trade.timestamp,
                )
            
            profile = self.profiles[wallet_lower]
            
            # Update stats
            profile.total_trades += 1
            profile.total_volume_usd += trade.value_usd
            profile.markets_traded.add(trade.market_id)
            profile.last_seen = max(profile.last_seen, trade.timestamp)
            profile.first_seen = min(profile.first_seen, trade.timestamp)
            profile.largest_trade_usd = max(profile.largest_trade_usd, trade.value_usd)
            
            if trade.side == "BUY":
                profile.total_buys += 1
                profile.buy_volume_usd += trade.value_usd
            else:
                profile.total_sells += 1
                profile.sell_volume_usd += trade.value_usd
        
        # Calculate averages
        for profile in self.profiles.values():
            if profile.total_trades > 0:
                profile.avg_trade_size_usd = profile.total_volume_usd / profile.total_trades
        
        logger.info(
            f"Built profiles for {len(self.profiles)} wallets from {len(trades)} trades"
        )
        
        # Log smart money wallets found
        smart_money_count = sum(1 for p in self.profiles.values() if p.is_smart_money)
        whale_count = sum(1 for p in self.profiles.values() if p.is_whale)
        logger.info(f"Found {smart_money_count} smart money wallets, {whale_count} whales")
        
        return len(self.profiles)
    
    def update_with_chain_data(self, chain_data: Dict[str, dict]):
        """
        Update wallet profiles with on-chain validation data.
        
        Args:
            chain_data: Dict mapping wallet -> {creation_date, tx_count, ...}
        """
        updated = 0
        for wallet, data in chain_data.items():
            wallet_lower = wallet.lower()
            if wallet_lower in self.profiles:
                profile = self.profiles[wallet_lower]
                profile.chain_creation_date = data.get("creation_date")
                profile.chain_tx_count = data.get("tx_count", 0)
                profile.chain_age_checked = True
                updated += 1
        
        logger.info(f"Updated {updated} profiles with on-chain data")
    
    def get_profile(self, wallet: str) -> Optional[WalletStats]:
        """Get profile for a wallet address."""
        return self.profiles.get(wallet.lower())
    
    def is_smart_money(self, wallet: str) -> bool:
        """Check if a wallet is classified as smart money."""
        profile = self.get_profile(wallet)
        return profile is not None and profile.is_smart_money
    
    def is_whale(self, wallet: str) -> bool:
        """Check if a wallet is a whale."""
        profile = self.get_profile(wallet)
        return profile is not None and profile.is_whale
    
    def is_oversized_bet(self, trade: TradeData) -> bool:
        """
        Check if a trade is an oversized bet.
        
        An oversized bet is significantly larger than the market average.
        """
        if trade.value_usd < OVERSIZED_BET_MIN_USD:
            return False
        
        avg_size = self.market_avg_trade_size.get(trade.market_id, 100)
        return trade.value_usd > avg_size * OVERSIZED_BET_MULTIPLIER
    
    def find_coordinated_trades(
        self,
        trade: TradeData,
        window_seconds: int = COORDINATED_WALLET_WINDOW_SECONDS,
        min_wallets: int = MIN_COORDINATED_WALLETS,
    ) -> List[str]:
        """
        Find wallets that traded in coordination with this trade.
        
        Looks for multiple unique wallets trading the same side
        within a time window.
        
        Args:
            trade: The trade to check
            window_seconds: Time window to look for coordinated trades
            min_wallets: Minimum wallets needed to be "coordinated"
            
        Returns:
            List of coordinated wallet addresses (empty if none found)
        """
        window_start = trade.timestamp - timedelta(seconds=window_seconds)
        window_end = trade.timestamp + timedelta(seconds=window_seconds)
        
        # Find trades in window with same market and side
        coordinated_wallets = set()
        
        for t in self._trades_by_market.get(trade.market_id, []):
            if window_start <= t.timestamp <= window_end:
                if t.side == trade.side and t.active_wallet:
                    coordinated_wallets.add(t.active_wallet.lower())
        
        if len(coordinated_wallets) >= min_wallets:
            return list(coordinated_wallets)
        
        return []
    
    def get_smart_money_wallets(self) -> List[str]:
        """Get list of all smart money wallet addresses."""
        return [
            wallet for wallet, profile in self.profiles.items()
            if profile.is_smart_money
        ]
    
    def get_whale_wallets(self) -> List[str]:
        """Get list of all whale wallet addresses."""
        return [
            wallet for wallet, profile in self.profiles.items()
            if profile.is_whale
        ]
    
    def get_stats_summary(self) -> dict:
        """Get summary statistics of all profiles."""
        if not self.profiles:
            return {}
        
        volumes = [p.total_volume_usd for p in self.profiles.values()]
        trades = [p.total_trades for p in self.profiles.values()]
        
        return {
            "total_wallets": len(self.profiles),
            "smart_money_count": sum(1 for p in self.profiles.values() if p.is_smart_money),
            "whale_count": sum(1 for p in self.profiles.values() if p.is_whale),
            "fresh_wallet_count": sum(1 for p in self.profiles.values() if p.is_fresh_wallet),
            "cold_wallet_count": sum(1 for p in self.profiles.values() if p.is_cold_wallet),
            "avg_volume_usd": sum(volumes) / len(volumes) if volumes else 0,
            "avg_trades": sum(trades) / len(trades) if trades else 0,
            "total_volume_usd": sum(volumes),
            "total_trades": sum(trades),
        }
    
    def print_summary(self):
        """Print a summary of wallet profiles."""
        stats = self.get_stats_summary()
        
        print("\n" + "=" * 50)
        print("WALLET PROFILE SUMMARY")
        print("=" * 50)
        print(f"Total Wallets:      {stats.get('total_wallets', 0):,}")
        print(f"Smart Money:        {stats.get('smart_money_count', 0):,}")
        print(f"Whales:             {stats.get('whale_count', 0):,}")
        print(f"Fresh Wallets:      {stats.get('fresh_wallet_count', 0):,}")
        print(f"Cold Wallets:       {stats.get('cold_wallet_count', 0):,}")
        print(f"Total Volume:       ${stats.get('total_volume_usd', 0):,.2f}")
        print(f"Total Trades:       {stats.get('total_trades', 0):,}")
        print("=" * 50 + "\n")

