"""
Flow Strategy Enhancements Module.

Contains improvements to address the flow strategy V3 optimization issues:
1. Wallet reputation scoring - Track wallet historical accuracy
2. Multi-signal confirmation - Require multiple confirming signals
3. Information timing filter - Only trade on early signals
4. Contrarian signal detection - Fade retail when smart money disagrees
5. Market context filters - Adjust signal weight based on market characteristics
6. Dynamic position sizing - Scale position with signal confidence
7. Adaptive exits - Exit parameters based on market dynamics

These improvements address the mismatch between the simple backtest (price momentum only)
and the live strategy (rich flow detection with wallet profiles, coordinated activity, etc.)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Set, Tuple, Any
from collections import defaultdict

from ..components.exit_strategies import ExitConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Priority 1: Wallet Reputation Scoring
# =============================================================================

@dataclass
class WalletScore:
    """
    Track wallet historical accuracy before copying trades.

    Wallets must demonstrate consistent profitability before
    their trades are copied with confidence.
    """
    address: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    markets_traded: Set[str] = field(default_factory=set)
    last_trade_time: Optional[datetime] = None

    # Track recent performance for recency weighting
    recent_trades: int = 0  # Trades in last 30 days
    recent_wins: int = 0
    recent_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def recent_win_rate(self) -> float:
        """Win rate in last 30 days."""
        return self.recent_wins / self.recent_trades if self.recent_trades > 0 else 0.0

    @property
    def is_smart_money(self) -> bool:
        """
        Determine if wallet qualifies as smart money.

        Requirements:
        - At least 10 trades total
        - Win rate >= 55%
        - Positive total PnL
        """
        return (
            self.trades >= 10 and
            self.win_rate >= 0.55 and
            self.total_pnl > 0
        )

    @property
    def is_elite_trader(self) -> bool:
        """
        Identify truly exceptional traders.

        Requirements:
        - At least 25 trades
        - Win rate >= 60%
        - Avg PnL per trade > $100
        - Recent performance remains strong (>= 55% recent win rate)
        """
        return (
            self.trades >= 25 and
            self.win_rate >= 0.60 and
            self.avg_pnl_per_trade > 100 and
            (self.recent_trades < 5 or self.recent_win_rate >= 0.55)
        )

    @property
    def reputation_score(self) -> float:
        """
        Calculate overall reputation score (0-100).

        Components:
        - Win rate contribution (0-40 points)
        - Trade volume contribution (0-20 points)
        - PnL contribution (0-20 points)
        - Recency contribution (0-20 points)
        """
        score = 0.0

        # Win rate: 0-40 points
        # 50% = 0, 60% = 20, 70% = 40
        if self.trades >= 5:
            wr_score = max(0, (self.win_rate - 0.5) * 200)  # 0-40
            score += min(40, wr_score)

        # Trade volume: 0-20 points
        # More trades = more confidence in the signal
        volume_score = min(20, self.trades / 5)  # Full points at 100 trades
        score += volume_score

        # PnL: 0-20 points
        if self.total_pnl > 0:
            pnl_score = min(20, self.total_pnl / 5000 * 20)  # Full points at $5000 profit
            score += pnl_score

        # Recency: 0-20 points
        # Recent performance should match historical
        if self.recent_trades >= 3:
            recency_score = self.recent_win_rate * 20  # 0-20
            score += recency_score
        elif self.trades >= 10:
            # Use historical if not enough recent data
            score += self.win_rate * 10

        return min(100, score)

    def update_trade_result(
        self,
        pnl: float,
        market_id: str,
        trade_time: Optional[datetime] = None
    ) -> None:
        """
        Update wallet score with a new trade result.

        Args:
            pnl: Profit/loss from the trade
            market_id: Market condition ID
            trade_time: When the trade closed (default: now)
        """
        self.trades += 1
        self.total_pnl += pnl
        self.markets_traded.add(market_id)

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        self.avg_pnl_per_trade = self.total_pnl / self.trades

        trade_time = trade_time or datetime.now(timezone.utc)
        self.last_trade_time = trade_time

        # Update recent stats (last 30 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        if trade_time > cutoff:
            self.recent_trades += 1
            self.recent_pnl += pnl
            if pnl > 0:
                self.recent_wins += 1


class WalletReputationTracker:
    """
    Centralized tracking of wallet reputation scores.

    Maintains a database of wallet performance and determines
    whether trades from specific wallets should be copied.
    """

    def __init__(
        self,
        min_trades_for_copy: int = 10,
        min_win_rate: float = 0.55,
        elite_min_trades: int = 25,
        elite_min_win_rate: float = 0.60,
    ):
        self.min_trades_for_copy = min_trades_for_copy
        self.min_win_rate = min_win_rate
        self.elite_min_trades = elite_min_trades
        self.elite_min_win_rate = elite_min_win_rate

        self._wallets: Dict[str, WalletScore] = {}

    def get_wallet(self, address: str) -> WalletScore:
        """Get or create wallet score."""
        address = address.lower()
        if address not in self._wallets:
            self._wallets[address] = WalletScore(address=address)
        return self._wallets[address]

    def update_result(
        self,
        address: str,
        pnl: float,
        market_id: str,
        trade_time: Optional[datetime] = None
    ) -> WalletScore:
        """Update wallet with trade result."""
        wallet = self.get_wallet(address)
        wallet.update_trade_result(pnl, market_id, trade_time)
        return wallet

    def should_copy_trade(
        self,
        wallet: WalletScore,
        trade_size: float,
        is_urgent: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if we should copy a trade from this wallet.

        Args:
            wallet: Wallet score to evaluate
            trade_size: Size of the trade in USD
            is_urgent: If True, lower requirements for time-sensitive trades

        Returns:
            Tuple of (should_copy, reason)
        """
        # Elite traders: always copy if trade is significant
        if wallet.is_elite_trader:
            if trade_size >= 1000:
                return True, f"Elite trader ({wallet.win_rate:.1%} WR, {wallet.trades} trades)"
            # Even small trades from elite traders are worth watching
            if trade_size >= 500:
                return True, f"Elite trader small position"

        # Smart money: copy larger trades
        if wallet.is_smart_money:
            # Require larger trades from newer smart money wallets
            min_size = 1000 if wallet.trades >= 50 else 5000
            if trade_size >= min_size:
                return True, f"Smart money ({wallet.win_rate:.1%} WR, ${trade_size:,.0f})"

        # Unknown wallet with very large trade - may be worth watching
        if wallet.trades < self.min_trades_for_copy:
            if trade_size >= 25000:
                return True, f"Large unknown wallet trade (${trade_size:,.0f})"
            if is_urgent and trade_size >= 10000:
                return True, f"Urgent large trade from new wallet"
            return False, f"Insufficient history ({wallet.trades} trades)"

        # Known wallet but poor performance
        if wallet.win_rate < self.min_win_rate:
            return False, f"Poor win rate ({wallet.win_rate:.1%})"

        return False, "Does not meet criteria"

    def get_top_wallets(self, limit: int = 10) -> List[WalletScore]:
        """Get top performing wallets by reputation score."""
        wallets = list(self._wallets.values())
        wallets.sort(key=lambda w: w.reputation_score, reverse=True)
        return wallets[:limit]


# =============================================================================
# Priority 2: Multi-Signal Confirmation
# =============================================================================

@dataclass
class SignalCluster:
    """
    Group of related signals that confirm a trade direction.

    Requires multiple independent signal types before trading
    to reduce false positives.
    """
    token_id: str
    market_id: str
    direction: str  # "BUY" or "SELL"
    signals: List[str] = field(default_factory=list)  # Signal types
    total_score: float = 0.0
    wallet_addresses: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def unique_signal_types(self) -> Set[str]:
        """Get unique signal type categories."""
        # Extract category from signal type (e.g., "whale_buy" -> "whale")
        categories = set()
        for signal in self.signals:
            # Handle both underscore and camelCase
            if "_" in signal:
                categories.add(signal.split("_")[0].lower())
            else:
                # CamelCase: extract first word
                import re
                match = re.match(r'([A-Z]?[a-z]+)', signal)
                if match:
                    categories.add(match.group(1).lower())
                else:
                    categories.add(signal.lower())
        return categories

    @property
    def signal_count(self) -> int:
        """Number of unique signal types."""
        return len(self.unique_signal_types)

    @property
    def is_strong(self) -> bool:
        """
        Determine if signal cluster is strong enough to trade.

        Requirements:
        - At least 2 unique signal type categories
        - Total score >= 50
        """
        return self.signal_count >= 2 and self.total_score >= 50

    @property
    def is_very_strong(self) -> bool:
        """
        Identify exceptionally strong signal clusters.

        Requirements:
        - At least 3 unique signal types
        - Total score >= 80
        - Multiple wallets involved
        """
        return (
            self.signal_count >= 3 and
            self.total_score >= 80 and
            len(self.wallet_addresses) >= 2
        )

    @property
    def confirmation_score(self) -> float:
        """
        Calculate confirmation strength (0-100).

        Higher scores indicate more independent confirmation.
        """
        base_score = min(50, self.total_score / 2)

        # Bonus for multiple signal types
        type_bonus = self.signal_count * 10  # Up to 30 points

        # Bonus for multiple wallets
        wallet_bonus = min(20, len(self.wallet_addresses) * 5)

        return min(100, base_score + type_bonus + wallet_bonus)

    def add_signal(
        self,
        signal_type: str,
        score: float,
        wallet: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add a signal to the cluster."""
        self.signals.append(signal_type)
        self.total_score += score

        if wallet:
            self.wallet_addresses.add(wallet.lower())

        if metadata:
            self.metadata.update(metadata)


class MultiSignalConfirmation:
    """
    Aggregates and confirms signals before trading.

    Requires multiple independent signal types to fire within
    a time window before generating a confirmed trade signal.
    """

    def __init__(
        self,
        confirmation_window_seconds: int = 60,
        min_signal_types: int = 2,
        min_total_score: float = 50.0,
    ):
        self.confirmation_window = confirmation_window_seconds
        self.min_signal_types = min_signal_types
        self.min_total_score = min_total_score

        # Track pending signals by token_id
        self._pending: Dict[str, SignalCluster] = {}

    def add_signal(
        self,
        token_id: str,
        market_id: str,
        signal_type: str,
        direction: str,
        score: float,
        wallet: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[SignalCluster]:
        """
        Add a signal and check for confirmation.

        Args:
            token_id: Token identifier
            market_id: Market condition ID
            signal_type: Type of signal (e.g., "whale", "smart_money")
            direction: "BUY" or "SELL"
            score: Signal score
            wallet: Optional wallet address
            metadata: Optional signal metadata

        Returns:
            SignalCluster if confirmed, None otherwise
        """
        now = datetime.now(timezone.utc)
        key = f"{token_id}:{direction}"

        # Clean up expired clusters
        self._cleanup_expired(now)

        # Get or create cluster
        if key not in self._pending:
            self._pending[key] = SignalCluster(
                token_id=token_id,
                market_id=market_id,
                direction=direction,
                timestamp=now,
            )

        cluster = self._pending[key]
        cluster.add_signal(signal_type, score, wallet, metadata)

        # Check if cluster is confirmed
        if cluster.is_strong:
            # Remove from pending and return
            del self._pending[key]
            logger.info(
                f"Signal CONFIRMED: {direction} {token_id[:16]}... "
                f"({cluster.signal_count} types, score={cluster.total_score:.1f})"
            )
            return cluster

        return None

    def _cleanup_expired(self, now: datetime) -> None:
        """Remove expired signal clusters."""
        cutoff = now - timedelta(seconds=self.confirmation_window)
        expired = [
            k for k, v in self._pending.items()
            if v.timestamp < cutoff
        ]
        for key in expired:
            del self._pending[key]

    def get_pending_clusters(self) -> List[SignalCluster]:
        """Get all pending (unconfirmed) clusters."""
        return list(self._pending.values())


# =============================================================================
# Priority 3: Information Timing Filter
# =============================================================================

class InformationTimingFilter:
    """
    Filter to only trade on EARLY signals, not stale information.

    A signal is "early" if:
    - It's among the first large trades in that direction
    - Price hasn't already moved significantly
    - Volume hasn't spiked yet
    """

    def __init__(
        self,
        max_prior_large_trades: int = 2,
        large_trade_threshold: float = 2000.0,
        max_price_change: float = 0.03,  # 3% max prior price move
        max_hour_volume: float = 50000.0,  # Max volume in last hour
        lookback_trades: int = 20,
    ):
        self.max_prior_large_trades = max_prior_large_trades
        self.large_trade_threshold = large_trade_threshold
        self.max_price_change = max_price_change
        self.max_hour_volume = max_hour_volume
        self.lookback_trades = lookback_trades

    def is_early_signal(
        self,
        trade_side: str,
        trade_value: float,
        recent_trades: List[Dict],
        price_change_1h: Optional[float] = None,
        volume_1h: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if this is an early signal worth acting on.

        Args:
            trade_side: "BUY" or "SELL"
            trade_value: Value of the triggering trade
            recent_trades: List of recent trades with 'side' and 'value_usd' keys
            price_change_1h: Price change in last hour (optional)
            volume_1h: Volume in last hour (optional)

        Returns:
            Tuple of (is_early, reason)
        """
        # Check if there are already many large trades in same direction
        same_direction_large = [
            t for t in recent_trades[-self.lookback_trades:]
            if t.get("side") == trade_side and
            t.get("value_usd", 0) >= self.large_trade_threshold
        ]

        if len(same_direction_large) > self.max_prior_large_trades:
            return False, f"Too many prior large {trade_side} trades ({len(same_direction_large)})"

        # Check if price has already moved significantly
        if price_change_1h is not None:
            # For BUY signals, price shouldn't have already gone up much
            # For SELL signals, price shouldn't have already dropped much
            if trade_side == "BUY" and price_change_1h > self.max_price_change:
                return False, f"Price already up {price_change_1h:.1%} - late signal"
            if trade_side == "SELL" and price_change_1h < -self.max_price_change:
                return False, f"Price already down {abs(price_change_1h):.1%} - late signal"

        # Check if volume has already spiked
        if volume_1h is not None and volume_1h > self.max_hour_volume:
            return False, f"High volume already (${volume_1h:,.0f}) - late signal"

        return True, "Early signal - good entry timing"


# =============================================================================
# Priority 4: Contrarian Signal - Fade the Crowd
# =============================================================================

@dataclass
class ContrarianSignal:
    """Signal generated when retail and smart money disagree."""
    direction: str  # "BUY" or "SELL"
    reason: str
    confidence: float  # 0-100
    retail_sentiment: str  # "bullish" or "bearish"
    smart_money_sentiment: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContrarianDetector:
    """
    Detect when retail is wrong and fade them.

    Generates signals when:
    - Retail is FOMO buying while whales sell
    - Retail is panic selling while whales buy
    """

    def __init__(
        self,
        small_trade_threshold: float = 500.0,
        large_trade_threshold: float = 5000.0,
        retail_ratio_threshold: float = 3.0,  # 3x more retail in one direction
        whale_ratio_threshold: float = 2.0,   # 2x more whales in opposite direction
    ):
        self.small_trade_threshold = small_trade_threshold
        self.large_trade_threshold = large_trade_threshold
        self.retail_ratio_threshold = retail_ratio_threshold
        self.whale_ratio_threshold = whale_ratio_threshold

    def detect_contrarian_signal(
        self,
        recent_trades: List[Dict],
    ) -> Optional[ContrarianSignal]:
        """
        Analyze recent trades to detect retail/smart money divergence.

        Args:
            recent_trades: List of recent trades with 'side' and 'value_usd' keys
                          Last 50 trades recommended

        Returns:
            ContrarianSignal if divergence detected, None otherwise
        """
        if len(recent_trades) < 20:
            return None

        # Count trades by size and direction
        small_buys = sum(
            1 for t in recent_trades
            if t.get("side") == "BUY" and t.get("value_usd", 0) < self.small_trade_threshold
        )
        small_sells = sum(
            1 for t in recent_trades
            if t.get("side") == "SELL" and t.get("value_usd", 0) < self.small_trade_threshold
        )
        large_buys = sum(
            1 for t in recent_trades
            if t.get("side") == "BUY" and t.get("value_usd", 0) >= self.large_trade_threshold
        )
        large_sells = sum(
            1 for t in recent_trades
            if t.get("side") == "SELL" and t.get("value_usd", 0) >= self.large_trade_threshold
        )

        # Detect retail FOMO while whales sell
        if (small_buys > small_sells * self.retail_ratio_threshold and
            large_sells > large_buys * self.whale_ratio_threshold and
            small_buys >= 5 and large_sells >= 2):

            confidence = min(100, (small_buys / max(1, small_sells) +
                                   large_sells / max(1, large_buys)) * 20)

            return ContrarianSignal(
                direction="SELL",
                reason="fade_retail_fomo",
                confidence=confidence,
                retail_sentiment="bullish",
                smart_money_sentiment="bearish",
                metadata={
                    "small_buys": small_buys,
                    "small_sells": small_sells,
                    "large_buys": large_buys,
                    "large_sells": large_sells,
                }
            )

        # Detect retail panic while whales buy
        if (small_sells > small_buys * self.retail_ratio_threshold and
            large_buys > large_sells * self.whale_ratio_threshold and
            small_sells >= 5 and large_buys >= 2):

            confidence = min(100, (small_sells / max(1, small_buys) +
                                   large_buys / max(1, large_sells)) * 20)

            return ContrarianSignal(
                direction="BUY",
                reason="fade_retail_panic",
                confidence=confidence,
                retail_sentiment="bearish",
                smart_money_sentiment="bullish",
                metadata={
                    "small_buys": small_buys,
                    "small_sells": small_sells,
                    "large_buys": large_buys,
                    "large_sells": large_sells,
                }
            )

        return None


# =============================================================================
# Priority 5: Market Context Filters
# =============================================================================

@dataclass
class MarketContext:
    """Market characteristics for signal adjustment."""
    hours_to_resolution: Optional[float]
    spread_pct: Optional[float]
    current_price: float
    volume_24h: Optional[float]
    volatility: Optional[float]


class MarketContextFilter:
    """
    Adjust signal weight based on market characteristics.

    Factors considered:
    - Time to resolution (near-term = higher confidence)
    - Liquidity (tight spread = easier execution)
    - Price extremes (avoid very high/low priced outcomes)
    - Volume and volatility
    """

    def __init__(
        self,
        near_term_hours: float = 24.0,
        far_term_hours: float = 720.0,  # 30 days
        high_spread_threshold: float = 0.05,  # 5%
        low_spread_threshold: float = 0.02,   # 2%
        price_extreme_low: float = 0.10,
        price_extreme_high: float = 0.90,
    ):
        self.near_term_hours = near_term_hours
        self.far_term_hours = far_term_hours
        self.high_spread_threshold = high_spread_threshold
        self.low_spread_threshold = low_spread_threshold
        self.price_extreme_low = price_extreme_low
        self.price_extreme_high = price_extreme_high

    def calculate_context_multiplier(
        self,
        context: MarketContext,
    ) -> Tuple[float, List[str]]:
        """
        Calculate signal multiplier based on market context.

        Args:
            context: Market context data

        Returns:
            Tuple of (multiplier, reasons)
        """
        multiplier = 1.0
        reasons = []

        # Time to resolution factor
        if context.hours_to_resolution is not None:
            if context.hours_to_resolution < self.near_term_hours:
                multiplier *= 1.5
                reasons.append(f"Near-term ({context.hours_to_resolution:.1f}h)")
            elif context.hours_to_resolution > self.far_term_hours:
                multiplier *= 0.5
                reasons.append(f"Far-term ({context.hours_to_resolution:.0f}h)")

        # Liquidity factor (spread)
        if context.spread_pct is not None:
            if context.spread_pct > self.high_spread_threshold:
                multiplier *= 0.5
                reasons.append(f"Wide spread ({context.spread_pct:.1%})")
            elif context.spread_pct < self.low_spread_threshold:
                multiplier *= 1.2
                reasons.append(f"Tight spread ({context.spread_pct:.1%})")

        # Price extremes factor
        if context.current_price < self.price_extreme_low:
            multiplier *= 0.7
            reasons.append(f"Low price (${context.current_price:.2f})")
        elif context.current_price > self.price_extreme_high:
            multiplier *= 0.7
            reasons.append(f"High price (${context.current_price:.2f})")

        # Volatility factor
        if context.volatility is not None:
            if context.volatility > 0.10:  # High volatility
                multiplier *= 0.8
                reasons.append("High volatility")
            elif context.volatility < 0.02:  # Low volatility
                multiplier *= 1.1
                reasons.append("Low volatility")

        return multiplier, reasons

    def should_trade(
        self,
        context: MarketContext,
        min_multiplier: float = 0.3,
    ) -> Tuple[bool, str]:
        """
        Determine if market conditions allow trading.

        Args:
            context: Market context
            min_multiplier: Minimum multiplier to allow trade

        Returns:
            Tuple of (should_trade, reason)
        """
        multiplier, reasons = self.calculate_context_multiplier(context)

        if multiplier < min_multiplier:
            return False, f"Poor conditions: {', '.join(reasons)}"

        return True, f"Favorable: {', '.join(reasons) or 'Standard conditions'}"


# =============================================================================
# Priority 6: Dynamic Position Sizing
# =============================================================================

class DynamicPositionSizer:
    """
    Scale position size based on signal confidence and context.

    Higher confidence signals get larger positions, with caps
    to manage risk.

    V4 Optimized: 20% max position with 0.20 confirmation weight
    """

    def __init__(
        self,
        base_position_pct: float = 0.10,  # V4: 10% base (was 5%)
        max_position_pct: float = 0.20,   # V4: 20% max (was 15%)
        min_position_pct: float = 0.05,   # V4: 5% min (was 2%)
    ):
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct

    def calculate_position_size(
        self,
        signal_cluster: Optional[SignalCluster] = None,
        wallet_score: Optional[WalletScore] = None,
        context_multiplier: float = 1.0,
        available_capital: float = 10000.0,
    ) -> Tuple[float, float, str]:
        """
        Calculate position size based on signal quality.

        Args:
            signal_cluster: Confirmed signal cluster (optional)
            wallet_score: Wallet reputation (optional)
            context_multiplier: Market context adjustment
            available_capital: Capital available for trading

        Returns:
            Tuple of (position_usd, position_pct, reasoning)
        """
        size_pct = self.base_position_pct
        reasons = []

        # Adjust for signal cluster strength
        if signal_cluster:
            if signal_cluster.is_very_strong:
                size_pct *= 2.0
                reasons.append("Very strong signal (2x)")
            elif signal_cluster.is_strong:
                size_pct *= 1.5
                reasons.append("Strong signal (1.5x)")

        # Adjust for wallet reputation
        if wallet_score and wallet_score.is_smart_money:
            # Scale by win rate above baseline (55%)
            win_rate_bonus = 1 + (wallet_score.win_rate - 0.55)
            size_pct *= win_rate_bonus
            reasons.append(f"Smart money ({wallet_score.win_rate:.1%} WR)")

        # Apply context multiplier
        size_pct *= context_multiplier
        if context_multiplier != 1.0:
            reasons.append(f"Context ({context_multiplier:.2f}x)")

        # Clamp to min/max
        size_pct = max(self.min_position_pct, min(self.max_position_pct, size_pct))

        position_usd = available_capital * size_pct

        reason_str = "; ".join(reasons) if reasons else "Base position"

        return position_usd, size_pct, reason_str


# =============================================================================
# Priority 7: Adaptive Exits
# =============================================================================

class AdaptiveExitCalculator:
    """
    Calculate exit parameters based on market dynamics.

    Near-resolution markets: Hold for outcome, tight stops
    Medium-term markets: Standard exits
    Long-term markets: Quick scalps
    """

    def __init__(self):
        pass

    def get_exit_params(
        self,
        hours_to_resolution: Optional[float],
        signal_strength: float = 50.0,
        is_contrarian: bool = False,
    ) -> ExitConfig:
        """
        Calculate optimal exit parameters.

        Args:
            hours_to_resolution: Hours until market resolves
            signal_strength: Signal confidence (0-100)
            is_contrarian: Whether this is a contrarian trade

        Returns:
            ExitConfig with adapted parameters
        """
        # Default parameters
        take_profit = 0.06
        stop_loss = 0.08
        trailing_activation = 0.03
        trailing_distance = 0.02
        max_hold_minutes = 120

        if hours_to_resolution is not None:
            if hours_to_resolution < 24:
                # Near resolution: hold for outcome
                take_profit = 0.20  # Higher target - let it ride
                stop_loss = 0.05   # Tighter stop - protect capital
                trailing_activation = 0.05
                trailing_distance = 0.02
                max_hold_minutes = None  # Don't force exit

            elif hours_to_resolution < 168:  # 1 week
                # Medium-term: standard exits
                take_profit = 0.08
                stop_loss = 0.06
                trailing_activation = 0.04
                trailing_distance = 0.02
                max_hold_minutes = 240

            else:
                # Long-term: quick scalps
                take_profit = 0.05
                stop_loss = 0.08
                trailing_activation = 0.03
                trailing_distance = 0.015
                max_hold_minutes = 60

        # Adjust for signal strength
        if signal_strength >= 80:
            # Strong signal: wider stops, let winners run
            stop_loss *= 1.2
            take_profit *= 1.3
        elif signal_strength < 40:
            # Weak signal: tighter management
            stop_loss *= 0.8
            take_profit *= 0.8

        # Adjust for contrarian trades
        if is_contrarian:
            # Contrarian trades need more room to work
            stop_loss *= 1.3
            take_profit *= 1.5
            if max_hold_minutes:
                max_hold_minutes = int(max_hold_minutes * 1.5)

        return ExitConfig(
            take_profit_pct=take_profit,
            stop_loss_pct=stop_loss,
            trailing_stop_enabled=True,
            trailing_stop_activation_pct=trailing_activation,
            trailing_stop_distance_pct=trailing_distance,
            time_exit_enabled=max_hold_minutes is not None,
            max_hold_minutes=max_hold_minutes or 120,
        )


# =============================================================================
# Integrated Enhanced Flow Signal Source
# =============================================================================

class EnhancedFlowSignalSource:
    """
    Enhanced flow signal source integrating all improvements.

    Combines:
    - Wallet reputation tracking
    - Multi-signal confirmation
    - Information timing filters
    - Contrarian detection
    - Market context adjustment
    - Dynamic position sizing
    - Adaptive exits

    V4 Optimized Parameters (Sharpe 4.60, 49.5% return):
    - min_score: 46.0
    - confirmation_weight: 0.20
    - context_weight: 0.10
    """

    def __init__(
        self,
        min_score: float = 46.0,  # V4 Optimized (was 30.0)
        require_confirmation: bool = True,
        enable_contrarian: bool = True,
        enable_timing_filter: bool = True,
    ):
        self.min_score = min_score
        self.require_confirmation = require_confirmation
        self.enable_contrarian = enable_contrarian
        self.enable_timing_filter = enable_timing_filter

        # Initialize components
        self.wallet_tracker = WalletReputationTracker()
        self.signal_confirmer = MultiSignalConfirmation()
        self.timing_filter = InformationTimingFilter()
        self.contrarian_detector = ContrarianDetector()
        self.context_filter = MarketContextFilter()
        self.position_sizer = DynamicPositionSizer()
        self.exit_calculator = AdaptiveExitCalculator()

    def process_alert(
        self,
        alert_type: str,
        token_id: str,
        market_id: str,
        direction: str,
        score: float,
        wallet: Optional[str] = None,
        trade_value: Optional[float] = None,
        recent_trades: Optional[List[Dict]] = None,
        market_context: Optional[MarketContext] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a flow alert through all enhancement filters.

        Args:
            alert_type: Type of alert
            token_id: Token identifier
            market_id: Market condition ID
            direction: "BUY" or "SELL"
            score: Raw alert score
            wallet: Wallet address (optional)
            trade_value: Trade value in USD (optional)
            recent_trades: Recent trades for timing analysis (optional)
            market_context: Market context (optional)
            metadata: Additional metadata (optional)

        Returns:
            Enhanced signal dict if valid, None if filtered out
        """
        # 1. Check wallet reputation
        wallet_score = None
        if wallet:
            wallet_score = self.wallet_tracker.get_wallet(wallet)
            should_copy, copy_reason = self.wallet_tracker.should_copy_trade(
                wallet_score,
                trade_value or 0,
            )
            if not should_copy and wallet_score.trades >= 10:
                logger.debug(f"Filtered: {copy_reason}")
                # Don't completely filter - reduce weight instead
                score *= 0.5

        # 2. Check timing if enabled
        if self.enable_timing_filter and recent_trades:
            is_early, timing_reason = self.timing_filter.is_early_signal(
                direction,
                trade_value or 0,
                recent_trades,
                market_context.hours_to_resolution if market_context else None,
            )
            if not is_early:
                logger.debug(f"Filtered: {timing_reason}")
                score *= 0.6

        # 3. Add to signal confirmation
        cluster = None
        if self.require_confirmation:
            cluster = self.signal_confirmer.add_signal(
                token_id=token_id,
                market_id=market_id,
                signal_type=alert_type,
                direction=direction,
                score=score,
                wallet=wallet,
                metadata=metadata,
            )

            if not cluster:
                # Signal added to pending cluster but not confirmed yet
                return None
        else:
            # Create a single-signal cluster
            cluster = SignalCluster(
                token_id=token_id,
                market_id=market_id,
                direction=direction,
            )
            cluster.add_signal(alert_type, score, wallet, metadata)

        # 4. Check contrarian signals
        contrarian_signal = None
        if self.enable_contrarian and recent_trades:
            contrarian_signal = self.contrarian_detector.detect_contrarian_signal(
                recent_trades
            )
            if contrarian_signal:
                # If contrarian agrees with our direction, boost score
                if contrarian_signal.direction == direction:
                    cluster.total_score *= 1.3
                    cluster.signals.append(f"contrarian_{contrarian_signal.reason}")

        # 5. Apply market context filter
        context_multiplier = 1.0
        context_reasons = []
        if market_context:
            should_trade, context_reason = self.context_filter.should_trade(market_context)
            if not should_trade:
                logger.debug(f"Context filter: {context_reason}")
                return None

            context_multiplier, context_reasons = self.context_filter.calculate_context_multiplier(
                market_context
            )

        # 6. Calculate position size
        position_usd, position_pct, size_reason = self.position_sizer.calculate_position_size(
            signal_cluster=cluster,
            wallet_score=wallet_score,
            context_multiplier=context_multiplier,
        )

        # 7. Calculate exit parameters
        exit_config = self.exit_calculator.get_exit_params(
            hours_to_resolution=market_context.hours_to_resolution if market_context else None,
            signal_strength=cluster.confirmation_score,
            is_contrarian=contrarian_signal is not None,
        )

        # Build enhanced signal
        return {
            "token_id": token_id,
            "market_id": market_id,
            "direction": direction,
            "score": cluster.total_score,
            "confirmation_score": cluster.confirmation_score,
            "signal_types": cluster.signals,
            "wallet_score": wallet_score.reputation_score if wallet_score else None,
            "context_multiplier": context_multiplier,
            "context_reasons": context_reasons,
            "position_pct": position_pct,
            "position_usd": position_usd,
            "size_reason": size_reason,
            "exit_config": exit_config,
            "is_contrarian": contrarian_signal is not None,
            "contrarian_reason": contrarian_signal.reason if contrarian_signal else None,
            "metadata": metadata or {},
        }
