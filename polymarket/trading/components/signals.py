"""
Signal source components.

Signal sources generate trading signals that the bot acts upon.
Different sources can be combined or swapped to create different strategies.
"""

import math
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Callable, TYPE_CHECKING

from ...core.models import Signal, SignalDirection, Market, Token
from ...core.config import RiskConfig

if TYPE_CHECKING:
    from ...flow_detector import FlowAlert, TradeFeedFlowDetector

logger = logging.getLogger(__name__)


# Signal weights for flow alerts
SIGNAL_WEIGHTS = {
    "SMART_MONEY_ACTIVITY": 30,
    "OVERSIZED_BET": 25,
    "COORDINATED_WALLETS": 25,
    "SHORT_DURATION_MOMENTUM": 20,  # Momentum signals for 15-min markets
    "SUDDEN_PRICE_MOVEMENT": 20,  # Rapid price moves - momentum opportunity
    "VOLUME_SPIKE": 10,
    "PRICE_ACCELERATION": 10,
    "FRESH_WALLET_ACTIVITY": 5,
    "COLD_WALLET_ACTIVITY": 5,
    "LOW_ACTIVITY_WALLET": 3,
    "CORRELATED_MOVEMENT": 5,
}

# Severity multipliers
SEVERITY_MULTIPLIERS = {
    "LOW": 0.5,
    "MEDIUM": 1.0,
    "HIGH": 1.5,
    "CRITICAL": 2.0,
}

# Sports market keywords - these markets have efficient pricing, low alpha
SPORTS_KEYWORDS = [
    "nba", "nfl", "nhl", "mlb", "cfb", "cbb", "mls", "ufc", "pga",
    "premier league", "la liga", "bundesliga", "serie a", "ligue 1",
    "champions league", "world cup", "euro 2024", "olympics",
    "wimbledon", "us open", "french open", "australian open",
    "super bowl", "world series", "stanley cup", "nba finals",
    "march madness", "college football", "college basketball",
    # Common sports patterns
    " vs ", " v ", "game ", "match ", "fight ", "bout ",
]

# Category-specific parameters for flow signals
CATEGORY_PARAMS = {
    "crypto": {
        "min_score": 50,
        "position_pct": 0.15,
        "take_profit": 0.10,
        "stop_loss": 0.15,
        "min_lifetime_hours": 1,
    },
    "politics": {
        "min_score": 40,
        "position_pct": 0.20,
        "take_profit": 0.20,
        "stop_loss": 0.25,
        "min_lifetime_hours": 24,
    },
    "finance": {
        "min_score": 45,
        "position_pct": 0.15,
        "take_profit": 0.15,
        "stop_loss": 0.20,
        "min_lifetime_hours": 4,
    },
    "other": {
        "min_score": 45,
        "position_pct": 0.10,
        "take_profit": 0.15,
        "stop_loss": 0.20,
        "min_lifetime_hours": 2,
    },
}


class SignalSource(ABC):
    """
    Abstract base class for signal sources.
    
    Signal sources generate trading signals that indicate when
    and how to trade.
    """
    
    @abstractmethod
    async def get_signals(self) -> List[Signal]:
        """
        Get current trading signals.
        
        Returns a list of signals, each indicating a potential trade.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this signal source"""
        pass


class ExpiringMarketSignals(SignalSource):
    """
    Signal source for expiring market strategy.

    Generates signals for markets that are:
    - Near expiration
    - Priced in a specific range (e.g., 95c-98c)
    - Likely to resolve to $1

    Includes deduplication for recycled markets (same question reused for
    different time periods, e.g., "BTC above $X at 3PM" for different days).
    """

    def __init__(
        self,
        min_price: float = 0.95,
        max_price: float = 0.98,
        min_seconds_left: int = 60,
        max_seconds_left: int = 1800,
        dedup_window_hours: float = 24.0,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_seconds_left = min_seconds_left
        self.max_seconds_left = max_seconds_left
        self.dedup_window_hours = dedup_window_hours
        self._markets: List[Market] = []
        # Track recently signaled markets to avoid recycled duplicates
        # Key: normalized question, Value: (condition_id, timestamp)
        self._recently_signaled: Dict[str, tuple] = {}

    @property
    def name(self) -> str:
        return "expiring_markets"

    def _is_short_term_recurring(self, market: Market) -> bool:
        """
        Check if this is a short-term recurring market (like 15-min BTC).
        These should NOT be deduplicated - each window is a new opportunity.
        """
        q = market.question.lower()
        # 15-minute crypto markets
        if "15m" in q or "15 min" in q or "15-min" in q:
            return True
        # Hourly markets
        if any(pattern in q for pattern in ["-1h", "1 hour", "hourly"]):
            return True
        # Time range patterns like "11:00PM-11:15PM"
        if "-" in q and ("am" in q or "pm" in q or ":" in q):
            import re
            # Pattern like "11:00PM-11:15PM" or "11:00 PM - 11:15 PM"
            if re.search(r'\d{1,2}(:\d{2})?\s*(am|pm)\s*-\s*\d{1,2}(:\d{2})?\s*(am|pm)', q, re.I):
                return True
        return False

    def _normalize_question(self, question: str) -> str:
        """
        Normalize question for deduplication.
        Removes date references to detect recycled markets.
        KEEPS time ranges for short-term markets so different windows aren't merged.
        """
        import re
        q = question.lower().strip()
        # Remove common date patterns (e.g., "jan 7", "1/7", "2025-01-07")
        q = re.sub(r'\b\d{1,2}/\d{1,2}(/\d{2,4})?\b', '', q)  # 1/7 or 1/7/25
        q = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', q)  # 2025-01-07
        q = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(uary|ruary|ch|il|e|y|ust|ember|ober|vember|cember)?\s+\d{1,2}\b', '', q, flags=re.I)
        # Remove standalone day references like "january 7" but keep time ranges
        q = re.sub(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', '', q, flags=re.I)
        # DO NOT remove time patterns - they distinguish different windows!
        # Remove timezone references (but keep the time)
        q = re.sub(r'\b(est|pst|cst|mst|utc|gmt|et|pt)\b', '', q, flags=re.I)
        # Collapse whitespace
        q = re.sub(r'\s+', ' ', q).strip()
        return q

    def _is_duplicate(self, market: Market) -> bool:
        """Check if this market is a recycled duplicate we recently signaled."""
        # Skip deduplication for short-term recurring markets
        # Each 15-minute window is a new opportunity, not a duplicate!
        if self._is_short_term_recurring(market):
            return False

        norm_q = self._normalize_question(market.question)
        now = datetime.now(timezone.utc)

        # Clean up old entries
        cutoff = now - timedelta(hours=self.dedup_window_hours)
        self._recently_signaled = {
            k: v for k, v in self._recently_signaled.items()
            if v[1] > cutoff
        }

        if norm_q in self._recently_signaled:
            old_id, old_time = self._recently_signaled[norm_q]
            # If it's a different condition_id but same normalized question,
            # it's likely a recycled market
            if old_id != market.condition_id:
                logger.debug(
                    f"Skipping recycled market: '{market.question[:50]}...' "
                    f"(similar to {old_id[:8]}... from {old_time})"
                )
                return True
        return False

    def _mark_signaled(self, market: Market):
        """Mark this market as recently signaled."""
        # Don't track short-term recurring markets for deduplication
        if self._is_short_term_recurring(market):
            return
        norm_q = self._normalize_question(market.question)
        self._recently_signaled[norm_q] = (market.condition_id, datetime.now(timezone.utc))

    def update_markets(self, markets: List[Market]):
        """Update the list of markets to scan"""
        self._markets = markets

    async def get_signals(self) -> List[Signal]:
        """Get signals for expiring markets in target price range"""
        signals = []

        for market in self._markets:
            # Check time to expiry
            if market.seconds_left < self.min_seconds_left:
                continue
            if market.seconds_left > self.max_seconds_left:
                continue

            # Skip recycled duplicates
            if self._is_duplicate(market):
                continue

            # Check each token
            for token in market.tokens:
                if self.min_price <= token.price <= self.max_price:
                    # Calculate signal strength based on price proximity to 1.0
                    # Higher price = higher confidence
                    price_factor = (token.price - self.min_price) / (self.max_price - self.min_price)

                    # Time factor: closer to expiry = higher score
                    time_factor = 1.0 - (market.seconds_left / self.max_seconds_left)

                    # Combined score (0-100)
                    score = (price_factor * 0.6 + time_factor * 0.4) * 100

                    signals.append(Signal(
                        market_id=market.condition_id,
                        token_id=token.token_id,
                        direction=SignalDirection.BUY,
                        score=score,
                        source=self.name,
                        metadata={
                            "price": token.price,
                            "seconds_left": market.seconds_left,
                            "outcome": token.outcome,
                            "question": market.question[:100],
                        }
                    ))

                    # Mark as signaled to prevent duplicates
                    self._mark_signaled(market)

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals


class FlowAlertSignals(SignalSource):
    """
    Signal source that converts flow alerts to trading signals.

    Features:
    - Deduplicates correlated alerts from same underlying event
    - Applies weighted scoring with severity multipliers
    - Exponential decay based on alert age
    - Filters out sports markets (efficiently priced, low alpha)
    - Filters out resolved markets (price near 0 or 1)
    - Filters out ultra-short markets (< 2 hours to resolution)
    - Category-specific parameter adjustment
    """

    def __init__(
        self,
        dedup_window_seconds: int = 10,  # V5: 10s (was 30s) - faster cycling
        decay_half_life_seconds: float = 15.0,  # V5: 15s (was 60s) - faster decay
        min_score: float = 55.0,  # V5: 55 (was 30) - higher bar for quality
        filter_sports: bool = True,
        filter_resolved: bool = True,
        min_market_lifetime_hours: float = 2.0,
        min_liquidity_volume: float = 5000.0,
    ):
        self.dedup_window = dedup_window_seconds
        self.decay_half_life = decay_half_life_seconds
        self.min_score = min_score
        self.filter_sports = filter_sports
        self.filter_resolved = filter_resolved
        self.min_market_lifetime_hours = min_market_lifetime_hours
        self.min_liquidity_volume = min_liquidity_volume
        self.recent_alerts: Dict[str, List] = defaultdict(list)
        self._alert_callback: Optional[Callable] = None

        # Stats for filtered signals
        self._filter_stats = {
            "sports_filtered": 0,
            "resolved_filtered": 0,
            "lifetime_filtered": 0,
            "liquidity_filtered": 0,
            "low_score_filtered": 0,
            "passed": 0,
        }

    def _is_sports_market(self, question: str) -> bool:
        """Check if market is a sports betting market."""
        if not question:
            return False
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in SPORTS_KEYWORDS)

    def _is_resolved_market(self, price: Optional[float]) -> bool:
        """Check if market is effectively resolved (price near 0 or 1)."""
        if price is None:
            return False
        return price >= 0.97 or price <= 0.03

    def _has_sufficient_lifetime(self, lifetime_hours: Optional[float]) -> bool:
        """Check if market has enough time for flow signals to play out."""
        if lifetime_hours is None:
            return True  # Unknown, allow it
        return lifetime_hours >= self.min_market_lifetime_hours

    def _get_category(self, question: str) -> str:
        """Determine market category from question text."""
        if not question:
            return "other"
        q_lower = question.lower()

        # Crypto keywords
        crypto_kw = ["btc", "bitcoin", "eth", "ethereum", "crypto", "token",
                     "solana", "sol", "xrp", "doge", "bnb", "cardano", "ada"]
        if any(kw in q_lower for kw in crypto_kw):
            return "crypto"

        # Politics keywords
        politics_kw = ["president", "election", "congress", "senate", "vote",
                       "trump", "biden", "republican", "democrat", "governor"]
        if any(kw in q_lower for kw in politics_kw):
            return "politics"

        # Finance keywords
        finance_kw = ["fed", "interest rate", "inflation", "gdp", "stock",
                      "s&p", "nasdaq", "dow", "economy", "treasury"]
        if any(kw in q_lower for kw in finance_kw):
            return "finance"

        return "other"

    def get_filter_stats(self) -> Dict[str, int]:
        """Get statistics on filtered signals."""
        return self._filter_stats.copy()
    
    @property
    def name(self) -> str:
        return "flow_alerts"
    
    def set_alert_callback(self, detector: "TradeFeedFlowDetector"):
        """Register as callback for flow detector alerts"""
        def callback(alert: "FlowAlert"):
            self._on_alert(alert)
        
        self._alert_callback = callback
        # The detector should call set_alert_callback on us
        if hasattr(detector, 'alert_callback'):
            detector.alert_callback = callback
    
    def _on_alert(self, alert: "FlowAlert"):
        """Handle incoming alert from flow detector"""
        key = f"{alert.market_id}:{alert.token_id}"
        self.recent_alerts[key].append(alert)
        logger.debug(f"Received alert: {alert.alert_type} for {key[:20]}...")
    
    def add_alert(self, alert: "FlowAlert"):
        """Manually add an alert (for testing or external sources)"""
        self._on_alert(alert)
    
    async def get_signals(self) -> List[Signal]:
        """Get deduplicated, filtered, and scored signals from recent alerts"""
        signals = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.dedup_window)

        for key, alerts in list(self.recent_alerts.items()):
            # Filter to recent alerts
            recent = [a for a in alerts if a.timestamp > cutoff]

            if not recent:
                # Remove empty key
                del self.recent_alerts[key]
                continue

            # Update stored alerts
            self.recent_alerts[key] = recent

            # Deduplicate correlated alerts
            deduped = self._deduplicate_alerts(recent)

            if not deduped:
                continue

            # Extract metadata for filtering
            question = ""
            original_price = None
            market_lifetime_hours = None
            market_start_date = None
            market_end_date = None
            volume_24h = None

            for alert in deduped:
                if hasattr(alert, 'question') and not question:
                    question = alert.question
                details = alert.details or {}
                if details.get("price") and original_price is None:
                    original_price = details["price"]
                if details.get("market_lifetime_hours") and market_lifetime_hours is None:
                    market_lifetime_hours = details["market_lifetime_hours"]
                if details.get("market_start_date") and market_start_date is None:
                    market_start_date = details["market_start_date"]
                if details.get("market_end_date") and market_end_date is None:
                    market_end_date = details["market_end_date"]
                if details.get("volume_24h") and volume_24h is None:
                    volume_24h = details["volume_24h"]

            # --- FILTER 1: Skip sports markets (efficiently priced) ---
            if self.filter_sports and self._is_sports_market(question):
                self._filter_stats["sports_filtered"] += 1
                logger.debug(f"Filtered sports market: {question[:50]}...")
                continue

            # --- FILTER 2: Skip resolved markets (price near 0 or 1) ---
            if self.filter_resolved and self._is_resolved_market(original_price):
                self._filter_stats["resolved_filtered"] += 1
                logger.debug(f"Filtered resolved market: price={original_price}")
                continue

            # --- FILTER 3: Skip ultra-short markets ---
            # EXCEPTION: Momentum alerts are specifically for short markets
            MOMENTUM_ALERT_TYPES = {"SHORT_DURATION_MOMENTUM", "SUDDEN_PRICE_MOVEMENT"}
            is_momentum_alert = any(a.alert_type in MOMENTUM_ALERT_TYPES for a in deduped)
            if not is_momentum_alert and not self._has_sufficient_lifetime(market_lifetime_hours):
                self._filter_stats["lifetime_filtered"] += 1
                logger.debug(f"Filtered short-lived market: {market_lifetime_hours}h lifetime")
                continue

            # --- FILTER 4: Skip low liquidity markets ---
            if volume_24h is not None and volume_24h < self.min_liquidity_volume:
                self._filter_stats["liquidity_filtered"] += 1
                logger.debug(f"Filtered low liquidity market: ${volume_24h} < ${self.min_liquidity_volume}")
                continue

            # Calculate composite score with signal staleness decay
            score = self._calculate_composite_score(deduped, now)

            # Get category-specific min score
            category = self._get_category(question)
            category_params = CATEGORY_PARAMS.get(category, CATEGORY_PARAMS["other"])
            effective_min_score = max(self.min_score, category_params["min_score"])

            if score < effective_min_score:
                self._filter_stats["low_score_filtered"] += 1
                continue

            # Determine direction
            direction = self._determine_direction(deduped)

            self._filter_stats["passed"] += 1

            signals.append(Signal(
                market_id=deduped[0].market_id,
                token_id=deduped[0].token_id,
                direction=direction,
                score=score,
                source=self.name,
                metadata={
                    "alert_types": [a.alert_type for a in deduped],
                    "severities": [a.severity for a in deduped],
                    "alert_count": len(deduped),
                    "question": question[:100] if question else "",
                    "price": original_price,
                    "market_lifetime_hours": market_lifetime_hours,
                    "market_start_date": market_start_date,
                    "market_end_date": market_end_date,
                    "category": category,
                    "category_params": category_params,
                    "volume_24h": volume_24h,
                }
            ))

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
    
    def _deduplicate_alerts(self, alerts: List) -> List:
        """
        Remove correlated alerts from same underlying event.
        
        Rule: If two alerts fire within 5 seconds and involve the same
        wallet/trade value, they're likely from the same event.
        Keep only the higher-weighted one.
        """
        if len(alerts) <= 1:
            return alerts
        
        # Group by approximate timestamp (5s buckets)
        buckets: Dict[int, List] = defaultdict(list)
        for alert in alerts:
            bucket_key = int(alert.timestamp.timestamp() // 5)
            buckets[bucket_key].append(alert)
        
        deduped = []
        for bucket_alerts in buckets.values():
            if len(bucket_alerts) == 1:
                deduped.append(bucket_alerts[0])
            else:
                # Check for correlated alerts
                correlated_groups = self._find_correlated_groups(bucket_alerts)
                for group in correlated_groups:
                    # Keep highest-weighted alert from group
                    best = max(
                        group,
                        key=lambda a: SIGNAL_WEIGHTS.get(a.alert_type, 0)
                    )
                    deduped.append(best)
        
        return deduped
    
    def _find_correlated_groups(self, alerts: List) -> List[List]:
        """
        Group correlated alerts together using O(n) hash-based grouping.

        Alerts are correlated if they have the same wallet or similar trade value.
        Optimized from O(n²) to O(n) using hash maps.
        """
        if len(alerts) <= 1:
            return [alerts] if alerts else []

        # Use Union-Find for efficient grouping
        parent = list(range(len(alerts)))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build hash maps for O(1) lookup
        wallet_to_indices: Dict[str, List[int]] = defaultdict(list)
        value_buckets: Dict[int, List[int]] = defaultdict(list)

        for i, alert in enumerate(alerts):
            details = alert.details or {}

            # Index by wallet
            wallet = details.get("wallet", "")
            if wallet:
                wallet_to_indices[wallet.lower()].append(i)

            # Index by value bucket (round to nearest 10%)
            value = details.get("trade_value_usd", 0)
            if value > 0:
                bucket = int(math.log10(max(1, value)) * 10)  # Log-scale bucketing
                value_buckets[bucket].append(i)

        # Union alerts with same wallet
        for indices in wallet_to_indices.values():
            if len(indices) > 1:
                for idx in indices[1:]:
                    union(indices[0], idx)

        # Union alerts with similar trade values (same bucket)
        for indices in value_buckets.values():
            if len(indices) > 1:
                for idx in indices[1:]:
                    union(indices[0], idx)

        # Collect groups
        group_map: Dict[int, List] = defaultdict(list)
        for i, alert in enumerate(alerts):
            group_map[find(i)].append(alert)

        return list(group_map.values())
    
    def _calculate_composite_score(self, alerts: List, now: datetime) -> float:
        """Calculate composite score with severity multipliers and NON-LINEAR decay.

        Information edge in prediction markets decays non-linearly:
        - First 5s: ~90% of edge captured by fast traders
        - 5-30s: ~9% remaining edge
        - 30s+: mostly noise

        We use a two-phase decay to model this:
        1. Steep power-law decay for first 30s
        2. Standard exponential decay thereafter
        """
        total = 0.0

        for alert in alerts:
            base_weight = SIGNAL_WEIGHTS.get(alert.alert_type, 0)
            severity_mult = SEVERITY_MULTIPLIERS.get(alert.severity, 1.0)
            total += base_weight * severity_mult

        # Apply NON-LINEAR decay based on oldest alert age
        oldest = min(a.timestamp for a in alerts)
        age_seconds = (now - oldest).total_seconds()

        # Two-phase decay model
        if age_seconds <= 5:
            # First 5 seconds: retain 90% -> 50% (steep but not total loss)
            decay = 0.9 - (age_seconds / 5) * 0.4  # 0.9 -> 0.5
        elif age_seconds <= 30:
            # 5-30 seconds: retain 50% -> 10% (continued decay)
            decay = 0.5 - ((age_seconds - 5) / 25) * 0.4  # 0.5 -> 0.1
        else:
            # 30+ seconds: standard exponential decay from 10%
            remaining_age = age_seconds - 30
            decay = 0.1 * math.exp(-remaining_age * math.log(2) / self.decay_half_life)

        return total * decay
    
    def _determine_direction(self, alerts: List) -> SignalDirection:
        """Determine signal direction from alerts"""
        buy_weight = 0
        sell_weight = 0

        for alert in alerts:
            weight = SIGNAL_WEIGHTS.get(alert.alert_type, 0)
            details = alert.details or {}

            # Check for momentum_direction (SHORT_DURATION_MOMENTUM alerts)
            # or side field (other alerts)
            side = details.get("momentum_direction", details.get("side", "")).upper()
            if side == "BUY":
                buy_weight += weight
            elif side == "SELL":
                sell_weight += weight
            else:
                # Default to BUY for most alert types (smart money, oversized bets)
                buy_weight += weight * 0.6
        
        if buy_weight > sell_weight * 1.5:
            return SignalDirection.BUY
        elif sell_weight > buy_weight * 1.5:
            return SignalDirection.SELL
        else:
            return SignalDirection.NEUTRAL
    
    def clear_old_alerts(self, max_age_seconds: int = 300):
        """Clear alerts older than max_age"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        
        for key in list(self.recent_alerts.keys()):
            self.recent_alerts[key] = [
                a for a in self.recent_alerts[key]
                if a.timestamp > cutoff
            ]
            if not self.recent_alerts[key]:
                del self.recent_alerts[key]


