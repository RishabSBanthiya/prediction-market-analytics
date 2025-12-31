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
    "VOLUME_SPIKE": 10,
    "PRICE_ACCELERATION": 10,
    "SUDDEN_PRICE_MOVEMENT": 8,
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
    """
    
    def __init__(
        self,
        min_price: float = 0.95,
        max_price: float = 0.98,
        min_seconds_left: int = 60,
        max_seconds_left: int = 1800,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.min_seconds_left = min_seconds_left
        self.max_seconds_left = max_seconds_left
        self._markets: List[Market] = []
    
    @property
    def name(self) -> str:
        return "expiring_markets"
    
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
    """
    
    def __init__(
        self,
        dedup_window_seconds: int = 30,
        decay_half_life_seconds: float = 60.0,
        min_score: float = 30.0,
    ):
        self.dedup_window = dedup_window_seconds
        self.decay_half_life = decay_half_life_seconds
        self.min_score = min_score
        self.recent_alerts: Dict[str, List] = defaultdict(list)
        self._alert_callback: Optional[Callable] = None
    
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
        """Get deduplicated and scored signals from recent alerts"""
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
            
            # Calculate composite score
            score = self._calculate_composite_score(deduped, now)
            
            if score < self.min_score:
                continue
            
            # Determine direction
            direction = self._determine_direction(deduped)
            
            # Extract the original trade price and market timing from alert details
            # Use the most recent alert's data as the reference
            original_price = None
            market_lifetime_hours = None
            market_start_date = None
            market_end_date = None
            
            for alert in deduped:
                details = alert.details or {}
                if details.get("price") and original_price is None:
                    original_price = details["price"]
                if details.get("market_lifetime_hours") and market_lifetime_hours is None:
                    market_lifetime_hours = details["market_lifetime_hours"]
                if details.get("market_start_date") and market_start_date is None:
                    market_start_date = details["market_start_date"]
                if details.get("market_end_date") and market_end_date is None:
                    market_end_date = details["market_end_date"]
            
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
                    "question": deduped[0].question[:100] if hasattr(deduped[0], 'question') else "",
                    "price": original_price,  # Original trade price from flow alert
                    "market_lifetime_hours": market_lifetime_hours,  # For time-based slippage
                    "market_start_date": market_start_date,
                    "market_end_date": market_end_date,
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
        Group correlated alerts together.
        
        Alerts are correlated if they have the same wallet or similar trade value.
        """
        if len(alerts) <= 1:
            return [alerts] if alerts else []
        
        # For simplicity, group by alert type overlap
        # More sophisticated: check wallet addresses, trade values
        groups = []
        used = set()
        
        for i, alert in enumerate(alerts):
            if i in used:
                continue
            
            group = [alert]
            used.add(i)
            
            # Check for correlation with other alerts
            for j, other in enumerate(alerts):
                if j in used:
                    continue
                
                # Check if same wallet in details
                a_wallet = (alert.details or {}).get("wallet", "")
                b_wallet = (other.details or {}).get("wallet", "")
                
                if a_wallet and a_wallet == b_wallet:
                    group.append(other)
                    used.add(j)
                    continue
                
                # Check if similar trade value
                a_value = (alert.details or {}).get("trade_value_usd", 0)
                b_value = (other.details or {}).get("trade_value_usd", 0)
                
                if a_value and b_value and abs(a_value - b_value) / max(a_value, b_value) < 0.1:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _calculate_composite_score(self, alerts: List, now: datetime) -> float:
        """Calculate composite score with severity multipliers and decay"""
        total = 0.0
        
        for alert in alerts:
            base_weight = SIGNAL_WEIGHTS.get(alert.alert_type, 0)
            severity_mult = SEVERITY_MULTIPLIERS.get(alert.severity, 1.0)
            total += base_weight * severity_mult
        
        # Apply exponential decay based on oldest alert age
        oldest = min(a.timestamp for a in alerts)
        age_seconds = (now - oldest).total_seconds()
        
        # Exponential decay with configurable half-life
        decay = math.exp(-age_seconds * math.log(2) / self.decay_half_life)
        
        return total * decay
    
    def _determine_direction(self, alerts: List) -> SignalDirection:
        """Determine signal direction from alerts"""
        buy_weight = 0
        sell_weight = 0
        
        for alert in alerts:
            weight = SIGNAL_WEIGHTS.get(alert.alert_type, 0)
            details = alert.details or {}
            
            # Check side in details
            side = details.get("side", "").upper()
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


