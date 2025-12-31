"""
Core data models for Polymarket trading system.

All dataclasses used across trading and backtesting are defined here
to ensure consistency and avoid circular imports.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum


class Side(Enum):
    """Trade side"""
    BUY = "BUY"
    SELL = "SELL"


class SignalDirection(Enum):
    """Signal direction for trading decisions"""
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class PositionStatus(Enum):
    """Position lifecycle status"""
    OPEN = "open"
    CLOSED = "closed"
    ORPHAN = "orphan"  # Found on-chain but not in DB


class ReservationStatus(Enum):
    """Reservation lifecycle status"""
    PENDING = "pending"
    EXECUTED = "executed"
    RELEASED = "released"
    EXPIRED = "expired"


class AgentStatus(Enum):
    """Agent lifecycle status"""
    ACTIVE = "active"
    STOPPED = "stopped"
    CRASHED = "crashed"


@dataclass
class Token:
    """Token data for a market outcome"""
    token_id: str
    outcome: str
    price: float = 0.0
    
    def __hash__(self):
        return hash(self.token_id)


@dataclass
class Market:
    """Market data"""
    condition_id: str
    question: str
    slug: str
    end_date: datetime
    tokens: List[Token]
    start_date: Optional[datetime] = None
    category: Optional[str] = None
    closed: bool = False
    resolved: bool = False
    winning_outcome: Optional[str] = None
    
    @property
    def seconds_left(self) -> float:
        """Seconds until market expires"""
        now = datetime.now(timezone.utc)
        if self.end_date.tzinfo is None:
            end = self.end_date.replace(tzinfo=timezone.utc)
        else:
            end = self.end_date
        return max(0, (end - now).total_seconds())
    
    @property
    def is_expired(self) -> bool:
        return self.seconds_left <= 0
    
    @property
    def lifetime_hours(self) -> Optional[float]:
        """Total market lifetime in hours (from start_date to end_date)."""
        if self.start_date is None:
            return None
        
        start = self.start_date
        end = self.end_date
        
        # Ensure both have timezone info
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        
        return (end - start).total_seconds() / 3600


def calculate_time_based_slippage_threshold(lifetime_hours: Optional[float]) -> float:
    """Calculate max slippage threshold based on market lifetime.
    
    Short markets (< 30 min) need tight slippage control because
    small price movements are significant relative to potential profit.
    Longer markets can tolerate much more price drift.
    
    Args:
        lifetime_hours: Total market lifetime in hours (start to end).
                       If None, defaults to loose threshold (10%).
    
    Returns:
        Maximum allowed slippage as a decimal (e.g., 0.03 = 3%)
    
    Thresholds:
        - <= 5 minutes (0.0833 hours): 1%
        - <= 30 minutes (0.5 hours): 3%
        - > 30 minutes: 10%
    """
    if lifetime_hours is None:
        return 0.10  # Default to loose threshold if unknown
    
    if lifetime_hours <= 0.0833:  # 5 minutes
        return 0.01  # 1%
    elif lifetime_hours <= 0.5:  # 30 minutes
        return 0.03  # 3%
    else:  # > 30 minutes
        return 0.10  # 10%


@dataclass
class Position:
    """Position in a market"""
    id: Optional[int] = None  # DB ID
    agent_id: str = ""
    market_id: str = ""  # condition_id
    token_id: str = ""
    outcome: str = ""
    shares: float = 0.0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    current_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    
    @property
    def cost_basis(self) -> float:
        """Total cost of position"""
        return self.shares * self.entry_price
    
    @property
    def current_value(self) -> float:
        """Current market value"""
        if self.current_price is not None:
            return self.shares * self.current_price
        return self.cost_basis
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L"""
        return self.current_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage"""
        if self.cost_basis > 0:
            return self.unrealized_pnl / self.cost_basis
        return 0.0


@dataclass
class Trade:
    """Executed trade"""
    trade_id: str
    market_id: str
    token_id: str
    side: Side
    shares: float
    price: float
    timestamp: datetime
    maker_address: Optional[str] = None
    taker_address: Optional[str] = None
    fee: float = 0.0
    
    @property
    def value_usd(self) -> float:
        return self.shares * self.price


@dataclass
class HistoricalPrice:
    """A single price point in history"""
    timestamp: int  # Unix timestamp
    price: float
    
    @property
    def datetime(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


@dataclass
class OrderbookSnapshot:
    """Orderbook state at a point in time"""
    token_id: str
    timestamp: datetime
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_size: float = 0.0
    ask_size: float = 0.0
    bid_depth: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    ask_depth: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Spread as percentage of midpoint (standard calculation)"""
        if self.best_bid and self.best_ask:
            midpoint = (self.best_bid + self.best_ask) / 2
            if midpoint > 0:
                return (self.best_ask - self.best_bid) / midpoint
        return None
    
    @property
    def midpoint(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None


@dataclass
class Signal:
    """Trading signal from a signal source"""
    market_id: str
    token_id: str
    direction: SignalDirection
    score: float  # 0-100 composite score
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = ""  # e.g., "flow_alerts", "expiring_markets"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_buy(self) -> bool:
        return self.direction == SignalDirection.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.direction == SignalDirection.SELL


@dataclass
class Reservation:
    """Capital reservation for pending trade"""
    id: str
    agent_id: str
    market_id: str
    token_id: str
    amount_usd: float
    reserved_at: datetime
    expires_at: datetime
    status: ReservationStatus = ReservationStatus.PENDING
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def is_active(self) -> bool:
        return self.status == ReservationStatus.PENDING and not self.is_expired


@dataclass
class AgentInfo:
    """Information about a registered agent"""
    agent_id: str
    agent_type: str  # e.g., "bond", "flow"
    started_at: datetime
    last_heartbeat: datetime
    status: AgentStatus = AgentStatus.ACTIVE
    
    @property
    def seconds_since_heartbeat(self) -> float:
        return (datetime.now(timezone.utc) - self.last_heartbeat).total_seconds()


@dataclass
class WalletState:
    """Current state of the trading wallet"""
    wallet_address: str
    usdc_balance: float
    total_positions_value: float
    total_reserved: float
    positions: List[Position] = field(default_factory=list)
    reservations: List[Reservation] = field(default_factory=list)
    agents: List[AgentInfo] = field(default_factory=list)
    
    @property
    def available_capital(self) -> float:
        """Capital available for new trades"""
        return self.usdc_balance - self.total_reserved
    
    @property
    def total_exposure(self) -> float:
        """Total capital at risk (positions + reservations)"""
        return self.total_positions_value + self.total_reserved
    
    @property
    def exposure_pct(self) -> float:
        """Exposure as percentage of total equity"""
        total_equity = self.usdc_balance + self.total_positions_value
        if total_equity > 0:
            return self.total_exposure / total_equity
        return 0.0


@dataclass
class ExecutionResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str] = None
    filled_shares: float = 0.0
    filled_price: float = 0.0
    requested_shares: float = 0.0
    requested_price: float = 0.0
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def fill_ratio(self) -> float:
        """Ratio of filled to requested shares"""
        if self.requested_shares > 0:
            return self.filled_shares / self.requested_shares
        return 0.0
    
    @property
    def slippage(self) -> float:
        """Price slippage from requested to filled"""
        if self.requested_price > 0:
            return abs(self.filled_price - self.requested_price) / self.requested_price
        return 0.0


