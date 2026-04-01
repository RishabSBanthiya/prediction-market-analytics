"""
Unified data models for omnitrade.

All exchanges map to these models. Bots and components only use these types.
Platform-specific details are handled by exchange adapters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

from .enums import (
    ExchangeId, Side, SignalDirection, OrderType, OrderStatus,
    InstrumentType, PositionStatus, ExitReason,
)


@dataclass
class Instrument:
    """
    A tradeable instrument on any exchange.

    This is the atomic unit - each Polymarket YES/NO token, each Kalshi contract,
    each Hyperliquid perp is one Instrument.

    Prices are normalized to 0-1 for binary/event types, USD for perpetuals.
    """
    instrument_id: str           # Platform-specific ID (token_id, ticker, symbol)
    exchange: ExchangeId
    instrument_type: InstrumentType
    name: str                    # Human-readable name

    # Pricing
    price: float = 0.0           # Last price (0-1 for binary, USD for perps)
    bid: float = 0.0
    ask: float = 0.0

    # Grouping
    market_id: str = ""          # Links related instruments (e.g., YES/NO pair)
    outcome: str = ""            # "YES"/"NO" for binary, "" for perps

    # Status
    active: bool = True
    closed: bool = False

    # Metadata varies by platform
    min_order_size: float = 0.0
    tick_size: float = 0.01

    # Perps-specific (Hyperliquid)
    max_leverage: float = 1.0
    funding_rate: float = 0.0

    # Event-specific
    expiry: Optional[datetime] = None

    # Raw platform data for adapter use
    raw: dict = field(default_factory=dict)


@dataclass
class OrderbookLevel:
    """Single price level in an orderbook."""
    price: float
    size: float


@dataclass
class OrderbookSnapshot:
    """Point-in-time orderbook state."""
    instrument_id: str
    bids: list[OrderbookLevel] = field(default_factory=list)  # Sorted highest first
    asks: list[OrderbookLevel] = field(default_factory=list)  # Sorted lowest first
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            mid = self.midpoint
            if mid and mid > 0:
                return (self.best_ask - self.best_bid) / mid
        return None


@dataclass
class Signal:
    """
    Trading signal from any signal source.

    Uses LONG/SHORT for directional intent. The executor maps
    to BUY/SELL per platform.
    """
    instrument_id: str
    direction: SignalDirection
    score: float              # Signal strength (arbitrary scale, higher = stronger)
    source: str               # Which signal source generated this
    price: float = 0.0        # Price at signal generation time

    # Optional context
    market_id: str = ""
    exchange: Optional[ExchangeId] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.NEUTRAL and self.score > 0


@dataclass
class OrderRequest:
    """Request to place an order. Platform-agnostic."""
    instrument_id: str
    side: Side
    size: float              # In base units (shares for binary, contracts for perps)
    price: float             # Limit price (ignored for market orders)
    order_type: OrderType = OrderType.LIMIT

    # Optional
    time_in_force: str = "GTC"
    reduce_only: bool = False
    leverage: Optional[float] = None  # Perps only

    metadata: dict = field(default_factory=dict)


@dataclass
class OrderResult:
    """Result of an order placement attempt."""
    success: bool
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING

    filled_size: float = 0.0
    filled_price: float = 0.0

    requested_size: float = 0.0
    requested_price: float = 0.0

    error_message: str = ""
    is_rejection: bool = False   # True if rejected by safety checks (not a system error)

    fees: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CancelDetail:
    """Per-order result from a batch cancel operation."""
    order_id: str
    success: bool
    error_code: str = ""
    error_message: str = ""


@dataclass
class CancelResult:
    """Structured result from cancel_orders() with per-order details.

    Attributes:
        cancelled: Number of orders successfully cancelled.
        failed: Number of orders that failed to cancel.
        already_filled: Number of orders that were already filled/expired.
        details: Per-order success/failure information.
    """
    cancelled: int = 0
    failed: int = 0
    already_filled: int = 0
    details: list[CancelDetail] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of orders processed."""
        return self.cancelled + self.failed + self.already_filled

    @property
    def failed_order_ids(self) -> list[str]:
        """Order IDs that failed to cancel (excludes already-filled)."""
        return [d.order_id for d in self.details if not d.success and d.error_code != "not_found"]


@dataclass
class OpenOrder:
    """An open/pending order on the exchange."""
    order_id: str
    instrument_id: str
    side: Side
    size: float
    filled_size: float
    price: float
    order_type: OrderType
    status: OrderStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccountBalance:
    """Account balance on an exchange."""
    exchange: ExchangeId
    total_equity: float = 0.0        # Total account value
    available_balance: float = 0.0   # Free to trade
    reserved: float = 0.0           # In open orders/positions margin
    unrealized_pnl: float = 0.0
    currency: str = "USD"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExchangePosition:
    """A position as reported by the exchange."""
    instrument_id: str
    exchange: ExchangeId
    side: Side
    size: float                    # Absolute size
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    liquidation_price: Optional[float] = None  # Perps only
    leverage: float = 1.0

    @property
    def cost_basis(self) -> float:
        return self.size * self.entry_price

    @property
    def market_value(self) -> float:
        return self.size * self.current_price if self.current_price else self.cost_basis


@dataclass
class Quote:
    """A two-sided quote for market making."""
    instrument_id: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float

    @property
    def spread(self) -> float:
        mid = (self.bid_price + self.ask_price) / 2
        return (self.ask_price - self.bid_price) / mid if mid > 0 else 0.0


@dataclass
class PositionState:
    """
    Tracked state for exit monitoring.
    Ported from polymarket-analytics with multi-exchange support.
    """
    instrument_id: str
    entry_price: float
    entry_time: datetime
    size: float
    side: Side = Side.BUY

    # Tracking
    peak_price: float = 0.0
    trough_price: float = float('inf')
    trailing_stop_activated: bool = False
    trailing_stop_level: float = 0.0

    # Safety order
    safety_order_id: Optional[str] = None

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.trough_price == float('inf'):
            self.trough_price = self.entry_price

    @property
    def unrealized_return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.peak_price - self.entry_price) / self.entry_price


# ==================== CROSS-EXCHANGE MODELS ====================


@dataclass
class SignalLeg:
    """One leg of a multi-exchange signal."""
    exchange: ExchangeId
    instrument_id: str
    direction: SignalDirection
    weight: float = 1.0          # Relative sizing (1.0 = full size)
    price: float = 0.0
    leverage: Optional[float] = None  # For perps hedges
    metadata: dict = field(default_factory=dict)


@dataclass
class MultiLegSignal:
    """
    Cross-exchange trading signal.

    Example: Long "BTC Up" on Polymarket + Short BTC-PERP on Hyperliquid.
    Each leg specifies exchange, instrument, direction, and relative weight.
    """
    legs: list[SignalLeg]
    strategy_type: str           # e.g. "hedge", "arb", "basis"
    score: float = 0.0
    source: str = ""
    edge_bps: float = 0.0       # Estimated edge in basis points
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def exchanges_involved(self) -> set[ExchangeId]:
        return {leg.exchange for leg in self.legs}

    @property
    def is_actionable(self) -> bool:
        return self.score > 0 and len(self.legs) >= 2


@dataclass
class LegResult:
    """Result of executing one leg of a multi-leg strategy."""
    leg: SignalLeg
    order_result: OrderResult
    reservation_id: str = ""

    @property
    def success(self) -> bool:
        return self.order_result.success


@dataclass
class MultiLegResult:
    """Result of a full multi-leg execution."""
    leg_results: list[LegResult]
    strategy_type: str
    needs_rollback: bool = False

    @property
    def success(self) -> bool:
        return all(lr.success for lr in self.leg_results)

    @property
    def total_cost(self) -> float:
        return sum(
            lr.order_result.filled_size * lr.order_result.filled_price
            for lr in self.leg_results if lr.success
        )
