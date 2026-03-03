"""Enums for the omnitrade multi-platform trading system."""

from enum import Enum


class ExchangeId(str, Enum):
    """Supported exchanges."""
    POLYMARKET = "polymarket"
    KALSHI = "kalshi"
    HYPERLIQUID = "hyperliquid"


class Side(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class SignalDirection(str, Enum):
    """Directional intent from a signal source."""
    LONG = "LONG"    # Go long / buy
    SHORT = "SHORT"  # Go short / sell
    NEUTRAL = "NEUTRAL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    GTC = "GTC"       # Good til cancelled


class OrderStatus(str, Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class InstrumentType(str, Enum):
    """Type of tradeable instrument."""
    BINARY_OUTCOME = "binary_outcome"     # Polymarket YES/NO tokens
    EVENT_CONTRACT = "event_contract"     # Kalshi contracts
    PERPETUAL = "perpetual"              # Hyperliquid perps


class PositionStatus(str, Enum):
    """Position lifecycle."""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"


class ReservationStatus(str, Enum):
    """Capital reservation lifecycle."""
    PENDING = "pending"
    EXECUTED = "executed"
    RELEASED = "released"
    EXPIRED = "expired"


class AgentStatus(str, Enum):
    """Bot agent lifecycle."""
    ACTIVE = "active"
    STOPPED = "stopped"
    CRASHED = "crashed"


class ExitReason(str, Enum):
    """Why a position was exited."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_LIMIT = "time_limit"
    NEAR_RESOLUTION = "near_resolution"
    MARKET_CLOSE = "market_close"
    LIQUIDATION = "liquidation"
    MANUAL = "manual"
    SIGNAL_REVERSAL = "signal_reversal"


class Environment(str, Enum):
    """Trading environment."""
    PAPER = "paper"      # Paper trading (dry run with simulated fills)
    LIVE = "live"        # Real money
