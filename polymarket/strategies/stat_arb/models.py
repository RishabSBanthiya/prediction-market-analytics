"""
Data models for statistical and cross-market arbitrage.

These models represent:
- Arbitrage opportunity types
- Correlated market pairs
- Market clusters (semantic groupings)
- Multi-leg arbitrage positions
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any
import uuid


class ArbType(Enum):
    """Types of statistical/cross-market arbitrage."""
    PAIR_SPREAD = "pair_spread"           # Correlated pair mean reversion
    MULTI_OUTCOME_SUM = "multi_outcome"   # Sum != 100% arbitrage
    CONDITIONAL_PROB = "conditional"      # P(A&B) vs P(A)*P(B|A) mispricing
    DUPLICATE_MARKET = "duplicate"        # Same question, different prices


class CorrelationType(Enum):
    """How market correlation was determined."""
    PRICE = "price"           # Historical price correlation
    SEMANTIC = "semantic"     # NLP-based question similarity
    MANUAL = "manual"         # Manual configuration


class StatArbPositionStatus(Enum):
    """Status of a multi-leg arbitrage position."""
    PENDING = "pending"       # Orders placed, waiting for fills
    PARTIAL = "partial"       # Some legs filled, waiting for others
    FILLED = "filled"         # All legs filled, position active
    CLOSING = "closing"       # Exit orders placed
    CLOSED = "closed"         # Position fully closed
    FAILED = "failed"         # Execution failed, unwound


@dataclass
class MarketPair:
    """A pair of correlated markets for pair trading."""
    market_a_id: str
    market_b_id: str
    market_a_question: str
    market_b_question: str
    token_a_id: str              # Token to trade in market A
    token_b_id: str              # Token to trade in market B
    correlation: float           # Correlation coefficient (-1 to 1)
    correlation_type: CorrelationType
    lookback_days: int
    last_updated: datetime

    # Spread statistics for mean reversion
    spread_mean: float = 0.0     # Historical mean of price_a - price_b
    spread_std: float = 0.0      # Standard deviation
    half_life_hours: float = 0.0 # Mean reversion half-life

    @property
    def is_valid(self) -> bool:
        """Check if correlation data is recent enough."""
        age_hours = (datetime.now(timezone.utc) - self.last_updated).total_seconds() / 3600
        return age_hours < 24  # Valid for 24 hours


@dataclass
class MarketCluster:
    """A group of semantically related markets."""
    cluster_id: str
    name: str
    market_ids: List[str]
    questions: List[str]
    category: str                # POLITICS, SPORTS, etc.
    similarity_threshold: float  # Min similarity to join cluster
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __len__(self) -> int:
        return len(self.market_ids)

    def add_market(self, market_id: str, question: str) -> None:
        """Add a market to the cluster."""
        if market_id not in self.market_ids:
            self.market_ids.append(market_id)
            self.questions.append(question)


@dataclass
class MultiOutcomeMarket:
    """A market with 3+ mutually exclusive outcomes."""
    condition_id: str
    question: str
    outcomes: List[Dict[str, Any]]  # [{token_id, outcome_name, ask_price, bid_price}, ...]
    total_ask: float               # Sum of all ask prices
    total_bid: float               # Sum of all bid prices

    @property
    def is_underpriced(self) -> bool:
        """Sum of asks < 100%, can buy all outcomes profitably."""
        return self.total_ask < 1.0

    @property
    def is_overpriced(self) -> bool:
        """Sum of bids > 100%, can sell all outcomes profitably."""
        return self.total_bid > 1.0

    @property
    def buy_edge_bps(self) -> int:
        """Edge in basis points if buying all outcomes."""
        if self.total_ask >= 1.0:
            return 0
        return int((1.0 - self.total_ask) * 10000)

    @property
    def sell_edge_bps(self) -> int:
        """Edge in basis points if selling all outcomes."""
        if self.total_bid <= 1.0:
            return 0
        return int((self.total_bid - 1.0) * 10000)


@dataclass
class ArbLeg:
    """A single leg of a multi-leg arbitrage trade."""
    token_id: str
    market_id: str
    outcome: str                 # Outcome name (e.g., "Yes", "Trump", etc.)
    side: str                    # "BUY" or "SELL"
    target_price: float          # Price we want
    target_shares: float         # Shares we want

    # Execution state
    order_id: Optional[str] = None
    filled: bool = False
    fill_price: float = 0.0
    fill_shares: float = 0.0
    fill_time: Optional[datetime] = None

    @property
    def fill_cost(self) -> float:
        """Cost of filled shares."""
        return self.fill_shares * self.fill_price

    @property
    def target_cost(self) -> float:
        """Expected cost of target shares."""
        return self.target_shares * self.target_price


@dataclass
class StatArbOpportunity:
    """A statistical arbitrage opportunity."""
    opportunity_id: str
    arb_type: ArbType
    detected_at: datetime

    # Market references
    market_ids: List[str]
    token_ids: List[str]
    questions: List[str]

    # Opportunity metrics
    edge_bps: int                # Edge in basis points
    z_score: float = 0.0         # For pair trades: deviation in std devs
    confidence: float = 0.0      # 0-1 confidence score

    # Execution details
    legs: List[ArbLeg] = field(default_factory=list)
    total_cost: float = 0.0      # Sum of all leg costs
    expected_profit: float = 0.0 # Expected profit at resolution/exit

    # Risk parameters
    stop_loss_pct: float = 0.0   # Stop loss percentage
    take_profit_pct: float = 0.0 # Take profit percentage

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if opportunity is still actionable."""
        return self.edge_bps > 0 and len(self.legs) >= 2

    @property
    def profit_pct(self) -> float:
        """Expected profit as percentage of cost."""
        if self.total_cost <= 0:
            return 0.0
        return (self.expected_profit / self.total_cost) * 100


@dataclass
class StatArbPosition:
    """A multi-leg stat arb position."""
    position_id: str
    agent_id: str
    arb_type: ArbType
    status: StatArbPositionStatus

    # Markets involved
    market_ids: List[str]

    # Legs
    legs: List[ArbLeg] = field(default_factory=list)

    # Entry metrics
    entry_spread: float = 0.0    # For pair trades: spread at entry
    entry_z_score: float = 0.0   # Z-score at entry
    total_entry_cost: float = 0.0

    # Current state
    current_spread: float = 0.0
    current_z_score: float = 0.0
    unrealized_pnl: float = 0.0

    # Exit metrics
    realized_pnl: float = 0.0

    # Risk parameters
    target_spread: float = 0.0   # Target for mean reversion exit
    stop_spread: float = 0.0     # Stop loss spread level

    # Timing
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_close_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_opportunity(
        cls,
        opportunity: StatArbOpportunity,
        agent_id: str,
    ) -> "StatArbPosition":
        """Create a position from an opportunity."""
        return cls(
            position_id=str(uuid.uuid4()),
            agent_id=agent_id,
            arb_type=opportunity.arb_type,
            status=StatArbPositionStatus.PENDING,
            market_ids=opportunity.market_ids,
            legs=opportunity.legs,
            entry_spread=opportunity.metadata.get("current_spread", 0.0),
            entry_z_score=opportunity.z_score,
            total_entry_cost=opportunity.total_cost,
            target_spread=opportunity.metadata.get("target_spread", 0.0),
            stop_spread=opportunity.metadata.get("stop_spread", 0.0),
            metadata=opportunity.metadata,
        )

    @property
    def is_fully_filled(self) -> bool:
        """Check if all legs are filled."""
        return all(leg.filled for leg in self.legs)

    @property
    def filled_leg_count(self) -> int:
        """Number of filled legs."""
        return sum(1 for leg in self.legs if leg.filled)

    @property
    def total_fill_cost(self) -> float:
        """Total cost of filled legs."""
        return sum(leg.fill_cost for leg in self.legs)

    def update_from_prices(self, prices: Dict[str, float]) -> None:
        """Update unrealized P&L from current prices."""
        # Calculate current value of position
        current_value = 0.0
        for leg in self.legs:
            if leg.filled and leg.token_id in prices:
                current_price = prices[leg.token_id]
                if leg.side == "BUY":
                    # We own shares, value is current price
                    current_value += leg.fill_shares * current_price
                else:
                    # We owe shares, value is negative
                    current_value -= leg.fill_shares * current_price

        # P&L depends on position type
        if self.arb_type == ArbType.PAIR_SPREAD:
            # For pair trades, track spread
            self.unrealized_pnl = current_value - self.total_fill_cost
        else:
            # For other types, simple P&L
            self.unrealized_pnl = current_value - self.total_fill_cost
