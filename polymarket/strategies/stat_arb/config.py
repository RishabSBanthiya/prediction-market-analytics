"""
Configuration for statistical arbitrage strategies.

Provides sensible defaults with full configurability for:
- Correlation detection parameters
- Entry/exit thresholds for each arb type
- Risk management limits
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .models import ArbType


@dataclass
class CorrelationConfig:
    """Configuration for correlation detection."""
    # Price correlation settings
    min_price_correlation: float = 0.7       # Minimum Pearson correlation
    lookback_days: int = 30                  # Days of history for correlation
    min_data_points: int = 50                # Minimum aligned price points

    # Semantic similarity settings
    min_semantic_similarity: float = 0.85    # TF-IDF cosine similarity threshold

    # Update frequency
    update_interval_hours: int = 6           # Re-compute correlations every N hours

    # Spread statistics
    spread_lookback_days: int = 14           # Days for spread mean/std calculation
    min_spread_observations: int = 100       # Minimum observations for stats


@dataclass
class PairTradingConfig:
    """Configuration for pair spread (mean reversion) trading."""
    # Entry thresholds
    entry_z_score: float = 2.0               # Enter when spread is N std devs from mean
    min_correlation: float = 0.7             # Minimum correlation to trade pair

    # Exit thresholds
    exit_z_score: float = 0.5                # Exit when spread returns near mean
    stop_z_score: float = 3.5                # Stop loss at extreme deviation

    # Position limits
    max_holding_hours: int = 48              # Max time to hold position
    max_position_pct: float = 0.10           # Max position as % of capital

    # Minimum edge to trade
    min_edge_bps: int = 30                   # Minimum expected edge in bps


@dataclass
class MultiOutcomeConfig:
    """Configuration for multi-outcome sum arbitrage."""
    # Entry thresholds (optimized via backtest)
    min_edge_bps: int = 30                   # Minimum edge (sum deviation) in bps
    min_outcomes: int = 3                    # Minimum outcomes to consider
    max_outcomes: int = 10                   # Maximum outcomes (complexity limit)

    # Liquidity requirements
    min_liquidity_usd: float = 100.0         # Min orderbook depth per outcome
    max_spread_bps: int = 200                # Max bid-ask spread per outcome

    # Position limits
    max_position_pct: float = 0.10           # Max position as % of capital


@dataclass
class DuplicateConfig:
    """Configuration for duplicate market arbitrage."""
    # Detection thresholds (optimized via backtest: Sharpe=7.11, WinRate=85%)
    min_similarity: float = 0.85             # Question similarity threshold
    min_edge_bps: int = 30                   # Minimum price difference in bps

    # Market matching
    max_end_date_diff_hours: int = 24        # Markets must have similar end times
    require_same_category: bool = True       # Must be same category

    # Position limits
    max_position_pct: float = 0.10           # Max position as % of capital


@dataclass
class ConditionalConfig:
    """Configuration for conditional probability arbitrage."""
    # Detection thresholds
    min_edge_bps: int = 50                   # Minimum mispricing in bps
    min_confidence: float = 0.8              # Confidence in conditional relationship

    # Relationship requirements
    require_same_category: bool = True       # A and B must be same category
    require_same_event: bool = False         # A and B from same event (stricter)

    # Position limits
    max_position_pct: float = 0.10           # Max position as % of capital


@dataclass
class StatArbConfig:
    """Main configuration for statistical arbitrage bot."""
    # Sub-configs for each strategy type
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    pair_trading: PairTradingConfig = field(default_factory=PairTradingConfig)
    multi_outcome: MultiOutcomeConfig = field(default_factory=MultiOutcomeConfig)
    duplicate: DuplicateConfig = field(default_factory=DuplicateConfig)
    conditional: ConditionalConfig = field(default_factory=ConditionalConfig)

    # Enabled arbitrage types (None = all enabled)
    enabled_types: Optional[List[ArbType]] = None

    # Global limits
    max_total_positions: int = 10            # Max concurrent positions across all types
    max_wallet_exposure_pct: float = 0.50    # Max total exposure as % of wallet
    max_per_market_pct: float = 0.15         # Max exposure to single market

    # Scanning settings
    scan_interval_seconds: int = 30          # How often to scan for opportunities
    market_refresh_seconds: int = 60         # How often to refresh market list

    # Execution settings
    order_timeout_seconds: int = 60          # Cancel unfilled orders after
    partial_fill_action: str = "unwind"      # "unwind" or "wait" on partial fills

    # Fee settings (Polymarket defaults)
    taker_fee_bps: int = 10                  # 10 bps taker fee
    maker_fee_bps: int = 0                   # 0 bps maker fee (rebate program)

    def get_enabled_types(self) -> List[ArbType]:
        """Get list of enabled arbitrage types."""
        if self.enabled_types is None:
            return list(ArbType)
        return self.enabled_types

    def is_type_enabled(self, arb_type: ArbType) -> bool:
        """Check if a specific arb type is enabled."""
        return arb_type in self.get_enabled_types()

    @classmethod
    def conservative(cls) -> "StatArbConfig":
        """Create a conservative configuration with higher thresholds."""
        return cls(
            pair_trading=PairTradingConfig(
                entry_z_score=2.5,
                stop_z_score=4.0,
                min_edge_bps=50,
            ),
            multi_outcome=MultiOutcomeConfig(
                min_edge_bps=75,
                min_liquidity_usd=200.0,
            ),
            duplicate=DuplicateConfig(
                min_similarity=0.95,
                min_edge_bps=50,
            ),
            conditional=ConditionalConfig(
                min_edge_bps=75,
                min_confidence=0.9,
            ),
            max_total_positions=5,
            max_wallet_exposure_pct=0.30,
        )

    @classmethod
    def aggressive(cls) -> "StatArbConfig":
        """Create an aggressive configuration with lower thresholds."""
        return cls(
            pair_trading=PairTradingConfig(
                entry_z_score=1.5,
                stop_z_score=3.0,
                min_edge_bps=20,
            ),
            multi_outcome=MultiOutcomeConfig(
                min_edge_bps=30,
                min_liquidity_usd=50.0,
            ),
            duplicate=DuplicateConfig(
                min_similarity=0.85,
                min_edge_bps=20,
            ),
            conditional=ConditionalConfig(
                min_edge_bps=30,
                min_confidence=0.7,
            ),
            max_total_positions=15,
            max_wallet_exposure_pct=0.60,
        )
