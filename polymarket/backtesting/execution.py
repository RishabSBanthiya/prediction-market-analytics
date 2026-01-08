"""
Simulated execution with realistic assumptions.

Provides execution simulation with:
- Slippage modeling (no fees on Polymarket)
- Spread checks
- Liquidity constraints
- Price-history based spread estimation
- Historical orderbook-based execution (when data available)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, TYPE_CHECKING

from ..core.models import OrderbookSnapshot, Side

if TYPE_CHECKING:
    from .liquidity_provider import HistoricalLiquidityProvider

logger = logging.getLogger(__name__)


# Constants for realistic execution
# NOTE: These defaults are calibrated conservatively based on Polymarket observations
# Real slippage varies 0.5-5% depending on market liquidity and timing
DEFAULT_SLIPPAGE_PCT = 0.015  # 1.5% base slippage (conservative)
MAX_SPREAD_PCT = 0.03         # 3% max acceptable spread
MIN_LIQUIDITY_USD = 100       # Minimum $100 liquidity to trade
PRICE_IMPACT_FACTOR = 0.002   # 0.2% impact per $1000 traded (doubled for realism)

# Survivorship bias penalty - applied to final returns
# Accounts for cancelled/disputed markets not in backtest data
SURVIVORSHIP_BIAS_PENALTY = 0.15  # 15% reduction in expected returns


@dataclass
class SimulatedExecution:
    """
    Simulates order execution with realistic assumptions.

    Features:
    - NO transaction fees (Polymarket has no fees)
    - Buy/sell slippage based on liquidity
    - Spread checks
    - Price impact modeling

    IMPORTANT: These estimates are inherently optimistic because:
    1. No historical orderbook depth available
    2. Spread estimation from price patterns is approximate
    3. Real fills compete with other market participants
    """

    transaction_fee_pct: float = 0.0      # No fees on Polymarket
    buy_slippage_pct: float = 0.015       # 1.5% base slippage (conservative)
    sell_slippage_pct: float = 0.015      # 1.5% base slippage (conservative)
    max_spread_pct: float = 0.03          # 3% max spread to trade
    min_liquidity_usd: float = 100.0      # Minimum $100 liquidity
    
    def check_spread(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float]
    ) -> Tuple[bool, float]:
        """
        Check if spread is acceptable for trading.
        
        Returns:
            (is_acceptable, spread_pct)
        """
        if best_bid is None or best_ask is None or best_bid <= 0:
            return False, 1.0  # Unknown spread, assume too wide
        
        spread_pct = (best_ask - best_bid) / best_bid
        return spread_pct <= self.max_spread_pct, spread_pct
    
    def estimate_slippage(
        self,
        trade_value_usd: float,
        liquidity_usd: float = 1000.0
    ) -> float:
        """
        Estimate slippage based on trade size relative to liquidity.
        
        Larger trades in less liquid markets have more slippage.
        """
        if liquidity_usd <= 0:
            return 0.10  # 10% slippage for no liquidity
        
        # Base slippage + impact based on trade size
        size_ratio = trade_value_usd / liquidity_usd
        impact = size_ratio * PRICE_IMPACT_FACTOR * 10  # Scale factor
        
        return min(0.10, self.buy_slippage_pct + impact)
    
    def estimate_spread_from_history(
        self,
        prices: List[float],
        min_spread: float = 0.01,
        max_spread: float = 0.15,
        liquidity_hint: Optional[float] = None
    ) -> float:
        """
        Estimate bid-ask spread from price history using price bounce patterns.

        High-frequency price reversals suggest wider spreads as trades
        bounce between bid and ask prices.

        LIMITATIONS (be aware):
        - Roll's estimator assumes efficient markets (Polymarket is NOT)
        - Information asymmetry causes price jumps, not spread bounces
        - This is a rough approximation; actual spreads vary 0.5-15%

        Args:
            prices: List of historical prices
            min_spread: Minimum spread to return (1% - more realistic)
            max_spread: Maximum spread to return (15% for illiquid)
            liquidity_hint: Optional estimated liquidity in USD

        Returns:
            Estimated spread as a decimal (e.g., 0.02 for 2%)
        """
        if len(prices) < 10:
            return 0.03  # Default 3% spread for insufficient data (more conservative)

        # Method 1: Average absolute tick-to-tick change
        # This approximates the bid-ask bounce
        tick_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_tick_change = sum(tick_changes) / len(tick_changes) if tick_changes else 0

        # Method 2: Count direction reversals
        # More reversals = trades bouncing between bid and ask = wider spread
        reversals = 0
        for i in range(2, len(prices)):
            prev_dir = prices[i-1] - prices[i-2]
            curr_dir = prices[i] - prices[i-1]
            if prev_dir * curr_dir < 0:  # Direction changed
                reversals += 1

        reversal_rate = reversals / (len(prices) - 2) if len(prices) > 2 else 0

        # Method 3: Roll's spread estimator (simplified)
        # Based on negative autocovariance of price changes
        signed_changes = [prices[j] - prices[j-1] for j in range(1, len(prices))]
        if len(signed_changes) >= 2:
            autocovar = 0
            for i in range(1, len(signed_changes)):
                autocovar += signed_changes[i] * signed_changes[i-1]
            autocovar /= (len(signed_changes) - 1)

            # Roll's estimator: spread = 2 * sqrt(-autocovariance)
            # Only valid if autocovariance is negative
            if autocovar < 0:
                roll_spread = 2 * (abs(autocovar) ** 0.5)
            else:
                # Positive autocovariance suggests trending, use higher spread
                roll_spread = avg_tick_change * 3
        else:
            roll_spread = avg_tick_change * 2

        # Method 4: Liquidity-based adjustment
        liquidity_factor = 1.0
        if liquidity_hint is not None:
            if liquidity_hint < 500:
                liquidity_factor = 1.5  # Low liquidity = wider spread
            elif liquidity_hint < 2000:
                liquidity_factor = 1.2
            elif liquidity_hint > 10000:
                liquidity_factor = 0.8  # High liquidity = tighter spread

        # Combine estimates with weights
        # - Tick change: 35% (direct observation)
        # - Roll estimator: 35% (statistical)
        # - Reversal rate: 15% (behavioral)
        # - Base uncertainty: 15% (acknowledge we're guessing)
        base_estimate = (
            avg_tick_change * 2.5 * 0.35 +  # Multiplier increased
            roll_spread * 0.35 +
            reversal_rate * 0.08 * 0.15 +  # Scale reversal rate to spread
            0.015 * 0.15  # Base uncertainty premium
        )

        # Apply liquidity adjustment
        spread_estimate = base_estimate * liquidity_factor

        # Ensure spread is within bounds
        return max(min_spread, min(max_spread, spread_estimate))
    
    def execute_buy(
        self,
        price: float,
        size_shares: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """
        Simulate buy execution.
        
        Args:
            price: Target/market price
            size_shares: Number of shares to buy
            orderbook: Current orderbook state (optional)
            liquidity_usd: Estimated market liquidity
            price_history: Historical prices for spread estimation (optional)
        
        Returns:
            Tuple of (execution_price, filled_shares, fee=0)
        """
        trade_value = size_shares * price
        
        # Calculate dynamic slippage
        base_slippage = self.estimate_slippage(trade_value, liquidity_usd)
        
        # If price history provided, estimate spread and add half-spread to slippage
        # (buyer pays the ask, which is mid + half spread)
        if price_history and len(price_history) >= 10:
            estimated_spread = self.estimate_spread_from_history(price_history)
            # Add half spread to slippage (crossing the spread)
            total_slippage = base_slippage + (estimated_spread / 2)
        else:
            total_slippage = base_slippage
        
        exec_price = price * (1 + total_slippage)
        
        # Check liquidity if orderbook provided
        filled_shares = size_shares
        if orderbook and orderbook.ask_depth:
            available = self._get_available_liquidity(orderbook.ask_depth, exec_price)
            filled_shares = min(size_shares, available)
        
        # Cap based on estimated liquidity
        max_shares_from_liquidity = (liquidity_usd * 0.1) / price if price > 0 else 0
        filled_shares = min(filled_shares, max_shares_from_liquidity) if max_shares_from_liquidity > 0 else filled_shares
        
        # No fees on Polymarket
        fee = 0.0
        
        return exec_price, filled_shares, fee
    
    def execute_sell(
        self,
        price: float,
        size_shares: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """
        Simulate sell execution.
        
        Args:
            price: Target/market price
            size_shares: Number of shares to sell
            orderbook: Current orderbook state (optional)
            liquidity_usd: Estimated market liquidity
            price_history: Historical prices for spread estimation (optional)
        
        Returns:
            Tuple of (execution_price, filled_shares, fee=0)
        """
        trade_value = size_shares * price
        
        # Calculate dynamic slippage
        base_slippage = self.estimate_slippage(trade_value, liquidity_usd)
        
        # If price history provided, estimate spread and add half-spread to slippage
        # (seller receives the bid, which is mid - half spread)
        if price_history and len(price_history) >= 10:
            estimated_spread = self.estimate_spread_from_history(price_history)
            # Add half spread to slippage (crossing the spread)
            total_slippage = base_slippage + (estimated_spread / 2)
        else:
            total_slippage = base_slippage
        
        exec_price = price * (1 - total_slippage)
        
        # Check liquidity if orderbook provided
        filled_shares = size_shares
        if orderbook and orderbook.bid_depth:
            available = self._get_available_liquidity(orderbook.bid_depth, exec_price)
            filled_shares = min(size_shares, available)
        
        # Cap based on estimated liquidity
        max_shares_from_liquidity = (liquidity_usd * 0.1) / price if price > 0 else 0
        filled_shares = min(filled_shares, max_shares_from_liquidity) if max_shares_from_liquidity > 0 else filled_shares
        
        # No fees on Polymarket
        fee = 0.0
        
        return exec_price, filled_shares, fee
    
    def _get_available_liquidity(
        self,
        depth: list,
        max_price: float
    ) -> float:
        """
        Calculate available liquidity up to max price.
        
        Args:
            depth: List of (price, size) tuples
            max_price: Maximum price willing to pay/accept
        
        Returns:
            Total available shares
        """
        total = 0.0
        for price, size in depth:
            if price <= max_price:
                total += size
            else:
                break
        return total
    
    def estimate_impact(
        self,
        side: Side,
        size_shares: float,
        orderbook: OrderbookSnapshot
    ) -> Optional[float]:
        """
        Estimate price impact of an order.
        
        Args:
            side: BUY or SELL
            size_shares: Order size
            orderbook: Current orderbook
        
        Returns:
            Estimated execution price, or None if insufficient liquidity
        """
        if side == Side.BUY:
            depth = orderbook.ask_depth
        else:
            depth = orderbook.bid_depth
        
        if not depth:
            return None
        
        remaining = size_shares
        total_cost = 0.0
        
        for price, size in depth:
            if remaining <= 0:
                break
            
            fill = min(remaining, size)
            total_cost += fill * price
            remaining -= fill
        
        if remaining > 0:
            # Insufficient liquidity
            return None
        
        return total_cost / size_shares


class RealisticExecution(SimulatedExecution):
    """
    More realistic execution simulation.
    
    Adds:
    - Execution delay modeling
    - Market impact estimation
    - Partial fill simulation
    - Spread checking
    """
    
    def __init__(
        self,
        buy_slippage_pct: float = 0.005,
        sell_slippage_pct: float = 0.005,
        max_spread_pct: float = 0.03,
        fill_probability: float = 0.95,
        partial_fill_probability: float = 0.10,
    ):
        super().__init__()
        self.buy_slippage_pct = buy_slippage_pct
        self.sell_slippage_pct = sell_slippage_pct
        self.max_spread_pct = max_spread_pct
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
    
    def can_trade(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float],
        liquidity_usd: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check if trading conditions are acceptable.
        
        Returns:
            (can_trade, reason)
        """
        # Check spread
        acceptable_spread, spread = self.check_spread(best_bid, best_ask)
        if not acceptable_spread:
            return False, f"Spread too wide: {spread:.1%} > {self.max_spread_pct:.1%}"
        
        # Check liquidity
        if liquidity_usd < self.min_liquidity_usd:
            return False, f"Insufficient liquidity: ${liquidity_usd:.0f} < ${self.min_liquidity_usd:.0f}"
        
        return True, "OK"
    
    def execute_buy(
        self,
        price: float,
        size_shares: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """Execute buy with realistic fill simulation"""
        import random
        
        # Check if order fills at all
        if random.random() > self.fill_probability:
            return price, 0.0, 0.0
        
        exec_price, filled_shares, fee = super().execute_buy(
            price, size_shares, orderbook, liquidity_usd, price_history
        )
        
        # Check for partial fill
        if random.random() < self.partial_fill_probability:
            fill_ratio = random.uniform(0.3, 0.9)
            filled_shares *= fill_ratio
        
        return exec_price, filled_shares, fee
    
    def execute_sell(
        self,
        price: float,
        size_shares: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """Execute sell with realistic fill simulation"""
        import random
        
        # Check if order fills at all
        if random.random() > self.fill_probability:
            return price, 0.0, 0.0
        
        exec_price, filled_shares, fee = super().execute_sell(
            price, size_shares, orderbook, liquidity_usd, price_history
        )

        # Check for partial fill
        if random.random() < self.partial_fill_probability:
            fill_ratio = random.uniform(0.3, 0.9)
            filled_shares *= fill_ratio

        return exec_price, filled_shares, fee


class HistoricalLiquidityExecution(SimulatedExecution):
    """
    Execution simulation using recorded historical orderbook data.

    Uses actual recorded orderbook snapshots for realistic slippage
    and fill simulation. Falls back to estimation when data unavailable.

    Usage:
        from polymarket.backtesting.liquidity_provider import HistoricalLiquidityProvider

        provider = HistoricalLiquidityProvider()
        execution = HistoricalLiquidityExecution(provider)

        # Execute with historical data
        price, shares, fee = execution.execute_buy_at(
            token_id="0x123...",
            timestamp=datetime(2024, 1, 15, 12, 0),
            price=0.65,
            size_shares=100
        )
    """

    def __init__(
        self,
        liquidity_provider: "HistoricalLiquidityProvider",
        fallback_slippage_pct: float = 0.015,
        max_spread_pct: float = 0.03,
        tolerance_seconds: int = 60,
    ):
        """
        Initialize with historical liquidity provider.

        Args:
            liquidity_provider: Provider for historical orderbook data
            fallback_slippage_pct: Slippage to use when no historical data
            max_spread_pct: Maximum acceptable spread
            tolerance_seconds: Max time difference for snapshot lookup
        """
        super().__init__()
        self.provider = liquidity_provider
        self.fallback_slippage_pct = fallback_slippage_pct
        self.max_spread_pct = max_spread_pct
        self.tolerance_seconds = tolerance_seconds

        # Track usage stats
        self.historical_fills = 0
        self.fallback_fills = 0

    def execute_buy_at(
        self,
        token_id: str,
        timestamp: datetime,
        price: float,
        size_shares: float,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """
        Execute buy using historical orderbook at specific timestamp.

        Args:
            token_id: Token to buy
            timestamp: Time of trade
            price: Target price
            size_shares: Shares to buy
            liquidity_usd: Fallback liquidity estimate
            price_history: Fallback price history for spread estimation

        Returns:
            (execution_price, filled_shares, fee=0)
        """
        trade_value = size_shares * price

        # Try to get historical slippage estimate
        slippage_result = self.provider.estimate_slippage(
            token_id=token_id,
            timestamp=timestamp,
            size=trade_value,
            side=Side.BUY,
            tolerance_seconds=self.tolerance_seconds
        )

        if slippage_result:
            self.historical_fills += 1
            exec_price = slippage_result.average_price
            filled_shares = slippage_result.filled_size / exec_price if exec_price > 0 else 0

            logger.debug(
                f"Historical execution: {size_shares:.2f} shares @ {exec_price:.4f}, "
                f"slippage={slippage_result.slippage_pct:.2%}, "
                f"levels={slippage_result.levels_consumed}"
            )

            return exec_price, filled_shares, 0.0

        # Fallback to estimation
        self.fallback_fills += 1
        logger.debug(f"No historical data for {token_id} at {timestamp}, using fallback")

        return super().execute_buy(
            price=price,
            size_shares=size_shares,
            orderbook=None,
            liquidity_usd=liquidity_usd,
            price_history=price_history
        )

    def execute_sell_at(
        self,
        token_id: str,
        timestamp: datetime,
        price: float,
        size_shares: float,
        liquidity_usd: float = 1000.0,
        price_history: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """
        Execute sell using historical orderbook at specific timestamp.

        Args:
            token_id: Token to sell
            timestamp: Time of trade
            price: Target price
            size_shares: Shares to sell
            liquidity_usd: Fallback liquidity estimate
            price_history: Fallback price history for spread estimation

        Returns:
            (execution_price, filled_shares, fee=0)
        """
        trade_value = size_shares * price

        # Try to get historical slippage estimate
        slippage_result = self.provider.estimate_slippage(
            token_id=token_id,
            timestamp=timestamp,
            size=trade_value,
            side=Side.SELL,
            tolerance_seconds=self.tolerance_seconds
        )

        if slippage_result:
            self.historical_fills += 1
            exec_price = slippage_result.average_price
            filled_shares = slippage_result.filled_size / price if price > 0 else 0

            logger.debug(
                f"Historical execution: {size_shares:.2f} shares @ {exec_price:.4f}, "
                f"slippage={slippage_result.slippage_pct:.2%}, "
                f"levels={slippage_result.levels_consumed}"
            )

            return exec_price, filled_shares, 0.0

        # Fallback to estimation
        self.fallback_fills += 1
        logger.debug(f"No historical data for {token_id} at {timestamp}, using fallback")

        return super().execute_sell(
            price=price,
            size_shares=size_shares,
            orderbook=None,
            liquidity_usd=liquidity_usd,
            price_history=price_history
        )

    def can_trade_at(
        self,
        token_id: str,
        timestamp: datetime,
        min_liquidity_usd: float = 100.0
    ) -> Tuple[bool, str]:
        """
        Check if trading conditions were acceptable at a point in time.

        Args:
            token_id: Token to check
            timestamp: Time to check
            min_liquidity_usd: Minimum required liquidity

        Returns:
            (can_trade, reason)
        """
        metrics = self.provider.get_liquidity_metrics(
            token_id=token_id,
            timestamp=timestamp,
            tolerance_seconds=self.tolerance_seconds
        )

        if not metrics:
            return True, "No historical data, assuming tradeable"

        # Check spread
        if metrics.spread_pct and metrics.spread_pct > self.max_spread_pct:
            return False, f"Spread too wide: {metrics.spread_pct:.1%} > {self.max_spread_pct:.1%}"

        # Check liquidity
        total_liq = metrics.total_bid_liquidity + metrics.total_ask_liquidity
        if total_liq < min_liquidity_usd:
            return False, f"Insufficient liquidity: ${total_liq:.0f} < ${min_liquidity_usd:.0f}"

        return True, "OK"

    def get_stats(self) -> dict:
        """Get execution statistics."""
        total = self.historical_fills + self.fallback_fills
        return {
            "historical_fills": self.historical_fills,
            "fallback_fills": self.fallback_fills,
            "total_fills": total,
            "historical_rate": self.historical_fills / total if total > 0 else 0,
        }

