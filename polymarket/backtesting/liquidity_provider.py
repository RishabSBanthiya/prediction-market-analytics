"""
Historical liquidity provider for backtesting.

Provides an interface to query recorded orderbook data for realistic
execution simulation during backtests.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

from ..core.models import OrderbookSnapshot, Side
from ..data.orderbook_storage import OrderbookStorage, StoredSnapshot

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Result of slippage estimation."""
    size: float
    side: Side
    average_price: float
    worst_price: float
    slippage_pct: float
    fully_filled: bool
    filled_size: float
    levels_consumed: int


@dataclass
class LiquidityMetrics:
    """Liquidity metrics at a point in time."""
    timestamp: datetime
    token_id: str
    spread_pct: Optional[float]
    midpoint: Optional[float]
    total_bid_liquidity: float
    total_ask_liquidity: float
    bid_depth_1pct: float  # Liquidity within 1% of mid
    ask_depth_1pct: float
    bid_depth_5pct: float  # Liquidity within 5% of mid
    ask_depth_5pct: float


class HistoricalLiquidityProvider:
    """
    Provides historical orderbook data for backtesting.

    Usage:
        provider = HistoricalLiquidityProvider()

        # Get orderbook at specific time
        snapshot = provider.get_orderbook_at(token_id, timestamp)

        # Estimate slippage for a trade
        slippage = provider.estimate_slippage(token_id, timestamp, 1000, Side.BUY)

        # Get spread history for analysis
        spreads = provider.get_spread_series(token_id, start, end)
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize provider.

        Args:
            db_path: Path to orderbook database. Uses default if None.
        """
        self.storage = OrderbookStorage(db_path)
        self._cache: Dict[str, StoredSnapshot] = {}
        self._cache_window_seconds = 60

    def has_data_for_period(
        self,
        token_id: str,
        start: datetime,
        end: datetime,
        min_snapshots: int = 10
    ) -> bool:
        """
        Check if sufficient orderbook data exists for a time period.

        Args:
            token_id: Token to check
            start: Start of period
            end: End of period
            min_snapshots: Minimum snapshots required

        Returns:
            True if sufficient data exists
        """
        count = self.storage.get_snapshot_count(token_id, start, end)
        return count >= min_snapshots

    def get_orderbook_at(
        self,
        token_id: str,
        timestamp: datetime,
        tolerance_seconds: int = 60
    ) -> Optional[OrderbookSnapshot]:
        """
        Get orderbook snapshot closest to a timestamp.

        Args:
            token_id: Token to query
            timestamp: Target timestamp
            tolerance_seconds: Max seconds difference allowed

        Returns:
            OrderbookSnapshot or None if no data within tolerance
        """
        stored = self.storage.get_snapshot_at(token_id, timestamp, tolerance_seconds)
        if not stored:
            return None

        return OrderbookSnapshot(
            token_id=stored.token_id,
            timestamp=stored.timestamp,
            best_bid=stored.best_bid,
            best_ask=stored.best_ask,
            bid_size=stored.bid_size,
            ask_size=stored.ask_size,
            bid_depth=stored.bid_depth,
            ask_depth=stored.ask_depth,
        )

    def estimate_slippage(
        self,
        token_id: str,
        timestamp: datetime,
        size: float,
        side: Side,
        tolerance_seconds: int = 60
    ) -> Optional[SlippageEstimate]:
        """
        Estimate slippage for a trade by walking the historical orderbook.

        Args:
            token_id: Token to trade
            timestamp: Time of trade
            size: Trade size in USD
            side: BUY or SELL
            tolerance_seconds: Max timestamp difference for snapshot

        Returns:
            SlippageEstimate with execution details, or None if no data
        """
        snapshot = self.storage.get_snapshot_at(token_id, timestamp, tolerance_seconds)
        if not snapshot:
            return None

        # Determine which side of the book to consume
        if side == Side.BUY:
            # Buying: consume asks (price ascending)
            depth = snapshot.ask_depth
            best_price = snapshot.best_ask
        else:
            # Selling: consume bids (price descending)
            depth = snapshot.bid_depth
            best_price = snapshot.best_bid

        if not depth or best_price is None:
            return None

        # Walk the book
        remaining = size
        total_cost = 0.0
        levels_consumed = 0
        worst_price = best_price

        for price, level_size in depth:
            if remaining <= 0:
                break

            # Calculate how much we can fill at this level
            level_value = level_size * price
            fill_value = min(remaining, level_value)
            fill_shares = fill_value / price if price > 0 else 0

            total_cost += fill_value
            remaining -= fill_value
            levels_consumed += 1
            worst_price = price

        filled_size = size - remaining
        fully_filled = remaining <= 0

        # Calculate average and slippage
        if filled_size > 0:
            # Average price weighted by fill
            shares_bought = total_cost / ((total_cost / filled_size) if filled_size > 0 else best_price)
            avg_price = total_cost / shares_bought if shares_bought > 0 else best_price

            # Slippage vs best price
            if side == Side.BUY:
                slippage_pct = (avg_price - best_price) / best_price if best_price > 0 else 0
            else:
                slippage_pct = (best_price - avg_price) / best_price if best_price > 0 else 0
        else:
            avg_price = best_price
            slippage_pct = 0

        return SlippageEstimate(
            size=size,
            side=side,
            average_price=avg_price,
            worst_price=worst_price,
            slippage_pct=max(0, slippage_pct),
            fully_filled=fully_filled,
            filled_size=filled_size,
            levels_consumed=levels_consumed,
        )

    def get_spread_series(
        self,
        token_id: str,
        start: datetime,
        end: datetime
    ) -> List[Tuple[datetime, Optional[float]]]:
        """
        Get spread percentage time series.

        Args:
            token_id: Token to query
            start: Start of period
            end: End of period

        Returns:
            List of (timestamp, spread_pct) tuples
        """
        return self.storage.get_spread_history(token_id, start, end)

    def get_liquidity_metrics(
        self,
        token_id: str,
        timestamp: datetime,
        tolerance_seconds: int = 60
    ) -> Optional[LiquidityMetrics]:
        """
        Get comprehensive liquidity metrics at a point in time.

        Args:
            token_id: Token to query
            timestamp: Target timestamp
            tolerance_seconds: Max timestamp difference

        Returns:
            LiquidityMetrics or None
        """
        snapshot = self.storage.get_snapshot_at(token_id, timestamp, tolerance_seconds)
        if not snapshot:
            return None

        # Calculate depth at various distances from midpoint
        mid = snapshot.midpoint
        bid_1pct = 0.0
        ask_1pct = 0.0
        bid_5pct = 0.0
        ask_5pct = 0.0

        if mid and mid > 0:
            for price, size in snapshot.bid_depth:
                distance = (mid - price) / mid
                if distance <= 0.01:
                    bid_1pct += size
                if distance <= 0.05:
                    bid_5pct += size

            for price, size in snapshot.ask_depth:
                distance = (price - mid) / mid
                if distance <= 0.01:
                    ask_1pct += size
                if distance <= 0.05:
                    ask_5pct += size

        return LiquidityMetrics(
            timestamp=snapshot.timestamp,
            token_id=snapshot.token_id,
            spread_pct=snapshot.spread_pct,
            midpoint=snapshot.midpoint,
            total_bid_liquidity=snapshot.total_bid_liquidity,
            total_ask_liquidity=snapshot.total_ask_liquidity,
            bid_depth_1pct=bid_1pct,
            ask_depth_1pct=ask_1pct,
            bid_depth_5pct=bid_5pct,
            ask_depth_5pct=ask_5pct,
        )

    def get_average_spread(
        self,
        token_id: str,
        start: datetime,
        end: datetime
    ) -> Optional[float]:
        """
        Get average spread percentage over a period.

        Args:
            token_id: Token to query
            start: Start of period
            end: End of period

        Returns:
            Average spread percentage or None if no data
        """
        spreads = self.storage.get_spread_history(token_id, start, end)
        valid_spreads = [s for _, s in spreads if s is not None]

        if not valid_spreads:
            return None

        return sum(valid_spreads) / len(valid_spreads)

    def get_average_liquidity(
        self,
        token_id: str,
        start: datetime,
        end: datetime,
        side: Optional[Side] = None
    ) -> Optional[float]:
        """
        Get average total liquidity over a period.

        Args:
            token_id: Token to query
            start: Start of period
            end: End of period
            side: If specified, only count that side

        Returns:
            Average liquidity in USD or None if no data
        """
        snapshots = self.storage.get_snapshots(token_id, start, end)
        if not snapshots:
            return None

        total = 0.0
        for s in snapshots:
            if side == Side.BUY:
                total += s.total_ask_liquidity
            elif side == Side.SELL:
                total += s.total_bid_liquidity
            else:
                total += s.total_bid_liquidity + s.total_ask_liquidity

        return total / len(snapshots)

    def get_available_markets(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[str]:
        """
        Get list of market IDs with recorded data.

        Args:
            start: Optional start time filter
            end: Optional end time filter

        Returns:
            List of market condition_ids
        """
        markets = self.storage.get_available_markets(start, end)
        return [m.market_id for m in markets]

    def get_data_coverage(
        self,
        token_id: str,
        start: datetime,
        end: datetime,
        expected_interval_seconds: int = 30
    ) -> float:
        """
        Calculate what percentage of expected snapshots exist.

        Args:
            token_id: Token to check
            start: Start of period
            end: End of period
            expected_interval_seconds: Expected recording interval

        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        actual = self.storage.get_snapshot_count(token_id, start, end)
        duration = (end - start).total_seconds()
        expected = duration / expected_interval_seconds

        if expected <= 0:
            return 0.0

        return min(1.0, actual / expected)
