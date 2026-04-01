"""
Market making bot.

Loop: select markets -> generate quotes -> place/cancel orders -> manage inventory
Works with any ExchangeClient through the unified interface.

Includes inlined MM-specific components:
- AdaptiveQuoter: Volatility-scaled spreads, fair value model, asymmetric sizing, toxicity awareness
- ActiveMarketSelector: Instrument filtering
- InventoryManager: Position tracking with cross-instrument netting
- VolatilityTracker: Short-term mid-price volatility and drift estimation
- FairValueEstimator: Orderbook imbalance-based fair value
- FillToxicityTracker: Detects toxic (informed) vs passive fills
"""

import asyncio
import collections
import logging
import math
import time
from typing import Optional, Protocol

from ..core.enums import Side, Environment, OrderType
from ..core.models import Instrument, OrderRequest, OrderbookSnapshot, Quote
from ..core.errors import OmniTradeError
from ..exchanges.base import ExchangeClient
from ..risk.coordinator import RiskCoordinator

logger = logging.getLogger(__name__)


# === Protocols ===


class QuoteEngine(Protocol):
    """Protocol for quote engines."""

    async def generate_quote(
        self,
        client: ExchangeClient,
        instrument_id: str,
        inventory: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> Optional[Quote]: ...


# === Adaptive MM Components ===


class VolatilityTracker:
    """
    Tracks recent mid-price changes to estimate short-term volatility and drift.

    Used by AdaptiveQuoter to:
    - Scale spread width with realized volatility (wider when volatile)
    - Detect price trend direction for asymmetric sizing
    """

    def __init__(self, window: int = 20, min_samples: int = 3):
        self.window = window
        self.min_samples = min_samples
        self._mids: dict[str, collections.deque[float]] = {}

    def update(self, instrument_id: str, mid: float) -> None:
        if instrument_id not in self._mids:
            self._mids[instrument_id] = collections.deque(maxlen=self.window)
        self._mids[instrument_id].append(mid)

    def get_volatility(self, instrument_id: str) -> float:
        """Stdev of log-returns of recent mids. Returns 0 if insufficient data."""
        mids = self._mids.get(instrument_id)
        if not mids or len(mids) < self.min_samples:
            return 0.0
        log_returns = []
        mids_list = list(mids)
        for i in range(1, len(mids_list)):
            if mids_list[i - 1] > 0 and mids_list[i] > 0:
                log_returns.append(math.log(mids_list[i] / mids_list[i - 1]))
        if len(log_returns) < 2:
            return 0.0
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        return math.sqrt(variance)

    def get_drift(self, instrument_id: str) -> float:
        """Mean of recent log-returns. Positive = price rising, negative = falling."""
        mids = self._mids.get(instrument_id)
        if not mids or len(mids) < self.min_samples:
            return 0.0
        log_returns = []
        mids_list = list(mids)
        for i in range(1, len(mids_list)):
            if mids_list[i - 1] > 0 and mids_list[i] > 0:
                log_returns.append(math.log(mids_list[i] / mids_list[i - 1]))
        if not log_returns:
            return 0.0
        return sum(log_returns) / len(log_returns)


class FairValueEstimator:
    """
    Estimates fair value from orderbook imbalance and recent price drift.

    Instead of quoting around the raw midpoint, the MM quotes around an
    estimated fair value that accounts for:
    - Orderbook imbalance: heavy bid depth -> fair value above mid
    - Price drift: trending market shifts fair value in trend direction
    """

    def __init__(self, imbalance_weight: float = 0.3, drift_weight: float = 0.2):
        self.imbalance_weight = imbalance_weight
        self.drift_weight = drift_weight

    def estimate(
        self, orderbook: OrderbookSnapshot, drift: float = 0.0
    ) -> Optional[float]:
        mid = orderbook.midpoint
        if mid is None or mid <= 0:
            return None

        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        if best_bid is None or best_ask is None:
            return mid

        spread = best_ask - best_bid
        if spread <= 0:
            return mid

        # Orderbook imbalance from top-of-book depth
        bid_depth = sum(level.size for level in orderbook.bids[:5])
        ask_depth = sum(level.size for level in orderbook.asks[:5])
        total = bid_depth + ask_depth
        if total > 0:
            imbalance = (bid_depth - ask_depth) / total  # -1 to +1
        else:
            imbalance = 0.0

        fair_value = (
            mid
            + self.imbalance_weight * imbalance * spread
            + self.drift_weight * drift * spread
        )
        return fair_value


class FillToxicityTracker:
    """
    Tracks whether fills are toxic (immediate) or passive (delayed).

    An order that fills within `toxic_threshold_seconds` of placement is
    likely informed flow picking off a stale quote. High toxic ratios
    cause the adaptive quoter to widen spreads defensively.
    """

    def __init__(
        self,
        toxic_threshold_seconds: float = 1.0,
        window: int = 50,
        spread_penalty_scale: float = 0.5,
        max_tracked_orders: int = 10_000,
        stale_threshold_seconds: float = 60.0,
    ):
        self.toxic_threshold_seconds = toxic_threshold_seconds
        self.window = window
        self.spread_penalty_scale = spread_penalty_scale
        self.max_tracked_orders = max_tracked_orders
        self.stale_threshold_seconds = stale_threshold_seconds
        self._fills: dict[str, collections.deque[bool]] = {}
        self._order_timestamps: dict[str, float] = {}  # order_id -> placement monotonic time

    def record_order_placed(self, order_id: str) -> None:
        # Enforce hard cap before inserting to guarantee the new entry survives
        if len(self._order_timestamps) >= self.max_tracked_orders:
            self._evict_oldest()
        self._order_timestamps[order_id] = time.monotonic()

    def record_fill(self, order_id: str, instrument_id: str) -> None:
        placed_at = self._order_timestamps.pop(order_id, None)
        if placed_at is None:
            return
        elapsed = time.monotonic() - placed_at
        is_toxic = elapsed < self.toxic_threshold_seconds
        if instrument_id not in self._fills:
            self._fills[instrument_id] = collections.deque(maxlen=self.window)
        self._fills[instrument_id].append(is_toxic)

    def get_toxic_ratio(self, instrument_id: str) -> float:
        fills = self._fills.get(instrument_id)
        if not fills:
            return 0.0
        return sum(fills) / len(fills)

    def get_spread_penalty(self, instrument_id: str) -> float:
        """Additional half-spread to add based on toxic flow. 0.0 = no penalty."""
        return self.get_toxic_ratio(instrument_id) * self.spread_penalty_scale

    def cleanup_stale(self, max_age_seconds: Optional[float] = None) -> int:
        """Remove order timestamps older than max_age to prevent memory leak.

        Args:
            max_age_seconds: Override staleness threshold. Defaults to
                ``self.stale_threshold_seconds`` (60 s).

        Returns:
            Number of stale entries removed.
        """
        threshold = max_age_seconds if max_age_seconds is not None else self.stale_threshold_seconds
        now = time.monotonic()
        stale = [oid for oid, t in self._order_timestamps.items() if now - t > threshold]
        for oid in stale:
            del self._order_timestamps[oid]
        return len(stale)

    def _evict_oldest(self) -> None:
        """Evict the oldest half of tracked orders when hard cap is exceeded."""
        sorted_entries = sorted(self._order_timestamps.items(), key=lambda kv: kv[1])
        evict_count = len(sorted_entries) // 2
        for oid, _ in sorted_entries[:evict_count]:
            del self._order_timestamps[oid]
        logger.warning(
            "FillToxicityTracker: evicted %d oldest order timestamps (cap=%d)",
            evict_count, self.max_tracked_orders,
        )


# === Quote Engine ===


class AdaptiveQuoter:
    """
    Adaptive market making quoter for latency-disadvantaged environments.

    - Volatility-scaled spreads: wider when market is volatile, tighter when calm
    - Fair value model: quotes around estimated fair value (orderbook imbalance + drift)
      instead of raw midpoint
    - Asymmetric sizing: reduces size on the toxic side (the side more likely to
      get adversely selected based on recent price trend)
    - Toxicity awareness: widens spread when recent fills are disproportionately
      toxic (immediate fills from informed flow)
    - Quadratic inventory skew: more aggressive inventory reduction at high levels
    """

    def __init__(
        self,
        # Spread parameters
        base_half_spread: float = 0.015,
        vol_scale: float = 2.0,
        min_half_spread: float = 0.005,
        max_half_spread: float = 0.08,
        # Sizing
        size_usd: float = 25.0,
        max_contracts: float = 200.0,
        # Inventory
        max_inventory: float = 500.0,
        inventory_skew: float = 0.5,
        # Asymmetric sizing
        toxic_size_scale: float = 10.0,
        max_toxic_reduction: float = 0.5,
        # Volatility tracking
        vol_window: int = 20,
        vol_min_samples: int = 3,
        # Sub-components (created if not provided)
        volatility_tracker: Optional[VolatilityTracker] = None,
        fair_value_estimator: Optional[FairValueEstimator] = None,
        toxicity_tracker: Optional[FillToxicityTracker] = None,
    ):
        self.base_half_spread = base_half_spread
        self.vol_scale = vol_scale
        self.min_half_spread = min_half_spread
        self.max_half_spread = max_half_spread
        self.size_usd = size_usd
        self.max_contracts = max_contracts
        self.max_inventory = max_inventory
        self.inventory_skew = inventory_skew
        self.toxic_size_scale = toxic_size_scale
        self.max_toxic_reduction = max_toxic_reduction

        self.volatility_tracker = volatility_tracker or VolatilityTracker(
            window=vol_window, min_samples=vol_min_samples,
        )
        self.fair_value_estimator = fair_value_estimator or FairValueEstimator()
        self.toxicity_tracker = toxicity_tracker

    async def generate_quote(
        self,
        client: ExchangeClient,
        instrument_id: str,
        inventory: float,
        orderbook: Optional[OrderbookSnapshot] = None,
    ) -> Optional[Quote]:
        if orderbook is None:
            orderbook = await client.get_orderbook(instrument_id, depth=5)

        mid = orderbook.midpoint
        if mid is None or mid <= 0:
            return None

        # Update volatility tracker with latest mid
        self.volatility_tracker.update(instrument_id, mid)

        # Get volatility and drift
        vol = self.volatility_tracker.get_volatility(instrument_id)
        drift = self.volatility_tracker.get_drift(instrument_id)

        # Estimate fair value (replaces raw midpoint)
        fair_value = self.fair_value_estimator.estimate(orderbook, drift)
        if fair_value is None or fair_value <= 0:
            return None

        # Dynamic half-spread: base + volatility component + toxicity penalty
        vol_component = self.vol_scale * vol
        toxicity_penalty = (
            self.toxicity_tracker.get_spread_penalty(instrument_id)
            if self.toxicity_tracker else 0.0
        )
        effective_half_spread = self.base_half_spread + vol_component + toxicity_penalty
        effective_half_spread = max(
            self.min_half_spread,
            min(self.max_half_spread, effective_half_spread),
        )

        # Quadratic inventory skew: more aggressive at high inventory levels
        skew = 0.0
        if abs(inventory) > 0 and self.max_inventory > 0:
            ratio = inventory / self.max_inventory  # -1 to +1
            skew = (ratio * abs(ratio)) * self.inventory_skew * effective_half_spread

        bid_price = fair_value - effective_half_spread - skew
        ask_price = fair_value + effective_half_spread - skew

        if bid_price <= 0 or ask_price <= 0:
            return None

        # Base sizes
        bid_size = min(self.size_usd / bid_price, self.max_contracts)
        ask_size = min(self.size_usd / ask_price, self.max_contracts)

        # Asymmetric sizing: reduce size on the toxic side
        toxic_reduction = min(abs(drift) * self.toxic_size_scale, self.max_toxic_reduction)
        if drift > 0:
            # Price rising -> ask side is toxic (informed buyers lifting offers)
            ask_size *= (1.0 - toxic_reduction)
        else:
            # Price falling -> bid side is toxic (informed sellers hitting bids)
            bid_size *= (1.0 - toxic_reduction)

        return Quote(
            instrument_id=instrument_id,
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
        )


# === Market Selector ===


class ActiveMarketSelector:
    """Select all active instruments matching filters."""

    def __init__(
        self, min_price: float = 0.05, max_price: float = 0.95,
        max_instruments: int = 20,
    ):
        self.min_price = min_price
        self.max_price = max_price
        self.max_instruments = max_instruments

    async def select(self, client: ExchangeClient) -> list[Instrument]:
        instruments = await client.get_instruments(active_only=True)
        filtered = [
            i for i in instruments
            if self.min_price <= i.price <= self.max_price
        ]
        # Sort by most liquid (tightest spread)
        filtered.sort(key=lambda i: abs(i.ask - i.bid) if i.ask > 0 and i.bid > 0 else float('inf'))
        return filtered[:self.max_instruments]


# === Inventory Manager ===


class InventoryManager:
    """
    Tracks inventory across instruments for a market making bot.

    Provides current inventory levels and whether the bot should
    skew quotes to reduce exposure. Supports cross-instrument netting
    for YES/NO pairs on the same event.
    """

    def __init__(self, max_inventory_usd: float = 500.0):
        self.max_inventory_usd = max_inventory_usd
        self._inventory: dict[str, float] = {}  # instrument_id -> signed USD exposure
        self._pair_map: dict[str, str] = {}  # instrument_id -> paired instrument_id

    def register_pair(self, instrument_id_a: str, instrument_id_b: str) -> None:
        """Register two instruments as a YES/NO pair for inventory netting."""
        self._pair_map[instrument_id_a] = instrument_id_b
        self._pair_map[instrument_id_b] = instrument_id_a

    def update_from_fill(self, instrument_id: str, side: Side, size_usd: float) -> None:
        """Update inventory after a fill."""
        current = self._inventory.get(instrument_id, 0.0)
        if side == Side.BUY:
            self._inventory[instrument_id] = current + size_usd
        else:
            self._inventory[instrument_id] = current - size_usd

    def get_inventory(self, instrument_id: str) -> float:
        """Get current inventory in USD (positive = long, negative = short)."""
        return self._inventory.get(instrument_id, 0.0)

    def get_net_inventory(self, instrument_id: str) -> float:
        """Get inventory netted against the paired instrument.

        If long $100 YES and long $40 NO on the same event,
        net YES = 100 - 40 = $60, net NO = 40 - 100 = -$60.
        Returns raw inventory if no pair is registered.
        """
        raw = self.get_inventory(instrument_id)
        pair_id = self._pair_map.get(instrument_id)
        if pair_id is None:
            return raw
        pair_inv = self.get_inventory(pair_id)
        return raw - pair_inv

    def get_inventory_ratio(self, instrument_id: str) -> float:
        """Inventory as fraction of max (-1 to 1)."""
        inv = self.get_inventory(instrument_id)
        if self.max_inventory_usd <= 0:
            return 0.0
        return max(-1.0, min(1.0, inv / self.max_inventory_usd))

    @property
    def total_exposure(self) -> float:
        """Total absolute exposure across all instruments."""
        return sum(abs(v) for v in self._inventory.values())

    def is_at_limit(self, instrument_id: str) -> bool:
        """Check if inventory is at max."""
        return abs(self.get_inventory(instrument_id)) >= self.max_inventory_usd

    async def sync_from_exchange(self, client: ExchangeClient) -> None:
        """Sync inventory from exchange positions."""
        positions = await client.get_positions()
        self._inventory.clear()
        for pos in positions:
            value = pos.size
            if pos.side == Side.SELL:
                value = -value
            self._inventory[pos.instrument_id] = value
        logger.info(f"Synced inventory: {len(positions)} positions, ${self.total_exposure:.2f} total")


# === Market Making Bot ===


class MarketMakingBot:
    """
    Market making bot.

    Generates two-sided quotes and manages inventory.
    Exchange-agnostic through ExchangeClient interface.
    """

    def __init__(
        self,
        agent_id: str,
        client: ExchangeClient,
        quote_engine: QuoteEngine,
        market_selector: ActiveMarketSelector,
        risk: RiskCoordinator,
        inventory: Optional[InventoryManager] = None,
        toxicity_tracker: Optional[FillToxicityTracker] = None,
        environment: Environment = Environment.PAPER,
        max_instruments: int = 5,
        refresh_interval: float = 1.5,
    ):
        self.agent_id = agent_id
        self.client = client
        self.quote_engine = quote_engine
        self.market_selector = market_selector
        self.risk = risk
        self.inventory = inventory or InventoryManager()
        self.toxicity_tracker = toxicity_tracker
        self.environment = environment
        self.max_instruments = max_instruments
        self.refresh_interval = refresh_interval
        self._running = False
        self._active_orders: dict[str, list[str]] = {}  # instrument_id -> [order_ids]

    async def start(self) -> None:
        if not self.client.is_connected:
            await self.client.connect()
        self.risk.startup(self.agent_id, "market_making", self.client.exchange_id)
        await self.inventory.sync_from_exchange(self.client)
        logger.info(f"MarketMakingBot '{self.agent_id}' started on {self.client.exchange_id.value}")

    async def stop(self) -> None:
        self._running = False
        await self._cancel_all_tracked_orders()
        self.risk.shutdown(self.agent_id)
        logger.info(f"MarketMakingBot '{self.agent_id}' stopped")

    async def run(self, interval_seconds: Optional[float] = None) -> None:
        """Main loop."""
        interval = interval_seconds or self.refresh_interval
        self._running = True
        await self.start()

        try:
            while self._running:
                try:
                    await self._iteration()
                except OmniTradeError as e:
                    logger.error(f"MM error: {e}")
                    self.risk.record_failure()
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    self.risk.record_failure()

                self.risk.heartbeat(self.agent_id)
                await asyncio.sleep(interval)
        finally:
            await self.stop()

    async def _cancel_all_tracked_orders(self) -> int:
        """Cancel all tracked orders via batch cancel and clear tracking state."""
        all_ids = [oid for oids in self._active_orders.values() for oid in oids]
        if not all_ids:
            self._active_orders.clear()
            return 0
        logger.info(
            "Cancelling %d tracked orders: %s",
            len(all_ids), all_ids,
        )
        cancel_result = await self.client.cancel_orders(all_ids)
        cancelled = cancel_result.cancelled
        if cancel_result.failed_order_ids:
            logger.warning(
                "Failed to cancel %d orders: %s",
                len(cancel_result.failed_order_ids), cancel_result.failed_order_ids,
            )
        logger.info(
            "Cancel result: %d/%d orders cancelled", cancelled, len(all_ids),
        )
        self._active_orders.clear()
        return cancelled

    async def _iteration(self) -> None:
        """Single MM iteration: select -> cancel old -> quote -> place."""
        if not hasattr(self, '_mm_iteration_count'):
            self._mm_iteration_count = 0
        self._mm_iteration_count += 1
        is_verbose = self._mm_iteration_count % 10 == 1

        # Update balance and check drawdown
        balance = await self.client.get_balance()
        trading_allowed = self.risk.update_equity(balance.total_equity)

        if not trading_allowed:
            logger.warning(
                "Drawdown limit breached — cancelling all orders and stopping bot"
            )
            await self._cancel_all_tracked_orders()
            self._running = False
            return

        if is_verbose:
            logger.info(
                "[mm iter %d] balance=$%.2f, inventory=$%.2f, active_orders=%d",
                self._mm_iteration_count, balance.total_equity,
                self.inventory.total_exposure,
                sum(len(v) for v in self._active_orders.values()),
            )

        # Select markets
        instruments = await self.market_selector.select(self.client)
        instruments = instruments[:self.max_instruments]

        if is_verbose:
            names = [f"{i.instrument_id[:25]}@{i.price:.4f}" for i in instruments[:5]]
            logger.info(
                "[mm iter %d] quoting %d instruments: %s",
                self._mm_iteration_count, len(instruments), ", ".join(names),
            )

        # Deduplicate by instrument_id (API/filter can return the same instrument twice)
        seen_ids: set[str] = set()
        unique_instruments: list[Instrument] = []
        for inst in instruments:
            if inst.instrument_id not in seen_ids:
                seen_ids.add(inst.instrument_id)
                unique_instruments.append(inst)
        instruments = unique_instruments

        # Register YES/NO pairs for inventory netting
        by_market: dict[str, list[Instrument]] = {}
        for inst in instruments:
            if inst.market_id:
                by_market.setdefault(inst.market_id, []).append(inst)
        for group in by_market.values():
            if len(group) == 2:
                self.inventory.register_pair(
                    group[0].instrument_id, group[1].instrument_id
                )

        # Cancel ALL old orders before placing new ones — including orders for
        # instruments no longer in the current selection (dropped by selector).
        tracked_count = sum(len(v) for v in self._active_orders.values())
        logger.info(
            "[mm iter %d] tracked_orders=%d, instruments=%d",
            self._mm_iteration_count, tracked_count, len(instruments),
        )
        if self._active_orders:
            try:
                # Record fills for toxicity tracking before cancelling
                if self.toxicity_tracker:
                    for iid, oids in self._active_orders.items():
                        for oid in oids:
                            self.toxicity_tracker.record_fill(oid, iid)
                await self._cancel_all_tracked_orders()
            except Exception as e:
                logger.info("SKIP iteration: cancel failed (%s)", e)
                return
        else:
            logger.info("[mm iter %d] no tracked orders to cancel", self._mm_iteration_count)

        for inst in instruments:
            instrument_id = inst.instrument_id

            # Get net inventory (accounts for YES/NO pair netting)
            inv = self.inventory.get_net_inventory(instrument_id)

            # Generate quote
            quote = await self.quote_engine.generate_quote(
                self.client, instrument_id, inv
            )
            if quote is None:
                logger.debug("No quote for %s (no midpoint?)", instrument_id)
                continue

            # Place orders (paper mode just logs)
            if self.environment == Environment.PAPER:
                logger.info(
                    "QUOTE %s: BID %.2f@%.4f | ASK %.2f@%.4f (spread=%.2f%%, inv=$%.0f)",
                    instrument_id[:25], quote.bid_size, quote.bid_price,
                    quote.ask_size, quote.ask_price,
                    quote.spread * 100, inv,
                )
                continue

            # Determine which sides to quote based on inventory limits
            inv_ratio = self.inventory.get_inventory_ratio(instrument_id)
            place_bid = inv_ratio < 1.0   # don't buy more when at max long
            place_ask = inv_ratio > -1.0  # don't sell more when at max short

            # Place bid
            bid_result = None
            if place_bid:
                bid_result = await self.client.place_order(OrderRequest(
                    instrument_id=instrument_id,
                    side=Side.BUY,
                    size=quote.bid_size,
                    price=quote.bid_price,
                    order_type=OrderType.LIMIT,
                ))
                if bid_result.success and bid_result.order_id:
                    self._active_orders.setdefault(instrument_id, []).append(bid_result.order_id)
                    if self.toxicity_tracker:
                        self.toxicity_tracker.record_order_placed(bid_result.order_id)

            # Place ask
            ask_result = None
            if place_ask:
                ask_result = await self.client.place_order(OrderRequest(
                    instrument_id=instrument_id,
                    side=Side.SELL,
                    size=quote.ask_size,
                    price=quote.ask_price,
                    order_type=OrderType.LIMIT,
                ))
                if ask_result.success and ask_result.order_id:
                    self._active_orders.setdefault(instrument_id, []).append(ask_result.order_id)
                    if self.toxicity_tracker:
                        self.toxicity_tracker.record_order_placed(ask_result.order_id)

            # Log placed orders
            bid_status = "SKIP" if bid_result is None else ("FILLED" if bid_result.filled_size > 0 else bid_result.status.value)
            ask_status = "SKIP" if ask_result is None else ("FILLED" if ask_result.filled_size > 0 else ask_result.status.value)
            logger.info(
                "PLACED %s: BID %.2f@%.4f (%s) | ASK %.2f@%.4f (%s)",
                instrument_id[:25],
                quote.bid_size, quote.bid_price, bid_status,
                quote.ask_size, quote.ask_price, ask_status,
            )

            # Update inventory if filled
            if bid_result and bid_result.filled_size > 0:
                self.inventory.update_from_fill(instrument_id, Side.BUY, bid_result.filled_size)
                logger.info(
                    "FILL BUY %s: %.4f @ $%.4f (inv now $%.0f)",
                    instrument_id[:25], bid_result.filled_size, bid_result.filled_price,
                    self.inventory.get_inventory(instrument_id),
                )
            if ask_result and ask_result.filled_size > 0:
                self.inventory.update_from_fill(instrument_id, Side.SELL, ask_result.filled_size)
                logger.info(
                    "FILL SELL %s: %.4f @ $%.4f (inv now $%.0f)",
                    instrument_id[:25], ask_result.filled_size, ask_result.filled_price,
                    self.inventory.get_inventory(instrument_id),
                )

        # Cleanup stale order timestamps every iteration to bound memory usage
        if self.toxicity_tracker:
            self.toxicity_tracker.cleanup_stale()
