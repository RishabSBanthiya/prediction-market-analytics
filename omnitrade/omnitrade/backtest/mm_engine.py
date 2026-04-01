"""
Market-making backtest engine.

Extends the base backtest engine with limit-order fill simulation so the
MarketMakingBot can run in LIVE mode against real orderbook data.
"""

import logging
import time
import tempfile
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..core.enums import (
    ExchangeId, Side, OrderStatus, OrderType,
    InstrumentType, Environment,
)
from ..core.models import (
    Instrument, OrderbookSnapshot,
    OrderRequest, OrderResult, OpenOrder,
)
from ..core.config import RiskConfig
from ..bots.market_making import (
    AdaptiveQuoter, ActiveMarketSelector, InventoryManager, MarketMakingBot, QuoteEngine,
)
from ..risk.coordinator import RiskCoordinator
from ..storage.sqlite import SQLiteStorage

from .engine import (
    BacktestExchangeClient,
    BacktestResult,
    BacktestProgress,
    ProgressCallback,
    compute_sharpe,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limit-order fill simulation client
# ---------------------------------------------------------------------------

@dataclass
class _OpenOrder:
    """Internal resting order."""
    order_id: str
    instrument_id: str
    side: Side
    size: float
    price: float
    created_step: int


@dataclass
class _Fill:
    """Record of a single fill."""
    order_id: str
    side: Side
    price: float
    size: float
    step: int
    timestamp: datetime


class MMBacktestExchangeClient(BacktestExchangeClient):
    """BacktestExchangeClient with limit-order resting / fill simulation."""

    def __init__(
        self,
        snapshots: list[OrderbookSnapshot],
        instrument: Instrument,
        initial_balance: float = 10_000,
        exchange_id: ExchangeId = ExchangeId.POLYMARKET,
    ):
        super().__init__(snapshots, instrument, initial_balance, exchange_id)
        self._open_orders: dict[str, _OpenOrder] = {}
        self._fills: list[_Fill] = []
        self._cancelled_count = 0

    # -- trading (limit fill simulation) --

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self._order_counter += 1
        order_id = f"MM-{self._order_counter:06d}"
        snap = self.current_snapshot

        # Check for immediate fill
        if request.side == Side.BUY:
            if snap.best_ask is not None and snap.best_ask <= request.price:
                fill_price = snap.best_ask
                fill = _Fill(
                    order_id=order_id,
                    side=Side.BUY,
                    price=fill_price,
                    size=request.size,
                    step=self._step,
                    timestamp=snap.timestamp,
                )
                self._fills.append(fill)
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_size=request.size,
                    filled_price=fill_price,
                    requested_size=request.size,
                    requested_price=request.price,
                )
        else:  # SELL
            if snap.best_bid is not None and snap.best_bid >= request.price:
                fill_price = snap.best_bid
                fill = _Fill(
                    order_id=order_id,
                    side=Side.SELL,
                    price=fill_price,
                    size=request.size,
                    step=self._step,
                    timestamp=snap.timestamp,
                )
                self._fills.append(fill)
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_size=request.size,
                    filled_price=fill_price,
                    requested_size=request.size,
                    requested_price=request.price,
                )

        # No immediate fill → resting order
        self._open_orders[order_id] = _OpenOrder(
            order_id=order_id,
            instrument_id=request.instrument_id,
            side=request.side,
            size=request.size,
            price=request.price,
            created_step=self._step,
        )
        return OrderResult(
            success=True,
            order_id=order_id,
            status=OrderStatus.OPEN,
            filled_size=0.0,
            filled_price=0.0,
            requested_size=request.size,
            requested_price=request.price,
        )

    def advance(self) -> None:
        """Move to next snapshot and check resting orders for fills."""
        super().advance()
        self._check_fills()

    def _check_fills(self) -> None:
        """Scan open orders against current snapshot and fill any that cross."""
        snap = self.current_snapshot
        filled_ids: list[str] = []

        for order_id, order in self._open_orders.items():
            if order.side == Side.BUY:
                if snap.best_ask is not None and snap.best_ask <= order.price:
                    fill = _Fill(
                        order_id=order_id,
                        side=Side.BUY,
                        price=snap.best_ask,
                        size=order.size,
                        step=self._step,
                        timestamp=snap.timestamp,
                    )
                    self._fills.append(fill)
                    filled_ids.append(order_id)
            else:  # SELL
                if snap.best_bid is not None and snap.best_bid >= order.price:
                    fill = _Fill(
                        order_id=order_id,
                        side=Side.SELL,
                        price=snap.best_bid,
                        size=order.size,
                        step=self._step,
                        timestamp=snap.timestamp,
                    )
                    self._fills.append(fill)
                    filled_ids.append(order_id)

        for oid in filled_ids:
            del self._open_orders[oid]

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        if order_id in self._open_orders:
            del self._open_orders[order_id]
            self._cancelled_count += 1
            return True
        return False

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        if instrument_id:
            to_cancel = [
                oid for oid, o in self._open_orders.items()
                if o.instrument_id == instrument_id
            ]
        else:
            to_cancel = list(self._open_orders.keys())
        for oid in to_cancel:
            del self._open_orders[oid]
        self._cancelled_count += len(to_cancel)
        return len(to_cancel)

    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        result = []
        for o in self._open_orders.values():
            if instrument_id and o.instrument_id != instrument_id:
                continue
            result.append(OpenOrder(
                order_id=o.order_id,
                instrument_id=o.instrument_id,
                side=o.side,
                size=o.size,
                filled_size=0.0,
                price=o.price,
                order_type=OrderType.LIMIT,
                status=OrderStatus.OPEN,
            ))
        return result

    # -- metrics accessors --

    @property
    def fills(self) -> list[_Fill]:
        return list(self._fills)

    @property
    def total_bid_fills(self) -> int:
        return sum(1 for f in self._fills if f.side == Side.BUY)

    @property
    def total_ask_fills(self) -> int:
        return sum(1 for f in self._fills if f.side == Side.SELL)

    @property
    def cancelled_count(self) -> int:
        return self._cancelled_count


# ---------------------------------------------------------------------------
# MM-specific result
# ---------------------------------------------------------------------------

@dataclass
class MMBacktestResult(BacktestResult):
    """BacktestResult extended with market-making metrics."""
    total_quotes: int = 0
    bid_fill_rate: float = 0.0
    ask_fill_rate: float = 0.0
    total_volume: float = 0.0
    spread_captured: float = 0.0
    peak_inventory: float = 0.0
    avg_inventory: float = 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MMBacktestRunner:
    """Runs MarketMakingBot in LIVE mode against real orderbook data."""

    def __init__(
        self,
        snapshots: list[OrderbookSnapshot],
        instrument_id: str = "EXT-001",
        scenario_name: str = "real_data",
        quote_engine: Optional[QuoteEngine] = None,
        initial_balance: float = 10_000,
        exchange_id: ExchangeId = ExchangeId.POLYMARKET,
        on_progress: Optional[ProgressCallback] = None,
        progress_interval: int = 5000,
        subsample: int = 1,
    ):
        self.quote_engine = quote_engine or AdaptiveQuoter()
        self.initial_balance = initial_balance
        self.exchange_id = exchange_id
        self._snapshots = snapshots
        self._instrument_id = instrument_id
        self._scenario_name = scenario_name
        self._on_progress = on_progress
        self._progress_interval = progress_interval
        self._subsample = subsample

    async def run(self) -> MMBacktestResult:
        snapshots = self._snapshots

        instrument = Instrument(
            instrument_id=self._instrument_id,
            exchange=self.exchange_id,
            instrument_type=InstrumentType.BINARY_OUTCOME,
            name=f"MM Backtest {self._scenario_name}",
            market_id=f"mm-{self._scenario_name}",
            outcome="YES",
            active=True,
            price=snapshots[0].midpoint or 0.5,
            bid=snapshots[0].best_bid or 0.48,
            ask=snapshots[0].best_ask or 0.52,
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()

        try:
            return await self._execute(db_path, snapshots, instrument)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

    async def _execute(
        self,
        db_path: str,
        snapshots: list[OrderbookSnapshot],
        instrument: Instrument,
    ) -> MMBacktestResult:
        storage = SQLiteStorage(db_path)
        storage.initialize()

        exchange_str = self.exchange_id.value
        client = MMBacktestExchangeClient(snapshots, instrument, self.initial_balance, self.exchange_id)

        storage.update_balance(exchange_str, "", self.initial_balance)

        risk_config = RiskConfig(
            max_wallet_exposure_pct=0.80,
            max_per_agent_exposure_pct=0.60,
            max_per_market_exposure_pct=0.60,
            min_trade_value_usd=1.0,
            max_trade_value_usd=5000.0,
            max_daily_drawdown_pct=0.30,
            max_total_drawdown_pct=0.50,
        )

        risk = RiskCoordinator(storage, risk_config)
        risk.register_account(self.exchange_id, "")

        market_selector = ActiveMarketSelector(
            min_price=0.01, max_price=0.99, max_instruments=5,
        )
        inventory = InventoryManager(max_inventory_usd=500.0)

        agent_id = f"mm-bt-{self._scenario_name}"

        bot = MarketMakingBot(
            agent_id=agent_id,
            client=client,
            quote_engine=self.quote_engine,
            market_selector=market_selector,
            risk=risk,
            inventory=inventory,
            environment=Environment.LIVE,  # Must be LIVE so place_order is called
            max_instruments=1,
        )

        # -- Snapshot deduplication: skip steps where midpoint is unchanged --
        active_indices = [0]
        for i in range(1, len(snapshots)):
            if snapshots[i].midpoint != snapshots[i - 1].midpoint:
                active_indices.append(i)
        # Always include the last snapshot
        if active_indices[-1] != len(snapshots) - 1:
            active_indices.append(len(snapshots) - 1)

        if self._subsample > 1:
            # Keep every Nth active index, always keeping the last
            subsampled = active_indices[::self._subsample]
            if subsampled[-1] != active_indices[-1]:
                subsampled.append(active_indices[-1])
            active_indices = subsampled
            logger.info("  Subsampled to %d steps (every %d)", len(active_indices), self._subsample)

        skipped = len(snapshots) - len(active_indices)
        logger.info(
            "  MM engine: %d active steps (skipped %d unchanged snapshots)",
            len(active_indices), skipped,
        )

        await bot.start()
        equity_curve = [self.initial_balance]
        inventory_history: list[float] = [0.0]
        total_steps = len(active_indices) - 1
        t_start = time.monotonic()

        # Track total quotes placed (each iteration places bid+ask = 1 quote pair)
        total_quotes = 0

        # -- Persistent FIFO PnL state (incremental, not recomputed each step) --
        fifo_buys: deque[list[float]] = deque()  # each element: [price, size]
        net_position = 0.0
        realized_pnl = 0.0
        fills_processed = 0

        # Throttle storage writes: only every N steps or on new fills
        _BALANCE_UPDATE_INTERVAL = 100

        for loop_idx in range(total_steps):
            target_step = active_indices[loop_idx + 1]

            # Jump client to the target snapshot (same pattern as directional engine)
            client._step = target_step

            fills_before = len(client._fills)
            try:
                await bot._iteration()
            except Exception as e:
                logger.debug("MM Iteration %d error: %s", loop_idx, e)

            fills_after = len(client._fills)
            # Each iteration that produces orders counts as a quote
            if fills_after > fills_before or len(client._open_orders) > 0:
                total_quotes += 1

            client.advance()

            # -- Incremental FIFO PnL: only process NEW fills --
            mid = client.current_snapshot.midpoint or 0.5
            new_fill_count = len(client._fills)
            for fi in range(fills_processed, new_fill_count):
                fill = client._fills[fi]
                if fill.side == Side.BUY:
                    fifo_buys.append([fill.price, fill.size])
                    net_position += fill.size
                else:
                    shares_to_close = fill.size
                    while shares_to_close > 0 and fifo_buys:
                        entry = fifo_buys[0]
                        matched = min(entry[1], shares_to_close)
                        realized_pnl += matched * (fill.price - entry[0])
                        entry[1] -= matched
                        if entry[1] <= 1e-10:
                            fifo_buys.popleft()
                        shares_to_close -= matched
                        net_position -= matched
            fills_processed = new_fill_count

            unrealized_pnl = sum(e[1] * (mid - e[0]) for e in fifo_buys)
            equity = self.initial_balance + realized_pnl + unrealized_pnl
            inv_usd = abs(net_position * mid)

            client._equity = equity
            client._available = max(0.0, equity - inv_usd)

            # Throttle storage writes
            has_new_fills = new_fill_count > fills_before
            if has_new_fills or loop_idx % _BALANCE_UPDATE_INTERVAL == 0 or loop_idx == total_steps - 1:
                storage.update_balance(exchange_str, "", equity)

            equity_curve.append(equity)
            inventory_history.append(inv_usd)

            # Update instrument price for ActiveMarketSelector
            instrument.price = mid
            instrument.bid = client.current_snapshot.best_bid or mid - 0.01
            instrument.ask = client.current_snapshot.best_ask or mid + 0.01

            # Progress callback
            if self._on_progress and (loop_idx + 1) % self._progress_interval == 0:
                self._on_progress(BacktestProgress(
                    step=loop_idx + 1,
                    total_steps=total_steps,
                    elapsed_secs=time.monotonic() - t_start,
                    equity=equity,
                    pnl=equity - self.initial_balance,
                    open_positions=len(client._open_orders),
                    closed_trades=len(client._fills),
                    mid_price=mid,
                    market_time=client.current_snapshot.timestamp,
                ))

        await bot.stop()

        # Compute data duration from snapshot timestamps for Sharpe
        first_ts = snapshots[0].timestamp
        last_ts = snapshots[-1].timestamp
        total_duration_secs = (last_ts - first_ts).total_seconds()

        result = self._compute_mm_metrics(
            client, equity_curve, inventory_history, total_quotes,
            self._scenario_name, total_duration_secs,
        )
        storage.close()
        return result

    def _compute_mm_metrics(
        self,
        client: MMBacktestExchangeClient,
        equity_curve: list[float],
        inventory_history: list[float],
        total_quotes: int,
        scenario_label: str,
        total_duration_secs: float = 0.0,
    ) -> MMBacktestResult:

        fills = client.fills
        bid_fills = [f for f in fills if f.side == Side.BUY]
        ask_fills = [f for f in fills if f.side == Side.SELL]

        total_trades = len(fills)
        total_volume = sum(f.price * f.size for f in fills)

        # Bid/ask fill rates
        total_bids_placed = len(bid_fills) + client.cancelled_count // 2
        total_asks_placed = len(ask_fills) + client.cancelled_count // 2
        bid_fill_rate = (
            len(bid_fills) / max(1, total_bids_placed)
        )
        ask_fill_rate = (
            len(ask_fills) / max(1, total_asks_placed)
        )

        # Spread captured: average spread between bid and ask fill prices
        # in round-trip pairs
        spread_captured = 0.0
        if bid_fills and ask_fills:
            avg_bid = sum(f.price for f in bid_fills) / len(bid_fills)
            avg_ask = sum(f.price for f in ask_fills) / len(ask_fills)
            spread_captured = avg_ask - avg_bid

        # Inventory metrics
        peak_inventory = max(inventory_history) if inventory_history else 0.0
        avg_inventory = (
            sum(inventory_history) / len(inventory_history)
            if inventory_history else 0.0
        )

        # PnL — realized from round trips
        buys: list[tuple[float, float]] = []
        realized_pnl = 0.0
        winning = 0
        losing = 0
        for fill in fills:
            if fill.side == Side.BUY:
                buys.append((fill.price, fill.size))
            else:
                shares_to_close = fill.size
                while shares_to_close > 0 and buys:
                    bp, bs = buys[0]
                    matched = min(bs, shares_to_close)
                    trip_pnl = matched * (fill.price - bp)
                    realized_pnl += trip_pnl
                    if trip_pnl > 0:
                        winning += 1
                    elif trip_pnl < 0:
                        losing += 1
                    buys[0] = (bp, bs - matched)
                    if buys[0][1] <= 1e-10:
                        buys.pop(0)
                    shares_to_close -= matched

        final_equity = equity_curve[-1] if equity_curve else self.initial_balance
        total_pnl = final_equity - self.initial_balance

        win_rate = winning / (winning + losing) if (winning + losing) > 0 else 0.0
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        sharpe = compute_sharpe(equity_curve, total_duration_secs)

        return MMBacktestResult(
            signal_name="market_making",
            scenario_name=scenario_label,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_trade_pnl=avg_trade_pnl,
            equity_curve=equity_curve,
            final_equity=final_equity,
            # MM-specific
            total_quotes=total_quotes,
            bid_fill_rate=bid_fill_rate,
            ask_fill_rate=ask_fill_rate,
            total_volume=total_volume,
            spread_captured=spread_captured,
            peak_inventory=peak_inventory,
            avg_inventory=avg_inventory,
        )
