"""
Backtesting engine for OmniTrade.

Replays real orderbook data through the DirectionalBot pipeline and computes
performance metrics. Designed for comparing signal sources against historical
market data.
"""

import math
import logging
import time
import tempfile
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable

from ..core.enums import ExchangeId, OrderStatus, InstrumentType, Environment
from ..core.models import (
    Instrument, OrderbookSnapshot,
    OrderRequest, OrderResult, AccountBalance, ExchangePosition, OpenOrder,
)
from ..core.config import ExchangeConfig, RiskConfig
from ..exchanges.base import ExchangeClient
from ..components.signals import SignalSource
from ..components.trading import FixedFractionSizer, ExitConfig
from ..exchanges.base import PaperClient
from ..risk.coordinator import RiskCoordinator
from ..storage.sqlite import SQLiteStorage
from ..bots.directional import DirectionalBot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Sharpe calculation
# ---------------------------------------------------------------------------

def compute_sharpe(
    equity_curve: list[float],
    total_duration_seconds: float,
) -> float:
    """Compute annualized Sharpe ratio from an equity curve.

    Splits the equity curve into N equal-duration periods (targeting ~1 day
    each) and computes period returns.  Annualizes with sqrt(periods_per_year).

    The number of periods is derived from the total data duration, NOT from
    the number of equity-curve points — this keeps the calculation correct
    even when snapshot dedup causes the curve to have fewer points than
    the original timeline would suggest.

    Args:
        equity_curve: List of equity values (one per backtest step).
        total_duration_seconds: Wall-clock duration of the data (first to
            last snapshot timestamp).

    Returns:
        Annualized Sharpe ratio, or 0.0 if it cannot be computed.
    """
    n = len(equity_curve)
    if n < 2 or total_duration_seconds <= 0:
        return 0.0

    # How many ~1-day periods does the data span?
    total_days = total_duration_seconds / 86_400
    num_periods = max(2, int(round(total_days)))

    # Clamp: can't have more periods than equity-curve points
    if num_periods >= n:
        num_periods = max(2, n // 2)

    # Split equity curve into num_periods equal-sized buckets
    period_returns: list[float] = []
    for p in range(num_periods):
        i_start = int(p * (n - 1) / num_periods)
        i_end = int((p + 1) * (n - 1) / num_periods)
        start_eq = equity_curve[i_start]
        end_eq = equity_curve[i_end]
        if start_eq > 0:
            period_returns.append((end_eq - start_eq) / start_eq)

    if len(period_returns) < 2:
        return 0.0

    mean_ret = sum(period_returns) / len(period_returns)
    variance = sum((r - mean_ret) ** 2 for r in period_returns) / len(period_returns)
    std_ret = variance ** 0.5
    if std_ret == 0:
        return 0.0

    # Annualize: each period is ~(total_days / num_periods) days long
    days_per_period = total_days / num_periods
    periods_per_year = 365.0 / days_per_period
    return mean_ret / std_ret * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# Replay exchange client
# ---------------------------------------------------------------------------

class BacktestExchangeClient(ExchangeClient):
    """Exchange client that replays pre-generated orderbook snapshots."""

    def __init__(
        self,
        snapshots: list[OrderbookSnapshot],
        instrument: Instrument,
        initial_balance: float = 10_000,
        exchange_id: ExchangeId = ExchangeId.POLYMARKET,
    ):
        config = ExchangeConfig(exchange=exchange_id)
        super().__init__(config)
        self._exchange_id = exchange_id
        self._snapshots = snapshots
        self._instrument = instrument
        self._step = 0
        self._initial_balance = initial_balance
        self._equity = initial_balance
        self._available = initial_balance
        self._order_counter = 0

    @property
    def exchange_id(self) -> ExchangeId:
        return self._exchange_id

    @property
    def current_snapshot(self) -> OrderbookSnapshot:
        return self._snapshots[min(self._step, len(self._snapshots) - 1)]

    def advance(self) -> None:
        """Move to the next orderbook snapshot."""
        self._step = min(self._step + 1, len(self._snapshots) - 1)

    # -- lifecycle --

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    # -- market data --

    async def get_instruments(self, active_only: bool = True, **filters) -> list[Instrument]:
        return [self._instrument]

    async def get_instrument(self, instrument_id: str) -> Optional[Instrument]:
        if instrument_id == self._instrument.instrument_id:
            return self._instrument
        return None

    async def get_orderbook(self, instrument_id: str, depth: int = 10) -> OrderbookSnapshot:
        return self.current_snapshot

    # -- trading (immediate fill) --

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self._order_counter += 1
        return OrderResult(
            success=True,
            order_id=f"BT-{self._order_counter:06d}",
            status=OrderStatus.FILLED,
            filled_size=request.size,
            filled_price=request.price,
            requested_size=request.size,
            requested_price=request.price,
        )

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        return True

    async def cancel_all_orders(self, instrument_id: str | None = None) -> int:
        return 0

    async def get_open_orders(self, instrument_id: str | None = None) -> list[OpenOrder]:
        return []

    # -- account --

    async def get_balance(self) -> AccountBalance:
        return AccountBalance(
            exchange=self._exchange_id,
            total_equity=self._equity,
            available_balance=max(0.0, self._available),
        )

    async def get_positions(self) -> list[ExchangePosition]:
        return []


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestProgress:
    """Snapshot of backtest state at a given step."""
    step: int
    total_steps: int
    elapsed_secs: float
    equity: float
    pnl: float
    open_positions: int
    closed_trades: int
    mid_price: float
    market_time: Optional[datetime] = None

    @property
    def pct_complete(self) -> float:
        return self.step / self.total_steps if self.total_steps > 0 else 0.0

    @property
    def eta_secs(self) -> float:
        if self.step == 0 or self.elapsed_secs == 0:
            return 0.0
        rate = self.step / self.elapsed_secs
        return (self.total_steps - self.step) / rate


# Callback type: called periodically during backtest iteration
ProgressCallback = Callable[[BacktestProgress], None]


@dataclass
class BacktestResult:
    """Performance metrics from a single backtest run."""

    signal_name: str
    scenario_name: str
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    equity_curve: list[float]
    final_equity: float


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """Orchestrates a single signal x data backtest through the real DirectionalBot pipeline."""

    def __init__(
        self,
        signal_source: SignalSource,
        snapshots: list[OrderbookSnapshot],
        instrument_id: str = "EXT-001",
        scenario_name: str = "real_data",
        initial_balance: float = 10_000,
        exchange_id: ExchangeId = ExchangeId.POLYMARKET,
        on_progress: Optional[ProgressCallback] = None,
        progress_interval: int = 5000,
        subsample: int = 1,
    ):
        self.signal_source = signal_source
        self.initial_balance = initial_balance
        self.exchange_id = exchange_id
        self._snapshots = snapshots
        self._instrument_id = instrument_id
        self._scenario_name = scenario_name
        self._on_progress = on_progress
        self._progress_interval = progress_interval
        self._subsample = subsample

    async def run(self) -> BacktestResult:
        """Run the backtest and return metrics."""
        instrument = Instrument(
            instrument_id=self._instrument_id,
            exchange=self.exchange_id,
            instrument_type=InstrumentType.BINARY_OUTCOME,
            name=f"Backtest {self._scenario_name}",
            market_id=f"bt-{self._scenario_name}",
            outcome="YES",
            active=True,
        )

        # Temp SQLite DB
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = tmp.name
        tmp.close()

        try:
            return await self._execute(db_path, self._snapshots, instrument)
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
    ) -> BacktestResult:
        storage = SQLiteStorage(db_path)
        storage.initialize()

        exchange_str = self.exchange_id.value
        client = BacktestExchangeClient(snapshots, instrument, self.initial_balance, self.exchange_id)

        # Seed storage balance
        storage.update_balance(exchange_str, "", self.initial_balance)

        # Relaxed risk config for backtest
        risk_config = RiskConfig(
            max_wallet_exposure_pct=0.80,
            max_per_agent_exposure_pct=0.60,
            max_per_market_exposure_pct=0.60,
            min_trade_value_usd=5.0,
            max_trade_value_usd=2000.0,
            max_daily_drawdown_pct=0.20,
            max_total_drawdown_pct=0.30,
        )

        risk = RiskCoordinator(storage, risk_config)
        risk.register_account(self.exchange_id, "")

        exit_config = ExitConfig(
            take_profit_pct=0.08,
            stop_loss_pct=0.30,
            max_hold_minutes=120,
            trailing_stop_activation_pct=0.03,
            trailing_stop_distance_pct=0.015,
        )

        sizer = FixedFractionSizer(fraction=0.05, min_usd=10.0, max_usd=200.0)
        agent_id = f"bt-{self.signal_source.name}-{self._scenario_name}"

        paper_client = PaperClient(client, slippage_pct=0.001)

        bot = DirectionalBot(
            agent_id=agent_id,
            client=paper_client,
            signal_source=self.signal_source,
            sizer=sizer,
            risk=risk,
            exit_config=exit_config,
            max_positions=5,
        )

        # Disable reconciliation — backtest exchange state is always consistent
        async def _noop():
            pass
        bot._reconcile_positions = _noop

        # Pre-filter: deduplicate consecutive identical snapshots to skip
        # empty carry-forward windows (common with sparse trade data)
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

        logger.info(
            "  Running %d active steps (skipped %d unchanged snapshots)",
            len(active_indices), len(snapshots) - len(active_indices),
        )

        await bot.start()
        equity_curve = [self.initial_balance]
        total_steps = len(active_indices) - 1
        t_start = time.monotonic()

        equity = self.initial_balance
        cached_open_count = 0
        cached_closed_count = 0

        # Incremental closed PnL tracking: avoid re-summing all closed
        # positions every MTM tick. Only sum newly closed positions.
        cumulative_closed_pnl = 0.0
        last_closed_count = 0
        last_balance_equity = self.initial_balance
        balance_write_interval = max(1, min(100, total_steps // 100))
        balance_writes_since_last = 0

        for loop_idx in range(total_steps):
            target_step = active_indices[loop_idx + 1]

            # Jump client to the target snapshot
            client._step = target_step

            try:
                await bot._iteration()
            except Exception as e:
                logger.debug("Iteration %d error: %s", loop_idx, e)

            mid = client.current_snapshot.midpoint or 0.5

            # Query positions every step for accurate equity curve
            open_pos = storage.get_agent_positions(agent_id, "open")
            closed_pos = storage.get_agent_positions(agent_id, "closed")

            # Incrementally sum PnL from only newly closed positions
            current_closed_count = len(closed_pos)
            if current_closed_count > last_closed_count:
                cumulative_closed_pnl += sum(
                    p.get("pnl", 0) or 0
                    for p in closed_pos[last_closed_count:]
                )
                last_closed_count = current_closed_count

            cached_open_count = len(open_pos)
            cached_closed_count = current_closed_count

            open_unrealized = sum(
                (mid - p["entry_price"]) * p["size"] if p["side"] == "BUY"
                else (p["entry_price"] - mid) * p["size"]
                for p in open_pos
            )
            open_cost = sum(p["size"] * p["entry_price"] for p in open_pos)

            equity = self.initial_balance + cumulative_closed_pnl + open_unrealized
            available = self.initial_balance + cumulative_closed_pnl - open_cost

            client._equity = equity
            client._available = max(0.0, available)

            # Throttle storage balance writes (not position reads)
            is_last_step = loop_idx == total_steps - 1
            balance_writes_since_last += 1
            if is_last_step or balance_writes_since_last >= balance_write_interval:
                equity_drift = (
                    abs(equity - last_balance_equity) / last_balance_equity
                    if last_balance_equity > 0 else 0.0
                )
                if is_last_step or equity_drift > 0.001:
                    storage.update_balance(exchange_str, "", equity)
                    last_balance_equity = equity
                balance_writes_since_last = 0

            equity_curve.append(equity)

            # Progress callback
            if self._on_progress and (loop_idx + 1) % self._progress_interval == 0:
                self._on_progress(BacktestProgress(
                    step=loop_idx + 1,
                    total_steps=total_steps,
                    elapsed_secs=time.monotonic() - t_start,
                    equity=equity,
                    pnl=equity - self.initial_balance,
                    open_positions=cached_open_count,
                    closed_trades=cached_closed_count,
                    mid_price=mid,
                    market_time=client.current_snapshot.timestamp,
                ))

        # Final balance update
        storage.update_balance(exchange_str, "", equity_curve[-1])
        await bot.stop()

        # Compute data duration from snapshot timestamps for Sharpe
        first_ts = snapshots[0].timestamp
        last_ts = snapshots[-1].timestamp
        total_duration_secs = (last_ts - first_ts).total_seconds()

        result = self._compute_metrics(storage, agent_id, equity_curve, total_duration_secs)
        storage.close()
        return result

    def _compute_metrics(
        self,
        storage: SQLiteStorage,
        agent_id: str,
        equity_curve: list[float],
        total_duration_secs: float,
    ) -> BacktestResult:
        closed = storage.get_agent_positions(agent_id, "closed")

        total_trades = len(closed)
        winning = sum(1 for p in closed if (p.get("pnl") or 0) > 0)
        losing = sum(1 for p in closed if (p.get("pnl") or 0) < 0)
        closed_pnl = sum(p.get("pnl", 0) or 0 for p in closed)

        win_rate = winning / total_trades if total_trades > 0 else 0.0
        avg_trade_pnl = closed_pnl / total_trades if total_trades > 0 else 0.0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        sharpe = compute_sharpe(equity_curve, total_duration_secs)

        final_equity = equity_curve[-1] if equity_curve else self.initial_balance

        return BacktestResult(
            signal_name=self.signal_source.name,
            scenario_name=self._scenario_name,
            total_pnl=final_equity - self.initial_balance,
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            avg_trade_pnl=avg_trade_pnl,
            equity_curve=equity_curve,
            final_equity=final_equity,
        )
