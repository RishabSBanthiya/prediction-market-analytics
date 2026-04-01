"""
Directional trading bot.

Loop: generate signals -> size positions -> check risk -> execute
Works with any ExchangeClient through the unified interface.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import Side, SignalDirection, ExchangeId, Environment
from ..core.models import Signal, PositionState, ExchangePosition
from ..core.config import Config
from ..core.errors import RiskLimitError, OmniTradeError
from ..exchanges.base import ExchangeClient
from ..components.signals import SignalSource
from ..core.models import OrderRequest
from ..components.trading import PositionSizer, direction_to_side, ExitMonitor, ExitConfig
from ..risk.coordinator import RiskCoordinator

logger = logging.getLogger(__name__)


class DirectionalBot:
    """
    Directional trading bot.

    Composition-based: signal source, sizer, executor, exit monitor
    are all pluggable. The bot never imports platform-specific code.
    """

    def __init__(
        self,
        agent_id: str,
        client: ExchangeClient,
        signal_source: SignalSource,
        sizer: PositionSizer,
        risk: RiskCoordinator,
        exit_config: Optional[ExitConfig] = None,
        max_positions: int = 10,
        min_price: float = 0.05,
        max_price: float = 0.95,
    ):
        self.agent_id = agent_id
        self.client = client
        self.signal_source = signal_source
        self.sizer = sizer
        self.risk = risk
        self.max_positions = max_positions
        self.min_price = min_price
        self.max_price = max_price
        self._running = False

        self.exit_monitor = ExitMonitor(exit_config)
        self._iteration_count = 0
        self._has_open_positions = False

    async def start(self) -> None:
        """Initialize and register with risk coordinator."""
        if not self.client.is_connected:
            await self.client.connect()
        self.risk.startup(self.agent_id, "directional", self.client.exchange_id)
        self._restore_exit_states()
        logger.info(f"DirectionalBot '{self.agent_id}' started on {self.client.exchange_id.value}")

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self.risk.shutdown(self.agent_id)
        logger.info(f"DirectionalBot '{self.agent_id}' stopped")

    async def run(self, interval_seconds: float = 30.0) -> None:
        """Main loop."""
        self._running = True
        await self.start()

        try:
            while self._running:
                try:
                    await self._iteration()
                except OmniTradeError as e:
                    logger.error(f"Trading error: {e}")
                    self.risk.record_failure()
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    self.risk.record_failure()

                self.risk.heartbeat(self.agent_id)
                await asyncio.sleep(interval_seconds)
        finally:
            await self.stop()

    async def _iteration(self) -> None:
        """Single trading iteration."""
        self._iteration_count += 1
        is_verbose_tick = self._iteration_count % 10 == 1  # Log details every 10 iterations

        # Update balance for drawdown tracking
        balance = await self.client.get_balance()
        self.risk.update_equity(balance.total_equity)

        if is_verbose_tick:
            logger.info(
                "[iter %d] balance=$%.2f (avail=$%.2f)",
                self._iteration_count, balance.total_equity, balance.available_balance,
            )

        # Monitor existing positions for exits
        await self._monitor_positions()

        # Reconcile with exchange every 10 iterations (~5 min at 30s interval)
        if self._iteration_count % 10 == 0:
            await self._reconcile_positions()

        # Generate new signals
        signals = await self.signal_source.generate(self.client)

        if is_verbose_tick:
            instruments = await self.client.get_instruments(active_only=True)
            logger.info(
                "[iter %d] scanning %d instruments via %s -> %d signals",
                self._iteration_count, len(instruments),
                self.signal_source.name, len(signals),
            )

        if not signals:
            return

        # Check position limit
        current_positions = self.risk.storage.get_agent_positions(self.agent_id, "open")
        if len(current_positions) >= self.max_positions:
            logger.info(
                "[iter %d] at position limit (%d/%d), skipping new signals",
                self._iteration_count, len(current_positions), self.max_positions,
            )
            return

        # Process best signal (try next if top signal is filtered out)
        signals.sort(key=lambda s: s.score, reverse=True)

        if is_verbose_tick and signals:
            top = signals[0]
            logger.info(
                "[iter %d] top signal: %s %s @ %.4f (score=%.1f)",
                self._iteration_count, top.direction.value, top.instrument_id,
                top.price, top.score,
            )

        for signal in signals:
            if not signal.is_actionable:
                continue
            traded = await self._process_signal(signal)
            if traded:
                break  # One trade per iteration

    async def _process_signal(self, signal: Signal) -> bool:
        """Process a single signal: size -> reserve -> execute. Returns True if trade was placed."""
        # Price filter
        if signal.price < self.min_price or signal.price > self.max_price:
            logger.info(
                "SKIP %s: price %.4f outside [%.2f, %.2f]",
                signal.instrument_id, signal.price, self.min_price, self.max_price,
            )
            return False

        # Get current balance for sizing
        balance = await self.client.get_balance()
        available = balance.available_balance

        # Size the position
        size_usd = self.sizer.calculate_size(signal, available, signal.price)
        if size_usd <= 0:
            logger.info(
                "SKIP %s: sizer returned $0 (avail=$%.2f, price=%.4f)",
                signal.instrument_id, available, signal.price,
            )
            return False

        # Map direction to side
        side = direction_to_side(signal.direction)

        logger.info(
            "CONSIDERING %s %s %s: $%.2f @ %.4f (score=%.1f)",
            side.value, signal.direction.value, signal.instrument_id,
            size_usd, signal.price, signal.score,
        )

        # Reserve capital
        try:
            reservation_id = self.risk.atomic_reserve(
                agent_id=self.agent_id,
                exchange=self.client.exchange_id,
                instrument_id=signal.instrument_id,
                amount_usd=size_usd,
            )
        except (RiskLimitError, Exception) as e:
            logger.info("BLOCKED %s: risk check -> %s", signal.instrument_id, e)
            return False

        # Execute
        try:
            shares = size_usd / signal.price if signal.price > 0 else 0
            if shares <= 0:
                self.risk.release_reservation(reservation_id)
                return False
            result = await self.client.place_order(OrderRequest(
                instrument_id=signal.instrument_id,
                side=side,
                size=shares,
                price=signal.price,
            ))

            if result.success and result.filled_size > 0:
                # Confirm with risk coordinator
                position_id = self.risk.confirm_execution(
                    reservation_id=reservation_id,
                    agent_id=self.agent_id,
                    exchange=self.client.exchange_id,
                    instrument_id=signal.instrument_id,
                    side=side.value,
                    size=result.filled_size,
                    price=result.filled_price,
                    order_id=result.order_id,
                    fees=result.fees,
                )

                # Register for exit monitoring
                self.exit_monitor.register(
                    signal.instrument_id,
                    PositionState(
                        instrument_id=signal.instrument_id,
                        entry_price=result.filled_price,
                        entry_time=datetime.now(timezone.utc),
                        size=result.filled_size,
                        side=side,
                    ),
                )

                self._has_open_positions = True
                logger.info(
                    "OPENED %s %.4f @ $%.4f on %s (score=%.1f, order=%s)",
                    side.value, result.filled_size, result.filled_price,
                    signal.instrument_id, signal.score, result.order_id,
                )
                return True
            else:
                self.risk.release_reservation(reservation_id)
                if not result.is_rejection:
                    self.risk.record_failure()
                logger.info(
                    "FAILED %s: %s",
                    signal.instrument_id, result.error_message or "no fill",
                )
                return False

        except Exception as e:
            self.risk.release_reservation(reservation_id)
            self.risk.record_failure()
            raise

    def _restore_exit_states(self) -> None:
        """Restore exit monitor state from storage after restart."""
        positions = self.risk.storage.get_agent_positions(self.agent_id, "open")
        if positions:
            self._has_open_positions = True
        for pos in positions:
            instrument_id = pos["instrument_id"]
            if self.exit_monitor.get_state(instrument_id) is not None:
                continue
            entry_price = pos["entry_price"]
            state = PositionState(
                instrument_id=instrument_id,
                entry_price=entry_price,
                entry_time=datetime.fromisoformat(pos["opened_at"]),
                size=pos["size"],
                side=Side.BUY if pos["side"] == "BUY" else Side.SELL,
                peak_price=pos.get("peak_price") or entry_price,
                trough_price=pos.get("trough_price") or entry_price,
                trailing_stop_activated=bool(pos.get("trailing_stop_activated")),
                trailing_stop_level=pos.get("trailing_stop_level") or 0.0,
            )
            self.exit_monitor.register(instrument_id, state)
            logger.info(f"Restored exit state for {instrument_id} (position {pos['position_id']})")

    async def _monitor_positions(self) -> None:
        """Check exit conditions on all open positions."""
        # Fast path: skip SQLite query when we know there are no positions
        if not self._has_open_positions and self._iteration_count % 10 != 0:
            return

        positions = self.risk.storage.get_agent_positions(self.agent_id, "open")
        now = datetime.now(timezone.utc)

        if not positions:
            self._has_open_positions = False
            return

        if positions and self._iteration_count % 10 == 1:
            pos_summary = ", ".join(
                f"{p['instrument_id'][:20]} {p['side']} {p['size']:.2f}@{p['entry_price']:.4f}"
                for p in positions[:5]
            )
            if len(positions) > 5:
                pos_summary += f" (+{len(positions) - 5} more)"
            logger.info("[positions] %d open: %s", len(positions), pos_summary)

        for pos in positions:
            instrument_id = pos["instrument_id"]
            state = self.exit_monitor.get_state(instrument_id)
            if state is None:
                continue

            # Get current price
            mid = await self.client.get_midpoint(instrument_id)
            if mid is None:
                continue

            # Check exit conditions
            exit_result = self.exit_monitor.check(state, mid, now)

            # Persist exit state every cycle
            self.risk.storage.update_position_exit_state(
                pos["position_id"],
                current_price=mid,
                peak_price=state.peak_price,
                trough_price=state.trough_price,
                trailing_stop_activated=state.trailing_stop_activated,
                trailing_stop_level=state.trailing_stop_level,
            )

            if exit_result is None:
                continue

            reason, exit_price, description = exit_result
            pnl_pct = (mid - state.entry_price) / state.entry_price * 100 if state.entry_price > 0 else 0
            logger.info(
                "EXIT SIGNAL [%s] %s: %s (entry=%.4f, now=%.4f, pnl=%.1f%%)",
                reason.value, instrument_id, description,
                state.entry_price, mid, pnl_pct,
            )

            # Execute exit
            exit_side = Side.SELL if state.side == Side.BUY else Side.BUY
            result = await self.client.place_order(OrderRequest(
                instrument_id=instrument_id,
                side=exit_side,
                size=state.size,
                price=exit_price,
            ))

            if result.success:
                pnl = (result.filled_price - state.entry_price) * state.size
                if state.side == Side.SELL:
                    pnl = -pnl
                self.risk.storage.close_position(
                    pos["position_id"], result.filled_price, reason.value
                )
                self.exit_monitor.unregister(instrument_id)
                logger.info(
                    "CLOSED %s: %s @ %.4f -> %.4f (PnL=$%.2f)",
                    instrument_id, reason.value, state.entry_price,
                    result.filled_price, pnl,
                )

        # Update flag: if we closed all positions this cycle, mark as empty
        remaining = self.risk.storage.get_agent_positions(self.agent_id, "open")
        if not remaining:
            self._has_open_positions = False

    async def _reconcile_positions(self) -> None:
        """Compare DB positions with exchange positions and log discrepancies."""
        try:
            exchange_positions = await self.client.get_positions()
        except Exception as e:
            logger.warning(f"Reconciliation skipped, get_positions failed: {e}")
            return

        db_positions = self.risk.storage.get_agent_positions(self.agent_id, "open")

        # Index exchange positions by instrument_id
        exchange_by_id: dict[str, ExchangePosition] = {
            ep.instrument_id: ep for ep in exchange_positions
        }
        db_seen_ids: set[str] = set()

        for pos in db_positions:
            iid = pos["instrument_id"]
            db_seen_ids.add(iid)
            ep = exchange_by_id.get(iid)
            if ep is None:
                logger.debug(
                    "RECONCILIATION: DB position %s (id=%s) not found on exchange",
                    iid, pos['position_id'],
                )
                continue
            # Size mismatch
            if abs(ep.size - pos["size"]) > 0.001:
                logger.warning(
                    f"RECONCILIATION: {iid} size mismatch — "
                    f"DB={pos['size']:.4f}, exchange={ep.size:.4f}"
                )
            # Side mismatch
            if ep.side.value != pos["side"]:
                logger.warning(
                    f"RECONCILIATION: {iid} side mismatch — "
                    f"DB={pos['side']}, exchange={ep.side.value}"
                )
            # Update current_price from exchange
            if ep.current_price and ep.current_price > 0:
                self.risk.storage.update_position_price(pos["position_id"], ep.current_price)

        # Exchange positions not tracked in DB
        for iid, ep in exchange_by_id.items():
            if iid not in db_seen_ids:
                logger.warning(
                    f"RECONCILIATION: exchange position {iid} "
                    f"({ep.side.value} {ep.size:.4f}) not tracked in DB"
                )
