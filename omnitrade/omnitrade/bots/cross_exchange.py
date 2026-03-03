"""
Cross-exchange strategy bot.

Executes multi-leg strategies across different exchanges simultaneously.
Key use case: long binary outcome on Polymarket/Kalshi + hedge with perps on Hyperliquid.

The bot holds multiple ExchangeClient instances and coordinates execution
across them. Legs are executed sequentially with rollback on failure.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import Side, SignalDirection, ExchangeId, Environment
from ..core.models import (
    MultiLegSignal, LegResult, MultiLegResult,
    PositionState, ExchangePosition,
)
from ..core.errors import RiskLimitError, OmniTradeError
from ..exchanges.base import ExchangeClient
from ..components.executors import Executor, DryRunExecutor
from ..components.exit_strategies import ExitMonitor, ExitConfig
from ..risk.coordinator import RiskCoordinator

logger = logging.getLogger(__name__)


class CrossExchangeSignalSource:
    """Abstract base for cross-exchange signal sources."""

    async def generate(
        self, clients: dict[ExchangeId, ExchangeClient]
    ) -> list[MultiLegSignal]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "cross_exchange"


class CrossExchangeBot:
    """
    Cross-exchange strategy bot.

    Holds multiple ExchangeClient instances and executes multi-leg
    strategies across them. Each leg is sized, risk-checked, and
    executed independently with rollback support.

    Example strategies:
    - Long "BTC Up" on Polymarket + Short BTC-PERP on Hyperliquid
    - Long event on Kalshi + Long opposite event on Polymarket (arb)
    - Long binary at 0.95 on Polymarket + sell perp as delta hedge
    """

    def __init__(
        self,
        agent_id: str,
        clients: dict[ExchangeId, ExchangeClient],
        signal_source: CrossExchangeSignalSource,
        executors: dict[ExchangeId, Executor],
        risk: RiskCoordinator,
        exit_config: Optional[ExitConfig] = None,
        environment: Environment = Environment.PAPER,
        base_size_usd: float = 50.0,
        max_strategies: int = 5,
    ):
        self.agent_id = agent_id
        self.clients = clients
        self.signal_source = signal_source
        self.risk = risk
        self.base_size_usd = base_size_usd
        self.max_strategies = max_strategies
        self.environment = environment
        self._running = False

        # Per-exchange executors (paper mode wraps all with DryRun)
        if environment == Environment.PAPER:
            self.executors = {
                ex: DryRunExecutor() for ex in clients
            }
            logger.info("Paper mode: all executors wrapped with DryRunExecutor")
        else:
            self.executors = executors

        self.exit_monitor = ExitMonitor(exit_config)

        # Track active multi-leg positions: strategy_key -> leg details
        self._active_strategies: dict[str, MultiLegSignal] = {}
        self._iteration_count = 0
        # Restored positions from DB that can't be added to _active_strategies
        # key: "restored-{position_id}", value: (sub_agent, position dict)
        self._restored_positions: dict[str, tuple[str, dict]] = {}

    async def start(self) -> None:
        """Connect all clients and register with risk."""
        for exchange_id, client in self.clients.items():
            if not client.is_connected:
                await client.connect()
        # Register agent on each exchange
        for exchange_id in self.clients:
            self.risk.startup(
                f"{self.agent_id}-{exchange_id.value}",
                "cross_exchange",
                exchange_id,
            )
        self._restore_exit_states()
        logger.info(
            f"CrossExchangeBot '{self.agent_id}' started on "
            f"{[e.value for e in self.clients]}"
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        for exchange_id in self.clients:
            self.risk.shutdown(f"{self.agent_id}-{exchange_id.value}")
        logger.info(f"CrossExchangeBot '{self.agent_id}' stopped")

    async def run(self, interval_seconds: float = 30.0) -> None:
        """Main loop."""
        self._running = True
        await self.start()

        try:
            while self._running:
                try:
                    await self._iteration()
                except OmniTradeError as e:
                    logger.error(f"Strategy error: {e}")
                    self.risk.record_failure()
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    self.risk.record_failure()

                # Heartbeat all sub-agents
                for exchange_id in self.clients:
                    self.risk.heartbeat(f"{self.agent_id}-{exchange_id.value}")
                await asyncio.sleep(interval_seconds)
        finally:
            await self.stop()

    async def _iteration(self) -> None:
        """Single strategy iteration."""
        self._iteration_count += 1

        # Update balances across all exchanges
        for _, client in self.clients.items():
            balance = await client.get_balance()
            self.risk.update_equity(balance.total_equity)

        # Monitor existing strategies for exits
        await self._monitor_strategies()

        # Monitor individually-restored positions
        await self._monitor_restored_positions()

        # Reconcile with exchanges every 10 iterations
        if self._iteration_count % 10 == 0:
            await self._reconcile_positions()

        # Check strategy limit
        if len(self._active_strategies) >= self.max_strategies:
            logger.debug(f"At strategy limit ({self.max_strategies})")
            return

        # Generate cross-exchange signals
        signals = await self.signal_source.generate(self.clients)
        if not signals:
            return

        # Process best signal
        signals.sort(key=lambda s: s.score, reverse=True)
        for signal in signals:
            if not signal.is_actionable:
                continue
            # Check all required exchanges are connected
            if not signal.exchanges_involved.issubset(self.clients.keys()):
                missing = signal.exchanges_involved - self.clients.keys()
                logger.warning(f"Missing clients for: {missing}")
                continue
            await self._execute_strategy(signal)
            break  # One strategy per iteration

    async def _execute_strategy(self, signal: MultiLegSignal) -> None:
        """
        Execute a multi-leg strategy.

        Legs are executed sequentially. If any leg fails after others
        succeeded, the bot attempts to roll back completed legs.
        """
        strategy_key = f"{signal.strategy_type}-{datetime.now(timezone.utc).timestamp():.0f}"
        leg_results: list[LegResult] = []
        reservations: list[tuple[str, ExchangeId]] = []  # (reservation_id, exchange)

        logger.info(
            f"Executing {signal.strategy_type} strategy ({len(signal.legs)} legs, "
            f"score={signal.score:.1f}, edge={signal.edge_bps:.0f}bps)"
        )

        for i, leg in enumerate(signal.legs):
            sub_agent = f"{self.agent_id}-{leg.exchange.value}"
            size_usd = self.base_size_usd * leg.weight

            # Reserve capital for this leg
            try:
                reservation_id = self.risk.atomic_reserve(
                    agent_id=sub_agent,
                    exchange=leg.exchange,
                    instrument_id=leg.instrument_id,
                    amount_usd=size_usd,
                )
                reservations.append((reservation_id, leg.exchange))
            except (RiskLimitError, Exception) as e:
                logger.warning(f"Leg {i} risk check failed: {e}")
                # Roll back previous reservations
                for rid, _ in reservations:
                    self.risk.release_reservation(rid)
                return

            # Execute leg
            executor = self.executors.get(leg.exchange)
            if executor is None:
                logger.error(f"No executor for {leg.exchange}")
                for rid, _ in reservations:
                    self.risk.release_reservation(rid)
                return

            client = self.clients[leg.exchange]
            side = Side.BUY if leg.direction == SignalDirection.LONG else Side.SELL

            result = await executor.execute(
                client=client,
                instrument_id=leg.instrument_id,
                side=side,
                size_usd=size_usd,
                price=leg.price,
            )

            leg_results.append(LegResult(
                leg=leg,
                order_result=result,
                reservation_id=reservation_id,
            ))

            if result.success and result.filled_size > 0:
                # Confirm with risk
                self.risk.confirm_execution(
                    reservation_id=reservation_id,
                    agent_id=sub_agent,
                    exchange=leg.exchange,
                    instrument_id=leg.instrument_id,
                    side=side.value,
                    size=result.filled_size,
                    price=result.filled_price,
                    order_id=result.order_id,
                    fees=result.fees,
                )
                logger.info(
                    f"  Leg {i}: {side.value} {result.filled_size:.4f} @ "
                    f"${result.filled_price:.4f} on {leg.exchange.value}/{leg.instrument_id}"
                )
            else:
                # Leg failed — roll back all previous legs
                logger.warning(
                    f"  Leg {i} FAILED on {leg.exchange.value}: {result.error_message}"
                )
                await self._rollback(leg_results, reservations)
                self.risk.record_failure()
                return

        # All legs succeeded
        multi_result = MultiLegResult(
            leg_results=leg_results,
            strategy_type=signal.strategy_type,
        )

        self._active_strategies[strategy_key] = signal

        # Register primary leg for exit monitoring
        primary = signal.legs[0]
        primary_result = leg_results[0].order_result
        self.exit_monitor.register(
            strategy_key,
            PositionState(
                instrument_id=primary.instrument_id,
                entry_price=primary_result.filled_price,
                entry_time=datetime.now(timezone.utc),
                size=primary_result.filled_size,
                side=Side.BUY if primary.direction == SignalDirection.LONG else Side.SELL,
            ),
        )

        logger.info(
            f"Strategy {strategy_key} OPENED: {len(leg_results)} legs, "
            f"total cost ${multi_result.total_cost:.2f}"
        )

    async def _rollback(
        self,
        leg_results: list[LegResult],
        reservations: list[tuple[str, ExchangeId]],
    ) -> None:
        """Roll back completed legs by placing opposing orders."""
        logger.warning("Rolling back completed legs...")

        for lr in leg_results:
            if not lr.success:
                # Release reservation for failed leg
                self.risk.release_reservation(lr.reservation_id)
                continue

            # Place opposing order to unwind
            leg = lr.leg
            client = self.clients[leg.exchange]
            opposite_side = Side.SELL if leg.direction == SignalDirection.LONG else Side.BUY
            executor = self.executors.get(leg.exchange)

            if executor:
                rollback_result = await executor.execute(
                    client=client,
                    instrument_id=leg.instrument_id,
                    side=opposite_side,
                    size_usd=lr.order_result.filled_size * lr.order_result.filled_price,
                    price=lr.order_result.filled_price,
                )
                if rollback_result.success:
                    logger.info(f"  Rolled back {leg.exchange.value}/{leg.instrument_id}")
                else:
                    logger.error(
                        f"  ROLLBACK FAILED for {leg.exchange.value}/{leg.instrument_id}: "
                        f"{rollback_result.error_message}"
                    )

    async def _monitor_strategies(self) -> None:
        """Monitor active strategies for exit conditions."""
        now = datetime.now(timezone.utc)
        to_close = []

        for strategy_key, signal in list(self._active_strategies.items()):
            state = self.exit_monitor.get_state(strategy_key)
            if state is None:
                continue

            # Check primary leg price
            primary = signal.legs[0]
            client = self.clients[primary.exchange]
            mid = await client.get_midpoint(primary.instrument_id)
            if mid is None:
                continue

            exit_result = self.exit_monitor.check(state, mid, now)

            # Persist exit state for all sub-agent positions
            for leg in signal.legs:
                sub_agent = f"{self.agent_id}-{leg.exchange.value}"
                positions = self.risk.storage.get_agent_positions(sub_agent, "open")
                for pos in positions:
                    if pos["instrument_id"] == leg.instrument_id:
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

            reason, _, description = exit_result
            logger.info(f"STRATEGY EXIT [{strategy_key}] {reason.value}: {description}")
            to_close.append((strategy_key, signal, reason))

        # Close strategies (unwind all legs)
        for strategy_key, signal, reason in to_close:
            await self._close_strategy(strategy_key, signal, reason.value)

    async def _close_strategy(
        self, strategy_key: str, signal: MultiLegSignal, exit_reason: str
    ) -> None:
        """Close all legs of a strategy."""
        for leg in signal.legs:
            sub_agent = f"{self.agent_id}-{leg.exchange.value}"
            client = self.clients[leg.exchange]
            opposite_side = Side.SELL if leg.direction == SignalDirection.LONG else Side.BUY
            executor = self.executors.get(leg.exchange)

            if executor:
                # Get current price for sizing
                mid = await client.get_midpoint(leg.instrument_id)
                if mid and mid > 0:
                    positions = self.risk.storage.get_agent_positions(sub_agent, "open")
                    for pos in positions:
                        if pos["instrument_id"] == leg.instrument_id:
                            result = await executor.execute(
                                client=client,
                                instrument_id=leg.instrument_id,
                                side=opposite_side,
                                size_usd=pos["size"] * mid,
                                price=mid,
                            )
                            if result.success:
                                self.risk.storage.close_position(
                                    pos["position_id"], result.filled_price, exit_reason
                                )

        self.exit_monitor.unregister(strategy_key)
        del self._active_strategies[strategy_key]
        logger.info(f"Strategy {strategy_key} CLOSED ({exit_reason})")

    def _restore_exit_states(self) -> None:
        """Restore exit monitor state from storage after restart."""
        for exchange_id in self.clients:
            sub_agent = f"{self.agent_id}-{exchange_id.value}"
            positions = self.risk.storage.get_agent_positions(sub_agent, "open")
            for pos in positions:
                monitor_key = f"restored-{pos['position_id']}"
                if self.exit_monitor.get_state(monitor_key) is not None:
                    continue
                entry_price = pos["entry_price"]
                state = PositionState(
                    instrument_id=pos["instrument_id"],
                    entry_price=entry_price,
                    entry_time=datetime.fromisoformat(pos["opened_at"]),
                    size=pos["size"],
                    side=Side.BUY if pos["side"] == "BUY" else Side.SELL,
                    peak_price=pos.get("peak_price") or entry_price,
                    trough_price=pos.get("trough_price") or entry_price,
                    trailing_stop_activated=bool(pos.get("trailing_stop_activated")),
                    trailing_stop_level=pos.get("trailing_stop_level") or 0.0,
                )
                self.exit_monitor.register(monitor_key, state)
                self._restored_positions[monitor_key] = (sub_agent, pos)
                logger.info(
                    f"Restored exit state for {pos['instrument_id']} "
                    f"(position {pos['position_id']}, sub-agent {sub_agent})"
                )

    async def _monitor_restored_positions(self) -> None:
        """Monitor positions restored from DB that aren't in _active_strategies."""
        now = datetime.now(timezone.utc)
        to_close: list[tuple[str, str, dict, str]] = []  # (key, sub_agent, pos, reason)

        for monitor_key, (sub_agent, pos) in list(self._restored_positions.items()):
            state = self.exit_monitor.get_state(monitor_key)
            if state is None:
                continue

            instrument_id = pos["instrument_id"]
            # Find the right client for this sub-agent
            exchange_value = sub_agent.split("-")[-1]
            client = None
            for eid, c in self.clients.items():
                if eid.value == exchange_value:
                    client = c
                    break
            if client is None:
                continue

            mid = await client.get_midpoint(instrument_id)
            if mid is None:
                continue

            exit_result = self.exit_monitor.check(state, mid, now)

            # Persist exit state
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

            reason, _, description = exit_result
            logger.info(f"RESTORED EXIT [{monitor_key}] {reason.value}: {description}")
            to_close.append((monitor_key, sub_agent, pos, reason.value))

        for monitor_key, sub_agent, pos, reason_val in to_close:
            instrument_id = pos["instrument_id"]
            exchange_value = sub_agent.split("-")[-1]
            client = None
            executor = None
            for eid, c in self.clients.items():
                if eid.value == exchange_value:
                    client = c
                    executor = self.executors.get(eid)
                    break

            if client and executor:
                exit_side = Side.SELL if pos["side"] == "BUY" else Side.BUY
                mid = await client.get_midpoint(instrument_id)
                if mid and mid > 0:
                    result = await executor.execute(
                        client=client,
                        instrument_id=instrument_id,
                        side=exit_side,
                        size_usd=pos["size"] * mid,
                        price=mid,
                    )
                    if result.success:
                        self.risk.storage.close_position(
                            pos["position_id"], result.filled_price, reason_val
                        )

            self.exit_monitor.unregister(monitor_key)
            del self._restored_positions[monitor_key]

    async def _reconcile_positions(self) -> None:
        """Compare DB positions with exchange positions and log discrepancies."""
        for exchange_id, client in self.clients.items():
            sub_agent = f"{self.agent_id}-{exchange_id.value}"
            try:
                exchange_positions = await client.get_positions()
            except Exception as e:
                logger.warning(
                    f"Reconciliation skipped for {exchange_id.value}: {e}"
                )
                continue

            db_positions = self.risk.storage.get_agent_positions(sub_agent, "open")
            exchange_by_id: dict[str, ExchangePosition] = {
                ep.instrument_id: ep for ep in exchange_positions
            }
            db_seen_ids: set[str] = set()

            for pos in db_positions:
                iid = pos["instrument_id"]
                db_seen_ids.add(iid)
                ep = exchange_by_id.get(iid)
                if ep is None:
                    logger.warning(
                        f"RECONCILIATION: DB position {iid} (id={pos['position_id']}) "
                        f"not found on {exchange_id.value}"
                    )
                    continue
                if abs(ep.size - pos["size"]) > 0.001:
                    logger.warning(
                        f"RECONCILIATION: {iid} size mismatch on {exchange_id.value} — "
                        f"DB={pos['size']:.4f}, exchange={ep.size:.4f}"
                    )
                if ep.side.value != pos["side"]:
                    logger.warning(
                        f"RECONCILIATION: {iid} side mismatch on {exchange_id.value} — "
                        f"DB={pos['side']}, exchange={ep.side.value}"
                    )
                if ep.current_price and ep.current_price > 0:
                    self.risk.storage.update_position_price(
                        pos["position_id"], ep.current_price
                    )

            for iid, ep in exchange_by_id.items():
                if iid not in db_seen_ids:
                    logger.warning(
                        f"RECONCILIATION: exchange position {iid} on {exchange_id.value} "
                        f"({ep.side.value} {ep.size:.4f}) not tracked in DB"
                    )
