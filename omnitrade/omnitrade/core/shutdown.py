"""
Graceful shutdown and startup recovery for trading bots.

Handles SIGINT/SIGTERM to cleanly wind down bot operations:
- Cancel open orders on the exchange
- Release risk reservations to prevent capital leaks
- Persist bot state for recovery on next startup
- Coordinate shutdown timeout so we don't hang indefinitely

Startup recovery reconciles state after an unclean exit:
- Detect and cancel orphaned orders left on exchanges
- Clean up leaked reservations and stale agent records
- Log discrepancies between DB and exchange state
"""

import asyncio
import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from ..exchanges.base import ExchangeClient
from ..risk.coordinator import RiskCoordinator
from ..storage.base import StorageBackend

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of the graceful shutdown sequence."""
    RUNNING = "running"
    STOPPING = "stopping"
    CANCELLING_ORDERS = "cancelling_orders"
    RELEASING_RESERVATIONS = "releasing_reservations"
    PERSISTING_STATE = "persisting_state"
    COMPLETE = "complete"


@dataclass
class ShutdownState:
    """Tracks progress through the shutdown sequence."""
    phase: ShutdownPhase = ShutdownPhase.RUNNING
    signal_received: Optional[str] = None
    orders_cancelled: int = 0
    reservations_released: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def is_shutting_down(self) -> bool:
        """True if shutdown has been initiated."""
        return self.phase != ShutdownPhase.RUNNING

    @property
    def duration_seconds(self) -> Optional[float]:
        """How long shutdown took, or has been running."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class ShutdownManager:
    """
    Coordinates graceful shutdown of a trading bot.

    Registers signal handlers for SIGINT and SIGTERM, then on signal:
    1. Signals the bot's run loop to stop (via asyncio.Event)
    2. Cancels all open orders on the exchange
    3. Releases pending risk reservations
    4. Marks agent as stopped in storage
    5. Persists final state

    Usage:
        manager = ShutdownManager(client, risk, storage, agent_id)
        manager.install_signal_handlers(loop)

        # In run loop:
        while not manager.should_stop:
            await bot_iteration()

        # After loop exits:
        state = await manager.execute_shutdown()
    """

    def __init__(
        self,
        clients: dict | ExchangeClient,
        risk: RiskCoordinator,
        storage: StorageBackend,
        agent_id: str,
        shutdown_timeout_seconds: float = 30.0,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        # Normalize to dict for uniform handling (single + cross-exchange bots)
        if isinstance(clients, dict):
            self._clients: dict[str, ExchangeClient] = {
                str(k): v for k, v in clients.items()
            }
        else:
            self._clients = {"default": clients}

        self._risk = risk
        self._storage = storage
        self._agent_id = agent_id
        self._timeout = shutdown_timeout_seconds
        self._on_stop = on_stop

        self._stop_event = asyncio.Event()
        self._state = ShutdownState()
        self._signal_handlers_installed = False
        self._shutdown_executed = False

    @property
    def should_stop(self) -> bool:
        """Check if the bot should stop its run loop."""
        return self._stop_event.is_set()

    @property
    def state(self) -> ShutdownState:
        """Current shutdown state."""
        return self._state

    def request_stop(self, reason: str = "manual") -> None:
        """Request the bot to stop. Can be called from any context."""
        if not self._stop_event.is_set():
            logger.info("Shutdown requested: %s", reason)
            self._state.signal_received = reason
            self._stop_event.set()

    def install_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> None:
        """Install SIGINT and SIGTERM handlers on the event loop.

        Args:
            loop: The asyncio event loop to install handlers on.
        """
        if self._signal_handlers_installed:
            return

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: self._handle_signal(s),
            )
        self._signal_handlers_installed = True
        logger.debug("Signal handlers installed for SIGINT, SIGTERM")

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle an OS signal by requesting shutdown.

        First signal: sets the stop event and calls the on_stop callback
        (which should set bot._running = False so the run loop exits).
        Second signal: forces immediate exit via os._exit.
        """
        sig_name = sig.name
        if self._stop_event.is_set():
            logger.warning(
                "Received %s again during shutdown — forcing exit", sig_name
            )
            # Second signal = force exit without cleanup
            os._exit(128 + sig.value)

        logger.info("Received %s — initiating graceful shutdown", sig_name)
        self._state.signal_received = sig_name
        self._stop_event.set()

        # Notify the bot to stop its run loop (sets _running = False)
        if self._on_stop is not None:
            try:
                self._on_stop()
            except Exception as e:
                logger.error("on_stop callback failed: %s", e)

    async def execute_shutdown(self) -> ShutdownState:
        """Execute the full shutdown sequence with timeout.

        Idempotent: calling this multiple times is safe; only the first
        invocation performs cleanup.

        Returns:
            ShutdownState with results of the shutdown process.
        """
        if self._shutdown_executed:
            return self._state

        self._shutdown_executed = True
        self._state.started_at = datetime.now(timezone.utc)
        self._state.phase = ShutdownPhase.STOPPING

        logger.info(
            "Beginning graceful shutdown (timeout=%ds)", self._timeout
        )

        try:
            await asyncio.wait_for(
                self._shutdown_sequence(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            self._state.errors.append(
                f"Shutdown timed out after {self._timeout}s"
            )
            logger.error(
                "Shutdown timed out after %ds — some cleanup may be incomplete",
                self._timeout,
            )

        self._state.phase = ShutdownPhase.COMPLETE
        self._state.completed_at = datetime.now(timezone.utc)

        duration = self._state.duration_seconds
        logger.info(
            "Shutdown complete in %.1fs: cancelled %d orders, "
            "released %d reservations, %d errors",
            duration or 0,
            self._state.orders_cancelled,
            self._state.reservations_released,
            len(self._state.errors),
        )

        return self._state

    async def _shutdown_sequence(self) -> None:
        """Internal shutdown steps, run under the timeout."""
        # Phase 1: Cancel open orders on all exchanges
        self._state.phase = ShutdownPhase.CANCELLING_ORDERS
        await self._cancel_all_orders()

        # Phase 2: Release pending reservations
        self._state.phase = ShutdownPhase.RELEASING_RESERVATIONS
        self._release_reservations()

        # Phase 3: Mark agent stopped and persist state
        self._state.phase = ShutdownPhase.PERSISTING_STATE
        self._finalize_agent_state()

    async def _cancel_all_orders(self) -> None:
        """Cancel all open orders across all exchange clients."""
        for label, client in self._clients.items():
            try:
                cancelled = await client.cancel_all_orders()
                self._state.orders_cancelled += cancelled
                if cancelled > 0:
                    logger.info(
                        "Cancelled %d orders on %s", cancelled, label
                    )
            except Exception as e:
                msg = f"Failed to cancel orders on {label}: {e}"
                self._state.errors.append(msg)
                logger.error(msg)

    def _release_reservations(self) -> None:
        """Release all pending reservations for this agent."""
        try:
            released = self._storage.cleanup_expired_reservations(
                agent_id=self._agent_id
            )
            self._state.reservations_released = released
            if released > 0:
                logger.info("Released %d expired reservations", released)
        except Exception as e:
            msg = f"Failed to release reservations: {e}"
            self._state.errors.append(msg)
            logger.error(msg)

    def _finalize_agent_state(self) -> None:
        """Mark the agent as stopped in storage.

        Skipped if the on_stop callback was provided, because bot.stop()
        already calls risk.shutdown(). This prevents duplicate cleanup calls.
        """
        if self._on_stop is not None:
            # bot.stop() already called risk.shutdown — skip to avoid duplicate
            logger.debug("Skipping risk.shutdown — bot.stop() handles it")
            return

        try:
            self._risk.shutdown(self._agent_id)
        except Exception as e:
            msg = f"Failed to mark agent stopped: {e}"
            self._state.errors.append(msg)
            logger.error(msg)


class StartupRecovery:
    """
    Reconciles bot state on startup after a potentially unclean exit.

    Checks for:
    - Orphaned open orders on the exchange that we don't track locally
    - Stale agent records (crashed agents that never called shutdown)
    - Expired/leaked capital reservations
    - DB positions with no corresponding exchange position

    Call recover() before starting the bot's main loop.
    """

    def __init__(
        self,
        client: ExchangeClient,
        storage: StorageBackend,
        risk: RiskCoordinator,
        agent_id: str,
    ):
        self._client = client
        self._storage = storage
        self._risk = risk
        self._agent_id = agent_id

    async def recover(self) -> dict[str, int]:
        """Run all recovery checks and return a summary.

        Returns:
            Dict with counts: stale_agents, expired_reservations,
            orphaned_orders_cancelled, position_mismatches.
        """
        logger.info(
            "Running startup recovery for agent '%s'", self._agent_id
        )
        results: dict[str, int] = {
            "stale_agents": 0,
            "expired_reservations": 0,
            "orphaned_orders_cancelled": 0,
            "position_mismatches": 0,
        }

        # 1. Clean up stale agents
        results["stale_agents"] = self._cleanup_stale_agents()

        # 2. Release expired reservations
        results["expired_reservations"] = self._cleanup_reservations()

        # 3. Cancel orphaned orders
        results["orphaned_orders_cancelled"] = await self._cancel_orphaned_orders()

        # 4. Check position mismatches
        results["position_mismatches"] = await self._check_position_mismatches()

        logger.info("Startup recovery complete: %s", results)
        return results

    def _cleanup_stale_agents(self) -> int:
        """Mark any crashed agents as stale."""
        try:
            count = self._storage.cleanup_stale_agents(
                self._risk.config.stale_agent_threshold_seconds
            )
            if count > 0:
                logger.warning("Cleaned up %d stale agent(s)", count)
            return count
        except Exception as e:
            logger.error("Failed to cleanup stale agents: %s", e)
            return 0

    def _cleanup_reservations(self) -> int:
        """Release expired capital reservations."""
        try:
            count = self._storage.cleanup_expired_reservations()
            if count > 0:
                logger.info("Released %d expired reservation(s)", count)
            return count
        except Exception as e:
            logger.error("Failed to cleanup reservations: %s", e)
            return 0

    async def _cancel_orphaned_orders(self) -> int:
        """Cancel any open orders on the exchange that we don't expect.

        On startup, the bot has no active orders tracked in memory.
        Any open orders on the exchange are orphans from a previous run.
        """
        try:
            open_orders = await self._client.get_open_orders()
            if not open_orders:
                return 0

            logger.warning(
                "Found %d orphaned order(s) on exchange — cancelling",
                len(open_orders),
            )
            order_ids = [o.order_id for o in open_orders]
            cancelled = await self._client.cancel_orders(order_ids)
            logger.info("Cancelled %d/%d orphaned orders", cancelled, len(order_ids))
            return cancelled
        except Exception as e:
            logger.error("Failed to cancel orphaned orders: %s", e)
            return 0

    async def _check_position_mismatches(self) -> int:
        """Log mismatches between DB positions and exchange positions.

        Does not auto-close anything — just logs warnings for the operator.
        Returns count of mismatches found.
        """
        mismatches = 0
        try:
            exchange_positions = await self._client.get_positions()
            db_positions = self._storage.get_agent_positions(
                self._agent_id, "open"
            )

            exchange_by_id = {ep.instrument_id: ep for ep in exchange_positions}
            db_ids = set()

            for pos in db_positions:
                iid = pos["instrument_id"]
                db_ids.add(iid)
                ep = exchange_by_id.get(iid)
                if ep is None:
                    logger.warning(
                        "RECOVERY: DB position %s (id=%s) not found on exchange "
                        "— may have been liquidated or expired",
                        iid,
                        pos["position_id"],
                    )
                    mismatches += 1
                elif abs(ep.size - pos["size"]) > 0.001:
                    logger.warning(
                        "RECOVERY: %s size mismatch — DB=%.4f, exchange=%.4f",
                        iid,
                        pos["size"],
                        ep.size,
                    )
                    mismatches += 1

            for iid, ep in exchange_by_id.items():
                if iid not in db_ids:
                    logger.warning(
                        "RECOVERY: exchange position %s (%s %.4f) not tracked in DB",
                        iid,
                        ep.side.value,
                        ep.size,
                    )
                    mismatches += 1

        except Exception as e:
            logger.error("Failed to check position mismatches: %s", e)

        return mismatches


class CrossExchangeStartupRecovery:
    """Startup recovery for cross-exchange bots with multiple clients.

    Runs StartupRecovery on each exchange independently and aggregates results.
    """

    def __init__(
        self,
        clients: dict,
        storage: StorageBackend,
        risk: RiskCoordinator,
        agent_id: str,
    ):
        self._clients = clients
        self._storage = storage
        self._risk = risk
        self._agent_id = agent_id

    async def recover(self) -> dict[str, int]:
        """Run recovery across all exchanges.

        Returns:
            Aggregated counts across all exchanges.
        """
        totals: dict[str, int] = {
            "stale_agents": 0,
            "expired_reservations": 0,
            "orphaned_orders_cancelled": 0,
            "position_mismatches": 0,
        }

        # Shared cleanup (runs once, not per-exchange)
        try:
            totals["stale_agents"] = self._storage.cleanup_stale_agents(
                self._risk.config.stale_agent_threshold_seconds
            )
        except Exception as e:
            logger.error("Failed to cleanup stale agents: %s", e)

        try:
            totals["expired_reservations"] = (
                self._storage.cleanup_expired_reservations()
            )
        except Exception as e:
            logger.error("Failed to cleanup reservations: %s", e)

        # Per-exchange recovery
        for exchange_id, client in self._clients.items():
            sub_agent = f"{self._agent_id}-{exchange_id.value}"
            recovery = StartupRecovery(
                client=client,
                storage=self._storage,
                risk=self._risk,
                agent_id=sub_agent,
            )
            # Only run exchange-specific checks (skip shared cleanup)
            cancelled = await recovery._cancel_orphaned_orders()
            mismatches = await recovery._check_position_mismatches()
            totals["orphaned_orders_cancelled"] += cancelled
            totals["position_mismatches"] += mismatches

        logger.info("Cross-exchange recovery complete: %s", totals)
        return totals
