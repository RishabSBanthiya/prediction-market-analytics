"""
Metrics collector for OmniTrade bots.

Lightweight, thread-safe metrics collection that bots push data into.
The collector maintains a rolling window of metrics per bot agent,
exposable via the HTTP metrics server.
"""

import copy
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BotMetrics:
    """Per-bot metric state. Updated by the bot on each iteration."""

    agent_id: str
    agent_type: str
    exchange: str

    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Balance
    total_equity: float = 0.0
    available_balance: float = 0.0

    # Positions
    open_position_count: int = 0

    # Orders
    orders_placed: int = 0
    orders_filled: int = 0
    orders_failed: int = 0

    # Exchange health
    exchange_latency_ms: float = 0.0
    last_successful_request: Optional[datetime] = None

    # Safety
    circuit_breaker_state: str = "CLOSED"
    daily_drawdown_pct: float = 0.0
    total_drawdown_pct: float = 0.0

    # Timing
    last_iteration_at: Optional[datetime] = None
    iteration_count: int = 0
    uptime_seconds: float = 0.0
    started_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "exchange": self.exchange,
            "pnl": {
                "realized": round(self.realized_pnl, 4),
                "unrealized": round(self.unrealized_pnl, 4),
                "total": round(self.realized_pnl + self.unrealized_pnl, 4),
            },
            "balance": {
                "total_equity": round(self.total_equity, 2),
                "available": round(self.available_balance, 2),
            },
            "positions": {
                "open_count": self.open_position_count,
            },
            "orders": {
                "placed": self.orders_placed,
                "filled": self.orders_filled,
                "failed": self.orders_failed,
                "fill_rate": (
                    round(self.orders_filled / self.orders_placed, 4)
                    if self.orders_placed > 0
                    else 0.0
                ),
            },
            "exchange_health": {
                "latency_ms": round(self.exchange_latency_ms, 1),
                "last_successful_request": (
                    self.last_successful_request.isoformat()
                    if self.last_successful_request
                    else None
                ),
            },
            "safety": {
                "circuit_breaker_state": self.circuit_breaker_state,
                "daily_drawdown_pct": round(self.daily_drawdown_pct, 4),
                "total_drawdown_pct": round(self.total_drawdown_pct, 4),
            },
            "timing": {
                "last_iteration_at": (
                    self.last_iteration_at.isoformat()
                    if self.last_iteration_at
                    else None
                ),
                "iteration_count": self.iteration_count,
                "uptime_seconds": round(self.uptime_seconds, 1),
                "started_at": (
                    self.started_at.isoformat() if self.started_at else None
                ),
            },
        }


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of all bot metrics."""

    timestamp: datetime
    bots: dict[str, dict] = field(default_factory=dict)
    system: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "bots": self.bots,
            "system": self.system,
        }


class MetricsCollector:
    """
    Central metrics collector for all running bots.

    Thread-safe: bots call record_* methods from async loops,
    the HTTP server reads via snapshot() from a different thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bots: dict[str, BotMetrics] = {}
        self._start_time = time.monotonic()
        self._alert_count = 0

    def register_bot(self, agent_id: str, agent_type: str, exchange: str) -> None:
        """Register a bot for metrics tracking."""
        with self._lock:
            self._bots[agent_id] = BotMetrics(
                agent_id=agent_id,
                agent_type=agent_type,
                exchange=exchange,
                started_at=datetime.now(timezone.utc),
            )
        logger.info("Metrics registered for bot %s (%s on %s)", agent_id, agent_type, exchange)

    def unregister_bot(self, agent_id: str) -> None:
        """Remove a bot from tracking."""
        with self._lock:
            self._bots.pop(agent_id, None)
        logger.info("Metrics unregistered for bot %s", agent_id)

    def record_iteration(
        self,
        agent_id: str,
        *,
        total_equity: float = 0.0,
        available_balance: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        open_position_count: int = 0,
        exchange_latency_ms: float = 0.0,
        circuit_breaker_state: str = "CLOSED",
        daily_drawdown_pct: float = 0.0,
        total_drawdown_pct: float = 0.0,
    ) -> None:
        """Record metrics from a single bot iteration."""
        with self._lock:
            bot = self._bots.get(agent_id)
            if bot is None:
                return

            bot.total_equity = total_equity
            bot.available_balance = available_balance
            bot.realized_pnl = realized_pnl
            bot.unrealized_pnl = unrealized_pnl
            bot.open_position_count = open_position_count
            bot.exchange_latency_ms = exchange_latency_ms
            bot.circuit_breaker_state = circuit_breaker_state
            bot.daily_drawdown_pct = daily_drawdown_pct
            bot.total_drawdown_pct = total_drawdown_pct
            bot.last_iteration_at = datetime.now(timezone.utc)
            bot.iteration_count += 1

            if bot.started_at:
                bot.uptime_seconds = (
                    datetime.now(timezone.utc) - bot.started_at
                ).total_seconds()

            if exchange_latency_ms > 0:
                bot.last_successful_request = datetime.now(timezone.utc)

    def record_order(self, agent_id: str, *, filled: bool, failed: bool = False) -> None:
        """Record an order attempt result."""
        with self._lock:
            bot = self._bots.get(agent_id)
            if bot is None:
                return

            bot.orders_placed += 1
            if filled:
                bot.orders_filled += 1
            if failed:
                bot.orders_failed += 1

    def set_alert_count(self, count: int) -> None:
        """Update the current active alert count (set by AlertManager)."""
        with self._lock:
            self._alert_count = count

    def get_bot_metrics(self, agent_id: str) -> Optional[BotMetrics]:
        """Get current metrics for a specific bot. Returns a defensive copy."""
        with self._lock:
            bot = self._bots.get(agent_id)
            if bot is None:
                return None
            return replace(bot)

    def snapshot(self) -> MetricSnapshot:
        """Take a point-in-time snapshot of all metrics."""
        with self._lock:
            bots = {
                agent_id: bot.to_dict() for agent_id, bot in self._bots.items()
            }
            system = {
                "collector_uptime_seconds": round(
                    time.monotonic() - self._start_time, 1
                ),
                "active_bots": len(self._bots),
                "active_alerts": self._alert_count,
            }

        return MetricSnapshot(
            timestamp=datetime.now(timezone.utc),
            bots=bots,
            system=system,
        )
