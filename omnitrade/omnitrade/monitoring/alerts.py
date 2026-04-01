"""
Alert threshold checking for OmniTrade monitoring.

Checks bot metrics against configurable thresholds and emits alerts
via standard logging. Alerts have severity levels and cooldown periods
to avoid spam.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .collector import MetricsCollector, BotMetrics

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Configurable alert thresholds."""

    # Drawdown thresholds (fraction, e.g. 0.03 = 3%)
    daily_drawdown_warning_pct: float = 0.03
    daily_drawdown_critical_pct: float = 0.05
    total_drawdown_warning_pct: float = 0.10
    total_drawdown_critical_pct: float = 0.15

    # Fill rate (fraction, e.g. 0.5 = 50%)
    min_fill_rate: float = 0.3

    # Exchange latency (milliseconds)
    latency_warning_ms: float = 2000.0
    latency_critical_ms: float = 5000.0

    # Inactivity (seconds since last iteration)
    inactivity_warning_seconds: float = 120.0
    inactivity_critical_seconds: float = 300.0

    # No fills threshold: alert if zero fills after N orders
    no_fills_after_orders: int = 10

    # Cooldown between repeated alerts of the same type (seconds)
    alert_cooldown_seconds: float = 300.0


@dataclass
class Alert:
    """A single alert event."""

    alert_id: str
    agent_id: str
    severity: AlertSeverity
    category: str
    message: str
    value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "alert_id": self.alert_id,
            "agent_id": self.agent_id,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4),
            "timestamp": self.timestamp.isoformat(),
        }


class AlertManager:
    """
    Checks bot metrics against thresholds and manages alert lifecycle.

    Alerts are emitted via logging and stored in a rolling buffer.
    Each alert type per bot has a cooldown to prevent spam.
    """

    def __init__(
        self,
        collector: MetricsCollector,
        config: Optional[AlertConfig] = None,
        max_alerts: int = 100,
    ) -> None:
        self._collector = collector
        self._config = config or AlertConfig()
        self._max_alerts = max_alerts
        self._alerts: list[Alert] = []
        self._cooldowns: dict[str, float] = {}  # "agent_id:category" -> last_fired_monotonic

    @property
    def config(self) -> AlertConfig:
        """Current alert configuration."""
        return self._config

    @property
    def active_alerts(self) -> list[Alert]:
        """List of all stored alerts (most recent first)."""
        return list(reversed(self._alerts))

    def run_checks(self) -> list[Alert]:
        """
        Check all registered bots against thresholds.

        Call this periodically from the bot's main loop (not from HTTP handlers).
        Returns list of newly fired alerts.
        """
        snapshot = self._collector.snapshot()
        new_alerts: list[Alert] = []

        for agent_id, bot_data in snapshot.bots.items():
            bot_metrics = self._collector.get_bot_metrics(agent_id)
            if bot_metrics is None:
                continue

            new_alerts.extend(self._check_bot(bot_metrics))

        # Update the collector with current alert count
        self._collector.set_alert_count(len(self._alerts))

        return new_alerts

    def check_all(self) -> list[Alert]:
        """Alias for run_checks() for backward compatibility."""
        return self.run_checks()

    def _check_bot(self, bot: BotMetrics) -> list[Alert]:
        """Run all threshold checks for a single bot."""
        new_alerts: list[Alert] = []
        cfg = self._config

        # --- Drawdown checks ---
        if bot.daily_drawdown_pct >= cfg.daily_drawdown_critical_pct:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.CRITICAL,
                category="daily_drawdown",
                message=(
                    f"Daily drawdown {bot.daily_drawdown_pct:.1%} exceeds "
                    f"critical threshold {cfg.daily_drawdown_critical_pct:.1%}"
                ),
                value=bot.daily_drawdown_pct,
                threshold=cfg.daily_drawdown_critical_pct,
            )
            if alert:
                new_alerts.append(alert)
        elif bot.daily_drawdown_pct >= cfg.daily_drawdown_warning_pct:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.WARNING,
                category="daily_drawdown",
                message=(
                    f"Daily drawdown {bot.daily_drawdown_pct:.1%} exceeds "
                    f"warning threshold {cfg.daily_drawdown_warning_pct:.1%}"
                ),
                value=bot.daily_drawdown_pct,
                threshold=cfg.daily_drawdown_warning_pct,
            )
            if alert:
                new_alerts.append(alert)

        if bot.total_drawdown_pct >= cfg.total_drawdown_critical_pct:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.CRITICAL,
                category="total_drawdown",
                message=(
                    f"Total drawdown {bot.total_drawdown_pct:.1%} exceeds "
                    f"critical threshold {cfg.total_drawdown_critical_pct:.1%}"
                ),
                value=bot.total_drawdown_pct,
                threshold=cfg.total_drawdown_critical_pct,
            )
            if alert:
                new_alerts.append(alert)
        elif bot.total_drawdown_pct >= cfg.total_drawdown_warning_pct:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.WARNING,
                category="total_drawdown",
                message=(
                    f"Total drawdown {bot.total_drawdown_pct:.1%} exceeds "
                    f"warning threshold {cfg.total_drawdown_warning_pct:.1%}"
                ),
                value=bot.total_drawdown_pct,
                threshold=cfg.total_drawdown_warning_pct,
            )
            if alert:
                new_alerts.append(alert)

        # --- Fill rate check ---
        if (
            bot.orders_placed >= cfg.no_fills_after_orders
            and bot.orders_filled == 0
        ):
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.CRITICAL,
                category="no_fills",
                message=(
                    f"Zero fills after {bot.orders_placed} orders placed"
                ),
                value=0.0,
                threshold=float(cfg.no_fills_after_orders),
            )
            if alert:
                new_alerts.append(alert)
        elif bot.orders_placed > 0:
            fill_rate = bot.orders_filled / bot.orders_placed
            if fill_rate < cfg.min_fill_rate:
                alert = self._fire(
                    agent_id=bot.agent_id,
                    severity=AlertSeverity.WARNING,
                    category="low_fill_rate",
                    message=(
                        f"Fill rate {fill_rate:.1%} below threshold "
                        f"{cfg.min_fill_rate:.1%} "
                        f"({bot.orders_filled}/{bot.orders_placed})"
                    ),
                    value=fill_rate,
                    threshold=cfg.min_fill_rate,
                )
                if alert:
                    new_alerts.append(alert)

        # --- Latency checks ---
        if bot.exchange_latency_ms >= cfg.latency_critical_ms:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.CRITICAL,
                category="exchange_latency",
                message=(
                    f"Exchange latency {bot.exchange_latency_ms:.0f}ms exceeds "
                    f"critical threshold {cfg.latency_critical_ms:.0f}ms"
                ),
                value=bot.exchange_latency_ms,
                threshold=cfg.latency_critical_ms,
            )
            if alert:
                new_alerts.append(alert)
        elif bot.exchange_latency_ms >= cfg.latency_warning_ms:
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.WARNING,
                category="exchange_latency",
                message=(
                    f"Exchange latency {bot.exchange_latency_ms:.0f}ms exceeds "
                    f"warning threshold {cfg.latency_warning_ms:.0f}ms"
                ),
                value=bot.exchange_latency_ms,
                threshold=cfg.latency_warning_ms,
            )
            if alert:
                new_alerts.append(alert)

        # --- Inactivity check ---
        if bot.last_iteration_at is not None:
            inactive_seconds = (
                datetime.now(timezone.utc) - bot.last_iteration_at
            ).total_seconds()

            if inactive_seconds >= cfg.inactivity_critical_seconds:
                alert = self._fire(
                    agent_id=bot.agent_id,
                    severity=AlertSeverity.CRITICAL,
                    category="inactivity",
                    message=(
                        f"Bot inactive for {inactive_seconds:.0f}s "
                        f"(critical threshold: {cfg.inactivity_critical_seconds:.0f}s)"
                    ),
                    value=inactive_seconds,
                    threshold=cfg.inactivity_critical_seconds,
                )
                if alert:
                    new_alerts.append(alert)
            elif inactive_seconds >= cfg.inactivity_warning_seconds:
                alert = self._fire(
                    agent_id=bot.agent_id,
                    severity=AlertSeverity.WARNING,
                    category="inactivity",
                    message=(
                        f"Bot inactive for {inactive_seconds:.0f}s "
                        f"(warning threshold: {cfg.inactivity_warning_seconds:.0f}s)"
                    ),
                    value=inactive_seconds,
                    threshold=cfg.inactivity_warning_seconds,
                )
                if alert:
                    new_alerts.append(alert)

        # --- Circuit breaker check ---
        if bot.circuit_breaker_state == "OPEN":
            alert = self._fire(
                agent_id=bot.agent_id,
                severity=AlertSeverity.CRITICAL,
                category="circuit_breaker",
                message="Circuit breaker is OPEN - trading halted",
                value=1.0,
                threshold=0.0,
            )
            if alert:
                new_alerts.append(alert)

        return new_alerts

    def _fire(
        self,
        agent_id: str,
        severity: AlertSeverity,
        category: str,
        message: str,
        value: float = 0.0,
        threshold: float = 0.0,
    ) -> Optional[Alert]:
        """
        Fire an alert if not in cooldown.

        Returns the Alert if fired, None if suppressed by cooldown.
        """
        cooldown_key = f"{agent_id}:{category}:{severity.value}"
        now = time.monotonic()

        last_fired = self._cooldowns.get(cooldown_key, 0.0)
        if now - last_fired < self._config.alert_cooldown_seconds:
            return None

        self._cooldowns[cooldown_key] = now

        alert = Alert(
            alert_id=f"{agent_id}:{category}:{int(now)}",
            agent_id=agent_id,
            severity=severity,
            category=category,
            message=message,
            value=value,
            threshold=threshold,
        )

        # Store (bounded)
        self._alerts.append(alert)
        if len(self._alerts) > self._max_alerts:
            self._alerts = self._alerts[-self._max_alerts :]

        # Log
        log_fn = logger.warning if severity == AlertSeverity.WARNING else logger.error
        log_fn("ALERT [%s] %s: %s", severity.value.upper(), agent_id, message)

        return alert

    def clear_cooldowns(self) -> None:
        """Clear all cooldowns (useful for testing)."""
        self._cooldowns.clear()

    def get_alerts_for_bot(self, agent_id: str) -> list[Alert]:
        """Get alerts for a specific bot."""
        return [a for a in self._alerts if a.agent_id == agent_id]
