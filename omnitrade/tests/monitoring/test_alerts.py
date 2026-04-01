"""Tests for AlertManager and alert threshold checking."""

import time
from datetime import datetime, timezone, timedelta

from omnitrade.monitoring.collector import MetricsCollector
from omnitrade.monitoring.alerts import (
    AlertManager,
    AlertConfig,
    Alert,
    AlertSeverity,
)


def _make_manager(
    config: AlertConfig | None = None,
) -> tuple[MetricsCollector, AlertManager]:
    """Helper to create a collector + alert manager pair."""
    collector = MetricsCollector()
    mgr = AlertManager(
        collector,
        config=config or AlertConfig(alert_cooldown_seconds=0),
    )
    return collector, mgr


class TestAlertManagerDrawdown:
    """Drawdown alert tests."""

    def test_no_alerts_when_healthy(self):
        """No alerts when metrics are within thresholds."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            daily_drawdown_pct=0.01,
            total_drawdown_pct=0.05,
        )
        alerts = mgr.check_all()
        assert len(alerts) == 0

    def test_daily_drawdown_warning(self):
        """Warning alert when daily drawdown exceeds warning threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=970.0,
            daily_drawdown_pct=0.04,  # > 0.03 warning
        )
        alerts = mgr.check_all()
        dd_alerts = [a for a in alerts if a.category == "daily_drawdown"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == AlertSeverity.WARNING

    def test_daily_drawdown_critical(self):
        """Critical alert when daily drawdown exceeds critical threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=940.0,
            daily_drawdown_pct=0.06,  # > 0.05 critical
        )
        alerts = mgr.check_all()
        dd_alerts = [a for a in alerts if a.category == "daily_drawdown"]
        assert len(dd_alerts) == 1
        assert dd_alerts[0].severity == AlertSeverity.CRITICAL

    def test_total_drawdown_warning(self):
        """Warning alert when total drawdown exceeds warning threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=880.0,
            total_drawdown_pct=0.12,  # > 0.10 warning
        )
        alerts = mgr.check_all()
        td_alerts = [a for a in alerts if a.category == "total_drawdown"]
        assert len(td_alerts) == 1
        assert td_alerts[0].severity == AlertSeverity.WARNING

    def test_total_drawdown_critical(self):
        """Critical alert when total drawdown exceeds critical threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=840.0,
            total_drawdown_pct=0.16,  # > 0.15 critical
        )
        alerts = mgr.check_all()
        td_alerts = [a for a in alerts if a.category == "total_drawdown"]
        assert len(td_alerts) == 1
        assert td_alerts[0].severity == AlertSeverity.CRITICAL


class TestAlertManagerFillRate:
    """Fill rate alert tests."""

    def test_no_fills_after_threshold_orders(self):
        """Critical alert when zero fills after N orders."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")

        # Place 10 orders, none filled
        for _ in range(10):
            collector.record_order("bot-1", filled=False, failed=True)

        collector.record_iteration("bot-1", total_equity=1000.0)
        alerts = mgr.check_all()
        nf_alerts = [a for a in alerts if a.category == "no_fills"]
        assert len(nf_alerts) == 1
        assert nf_alerts[0].severity == AlertSeverity.CRITICAL

    def test_low_fill_rate_warning(self):
        """Warning when fill rate is below threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")

        # 1 fill out of 5 orders = 20% < 30%
        collector.record_order("bot-1", filled=True)
        for _ in range(4):
            collector.record_order("bot-1", filled=False)

        collector.record_iteration("bot-1", total_equity=1000.0)
        alerts = mgr.check_all()
        fr_alerts = [a for a in alerts if a.category == "low_fill_rate"]
        assert len(fr_alerts) == 1
        assert fr_alerts[0].severity == AlertSeverity.WARNING

    def test_healthy_fill_rate_no_alert(self):
        """No alert when fill rate is above threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")

        # 3 fills out of 5 = 60% > 30%
        for _ in range(3):
            collector.record_order("bot-1", filled=True)
        for _ in range(2):
            collector.record_order("bot-1", filled=False)

        collector.record_iteration("bot-1", total_equity=1000.0)
        alerts = mgr.check_all()
        fr_alerts = [a for a in alerts if a.category in ("low_fill_rate", "no_fills")]
        assert len(fr_alerts) == 0


class TestAlertManagerLatency:
    """Exchange latency alert tests."""

    def test_latency_warning(self):
        """Warning when latency exceeds warning threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            exchange_latency_ms=3000.0,  # > 2000 warning
        )
        alerts = mgr.check_all()
        lat_alerts = [a for a in alerts if a.category == "exchange_latency"]
        assert len(lat_alerts) == 1
        assert lat_alerts[0].severity == AlertSeverity.WARNING

    def test_latency_critical(self):
        """Critical when latency exceeds critical threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            exchange_latency_ms=6000.0,  # > 5000 critical
        )
        alerts = mgr.check_all()
        lat_alerts = [a for a in alerts if a.category == "exchange_latency"]
        assert len(lat_alerts) == 1
        assert lat_alerts[0].severity == AlertSeverity.CRITICAL


class TestAlertManagerInactivity:
    """Bot inactivity alert tests."""

    def test_inactivity_warning(self):
        """Warning when bot has not iterated for too long."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")

        # Manually set last_iteration to 150 seconds ago on the live object
        with collector._lock:
            collector._bots["bot-1"].last_iteration_at = (
                datetime.now(timezone.utc) - timedelta(seconds=150)
            )

        alerts = mgr.check_all()
        inact_alerts = [a for a in alerts if a.category == "inactivity"]
        assert len(inact_alerts) == 1
        assert inact_alerts[0].severity == AlertSeverity.WARNING

    def test_inactivity_critical(self):
        """Critical when bot has been inactive beyond critical threshold."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")

        with collector._lock:
            collector._bots["bot-1"].last_iteration_at = (
                datetime.now(timezone.utc) - timedelta(seconds=400)
            )

        alerts = mgr.check_all()
        inact_alerts = [a for a in alerts if a.category == "inactivity"]
        assert len(inact_alerts) == 1
        assert inact_alerts[0].severity == AlertSeverity.CRITICAL


class TestAlertManagerCircuitBreaker:
    """Circuit breaker alert tests."""

    def test_circuit_breaker_open_alert(self):
        """Critical alert when circuit breaker is OPEN."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )
        alerts = mgr.check_all()
        cb_alerts = [a for a in alerts if a.category == "circuit_breaker"]
        assert len(cb_alerts) == 1
        assert cb_alerts[0].severity == AlertSeverity.CRITICAL

    def test_circuit_breaker_closed_no_alert(self):
        """No alert when circuit breaker is CLOSED."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="CLOSED",
        )
        alerts = mgr.check_all()
        cb_alerts = [a for a in alerts if a.category == "circuit_breaker"]
        assert len(cb_alerts) == 0


class TestAlertCooldown:
    """Alert cooldown tests."""

    def test_cooldown_suppresses_duplicate(self):
        """Same alert type is suppressed during cooldown."""
        config = AlertConfig(alert_cooldown_seconds=10.0)
        collector, mgr = _make_manager(config)
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )

        alerts1 = mgr.check_all()
        alerts2 = mgr.check_all()

        cb_1 = [a for a in alerts1 if a.category == "circuit_breaker"]
        cb_2 = [a for a in alerts2 if a.category == "circuit_breaker"]
        assert len(cb_1) == 1
        assert len(cb_2) == 0  # suppressed

    def test_cooldown_does_not_suppress_different_severity(self):
        """A warning cooldown must not suppress a subsequent critical alert."""
        config = AlertConfig(
            alert_cooldown_seconds=10.0,
            daily_drawdown_warning_pct=0.03,
            daily_drawdown_critical_pct=0.05,
        )
        collector, mgr = _make_manager(config)
        collector.register_bot("bot-1", "directional", "polymarket")

        # First: trigger a warning
        collector.record_iteration("bot-1", total_equity=1000.0, daily_drawdown_pct=0.04)
        alerts1 = mgr.check_all()
        dd_warn = [a for a in alerts1 if a.category == "daily_drawdown"]
        assert len(dd_warn) == 1
        assert dd_warn[0].severity == AlertSeverity.WARNING

        # Now: situation worsens to critical (within cooldown window)
        collector.record_iteration("bot-1", total_equity=1000.0, daily_drawdown_pct=0.06)
        alerts2 = mgr.check_all()
        dd_crit = [a for a in alerts2 if a.category == "daily_drawdown"]
        assert len(dd_crit) == 1
        assert dd_crit[0].severity == AlertSeverity.CRITICAL

    def test_clear_cooldowns(self):
        """clear_cooldowns allows same alert to fire again."""
        config = AlertConfig(alert_cooldown_seconds=10.0)
        collector, mgr = _make_manager(config)
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )

        mgr.check_all()
        mgr.clear_cooldowns()
        alerts = mgr.check_all()

        cb_alerts = [a for a in alerts if a.category == "circuit_breaker"]
        assert len(cb_alerts) == 1


class TestAlertManagerMultipleBots:
    """Tests with multiple bots."""

    def test_alerts_from_multiple_bots(self):
        """Alerts are tracked independently per bot."""
        collector, mgr = _make_manager()
        collector.register_bot("bot-a", "directional", "polymarket")
        collector.register_bot("bot-b", "mm", "kalshi")

        collector.record_iteration(
            "bot-a",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )
        collector.record_iteration(
            "bot-b",
            total_equity=500.0,
            daily_drawdown_pct=0.06,
        )

        alerts = mgr.check_all()
        a_alerts = mgr.get_alerts_for_bot("bot-a")
        b_alerts = mgr.get_alerts_for_bot("bot-b")

        assert any(a.category == "circuit_breaker" for a in a_alerts)
        assert any(a.category == "daily_drawdown" for a in b_alerts)


class TestAlertSerialization:
    """Alert serialization tests."""

    def test_alert_to_dict(self):
        """Alert.to_dict produces valid structure."""
        alert = Alert(
            alert_id="test:category:123",
            agent_id="bot-1",
            severity=AlertSeverity.WARNING,
            category="daily_drawdown",
            message="test message",
            value=0.04,
            threshold=0.03,
        )
        d = alert.to_dict()
        assert d["alert_id"] == "test:category:123"
        assert d["severity"] == "warning"
        assert d["category"] == "daily_drawdown"
        assert d["value"] == 0.04
        assert "timestamp" in d
