"""Tests for MetricsCollector."""

import time
from datetime import datetime, timezone

from omnitrade.monitoring.collector import MetricsCollector, BotMetrics


class TestMetricsCollector:
    """Tests for the metrics collector."""

    def test_register_and_snapshot(self):
        """Registered bot appears in snapshot."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")

        snap = collector.snapshot()
        assert "bot-1" in snap.bots
        assert snap.bots["bot-1"]["agent_type"] == "directional"
        assert snap.bots["bot-1"]["exchange"] == "polymarket"
        assert snap.system["active_bots"] == 1

    def test_unregister_bot(self):
        """Unregistered bot is removed from snapshot."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.unregister_bot("bot-1")

        snap = collector.snapshot()
        assert "bot-1" not in snap.bots
        assert snap.system["active_bots"] == 0

    def test_record_iteration_updates_metrics(self):
        """record_iteration updates the bot's metric state."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")

        collector.record_iteration(
            "bot-1",
            total_equity=1500.0,
            available_balance=1200.0,
            realized_pnl=50.0,
            unrealized_pnl=-10.0,
            open_position_count=3,
            exchange_latency_ms=150.0,
            circuit_breaker_state="CLOSED",
            daily_drawdown_pct=0.02,
            total_drawdown_pct=0.05,
        )

        snap = collector.snapshot()
        bot = snap.bots["bot-1"]
        assert bot["balance"]["total_equity"] == 1500.0
        assert bot["balance"]["available"] == 1200.0
        assert bot["pnl"]["realized"] == 50.0
        assert bot["pnl"]["unrealized"] == -10.0
        assert bot["pnl"]["total"] == 40.0
        assert bot["positions"]["open_count"] == 3
        assert bot["exchange_health"]["latency_ms"] == 150.0
        assert bot["safety"]["circuit_breaker_state"] == "CLOSED"
        assert bot["safety"]["daily_drawdown_pct"] == 0.02
        assert bot["timing"]["iteration_count"] == 1

    def test_record_iteration_increments_count(self):
        """Multiple iterations increment the count."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "mm", "kalshi")

        collector.record_iteration("bot-1", total_equity=100.0)
        collector.record_iteration("bot-1", total_equity=101.0)
        collector.record_iteration("bot-1", total_equity=102.0)

        snap = collector.snapshot()
        assert snap.bots["bot-1"]["timing"]["iteration_count"] == 3

    def test_record_iteration_unknown_agent_is_noop(self):
        """Recording for unknown agent does not raise."""
        collector = MetricsCollector()
        collector.record_iteration("nonexistent", total_equity=100.0)
        snap = collector.snapshot()
        assert len(snap.bots) == 0

    def test_record_order_tracks_fills_and_failures(self):
        """record_order tracks placed, filled, and failed counts."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")

        collector.record_order("bot-1", filled=True)
        collector.record_order("bot-1", filled=True)
        collector.record_order("bot-1", filled=False, failed=True)
        collector.record_order("bot-1", filled=False)

        snap = collector.snapshot()
        orders = snap.bots["bot-1"]["orders"]
        assert orders["placed"] == 4
        assert orders["filled"] == 2
        assert orders["failed"] == 1
        assert orders["fill_rate"] == 0.5

    def test_record_order_unknown_agent_is_noop(self):
        """Recording orders for unknown agent does not raise."""
        collector = MetricsCollector()
        collector.record_order("nonexistent", filled=True)

    def test_multiple_bots_independent(self):
        """Multiple bots track metrics independently."""
        collector = MetricsCollector()
        collector.register_bot("bot-a", "directional", "polymarket")
        collector.register_bot("bot-b", "mm", "kalshi")

        collector.record_iteration("bot-a", total_equity=1000.0)
        collector.record_iteration("bot-b", total_equity=2000.0)

        snap = collector.snapshot()
        assert snap.bots["bot-a"]["balance"]["total_equity"] == 1000.0
        assert snap.bots["bot-b"]["balance"]["total_equity"] == 2000.0
        assert snap.system["active_bots"] == 2

    def test_snapshot_timestamp_is_utc(self):
        """Snapshot timestamp is in UTC."""
        collector = MetricsCollector()
        snap = collector.snapshot()
        assert snap.timestamp.tzinfo == timezone.utc

    def test_get_bot_metrics_returns_none_for_unknown(self):
        """get_bot_metrics returns None for unregistered bot."""
        collector = MetricsCollector()
        assert collector.get_bot_metrics("nonexistent") is None

    def test_get_bot_metrics_returns_data(self):
        """get_bot_metrics returns the BotMetrics object."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration("bot-1", total_equity=500.0)

        bot = collector.get_bot_metrics("bot-1")
        assert bot is not None
        assert bot.total_equity == 500.0
        assert bot.agent_id == "bot-1"

    def test_get_bot_metrics_returns_defensive_copy(self):
        """get_bot_metrics returns a copy, not the live mutable object."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration("bot-1", total_equity=500.0)

        copy1 = collector.get_bot_metrics("bot-1")
        assert copy1 is not None
        copy1.total_equity = 9999.0  # mutate the copy

        copy2 = collector.get_bot_metrics("bot-1")
        assert copy2 is not None
        assert copy2.total_equity == 500.0  # internal state unchanged

    def test_last_successful_request_none_when_no_latency(self):
        """last_successful_request stays None when latency is 0 (no real request)."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration("bot-1", total_equity=100.0, exchange_latency_ms=0.0)

        bot = collector.get_bot_metrics("bot-1")
        assert bot is not None
        assert bot.last_successful_request is None

    def test_last_successful_request_set_when_positive_latency(self):
        """last_successful_request is set when latency > 0."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration("bot-1", total_equity=100.0, exchange_latency_ms=50.0)

        bot = collector.get_bot_metrics("bot-1")
        assert bot is not None
        assert bot.last_successful_request is not None

    def test_set_alert_count(self):
        """set_alert_count is reflected in snapshot."""
        collector = MetricsCollector()
        collector.set_alert_count(5)
        snap = collector.snapshot()
        assert snap.system["active_alerts"] == 5

    def test_bot_metrics_to_dict_fill_rate_zero_orders(self):
        """Fill rate is 0.0 when no orders have been placed."""
        metrics = BotMetrics(agent_id="test", agent_type="directional", exchange="polymarket")
        d = metrics.to_dict()
        assert d["orders"]["fill_rate"] == 0.0

    def test_snapshot_to_dict(self):
        """MetricSnapshot.to_dict produces valid structure."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        snap = collector.snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        assert "bots" in d
        assert "system" in d
        assert isinstance(d["timestamp"], str)
