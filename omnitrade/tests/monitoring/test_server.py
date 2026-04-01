"""Tests for MetricsServer HTTP endpoints."""

import json
import urllib.request
import time

from omnitrade.monitoring.collector import MetricsCollector
from omnitrade.monitoring.alerts import AlertManager, AlertConfig
from omnitrade.monitoring.server import MetricsServer


def _get_json(port: int, path: str) -> dict:
    """Helper to GET a JSON endpoint."""
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestMetricsServer:
    """Tests for the HTTP metrics server."""

    def test_health_endpoint(self):
        """GET /health returns ok status."""
        collector = MetricsCollector()
        server = MetricsServer(collector, host="127.0.0.1", port=0)

        # Port 0 lets the OS assign a free port
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            data = _get_json(actual_port, "/health")
            assert data["status"] == "ok"
            assert "active_bots" in data
            assert "uptime_seconds" in data
        finally:
            server.stop()

    def test_metrics_endpoint(self):
        """GET /metrics returns full snapshot."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration("bot-1", total_equity=1000.0)

        server = MetricsServer(collector, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            data = _get_json(actual_port, "/metrics")
            assert "timestamp" in data
            assert "bots" in data
            assert "bot-1" in data["bots"]
            assert data["bots"]["bot-1"]["balance"]["total_equity"] == 1000.0
            assert data["system"]["active_bots"] == 1
        finally:
            server.stop()

    def test_alerts_endpoint_without_alert_manager(self):
        """GET /metrics/alerts without AlertManager returns empty list."""
        collector = MetricsCollector()
        server = MetricsServer(collector, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            data = _get_json(actual_port, "/metrics/alerts")
            assert data["alerts"] == []
        finally:
            server.stop()

    def test_alerts_endpoint_with_alert_manager(self):
        """GET /metrics/alerts returns cached alerts (read-only)."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )

        alert_mgr = AlertManager(
            collector, config=AlertConfig(alert_cooldown_seconds=0)
        )
        # Run checks before querying the endpoint (simulates periodic bot call)
        alert_mgr.run_checks()

        server = MetricsServer(collector, alert_manager=alert_mgr, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            data = _get_json(actual_port, "/metrics/alerts")
            assert data["alert_count"] > 0
            assert any(a["category"] == "circuit_breaker" for a in data["alerts"])
        finally:
            server.stop()

    def test_alerts_endpoint_is_read_only(self):
        """GET /metrics/alerts does not run checks or mutate state."""
        collector = MetricsCollector()
        collector.register_bot("bot-1", "directional", "polymarket")
        collector.record_iteration(
            "bot-1",
            total_equity=1000.0,
            circuit_breaker_state="OPEN",
        )

        alert_mgr = AlertManager(
            collector, config=AlertConfig(alert_cooldown_seconds=0)
        )
        # Do NOT call run_checks — endpoint should return empty
        server = MetricsServer(collector, alert_manager=alert_mgr, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            data = _get_json(actual_port, "/metrics/alerts")
            assert data["alert_count"] == 0
            assert data["alerts"] == []
        finally:
            server.stop()

    def test_404_for_unknown_path(self):
        """Unknown path returns 404 with endpoint list."""
        collector = MetricsCollector()
        server = MetricsServer(collector, host="127.0.0.1", port=0)
        server.start()
        try:
            actual_port = server._httpd.server_address[1]
            url = f"http://127.0.0.1:{actual_port}/unknown"
            try:
                urllib.request.urlopen(url, timeout=5)
                assert False, "Expected HTTP error"
            except urllib.error.HTTPError as e:
                assert e.code == 404
                body = json.loads(e.read().decode("utf-8"))
                assert "endpoints" in body
        finally:
            server.stop()

    def test_start_stop_lifecycle(self):
        """Server can be started and stopped cleanly."""
        collector = MetricsCollector()
        server = MetricsServer(collector, host="127.0.0.1", port=0)

        assert not server.is_running
        server.start()
        assert server.is_running
        server.stop()
        assert not server.is_running

    def test_double_start_is_noop(self):
        """Starting an already running server is a no-op."""
        collector = MetricsCollector()
        server = MetricsServer(collector, host="127.0.0.1", port=0)
        server.start()
        try:
            thread_id = server._thread.ident
            server.start()  # should not raise, should keep same thread
            assert server._thread.ident == thread_id
        finally:
            server.stop()
