"""
HTTP metrics server for OmniTrade monitoring.

Exposes bot metrics and alerts as JSON via a lightweight HTTP endpoint.
Uses Python's built-in http.server to avoid adding dependencies.
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .collector import MetricsCollector
from .alerts import AlertManager

logger = logging.getLogger(__name__)


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for metrics endpoints."""

    # Set by MetricsServer before starting
    collector: MetricsCollector
    alert_manager: Optional[AlertManager] = None

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/metrics" or self.path == "/metrics/":
            self._serve_metrics()
        elif self.path == "/metrics/alerts" or self.path == "/metrics/alerts/":
            self._serve_alerts()
        elif self.path == "/health" or self.path == "/health/":
            self._serve_health()
        else:
            self._send_json(
                {"error": "not found", "endpoints": ["/metrics", "/metrics/alerts", "/health"]},
                status=404,
            )

    def _serve_metrics(self) -> None:
        """Serve full metrics snapshot."""
        snapshot = self.collector.snapshot()
        self._send_json(snapshot.to_dict())

    def _serve_alerts(self) -> None:
        """Serve cached alerts (read-only, no side effects)."""
        if self.alert_manager is None:
            self._send_json({"alerts": [], "message": "alerting not configured"})
            return

        alerts = self.alert_manager.active_alerts
        self._send_json({
            "alert_count": len(alerts),
            "alerts": [a.to_dict() for a in alerts],
        })

    def _serve_health(self) -> None:
        """Simple health check."""
        snapshot = self.collector.snapshot()
        self._send_json({
            "status": "ok",
            "active_bots": snapshot.system.get("active_bots", 0),
            "uptime_seconds": snapshot.system.get("collector_uptime_seconds", 0),
        })

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default request logging to avoid noise."""
        pass


class MetricsServer:
    """
    Lightweight HTTP server exposing bot metrics as JSON.

    Runs in a daemon thread so it does not block the main asyncio loop.
    Endpoints:
        GET /metrics        - Full metrics snapshot (all bots)
        GET /metrics/alerts - Current alerts
        GET /health         - Simple health check

    Usage:
        collector = MetricsCollector()
        server = MetricsServer(collector, port=9090)
        server.start()
        # ... later ...
        server.stop()
    """

    def __init__(
        self,
        collector: MetricsCollector,
        alert_manager: Optional[AlertManager] = None,
        host: str = "0.0.0.0",
        port: int = 9090,
    ) -> None:
        self._collector = collector
        self._alert_manager = alert_manager
        self._host = host
        self._port = port
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        """The port this server is configured on."""
        return self._port

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the metrics server in a daemon thread."""
        if self.is_running:
            logger.warning("Metrics server already running on port %d", self._port)
            return

        # Inject dependencies into the handler class
        handler_class = type(
            "_BoundMetricsHandler",
            (_MetricsHandler,),
            {
                "collector": self._collector,
                "alert_manager": self._alert_manager,
            },
        )

        self._httpd = HTTPServer((self._host, self._port), handler_class)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="metrics-server",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Metrics server started on http://%s:%d/metrics",
            self._host,
            self._port,
        )

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Metrics server stopped")
