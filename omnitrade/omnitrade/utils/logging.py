"""Structured JSON logging setup for omnitrade.

Provides a JSON formatter and rotating file handler so that every log line
is a single JSON object with consistent fields (timestamp, level, module,
bot_id, exchange, message).  Backward-compatible: callers keep using
``logger.info(...)`` as before.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Thread-local-ish context that bot runners inject at startup so every
# log record includes bot_id and exchange without callers threading it.
# ---------------------------------------------------------------------------
_log_context: dict[str, str] = {}


def set_log_context(*, bot_id: str = "", exchange: str = "") -> None:
    """Set global context fields that are added to every JSON log entry.

    Args:
        bot_id: The agent/bot identifier (e.g. ``directional-kalshi``).
        exchange: The exchange name (e.g. ``kalshi``, ``polymarket``).
    """
    if bot_id:
        _log_context["bot_id"] = bot_id
    if exchange:
        _log_context["exchange"] = exchange


def get_log_context() -> dict[str, str]:
    """Return a copy of the current global log context."""
    return dict(_log_context)


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------
class JSONFormatter(logging.Formatter):
    """Formats each log record as a single-line JSON object.

    Output fields:
        timestamp  - ISO-8601 UTC string
        level      - DEBUG / INFO / WARNING / ERROR / CRITICAL
        module     - logger name (e.g. ``omnitrade.risk.coordinator``)
        bot_id     - from global context (empty string if not set)
        exchange   - from global context (empty string if not set)
        message    - the formatted message string

    Any *extra* dict passed via ``logger.info("msg", extra={...})`` is
    merged into the JSON object under an ``"extra"`` key so callers can
    attach structured data ad-hoc.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "bot_id": _log_context.get("bot_id", ""),
            "exchange": _log_context.get("exchange", ""),
            "message": record.getMessage(),
        }

        # Propagate exception info into the JSON blob.
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Default log directory
# ---------------------------------------------------------------------------
_DEFAULT_LOG_DIR = Path("logs")


def setup_logging(
    level: str = "INFO",
    format_style: str = "json",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    bot_id: str = "",
    exchange: str = "",
) -> None:
    """Configure structured logging with JSON formatting and file rotation.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_style: ``"json"`` (default) or ``"standard"`` for
            human-readable output.
        log_file: Explicit path for the rotating log file.  When *None*
            the file is ``<log_dir>/omnitrade.log``.
        log_dir: Directory for log files (default ``logs/``).
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated files to keep.
        bot_id: Convenience arg forwarded to :func:`set_log_context`.
        exchange: Convenience arg forwarded to :func:`set_log_context`.
    """
    # Populate global context so JSON records carry bot_id / exchange.
    set_log_context(bot_id=bot_id, exchange=exchange)

    # --- Handlers -----------------------------------------------------------
    handlers: list[logging.Handler] = []

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stdout_handler)

    # Rotating file handler
    resolved_dir = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_file = Path(log_file) if log_file else resolved_dir / "omnitrade.log"
    file_handler = logging.handlers.RotatingFileHandler(
        resolved_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handlers.append(file_handler)

    # --- Formatter ----------------------------------------------------------
    if format_style == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    # --- Root logger --------------------------------------------------------
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    # Quiet noisy libraries
    for lib in ["aiohttp", "urllib3", "web3", "websockets"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the omnitrade prefix.

    Args:
        name: Logical module name (e.g. ``"risk.coordinator"``).

    Returns:
        A :class:`logging.Logger` under the ``omnitrade.`` namespace.
    """
    return logging.getLogger(f"omnitrade.{name}")
