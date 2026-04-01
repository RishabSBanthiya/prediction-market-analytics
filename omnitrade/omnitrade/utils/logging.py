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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Thread-local context that bot runners inject at startup so every
# log record includes bot_id and exchange without callers threading it.
# Uses threading.local() for safety since asyncio.to_thread is used.
# ---------------------------------------------------------------------------
_log_context: threading.local = threading.local()


def set_log_context(*, bot_id: str = "", exchange: str = "") -> None:
    """Set thread-local context fields that are added to every JSON log entry.

    Args:
        bot_id: The agent/bot identifier (e.g. ``directional-kalshi``).
        exchange: The exchange name (e.g. ``kalshi``, ``polymarket``).
    """
    if bot_id:
        _log_context.bot_id = bot_id
    if exchange:
        _log_context.exchange = exchange


def get_log_context() -> dict[str, str]:
    """Return a copy of the current thread-local log context."""
    ctx: dict[str, str] = {}
    if hasattr(_log_context, "bot_id"):
        ctx["bot_id"] = _log_context.bot_id
    if hasattr(_log_context, "exchange"):
        ctx["exchange"] = _log_context.exchange
    return ctx


def clear_log_context() -> None:
    """Remove all fields from the thread-local log context."""
    for attr in ("bot_id", "exchange"):
        try:
            delattr(_log_context, attr)
        except AttributeError:
            pass


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
            "bot_id": getattr(_log_context, "bot_id", ""),
            "exchange": getattr(_log_context, "exchange", ""),
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
) -> None:
    """Configure structured logging with JSON formatting and optional file rotation.

    Call :func:`set_log_context` **before** this function if you need
    ``bot_id`` / ``exchange`` fields in the first log lines.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_style: ``"json"`` (default) or ``"standard"`` for
            human-readable output.
        log_file: Explicit path for the rotating log file.  When *None*
            no file handler is created (stdout only).
        log_dir: Parent directory for *log_file* when the file's own
            directory does not exist yet.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated files to keep.
    """

    # --- Handlers -----------------------------------------------------------
    handlers: list[logging.Handler] = []

    # Stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers.append(stdout_handler)

    # Rotating file handler — only created when a log_file path is provided,
    # preserving the original behaviour (the old code only added a file
    # handler when ``log_file`` was explicitly passed).
    if log_file is not None:
        resolved_dir = Path(log_dir) if log_dir else Path(log_file).parent
        resolved_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
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
