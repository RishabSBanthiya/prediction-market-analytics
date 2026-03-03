"""Structured logging setup for omnitrade."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_style: str = "standard",
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_style: "standard" or "json"
        log_file: Optional file path for log output
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )

    # Quiet noisy libraries
    for lib in ["aiohttp", "urllib3", "web3", "websockets"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the omnitrade prefix."""
    return logging.getLogger(f"omnitrade.{name}")
