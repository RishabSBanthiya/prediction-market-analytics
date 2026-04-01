"""Tests for structured JSON logging."""

import json
import logging
import logging.handlers
import os
import tempfile
from pathlib import Path

import pytest

from omnitrade.utils.logging import (
    JSONFormatter,
    clear_log_context,
    get_log_context,
    get_logger,
    set_log_context,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_log_context():
    """Clear thread-local log context between tests."""
    clear_log_context()
    yield
    clear_log_context()


@pytest.fixture()
def _reset_root_logger():
    """Remove handlers added by setup_logging so tests don't leak."""
    yield
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()


class TestJSONFormatter:
    """Tests for the JSONFormatter class."""

    def test_basic_format_produces_valid_json(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello world",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["module"] == "omnitrade.test"
        assert parsed["message"] == "hello world"
        assert "timestamp" in parsed

    def test_format_includes_context_fields(self):
        set_log_context(bot_id="dir-kalshi", exchange="kalshi")
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.risk",
            level=logging.WARNING,
            pathname="risk.py",
            lineno=10,
            msg="limit breached",
            args=None,
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))

        assert parsed["bot_id"] == "dir-kalshi"
        assert parsed["exchange"] == "kalshi"
        assert parsed["level"] == "WARNING"

    def test_format_without_context_gives_empty_strings(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="no context",
            args=None,
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["bot_id"] == ""
        assert parsed["exchange"] == ""

    def test_format_includes_exception_info(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="something broke",
            args=None,
            exc_info=exc_info,
        )
        parsed = json.loads(formatter.format(record))
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]

    def test_format_with_percent_style_args(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="price is %.4f on %s",
            args=(0.7512, "kalshi"),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["message"] == "price is 0.7512 on kalshi"

    def test_timestamp_is_iso8601_utc(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="ts check",
            args=None,
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        ts = parsed["timestamp"]
        # Should end with +00:00 (UTC)
        assert "+00:00" in ts

    def test_one_json_object_per_line(self):
        """Each formatted record must be a single line (no embedded newlines)."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="omnitrade.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="line one\nline two",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        # json.dumps escapes newlines inside strings, so the output itself
        # should be a single line.
        assert "\n" not in output


class TestSetLogContext:

    def test_set_and_get_context(self):
        set_log_context(bot_id="mm-poly", exchange="polymarket")
        ctx = get_log_context()
        assert ctx == {"bot_id": "mm-poly", "exchange": "polymarket"}

    def test_partial_update_preserves_existing(self):
        set_log_context(bot_id="a")
        set_log_context(exchange="kalshi")
        ctx = get_log_context()
        assert ctx["bot_id"] == "a"
        assert ctx["exchange"] == "kalshi"

    def test_get_returns_copy(self):
        set_log_context(bot_id="x")
        ctx = get_log_context()
        ctx["bot_id"] = "mutated"
        assert get_log_context()["bot_id"] == "x"


class TestSetupLogging:

    @pytest.mark.usefixtures("_reset_root_logger")
    def test_creates_log_file_when_path_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test_logs", "omnitrade.log")
            set_log_context(bot_id="test-bot", exchange="kalshi")
            setup_logging(
                level="DEBUG",
                format_style="json",
                log_file=log_path,
            )

            logger = logging.getLogger("omnitrade.test.setup")
            logger.info("hello from test")

            # Flush handlers
            for h in logging.getLogger().handlers:
                h.flush()

            assert Path(log_path).exists()

            content = Path(log_path).read_text().strip()
            assert content  # not empty
            parsed = json.loads(content.split("\n")[-1])
            assert parsed["bot_id"] == "test-bot"
            assert parsed["exchange"] == "kalshi"
            assert parsed["message"] == "hello from test"

    @pytest.mark.usefixtures("_reset_root_logger")
    def test_no_file_handler_when_log_file_omitted(self):
        """When log_file is not provided, no RotatingFileHandler is created."""
        setup_logging(level="INFO", format_style="json")
        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert file_handlers == []

    @pytest.mark.usefixtures("_reset_root_logger")
    def test_standard_format_style(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "omnitrade.log")
            setup_logging(
                level="INFO",
                format_style="standard",
                log_file=log_path,
            )
            logger = logging.getLogger("omnitrade.test.standard")
            logger.info("plain text")

            for h in logging.getLogger().handlers:
                h.flush()

            content = Path(log_path).read_text().strip()
            # Standard format should NOT be JSON
            with pytest.raises(json.JSONDecodeError):
                json.loads(content.split("\n")[-1])
            assert "plain text" in content

    @pytest.mark.usefixtures("_reset_root_logger")
    def test_explicit_log_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom.log")
            setup_logging(
                level="INFO",
                log_file=custom_path,
            )
            logger = logging.getLogger("omnitrade.test.custom")
            logger.info("custom path")

            for h in logging.getLogger().handlers:
                h.flush()

            assert Path(custom_path).exists()

    @pytest.mark.usefixtures("_reset_root_logger")
    def test_noisy_libraries_suppressed(self):
        setup_logging(level="DEBUG")

        for lib in ["aiohttp", "urllib3", "web3", "websockets"]:
            assert logging.getLogger(lib).level == logging.WARNING


class TestClearLogContext:

    def test_clear_removes_all_fields(self):
        set_log_context(bot_id="x", exchange="y")
        clear_log_context()
        assert get_log_context() == {}

    def test_clear_is_idempotent(self):
        clear_log_context()
        clear_log_context()
        assert get_log_context() == {}


class TestThreadSafety:

    def test_context_is_thread_local(self):
        """Context set in one thread must not leak into another."""
        import threading

        set_log_context(bot_id="main-thread", exchange="poly")
        child_ctx: dict[str, str] = {}

        def _child():
            # Child thread should see an empty context
            child_ctx.update(get_log_context())

        t = threading.Thread(target=_child)
        t.start()
        t.join()

        # Main thread context unchanged
        assert get_log_context() == {"bot_id": "main-thread", "exchange": "poly"}
        # Child thread saw no context
        assert child_ctx == {}


class TestGetLogger:

    def test_prefixes_name(self):
        logger = get_logger("risk.coordinator")
        assert logger.name == "omnitrade.risk.coordinator"

    def test_returns_standard_logger(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)
