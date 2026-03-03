"""Tests for safety modules."""

import pytest
from datetime import datetime, timezone, timedelta
from omnitrade.risk.safety import CircuitBreaker, DrawdownLimit, TradingHalt


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.can_execute()
        assert not cb.is_open
        assert cb.state == "CLOSED"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.can_execute()
        cb.record_failure()
        assert not cb.can_execute()
        assert cb.is_open

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        # Simulate timeout by manipulating last_failure_time
        cb.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=700)
        assert cb.can_execute()  # Half-open
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.can_execute()

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert not cb.can_execute()
        cb.reset()
        assert cb.can_execute()

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=10)
        cb.record_failure()
        assert not cb.can_execute()
        cb.last_failure_time = datetime.now(timezone.utc) - timedelta(seconds=11)
        assert cb.can_execute()
        assert cb.state == "HALF-OPEN"

    def test_seconds_until_reset_when_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.seconds_until_reset is None

    def test_seconds_until_reset_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_seconds=600)
        cb.record_failure()
        remaining = cb.seconds_until_reset
        assert remaining is not None
        assert 590 < remaining <= 600

    def test_failure_below_threshold_stays_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 2

    def test_record_success_from_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_success()
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestDrawdownLimit:
    def test_no_breach(self):
        dl = DrawdownLimit(max_daily_drawdown_pct=0.10, max_total_drawdown_pct=0.25)
        assert dl.update(1000)
        assert dl.update(990)
        assert not dl.is_breached

    def test_daily_breach(self):
        dl = DrawdownLimit(max_daily_drawdown_pct=0.05)
        dl.update(1000)  # Sets daily start
        result = dl.update(940)  # 6% drawdown
        assert not result
        assert dl.is_breached
        assert "DAILY" in dl.breach_reason

    def test_total_breach(self):
        dl = DrawdownLimit(max_daily_drawdown_pct=0.50, max_total_drawdown_pct=0.10)
        dl.update(1000)  # Peak
        result = dl.update(890)  # 11% from peak
        assert not result
        assert "TOTAL" in dl.breach_reason

    def test_recovery_clears_breach(self):
        dl = DrawdownLimit(max_daily_drawdown_pct=0.05)
        dl.update(1000)
        dl.update(940)  # Breach
        assert dl.is_breached
        # Manually reset
        dl.reset(1000)
        assert not dl.is_breached

    def test_drawdown_percentages(self):
        dl = DrawdownLimit()
        dl.update(1000)
        dl.update(950)
        assert abs(dl.daily_drawdown_pct - 0.05) < 0.001
        assert abs(dl.total_drawdown_pct - 0.05) < 0.001

    def test_initial_peak_is_zero(self):
        dl = DrawdownLimit()
        assert dl.peak_equity == 0.0
        assert dl.total_drawdown_pct == 0.0

    def test_peak_tracks_highest(self):
        dl = DrawdownLimit()
        dl.update(1000)
        dl.update(1050)
        dl.update(1020)
        assert dl.peak_equity == 1050

    def test_reset_clears_everything(self):
        dl = DrawdownLimit(max_daily_drawdown_pct=0.05)
        dl.update(1000)
        dl.update(940)
        assert dl.is_breached
        dl.reset(1000)
        assert not dl.is_breached
        assert dl.breach_reason is None
        assert dl.peak_equity == 1000


class TestTradingHalt:
    def test_not_halted_initially(self):
        th = TradingHalt()
        assert not th.is_halted

    def test_halt_with_reason(self):
        th = TradingHalt()
        th.add_reason("drawdown", "Daily limit breached")
        assert th.is_halted
        assert "drawdown" in th.reasons

    def test_multiple_reasons(self):
        th = TradingHalt()
        th.add_reason("drawdown", "Loss limit")
        th.add_reason("manual", "Operator halt")
        assert th.is_halted
        th.clear_reason("drawdown")
        assert th.is_halted  # Still halted (manual)
        th.clear_reason("manual")
        assert not th.is_halted

    def test_clear_all(self):
        th = TradingHalt()
        th.add_reason("a", "1")
        th.add_reason("b", "2")
        th.clear_all()
        assert not th.is_halted

    def test_reasons_returns_copy(self):
        th = TradingHalt()
        th.add_reason("key", "value")
        reasons = th.reasons
        reasons["extra"] = "should not affect original"
        assert "extra" not in th.reasons

    def test_clear_nonexistent_reason(self):
        th = TradingHalt()
        th.clear_reason("nonexistent")  # Should not raise
        assert not th.is_halted

    def test_overwrite_reason(self):
        th = TradingHalt()
        th.add_reason("key", "first")
        th.add_reason("key", "second")
        assert th.reasons["key"] == "second"
        assert th.is_halted
