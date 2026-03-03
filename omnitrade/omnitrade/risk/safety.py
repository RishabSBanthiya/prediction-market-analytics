"""
Safety components for trading bots.

Ported from polymarket-analytics. These are the last line of defense:
- CircuitBreaker: Stop trading on repeated failures
- DrawdownLimit: Stop trading on excessive losses
- TradingHalt: Manual/automatic trading halt with reasons
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for trading.

    States:
    - CLOSED: Normal operation, trades allowed
    - OPEN: Failures exceeded threshold, trades blocked
    - HALF-OPEN: Reset timeout passed, allow one test trade
    """

    failure_threshold: int = 3
    reset_timeout_seconds: int = 600
    failure_count: int = field(default=0, init=False)
    last_failure_time: Optional[datetime] = field(default=None, init=False)
    state: str = field(default="CLOSED", init=False)

    def record_success(self):
        self.failure_count = 0
        if self.state != "CLOSED":
            logger.info("Circuit breaker CLOSED (success after recovery)")
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                logger.error(f"CIRCUIT BREAKER OPEN: {self.failure_count} consecutive failures")
            self.state = "OPEN"
        else:
            logger.warning(f"Failure {self.failure_count}/{self.failure_threshold}")

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "HALF-OPEN":
            return True
        # OPEN - check reset timeout
        if self.last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
            if elapsed > self.reset_timeout_seconds:
                logger.info("Circuit breaker HALF-OPEN (reset timeout passed)")
                self.state = "HALF-OPEN"
                return True
        return False

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        logger.info("Circuit breaker manually reset")

    @property
    def is_open(self) -> bool:
        return self.state == "OPEN"

    @property
    def seconds_until_reset(self) -> Optional[float]:
        if self.state != "OPEN" or not self.last_failure_time:
            return None
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return max(0, self.reset_timeout_seconds - elapsed)


@dataclass
class DrawdownLimit:
    """
    Drawdown-based trading limit.

    Stops trading when losses exceed thresholds:
    - Daily drawdown: Losses from start of day
    - Total drawdown: Losses from peak equity
    """

    max_daily_drawdown_pct: float = 0.05
    max_total_drawdown_pct: float = 0.15

    daily_start_equity: Optional[float] = field(default=None, init=False)
    daily_start_date: Optional[datetime] = field(default=None, init=False)
    peak_equity: float = field(default=0.0, init=False)
    current_equity: float = field(default=0.0, init=False)
    is_breached: bool = field(default=False, init=False)
    breach_reason: Optional[str] = field(default=None, init=False)

    def update(self, current_equity: float) -> bool:
        """Update equity and check limits. Returns True if trading allowed."""
        now = datetime.now(timezone.utc)
        self.current_equity = current_equity

        # Reset daily at midnight UTC
        if self.daily_start_date is None or now.date() != self.daily_start_date.date():
            self.daily_start_equity = current_equity
            self.daily_start_date = now
            logger.info(f"Daily drawdown reset: starting equity ${current_equity:.2f}")

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Check daily
        if self.daily_start_equity and self.daily_start_equity > 0:
            daily_dd = (self.daily_start_equity - current_equity) / self.daily_start_equity
            if daily_dd > self.max_daily_drawdown_pct:
                self.is_breached = True
                self.breach_reason = f"DAILY DRAWDOWN: {daily_dd:.1%} > {self.max_daily_drawdown_pct:.1%}"
                logger.error(self.breach_reason)
                return False

        # Check total
        if self.peak_equity > 0:
            total_dd = (self.peak_equity - current_equity) / self.peak_equity
            if total_dd > self.max_total_drawdown_pct:
                self.is_breached = True
                self.breach_reason = f"TOTAL DRAWDOWN: {total_dd:.1%} > {self.max_total_drawdown_pct:.1%}"
                logger.error(self.breach_reason)
                return False

        if self.is_breached:
            self.is_breached = False
            self.breach_reason = None
            logger.info("Drawdown limits cleared (equity recovered)")

        return True

    def reset(self, current_equity: Optional[float] = None):
        equity = current_equity or self.current_equity
        self.daily_start_equity = equity
        self.daily_start_date = datetime.now(timezone.utc)
        self.peak_equity = equity
        self.is_breached = False
        self.breach_reason = None
        logger.info(f"Drawdown limits manually reset: equity ${equity:.2f}")

    @property
    def daily_drawdown_pct(self) -> float:
        if not self.daily_start_equity or self.daily_start_equity <= 0:
            return 0.0
        return max(0, (self.daily_start_equity - self.current_equity) / self.daily_start_equity)

    @property
    def total_drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0, (self.peak_equity - self.current_equity) / self.peak_equity)


class TradingHalt:
    """
    Composite trading halt with multiple reasons.

    Multiple systems can independently halt trading.
    Trading resumes only when ALL reasons are cleared.
    """

    def __init__(self):
        self._reasons: dict[str, str] = {}

    def add_reason(self, key: str, message: str) -> None:
        self._reasons[key] = message
        logger.warning(f"Trading HALTED [{key}]: {message}")

    def clear_reason(self, key: str) -> None:
        if key in self._reasons:
            del self._reasons[key]
            if not self._reasons:
                logger.info("All halt reasons cleared - trading resumed")

    @property
    def is_halted(self) -> bool:
        return len(self._reasons) > 0

    @property
    def reasons(self) -> dict[str, str]:
        return dict(self._reasons)

    def clear_all(self) -> None:
        self._reasons.clear()
        logger.info("All halt reasons force-cleared")
