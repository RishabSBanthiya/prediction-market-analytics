"""
Multi-exchange risk coordinator.

Manages capital allocation across multiple exchanges and bot agents.
Ported from polymarket-analytics with multi-exchange support.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from ..core.enums import ExchangeId
from ..core.config import RiskConfig
from ..core.errors import RiskLimitError, InsufficientBalanceError
from ..storage.base import StorageBackend
from .safety import CircuitBreaker, DrawdownLimit, TradingHalt

logger = logging.getLogger(__name__)


class RiskCoordinator:
    """
    Central risk management for multi-exchange trading.

    Responsibilities:
    - Atomic capital reservation with limit checks
    - Per-exchange, per-agent, per-instrument exposure limits
    - Circuit breaker and drawdown monitoring
    - Agent lifecycle management
    """

    def __init__(self, storage: StorageBackend, risk_config: RiskConfig):
        self.storage = storage
        self.config = risk_config

        # Safety components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=risk_config.circuit_breaker_failures,
            reset_timeout_seconds=risk_config.circuit_breaker_reset_seconds,
        )
        self.drawdown_limit = DrawdownLimit(
            max_daily_drawdown_pct=risk_config.max_daily_drawdown_pct,
            max_total_drawdown_pct=risk_config.max_total_drawdown_pct,
        )
        self.trading_halt = TradingHalt()

        self._account_ids: dict[ExchangeId, str] = {}

    def register_account(self, exchange: ExchangeId, account_id: str) -> None:
        """Register an account for an exchange."""
        self._account_ids[exchange] = account_id

    def startup(self, agent_id: str, agent_type: str, exchange: ExchangeId) -> bool:
        """Register a bot agent. Returns True if successfully registered."""
        return self.storage.register_agent(agent_id, agent_type, exchange.value)

    def heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat."""
        self.storage.update_heartbeat(agent_id)

    def shutdown(self, agent_id: str) -> None:
        """Mark agent as stopped."""
        self.storage.set_agent_status(agent_id, "stopped")

    def atomic_reserve(
        self,
        agent_id: str,
        exchange: ExchangeId,
        instrument_id: str,
        amount_usd: float,
    ) -> str:
        """
        Atomically reserve capital with all limit checks.

        Checks (all in one operation):
        1. Trade value within min/max bounds
        2. Total wallet exposure within limit
        3. Per-agent exposure within limit
        4. Per-instrument exposure within limit
        5. Sufficient available balance

        Returns reservation_id on success.
        Raises RiskLimitError or InsufficientBalanceError on failure.
        """
        # Safety checks first
        if self.trading_halt.is_halted:
            raise RiskLimitError("trading_halt", f"Trading halted: {self.trading_halt.reasons}")

        if not self.circuit_breaker.can_execute():
            raise RiskLimitError("circuit_breaker", "Circuit breaker is OPEN")

        if self.drawdown_limit.is_breached:
            raise RiskLimitError("drawdown", self.drawdown_limit.breach_reason or "Drawdown limit breached")

        # Trade size check
        if amount_usd < self.config.min_trade_value_usd:
            raise RiskLimitError("min_trade", f"${amount_usd:.2f} < min ${self.config.min_trade_value_usd:.2f}")
        if amount_usd > self.config.max_trade_value_usd:
            raise RiskLimitError("max_trade", f"${amount_usd:.2f} > max ${self.config.max_trade_value_usd:.2f}")

        account_id = self._account_ids.get(exchange, "")

        # Get current balance
        balance = self.storage.get_balance(exchange.value, account_id)
        if balance <= 0:
            raise InsufficientBalanceError(exchange.value, amount_usd, balance)

        total_equity = balance  # Simplified; in prod would include position values

        # Wallet exposure check
        current_exposure = self.storage.get_total_exposure(exchange.value, account_id)
        new_exposure = current_exposure + amount_usd
        if new_exposure / total_equity > self.config.max_wallet_exposure_pct:
            raise RiskLimitError(
                "wallet_exposure",
                f"Would be {new_exposure/total_equity:.1%} > {self.config.max_wallet_exposure_pct:.1%}"
            )

        # Agent exposure check
        agent_exposure = self.storage.get_agent_exposure(agent_id)
        new_agent_exposure = agent_exposure + amount_usd
        if new_agent_exposure / total_equity > self.config.max_per_agent_exposure_pct:
            raise RiskLimitError(
                "agent_exposure",
                f"Agent would be {new_agent_exposure/total_equity:.1%} > {self.config.max_per_agent_exposure_pct:.1%}"
            )

        # Instrument exposure check
        inst_exposure = self.storage.get_instrument_exposure(exchange.value, instrument_id)
        new_inst_exposure = inst_exposure + amount_usd
        if new_inst_exposure / total_equity > self.config.max_per_market_exposure_pct:
            raise RiskLimitError(
                "instrument_exposure",
                f"Instrument would be {new_inst_exposure/total_equity:.1%} > {self.config.max_per_market_exposure_pct:.1%}"
            )

        # Available capital check
        reserved = self.storage.get_reserved_amount(exchange.value, account_id)
        available = balance - reserved
        if amount_usd > available:
            raise InsufficientBalanceError(exchange.value, amount_usd, available)

        # All checks passed - create reservation
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.config.reservation_ttl_seconds)
        reservation_id = self.storage.create_reservation(
            agent_id=agent_id,
            exchange=exchange.value,
            instrument_id=instrument_id,
            amount_usd=amount_usd,
            expires_at=expires_at,
        )

        logger.info(f"Reserved ${amount_usd:.2f} for {agent_id} on {exchange.value}/{instrument_id}")
        return reservation_id

    def confirm_execution(
        self,
        reservation_id: str,
        agent_id: str,
        exchange: ExchangeId,
        instrument_id: str,
        side: str,
        size: float,
        price: float,
        order_id: str = "",
        fees: float = 0.0,
    ) -> int:
        """
        Confirm a trade executed against a reservation.
        Creates position record and logs execution.
        Returns position_id.
        """
        filled_amount = size * price
        self.storage.mark_reservation_executed(reservation_id, filled_amount)

        position_id = self.storage.create_position(
            agent_id=agent_id,
            exchange=exchange.value,
            instrument_id=instrument_id,
            side=side,
            size=size,
            entry_price=price,
        )

        self.storage.log_execution(
            agent_id=agent_id,
            exchange=exchange.value,
            instrument_id=instrument_id,
            side=side,
            size=size,
            price=price,
            order_id=order_id,
            fees=fees,
        )

        self.circuit_breaker.record_success()
        logger.info(f"Confirmed: {side} {size:.4f} @ ${price:.4f} on {exchange.value}")
        return position_id

    def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation (trade failed/cancelled)."""
        self.storage.release_reservation(reservation_id)

    def record_failure(self) -> None:
        """Record a trading failure for circuit breaker."""
        self.circuit_breaker.record_failure()

    def update_equity(self, current_equity: float) -> bool:
        """Update equity for drawdown tracking. Returns True if trading allowed."""
        return self.drawdown_limit.update(current_equity)

    def cleanup(self) -> None:
        """Periodic cleanup of stale agents and expired reservations."""
        stale = self.storage.cleanup_stale_agents(self.config.stale_agent_threshold_seconds)
        expired = self.storage.cleanup_expired_reservations()
        if stale or expired:
            logger.info(f"Cleanup: {stale} stale agents, {expired} expired reservations")
