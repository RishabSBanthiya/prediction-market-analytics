"""
Abstract storage backend interface.

All storage implementations inherit from StorageBackend.
Supports multi-exchange position tracking, reservations, and agent management.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Generator


class StorageBackend(ABC):
    """
    Abstract storage backend.

    All operations should be transactional where possible.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Create tables/schema if needed."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

    # ==================== BALANCE ====================

    @abstractmethod
    def get_balance(self, exchange: str, account_id: str) -> float:
        """Get cached balance for an exchange account."""
        pass

    @abstractmethod
    def update_balance(self, exchange: str, account_id: str, balance: float) -> None:
        """Update cached balance."""
        pass

    # ==================== AGENTS ====================

    @abstractmethod
    def register_agent(self, agent_id: str, agent_type: str, exchange: str) -> bool:
        """Register a bot agent. Returns False if already active."""
        pass

    @abstractmethod
    def update_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp."""
        pass

    @abstractmethod
    def set_agent_status(self, agent_id: str, status: str) -> None:
        """Update agent status."""
        pass

    @abstractmethod
    def cleanup_stale_agents(self, stale_threshold_seconds: int) -> int:
        """Mark stale agents as crashed. Returns count."""
        pass

    # ==================== RESERVATIONS ====================

    @abstractmethod
    def create_reservation(
        self,
        agent_id: str,
        exchange: str,
        instrument_id: str,
        amount_usd: float,
        expires_at: datetime,
    ) -> str:
        """Create a capital reservation. Returns reservation_id."""
        pass

    @abstractmethod
    def mark_reservation_executed(self, reservation_id: str, filled_amount: float) -> None:
        pass

    @abstractmethod
    def release_reservation(self, reservation_id: str) -> None:
        pass

    @abstractmethod
    def cleanup_expired_reservations(self) -> int:
        """Expire old reservations. Returns count."""
        pass

    @abstractmethod
    def get_reserved_amount(self, exchange: str, account_id: str) -> float:
        """Total reserved capital for an exchange account."""
        pass

    # ==================== POSITIONS ====================

    @abstractmethod
    def create_position(
        self,
        agent_id: str,
        exchange: str,
        instrument_id: str,
        side: str,
        size: float,
        entry_price: float,
    ) -> int:
        """Create a position record. Returns position_id."""
        pass

    @abstractmethod
    def get_agent_positions(self, agent_id: str, status: str = "open") -> list[dict]:
        """Get positions for an agent."""
        pass

    @abstractmethod
    def get_exchange_positions(self, exchange: str, status: str = "open") -> list[dict]:
        """Get all positions on an exchange."""
        pass

    @abstractmethod
    def close_position(self, position_id: int, exit_price: float, exit_reason: str) -> None:
        """Close a position."""
        pass

    @abstractmethod
    def update_position_price(self, position_id: int, current_price: float) -> None:
        pass

    @abstractmethod
    def update_position_exit_state(
        self,
        position_id: int,
        current_price: float,
        peak_price: float,
        trough_price: float,
        trailing_stop_activated: bool,
        trailing_stop_level: float,
    ) -> None:
        """Atomically persist current price and exit monitor state."""
        pass

    # ==================== EXPOSURE ====================

    @abstractmethod
    def get_total_exposure(self, exchange: str, account_id: str) -> float:
        """Total exposure (positions + reservations) for an account."""
        pass

    @abstractmethod
    def get_agent_exposure(self, agent_id: str) -> float:
        """Total exposure for an agent across exchanges."""
        pass

    @abstractmethod
    def get_instrument_exposure(self, exchange: str, instrument_id: str) -> float:
        """Total exposure on a specific instrument."""
        pass

    # ==================== EXECUTIONS ====================

    @abstractmethod
    def log_execution(
        self,
        agent_id: str,
        exchange: str,
        instrument_id: str,
        side: str,
        size: float,
        price: float,
        order_id: str,
        fees: float = 0.0,
    ) -> None:
        """Log a trade execution."""
        pass

    @abstractmethod
    def get_executions(
        self,
        agent_id: Optional[str] = None,
        exchange: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[dict]:
        """Get execution history."""
        pass
