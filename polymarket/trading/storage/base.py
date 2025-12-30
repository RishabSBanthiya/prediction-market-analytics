"""
Abstract storage backend interface.

All storage implementations must inherit from StorageBackend and implement
the required methods. This allows swapping between SQLite, Redis, etc.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import (
        Position, 
        Reservation, 
        AgentInfo, 
        WalletState,
        ReservationStatus,
        PositionStatus,
        AgentStatus,
    )


class StorageTransaction(ABC):
    """
    Abstract transaction interface.
    
    All operations within a transaction are atomic - either all succeed
    or all fail together.
    """
    
    # ==================== WALLET STATE ====================
    
    @abstractmethod
    def get_wallet_state(self, wallet_address: str) -> "WalletState":
        """Get current wallet state including positions and reservations"""
        pass
    
    @abstractmethod
    def get_usdc_balance(self, wallet_address: str) -> float:
        """Get cached USDC balance (call update_usdc_balance to refresh)"""
        pass
    
    @abstractmethod
    def update_usdc_balance(self, wallet_address: str, balance: float) -> None:
        """Update cached USDC balance"""
        pass
    
    # ==================== AGENTS ====================
    
    @abstractmethod
    def register_agent(self, agent_id: str, agent_type: str, wallet_address: str) -> bool:
        """
        Register a new agent.
        
        Returns False if agent_id already exists and is active.
        """
        pass
    
    @abstractmethod
    def update_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat timestamp"""
        pass
    
    @abstractmethod
    def get_agent(self, agent_id: str) -> Optional["AgentInfo"]:
        """Get agent info by ID"""
        pass
    
    @abstractmethod
    def get_all_agents(self, wallet_address: Optional[str] = None) -> List["AgentInfo"]:
        """Get all agents, optionally filtered by wallet"""
        pass
    
    @abstractmethod
    def update_agent_status(self, agent_id: str, status: "AgentStatus") -> None:
        """Update agent status"""
        pass
    
    @abstractmethod
    def cleanup_stale_agents(self, stale_threshold_seconds: int) -> int:
        """Mark agents as crashed if no heartbeat, returns count"""
        pass
    
    # ==================== RESERVATIONS ====================
    
    @abstractmethod
    def create_reservation(
        self,
        agent_id: str,
        market_id: str,
        token_id: str,
        amount_usd: float,
        expires_at: datetime
    ) -> str:
        """Create a capital reservation, returns reservation_id"""
        pass
    
    @abstractmethod
    def get_reservation(self, reservation_id: str) -> Optional["Reservation"]:
        """Get reservation by ID"""
        pass
    
    @abstractmethod
    def get_agent_reservations(self, agent_id: str) -> List["Reservation"]:
        """Get all active reservations for an agent"""
        pass
    
    @abstractmethod
    def get_all_reservations(
        self, 
        wallet_address: Optional[str] = None,
        status: Optional["ReservationStatus"] = None
    ) -> List["Reservation"]:
        """Get all reservations, optionally filtered"""
        pass
    
    @abstractmethod
    def mark_reservation_executed(
        self, 
        reservation_id: str, 
        filled_amount: float
    ) -> None:
        """Mark reservation as executed"""
        pass
    
    @abstractmethod
    def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation (trade cancelled/failed)"""
        pass
    
    @abstractmethod
    def release_all_reservations(self, agent_id: Optional[str] = None) -> int:
        """Release all reservations, optionally for specific agent. Returns count."""
        pass
    
    @abstractmethod
    def cleanup_expired_reservations(self) -> int:
        """Mark expired reservations as expired, returns count"""
        pass
    
    # ==================== POSITIONS ====================
    
    @abstractmethod
    def create_position(
        self,
        agent_id: str,
        market_id: str,
        token_id: str,
        outcome: str,
        shares: float,
        entry_price: float
    ) -> int:
        """Create a new position, returns position ID"""
        pass
    
    @abstractmethod
    def get_position(self, position_id: int) -> Optional["Position"]:
        """Get position by ID"""
        pass
    
    @abstractmethod
    def get_agent_positions(
        self, 
        agent_id: str,
        status: Optional["PositionStatus"] = None
    ) -> List["Position"]:
        """Get positions for an agent"""
        pass
    
    @abstractmethod
    def get_all_positions(
        self,
        wallet_address: Optional[str] = None,
        status: Optional["PositionStatus"] = None
    ) -> List["Position"]:
        """Get all positions, optionally filtered"""
        pass
    
    @abstractmethod
    def update_position_price(self, position_id: int, current_price: float) -> None:
        """Update current price for a position"""
        pass
    
    @abstractmethod
    def mark_position_closed(self, position_id: int) -> None:
        """Mark position as closed"""
        pass
    
    @abstractmethod
    def mark_position_closed_by_token(self, wallet_address: str, token_id: str) -> int:
        """Mark all positions with the given token_id as closed. Returns count."""
        pass
    
    @abstractmethod
    def add_orphan_position(
        self,
        wallet_address: str,
        token_id: str,
        market_id: str,
        shares: float,
        current_price: float
    ) -> int:
        """Add an orphan position (found on-chain but not in DB)"""
        pass
    
    # ==================== EXPOSURE CALCULATIONS ====================
    
    @abstractmethod
    def get_total_exposure(self, wallet_address: str) -> float:
        """Get total exposure (positions + reservations) for wallet"""
        pass
    
    @abstractmethod
    def get_agent_exposure(self, agent_id: str) -> float:
        """Get total exposure for an agent"""
        pass
    
    @abstractmethod
    def get_market_exposure(self, market_id: str, wallet_address: str) -> float:
        """Get exposure in a specific market"""
        pass
    
    # ==================== RATE LIMITING ====================
    
    @abstractmethod
    def log_request(self, agent_id: str, endpoint: str, timestamp: datetime) -> None:
        """Log an API request for rate limiting"""
        pass
    
    @abstractmethod
    def count_requests_since(self, since: datetime) -> int:
        """Count requests since timestamp"""
        pass
    
    @abstractmethod
    def cleanup_old_requests(self, before: datetime) -> int:
        """Remove old request logs, returns count deleted"""
        pass
    
    # ==================== EXECUTION HISTORY ====================
    
    @abstractmethod
    def save_execution(
        self,
        agent_id: str,
        market_id: str,
        token_id: str,
        side: "Side",
        shares: float,
        price: float,
        filled_price: float,
        signal_score: float,
        success: bool,
        error_message: Optional[str] = None
    ) -> int:
        """Save execution result, returns execution ID"""
        pass
    
    @abstractmethod
    def get_executions(
        self,
        agent_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        wallet_address: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """Get execution history with optional filters"""
        pass
    
    # ==================== FLOW ALERTS ====================
    
    @abstractmethod
    def save_alert(
        self,
        alert_type: str,
        market_id: str,
        token_id: str,
        question: str,
        timestamp: datetime,
        severity: str,
        reason: str,
        details: dict,
        category: str,
        score: Optional[float] = None
    ) -> int:
        """Save flow alert, returns alert ID"""
        pass
    
    @abstractmethod
    def get_alerts(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """Get alerts with optional filters"""
        pass
    
    @abstractmethod
    def get_alert_stats(self) -> dict:
        """Get alert statistics"""
        pass


class StorageBackend(ABC):
    """
    Abstract storage backend.
    
    Implementations must provide transactional access to storage.
    All operations should be performed within a transaction context.
    """
    
    @abstractmethod
    @contextmanager
    def transaction(self) -> Generator[StorageTransaction, None, None]:
        """
        Context manager for atomic transactions.
        
        Usage:
            with storage.transaction() as txn:
                txn.create_reservation(...)
                txn.update_heartbeat(...)
            # Committed on successful exit, rolled back on exception
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize storage (create tables, etc.)"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close storage connections"""
        pass
    
    # Convenience methods that wrap transactions
    
    def get_wallet_state(self, wallet_address: str) -> "WalletState":
        """Get current wallet state"""
        with self.transaction() as txn:
            return txn.get_wallet_state(wallet_address)
    
    def register_agent(self, agent_id: str, agent_type: str, wallet_address: str) -> bool:
        """Register a new agent"""
        with self.transaction() as txn:
            return txn.register_agent(agent_id, agent_type, wallet_address)
    
    def update_heartbeat(self, agent_id: str) -> None:
        """Update agent heartbeat"""
        with self.transaction() as txn:
            txn.update_heartbeat(agent_id)
    
    def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation"""
        with self.transaction() as txn:
            txn.release_reservation(reservation_id)
    
    def mark_position_closed_by_token(self, wallet_address: str, token_id: str) -> int:
        """Mark all positions with the given token_id as closed. Returns count."""
        with self.transaction() as txn:
            return txn.mark_position_closed_by_token(wallet_address, token_id)


