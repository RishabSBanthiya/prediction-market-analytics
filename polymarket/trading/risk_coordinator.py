"""
Multi-Agent Risk Coordinator.

Centralized risk management for multiple trading agents sharing the same wallet.
Provides:
- Atomic capital reservation (no race conditions)
- API-first state reconciliation (Polymarket API = source of truth)
- Agent heartbeat monitoring
- Exposure limit enforcement
- Discrepancy detection between API and transaction history

Architecture:
- api_positions table: SOURCE OF TRUTH (synced from Polymarket API)
- transactions table: AUDIT LOG (synced from chain for validation)
- reconciliation_issues table: Logs discrepancies for investigation
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING

from ..core.models import Position, WalletState, PositionStatus, ReservationStatus
from ..core.config import RiskConfig, Config, get_config
from ..core.api import PolymarketAPI
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteStorage
from .chain_sync import ChainSyncService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RiskCoordinator:
    """
    Centralized risk management for multi-agent trading.
    
    All agents MUST use the same RiskCoordinator instance (or at least
    the same storage backend) to ensure proper coordination.
    
    Usage:
        coordinator = RiskCoordinator(config=Config.from_env())
        await coordinator.startup("my-agent", "bond")
        
        # Before trading:
        reservation_id = coordinator.atomic_reserve(
            "my-agent", market_id, token_id, amount_usd
        )
        if reservation_id:
            try:
                result = await execute_trade(...)
                coordinator.confirm_execution(reservation_id, result.shares, result.price)
            except:
                coordinator.release_reservation(reservation_id)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        storage: Optional[StorageBackend] = None,
        api: Optional[PolymarketAPI] = None
    ):
        self.config = config or get_config()
        self.storage = storage or SQLiteStorage(self.config.db_path)
        self.api = api
        self.wallet_address = self.config.proxy_address or ""
        self._reconciled = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._validation_task: Optional[asyncio.Task] = None
        self._current_agent_id: Optional[str] = None
        self._chain_sync: Optional[ChainSyncService] = None
        self._current_prices: Dict[str, float] = {}  # Cache for current prices
        self._validation_interval_seconds = 300  # Validate every 5 minutes
    
    async def startup(self, agent_id: str, agent_type: str) -> bool:
        """
        Initialize coordinator for an agent.
        
        MUST be called before any trading operations.
        
        Steps:
        1. Register agent
        2. Reconcile DB state with on-chain reality
        3. Start heartbeat loop
        
        Returns True if startup successful.
        """
        if not self.wallet_address:
            logger.error("No wallet address configured")
            return False
        
        # Initialize API if not provided
        if self.api is None:
            self.api = PolymarketAPI(self.config)
            await self.api.connect()
        
        # Register agent (or restart if already exists)
        try:
            with self.storage.transaction() as txn:
                if not txn.register_agent(agent_id, agent_type, self.wallet_address):
                    logger.error(f"Failed to register agent {agent_id}")
                    return False
        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {e}")
            return False
        
        self._current_agent_id = agent_id

        # Reconcile state
        await self._reconcile_state()
        self._reconciled = True

        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(agent_id)
        )

        # Start background validation loop
        self._validation_task = asyncio.create_task(
            self._background_validation_loop()
        )

        logger.info(f"RiskCoordinator started for agent {agent_id} ({agent_type})")
        return True

    async def shutdown(self):
        """Shutdown coordinator gracefully"""
        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Cancel validation task
        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass

        # Mark agent as stopped
        if self._current_agent_id:
            from ..core.models import AgentStatus
            with self.storage.transaction() as txn:
                txn.update_agent_status(self._current_agent_id, AgentStatus.STOPPED)

        # Close chain sync service
        if self._chain_sync:
            await self._chain_sync.close()

        # Close API
        if self.api:
            await self.api.close()

        logger.info("RiskCoordinator shutdown complete")
    
    async def _reconcile_state(self):
        """
        API-FIRST RECONCILIATION: Uses Polymarket API as source of truth.

        This method:
        1. Fetches current positions from Polymarket API (SOURCE OF TRUTH)
        2. Stores them in api_positions table
        3. Fetches USDC balance from chain
        4. Runs chain sync in background for audit trail
        5. Detects and logs any discrepancies between API and computed positions
        """
        logger.info(f"{'='*60}")
        logger.info("RECONCILING STATE (API-FIRST)...")
        logger.info(f"{'='*60}")
        logger.info(f"  Wallet: {self.wallet_address}")

        # STEP 1: Fetch positions from Polymarket API (SOURCE OF TRUTH)
        logger.info("  [1/5] Fetching positions from Polymarket API...")
        try:
            api_positions = await self.api.fetch_positions(self.wallet_address)
            logger.info(f"        Found {len(api_positions)} positions from API")
        except Exception as e:
            logger.error(f"        Failed to fetch API positions: {e}")
            api_positions = []

        # STEP 2: Store API positions as source of truth
        logger.info("  [2/5] Storing API positions as source of truth...")
        with self.storage.transaction() as txn:
            positions_synced = txn.upsert_api_positions(self.wallet_address, api_positions)
            logger.info(f"        Synced {positions_synced} positions to api_positions table")

        # Update current prices cache from API positions
        self._current_prices = {
            p.token_id: p.current_price or p.entry_price or 0
            for p in api_positions
        }

        # STEP 3: Fetch USDC balance from chain
        logger.info("  [3/5] Fetching USDC balance from chain...")
        try:
            actual_balance = await self.api.fetch_usdc_balance(self.wallet_address)
            with self.storage.transaction() as txn:
                txn.update_usdc_balance(self.wallet_address, actual_balance)
            logger.info(f"        USDC Balance: ${actual_balance:.2f}")
        except Exception as e:
            logger.error(f"        Failed to fetch USDC balance: {e}")
            actual_balance = 0.0

        # STEP 4: Run chain sync for audit trail (non-blocking validation)
        logger.info("  [4/5] Running chain sync for audit trail...")
        await self._run_chain_sync_audit()

        # STEP 5: Compare API positions with computed positions and log discrepancies
        logger.info("  [5/5] Validating against transaction history...")
        discrepancy_count = await self._validate_and_log_discrepancies()

        # Calculate total positions value from API (source of truth)
        total_positions_value = sum(
            p.shares * (p.current_price or p.entry_price or 0)
            for p in api_positions
        )

        # Log summary
        logger.info(f"{'='*60}")
        logger.info(f"RECONCILIATION COMPLETE (API-FIRST)")
        logger.info(f"  API Positions: {len(api_positions)}")
        logger.info(f"  USDC Balance: ${actual_balance:.2f}")
        logger.info(f"  Positions Value: ${total_positions_value:.2f}")
        logger.info(f"  Total Equity: ${actual_balance + total_positions_value:.2f}")
        if discrepancy_count > 0:
            logger.warning(f"  Discrepancies Found: {discrepancy_count} (logged for review)")
        else:
            logger.info(f"  Discrepancies: None - API and computed positions match")
        logger.info(f"{'='*60}")

        # Cleanup stale reservations and agents
        with self.storage.transaction() as txn:
            released_count = txn.release_all_reservations()
            if released_count > 0:
                logger.info(f"  Released {released_count} stale reservations")

            crashed_count = txn.cleanup_stale_agents(
                self.config.risk.stale_agent_threshold_seconds
            )
            if crashed_count > 0:
                logger.warning(f"  Marked {crashed_count} stale agents as crashed")

    async def _run_chain_sync_audit(self):
        """
        Run chain sync for audit trail.
        This populates the transactions table for historical tracking
        but does NOT affect the source of truth (api_positions).
        """
        # Initialize chain sync service if needed
        if self._chain_sync is None:
            self._chain_sync = ChainSyncService(
                config=self.config,
                storage=self.storage,
                api=self.api
            )

        try:
            # Check if we need full sync or incremental
            with self.storage.transaction() as txn:
                sync_state = txn.get_chain_sync_state(self.wallet_address)

            if sync_state is None:
                logger.info("        First sync - performing full historical sync...")
                result = await self._chain_sync.full_sync(
                    self.wallet_address,
                    match_existing_executions=True
                )
            else:
                logger.info(f"        Incremental sync from block {sync_state['last_synced_block']:,}...")
                result = await self._chain_sync.incremental_sync(self.wallet_address)

            if result.success:
                logger.info(f"        Synced {result.transactions_synced} transactions for audit")
            else:
                logger.warning(f"        Chain sync had errors: {result.errors}")

            # Retry any failed gaps
            await self._retry_sync_gaps()

        except Exception as e:
            logger.error(f"        Chain sync audit failed: {e}")

    async def _retry_sync_gaps(self):
        """Retry any previously failed sync ranges"""
        with self.storage.transaction() as txn:
            gaps = txn.get_unresolved_gaps(self.wallet_address)

        if not gaps:
            return

        logger.info(f"        Retrying {len(gaps)} failed sync ranges...")

        for gap in gaps:
            if gap.get("retry_count", 0) >= 5:
                logger.warning(f"        Gap {gap['from_block']}-{gap['to_block']} exceeded max retries")
                continue

            try:
                # Attempt to sync this specific range
                result = await self._chain_sync.full_sync(
                    self.wallet_address,
                    from_block=gap["from_block"],
                    to_block=gap["to_block"]
                )
                if result.success:
                    with self.storage.transaction() as txn:
                        txn.resolve_gap(gap["id"])
                    logger.info(f"        Resolved gap {gap['from_block']}-{gap['to_block']}")
                else:
                    with self.storage.transaction() as txn:
                        txn.increment_gap_retry(gap["id"], str(result.errors))
            except Exception as e:
                with self.storage.transaction() as txn:
                    txn.increment_gap_retry(gap["id"], str(e))

    async def _validate_and_log_discrepancies(self) -> int:
        """
        Compare API positions (truth) with computed positions (audit).
        Log any discrepancies for investigation.

        Returns:
            Number of discrepancies found
        """
        discrepancy_count = 0

        with self.storage.transaction() as txn:
            api_positions = txn.get_api_positions(self.wallet_address)
            computed_positions = txn.get_computed_positions(self.wallet_address)

        # Build lookups
        api_by_token = {p["token_id"]: p for p in api_positions}
        computed_by_token = {p["token_id"]: p for p in computed_positions}

        all_tokens = set(api_by_token.keys()) | set(computed_by_token.keys())

        for token_id in all_tokens:
            api_pos = api_by_token.get(token_id)
            computed_pos = computed_by_token.get(token_id)

            api_shares = api_pos["shares"] if api_pos else 0
            computed_shares = computed_pos["shares"] if computed_pos else 0

            # Check for discrepancies (tolerance of 0.001 shares)
            if abs(api_shares - computed_shares) > 0.001:
                discrepancy_count += 1

                if api_shares > 0 and computed_shares == 0:
                    issue_type = "missing_tx"
                    details = f"API shows {api_shares:.4f} shares but no transactions found"
                elif api_shares == 0 and computed_shares > 0:
                    issue_type = "extra_tx"
                    details = f"Transactions show {computed_shares:.4f} shares but API shows none"
                else:
                    issue_type = "share_mismatch"
                    details = f"API: {api_shares:.4f}, Computed: {computed_shares:.4f}, Diff: {api_shares - computed_shares:.4f}"

                market_id = (api_pos or computed_pos or {}).get("market_id")

                with self.storage.transaction() as txn:
                    txn.log_reconciliation_issue(
                        wallet_address=self.wallet_address,
                        issue_type=issue_type,
                        api_value=api_shares,
                        computed_value=computed_shares,
                        token_id=token_id,
                        market_id=market_id,
                        details=details
                    )

                logger.warning(f"        Discrepancy [{issue_type}]: {token_id[:16]}... - {details}")

        return discrepancy_count

    async def validate_wallet_state(self) -> Tuple[bool, List[str]]:
        """
        Public method to validate wallet state on demand.

        Returns:
            (is_valid, list of issue descriptions)
        """
        issues = []

        # Re-fetch API positions
        api_positions = await self.api.fetch_positions(self.wallet_address)

        with self.storage.transaction() as txn:
            # Update API positions
            txn.upsert_api_positions(self.wallet_address, api_positions)

            # Get computed positions
            computed_positions = txn.get_computed_positions(self.wallet_address)

        api_by_token = {p.token_id: p for p in api_positions}
        computed_by_token = {p["token_id"]: p for p in computed_positions}

        all_tokens = set(api_by_token.keys()) | set(computed_by_token.keys())

        for token_id in all_tokens:
            api_pos = api_by_token.get(token_id)
            computed_pos = computed_by_token.get(token_id)

            api_shares = api_pos.shares if api_pos else 0
            computed_shares = computed_pos["shares"] if computed_pos else 0

            if abs(api_shares - computed_shares) > 0.001:
                issues.append(
                    f"Token {token_id[:16]}...: API={api_shares:.4f}, Computed={computed_shares:.4f}"
                )

        return len(issues) == 0, issues
    
    async def _heartbeat_loop(self, agent_id: str):
        """Background task to update heartbeat"""
        while True:
            try:
                self.storage.update_heartbeat(agent_id)
                await asyncio.sleep(self.config.risk.heartbeat_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _background_validation_loop(self):
        """
        Background task to periodically validate wallet state.

        Runs every _validation_interval_seconds to:
        1. Refresh API positions
        2. Compare with computed positions
        3. Log any discrepancies
        4. Update price cache

        This provides continuous monitoring for drift between
        API state (truth) and transaction history (audit).
        """
        while True:
            try:
                await asyncio.sleep(self._validation_interval_seconds)

                logger.debug("Running background validation...")

                # Fetch fresh API positions
                api_positions = await self.api.fetch_positions(self.wallet_address)

                # Update API positions cache
                with self.storage.transaction() as txn:
                    txn.upsert_api_positions(self.wallet_address, api_positions)

                # Update price cache
                self._current_prices = {
                    p.token_id: p.current_price or p.entry_price or 0
                    for p in api_positions
                }

                # Check for discrepancies
                discrepancy_count = await self._validate_and_log_discrepancies()

                if discrepancy_count > 0:
                    logger.warning(
                        f"Background validation found {discrepancy_count} discrepancies"
                    )
                else:
                    logger.debug("Background validation: no discrepancies")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background validation error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def atomic_reserve(
        self,
        agent_id: str,
        market_id: str,
        token_id: str,
        amount_usd: float,
        ttl_seconds: Optional[int] = None
    ) -> Optional[str]:
        """
        Atomically check limits AND reserve capital.
        
        This is a SINGLE TRANSACTION - no race conditions possible.
        
        Returns reservation_id if successful, None if denied.
        """
        if not self._reconciled:
            raise RuntimeError("Must call startup() before trading!")
        
        if amount_usd < self.config.risk.min_trade_value_usd:
            logger.warning(f"Trade value ${amount_usd:.2f} below minimum")
            return None
        
        if amount_usd > self.config.risk.max_trade_value_usd:
            logger.warning(f"Trade value ${amount_usd:.2f} above maximum")
            return None
        
        ttl = ttl_seconds or self.config.risk.reservation_ttl_seconds
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        
        with self.storage.transaction() as txn:
            # Get current wallet state
            wallet_state = txn.get_wallet_state(self.wallet_address)
            total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
            
            if total_equity <= 0:
                logger.warning("No equity available")
                return None
            
            # Check global wallet limit
            max_wallet_exposure = total_equity * self.config.risk.max_wallet_exposure_pct
            if wallet_state.total_exposure + amount_usd > max_wallet_exposure:
                logger.warning(
                    f"Wallet exposure limit: {wallet_state.total_exposure + amount_usd:.2f} > "
                    f"{max_wallet_exposure:.2f}"
                )
                return None
            
            # Check per-agent limit
            max_agent_exposure = total_equity * self.config.risk.max_per_agent_exposure_pct
            agent_exposure = txn.get_agent_exposure(agent_id)
            if agent_exposure + amount_usd > max_agent_exposure:
                logger.warning(
                    f"Agent exposure limit: {agent_exposure + amount_usd:.2f} > "
                    f"{max_agent_exposure:.2f}"
                )
                return None
            
            # Check per-market limit
            max_market_exposure = total_equity * self.config.risk.max_per_market_exposure_pct
            market_exposure = txn.get_market_exposure(market_id, self.wallet_address)
            if market_exposure + amount_usd > max_market_exposure:
                logger.warning(
                    f"Market exposure limit: {market_exposure + amount_usd:.2f} > "
                    f"{max_market_exposure:.2f}"
                )
                return None
            
            # Check available capital
            if amount_usd > wallet_state.available_capital:
                logger.warning(
                    f"Insufficient capital: need ${amount_usd:.2f}, "
                    f"available ${wallet_state.available_capital:.2f}"
                )
                return None
            
            # All checks passed - create reservation
            reservation_id = txn.create_reservation(
                agent_id=agent_id,
                market_id=market_id,
                token_id=token_id,
                amount_usd=amount_usd,
                expires_at=expires_at
            )
            
            logger.info(
                f"Reserved ${amount_usd:.2f} for {agent_id} "
                f"(reservation: {reservation_id[:8]}...)"
            )
            return reservation_id
    
    def confirm_execution(
        self,
        reservation_id: str,
        filled_shares: float,
        filled_price: float,
        requested_shares: Optional[float] = None
    ) -> bool:
        """
        Confirm trade executed, converting reservation to position.
        
        Handles partial fills correctly - releases unfilled portion back
        to available capital.
        
        Returns True if successful.
        """
        with self.storage.transaction() as txn:
            reservation = txn.get_reservation(reservation_id)
            
            if not reservation:
                logger.error(f"Reservation not found: {reservation_id}")
                return False
            
            if reservation.status != ReservationStatus.PENDING:
                logger.error(f"Reservation not pending: {reservation.status}")
                return False
            
            filled_amount = filled_shares * filled_price
            reserved_amount = reservation.amount_usd
            
            # Create position for filled portion
            txn.create_position(
                agent_id=reservation.agent_id,
                market_id=reservation.market_id,
                token_id=reservation.token_id,
                outcome="",  # Will be updated later
                shares=filled_shares,
                entry_price=filled_price
            )
            
            # Release unfilled portion back to available capital
            unfilled_amount = reserved_amount - filled_amount
            if unfilled_amount > 0.01:  # More than 1 cent unfilled
                logger.info(f"Partial fill: releasing ${unfilled_amount:.2f} back to pool")
            
            # Mark reservation as executed
            txn.mark_reservation_executed(reservation_id, filled_amount)
            
            logger.info(
                f"Confirmed execution: {filled_shares:.2f} shares @ ${filled_price:.4f} "
                f"(${filled_amount:.2f})"
            )
            return True
    
    def release_reservation(self, reservation_id: str) -> bool:
        """
        Release a reservation (trade failed or cancelled).
        
        Returns True if successful.
        """
        try:
            self.storage.release_reservation(reservation_id)
            logger.info(f"Released reservation {reservation_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Error releasing reservation: {e}")
            return False
    
    def can_trade(
        self,
        agent_id: str,
        amount_usd: float
    ) -> Tuple[bool, str]:
        """
        Check if agent can make a trade of given size.
        
        Returns (can_trade, reason).
        
        Note: This is a READ-ONLY check. Use atomic_reserve() for actual
        reservation to avoid race conditions.
        """
        if not self._reconciled:
            return False, "Coordinator not initialized"
        
        if amount_usd < self.config.risk.min_trade_value_usd:
            return False, f"Below minimum (${self.config.risk.min_trade_value_usd})"
        
        if amount_usd > self.config.risk.max_trade_value_usd:
            return False, f"Above maximum (${self.config.risk.max_trade_value_usd})"
        
        with self.storage.transaction() as txn:
            wallet_state = txn.get_wallet_state(self.wallet_address)
            total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
            
            if total_equity <= 0:
                return False, "No equity"
            
            # Check wallet limit
            max_wallet = total_equity * self.config.risk.max_wallet_exposure_pct
            if wallet_state.total_exposure + amount_usd > max_wallet:
                return False, f"Wallet limit (${max_wallet:.2f})"
            
            # Check agent limit
            max_agent = total_equity * self.config.risk.max_per_agent_exposure_pct
            agent_exposure = txn.get_agent_exposure(agent_id)
            if agent_exposure + amount_usd > max_agent:
                return False, f"Agent limit (${max_agent:.2f})"
            
            # Check available capital
            if amount_usd > wallet_state.available_capital:
                return False, f"Insufficient capital (${wallet_state.available_capital:.2f})"
        
        return True, "OK"
    
    def get_available_capital(self, agent_id: str) -> float:
        """
        Get capital available for an agent.
        
        Takes into account:
        - USDC balance
        - Active reservations
        - Agent-specific limits
        """
        with self.storage.transaction() as txn:
            wallet_state = txn.get_wallet_state(self.wallet_address)
            total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
            
            if total_equity <= 0:
                return 0.0
            
            # Agent's remaining allocation
            max_agent = total_equity * self.config.risk.max_per_agent_exposure_pct
            agent_exposure = txn.get_agent_exposure(agent_id)
            agent_remaining = max(0, max_agent - agent_exposure)
            
            # Wallet's remaining capital
            wallet_remaining = wallet_state.available_capital
            
            # Return the minimum
            return min(agent_remaining, wallet_remaining)
    
    def get_wallet_state(self) -> WalletState:
        """Get current wallet state"""
        return self.storage.get_wallet_state(self.wallet_address)
    
    async def refresh_balance(self) -> float:
        """Refresh USDC balance from chain"""
        if not self.api:
            return 0.0
        
        balance = await self.api.fetch_usdc_balance(self.wallet_address)
        
        with self.storage.transaction() as txn:
            txn.update_usdc_balance(self.wallet_address, balance)
        
        return balance
    
    async def fetch_actual_position(self, token_id: str) -> Optional[float]:
        """
        Fetch actual share balance for a token from the API.
        
        Returns actual shares held, or None if position doesn't exist or API unavailable.
        This queries the real on-chain balance, not our SQL tracking.
        """
        if not self.api:
            return None
        
        try:
            actual_positions = await self.api.fetch_positions(self.wallet_address)
            for pos in actual_positions:
                if pos.token_id == token_id:
                    return pos.shares
            # Position not found on-chain
            return 0.0
        except Exception as e:
            logger.warning(f"Failed to fetch actual position for {token_id[:20]}...: {e}")
            return None
    
    def mark_position_closed_by_token(self, token_id: str) -> int:
        """
        Mark position as closed by token_id.
        
        Returns number of positions closed.
        """
        with self.storage.transaction() as txn:
            return txn.mark_position_closed_by_token(self.wallet_address, token_id)
    
    def cleanup_stale(self) -> Tuple[int, int]:
        """
        Cleanup stale reservations and agents.
        
        Returns (reservations_cleaned, agents_cleaned).
        """
        with self.storage.transaction() as txn:
            reservations = txn.cleanup_expired_reservations()
            agents = txn.cleanup_stale_agents(
                self.config.risk.stale_agent_threshold_seconds
            )
        
        if reservations > 0:
            logger.info(f"Cleaned up {reservations} expired reservations")
        if agents > 0:
            logger.warning(f"Marked {agents} stale agents as crashed")
        
        return reservations, agents
    
    async def reconcile_positions(self) -> Tuple[int, int]:
        """
        Lightweight reconciliation of SQL positions with on-chain state.
        
        Performs incremental chain sync to update the transactions table.
        
        Returns (transactions_synced, actual_position_count).
        """
        if not self.api:
            return 0, 0
        
        try:
            if self._chain_sync is None:
                self._chain_sync = ChainSyncService(
                    config=self.config,
                    storage=self.storage,
                    api=self.api
                )
            
            result = await self._chain_sync.incremental_sync(self.wallet_address)
            
            # Update current prices cache
            actual_positions = await self.api.fetch_positions(self.wallet_address)
            self._current_prices = {
                p.token_id: p.current_price or p.entry_price or 0
                for p in actual_positions
            }
            
            # Also refresh USDC balance
            actual_balance = await self.api.fetch_usdc_balance(self.wallet_address)
            with self.storage.transaction() as txn:
                txn.update_usdc_balance(self.wallet_address, actual_balance)
            
            return result.transactions_synced, len(actual_positions)
            
        except Exception as e:
            logger.warning(f"Chain sync reconciliation failed: {e}")
            return 0, 0
    
    def get_computed_positions(self) -> List[dict]:
        """
        Get current positions computed from transaction history.
        
        This is the source-of-truth method when chain sync is enabled.
        """
        with self.storage.transaction() as txn:
            return txn.get_computed_positions(self.wallet_address)
    
    def get_computed_exposure(self, agent_id: Optional[str] = None) -> float:
        """
        Get exposure computed from transaction history.
        
        Args:
            agent_id: Optional agent ID to filter by. If None, returns total wallet exposure.
        
        Returns total position value based on transactions.
        """
        with self.storage.transaction() as txn:
            if agent_id:
                return txn.get_agent_computed_exposure(agent_id, self._current_prices)
            else:
                return txn.get_total_computed_exposure(self.wallet_address, self._current_prices)
    
    def get_transaction_history(
        self,
        transaction_type: Optional[str] = None,
        token_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[dict]:
        """
        Get transaction history from the transactions table.
        
        Args:
            transaction_type: Optional filter ('buy', 'sell', 'claim', 'deposit', 'withdrawal')
            token_id: Optional filter by token
            limit: Optional limit on results
        
        Returns list of transactions.
        """
        with self.storage.transaction() as txn:
            return txn.get_transactions(
                wallet_address=self.wallet_address,
                transaction_type=transaction_type,
                token_id=token_id,
                limit=limit
            )
    
    async def sync_transactions(self, full_sync: bool = False) -> int:
        """
        Manually trigger transaction sync.
        
        Args:
            full_sync: If True, performs full historical sync. Otherwise incremental.
        
        Returns number of transactions synced.
        """
        if self._chain_sync is None:
            self._chain_sync = ChainSyncService(
                config=self.config,
                storage=self.storage,
                api=self.api
            )
        
        if full_sync:
            result = await self._chain_sync.full_sync(self.wallet_address)
        else:
            result = await self._chain_sync.incremental_sync(self.wallet_address)
        
        return result.transactions_synced


