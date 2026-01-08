"""
Main TradingBot class with composition-based architecture.

The TradingBot is configured with pluggable components:
- SignalSource: Where trading signals come from
- PositionSizer: How to size positions
- ExecutionEngine: How to execute trades
- RiskCoordinator: Multi-agent risk management
"""

import asyncio
import logging
from typing import Optional, List, TYPE_CHECKING
from datetime import datetime, timezone

from ..core.models import Signal, Side, ExecutionResult, Position, PositionStatus, AgentStatus
from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from .risk_coordinator import RiskCoordinator
from .safety import CircuitBreaker, DrawdownLimit, TradingHalt
from .components.signals import SignalSource
from .components.sizers import PositionSizer
from .components.executors import ExecutionEngine, DryRunExecutor
from .components.exit_strategies import ExitMonitor, ExitConfig, ExitReason, PositionState

if TYPE_CHECKING:
    from py_clob_client.client import ClobClient

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Composition-based trading bot.
    
    Instead of inheritance, uses pluggable components for flexibility.
    
    Usage:
        bot = TradingBot(
            agent_id="my-bot",
            signal_source=FlowAlertSignals(),
            position_sizer=SignalScaledSizer(),
            executor=AggressiveExecutor(),
        )
        await bot.start()
        await bot.run()
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str = "generic",
        signal_source: Optional[SignalSource] = None,
        position_sizer: Optional[PositionSizer] = None,
        executor: Optional[ExecutionEngine] = None,
        config: Optional[Config] = None,
        dry_run: bool = False,
        min_price: float = 0.0,   # Minimum price filter (0 = no filter)
        max_price: float = 1.0,   # Maximum price filter (1 = no filter)
        exit_config: Optional[ExitConfig] = None,  # Exit strategy configuration
    ):
        """
        Initialize trading bot with components.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., "bond", "flow")
            signal_source: Source of trading signals
            position_sizer: Position sizing strategy
            executor: Order execution engine
            config: Configuration (uses default if not provided)
            dry_run: If True, use dry run executor
            min_price: Minimum token price to trade (0.20 = 20c)
            max_price: Maximum token price to trade (0.80 = 80c)
            exit_config: Exit strategy configuration (take-profit, stop-loss, etc.)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or get_config()
        self.dry_run = dry_run
        
        # Price range filters
        self.min_price = min_price
        self.max_price = max_price
        
        # Components
        self.signal_source = signal_source
        self.position_sizer = position_sizer
        self.executor = executor or (DryRunExecutor() if dry_run else None)
        
        # Exit strategy monitoring
        self.exit_config = exit_config or ExitConfig()
        self.exit_monitor = ExitMonitor(self.exit_config)
        
        # Infrastructure
        self.api: Optional[PolymarketAPI] = None
        self.client: Optional["ClobClient"] = None
        self.risk_coordinator: Optional[RiskCoordinator] = None
        
        # Safety components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.risk.circuit_breaker_failures,
            reset_timeout_seconds=self.config.risk.circuit_breaker_reset_seconds
        )
        self.drawdown_limit = DrawdownLimit(
            max_daily_drawdown_pct=self.config.risk.max_daily_drawdown_pct,
            max_total_drawdown_pct=self.config.risk.max_total_drawdown_pct
        )
        self.trading_halt = TradingHalt()
        
        # State
        self.running = False
        self._main_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """
        Start the trading bot.
        
        Initializes all components and starts the risk coordinator.
        """
        logger.info(f"{'='*60}")
        logger.info(f"🚀 STARTING TRADING BOT")
        logger.info(f"{'='*60}")
        logger.info(f"  Agent ID:   {self.agent_id}")
        logger.info(f"  Type:       {self.agent_type}")
        logger.info(f"  Mode:       {'🧪 DRY RUN' if self.dry_run else '💸 LIVE'}")
        
        if self.dry_run:
            logger.warning("⚠️  DRY RUN MODE - No real orders will be placed")
        
        # Validate configuration
        self.config.require_credentials()
        logger.info("  ✅ Credentials validated")
        
        # Initialize API
        self.api = PolymarketAPI(self.config)
        await self.api.connect()
        logger.info("  ✅ API connected")
        
        # Initialize CLOB client
        if not self.dry_run:
            from py_clob_client.client import ClobClient
            
            self.client = ClobClient(
                self.config.clob_host,
                key=self.config.private_key,
                chain_id=self.config.chain_id,
                signature_type=2,
                funder=self.config.proxy_address
            )
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
            logger.info("  ✅ CLOB client initialized")
        
        # Initialize risk coordinator
        self.risk_coordinator = RiskCoordinator(
            config=self.config,
            api=self.api
        )
        
        if not await self.risk_coordinator.startup(self.agent_id, self.agent_type):
            raise RuntimeError("Failed to start risk coordinator")
        logger.info("  ✅ Risk coordinator started")
        
        # Initialize drawdown tracking - load persisted state or start fresh
        wallet_state = self.risk_coordinator.get_wallet_state()
        total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
        
        # Try to load persisted drawdown state
        with self.risk_coordinator.storage.transaction() as txn:
            saved_state = txn.get_drawdown_state(self.config.proxy_address)
        
        if saved_state:
            # Restore saved state
            self.drawdown_limit.peak_equity = saved_state["peak_equity"]
            self.drawdown_limit.daily_start_equity = saved_state["daily_start_equity"]
            self.drawdown_limit.daily_start_date = saved_state["daily_start_date"]
            self.drawdown_limit.is_breached = saved_state["is_breached"]
            self.drawdown_limit.breach_reason = saved_state["breach_reason"]
            self.drawdown_limit.current_equity = total_equity
            
            # Check if we need to reset daily tracking (new day)
            now = datetime.now(timezone.utc)
            if saved_state["daily_start_date"] and now.date() != saved_state["daily_start_date"].date():
                self.drawdown_limit.daily_start_equity = total_equity
                self.drawdown_limit.daily_start_date = now
                logger.info(f"Daily drawdown reset (new day): starting equity ${total_equity:.2f}")
            
            # Update peak if current equity is higher
            if total_equity > self.drawdown_limit.peak_equity:
                self.drawdown_limit.peak_equity = total_equity
            
            logger.info(f"📊 Loaded persisted drawdown state:")
            logger.info(f"  Peak Equity: ${self.drawdown_limit.peak_equity:.2f}")
            logger.info(f"  Daily Start: ${self.drawdown_limit.daily_start_equity:.2f}")
            logger.info(f"  Current: ${total_equity:.2f}")
            logger.info(f"  Daily DD: {self.drawdown_limit.daily_drawdown_pct:.1%}")
            logger.info(f"  Total DD: {self.drawdown_limit.total_drawdown_pct:.1%}")
            
            if self.drawdown_limit.is_breached:
                logger.warning(f"⚠️  Drawdown limit still breached: {self.drawdown_limit.breach_reason}")
                self.trading_halt.add_reason("DRAWDOWN", self.drawdown_limit.breach_reason or "")
        else:
            # First run - initialize fresh
            self.drawdown_limit.reset(total_equity)
            logger.info(f"📊 Initialized fresh drawdown tracking: equity ${total_equity:.2f}")
        
        self.running = True
        
        logger.info(f"{'='*60}")
        logger.info(f"💰 WALLET STATE")
        logger.info(f"{'='*60}")
        logger.info(f"  USDC Balance:    ${wallet_state.usdc_balance:,.2f}")
        logger.info(f"  Positions Value: ${wallet_state.total_positions_value:,.2f}")
        logger.info(f"  Total Equity:    ${total_equity:,.2f}")
        logger.info(f"  Available:       ${wallet_state.available_capital:,.2f}")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Bot ready and running!")
    
    async def stop(self):
        """Stop the trading bot gracefully"""
        logger.info(f"{'='*60}")
        logger.info(f"🛑 STOPPING BOT: {self.agent_id}")
        logger.info(f"{'='*60}")
        
        self.running = False
        
        # Cancel main task
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown risk coordinator
        if self.risk_coordinator:
            await self.risk_coordinator.shutdown()
            logger.info("  ✅ Risk coordinator stopped")
        
        # Close API
        if self.api:
            await self.api.close()
            logger.info("  ✅ API disconnected")
        
        logger.info(f"{'='*60}")
        logger.info(f"👋 Bot stopped cleanly")
        logger.info(f"{'='*60}")
    
    async def run(self, interval_seconds: float = 5.0):
        """
        Main trading loop.
        
        Continuously:
        1. Check safety conditions
        2. Get signals from signal source
        3. Process signals and execute trades
        4. Sleep for interval
        """
        if not self.running:
            raise RuntimeError("Bot not started. Call start() first.")
        
        if not self.signal_source:
            raise RuntimeError("No signal source configured")
        
        if not self.position_sizer:
            raise RuntimeError("No position sizer configured")
        
        if not self.executor:
            raise RuntimeError("No executor configured")
        
        logger.info(f"Starting main loop (interval={interval_seconds}s)")
        
        self._main_task = asyncio.current_task()
        
        try:
            while self.running:
                try:
                    await self._trading_iteration()
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}")
                    self.circuit_breaker.record_failure()
                
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
    
    async def _trading_iteration(self):
        """Single iteration of the trading loop"""
        # 0. Check if we should stop (external stop command)
        if await self._check_should_stop():
            self.running = False
            logger.info("Received external stop signal - shutting down")
            return

        # 1. Reload drawdown state from DB (to respect external resets)
        await self._reload_drawdown_state()

        # 2. Check safety conditions (for new BUY signals - SELLs checked separately)
        # Note: We still check here to skip unnecessary work, but SELL signals
        # for exits will use the permissive check

        # 3. Update equity for drawdown tracking
        await self._update_equity()
        
        # 3. Monitor existing positions for exit conditions
        # This runs even if safety checks fail for BUYs - we always want to manage exits
        await self._monitor_positions()
        
        # 4. Check if we should process new signals
        if not self._check_safety():
            return
        
        # 5. Get signals
        signals = await self.signal_source.get_signals()
        
        if not signals:
            logger.debug("No signals")
            return
        
        logger.info(f"Got {len(signals)} signals")
        
        # 4. Process signals
        # Note: We don't do a strict safety check here because SELL signals
        # should be allowed through even during drawdown/circuit breaker conditions.
        # Full safety checks happen inside _process_signal based on signal direction.
        for signal in signals:
            if not self.running:
                break
            
            # Only check for manual halt at loop level - everything else checked per-signal
            if self.trading_halt.is_halted and "DRAWDOWN" not in self.trading_halt.reasons:
                logger.warning(f"Trading manually halted: {self.trading_halt.reason_summary}")
                break
            
            await self._process_signal(signal)
    
    def _check_safety(self) -> bool:
        """Check all safety conditions"""
        # Trading halt
        if self.trading_halt.is_halted:
            logger.warning(f"Trading halted: {self.trading_halt.reason_summary}")
            return False
        
        # Circuit breaker
        if not self.circuit_breaker.can_execute():
            remaining = self.circuit_breaker.seconds_until_reset
            logger.warning(f"Circuit breaker OPEN ({remaining:.0f}s until reset)")
            return False
        
        # Drawdown limit
        if self.drawdown_limit.is_breached:
            logger.warning(f"Drawdown limit breached: {self.drawdown_limit.breach_reason}")
            return False
        
        return True
    
    def _check_safety_for_sell(self) -> bool:
        """
        Check safety for SELL orders (more permissive - allows reducing exposure).
        
        SELL orders should be allowed even when drawdown limits or circuit breakers
        are triggered, because they reduce exposure and help recover from drawdowns.
        Only manual trading halts (non-drawdown) should block sells.
        """
        # Only trading halt should block sells - we always want to allow exiting positions
        if self.trading_halt.is_halted:
            # Check if halt is due to drawdown - if so, allow sells to reduce exposure
            if "DRAWDOWN" in self.trading_halt.reasons:
                logger.info("📉 Allowing SELL through drawdown halt to reduce exposure")
                return True
            # Non-drawdown halt (manual intervention) - block everything
            logger.warning(f"Trading halted (manual): {self.trading_halt.reason_summary}")
            return False
        
        # Circuit breaker and drawdown limits don't block sells
        # We WANT to reduce exposure when things are going badly
        return True
    
    async def _update_equity(self):
        """Update equity and check drawdown limits"""
        # Throttle reconciliation to every 30 seconds to avoid API rate limits
        # This ensures manually closed positions are detected promptly
        now = datetime.now(timezone.utc)
        if not hasattr(self, '_last_reconcile') or \
           (now - self._last_reconcile).total_seconds() > 30:
            closed, _ = await self.risk_coordinator.reconcile_positions()
            self._last_reconcile = now
            
            # If we closed positions, clear any stale drawdown halt
            # since our equity calculation was based on phantom positions
            if closed > 0 and self.trading_halt.is_halted:
                if "DRAWDOWN" in self.trading_halt.reasons:
                    logger.info(
                        f"🔄 Reconciled {closed} stale position(s) - "
                        f"re-checking drawdown with accurate equity"
                    )
        
        # Now get wallet state (will be accurate after reconciliation)
        wallet_state = self.risk_coordinator.get_wallet_state()
        total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
        
        if not self.drawdown_limit.update(total_equity):
            logger.error("Drawdown limit breached - halting trading")
            self.trading_halt.add_reason("DRAWDOWN", self.drawdown_limit.breach_reason or "")
        
        # Persist drawdown state (throttle to avoid excessive writes)
        if not hasattr(self, '_last_drawdown_save') or \
           (now - self._last_drawdown_save).total_seconds() > 60:
            with self.risk_coordinator.storage.transaction() as txn:
                txn.update_drawdown_state(
                    wallet_address=self.config.proxy_address,
                    peak_equity=self.drawdown_limit.peak_equity,
                    daily_start_equity=self.drawdown_limit.daily_start_equity or total_equity,
                    daily_start_date=self.drawdown_limit.daily_start_date or datetime.now(timezone.utc),
                    is_breached=self.drawdown_limit.is_breached,
                    breach_reason=self.drawdown_limit.breach_reason
                )
            self._last_drawdown_save = now

    async def _check_should_stop(self) -> bool:
        """
        Check if the bot should stop due to external command.

        Checks the agent status in the database - if marked as STOPPED,
        the bot will stop gracefully.
        """
        # Throttle DB checks to every 10 seconds
        now = datetime.now(timezone.utc)
        if hasattr(self, '_last_stop_check') and \
           (now - self._last_stop_check).total_seconds() < 10:
            return False
        self._last_stop_check = now

        try:
            with self.risk_coordinator.storage.transaction() as txn:
                agent = txn.get_agent(self.agent_id)
                if agent and agent.status == AgentStatus.STOPPED:
                    logger.info(f"Agent {self.agent_id} marked as STOPPED in database")
                    return True
        except Exception as e:
            logger.debug(f"Error checking agent status: {e}")

        return False

    async def _reload_drawdown_state(self):
        """
        Reload drawdown state from database to respect external resets.

        This allows the `reset-drawdown` command to take effect on running bots.
        """
        # Throttle DB checks to every 30 seconds
        now = datetime.now(timezone.utc)
        if hasattr(self, '_last_drawdown_reload') and \
           (now - self._last_drawdown_reload).total_seconds() < 30:
            return
        self._last_drawdown_reload = now

        try:
            with self.risk_coordinator.storage.transaction() as txn:
                saved_state = txn.get_drawdown_state(self.config.proxy_address)

            if saved_state:
                # Check if state was externally reset (is_breached cleared)
                db_breached = saved_state["is_breached"]
                db_peak = saved_state["peak_equity"]
                db_daily_start = saved_state["daily_start_equity"]
                db_daily_date = saved_state["daily_start_date"]

                # Detect external reset: DB shows not breached but we think it is
                # OR the daily_start_date changed (indicates manual reset)
                external_reset = False
                if self.drawdown_limit.is_breached and not db_breached:
                    external_reset = True
                elif db_daily_date and self.drawdown_limit.daily_start_date:
                    # Check if daily start date changed significantly (more than 1 minute difference)
                    time_diff = abs((db_daily_date - self.drawdown_limit.daily_start_date).total_seconds())
                    if time_diff > 60 and db_daily_start != self.drawdown_limit.daily_start_equity:
                        external_reset = True

                if external_reset:
                    # Full reload - someone used reset-drawdown command
                    logger.info("Drawdown state was externally reset - reloading all values")
                    logger.info(f"  Peak: ${self.drawdown_limit.peak_equity:.2f} -> ${db_peak:.2f}")
                    logger.info(f"  Daily start: ${self.drawdown_limit.daily_start_equity:.2f} -> ${db_daily_start:.2f}")

                    self.drawdown_limit.peak_equity = db_peak
                    self.drawdown_limit.daily_start_equity = db_daily_start
                    self.drawdown_limit.daily_start_date = db_daily_date
                    self.drawdown_limit.is_breached = db_breached
                    self.drawdown_limit.breach_reason = saved_state.get("breach_reason")

                    # Clear any trading halt from drawdown
                    self.trading_halt.clear_reason("DRAWDOWN")

        except Exception as e:
            logger.debug(f"Error reloading drawdown state: {e}")

    def _should_manage_position(self, position: Position) -> bool:
        """
        Determine if this bot should manage a given position.
        
        Ownership rules:
        1. Own positions: agent_id matches this bot's agent_id
        2. Orphan positions: status is ORPHAN or agent_id is 'unattributed'/empty/orphan-prefixed
        3. Other bots' positions: NOT managed (filtered out)
        
        This ensures:
        - Bond bot only exits bond positions (using bond exit config)
        - Flow bot only exits flow positions (using flow exit config)
        - Either bot can handle orphans (positions found on-chain without DB record)
        
        Returns:
            True if this bot should manage the position, False otherwise
        """
        # Check if this is our own position
        if position.agent_id == self.agent_id:
            return True
        
        # Check if this is an orphan position (any bot can manage)
        # Orphan indicators:
        # 1. Status is ORPHAN
        # 2. agent_id is empty, 'unattributed', or starts with 'orphan_'
        if position.status == PositionStatus.ORPHAN:
            return True
        
        orphan_indicators = ('', 'unattributed', 'orphan_')
        if not position.agent_id or position.agent_id in orphan_indicators[:2]:
            return True
        if position.agent_id.startswith('orphan_'):
            return True
        
        # This position belongs to another bot - don't manage it
        return False
    
    async def _monitor_positions(self):
        """
        Monitor existing positions for exit conditions.
        
        Checks all positions against configured exit strategies:
        - Take-profit: Exit when target profit reached
        - Trailing stop: Lock in profits with dynamic stop
        - Stop-loss: Exit when loss threshold reached
        - Time-based: Force exit after max hold time
        
        Only monitors positions owned by this agent or orphan positions.
        Other agents' positions are left for their respective bots to manage.
        
        Also reconciles with actual on-chain state to handle manual sells.
        """
        if not self.exit_monitor:
            return
        
        wallet_state = self.risk_coordinator.get_wallet_state()
        now = datetime.now(timezone.utc)
        
        # Filter to only positions this bot should manage
        managed_positions = [p for p in wallet_state.positions if self._should_manage_position(p)]
        
        # Get current position token IDs from managed positions only
        current_position_ids = {pos.token_id for pos in managed_positions}
        
        # Clean up exit monitor for positions that no longer exist
        # This handles manual sells or positions closed outside the bot
        # IMPORTANT: Add grace period for new positions - API may be slow to reflect them
        tracked_ids = list(self.exit_monitor._position_states.keys())
        for tracked_id in tracked_ids:
            if tracked_id not in current_position_ids:
                # Check grace period before cleanup - new positions may not be in API yet
                state = self.exit_monitor.get_state(tracked_id)
                if state and state.entry_time:
                    age_seconds = (now - state.entry_time).total_seconds()
                    if age_seconds < 30:  # 30 second grace period
                        logger.debug(
                            f"⏳ Skipping cleanup for {tracked_id[:16]}... "
                            f"(position only {age_seconds:.0f}s old - waiting for API sync)"
                        )
                        continue
                
                self.exit_monitor.remove_position(tracked_id)
                logger.info(
                    f"🧹 Cleaned up tracking for {tracked_id[:16]}... "
                    f"(position no longer exists - likely manual sell)"
                )
                # Also cancel any stale orders for this token
                await self._cancel_existing_orders(tracked_id)
        
        if not managed_positions:
            return
        
        for position in managed_positions:
            try:
                # Get or create position state for monitoring
                state = self.exit_monitor.get_state(position.token_id)
                
                if state is None:
                    # Check position age for grace period
                    entry_time = position.entry_time or now
                    position_age_seconds = (now - entry_time).total_seconds()
                    
                    # Before registering, verify position actually exists on-chain
                    api_shares = await self.risk_coordinator.fetch_actual_position(position.token_id)
                    if api_shares is not None and api_shares <= 0:
                        # Position doesn't exist on-chain
                        # For NEW positions (< 30s old), the API may not reflect them yet
                        if position_age_seconds < 30:
                            logger.debug(
                                f"⏳ API shows 0 shares for {position.token_id[:16]}... "
                                f"but position only {position_age_seconds:.0f}s old - trusting SQL"
                            )
                            api_shares = None  # Fall back to SQL shares
                        else:
                            # Position is old enough, trust the API
                            logger.info(
                                f"🧹 Position {position.token_id[:16]}... in SQL but not on-chain - marking closed"
                            )
                            self.risk_coordinator.mark_position_closed_by_token(position.token_id)
                            continue
                    
                    # Before registering, check if orderbook exists (market still active)
                    # This prevents failed safety sell attempts for resolved markets
                    bid, ask, _ = await self.api.get_spread(position.token_id)
                    if bid is None and ask is None:
                        # Market is closed/resolved - clean up immediately
                        logger.info(
                            f"🧹 Position {position.token_id[:16]}... has no orderbook - "
                            f"market closed/resolved. Cleaning up."
                        )
                        self.risk_coordinator.mark_position_closed_by_token(position.token_id)
                        await self._cancel_existing_orders(position.token_id)
                        continue
                    
                    # Position exists and market is active - register it for monitoring
                    actual_shares = api_shares if api_shares is not None else position.shares
                    state = self.exit_monitor.register_position(
                        position_id=position.token_id,
                        entry_price=position.entry_price,
                        entry_time=entry_time,
                        shares=actual_shares,
                    )
                    logger.info(
                        f"📝 Registered position for monitoring: {position.token_id[:16]}... "
                        f"@ ${position.entry_price:.4f} ({actual_shares:.4f} shares)"
                    )
                    
                    # Place safety sell order for this position (handles orphans and restarts)
                    # Only if position doesn't already have a safety order
                    if state and not state.has_safety_order():
                        # IMPORTANT: Cancel any existing orders first (from previous bot runs)
                        # This prevents "not enough balance" errors from duplicate orders
                        cancelled = await self._cancel_existing_orders(position.token_id)
                        if cancelled > 0:
                            logger.info(f"  🗑️  Cleared {cancelled} existing order(s) before placing safety sell")
                        
                        safety_order_id = await self._place_safety_sell(
                            token_id=position.token_id,
                            shares=actual_shares,
                            entry_price=position.entry_price,
                        )
                        if safety_order_id:
                            state.update_safety_order(safety_order_id)
                
                # Get current price
                bid, ask, spread = await self.api.get_spread(position.token_id)
                
                # If both bid and ask are None, the orderbook doesn't exist
                # This happens when a market is closed/resolved
                if bid is None and ask is None:
                    # Track failures to avoid removing on transient errors
                    if not hasattr(self, '_position_price_failures'):
                        self._position_price_failures = {}
                    
                    failures = self._position_price_failures.get(position.token_id, 0) + 1
                    self._position_price_failures[position.token_id] = failures
                    
                    if failures >= 3:
                        logger.warning(
                            f"🗑️ No orderbook for {position.token_id[:16]}... after {failures} attempts - "
                            f"market closed/resolved. Cleaning up position."
                        )
                        # Clean up the position from all tracking
                        self.exit_monitor.remove_position(position.token_id)
                        self.risk_coordinator.mark_position_closed_by_token(position.token_id)
                        await self._cancel_existing_orders(position.token_id)
                        # Clear failure counter
                        del self._position_price_failures[position.token_id]
                    continue
                
                # Reset failure counter on successful price fetch
                if hasattr(self, '_position_price_failures') and position.token_id in self._position_price_failures:
                    del self._position_price_failures[position.token_id]
                
                current_price = bid or position.current_price or position.entry_price
                
                if current_price <= 0:
                    continue
                
                # Check exit conditions
                exit_result = self.exit_monitor.check_exit_conditions(state, current_price, now)
                
                if exit_result:
                    reason, exit_price, description = exit_result
                    await self._execute_exit(position, reason, exit_price, description)
                    
            except Exception as e:
                logger.warning(f"Error monitoring position {position.token_id[:16]}...: {e}")
    
    async def _cancel_existing_orders(self, token_id: str) -> int:
        """
        Cancel any existing open orders for a token.
        
        This ensures stale orders (e.g., take-profit at 90c when price is now 20c)
        are cancelled before placing new orders based on current conditions.
        
        Returns:
            Number of orders cancelled
        """
        if not self.client or self.dry_run:
            return 0
        
        try:
            from py_clob_client.clob_types import OpenOrderParams
            
            # Query for existing orders on this token
            params = OpenOrderParams(asset_id=token_id)
            orders_response = self.client.get_orders(params)
            
            # Handle response format
            if isinstance(orders_response, dict):
                orders = orders_response.get('data', []) or orders_response.get('orders', [])
            elif hasattr(orders_response, 'data'):
                orders = orders_response.data or []
            else:
                orders = orders_response if isinstance(orders_response, list) else []
            
            if not orders:
                return 0
            
            cancelled = 0
            for order in orders:
                order_id = order.get('id') if isinstance(order, dict) else getattr(order, 'id', None)
                if order_id:
                    try:
                        self.client.cancel(order_id)
                        order_price = order.get('price') if isinstance(order, dict) else getattr(order, 'price', 'N/A')
                        order_side = order.get('side') if isinstance(order, dict) else getattr(order, 'side', 'N/A')
                        logger.info(
                            f"  🗑️  Cancelled stale order: {order_id[:16]}... "
                            f"({order_side} @ ${order_price})"
                        )
                        cancelled += 1
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to cancel order {order_id[:16]}...: {e}")
            
            return cancelled
            
        except Exception as e:
            logger.warning(f"Error checking/cancelling existing orders: {e}")
            return 0
    
    async def _place_safety_sell(
        self,
        token_id: str,
        shares: float,
        entry_price: float,
    ) -> Optional[str]:
        """
        Place a GTC limit sell order at $0.99 (safety sell).
        
        This captures value when the market resolves to the winning outcome,
        since Polymarket doesn't allow direct claims via API.
        
        Args:
            token_id: Token to place safety sell for
            shares: Number of shares to sell
            entry_price: Entry price (for logging)
            
        Returns:
            Order ID if successful, None otherwise
        """
        if not self.client or self.dry_run:
            if self.dry_run:
                logger.info(
                    f"  🛡️  [DRY RUN] Would place safety sell: "
                    f"{shares:.4f} shares @ ${self.exit_config.near_resolution_high_price:.3f}"
                )
            return None
        
        safety_price = self.exit_config.near_resolution_high_price
        
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import SELL
            
            # Apply 99.5% safety margin to avoid "not enough balance" errors
            safe_shares = shares * 0.995
            
            # Create and place GTC limit sell order at safety price
            order_args = OrderArgs(
                price=safety_price,
                size=safe_shares,
                side=SELL,
                token_id=token_id
            )
            
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)
            
            # Handle response - can be dict or object
            def get_response_field(field: str, default=None):
                if isinstance(response, dict):
                    return response.get(field, default)
                return getattr(response, field, default)
            
            success = get_response_field("success")
            order_id = get_response_field("orderID") or get_response_field("order_id") or ""
            
            if success and order_id:
                expected_value = safe_shares * safety_price
                logger.info(
                    f"  🛡️  Safety sell placed: {safe_shares:.4f} shares @ ${safety_price:.3f} "
                    f"(~${expected_value:.2f} if resolved)"
                )
                return order_id
            else:
                error_msg = get_response_field("errorMsg") or get_response_field("error_msg") or "Unknown error"
                logger.warning(f"  ⚠️  Failed to place safety sell: {error_msg}")
                return None
                
        except Exception as e:
            logger.warning(f"  ⚠️  Error placing safety sell for {token_id[:16]}...: {e}")
            return None
    
    async def _cancel_safety_order(self, token_id: str) -> bool:
        """
        Cancel the safety sell order for a position.
        
        Should be called before placing exit orders at different prices
        (e.g., take-profit, stop-loss, trailing stop).
        
        Args:
            token_id: Token ID of the position
            
        Returns:
            True if cancelled successfully or no order existed, False on error
        """
        if not self.client or self.dry_run:
            return True
        
        # Get the position state to find the safety order ID
        state = self.exit_monitor.get_state(token_id)
        if not state or not state.safety_order_id:
            return True  # No safety order to cancel
        
        try:
            self.client.cancel(state.safety_order_id)
            logger.info(f"  🗑️  Cancelled safety sell order: {state.safety_order_id[:16]}...")
            state.update_safety_order(None)
            return True
        except Exception as e:
            # Order may have already filled or been cancelled
            logger.debug(f"  ℹ️  Could not cancel safety order (may have filled): {e}")
            state.update_safety_order(None)
            return True
    
    async def _execute_exit(
        self,
        position: Position,
        reason: ExitReason,
        exit_price: float,
        description: str
    ):
        """Execute an exit for a position that triggered an exit condition."""
        logger.info(f"{'='*60}")
        logger.info(f"🚨 EXIT TRIGGERED: {reason.value.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"  📌 Token: {position.token_id[:32]}...")
        logger.info(f"  💰 Entry: ${position.entry_price:.4f}")
        logger.info(f"  💵 Exit:  ${exit_price:.4f}")
        logger.info(f"  📊 Tracked Shares: {position.shares:.4f}")
        logger.info(f"  📝 Reason: {description}")
        
        # Get share counts from BOTH SQL and API, use minimum for safety
        # SQL tracking
        wallet_state = self.risk_coordinator.get_wallet_state()
        sql_position = next(
            (p for p in wallet_state.positions if p.token_id == position.token_id), 
            None
        )
        sql_shares = sql_position.shares if sql_position else 0.0
        
        # API/on-chain balance (source of truth)
        api_shares = await self.risk_coordinator.fetch_actual_position(position.token_id)
        
        logger.info(f"  📊 Balance check: SQL={sql_shares:.4f}, API={api_shares if api_shares is not None else 'N/A'}")
        
        # Determine actual shares - use API if available, fall back to SQL
        if api_shares is not None:
            if api_shares <= 0:
                logger.info(f"  ℹ️  Position no longer exists on-chain (likely manual sell) - cleaning up")
                # Clean up ALL tracking: exit monitor, SQL, and orders
                self.exit_monitor.remove_position(position.token_id)
                self.risk_coordinator.mark_position_closed_by_token(position.token_id)
                await self._cancel_existing_orders(position.token_id)
                return
            actual_shares = api_shares
        elif sql_shares > 0:
            logger.warning(f"  ⚠️  API unavailable, using SQL balance: {sql_shares:.4f}")
            actual_shares = sql_shares
        else:
            logger.info(f"  ℹ️  Position no longer exists - cleaning up")
            # Clean up ALL tracking: exit monitor, SQL, and orders
            self.exit_monitor.remove_position(position.token_id)
            self.risk_coordinator.mark_position_closed_by_token(position.token_id)
            await self._cancel_existing_orders(position.token_id)
            return
        
        # Apply 99.5% safety margin to avoid "not enough balance" errors from rounding
        safe_shares = actual_shares * 0.995
        if safe_shares != actual_shares:
            logger.info(f"  🔄 Applying safety margin: {actual_shares:.4f} → {safe_shares:.4f}")
        
        # Cancel the safety sell order first (updates state tracking)
        await self._cancel_safety_order(position.token_id)
        
        # Cancel any other existing stale orders for this token before placing new one
        cancelled = await self._cancel_existing_orders(position.token_id)
        if cancelled > 0:
            logger.info(f"  🔄 Cancelled {cancelled} additional order(s)")
        
        # Calculate value to sell using SAFE shares (with margin)
        size_usd = safe_shares * exit_price
        
        # Check safety for sell (permissive check)
        if not self._check_safety_for_sell():
            logger.warning(f"⚠️  Exit blocked by safety check")
            return
        
        try:
            # Execute sell order
            result = await self.executor.execute(
                client=self.client,
                token_id=position.token_id,
                side=Side.SELL,
                size_usd=size_usd,
                price=exit_price,
                orderbook=None,
                original_signal_price=None  # Don't check price drift for exits
            )
            
            if result.success and result.filled_shares > 0:
                self.circuit_breaker.record_success()
                
                # Remove from exit monitor
                self.exit_monitor.remove_position(position.token_id)
                
                pnl = (result.filled_price - position.entry_price) * result.filled_shares
                pnl_pct = (result.filled_price - position.entry_price) / position.entry_price
                
                logger.info(f"{'='*60}")
                logger.info(f"✅ EXIT FILLED")
                logger.info(f"{'='*60}")
                logger.info(f"  📦 Shares sold: {result.filled_shares:.4f}")
                logger.info(f"  💰 Fill price: ${result.filled_price:.4f}")
                logger.info(f"  {'📈' if pnl >= 0 else '📉'} P&L: ${pnl:.2f} ({pnl_pct:+.2%})")
                logger.info(f"  🏷️  Exit reason: {reason.value}")
                logger.info(f"{'='*60}")
                
                # Save execution history
                try:
                    with self.risk_coordinator.storage.transaction() as txn:
                        txn.save_execution(
                            agent_id=self.agent_id,
                            market_id=position.market_id,
                            token_id=position.token_id,
                            side=Side.SELL,
                            shares=result.filled_shares,
                            price=exit_price,
                            filled_price=result.filled_price,
                            signal_score=0.0,  # Exit, not signal-based
                            success=True
                        )
                except Exception as e:
                    logger.warning(f"Failed to save exit execution: {e}")
            else:
                if result.error_message:
                    logger.warning(f"❌ Exit failed: {result.error_message}")
                else:
                    logger.info(f"⏳ Exit order placed but not filled (may fill later)")
                    
        except Exception as e:
            logger.error(f"❌ Error executing exit: {e}")
    
    async def _process_signal(self, signal: Signal):
        """Process a single signal"""
        question = signal.metadata.get('question', signal.token_id[:30])[:40]
        
        try:
            # Check if signal is actionable
            if signal.direction.value == "NEUTRAL":
                logger.debug(f"⏭️  Skip: Neutral signal for {question}...")
                return
            
            # Get available capital
            available = self.risk_coordinator.get_available_capital(self.agent_id)
            if available < self.config.risk.min_trade_value_usd:
                logger.warning(
                    f"⚠️  Skip: Insufficient capital (${available:.2f} < "
                    f"${self.config.risk.min_trade_value_usd:.2f} min)"
                )
                return
            
            # Get current price
            bid, ask, spread = await self.api.get_spread(signal.token_id)
            
            # Sanity check: bid should be less than ask
            if bid and ask and bid >= ask:
                logger.warning(
                    f"⚠️  Skip: Invalid orderbook (bid ${bid:.4f} >= ask ${ask:.4f})"
                )
                return
            
            # Note: Spread check moved to after we determine side
            # SELLs should always be allowed to place limit orders to reduce exposure
            
            # Track state for position handling
            held_position = None
            actual_token_id = signal.token_id  # The token we'll actually trade
            opposite_side_trade = False  # Flag to indicate SELL->opposite BUY
            
            if signal.is_buy:
                current_price = ask or signal.metadata.get("price", 0)
                side = Side.BUY
            else:
                # SELL signal: Buy the opposite side instead of shorting
                # This is equivalent economically - if smart money sells "Yes", buying "No"
                # captures the same trade direction
                
                # First, try to find the opposite token
                opposite_token_id = await self._get_opposite_token_id(signal.market_id, signal.token_id)
                
                if opposite_token_id:
                    # Trade the opposite token
                    logger.info(
                        f"📉 SELL signal -> Buying opposite token instead"
                    )
                    actual_token_id = opposite_token_id
                    opposite_side_trade = True
                    
                    # Get price for the opposite token
                    opp_bid, opp_ask, opp_spread = await self.api.get_spread(opposite_token_id)
                    current_price = opp_ask or 0
                    side = Side.BUY
                    
                    # Update spread check for the opposite token
                    if opp_spread and opp_spread > self.config.risk.max_spread_pct:
                        logger.info(
                            f"⏭️  Skip: Opposite token spread too wide ({opp_spread:.1%})"
                        )
                        return
                else:
                    # Fallback to original behavior: sell if we hold the token
                    current_price = bid or signal.metadata.get("price", 0)
                    side = Side.SELL
                    
                    # For SELL, verify we actually hold this token
                    wallet_state = self.risk_coordinator.get_wallet_state()
                    for pos in wallet_state.positions:
                        if pos.token_id == signal.token_id:
                            held_position = pos
                            break
                    
                    if not held_position:
                        logger.debug(
                            f"⏭️  Skip: SELL signal, no opposite token found and don't hold {signal.token_id[:16]}..."
                        )
                        return
            
            if not current_price or current_price <= 0:
                logger.debug(f"⏭️  Skip: No valid price for {question}...")
                return
            
            # Price range filter - avoid extreme prices
            if self.min_price > 0 and current_price < self.min_price:
                logger.info(
                    f"⏭️  Skip: Price ${current_price:.4f} below min ${self.min_price:.2f} "
                    f"(unlikely outcome)"
                )
                return
            
            if self.max_price < 1.0 and current_price > self.max_price:
                logger.info(
                    f"⏭️  Skip: Price ${current_price:.4f} above max ${self.max_price:.2f} "
                    f"(limited upside)"
                )
                return
            
            # Safety check - use permissive check for SELLs (allow through drawdown limits)
            # SELLs reduce exposure and should always be allowed to help recover from drawdowns
            if side == Side.SELL:
                if not self._check_safety_for_sell():
                    logger.info(f"⏭️  Skip: Safety check failed for SELL (manual halt)")
                    return
                # Note: No spread check for SELLs - we always want to allow limit orders to reduce exposure
            else:
                # BUY orders must pass full safety checks
                if not self._check_safety():
                    logger.info(f"⏭️  Skip: Safety check failed for BUY")
                    return
                
                # Spread check only for BUY orders - SELLs should always be able to exit
                if spread and spread > self.config.risk.max_spread_pct:
                    logger.info(
                        f"⏭️  Skip: Spread too wide ({spread:.1%} > {self.config.risk.max_spread_pct:.1%}) "
                        f"[bid=${bid:.4f}, ask=${ask:.4f}]"
                    )
                    return
            
            # Prevent buying in the same market twice
            if side == Side.BUY:
                wallet_state = self.risk_coordinator.get_wallet_state()
                for pos in wallet_state.positions:
                    # Check if we already have a position in this market (any token)
                    if pos.market_id == signal.market_id:
                        logger.info(
                            f"⏭️  Skip: Already have position in market {signal.market_id[:16]}... "
                            f"({pos.shares:.2f} shares of {pos.token_id[:12]}...)"
                        )
                        return
            
            # Calculate position size
            size_usd = self.position_sizer.calculate_size(
                signal, available, current_price
            )
            
            if size_usd < self.config.risk.min_trade_value_usd:
                logger.debug(f"⏭️  Skip: Position too small (${size_usd:.2f})")
                return
            
            # For SELL orders, cap to the shares we actually hold
            if side == Side.SELL and held_position:
                max_sell_value = held_position.shares * current_price
                if size_usd > max_sell_value:
                    logger.info(
                        f"📉 Capping SELL from ${size_usd:.2f} to ${max_sell_value:.2f} "
                        f"(hold {held_position.shares:.4f} shares)"
                    )
                    size_usd = max_sell_value
                    
                    # Re-check minimum after capping
                    if size_usd < self.config.risk.min_trade_value_usd:
                        logger.debug(f"⏭️  Skip: Capped position too small (${size_usd:.2f})")
                        return
            
            # Reserve capital for the token we're actually trading
            reservation_id = self.risk_coordinator.atomic_reserve(
                agent_id=self.agent_id,
                market_id=signal.market_id,
                token_id=actual_token_id,  # Use actual token (may be opposite)
                amount_usd=size_usd
            )
            
            if not reservation_id:
                logger.warning(f"⚠️  Skip: Could not reserve ${size_usd:.2f} capital")
                return
            
            try:
                # Log trade attempt
                side_emoji = "📈" if side == Side.BUY else "📉"
                logger.info(f"{'='*60}")
                if opposite_side_trade:
                    logger.info(f"{side_emoji} EXECUTING {side.value} ORDER (opposite side)")
                else:
                    logger.info(f"{side_emoji} EXECUTING {side.value} ORDER")
                logger.info(f"{'='*60}")
                logger.info(f"  📌 {question}...")
                if opposite_side_trade:
                    logger.info(f"  🔄 Trading opposite token (SELL signal -> BUY opposite)")
                    logger.info(f"  🎯 Token: {actual_token_id[:32]}...")
                logger.info(f"  💵 Size:   ${size_usd:.2f}")
                logger.info(f"  💰 Price:  ${current_price:.4f}")
                logger.info(f"  📊 Score:  {signal.score:.1f}")
                logger.info(f"  📈 Spread: {spread:.2%}" if spread else "  📈 Spread: N/A")
                logger.info(f"  💳 Available: ${available:.2f}")
                
                # Cancel any existing stale orders for this token before placing new one
                # This ensures we don't have conflicting orders (e.g., old take-profit while placing new entry)
                cancelled = await self._cancel_existing_orders(actual_token_id)
                if cancelled > 0:
                    logger.info(f"  🔄 Cancelled {cancelled} existing order(s)")
                
                # Get original signal price (from flow alert) for price drift check
                # For opposite side trades, don't apply price drift check
                original_signal_price = None if opposite_side_trade else signal.metadata.get("price")
                
                # Get market lifetime for time-based slippage threshold
                # Short-duration markets need tighter slippage control
                market_lifetime_hours = signal.metadata.get("market_lifetime_hours")
                
                result = await self.executor.execute(
                    client=self.client,
                    token_id=actual_token_id,  # Use actual token (may be opposite)
                    side=side,
                    size_usd=size_usd,
                    price=current_price,
                    orderbook=None,
                    original_signal_price=original_signal_price,
                    market_lifetime_hours=market_lifetime_hours
                )
                
                if result.success and result.filled_shares > 0:
                    # Confirm execution
                    self.risk_coordinator.confirm_execution(
                        reservation_id=reservation_id,
                        filled_shares=result.filled_shares,
                        filled_price=result.filled_price,
                        requested_shares=result.requested_shares
                    )
                    self.circuit_breaker.record_success()
                    
                    # Save execution history
                    try:
                        with self.risk_coordinator.storage.transaction() as txn:
                            txn.save_execution(
                                agent_id=self.agent_id,
                                market_id=signal.market_id,
                                token_id=actual_token_id,  # Use actual token traded
                                side=side,
                                shares=result.filled_shares,
                                price=current_price,
                                filled_price=result.filled_price,
                                signal_score=signal.score,
                                success=True
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save execution history: {e}")
                    
                    total_cost = result.filled_shares * result.filled_price
                    slippage = abs(result.filled_price - current_price) / current_price if current_price > 0 else 0
                    
                    logger.info(f"{'='*60}")
                    logger.info(f"✅ ORDER FILLED")
                    logger.info(f"{'='*60}")
                    logger.info(f"  📦 Shares:   {result.filled_shares:.4f}")
                    logger.info(f"  💰 Price:    ${result.filled_price:.4f}")
                    logger.info(f"  💵 Total:    ${total_cost:.2f}")
                    logger.info(f"  📉 Slippage: {slippage:.2%}")
                    logger.info(f"{'='*60}")
                    
                    # Register BUY positions with exit monitor for exit strategy tracking
                    if side == Side.BUY and self.exit_monitor:
                        state = self.exit_monitor.register_position(
                            position_id=actual_token_id,
                            entry_price=result.filled_price,
                            entry_time=datetime.now(timezone.utc),
                            shares=result.filled_shares,
                        )
                        logger.info(
                            f"  📝 Registered for exit monitoring "
                            f"(TP: {self.exit_config.take_profit_pct:.0%}, "
                            f"SL: {self.exit_config.stop_loss_pct:.0%}, "
                            f"Max: {self.exit_config.max_hold_minutes}min)"
                        )
                        
                        # Place safety sell order at $0.99 to capture value on resolution
                        safety_order_id = await self._place_safety_sell(
                            token_id=actual_token_id,
                            shares=result.filled_shares,
                            entry_price=result.filled_price,
                        )
                        if safety_order_id and state:
                            state.update_safety_order(safety_order_id)
                else:
                    # Release reservation
                    self.risk_coordinator.release_reservation(reservation_id)
                    
                    # Save failed execution
                    try:
                        with self.risk_coordinator.storage.transaction() as txn:
                            txn.save_execution(
                                agent_id=self.agent_id,
                                market_id=signal.market_id,
                                token_id=actual_token_id,  # Use actual token attempted
                                side=side,
                                shares=result.requested_shares,
                                price=current_price,
                                filled_price=0.0,
                                signal_score=signal.score,
                                success=False,
                                error_message=result.error_message
                            )
                    except Exception as e:
                        logger.warning(f"Failed to save execution history: {e}")
                    
                    if result.error_message:
                        logger.warning(f"❌ Execution failed: {result.error_message}")
                        # Only count actual system errors toward circuit breaker
                        # Protective rejections (price drift, spread, slippage) are expected
                        # behavior during volatile markets, not system failures
                        if not result.is_rejection:
                            self.circuit_breaker.record_failure()
                    else:
                        logger.info("⏳ Order placed but not filled (may fill later)")
                        
            except Exception as e:
                # Release reservation on error
                self.risk_coordinator.release_reservation(reservation_id)
                raise
                
        except Exception as e:
            logger.error(f"❌ Error processing signal for {question}: {e}")
            self.circuit_breaker.record_failure()
    
    async def _get_opposite_token_id(self, market_id: str, token_id: str) -> Optional[str]:
        """
        Get the opposite token ID for a given market and token.
        
        For binary markets (Yes/No), returns the other token.
        Returns None if market not found or not binary.
        """
        try:
            # First check cached market info from risk coordinator
            if self.risk_coordinator and hasattr(self.risk_coordinator, '_market_cache'):
                cached_market = self.risk_coordinator._market_cache.get(market_id)
                if cached_market:
                    tokens = cached_market.get('tokens', [])
                    for t in tokens:
                        if t.get('token_id') != token_id:
                            return t.get('token_id')
            
            # Fallback: fetch from API
            if self.api:
                # Try to get market info
                markets = await self.api.fetch_markets_batch(0, 100)
                for m in markets:
                    if m.get('condition_id') == market_id:
                        tokens = m.get('tokens', [])
                        for t in tokens:
                            if t.get('token_id') != token_id:
                                return t.get('token_id')
                        break
            
            return None
        except Exception as e:
            logger.warning(f"Failed to get opposite token for {market_id}: {e}")
            return None
    
    def get_status(self) -> dict:
        """Get current bot status"""
        wallet_state = None
        if self.risk_coordinator:
            wallet_state = self.risk_coordinator.get_wallet_state()
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "running": self.running,
            "dry_run": self.dry_run,
            "signal_source": self.signal_source.name if self.signal_source else None,
            "position_sizer": self.position_sizer.name if self.position_sizer else None,
            "executor": self.executor.name if self.executor else None,
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
            },
            "drawdown": self.drawdown_limit.get_status(),
            "trading_halt": {
                "is_halted": self.trading_halt.is_halted,
                "reasons": self.trading_halt.reasons,
            },
            "wallet": {
                "balance": wallet_state.usdc_balance if wallet_state else 0,
                "positions_value": wallet_state.total_positions_value if wallet_state else 0,
                "reserved": wallet_state.total_reserved if wallet_state else 0,
                "available": wallet_state.available_capital if wallet_state else 0,
            } if wallet_state else None,
        }


