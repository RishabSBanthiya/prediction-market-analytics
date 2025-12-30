"""
Bond Strategy - Expiring Market Trading Bot.

This strategy trades on markets near expiration where prices
are in the 95c-98c range, betting they'll resolve to $1.

Named "bond" because these trades behave like short-term bonds -
high probability of small gain, low probability of total loss.

Includes hedging capability to protect against fat-tail losses:
1. Arbitrage (YES + NO < 1)
2. Protective hedge (buy opposite outcome)
3. Partial exit (reduce exposure)
4. Stop-loss (full exit)
"""

import asyncio
import logging
from typing import Optional, List, Dict
from datetime import datetime, timezone

from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from ..core.models import Market, Signal, Position
from ..trading.bot import TradingBot
from ..trading.components.signals import ExpiringMarketSignals
from ..trading.components.sizers import KellyPositionSizer
from ..trading.components.executors import AggressiveExecutor, DryRunExecutor
from ..trading.components.hedge_monitor import (
    HedgeMonitor,
    HedgeConfig,
    HedgeRecommendation,
    HedgeAction,
)
from ..trading.components.hedge_strategies import HedgeExecutor

logger = logging.getLogger(__name__)


def format_time_remaining(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


# Time bucket configuration for diversification
# Format: (max_seconds, weight, max_positions)
TIME_BUCKETS = [
    (120, 4.0, 8),      # <2 min: 4x weight, up to 8 positions
    (300, 3.0, 6),      # 2-5 min: 3x weight, up to 6 positions
    (900, 2.0, 4),      # 5-15 min: 2x weight, up to 4 positions
    (1800, 1.5, 3),     # 15-30 min: 1.5x weight, up to 3 positions
    (float('inf'), 1.0, 2),  # 30+ min: 1x weight, up to 2 positions
]


class BondSignalSource(ExpiringMarketSignals):
    """
    Extended expiring market signal source with market fetching.
    
    Also tracks market tokens for hedge capabilities.
    """
    
    def __init__(
        self,
        api: PolymarketAPI,
        min_price: float = 0.95,
        max_price: float = 0.98,
        min_seconds_left: int = 60,
        max_seconds_left: int = 1800,
        refresh_interval: int = 30,
    ):
        super().__init__(min_price, max_price, min_seconds_left, max_seconds_left)
        self.api = api
        self.refresh_interval = refresh_interval
        self._last_refresh: Optional[datetime] = None
        self._scan_count = 0
        self._last_opportunity_count = 0
        
        # Track market data for hedging (token_id -> Market)
        self._token_to_market: Dict[str, Market] = {}
    
    async def get_signals(self) -> List[Signal]:
        """Get signals, refreshing markets if needed"""
        now = datetime.now(timezone.utc)
        
        # Refresh markets periodically
        if (self._last_refresh is None or 
            (now - self._last_refresh).total_seconds() > self.refresh_interval):
            await self._refresh_markets()
            self._last_refresh = now
        
        signals = await super().get_signals()
        
        self._scan_count += 1
        
        # Log scan summary periodically or when opportunities change
        if signals or self._scan_count % 12 == 0:  # Every ~minute at 5s interval
            self._log_scan_summary(signals)
        
        return signals
    
    def _log_scan_summary(self, signals: List[Signal]):
        """Log detailed scan summary"""
        expiring_count = len([m for m in self._markets if self._is_expiring_soon(m)])
        
        if not signals:
            if expiring_count > 0:
                logger.info(
                    f"📊 Scan #{self._scan_count}: {expiring_count} expiring markets, "
                    f"0 in price range ${self.min_price:.2f}-${self.max_price:.2f}"
                )
            else:
                logger.debug(f"📊 Scan #{self._scan_count}: No expiring markets found")
            return
        
        logger.info(f"{'='*60}")
        logger.info(f"🎯 BOND OPPORTUNITIES FOUND: {len(signals)}")
        logger.info(f"{'='*60}")
        
        for i, signal in enumerate(signals, 1):
            time_left = signal.metadata.get('seconds_left', 0)
            price = signal.metadata.get('price', 0)
            expected_return = ((1.0 / price) - 1.0) * 100 if price > 0 else 0
            question = signal.metadata.get('question', 'Unknown')[:50]
            
            # Determine time bucket
            bucket = "30m+"
            for max_sec, weight, _ in TIME_BUCKETS:
                if time_left <= max_sec:
                    if max_sec <= 120:
                        bucket = "<2m ⚡"
                    elif max_sec <= 300:
                        bucket = "2-5m 🔥"
                    elif max_sec <= 900:
                        bucket = "5-15m"
                    elif max_sec <= 1800:
                        bucket = "15-30m"
                    break
            
            logger.info(
                f"  [{i}] {question}..."
            )
            logger.info(
                f"      💰 Price: ${price:.4f} | "
                f"⏱️  Time: {format_time_remaining(time_left)} ({bucket}) | "
                f"📈 Expected: +{expected_return:.1f}%"
            )
        
        logger.info(f"{'='*60}")
        self._last_opportunity_count = len(signals)
    
    def _is_expiring_soon(self, market: Market) -> bool:
        """Check if market is expiring within our window"""
        if not market.end_date:
            return False
        now = datetime.now(timezone.utc)
        time_left = (market.end_date - now).total_seconds()
        return self.min_seconds_left <= time_left <= self.max_seconds_left
    
    async def _refresh_markets(self):
        """Fetch and parse active markets"""
        logger.info("🔄 Refreshing markets from Polymarket API...")
        
        raw_markets = await self.api.fetch_all_markets()
        
        markets = []
        expired_count = 0
        closed_count = 0
        
        for raw in raw_markets:
            market = self.api.parse_market(raw)
            if market:
                if market.is_expired:
                    expired_count += 1
                elif market.closed:
                    closed_count += 1
                else:
                    markets.append(market)
                    # Map token IDs to market for hedging
                    for token in market.tokens:
                        self._token_to_market[token.token_id] = market
        
        self.update_markets(markets)
        
        # Count expiring markets in our window
        expiring = [m for m in markets if self._is_expiring_soon(m)]
        
        logger.info(
            f"📥 Loaded {len(markets)} active markets "
            f"(skipped: {expired_count} expired, {closed_count} closed)"
        )
        if expiring:
            logger.info(
                f"⏰ {len(expiring)} markets expiring in {self.min_seconds_left}-{self.max_seconds_left}s window"
            )
    
    def get_market_for_token(self, token_id: str) -> Optional[Market]:
        """Get the market containing a token (for hedging)"""
        return self._token_to_market.get(token_id)


class HedgedBondBot:
    """
    Bond bot with integrated hedge monitoring.
    
    Wraps TradingBot and adds:
    - Position monitoring for adverse movements
    - Cascading hedge execution (arb -> hedge -> partial -> stop-loss)
    """
    
    def __init__(
        self,
        bot: TradingBot,
        api: PolymarketAPI,
        signal_source: BondSignalSource,
        hedge_config: Optional[HedgeConfig] = None,
        dry_run: bool = False,
    ):
        self.bot = bot
        self.api = api
        self.signal_source = signal_source
        self.hedge_config = hedge_config or HedgeConfig()
        self.dry_run = dry_run
        
        # Hedge components
        self.hedge_monitor: Optional[HedgeMonitor] = None
        self.hedge_executor: Optional[HedgeExecutor] = None
        
        # Track positions we've traded
        self._tracked_positions: Dict[str, Position] = {}
    
    async def start(self):
        """Start the bot and hedge monitor"""
        await self.bot.start()
        
        # Initialize hedge monitor
        self.hedge_monitor = HedgeMonitor(
            api=self.api,
            config=self.hedge_config,
            on_hedge_triggered=self._on_hedge_triggered,
        )
        
        # Initialize hedge executor
        self.hedge_executor = HedgeExecutor(
            api=self.api,
            executor=self.bot.executor,
            client=self.bot.client,
            config=self.hedge_config,
        )
        
        # Start monitoring
        await self.hedge_monitor.start()
        
        logger.info(f"{'='*60}")
        logger.info(f"🛡️ HEDGE MONITOR ACTIVE")
        logger.info(f"{'='*60}")
        logger.info(f"  Price Drop Trigger: {self.hedge_config.price_drop_trigger_pct:.0%}")
        logger.info(f"  Min Arb Profit:     {self.hedge_config.min_arb_profit_pct:.0%}")
        logger.info(f"  Stop-Loss:          {self.hedge_config.stop_loss_pct:.0%}")
        logger.info(f"{'='*60}")
    
    async def stop(self):
        """Stop the bot and hedge monitor"""
        if self.hedge_monitor:
            await self.hedge_monitor.stop()
        await self.bot.stop()
    
    async def run(self, interval_seconds: float = 5.0):
        """Run the trading loop with hedge monitoring"""
        if not self.bot.running:
            raise RuntimeError("Bot not started. Call start() first.")
        
        logger.info(f"Starting hedged bond bot loop (interval={interval_seconds}s)")
        
        try:
            while self.bot.running:
                try:
                    # Run trading iteration
                    await self.bot._trading_iteration()
                    
                    # Update tracked positions from risk coordinator
                    await self._sync_positions()
                    
                    # Check for hedge opportunities
                    await self._check_hedges()
                    
                except Exception as e:
                    logger.error(f"Error in trading iteration: {e}")
                    self.bot.circuit_breaker.record_failure()
                
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
    
    async def _sync_positions(self):
        """Sync positions from risk coordinator to hedge monitor"""
        if not self.bot.risk_coordinator or not self.hedge_monitor:
            return
        
        wallet_state = self.bot.risk_coordinator.get_wallet_state()
        
        for position in wallet_state.positions:
            # Skip if already tracked
            if position.token_id in self._tracked_positions:
                continue
            
            # Get market info for hedging
            market = self.signal_source.get_market_for_token(position.token_id)
            
            if market:
                self.hedge_monitor.add_position(
                    position=position,
                    market_tokens=market.tokens,
                )
                self._tracked_positions[position.token_id] = position
                logger.info(f"📊 Added position to hedge monitor: {position.outcome}")
    
    async def _check_hedges(self):
        """Check positions and execute hedges if needed"""
        if not self.hedge_monitor or not self.hedge_executor:
            return
        
        recommendations = await self.hedge_monitor.check_all_positions()
        
        for rec in recommendations:
            if self.dry_run:
                logger.info(
                    f"🧪 DRY RUN: Would execute {rec.action.value} for {rec.position.position.outcome}"
                )
                continue
            
            # Get available capital
            available = self.bot.risk_coordinator.get_available_capital(self.bot.agent_id)
            
            # Execute hedge
            result = await self.hedge_executor.execute_hedge(rec, available)
            
            if result.success:
                self.hedge_monitor.mark_hedge_executed(
                    rec.position.position.token_id,
                    rec.action,
                )
                
                # If stop-loss or full exit, remove from tracking
                if rec.action == HedgeAction.STOP_LOSS:
                    self._tracked_positions.pop(rec.position.position.token_id, None)
                    self.hedge_monitor.remove_position(rec.position.position.token_id)
    
    def _on_hedge_triggered(self, recommendation: HedgeRecommendation):
        """Callback when hedge is triggered"""
        logger.warning(
            f"⚠️ HEDGE TRIGGERED: {recommendation.action.value} for "
            f"{recommendation.position.position.outcome} - {recommendation.reason}"
        )


def create_bond_bot(
    agent_id: str = "bond-bot",
    config: Optional[Config] = None,
    dry_run: bool = False,
    min_price: float = 0.95,
    max_price: float = 0.98,
    enable_hedging: bool = True,
    hedge_config: Optional[HedgeConfig] = None,
) -> "HedgedBondBot":
    """
    Create a bond strategy trading bot with hedging.
    
    Args:
        agent_id: Unique identifier for this agent
        config: Configuration (uses default if not provided)
        dry_run: If True, simulate trades without execution
        min_price: Minimum price to consider (default 0.95)
        max_price: Maximum price to consider (default 0.98)
        enable_hedging: If True, enable hedge monitoring (default True)
        hedge_config: Hedge configuration (uses defaults if not provided)
    
    Returns:
        Configured HedgedBondBot ready to start
    """
    config = config or get_config()
    hedge_config = hedge_config or HedgeConfig()
    
    # Log strategy configuration
    logger.info(f"{'='*60}")
    logger.info(f"🏦 BOND STRATEGY CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Agent ID:      {agent_id}")
    logger.info(f"  Mode:          {'🧪 DRY RUN' if dry_run else '💸 LIVE TRADING'}")
    logger.info(f"  Price Range:   ${min_price:.2f} - ${max_price:.2f}")
    logger.info(f"  Time Window:   60s - 1800s (30 min)")
    logger.info(f"  Position Size: Half-Kelly (max 25%)")
    logger.info(f"  Hedging:       {'✅ ENABLED' if enable_hedging else '❌ DISABLED'}")
    expected_return_min = ((1.0 / max_price) - 1.0) * 100
    expected_return_max = ((1.0 / min_price) - 1.0) * 100
    logger.info(f"  Expected Returns: +{expected_return_min:.1f}% to +{expected_return_max:.1f}%")
    logger.info(f"{'='*60}")
    
    # Create API for signal source
    api = PolymarketAPI(config)
    
    # Create components
    signal_source = BondSignalSource(
        api=api,
        min_price=min_price,
        max_price=max_price,
        min_seconds_left=60,
        max_seconds_left=1800,
    )
    
    position_sizer = KellyPositionSizer(
        kelly_fraction=0.5,  # Half Kelly for safety
        min_edge=0.02,
        max_kelly=0.25,
        price_range=(min_price, max_price)
    )
    
    executor = DryRunExecutor() if dry_run else AggressiveExecutor(max_slippage=0.02)
    
    # Create base bot
    bot = TradingBot(
        agent_id=agent_id,
        agent_type="bond",
        signal_source=signal_source,
        position_sizer=position_sizer,
        executor=executor,
        config=config,
        dry_run=dry_run,
    )
    
    # Wrap with hedging if enabled
    return HedgedBondBot(
        bot=bot,
        api=api,
        signal_source=signal_source,
        hedge_config=hedge_config if enable_hedging else None,
        dry_run=dry_run,
    )


async def run_bond_bot(
    agent_id: str = "bond-bot",
    dry_run: bool = False,
    interval: float = 5.0,
    enable_hedging: bool = True,
    hedge_config: Optional[HedgeConfig] = None,
):
    """
    Run the bond strategy bot with hedging.
    
    This is a convenience function for running the bot directly.
    
    Args:
        agent_id: Unique identifier for this agent
        dry_run: If True, simulate trades without execution
        interval: Scan interval in seconds
        enable_hedging: If True, enable hedge monitoring
        hedge_config: Hedge configuration (uses defaults if not provided)
    """
    bot = create_bond_bot(
        agent_id=agent_id,
        dry_run=dry_run,
        enable_hedging=enable_hedging,
        hedge_config=hedge_config,
    )
    
    try:
        await bot.start()
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bond Strategy Trading Bot")
    parser.add_argument("--agent-id", default="bond-bot", help="Agent ID")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--interval", type=float, default=5.0, help="Scan interval")
    parser.add_argument("--no-hedge", action="store_true", help="Disable hedging")
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop-loss threshold (default 15%)")
    parser.add_argument("--hedge-trigger", type=float, default=0.05, help="Hedge trigger threshold (default 5%)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Build hedge config from args
    hedge_config = HedgeConfig(
        stop_loss_pct=args.stop_loss,
        price_drop_trigger_pct=args.hedge_trigger,
    )
    
    asyncio.run(run_bond_bot(
        agent_id=args.agent_id,
        dry_run=args.dry_run,
        interval=args.interval,
        enable_hedging=not args.no_hedge,
        hedge_config=hedge_config,
    ))


