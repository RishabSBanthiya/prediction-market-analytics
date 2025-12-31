"""
Flow Copy Strategy - Copy Trade Unusual Flow Alerts.

This strategy monitors flow detection alerts and copy trades
signals from smart money, oversized bets, and coordinated wallets.

Features:
- Signal deduplication to avoid double-counting
- Weighted composite scoring
- Signal-scaled position sizing
- Exit strategies (take-profit, trailing stop, time-based)
"""

import asyncio
import logging
from typing import Optional, List, Dict, TYPE_CHECKING
from datetime import datetime, timezone
from collections import defaultdict

from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from ..core.models import Signal
from ..trading.bot import TradingBot
from ..trading.components.signals import FlowAlertSignals, SIGNAL_WEIGHTS
from ..trading.components.sizers import SignalScaledSizer
from ..trading.components.executors import AggressiveExecutor, DryRunExecutor
from ..trading.components.exit_strategies import ExitConfig

if TYPE_CHECKING:
    from ..flow_detector import TradeFeedFlowDetector, FlowAlert
    from ..trading.storage.base import StorageBackend

logger = logging.getLogger(__name__)


# Alert type emojis for better visibility
ALERT_EMOJIS = {
    "smart_money": "🧠",
    "whale": "🐋",
    "coordinated": "🤝",
    "momentum": "📈",
    "unusual_size": "💰",
    "breakout": "🚀",
}

# Severity colors
SEVERITY_INDICATORS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}


class FlowCopySignalSource(FlowAlertSignals):
    """
    Extended flow alert signal source with integrated flow detector.
    """
    
    def __init__(
        self,
        dedup_window_seconds: int = 30,
        decay_half_life_seconds: float = 60.0,
        min_score: float = 30.0,
        min_trade_size: float = 100.0,
        category_filter: Optional[str] = None,
        storage: Optional["StorageBackend"] = None,
    ):
        super().__init__(dedup_window_seconds, decay_half_life_seconds, min_score)
        self.min_trade_size = min_trade_size
        self.category_filter = category_filter
        self.storage = storage
        self._detector: Optional["TradeFeedFlowDetector"] = None
        self._detector_task: Optional[asyncio.Task] = None
        
        # Stats tracking
        self._alert_count = 0
        self._alert_by_type: Dict[str, int] = defaultdict(int)
        self._signal_count = 0
        self._last_stats_log = datetime.now(timezone.utc)
    
    async def get_signals(self) -> List[Signal]:
        """Get signals and log activity"""
        signals = await super().get_signals()
        
        if signals:
            self._signal_count += len(signals)
            self._log_signals(signals)
        
        # Log stats every 5 minutes
        now = datetime.now(timezone.utc)
        if (now - self._last_stats_log).total_seconds() > 300:
            self._log_stats()
            self._last_stats_log = now
        
        return signals
    
    def _log_signals(self, signals: List[Signal]):
        """Log actionable signals"""
        logger.info(f"{'='*60}")
        logger.info(f"🎯 FLOW SIGNALS READY TO TRADE: {len(signals)}")
        logger.info(f"{'='*60}")
        
        for i, signal in enumerate(signals, 1):
            direction = "📈 BUY" if signal.is_buy else "📉 SELL"
            question = signal.metadata.get('question', 'Unknown')[:45]
            
            logger.info(f"  [{i}] {question}...")
            logger.info(
                f"      {direction} | "
                f"Score: {signal.score:.1f} | "
                f"Token: {signal.token_id[:16]}..."
            )
            
            # Log contributing factors if available
            if 'alert_types' in signal.metadata:
                types = signal.metadata['alert_types']
                type_str = ", ".join(
                    f"{ALERT_EMOJIS.get(t, '📊')} {t}" for t in types[:3]
                )
                logger.info(f"      Factors: {type_str}")
        
        logger.info(f"{'='*60}")
    
    def _log_stats(self):
        """Log periodic stats summary"""
        if self._alert_count == 0:
            logger.info("📊 Flow Stats: No alerts received in last 5 minutes")
            return
        
        logger.info(f"{'='*60}")
        logger.info(f"📊 FLOW DETECTOR STATS (Last 5 min)")
        logger.info(f"{'='*60}")
        logger.info(f"  Total Alerts:  {self._alert_count}")
        logger.info(f"  Signals Made:  {self._signal_count}")
        
        if self._alert_by_type:
            logger.info(f"  Alert Types:")
            for alert_type, count in sorted(self._alert_by_type.items(), key=lambda x: -x[1]):
                emoji = ALERT_EMOJIS.get(alert_type, "📊")
                logger.info(f"    {emoji} {alert_type}: {count}")
        
        logger.info(f"{'='*60}")
        
        # Reset counters
        self._alert_count = 0
        self._alert_by_type.clear()
        self._signal_count = 0
    
    async def start_detector(self):
        """Start the flow detector in background"""
        try:
            from ..flow_detector import TradeFeedFlowDetector, MarketCategory
            
            category = MarketCategory.ALL
            if self.category_filter:
                try:
                    category = MarketCategory(self.category_filter.lower())
                except ValueError:
                    logger.warning(f"Unknown category: {self.category_filter}, using ALL")
            
            self._detector = TradeFeedFlowDetector(
                min_trade_size=self.min_trade_size,
                verbose=False,
                category_filter=category,
                storage=self.storage
            )
            
            # Set up callback
            self._detector.alert_callback = self._on_alert
            
            # Start detector in background
            self._detector_task = asyncio.create_task(self._run_detector())
            
            logger.info("🔌 Flow detector connected to Polymarket WebSocket")
            logger.info(f"   Monitoring trades >= ${self.min_trade_size:.0f}")
            if self.category_filter:
                logger.info(f"   Category filter: {self.category_filter}")
            
        except ImportError as e:
            logger.error(f"Could not import flow_detector: {e}")
            raise
    
    async def _run_detector(self):
        """Run the flow detector"""
        try:
            await self._detector.start()
        except asyncio.CancelledError:
            logger.info("Flow detector cancelled")
        except Exception as e:
            logger.error(f"Flow detector error: {e}")
    
    async def stop_detector(self):
        """Stop the flow detector"""
        logger.info("🔌 Stopping flow detector...")
        
        if self._detector:
            await self._detector.stop()
        
        if self._detector_task:
            self._detector_task.cancel()
            try:
                await self._detector_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Flow detector stopped")
    
    def _on_alert(self, alert: "FlowAlert"):
        """Callback for flow detector alerts"""
        # Add directly to recent_alerts to avoid recursion 
        key = f"{alert.market_id}:{alert.token_id}"
        self.recent_alerts[key].append(alert)
        
        # Track stats
        self._alert_count += 1
        self._alert_by_type[alert.alert_type] += 1
        
        # Get emoji for alert type and severity
        type_emoji = ALERT_EMOJIS.get(alert.alert_type, "📊")
        sev_indicator = SEVERITY_INDICATORS.get(alert.severity, "⚪")
        
        # Format trade size
        size_str = f"${alert.trade_size:,.0f}" if hasattr(alert, 'trade_size') else ""
        
        logger.info(
            f"{sev_indicator} {type_emoji} ALERT: {alert.alert_type.upper()} | "
            f"{alert.severity} | {size_str}"
        )
        logger.info(
            f"   📌 {alert.question[:60]}{'...' if len(alert.question) > 60 else ''}"
        )
        
        # Log additional details for high-value alerts
        if alert.severity in ("critical", "high"):
            if hasattr(alert, 'wallet_address') and alert.wallet_address:
                logger.info(f"   👤 Wallet: {alert.wallet_address[:10]}...")
            if hasattr(alert, 'outcome') and alert.outcome:
                logger.info(f"   🎯 Outcome: {alert.outcome}")


def create_flow_bot(
    agent_id: str = "flow-bot",
    config: Optional[Config] = None,
    dry_run: bool = False,
    min_score: float = 30.0,
    min_trade_size: float = 100.0,
    category: Optional[str] = None,
    max_spread: float = 0.03,  # 3% max spread (optimized)
    max_price_drift: float = 0.10,  # 10% max price drift from original trade
    exit_config: Optional[ExitConfig] = None,  # Exit strategy configuration
    # Optimized parameters from Bayesian optimization (53.83% return, 72.5% win rate)
    base_position_pct: float = 0.035,  # 3.5% base position (optimized)
    max_position_multiplier: float = 4.0,  # 4x max multiplier (optimized)
    # Price range filter - avoid extreme prices
    min_price: float = 0.20,  # Don't buy tokens priced below 20c (unlikely outcomes)
    max_price: float = 0.80,  # Don't buy tokens priced above 80c (limited upside)
) -> TradingBot:
    """
    Create a flow copy strategy trading bot.
    
    Args:
        agent_id: Unique identifier for this agent
        config: Configuration (uses default if not provided)
        dry_run: If True, simulate trades without execution
        min_score: Minimum composite score to trade
        min_trade_size: Minimum trade size to track in flow detector
        category: Market category filter (crypto, sports, politics, etc.)
        max_spread: Maximum bid-ask spread allowed (default 3%)
        max_price_drift: Maximum price drift from original signal (default 10%)
    
    Returns:
        Configured TradingBot ready to start
    """
    config = config or get_config()
    
    # Log strategy configuration
    logger.info(f"{'='*60}")
    logger.info(f"🌊 FLOW COPY STRATEGY CONFIGURATION")
    logger.info(f"{'='*60}")
    logger.info(f"  Agent ID:       {agent_id}")
    logger.info(f"  Mode:           {'🧪 DRY RUN' if dry_run else '💸 LIVE TRADING'}")
    logger.info(f"  Min Score:      {min_score:.0f}")
    logger.info(f"  Min Trade Size: ${min_trade_size:,.0f}")
    logger.info(f"  Category:       {category or 'ALL'}")
    logger.info(f"  Position Size:  {base_position_pct:.1%} base, up to {max_position_multiplier:.0f}x for high scores")
    logger.info(f"  Max Spread:     {max_spread:.0%}")
    logger.info(f"  Max Price Drift: {max_price_drift:.0%} (fallback)")
    logger.info(f"  Time-Based Slippage: Enabled (1% for <5min, 3% for <30min, 10% for >30min)")
    logger.info(f"  Price Range:    ${min_price:.2f} - ${max_price:.2f}")
    
    # Exit strategy config with optimized defaults from Bayesian optimization
    exit_cfg = exit_config or ExitConfig(
        take_profit_pct=0.05,  # 5% take-profit (optimized)
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=0.02,  # 2% activation (optimized)
        trailing_stop_distance_pct=0.01,  # 1% trail distance (optimized)
        max_hold_minutes=75,  # 75 min max hold (optimized)
        stop_loss_pct=0.25,  # 25% stop-loss (optimized)
    )
    logger.info(f"{'='*60}")
    logger.info(f"  Exit Strategy:")
    logger.info(f"    Take-Profit:    {exit_cfg.take_profit_pct:.0%}")
    logger.info(f"    Trailing Stop:  {'Enabled' if exit_cfg.trailing_stop_enabled else 'Disabled'}")
    logger.info(f"    Max Hold Time:  {exit_cfg.max_hold_minutes} min")
    logger.info(f"    Stop-Loss:      {exit_cfg.stop_loss_pct:.0%}")
    logger.info(f"{'='*60}")
    logger.info(f"  Signal Weights:")
    for signal_type, weight in SIGNAL_WEIGHTS.items():
        emoji = ALERT_EMOJIS.get(signal_type, "📊")
        logger.info(f"    {emoji} {signal_type}: {weight:.1f}x")
    logger.info(f"{'='*60}")
    
    # Create storage for persisting flow alerts
    from ..trading.storage.sqlite import SQLiteStorage
    storage = SQLiteStorage(config.db_path)
    
    # Create components
    signal_source = FlowCopySignalSource(
        dedup_window_seconds=30,
        decay_half_life_seconds=60.0,
        min_score=min_score,
        min_trade_size=min_trade_size,
        category_filter=category,
        storage=storage,
    )
    
    position_sizer = SignalScaledSizer(
        base_fraction=base_position_pct,  # Optimized: 3.5% base position
        reference_score=50.0,  # Score of 50 = 1x
        scale_factor=1.0,
        min_score=min_score,
        max_multiplier=max_position_multiplier,  # Optimized: 4x max position
    )
    
    executor = DryRunExecutor() if dry_run else AggressiveExecutor(
        max_slippage=0.02,
        max_spread=max_spread,
        max_price_drift=max_price_drift
    )
    
    # Create bot with price range filter and exit strategy
    bot = TradingBot(
        agent_id=agent_id,
        agent_type="flow",
        signal_source=signal_source,
        position_sizer=position_sizer,
        executor=executor,
        config=config,
        dry_run=dry_run,
        min_price=min_price,  # Filter out unlikely outcomes (< 20c)
        max_price=max_price,  # Filter out limited upside (> 80c)
        exit_config=exit_cfg,  # Exit strategy (take-profit, stop-loss, trailing, time-based)
    )
    
    return bot


async def run_flow_bot(
    agent_id: str = "flow-bot",
    dry_run: bool = False,
    interval: float = 2.0,
    min_score: float = 30.0,
    min_trade_size: float = 100.0,
    category: Optional[str] = None,
    max_spread: float = 0.03,
    max_price_drift: float = 0.10,
    exit_config: Optional[ExitConfig] = None,
    min_price: float = 0.20,
    max_price: float = 0.80,
):
    """
    Run the flow copy strategy bot.
    
    This is a convenience function for running the bot directly.
    """
    bot = create_flow_bot(
        agent_id=agent_id,
        dry_run=dry_run,
        min_score=min_score,
        min_trade_size=min_trade_size,
        category=category,
        max_spread=max_spread,
        max_price_drift=max_price_drift,
        exit_config=exit_config,
        min_price=min_price,
        max_price=max_price,
    )
    
    signal_source: FlowCopySignalSource = bot.signal_source
    
    try:
        await bot.start()
        
        # Start flow detector
        await signal_source.start_detector()
        
        # Run main trading loop
        await bot.run(interval_seconds=interval)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop detector first
        await signal_source.stop_detector()
        await bot.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Flow Copy Strategy Trading Bot")
    parser.add_argument("--agent-id", default="flow-bot", help="Agent ID")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--interval", type=float, default=2.0, help="Scan interval")
    parser.add_argument("--min-score", type=float, default=30.0, help="Minimum signal score")
    parser.add_argument("--min-trade-size", type=float, default=100.0, help="Min trade size to track")
    parser.add_argument("--category", type=str, default=None, help="Market category filter")
    parser.add_argument("--max-spread", type=float, default=0.03, help="Max bid-ask spread (default: 0.03 = 3%%)")
    parser.add_argument("--max-price-drift", type=float, default=0.10, help="Max price drift from signal (default: 0.10 = 10%%)")
    parser.add_argument("--min-price", type=float, default=0.20, help="Min token price to trade (default: 0.20 = 20c)")
    parser.add_argument("--max-price", type=float, default=0.80, help="Max token price to trade (default: 0.80 = 80c)")
    parser.add_argument("--take-profit", type=float, default=0.05, help="Take-profit threshold (default: 0.05 = 5%%, optimized)")
    parser.add_argument("--trailing-activation", type=float, default=0.02, help="Trailing stop activation (default: 0.02 = 2%%, optimized)")
    parser.add_argument("--trailing-distance", type=float, default=0.01, help="Trailing stop distance (default: 0.01 = 1%%, optimized)")
    parser.add_argument("--max-hold-minutes", type=int, default=75, help="Max hold time in minutes (default: 75, optimized)")
    parser.add_argument("--stop-loss", type=float, default=0.25, help="Stop-loss threshold (default: 0.25 = 25%%, optimized)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Build exit config from args (optimized defaults)
    exit_cfg = ExitConfig(
        take_profit_pct=args.take_profit,
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=args.trailing_activation,
        trailing_stop_distance_pct=args.trailing_distance,
        max_hold_minutes=args.max_hold_minutes,
        stop_loss_pct=args.stop_loss,
    )
    
    asyncio.run(run_flow_bot(
        agent_id=args.agent_id,
        dry_run=args.dry_run,
        interval=args.interval,
        min_score=args.min_score,
        min_trade_size=args.min_trade_size,
        category=args.category,
        max_spread=args.max_spread,
        max_price_drift=args.max_price_drift,
        exit_config=exit_cfg,
        min_price=args.min_price,
        max_price=args.max_price,
    ))

