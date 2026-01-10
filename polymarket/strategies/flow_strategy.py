"""
Flow Copy Strategy - Copy Trade Unusual Flow Alerts (V6 Optimized).

This strategy monitors flow detection alerts and copy trades
signals from smart money, oversized bets, and coordinated wallets.

V6 Optimizations (based on real flow alert backtest):
- Stop Loss: 20% (wider to avoid whipsaws)
- Take Profit: 50% (let winners run to resolution)
- Price Range: 50c-90c (mid-range prices have best win rate)
- Alert Type: SMART_MONEY_ACTIVITY has 75% win rate
- Severity: HIGH outperforms CRITICAL
- Max Hold: 120 min (longer to capture resolution)

Features:
- Signal deduplication to avoid double-counting
- Weighted composite scoring
- Signal-scaled position sizing
- Exit strategies (take-profit, trailing stop, time-based)

Enhanced Features (V4+):
- Wallet reputation scoring - Track wallet historical accuracy
- Multi-signal confirmation - Require multiple confirming signals
- Information timing filter - Only trade on early signals
- Contrarian signal detection - Fade retail when smart money disagrees
- Market context filters - Adjust signal weight based on market characteristics
- Dynamic position sizing - Scale position with signal confidence
- Adaptive exits - Exit parameters based on market dynamics
"""

import asyncio
import logging
from typing import Optional, List, Dict, TYPE_CHECKING
from datetime import datetime, timezone
from collections import defaultdict

from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from ..core.models import Signal, SignalDirection
from ..trading.bot import TradingBot
from ..trading.components.signals import FlowAlertSignals, SIGNAL_WEIGHTS
from ..trading.components.sizers import SignalScaledSizer
from ..trading.components.executors import AggressiveExecutor, DryRunExecutor
from ..trading.components.exit_strategies import ExitConfig
from ..trading.components.flow_enhancements import (
    EnhancedFlowSignalSource,
    WalletReputationTracker,
    MultiSignalConfirmation,
    MarketContext,
    SignalCluster,
)

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
        min_trade_size: float = 250.0,
        category_filter: Optional[str] = None,
        storage: Optional["StorageBackend"] = None,
        # V5 Signal Quality Filters
        filter_sports: bool = True,
        filter_resolved: bool = True,
        min_market_lifetime_hours: float = 2.0,
        min_liquidity_volume: float = 5000.0,
    ):
        super().__init__(
            dedup_window_seconds=dedup_window_seconds,
            decay_half_life_seconds=decay_half_life_seconds,
            min_score=min_score,
            filter_sports=filter_sports,
            filter_resolved=filter_resolved,
            min_market_lifetime_hours=min_market_lifetime_hours,
            min_liquidity_volume=min_liquidity_volume,
        )
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


class EnhancedFlowCopySignalSource(FlowCopySignalSource):
    """
    Enhanced flow signal source with V4 improvements.

    Integrates:
    - Wallet reputation tracking
    - Multi-signal confirmation
    - Information timing filtering
    - Contrarian signal detection
    - Market context adjustment
    - Dynamic position sizing
    - Adaptive exit parameters
    - V5: Sports/resolved/liquidity filtering
    """

    def __init__(
        self,
        dedup_window_seconds: int = 30,
        decay_half_life_seconds: float = 60.0,
        min_score: float = 30.0,
        min_trade_size: float = 250.0,
        category_filter: Optional[str] = None,
        storage: Optional["StorageBackend"] = None,
        # V4 Enhancement options
        require_confirmation: bool = True,
        enable_contrarian: bool = True,
        enable_timing_filter: bool = True,
        enable_wallet_scoring: bool = True,
        # V5 Signal Quality Filters
        filter_sports: bool = True,
        filter_resolved: bool = True,
        min_market_lifetime_hours: float = 2.0,
        min_liquidity_volume: float = 5000.0,
    ):
        super().__init__(
            dedup_window_seconds=dedup_window_seconds,
            decay_half_life_seconds=decay_half_life_seconds,
            min_score=min_score,
            min_trade_size=min_trade_size,
            category_filter=category_filter,
            storage=storage,
            filter_sports=filter_sports,
            filter_resolved=filter_resolved,
            min_market_lifetime_hours=min_market_lifetime_hours,
            min_liquidity_volume=min_liquidity_volume,
        )

        # V4 Enhancement options
        self.require_confirmation = require_confirmation
        self.enable_contrarian = enable_contrarian
        self.enable_timing_filter = enable_timing_filter
        self.enable_wallet_scoring = enable_wallet_scoring

        # Initialize V4 components
        self._enhanced_source = EnhancedFlowSignalSource(
            min_score=min_score,
            require_confirmation=require_confirmation,
            enable_contrarian=enable_contrarian,
            enable_timing_filter=enable_timing_filter,
        )

        # Track enhanced signals
        self._enhanced_signals: List[Dict] = []
        self._pending_clusters: Dict[str, SignalCluster] = {}

    async def get_signals(self) -> List[Signal]:
        """Get enhanced signals with V4 improvements."""
        # Get base signals
        base_signals = await super().get_signals()

        if not base_signals:
            return []

        # Process through enhancement pipeline
        enhanced_signals = []
        for signal in base_signals:
            enhanced = self._enhance_signal(signal)
            if enhanced:
                enhanced_signals.append(enhanced)

        if enhanced_signals:
            self._log_enhanced_signals(enhanced_signals)

        return enhanced_signals

    def _enhance_signal(self, signal: Signal) -> Optional[Signal]:
        """Apply V4 enhancements to a base signal."""
        # Extract metadata
        metadata = signal.metadata or {}
        alert_types = metadata.get("alert_types", [])
        question = metadata.get("question", "")

        # Build market context from metadata
        market_context = None
        if "market_lifetime_hours" in metadata:
            market_context = MarketContext(
                hours_to_resolution=metadata.get("market_lifetime_hours"),
                spread_pct=metadata.get("spread_pct"),
                current_price=metadata.get("price", 0.5),
                volume_24h=metadata.get("volume_24h"),
                volatility=metadata.get("volatility"),
            )

        # Process through enhanced source
        for alert_type in alert_types or ["unknown"]:
            result = self._enhanced_source.process_alert(
                alert_type=alert_type,
                token_id=signal.token_id,
                market_id=signal.market_id,
                direction="BUY" if signal.is_buy else "SELL",
                score=signal.score / len(alert_types) if alert_types else signal.score,
                wallet=metadata.get("wallet"),
                trade_value=metadata.get("trade_value_usd"),
                market_context=market_context,
                metadata=metadata,
            )

            if result:
                # Create enhanced signal
                enhanced_metadata = {
                    **metadata,
                    "enhanced": True,
                    "confirmation_score": result["confirmation_score"],
                    "signal_types": result["signal_types"],
                    "wallet_score": result["wallet_score"],
                    "context_multiplier": result["context_multiplier"],
                    "position_pct": result["position_pct"],
                    "is_contrarian": result["is_contrarian"],
                }

                return Signal(
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    direction=signal.direction,
                    score=result["score"],
                    source=f"{signal.source}_v4",
                    metadata=enhanced_metadata,
                )

        # If confirmation not required, return original signal with context adjustments
        if not self.require_confirmation:
            # Apply context multiplier if available
            if market_context:
                from ..trading.components.flow_enhancements import MarketContextFilter
                ctx_filter = MarketContextFilter()
                multiplier, _ = ctx_filter.calculate_context_multiplier(market_context)
                signal.score *= multiplier

            return signal

        return None

    def _log_enhanced_signals(self, signals: List[Signal]):
        """Log enhanced signals with V4 details."""
        logger.info(f"{'='*60}")
        logger.info(f"🎯 ENHANCED FLOW SIGNALS (V4): {len(signals)}")
        logger.info(f"{'='*60}")

        for i, signal in enumerate(signals, 1):
            meta = signal.metadata or {}
            direction = "📈 BUY" if signal.is_buy else "📉 SELL"
            question = meta.get('question', 'Unknown')[:45]

            logger.info(f"  [{i}] {question}...")
            logger.info(
                f"      {direction} | "
                f"Score: {signal.score:.1f} | "
                f"Confirm: {meta.get('confirmation_score', 0):.1f}"
            )

            # Log V4 enhancement details
            if meta.get("enhanced"):
                types = meta.get("signal_types", [])
                wallet_score = meta.get("wallet_score")
                ctx_mult = meta.get("context_multiplier", 1.0)
                pos_pct = meta.get("position_pct", 0.05)

                logger.info(
                    f"      Types: {', '.join(types[:3])} | "
                    f"Wallet: {wallet_score:.0f}" if wallet_score else "N/A" + f" | "
                    f"Ctx: {ctx_mult:.2f}x | "
                    f"Size: {pos_pct:.1%}"
                )

                if meta.get("is_contrarian"):
                    logger.info(f"      CONTRARIAN SIGNAL")

        logger.info(f"{'='*60}")

    def get_exit_config_for_signal(self, signal: Signal) -> ExitConfig:
        """Get adaptive exit config for a signal."""
        metadata = signal.metadata or {}

        hours_to_resolution = metadata.get("market_lifetime_hours")
        confirmation_score = metadata.get("confirmation_score", 50)
        is_contrarian = metadata.get("is_contrarian", False)

        return self._enhanced_source.exit_calculator.get_exit_params(
            hours_to_resolution=hours_to_resolution,
            signal_strength=confirmation_score,
            is_contrarian=is_contrarian,
        )

    def update_wallet_result(
        self,
        wallet: str,
        pnl: float,
        market_id: str,
    ) -> None:
        """Update wallet reputation with trade result."""
        self._enhanced_source.wallet_tracker.update_result(
            address=wallet,
            pnl=pnl,
            market_id=market_id,
        )

    def get_wallet_stats(self) -> Dict:
        """Get wallet reputation statistics."""
        tracker = self._enhanced_source.wallet_tracker
        top_wallets = tracker.get_top_wallets(10)

        return {
            "total_wallets": len(tracker._wallets),
            "smart_money_count": sum(1 for w in tracker._wallets.values() if w.is_smart_money),
            "elite_count": sum(1 for w in tracker._wallets.values() if w.is_elite_trader),
            "top_wallets": [
                {
                    "address": w.address[:10] + "...",
                    "trades": w.trades,
                    "win_rate": f"{w.win_rate:.1%}",
                    "pnl": f"${w.total_pnl:,.0f}",
                    "score": f"{w.reputation_score:.0f}",
                }
                for w in top_wallets
            ],
        }


def create_flow_bot(
    agent_id: str = "flow-bot",
    config: Optional[Config] = None,
    dry_run: bool = False,
    # V6 OPTIMIZED PARAMETERS - Based on backtest analysis of real flow alerts
    min_score: float = 55.0,  # V6: 55 - higher quality bar
    min_trade_size: float = 500.0,  # V6: $500 - ignore noise
    category: Optional[str] = None,
    max_spread: float = 0.025,  # V6: 2.5% - tight spread requirement
    max_price_drift: float = 0.05,  # V6: 5% - tight drift tolerance
    exit_config: Optional[ExitConfig] = None,  # Exit strategy configuration
    # V6 Position sizing
    base_position_pct: float = 0.10,  # V6: 10% base position (optimized)
    max_position_multiplier: float = 1.25,  # V6: 1.25x max
    # V6 Price range filter - OPTIMIZED: 50c-90c range has best win rate
    min_price: float = 0.50,  # V6: 50c (was 15c) - only mid-range prices
    max_price: float = 0.90,  # V6: 90c (was 85c) - allow higher confidence plays
    # V6 Enhancement options - ON by default
    enhanced: bool = True,  # V6 enhanced signal source ON by default
    require_confirmation: bool = True,  # Require multi-signal confirmation
    enable_contrarian: bool = True,  # Enable contrarian signal detection
    enable_timing_filter: bool = True,  # Filter late signals
    enable_wallet_scoring: bool = True,  # Track wallet reputation
    # V6 Signal Quality Filters - ON by default
    filter_sports: bool = True,  # Filter out sports markets (efficiently priced)
    filter_resolved: bool = True,  # Filter out resolved markets (price near 0/1)
    min_market_lifetime_hours: float = 2.0,  # Skip ultra-short markets
    min_liquidity_volume: float = 5000.0,  # Skip low liquidity markets
    # V6 Alert Type Filter - SMART_MONEY_ACTIVITY has 75% win rate
    alert_type_filter: Optional[List[str]] = None,  # V6: Filter by alert type
    severity_filter: Optional[List[str]] = None,  # V6: Filter by severity (HIGH best)
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
        enhanced: If True, use V4 enhanced signal source with improvements
        require_confirmation: Require multi-signal confirmation (V4)
        enable_contrarian: Enable contrarian signal detection (V4)
        enable_timing_filter: Filter late signals (V4)
        enable_wallet_scoring: Track wallet reputation (V4)

    Returns:
        Configured TradingBot ready to start
    """
    config = config or get_config()
    
    # Log strategy configuration
    version = "V4 (Enhanced)" if enhanced else "V3"
    logger.info(f"{'='*60}")
    logger.info(f"🌊 FLOW COPY STRATEGY CONFIGURATION - {version}")
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

    # Log V4 enhancement options
    if enhanced:
        logger.info(f"  V4 Enhancements:")
        logger.info(f"    Multi-Signal Confirmation: {'ON' if require_confirmation else 'OFF'}")
        logger.info(f"    Contrarian Detection:      {'ON' if enable_contrarian else 'OFF'}")
        logger.info(f"    Timing Filter:             {'ON' if enable_timing_filter else 'OFF'}")
        logger.info(f"    Wallet Reputation:         {'ON' if enable_wallet_scoring else 'OFF'}")

    # Log V5 signal quality filters
    logger.info(f"  V5 Signal Quality Filters:")
    logger.info(f"    Filter Sports Markets:     {'ON' if filter_sports else 'OFF'}")
    logger.info(f"    Filter Resolved Markets:   {'ON' if filter_resolved else 'OFF'}")
    logger.info(f"    Min Market Lifetime:       {min_market_lifetime_hours}h")
    logger.info(f"    Min Liquidity Volume:      ${min_liquidity_volume:,.0f}")
    
    # Exit strategy config with V6 optimized defaults
    # V6 Optimization: Based on backtest - WIDER STOPS, HIGHER TARGETS
    # Analysis showed: -20% SL, +50% TP optimal for prediction markets
    exit_cfg = exit_config or ExitConfig(
        take_profit_pct=0.50,  # V6: 50% take-profit (was 8%) - let winners run
        trailing_stop_enabled=True,
        trailing_stop_activation_pct=0.20,  # V6: 20% activation (was 4%)
        trailing_stop_distance_pct=0.10,  # V6: 10% trail (was 2.5%)
        max_hold_minutes=120,  # V6: 120 min (was 30) - hold for resolution
        stop_loss_pct=0.20,  # V6: 20% stop-loss (was 10%) - wider to avoid whipsaws
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

    # Create signal source (V3 or V4/V5 enhanced)
    if enhanced:
        signal_source = EnhancedFlowCopySignalSource(
            dedup_window_seconds=30,
            decay_half_life_seconds=60.0,
            min_score=min_score,
            min_trade_size=min_trade_size,
            category_filter=category,
            storage=storage,
            require_confirmation=require_confirmation,
            enable_contrarian=enable_contrarian,
            enable_timing_filter=enable_timing_filter,
            enable_wallet_scoring=enable_wallet_scoring,
            # V5 Signal Quality Filters
            filter_sports=filter_sports,
            filter_resolved=filter_resolved,
            min_market_lifetime_hours=min_market_lifetime_hours,
            min_liquidity_volume=min_liquidity_volume,
        )
    else:
        signal_source = FlowCopySignalSource(
            dedup_window_seconds=30,
            decay_half_life_seconds=60.0,
            min_score=min_score,
            min_trade_size=min_trade_size,
            category_filter=category,
            storage=storage,
            # V5 Signal Quality Filters
            filter_sports=filter_sports,
            filter_resolved=filter_resolved,
            min_market_lifetime_hours=min_market_lifetime_hours,
            min_liquidity_volume=min_liquidity_volume,
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
    interval: float = 1.0,  # V6: 1s - fast polling
    min_score: float = 55.0,  # V6: 55
    min_trade_size: float = 500.0,  # V6: $500
    category: Optional[str] = None,
    max_spread: float = 0.025,  # V6: 2.5%
    max_price_drift: float = 0.05,  # V6: 5%
    exit_config: Optional[ExitConfig] = None,
    min_price: float = 0.50,  # V6: 50c (was 15c) - optimized range
    max_price: float = 0.90,  # V6: 90c (was 85c) - optimized range
    # V6 Enhancement options - ON by default
    enhanced: bool = True,  # V6: ON by default
    require_confirmation: bool = True,
    enable_contrarian: bool = True,
    enable_timing_filter: bool = True,
    enable_wallet_scoring: bool = True,
    # V6 Alert filters
    alert_type_filter: Optional[List[str]] = None,
    severity_filter: Optional[List[str]] = None,
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
        enhanced=enhanced,
        require_confirmation=require_confirmation,
        enable_contrarian=enable_contrarian,
        enable_timing_filter=enable_timing_filter,
        enable_wallet_scoring=enable_wallet_scoring,
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

    parser = argparse.ArgumentParser(description="Flow Copy Strategy Trading Bot V6")
    parser.add_argument("--agent-id", default="flow-bot", help="Agent ID")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--interval", type=float, default=1.0, help="Scan interval (V6: 1s)")
    parser.add_argument("--min-score", type=float, default=55.0, help="Minimum signal score (V6: 55)")
    parser.add_argument("--min-trade-size", type=float, default=500.0, help="Min trade size to track (V6: $500)")
    parser.add_argument("--category", type=str, default=None, help="Market category filter")
    parser.add_argument("--max-spread", type=float, default=0.025, help="Max bid-ask spread (V6: 2.5%%)")
    parser.add_argument("--max-price-drift", type=float, default=0.05, help="Max price drift from signal (V6: 5%%)")
    parser.add_argument("--min-price", type=float, default=0.50, help="Min token price to trade (V6: 50c)")
    parser.add_argument("--max-price", type=float, default=0.90, help="Max token price to trade (V6: 90c)")
    parser.add_argument("--take-profit", type=float, default=0.50, help="Take-profit threshold (V6: 50%%)")
    parser.add_argument("--trailing-activation", type=float, default=0.20, help="Trailing stop activation (V6: 20%%)")
    parser.add_argument("--trailing-distance", type=float, default=0.10, help="Trailing stop distance (V6: 10%%)")
    parser.add_argument("--max-hold-minutes", type=int, default=120, help="Max hold time in minutes (V6: 120)")
    parser.add_argument("--stop-loss", type=float, default=0.20, help="Stop-loss threshold (V6: 20%%)")

    # V6 Enhancement options - ON by default
    parser.add_argument("--enhanced", "-e", action="store_true", default=True, help="Use V6 enhanced signal source (default: ON)")
    parser.add_argument("--no-enhanced", action="store_true", help="Disable V6 enhanced signal source")
    parser.add_argument("--no-confirmation", action="store_true", help="Disable multi-signal confirmation")
    parser.add_argument("--no-contrarian", action="store_true", help="Disable contrarian detection")
    parser.add_argument("--no-timing-filter", action="store_true", help="Disable timing filter")
    parser.add_argument("--no-wallet-scoring", action="store_true", help="Disable wallet scoring")

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
    
    # Handle enhanced flag (--no-enhanced overrides --enhanced)
    use_enhanced = args.enhanced and not getattr(args, 'no_enhanced', False)

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
        # V5 Enhancement options
        enhanced=use_enhanced,
        require_confirmation=not args.no_confirmation,
        enable_contrarian=not args.no_contrarian,
        enable_timing_filter=not args.no_timing_filter,
        enable_wallet_scoring=not args.no_wallet_scoring,
    ))

