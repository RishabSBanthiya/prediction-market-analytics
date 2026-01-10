"""
Statistical Arbitrage Strategy.

Main orchestration class that:
- Initializes all components (correlation engine, scanners, position manager)
- Runs scan and execute loop
- Monitors positions for exit conditions
- Integrates with TradingBot via SignalSource interface
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market
from polymarket.trading.risk_coordinator import RiskCoordinator

from .models import StatArbOpportunity, StatArbPosition, ArbType
from .config import StatArbConfig
from .correlation_engine import CorrelationEngine
from .signals import StatArbSignals
from .position_manager import StatArbPositionManager

logger = logging.getLogger(__name__)


class StatArbStrategy:
    """
    Statistical Arbitrage Strategy orchestrator.

    Coordinates:
    - Market data fetching
    - Correlation analysis
    - Opportunity scanning (all 4 types)
    - Position management
    - Exit monitoring

    Can be used standalone or integrated with TradingBot.
    """

    def __init__(
        self,
        api: PolymarketAPI,
        config: StatArbConfig,
        risk_coordinator: Optional[RiskCoordinator] = None,
        dry_run: bool = True,
        agent_id: str = "stat-arb",
    ):
        self.api = api
        self.config = config
        self.risk_coordinator = risk_coordinator
        self.dry_run = dry_run
        self.agent_id = agent_id

        # Initialize components
        self.correlation_engine = CorrelationEngine(api, config.correlation)

        self.signals = StatArbSignals(
            api=api,
            config=config,
            correlation_engine=self.correlation_engine,
        )

        self.position_manager = StatArbPositionManager(
            api=api,
            config=config,
            risk_coordinator=risk_coordinator,
            dry_run=dry_run,
        )

        # State
        self._initialized = False
        self._markets: List[Market] = []
        self._running = False

    async def initialize(self) -> None:
        """Initialize strategy with market data and correlations."""
        if self._initialized:
            return

        logger.info(f"Initializing stat arb strategy (agent={self.agent_id})")

        # Initialize signal source (which initializes correlation engine)
        await self.signals.initialize()

        self._initialized = True
        logger.info("Stat arb strategy initialized")

    async def run(self, interval_seconds: Optional[float] = None) -> None:
        """
        Run the strategy loop.

        Args:
            interval_seconds: Scan interval (default from config)
        """
        if not self._initialized:
            await self.initialize()

        interval = interval_seconds or self.config.scan_interval_seconds
        self._running = True

        logger.info(f"Starting stat arb strategy loop (interval={interval}s)")

        while self._running:
            try:
                await self._iteration()
            except Exception as e:
                logger.error(f"Strategy iteration failed: {e}")

            await asyncio.sleep(interval)

    async def stop(self) -> None:
        """Stop the strategy loop."""
        self._running = False
        logger.info("Stat arb strategy stopped")

    async def _iteration(self) -> None:
        """Single strategy iteration."""
        # Get opportunities via signals
        opportunities = await self.scan_opportunities()

        if opportunities:
            logger.info(f"Found {len(opportunities)} opportunities")

            # Execute top opportunities
            for opp in opportunities[:3]:  # Max 3 per iteration
                await self.execute_opportunity(opp)

        # Monitor existing positions
        await self.monitor_positions()

    async def scan_opportunities(self) -> List[StatArbOpportunity]:
        """
        Scan for all types of arbitrage opportunities.

        Returns:
            List of opportunities sorted by edge.
        """
        signals = await self.signals.get_signals()

        # Convert signals back to opportunities (they have full metadata)
        opportunities = []
        for signal in signals:
            opp = self._signal_to_opportunity(signal)
            if opp:
                opportunities.append(opp)

        return opportunities

    async def execute_opportunity(
        self,
        opportunity: StatArbOpportunity,
    ) -> Optional[StatArbPosition]:
        """
        Execute an arbitrage opportunity.

        Args:
            opportunity: Opportunity to execute

        Returns:
            Position if successful, None otherwise
        """
        # Calculate position size based on config and available capital
        position_size = await self._calculate_position_size(opportunity)

        if position_size <= 0:
            logger.debug(f"Position size too small for {opportunity.opportunity_id}")
            return None

        logger.info(
            f"Executing {opportunity.arb_type.value} opportunity "
            f"(edge={opportunity.edge_bps}bps, size=${position_size:.2f})"
        )

        return await self.position_manager.open_position(
            opportunity=opportunity,
            agent_id=self.agent_id,
            position_size_usd=position_size,
        )

    async def monitor_positions(self) -> None:
        """Monitor positions and close when exit conditions met."""
        updates = await self.position_manager.update_positions()

        for position_id, update in updates.items():
            if update.get("should_close"):
                reason = update.get("close_reason", "unknown")
                logger.info(f"Closing position {position_id[:8]}: {reason}")
                await self.position_manager.close_position(position_id, reason)

    async def _calculate_position_size(
        self,
        opportunity: StatArbOpportunity,
    ) -> float:
        """Calculate position size based on opportunity and limits."""
        # Get available capital
        if self.risk_coordinator:
            available = self.risk_coordinator.get_available_capital(self.agent_id)
        else:
            # Default to a fixed amount in dry run
            available = 1000.0

        # Apply per-type limits
        if opportunity.arb_type == ArbType.PAIR_SPREAD:
            max_pct = self.config.pair_trading.max_position_pct
        elif opportunity.arb_type == ArbType.MULTI_OUTCOME_SUM:
            max_pct = self.config.multi_outcome.max_position_pct
        elif opportunity.arb_type == ArbType.DUPLICATE_MARKET:
            max_pct = self.config.duplicate.max_position_pct
        elif opportunity.arb_type == ArbType.CONDITIONAL_PROB:
            max_pct = self.config.conditional.max_position_pct
        else:
            max_pct = 0.10

        position_size = available * max_pct

        # Scale by confidence
        position_size *= opportunity.confidence

        # Minimum position size
        min_size = 10.0
        if position_size < min_size:
            return 0.0

        return position_size

    def _signal_to_opportunity(self, signal) -> Optional[StatArbOpportunity]:
        """Convert a Signal back to StatArbOpportunity."""
        metadata = signal.metadata

        if "opportunity_id" not in metadata:
            return None

        # Reconstruct legs
        legs = []
        for leg_data in metadata.get("legs", []):
            from .models import ArbLeg
            leg = ArbLeg(
                token_id=leg_data["token_id"],
                market_id=leg_data["market_id"],
                outcome=leg_data.get("outcome", ""),
                side=leg_data["side"],
                target_price=leg_data["target_price"],
                target_shares=leg_data["target_shares"],
            )
            legs.append(leg)

        arb_type = ArbType(metadata.get("arb_type", "pair_spread"))

        return StatArbOpportunity(
            opportunity_id=metadata["opportunity_id"],
            arb_type=arb_type,
            detected_at=datetime.now(timezone.utc),
            market_ids=metadata.get("all_market_ids", [signal.market_id]),
            token_ids=metadata.get("all_token_ids", [signal.token_id]),
            questions=metadata.get("questions", []),
            edge_bps=metadata.get("edge_bps", 0),
            z_score=metadata.get("z_score", 0.0),
            confidence=signal.score / 100,
            legs=legs,
            total_cost=metadata.get("total_cost", 0.0),
            expected_profit=metadata.get("expected_profit", 0.0),
            metadata={
                k: v for k, v in metadata.items()
                if k not in ("opportunity_id", "arb_type", "edge_bps", "z_score",
                            "legs", "all_market_ids", "all_token_ids", "questions",
                            "total_cost", "expected_profit")
            },
        )

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        scanner_stats = self.signals.get_scanner_stats()
        position_stats = self.position_manager.get_stats()

        return {
            "agent_id": self.agent_id,
            "dry_run": self.dry_run,
            "initialized": self._initialized,
            "running": self._running,
            **scanner_stats,
            **position_stats,
        }
