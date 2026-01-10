#!/usr/bin/env python3
"""
Run the Statistical Arbitrage Bot.

This bot scans all Polymarket markets for statistical and cross-market
arbitrage opportunities including:
- Pair trading (correlated market mean reversion)
- Multi-outcome sum arbitrage (sum != 100%)
- Duplicate market arbitrage (same question, different prices)
- Conditional probability arbitrage (P(A|B) mispricings)

Usage:
    # Dry run mode (recommended first)
    python scripts/run_stat_arb_bot.py --dry-run

    # Live trading
    python scripts/run_stat_arb_bot.py

    # Enable only specific arb types
    python scripts/run_stat_arb_bot.py --types pair_spread,multi_outcome

    # Custom parameters
    python scripts/run_stat_arb_bot.py --entry-z 2.5 --max-positions 5
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.core.config import get_config
from polymarket.core.api import PolymarketAPI
from polymarket.trading.risk_coordinator import RiskCoordinator
from polymarket.strategies.stat_arb import (
    StatArbStrategy,
    StatArbConfig,
    CorrelationConfig,
    PairTradingConfig,
    MultiOutcomeConfig,
    DuplicateConfig,
    ConditionalConfig,
    ArbType,
)


def setup_logging(agent_id: str, log_level: str = "INFO"):
    """Configure logging."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{agent_id}.log"),
        ],
    )

    # Quiet noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Bot for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (safe, no real orders)
    python run_stat_arb_bot.py --dry-run

    # Only pair trading
    python run_stat_arb_bot.py --types pair_spread

    # Multiple types
    python run_stat_arb_bot.py --types pair_spread,multi_outcome,duplicate

    # Custom pair trading parameters
    python run_stat_arb_bot.py --entry-z 2.5 --exit-z 0.3 --stop-z 4.0

Arbitrage Types:
    pair_spread    - Mean reversion on correlated market pairs
    multi_outcome  - Sum of outcome prices != 100%
    duplicate      - Same question priced differently
    conditional    - Conditional probability mispricings (P(A|B))
        """,
    )

    # Basic options
    parser.add_argument(
        "--agent-id",
        type=str,
        default="stat-arb-bot",
        help="Unique agent identifier (default: stat-arb-bot)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Scan interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # Arb type selection
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help="Comma-separated arb types: pair_spread,multi_outcome,duplicate,conditional",
    )

    # Position limits
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Max concurrent positions (default: 10)",
    )
    parser.add_argument(
        "--max-position-pct",
        type=float,
        default=0.10,
        help="Max position size as %% of capital (default: 0.10)",
    )

    # Pair trading parameters
    parser.add_argument(
        "--entry-z",
        type=float,
        default=2.0,
        help="Pair trading: entry z-score (default: 2.0)",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.5,
        help="Pair trading: exit z-score (default: 0.5)",
    )
    parser.add_argument(
        "--stop-z",
        type=float,
        default=3.5,
        help="Pair trading: stop loss z-score (default: 3.5)",
    )
    parser.add_argument(
        "--min-correlation",
        type=float,
        default=0.7,
        help="Minimum correlation for pairs (default: 0.7)",
    )

    # Multi-outcome parameters
    parser.add_argument(
        "--multi-min-edge",
        type=int,
        default=50,
        help="Multi-outcome: min edge in bps (default: 50)",
    )

    # Duplicate parameters
    parser.add_argument(
        "--dup-min-edge",
        type=int,
        default=30,
        help="Duplicate: min edge in bps (default: 30)",
    )
    parser.add_argument(
        "--dup-similarity",
        type=float,
        default=0.90,
        help="Duplicate: min semantic similarity (default: 0.90)",
    )

    # Conditional parameters
    parser.add_argument(
        "--cond-min-edge",
        type=int,
        default=50,
        help="Conditional: min edge in bps (default: 50)",
    )

    return parser.parse_args()


def build_config(args) -> StatArbConfig:
    """Build StatArbConfig from command line arguments."""
    # Parse enabled types
    enabled_types = None
    if args.types:
        type_map = {
            "pair_spread": ArbType.PAIR_SPREAD,
            "multi_outcome": ArbType.MULTI_OUTCOME_SUM,
            "duplicate": ArbType.DUPLICATE_MARKET,
            "conditional": ArbType.CONDITIONAL_PROB,
        }
        enabled_types = []
        for t in args.types.split(","):
            t = t.strip().lower()
            if t in type_map:
                enabled_types.append(type_map[t])
            else:
                print(f"Warning: Unknown arb type '{t}'")

    return StatArbConfig(
        # Correlation
        correlation=CorrelationConfig(
            min_price_correlation=args.min_correlation,
        ),
        # Pair trading
        pair_trading=PairTradingConfig(
            entry_z_score=args.entry_z,
            exit_z_score=args.exit_z,
            stop_z_score=args.stop_z,
            min_correlation=args.min_correlation,
            max_position_pct=args.max_position_pct,
        ),
        # Multi-outcome
        multi_outcome=MultiOutcomeConfig(
            min_edge_bps=args.multi_min_edge,
            max_position_pct=args.max_position_pct,
        ),
        # Duplicate
        duplicate=DuplicateConfig(
            min_similarity=args.dup_similarity,
            min_edge_bps=args.dup_min_edge,
            max_position_pct=args.max_position_pct,
        ),
        # Conditional
        conditional=ConditionalConfig(
            min_edge_bps=args.cond_min_edge,
            max_position_pct=args.max_position_pct,
        ),
        # General
        enabled_types=enabled_types,
        max_total_positions=args.max_positions,
        scan_interval_seconds=args.interval,
    )


def print_banner(args, config: StatArbConfig):
    """Print startup banner."""
    enabled = config.get_enabled_types()
    types_str = ", ".join(t.value for t in enabled)

    print()
    print("=" * 60)
    print("STATISTICAL ARBITRAGE BOT")
    print("=" * 60)
    print(f"  Agent ID:       {args.agent_id}")
    print(f"  Mode:           {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Scan Interval:  {args.interval}s")
    print(f"  Max Positions:  {config.max_total_positions}")
    print()
    print(f"  Enabled Types:  {types_str}")
    print()

    if ArbType.PAIR_SPREAD in enabled:
        print("  PAIR TRADING:")
        print(f"    Entry Z-Score:  {config.pair_trading.entry_z_score}")
        print(f"    Exit Z-Score:   {config.pair_trading.exit_z_score}")
        print(f"    Stop Z-Score:   {config.pair_trading.stop_z_score}")
        print(f"    Min Correl:     {config.pair_trading.min_correlation}")
        print()

    if ArbType.MULTI_OUTCOME_SUM in enabled:
        print("  MULTI-OUTCOME:")
        print(f"    Min Edge:       {config.multi_outcome.min_edge_bps} bps")
        print()

    if ArbType.DUPLICATE_MARKET in enabled:
        print("  DUPLICATE MARKET:")
        print(f"    Min Edge:       {config.duplicate.min_edge_bps} bps")
        print(f"    Min Similarity: {config.duplicate.min_similarity}")
        print()

    if ArbType.CONDITIONAL_PROB in enabled:
        print("  CONDITIONAL PROB:")
        print(f"    Min Edge:       {config.conditional.min_edge_bps} bps")
        print()

    print("=" * 60)
    print()


class StatArbBot:
    """Main bot class."""

    def __init__(
        self,
        agent_id: str,
        config: StatArbConfig,
        dry_run: bool = True,
    ):
        self.agent_id = agent_id
        self.config = config
        self.dry_run = dry_run

        self.api: PolymarketAPI = None
        self.risk_coordinator: RiskCoordinator = None
        self.strategy: StatArbStrategy = None

        self._running = False

    async def start(self):
        """Initialize and start the bot."""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting stat arb bot: {self.agent_id}")

        # Initialize API
        app_config = get_config()
        self.api = PolymarketAPI(app_config)

        # Initialize risk coordinator
        try:
            self.risk_coordinator = RiskCoordinator(config=app_config, api=self.api)
            await self.risk_coordinator.startup(self.agent_id, "stat_arb")
        except Exception as e:
            logger.warning(f"Risk coordinator init failed: {e}")
            self.risk_coordinator = None

        # Print wallet state
        await self._print_wallet_state()

        # Initialize CLOB client for live trading
        clob_client = None
        if not self.dry_run:
            try:
                from py_clob_client.client import ClobClient

                clob_client = ClobClient(
                    app_config.clob_host,
                    key=app_config.private_key,
                    chain_id=app_config.chain_id,
                    signature_type=2,
                    funder=app_config.proxy_address
                )
                clob_client.set_api_creds(clob_client.create_or_derive_api_creds())
                logger.info("CLOB client initialized for live trading")
            except Exception as e:
                logger.error(f"Failed to initialize CLOB client: {e}")
                logger.warning("Falling back to dry run mode")
                self.dry_run = True

        # Initialize strategy
        self.strategy = StatArbStrategy(
            api=self.api,
            config=self.config,
            risk_coordinator=self.risk_coordinator,
            dry_run=self.dry_run,
            agent_id=self.agent_id,
        )

        # Set CLOB client on position manager for live trading
        if clob_client and not self.dry_run:
            self.strategy.position_manager.set_clob_client(clob_client)

        await self.strategy.initialize()

    async def run(self):
        """Run the main bot loop."""
        logger = logging.getLogger(__name__)
        self._running = True
        iteration = 0

        while self._running:
            try:
                iteration += 1

                # Scan and execute
                opportunities = await self.strategy.scan_opportunities()

                if opportunities:
                    logger.info(
                        f"Iteration {iteration}: Found {len(opportunities)} opportunities"
                    )
                    # Log top opportunities
                    for opp in opportunities[:5]:
                        logger.info(
                            f"  {opp.arb_type.value}: {opp.edge_bps}bps "
                            f"(z={opp.z_score:.2f}, conf={opp.confidence:.2f})"
                        )

                    # Execute top opportunities
                    for opp in opportunities[:3]:
                        await self.strategy.execute_opportunity(opp)

                # Monitor positions
                await self.strategy.monitor_positions()

                # Log stats periodically
                if iteration % 10 == 0:
                    stats = self.strategy.get_stats()
                    logger.info(
                        f"Stats: {stats['open_positions']} open positions, "
                        f"P&L=${stats['total_realized_pnl']:.2f}"
                    )

            except Exception as e:
                logger.error(f"Iteration {iteration} failed: {e}")

            await asyncio.sleep(self.config.scan_interval_seconds)

    async def stop(self):
        """Stop the bot gracefully."""
        logger = logging.getLogger(__name__)
        logger.info("Stopping stat arb bot...")

        self._running = False

        if self.strategy:
            await self.strategy.stop()

        # Print final stats
        if self.strategy:
            stats = self.strategy.get_stats()
            logger.info(f"Final stats: {stats}")

    async def _print_wallet_state(self):
        """Print wallet state."""
        try:
            config = get_config()
            wallet_address = config.proxy_address
            if not wallet_address:
                print("  (No wallet address configured)")
                print()
                return

            balance = await self.api.fetch_usdc_balance(wallet_address)
            print()
            print("WALLET STATE")
            print(f"  USDC Balance:    ${balance:.2f}")

            if self.risk_coordinator:
                available = self.risk_coordinator.get_available_capital(self.agent_id)
                print(f"  Available:       ${available:.2f}")
            print()
        except Exception as e:
            print(f"  (Could not fetch wallet state: {e})")
            print()


async def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.agent_id, args.log_level)

    logger = logging.getLogger(__name__)

    # Build config
    config = build_config(args)
    print_banner(args, config)

    # Create bot
    bot = StatArbBot(
        agent_id=args.agent_id,
        config=config,
        dry_run=args.dry_run,
    )

    # Handle signals - set _running = False immediately for fast shutdown
    def signal_handler():
        logger.info("Received shutdown signal")
        bot._running = False

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
        await bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
