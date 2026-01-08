#!/usr/bin/env python3
"""
Run the Delta-Neutral Arbitrage Bot.

This bot monitors 15-minute crypto markets for arbitrage opportunities
where both outcomes can be bought for less than $1 total.

Strategy:
1. Scan markets for UP_price + DOWN_price < 1.0 (after fees)
2. Place limit orders on BOTH sides at profitable prices
3. Wait patiently for fills (no latency advantage needed)
4. When both fill = guaranteed profit at resolution

Usage:
    # Dry run mode (recommended first)
    python scripts/run_arb_bot.py --dry-run

    # Live trading
    python scripts/run_arb_bot.py

    # Custom settings
    python scripts/run_arb_bot.py --min-edge 75 --size 50
"""

import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.core.config import get_config
from polymarket.core.api import PolymarketAPI
from polymarket.strategies.arb_strategy import ArbStrategy, ArbConfig


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

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Delta-Neutral Arbitrage Bot for Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (safe, no real orders)
    python run_arb_bot.py --dry-run

    # Conservative settings (higher edge required)
    python run_arb_bot.py --min-edge 100 --size 10

    # More aggressive
    python run_arb_bot.py --min-edge 30 --size 50 --max-positions 10

How it works:
    In a binary market (UP/DOWN), the outcomes sum to $1 at resolution.
    If we can buy BOTH for less than $1, we profit regardless of outcome.

    Example:
        UP ask = $0.48, DOWN ask = $0.48
        Total cost = $0.96 + fees (~$0.01) = $0.97
        Guaranteed return = $1.00
        Profit = $0.03 per share pair (3.1%)

    Without latency advantage, we place patient LIMIT orders and wait.
        """,
    )

    parser.add_argument(
        "--agent-id",
        type=str,
        default="arb-bot",
        help="Unique identifier for this agent (default: arb-bot)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real orders)",
    )

    parser.add_argument(
        "--min-edge",
        type=int,
        default=50,
        help="Minimum edge in basis points to open position (default: 50 = 0.5%%)",
    )

    parser.add_argument(
        "--size",
        type=float,
        default=20.0,
        help="Order size in USD per side (default: 20)",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Maximum open positions (default: 5)",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Scan interval in seconds (default: 30)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


class ArbBot:
    """Arbitrage bot runner."""

    def __init__(
        self,
        agent_id: str,
        arb_config: ArbConfig,
        dry_run: bool = True,
    ):
        self.agent_id = agent_id
        self.arb_config = arb_config
        self.dry_run = dry_run
        self.config = get_config()
        self.api: Optional[PolymarketAPI] = None
        self.strategy: Optional[ArbStrategy] = None
        self.running = False
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start the bot."""
        self.logger.info("=" * 60)
        self.logger.info("STARTING ARBITRAGE BOT")
        self.logger.info("=" * 60)
        self.logger.info(f"  Agent ID:     {self.agent_id}")
        self.logger.info(f"  Mode:         {'DRY RUN' if self.dry_run else 'LIVE'}")
        self.logger.info(f"  Min Edge:     {self.arb_config.min_edge_bps} bps")
        self.logger.info(f"  Order Size:   ${self.arb_config.order_size_usd}")
        self.logger.info(f"  Max Positions: {self.arb_config.max_positions}")

        if self.dry_run:
            self.logger.warning("DRY RUN MODE - No real orders will be placed")
        else:
            self.logger.warning("LIVE MODE - Real orders WILL be placed!")

        # Initialize API
        self.api = PolymarketAPI(self.config)
        await self.api.connect()
        self.logger.info("  API connected")

        # Initialize strategy
        self.strategy = ArbStrategy(
            api=self.api,
            config=self.arb_config,
            dry_run=self.dry_run,
        )

        # Setup CLOB client for live trading
        if not self.dry_run:
            try:
                from py_clob_client.client import ClobClient

                client = ClobClient(
                    self.config.clob_host,
                    key=self.config.private_key,
                    chain_id=self.config.chain_id,
                    signature_type=2,
                    funder=self.config.proxy_address,
                )
                client.set_api_creds(client.create_or_derive_api_creds())
                self.strategy.set_client(client)
                self.logger.info("  CLOB client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize CLOB client: {e}")
                raise

        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("Arbitrage Bot ready!")

    async def stop(self):
        """Stop the bot."""
        self.logger.info("=" * 60)
        self.logger.info(f"STOPPING ARBITRAGE BOT: {self.agent_id}")
        self.logger.info("=" * 60)

        self.running = False

        # Print final stats
        if self.strategy:
            stats = self.strategy.get_stats()
            self.logger.info("FINAL STATISTICS")
            self.logger.info(f"  Open Positions:   {stats['open_positions']}")
            self.logger.info(f"  Locked Positions: {stats['locked_positions']}")
            self.logger.info(f"  Total Profit:     ${stats['total_locked_profit']:.2f}")

        if self.api:
            await self.api.close()
            self.logger.info("  API disconnected")

        self.logger.info("Arbitrage Bot stopped")

    async def run(self, interval_seconds: float = 30.0):
        """Run the arbitrage scanning loop."""
        self.logger.info(f"Starting scan loop (interval={interval_seconds}s)")

        iteration = 0
        while self.running:
            try:
                iteration += 1

                # Update markets
                await self.strategy.update_markets()

                # Scan and execute
                opportunities = await self.strategy.scan_and_execute()

                # Log stats periodically
                if iteration % 10 == 0:
                    stats = self.strategy.get_stats()
                    self.logger.info(
                        f"Stats: {stats['active_markets']} markets | "
                        f"{stats['open_positions']} open | "
                        f"{stats['locked_positions']} locked | "
                        f"${stats['total_locked_profit']:.2f} profit"
                    )

                # Wait for next scan
                await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                self.logger.info("Scan loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry


async def main():
    """Main entry point."""
    args = parse_args()

    setup_logging(args.agent_id, args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("POLYMARKET ARBITRAGE BOT")
    logger.info("=" * 60)
    logger.info(f"  Agent ID:      {args.agent_id}")
    logger.info(f"  Mode:          {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Min Edge:      {args.min_edge} bps ({args.min_edge/100:.2f}%)")
    logger.info(f"  Order Size:    ${args.size} per side")
    logger.info(f"  Max Positions: {args.max_positions}")
    logger.info(f"  Scan Interval: {args.interval}s")
    logger.info("=" * 60)

    arb_config = ArbConfig(
        min_edge_bps=args.min_edge,
        order_size_usd=args.size,
        max_positions=args.max_positions,
    )

    bot = ArbBot(
        agent_id=args.agent_id,
        arb_config=arb_config,
        dry_run=args.dry_run,
    )

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def handle_signal(sig):
        logger.info(f"Received signal {sig.name}, shutting down...")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        await bot.start()
        await bot.run(interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if bot.running:
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
