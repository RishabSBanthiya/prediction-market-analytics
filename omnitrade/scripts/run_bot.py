#!/usr/bin/env python3
"""
OmniTrade unified bot CLI.

Usage:
    python scripts/run_bot.py directional --exchange polymarket --paper
    python scripts/run_bot.py mm --exchange kalshi --live
    python scripts/run_bot.py directional --exchange hyperliquid --paper --interval 15
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.core.config import Config, get_config, set_config
from omnitrade.core.enums import ExchangeId, Environment
from omnitrade.exchanges.registry import create_client
from omnitrade.storage.sqlite import SQLiteStorage
from omnitrade.risk.coordinator import RiskCoordinator
from omnitrade.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="OmniTrade Bot Runner")
    parser.add_argument(
        "bot_type",
        choices=["directional", "mm", "market-making", "hedge", "cross-arb"],
        help="Bot type to run",
    )
    parser.add_argument(
        "--exchange", "-e",
        choices=["polymarket", "kalshi", "hyperliquid"],
        default=None,
        help="Exchange to trade on (required for single-exchange bots)",
    )
    parser.add_argument(
        "--hedge-exchange",
        choices=["hyperliquid"],
        default="hyperliquid",
        help="Exchange for hedge leg (default: hyperliquid)",
    )
    parser.add_argument(
        "--paper", action="store_true", default=True,
        help="Paper trading mode (default)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live trading mode (real money!)",
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=30.0,
        help="Trading loop interval in seconds",
    )
    parser.add_argument(
        "--agent-id", type=str, default=None,
        help="Bot agent ID (auto-generated if not set)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Alias for --paper",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def run_directional(exchange_id, config, agent_id, interval, environment):
    """Run a directional bot."""
    from omnitrade.components.signals import MidpointDeviationSignal
    from omnitrade.components.sizers import SignalScaledSizer
    from omnitrade.components.executors import AggressiveExecutor, DryRunExecutor
    from omnitrade.bots.directional import DirectionalBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)
    client = create_client(exchange_id, config)

    executor = DryRunExecutor() if environment == Environment.PAPER else AggressiveExecutor()

    bot = DirectionalBot(
        agent_id=agent_id,
        client=client,
        signal_source=MidpointDeviationSignal(),
        sizer=SignalScaledSizer(),
        executor=executor,
        risk=risk,
        environment=environment,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


async def run_market_making(exchange_id, config, agent_id, interval, environment):
    """Run a market making bot."""
    from omnitrade.components.quote_engines import SimpleSpreadQuoter
    from omnitrade.components.market_selectors import ActiveMarketSelector
    from omnitrade.bots.market_making import MarketMakingBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)
    client = create_client(exchange_id, config)

    bot = MarketMakingBot(
        agent_id=agent_id,
        client=client,
        quote_engine=SimpleSpreadQuoter(),
        market_selector=ActiveMarketSelector(),
        risk=risk,
        environment=environment,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


async def run_hedge(binary_exchange_id, hedge_exchange_id, config, agent_id, interval, environment):
    """Run a cross-exchange hedge bot (binary + perp)."""
    from omnitrade.components.hedge_signals import BinaryPerpHedgeSignal
    from omnitrade.components.executors import DryRunExecutor, AggressiveExecutor
    from omnitrade.bots.cross_exchange import CrossExchangeBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)

    # Create clients for both exchanges
    clients = {
        binary_exchange_id: create_client(binary_exchange_id, config),
        hedge_exchange_id: create_client(hedge_exchange_id, config),
    }

    # Create executors for each exchange
    if environment == Environment.PAPER:
        executors = {ex: DryRunExecutor() for ex in clients}
    else:
        executors = {ex: AggressiveExecutor() for ex in clients}

    signal_source = BinaryPerpHedgeSignal(
        binary_exchange=binary_exchange_id,
        hedge_exchange=hedge_exchange_id,
    )

    bot = CrossExchangeBot(
        agent_id=agent_id,
        clients=clients,
        signal_source=signal_source,
        executors=executors,
        risk=risk,
        environment=environment,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


async def run_cross_arb(config, agent_id, interval, environment):
    """Run cross-exchange arb bot (Polymarket vs Kalshi)."""
    from omnitrade.components.hedge_signals import CrossExchangeArbSignal
    from omnitrade.components.executors import DryRunExecutor, AggressiveExecutor
    from omnitrade.bots.cross_exchange import CrossExchangeBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)

    clients = {
        ExchangeId.POLYMARKET: create_client(ExchangeId.POLYMARKET, config),
        ExchangeId.KALSHI: create_client(ExchangeId.KALSHI, config),
    }

    if environment == Environment.PAPER:
        executors = {ex: DryRunExecutor() for ex in clients}
    else:
        executors = {ex: AggressiveExecutor() for ex in clients}

    bot = CrossExchangeBot(
        agent_id=agent_id,
        clients=clients,
        signal_source=CrossExchangeArbSignal(),
        executors=executors,
        risk=risk,
        environment=environment,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    environment = Environment.LIVE if args.live else Environment.PAPER

    config = Config.from_env()
    config.environment = environment
    set_config(config)

    mode = "LIVE" if environment == Environment.LIVE else "PAPER"

    if environment == Environment.LIVE:
        print("WARNING: LIVE TRADING MODE - Real money at risk!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return

    # Cross-exchange bots don't need --exchange
    if args.bot_type == "hedge":
        exchange = args.exchange or "polymarket"
        exchange_id = ExchangeId(exchange)
        hedge_id = ExchangeId(args.hedge_exchange)
        agent_id = args.agent_id or f"hedge-{exchange}-{args.hedge_exchange}"

        print(f"OmniTrade hedge bot | {exchange} + {args.hedge_exchange} | {mode} mode")
        print(f"Agent: {agent_id} | Interval: {args.interval}s")
        print("-" * 50)

        asyncio.run(run_hedge(exchange_id, hedge_id, config, agent_id, args.interval, environment))

    elif args.bot_type == "cross-arb":
        agent_id = args.agent_id or "cross-arb-poly-kalshi"

        print(f"OmniTrade cross-arb bot | polymarket + kalshi | {mode} mode")
        print(f"Agent: {agent_id} | Interval: {args.interval}s")
        print("-" * 50)

        asyncio.run(run_cross_arb(config, agent_id, args.interval, environment))

    else:
        # Single-exchange bots require --exchange
        if not args.exchange:
            print("Error: --exchange is required for directional and mm bots")
            sys.exit(1)

        exchange_id = ExchangeId(args.exchange)
        agent_id = args.agent_id or f"{args.bot_type}-{args.exchange}"

        print(f"OmniTrade {args.bot_type} bot | {args.exchange} | {mode} mode")
        print(f"Agent: {agent_id} | Interval: {args.interval}s")
        print("-" * 50)

        if args.bot_type == "directional":
            asyncio.run(run_directional(exchange_id, config, agent_id, args.interval, environment))
        else:
            asyncio.run(run_market_making(exchange_id, config, agent_id, args.interval, environment))


if __name__ == "__main__":
    main()
