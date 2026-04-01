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

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omnitrade.core.config import Config, get_config, set_config
from omnitrade.core.enums import ExchangeId, Environment
from omnitrade.core.errors import ConfigError
from omnitrade.core.models import Instrument
from omnitrade.core.validation import validate_startup
from omnitrade.exchanges.registry import create_client
from omnitrade.storage.sqlite import SQLiteStorage
from omnitrade.risk.coordinator import RiskCoordinator
from omnitrade.utils.logging import setup_logging


class MarketFilteredClient:
    """Wraps an ExchangeClient to fetch only specific instruments.

    For each filter term, makes a targeted API call using the exchange's
    server-side filtering (event_ticker for Kalshi, slug_contains for
    Polymarket) rather than fetching all instruments and filtering locally.
    """

    def __init__(self, client, filters: list[str]):
        self._client = client
        self._filters = filters
        self._exchange_id = client.exchange_id
        self._cache: list[Instrument] | None = None
        self._cache_time: float = 0
        self._logger = logging.getLogger("omnitrade.filter")

    async def get_instruments(self, active_only: bool = True, **kwargs) -> list[Instrument]:
        # Re-fetch every 5 minutes (10 cycles at 30s), cache between
        import time
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_time) < 300:
            return self._cache

        all_matched: list[Instrument] = []
        seen_ids: set[str] = set()

        for term in self._filters:
            instruments = await self._fetch_for_term(term, active_only, **kwargs)
            for inst in instruments:
                if inst.instrument_id not in seen_ids:
                    all_matched.append(inst)
                    seen_ids.add(inst.instrument_id)

        if all_matched:
            self._logger.info(
                "Market filter %s: found %d instruments",
                self._filters, len(all_matched),
            )
            for m in all_matched[:10]:
                self._logger.info("  -> %s: %s (mid=%.4f)", m.instrument_id, m.name[:50], m.price)
            if len(all_matched) > 10:
                self._logger.info("  ... and %d more", len(all_matched) - 10)
        else:
            self._logger.warning(
                "Market filter %s: no matching instruments found on %s",
                self._filters, self._exchange_id.value,
            )

        self._cache = all_matched
        self._cache_time = now
        return all_matched

    async def _fetch_for_term(self, term: str, active_only: bool, **kwargs) -> list[Instrument]:
        """Fetch instruments for a single filter term using server-side filtering."""
        from omnitrade.core.enums import ExchangeId

        if self._exchange_id == ExchangeId.KALSHI:
            # Kalshi's API ignores unrecognized filter values and returns
            # default paginated results, so every API response must be
            # validated for relevance before being accepted.
            term_lower = term.lower()

            def _is_relevant(results: list[Instrument]) -> bool:
                """Check that results actually relate to the search term."""
                if not results:
                    return False
                # Sample up to 10 results — if none mention the term, it's junk
                sample = results[:10]
                return any(
                    term_lower in f"{i.instrument_id} {i.name} {i.market_id}".lower()
                    for i in sample
                )

            # 1. series_ticker (broad category, e.g. KXBTC)
            instruments = await self._client.get_instruments(
                active_only=active_only, series_ticker=term, limit=200, **kwargs,
            )
            if not _is_relevant(instruments):
                instruments = []

            # 2. event_ticker (specific event, e.g. KXBTC-25NOV1800)
            if not instruments:
                instruments = await self._client.get_instruments(
                    active_only=active_only, event_ticker=term, limit=200, **kwargs,
                )
                if not _is_relevant(instruments):
                    instruments = []

            # 3. exact ticker
            if not instruments:
                instruments = await self._client.get_instruments(
                    active_only=active_only, ticker=term, limit=200, **kwargs,
                )
                if not _is_relevant(instruments):
                    instruments = []

            # 4. Fetch a batch and filter locally by name
            if not instruments:
                all_inst = await self._client.get_instruments(
                    active_only=active_only, limit=500, **kwargs,
                )
                instruments = [
                    i for i in all_inst
                    if term_lower in f"{i.instrument_id} {i.name} {i.market_id}".lower()
                ]
        elif self._exchange_id == ExchangeId.POLYMARKET:
            # Polymarket Gamma API supports slug/tag search
            instruments = await self._client.get_instruments(
                active_only=active_only, slug=term, limit=200, **kwargs,
            )
            if not instruments:
                instruments = await self._client.get_instruments(
                    active_only=active_only, tag=term, limit=200, **kwargs,
                )
        else:
            # Fallback: fetch all and filter locally
            all_inst = await self._client.get_instruments(active_only=active_only, limit=500, **kwargs)
            term_lower = term.lower()
            instruments = [
                i for i in all_inst
                if term_lower in f"{i.instrument_id} {i.name} {i.market_id}".lower()
            ]

        return instruments

    def __getattr__(self, name):
        return getattr(self._client, name)


def parse_args():
    parser = argparse.ArgumentParser(description="OmniTrade Bot Runner")
    parser.add_argument(
        "bot_type",
        choices=["directional", "mm", "market-making", "hedge", "cross-arb", "copy"],
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--paper", "--dry-run", action="store_true", default=True,
        help="Paper trading mode (default)",
    )
    mode_group.add_argument(
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
        "--signal", "-s",
        choices=["midpoint", "orderbook", "longshot-bias"],
        default="midpoint",
        help="Signal source for directional bot (default: midpoint)",
    )
    parser.add_argument(
        "--market", "-m", type=str, action="append", default=[],
        help="Filter to specific market(s) by keyword or ID (can repeat: -m bitcoin -m ethereum)",
    )
    parser.add_argument(
        "--target", "-t", type=str, action="append", default=[],
        help="Target address/ID to copy trade (can repeat: -t 0xABC -t 0xDEF). "
             "Format: address or address:label:weight (e.g. 0xABC:whale:0.5)",
    )
    parser.add_argument(
        "--targets-file", type=str, default=None,
        help="Path to JSON file with target accounts (list of {address, label, weight})",
    )
    parser.add_argument(
        "--size-multiplier", type=float, default=1.0,
        help="Size multiplier for copy trades (default: 1.0)",
    )
    parser.add_argument(
        "--max-price-deviation", type=float, default=0.05,
        help="Max price deviation from target's entry before skipping (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--no-copy-exits", action="store_true",
        help="Don't copy position exits (only copy opens)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def run_directional(exchange_id, config, agent_id, interval, environment, signal_type="midpoint", market_filters=None):
    """Run a directional bot."""
    from omnitrade.components.signals import MidpointDeviationSignal, FavoriteLongshotSignal, OrderbookMicrostructureSignal
    from omnitrade.components.trading import SignalScaledSizer, FixedSizer
    from omnitrade.exchanges.base import PaperClient
    from omnitrade.bots.directional import DirectionalBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk_config = config.risk
    client = create_client(exchange_id, config)
    await client.connect()

    if market_filters:
        client = MarketFilteredClient(client, market_filters)

    # Paper mode: wrap client to simulate fills
    if environment == Environment.PAPER:
        client = PaperClient(client)

    # Sync balance into risk storage so atomic_reserve can check it
    balance = await client.get_balance()
    max_per_instrument = balance.total_equity * risk_config.max_per_market_exposure_pct

    # For small accounts, lower min trade and use fixed sizing
    if risk_config.min_trade_value_usd > max_per_instrument and max_per_instrument > 0:
        risk_config.min_trade_value_usd = max(1.0, max_per_instrument * 0.5)

    logger.info(
        "Account: $%.2f balance, max $%.2f/instrument, min trade $%.2f",
        balance.total_equity, max_per_instrument, risk_config.min_trade_value_usd,
    )

    risk = RiskCoordinator(storage, risk_config)
    risk.register_account(exchange_id, agent_id)
    storage.update_balance(exchange_id.value, agent_id, balance.total_equity)

    if signal_type == "orderbook":
        signal_source = OrderbookMicrostructureSignal()
    elif signal_type == "longshot-bias":
        signal_source = FavoriteLongshotSignal()
    else:
        signal_source = MidpointDeviationSignal()

    # Size trades to fit within per-instrument risk limits
    trade_size = max_per_instrument * 0.9  # 90% of per-instrument limit
    if signal_type == "longshot-bias":
        sizer = FixedSizer(max(risk_config.min_trade_value_usd, trade_size))
    elif balance.total_equity < 500:
        # Small account: SignalScaledSizer produces tiny sizes, use fixed instead
        sizer = FixedSizer(max(risk_config.min_trade_value_usd, trade_size))
        logger.info("Small account: using FixedSizer at $%.2f per trade", trade_size)
    else:
        sizer = SignalScaledSizer()

    # Widen price filter for longshot-bias (it trades extremes by design)
    price_bounds = {}
    if signal_type == "longshot-bias":
        price_bounds = {"min_price": 0.01, "max_price": 0.99}

    bot = DirectionalBot(
        agent_id=agent_id,
        client=client,
        signal_source=signal_source,
        sizer=sizer,
        risk=risk,
        **price_bounds,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


async def run_market_making(exchange_id, config, agent_id, interval, environment, market_filters=None):
    """Run a market making bot."""
    from omnitrade.bots.market_making import MarketMakingBot, AdaptiveQuoter, ActiveMarketSelector
    from omnitrade.exchanges.base import PaperClient

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk_config = config.risk
    risk = RiskCoordinator(storage, risk_config)
    client = create_client(exchange_id, config)
    await client.connect()

    if market_filters:
        client = MarketFilteredClient(client, market_filters)

    if environment == Environment.PAPER:
        client = PaperClient(client)

    # Scale quote size to account balance
    balance = await client.get_balance()
    risk.register_account(exchange_id, agent_id)
    storage.update_balance(exchange_id.value, agent_id, balance.total_equity)

    max_per_instrument = balance.total_equity * risk_config.max_per_market_exposure_pct
    quote_size = max(risk_config.min_trade_value_usd, max_per_instrument * 0.8)
    max_inventory = balance.total_equity * risk_config.max_per_agent_exposure_pct

    # For small accounts, also lower min_trade so risk checks don't block everything
    if risk_config.min_trade_value_usd > max_per_instrument and max_per_instrument > 0:
        risk_config.min_trade_value_usd = max(1.0, max_per_instrument * 0.5)

    logger.info(
        "Account: $%.2f balance, quote size=$%.2f, max inventory=$%.2f",
        balance.total_equity, quote_size, max_inventory,
    )

    bot = MarketMakingBot(
        agent_id=agent_id,
        client=client,
        quote_engine=AdaptiveQuoter(
            size_usd=quote_size,
            max_inventory=max_inventory,
        ),
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
    from omnitrade.components.signals import BinaryPerpHedgeSignal
    from omnitrade.exchanges.base import PaperClient
    from omnitrade.bots.cross_exchange import CrossExchangeBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)

    # Create clients for both exchanges
    clients = {
        binary_exchange_id: create_client(binary_exchange_id, config),
        hedge_exchange_id: create_client(hedge_exchange_id, config),
    }

    # Paper mode: wrap all clients
    if environment == Environment.PAPER:
        clients = {ex: PaperClient(c) for ex, c in clients.items()}

    signal_source = BinaryPerpHedgeSignal(
        binary_exchange=binary_exchange_id,
        hedge_exchange=hedge_exchange_id,
    )

    bot = CrossExchangeBot(
        agent_id=agent_id,
        clients=clients,
        signal_source=signal_source,
        risk=risk,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


async def run_cross_arb(config, agent_id, interval, environment):
    """Run cross-exchange arb bot (Polymarket vs Kalshi)."""
    from omnitrade.components.signals import CrossExchangeArbSignal
    from omnitrade.exchanges.base import PaperClient
    from omnitrade.bots.cross_exchange import CrossExchangeBot

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)

    clients = {
        ExchangeId.POLYMARKET: create_client(ExchangeId.POLYMARKET, config),
        ExchangeId.KALSHI: create_client(ExchangeId.KALSHI, config),
    }

    if environment == Environment.PAPER:
        clients = {ex: PaperClient(c) for ex, c in clients.items()}

    bot = CrossExchangeBot(
        agent_id=agent_id,
        clients=clients,
        signal_source=CrossExchangeArbSignal(),
        risk=risk,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    finally:
        storage.close()


def parse_targets(args) -> list:
    """Parse target accounts from CLI args and/or JSON file."""
    from omnitrade.bots.copy_trading import TargetAccount

    targets = []

    # Parse --target args: "address" or "address:label:weight"
    for t in args.target:
        parts = t.split(":", 2)
        address = parts[0]
        label = parts[1] if len(parts) > 1 else ""
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        targets.append(TargetAccount(address=address, label=label, weight=weight))

    # Parse --targets-file
    if args.targets_file:
        import json
        with open(args.targets_file) as f:
            data = json.load(f)
        for entry in data:
            if isinstance(entry, str):
                targets.append(TargetAccount(address=entry))
            elif isinstance(entry, dict):
                targets.append(TargetAccount(
                    address=entry["address"],
                    label=entry.get("label", ""),
                    weight=float(entry.get("weight", 1.0)),
                ))

    return targets


async def run_copy(exchange_id, config, agent_id, interval, environment, targets, copy_config):
    """Run a copy trading bot."""
    from omnitrade.bots.copy_trading import CopyTradingBot, TargetTracker
    from omnitrade.exchanges.base import PaperClient

    storage = SQLiteStorage(config.db_path)
    storage.initialize()

    risk = RiskCoordinator(storage, config.risk)
    client = create_client(exchange_id, config)
    await client.connect()

    if environment == Environment.PAPER:
        client = PaperClient(client)

    balance = await client.get_balance()
    risk.register_account(exchange_id, agent_id)
    storage.update_balance(exchange_id.value, agent_id, balance.total_equity)

    logger.info(
        "Account: $%.2f balance, copy multiplier=%.2fx",
        balance.total_equity, copy_config.size_multiplier,
    )

    tracker = TargetTracker()

    bot = CopyTradingBot(
        agent_id=agent_id,
        client=client,
        tracker=tracker,
        targets=targets,
        risk=risk,
        config=copy_config,
    )

    try:
        await bot.run(interval_seconds=interval)
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        if "No valid targets" in str(e):
            print(f"\nERROR: {e}")
            sys.exit(1)
        raise
    finally:
        storage.close()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    environment = Environment.LIVE if args.live else Environment.PAPER

    # Determine which exchange(s) we need for targeted validation
    target_exchange = None
    if args.bot_type in ("directional", "mm", "market-making") and args.exchange:
        target_exchange = ExchangeId(args.exchange)
    elif args.bot_type == "copy":
        target_exchange = ExchangeId.POLYMARKET
    elif args.bot_type == "hedge" and args.exchange:
        target_exchange = ExchangeId(args.exchange)

    # Validate configuration at startup with clear error messages
    try:
        config = validate_startup(exchange=target_exchange)
    except ConfigError as e:
        print(f"\nConfiguration error:\n{e}", file=sys.stderr)
        sys.exit(1)

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

    elif args.bot_type == "copy":
        # Copy trading only works on Polymarket (public position API).
        # Kalshi has no public portfolio API.
        if args.exchange and args.exchange != "polymarket":
            print(f"Error: copy trading only supports Polymarket (Kalshi has no public portfolio API)")
            sys.exit(1)

        exchange_id = ExchangeId.POLYMARKET

        targets = parse_targets(args)
        if not targets:
            print("Error: at least one --target or --targets-file is required for copy bot")
            print("Usage: python scripts/run_bot.py copy -t 0xADDRESS")
            print("       python scripts/run_bot.py copy -t 0xABC:whale:0.5 -t 0xDEF:degen")
            sys.exit(1)

        agent_id = args.agent_id or "copy-polymarket"

        from omnitrade.bots.copy_trading import CopyConfig
        copy_config = CopyConfig(
            size_multiplier=args.size_multiplier,
            max_price_deviation_pct=args.max_price_deviation,
            copy_exits=not args.no_copy_exits,
        )

        print(f"OmniTrade copy bot | polymarket | {mode} mode")
        print(f"Agent: {agent_id} | Interval: {args.interval}s")
        print(f"Targets: {len(targets)}")
        for t in targets:
            w = f" (weight={t.weight})" if t.weight != 1.0 else ""
            display = t.label or t.address
            print(f"  -> {display}{w}")
        print(f"Size multiplier: {copy_config.size_multiplier}x")
        print(f"Copy exits: {copy_config.copy_exits}")
        print("-" * 50)

        asyncio.run(run_copy(exchange_id, config, agent_id, args.interval, environment, targets, copy_config))

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

        market_filters = args.market if args.market else None
        if market_filters:
            print(f"Market filter: {', '.join(market_filters)}")

        if args.bot_type == "directional":
            asyncio.run(run_directional(exchange_id, config, agent_id, args.interval, environment, args.signal, market_filters))
        else:
            asyncio.run(run_market_making(exchange_id, config, agent_id, args.interval, environment, market_filters))


if __name__ == "__main__":
    main()
