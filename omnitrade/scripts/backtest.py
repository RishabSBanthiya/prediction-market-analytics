#!/usr/bin/env python3
"""
Backtest runner for all bot types using real exchange data.

Usage:
    python scripts/backtest.py --data-dir ./data/polymarket --list-markets
    python scripts/backtest.py --data-dir ./data/polymarket --market "presidential"
    python scripts/backtest.py --data-dir ./data/polymarket --market "presidential" --bot mm
    python scripts/backtest.py --data-dir ./data/polymarket --market "cricket" --jobs 4
    python scripts/backtest.py --data-dir ./data/kalshi --exchange kalshi --list-markets
"""

import argparse
import asyncio
import logging
import time
import sys
import os
from typing import Optional

# Force unbuffered stdout for real-time progress
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from omnitrade.backtest.engine import BacktestRunner, BacktestResult, BacktestProgress


def _detect_exchange(data_dir: str) -> Optional[str]:
    """Auto-detect exchange from data directory by peeking at parquet columns."""
    markets_dir = os.path.join(data_dir, "markets")
    if not os.path.isdir(markets_dir):
        return None
    for fname in sorted(os.listdir(markets_dir)):
        if fname.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(os.path.join(markets_dir, fname))
                cols = set(schema.names)
                if "ticker" in cols and "event_ticker" in cols:
                    return "kalshi"
                if "condition_id" in cols:
                    return "polymarket"
            except Exception:
                pass
            break
    return None


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

def _format_time(secs: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    if secs < 0:
        return "--:--"
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_market_time(dt) -> str:
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%d %H:%M")


def _print_progress(p: BacktestProgress) -> None:
    """Print a single-line progress update (overwrites previous line)."""
    pct = p.pct_complete * 100
    bar_width = 20
    filled = int(bar_width * p.pct_complete)
    bar = "#" * filled + "-" * (bar_width - filled)

    eta = _format_time(p.eta_secs)
    elapsed = _format_time(p.elapsed_secs)
    mkt_time = _format_market_time(p.market_time)

    line = (
        f"\r    [{bar}] {pct:5.1f}%  "
        f"elapsed {elapsed}  eta {eta}  "
        f"PnL ${p.pnl:+.2f}  "
        f"trades {p.closed_trades}  "
        f"mid {p.mid_price:.4f}"
    )
    if mkt_time:
        line += f"  @ {mkt_time}"

    # Pad to overwrite previous longer lines
    sys.stdout.write(f"{line:<120}")
    sys.stdout.flush()


def _finish_progress() -> None:
    """Clear the progress line after completion."""
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_results_table(results: list[BacktestResult]) -> None:
    """Print a formatted comparison table of backtest results."""
    print()
    print("=" * 105)
    print(f"{'Market':<18} {'Signal':<28} {'PnL':>10} {'Trades':>8} "
          f"{'Win%':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print("-" * 105)

    for r in results:
        print(
            f"{r.scenario_name:<18} {r.signal_name:<28} "
            f"${r.total_pnl:>+8.2f} {r.total_trades:>8} "
            f"{r.win_rate:>7.1%} {r.max_drawdown_pct:>7.2%} "
            f"{r.sharpe_ratio:>8.2f}"
        )

    print("=" * 105)

    # Per-signal totals
    signal_names = sorted(set(r.signal_name for r in results))
    print()
    print(f"{'Signal':<28} {'Total PnL':>12} {'Trades':>8} {'Avg Win%':>10} {'Avg Sharpe':>12}")
    print("-" * 75)
    for sn in signal_names:
        sig_results = [r for r in results if r.signal_name == sn]
        total_pnl = sum(r.total_pnl for r in sig_results)
        total_trades = sum(r.total_trades for r in sig_results)
        avg_win = sum(r.win_rate for r in sig_results) / len(sig_results) if sig_results else 0
        avg_sharpe = sum(r.sharpe_ratio for r in sig_results) / len(sig_results) if sig_results else 0
        print(f"{sn:<28} ${total_pnl:>+10.2f} {total_trades:>8} "
              f"{avg_win:>9.1%} {avg_sharpe:>12.2f}")
    print()


def _print_mm_results_table(results: list) -> None:
    """Print MM-specific results table."""
    from omnitrade.backtest.mm_engine import MMBacktestResult

    print()
    print("=" * 130)
    print(f"{'Market':<18} {'PnL':>10} {'Trades':>8} {'Win%':>8} {'MaxDD':>8} "
          f"{'Sharpe':>8} {'BidFill%':>9} {'AskFill%':>9} {'Volume':>10} {'PeakInv':>9}")
    print("-" * 130)

    for r in results:
        if isinstance(r, MMBacktestResult):
            print(
                f"{r.scenario_name:<18} "
                f"${r.total_pnl:>+8.2f} {r.total_trades:>8} "
                f"{r.win_rate:>7.1%} {r.max_drawdown_pct:>7.2%} "
                f"{r.sharpe_ratio:>8.2f} "
                f"{r.bid_fill_rate:>8.1%} {r.ask_fill_rate:>8.1%} "
                f"${r.total_volume:>8.0f} ${r.peak_inventory:>7.0f}"
            )

    print("=" * 130)

    # Summary
    total_pnl = sum(r.total_pnl for r in results)
    total_trades = sum(r.total_trades for r in results)
    total_volume = sum(r.total_volume for r in results if isinstance(r, MMBacktestResult))
    avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results) if results else 0
    print()
    print(f"  Total PnL: ${total_pnl:+.2f}  |  Trades: {total_trades}  |  "
          f"Volume: ${total_volume:,.0f}  |  Avg Sharpe: {avg_sharpe:.2f}")
    print()


# ---------------------------------------------------------------------------
# Directional mode
# ---------------------------------------------------------------------------

async def _load_market_data(
    loader, condition_id: str, args: argparse.Namespace,
    *, quiet: bool = False,
) -> tuple[list, str]:
    """Load trades and reconstruct orderbook snapshots for a single market.

    Args:
        loader: Exchange data loader instance.
        condition_id: Market condition ID to load data for.
        args: Parsed CLI arguments (needs .max_trades, .window).
        quiet: If True, suppress print output (used for parallel runs).

    Returns:
        (snapshots, condition_id).  Returns ([], condition_id) if no
        trades are found.
    """
    from omnitrade.backtest.data_loader import OrderbookReconstructor

    def _log(msg: str) -> None:
        if not quiet:
            print(msg)

    t0 = time.monotonic()
    _log("  Loading trades...")
    trades = loader.load_trades(condition_id=condition_id, max_trades=args.max_trades)
    load_time = time.monotonic() - t0
    _log(f"    Loaded {len(trades)} trades in {load_time:.1f}s")

    if not trades:
        _log("    No trades found — skipping this market.")
        return [], condition_id

    _log(f"    Time range: {trades[0].timestamp} -> {trades[-1].timestamp}")
    _log(f"  Reconstructing orderbooks (window={args.window}s)...")

    t0 = time.monotonic()
    reconstructor = OrderbookReconstructor(window_seconds=args.window)
    snapshots = reconstructor.reconstruct(trades, condition_id)
    recon_time = time.monotonic() - t0
    _log(f"    Generated {len(snapshots)} snapshots in {recon_time:.1f}s")
    return snapshots, condition_id


# ---------------------------------------------------------------------------
# Per-market coroutines (used by both sequential and parallel paths)
# ---------------------------------------------------------------------------

async def _backtest_one_directional(
    loader, market_info, args: argparse.Namespace, exchange_id,
    *, quiet: bool = False,
) -> list[BacktestResult]:
    """Run directional backtest on a single market.

    Args:
        loader: Exchange data loader (read-only market cache, safe to share).
        market_info: MarketInfo for the target market.
        args: Parsed CLI arguments.
        exchange_id: ExchangeId enum value for the target exchange.
        quiet: Suppress per-step progress output (for parallel runs).

    Returns:
        List of BacktestResult, one per signal. Empty list if no data.
    """
    from omnitrade.components.signals import (
        OrderbookMicrostructureSignal, MidpointDeviationSignal, FavoriteLongshotSignal,
    )

    condition_id = market_info.condition_id
    scenario_name = market_info.question[:30]

    snapshots, _ = await _load_market_data(loader, condition_id, args, quiet=quiet)
    if not snapshots:
        return []

    signals = [
        OrderbookMicrostructureSignal(),
        MidpointDeviationSignal(fair_value=0.5, min_deviation=0.03),
        FavoriteLongshotSignal(),
    ]

    results: list[BacktestResult] = []
    for signal in signals:
        runner = BacktestRunner(
            signal_source=signal,
            snapshots=snapshots,
            instrument_id=condition_id,
            scenario_name=scenario_name,
            initial_balance=args.initial_balance,
            exchange_id=exchange_id,
            on_progress=None if quiet else _print_progress,
            progress_interval=max(1, len(snapshots) // 40),
            subsample=getattr(args, "subsample", 1),
        )
        result = await runner.run()
        if not quiet:
            _finish_progress()
        results.append(result)
    return results


async def _backtest_one_mm(
    loader, market_info, args: argparse.Namespace, exchange_id,
    *, quiet: bool = False,
) -> list:
    """Run market-making backtest on a single market.

    Args:
        loader: Exchange data loader (read-only market cache, safe to share).
        market_info: MarketInfo for the target market.
        args: Parsed CLI arguments.
        exchange_id: ExchangeId enum value for the target exchange.
        quiet: Suppress per-step progress output (for parallel runs).

    Returns:
        List with a single MMBacktestResult, or empty list if no data.
    """
    from omnitrade.backtest.mm_engine import MMBacktestRunner
    from omnitrade.bots.market_making import AdaptiveQuoter, FillToxicityTracker

    condition_id = market_info.condition_id
    scenario_name = market_info.question[:30]

    snapshots, _ = await _load_market_data(loader, condition_id, args, quiet=quiet)
    if not snapshots:
        return []

    quote_engine = AdaptiveQuoter(
        base_half_spread=0.015, size_usd=25.0,
        toxicity_tracker=FillToxicityTracker(),
    )

    runner = MMBacktestRunner(
        snapshots=snapshots,
        instrument_id=condition_id,
        scenario_name=scenario_name,
        quote_engine=quote_engine,
        initial_balance=args.initial_balance,
        exchange_id=exchange_id,
        on_progress=None if quiet else _print_progress,
        progress_interval=max(1, len(snapshots) // 40),
        subsample=getattr(args, "subsample", 1),
    )
    result = await runner.run()
    if not quiet:
        _finish_progress()
    return [result]


# ---------------------------------------------------------------------------
# Directional mode
# ---------------------------------------------------------------------------

async def run_directional(args: argparse.Namespace) -> None:
    """Run directional backtest across all matched markets.

    When args.jobs > 1, markets are processed concurrently in batches
    using asyncio.gather.
    """
    from omnitrade.components.signals import OrderbookMicrostructureSignal, MidpointDeviationSignal, FavoriteLongshotSignal
    from omnitrade.core.enums import ExchangeId

    if args.exchange == "kalshi":
        from omnitrade.backtest.data_loader import KalshiDataLoader
        loader = KalshiDataLoader(args.data_dir)
    else:
        from omnitrade.backtest.data_loader import PolymarketDataLoader
        loader = PolymarketDataLoader(args.data_dir)

    # Find ALL matching markets
    matched = loader.find_markets(args.market)
    if not matched:
        print(f"Error: No markets found matching '{args.market}'.", file=sys.stderr)
        sys.exit(1)

    if args.max_markets:
        matched = matched[:args.max_markets]

    print(f"Found {len(matched)} market(s) matching '{args.market}':")
    for m in matched:
        q = m.question[:65] + "..." if len(m.question) > 65 else m.question
        print(f"  - {q}  (vol=${m.volume:,.0f})")
    print()

    exchange_id = ExchangeId(args.exchange)
    results: list[BacktestResult] = []

    if args.jobs > 1:
        # -- Parallel execution in batches of args.jobs --
        for batch_start in range(0, len(matched), args.jobs):
            batch = matched[batch_start:batch_start + args.jobs]
            print(f"\nRunning batch of {len(batch)} market(s) concurrently...")

            tasks = [
                _backtest_one_directional(
                    loader, m, args, exchange_id, quiet=True,
                )
                for m in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for m, market_results in zip(batch, batch_results):
                for r in market_results:
                    results.append(r)
                    print(f"  {m.question[:40]}: {r.signal_name} "
                          f"PnL=${r.total_pnl:+.2f}")
    else:
        # -- Sequential execution (original behaviour) --
        for midx, market_info in enumerate(matched, 1):
            condition_id = market_info.condition_id
            scenario_name = market_info.question[:30]
            print(f"\n{'='*80}")
            print(f"[{midx}/{len(matched)}] {market_info.question}")
            print(f"  condition_id: {condition_id}  |  volume: ${market_info.volume:,.0f}")
            print(f"{'='*80}")

            market_results = await _backtest_one_directional(
                loader, market_info, args, exchange_id, quiet=False,
            )
            for r in market_results:
                results.append(r)
                print(f"    PnL=${r.total_pnl:+.2f}, trades={r.total_trades}, "
                      f"win={r.win_rate:.1%}, sharpe={r.sharpe_ratio:.2f}")

    if results:
        _print_results_table(results)
    else:
        print("\nNo results — no trades found for any matched market.")


# ---------------------------------------------------------------------------
# Market-making mode
# ---------------------------------------------------------------------------

async def run_mm(args: argparse.Namespace) -> None:
    """Run market-making backtest across all matched markets.

    When args.jobs > 1, markets are processed concurrently in batches
    using asyncio.gather.
    """
    from omnitrade.core.enums import ExchangeId

    if args.exchange == "kalshi":
        from omnitrade.backtest.data_loader import KalshiDataLoader
        loader = KalshiDataLoader(args.data_dir)
    else:
        from omnitrade.backtest.data_loader import PolymarketDataLoader
        loader = PolymarketDataLoader(args.data_dir)

    # Find ALL matching markets
    matched = loader.find_markets(args.market)
    if not matched:
        print(f"Error: No markets found matching '{args.market}'.", file=sys.stderr)
        sys.exit(1)

    if args.max_markets:
        matched = matched[:args.max_markets]

    print(f"Found {len(matched)} market(s) matching '{args.market}':")
    for m in matched:
        q = m.question[:65] + "..." if len(m.question) > 65 else m.question
        print(f"  - {q}  (vol=${m.volume:,.0f})")
    print()

    exchange_id = ExchangeId(args.exchange)
    results = []

    if args.jobs > 1:
        # -- Parallel execution in batches of args.jobs --
        for batch_start in range(0, len(matched), args.jobs):
            batch = matched[batch_start:batch_start + args.jobs]
            print(f"\nRunning batch of {len(batch)} market(s) concurrently...")

            tasks = [
                _backtest_one_mm(
                    loader, m, args, exchange_id, quiet=True,
                )
                for m in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for m, market_results in zip(batch, batch_results):
                for r in market_results:
                    results.append(r)
                    print(f"  {m.question[:40]}: PnL=${r.total_pnl:+.2f}, "
                          f"volume=${r.total_volume:,.0f}")
    else:
        # -- Sequential execution (original behaviour) --
        for midx, market_info in enumerate(matched, 1):
            condition_id = market_info.condition_id
            print(f"\n{'='*80}")
            print(f"[{midx}/{len(matched)}] {market_info.question}")
            print(f"  condition_id: {condition_id}  |  volume: ${market_info.volume:,.0f}")
            print(f"{'='*80}")

            market_results = await _backtest_one_mm(
                loader, market_info, args, exchange_id, quiet=False,
            )
            for r in market_results:
                results.append(r)
                print(f"    PnL=${r.total_pnl:+.2f}, trades={r.total_trades}, "
                      f"volume=${r.total_volume:,.0f}")

    if results:
        _print_mm_results_table(results)
    else:
        print("\nNo results — no trades found for any matched market.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest runner using real exchange data (Polymarket, Kalshi)"
    )

    # Bot type
    parser.add_argument("--bot", choices=["directional", "mm"],
                        default="directional",
                        help="Bot type to backtest (default: directional)")
    parser.add_argument("--exchange", choices=["polymarket", "kalshi"],
                        default="polymarket",
                        help="Exchange data to backtest against (default: polymarket)")

    # Data options (required)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to Polymarket data directory")
    parser.add_argument("--market", type=str,
                        help="Market query: condition_id or question substring")
    parser.add_argument("--window", type=int, default=30,
                        help="Orderbook reconstruction window in seconds (default: 30)")
    parser.add_argument("--max-trades", type=int,
                        help="Limit number of trades loaded (for quick tests)")
    parser.add_argument("--list-markets", action="store_true",
                        help="List available markets and exit")
    parser.add_argument("--search", type=str,
                        help="Filter --list-markets by keyword (case-insensitive)")
    parser.add_argument("--min-volume", type=float, default=0,
                        help="Minimum volume filter for --list-markets")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of markets to show (default: 50)")

    # Parallelism
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of markets to backtest in parallel (default: 1)")

    # Performance / sampling
    parser.add_argument("--subsample", type=int, default=1,
                        help="Process every Nth snapshot (default: 1, no subsampling)")
    parser.add_argument("--max-markets", type=int,
                        help="Limit number of matched markets to backtest")

    # Shared
    parser.add_argument("--initial-balance", type=int, default=10000,
                        help="Initial balance USD (default: 10000)")

    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    # Show info-level for data loading / backtest progress, suppress noise
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("omnitrade").setLevel(logging.INFO)
    # Suppress noisy submodules during backtest iteration
    logging.getLogger("omnitrade.bots").setLevel(logging.CRITICAL)
    logging.getLogger("omnitrade.risk").setLevel(logging.CRITICAL)
    logging.getLogger("omnitrade.exchanges").setLevel(logging.CRITICAL)
    logging.getLogger("omnitrade.components").setLevel(logging.CRITICAL)
    logging.getLogger("omnitrade.storage").setLevel(logging.CRITICAL)

    # Auto-detect exchange from data if not explicitly set
    if args.exchange == "polymarket":
        detected = _detect_exchange(args.data_dir)
        if detected and detected != args.exchange:
            args.exchange = detected

    # List markets mode
    if args.list_markets:
        if args.exchange == "kalshi":
            from omnitrade.backtest.data_loader import KalshiDataLoader
            loader = KalshiDataLoader(args.data_dir)
        else:
            from omnitrade.backtest.data_loader import PolymarketDataLoader
            loader = PolymarketDataLoader(args.data_dir)
        markets = loader.list_markets(active_only=False, min_volume=args.min_volume)

        # Apply search filter
        if args.search:
            query = args.search.lower()
            markets = [m for m in markets if query in m.question.lower()]

        limit = args.top
        print(f"Found {len(markets)} markets", end="")
        if args.search:
            print(f" matching '{args.search}'", end="")
        if args.min_volume > 0:
            print(f" (min vol ${args.min_volume:,.0f})", end="")
        print(f":\n")
        print(f"{'Volume':>12}  {'Active':>6}  {'Question'}")
        print("-" * 80)
        for m in markets[:limit]:
            q = m.question[:55] + "..." if len(m.question) > 55 else m.question
            print(f"${m.volume:>10,.0f}  {'yes' if m.active else 'no':>6}  {q}")
        if len(markets) > limit:
            print(f"  ... and {len(markets) - limit} more")
        print()
        print("Use --market <substring> to select a market for backtesting.")
        return

    if not args.market:
        print("Error: --market is required (or use --list-markets).", file=sys.stderr)
        sys.exit(1)

    print(f"Backtest ({args.bot}): data_dir={args.data_dir}, balance=${args.initial_balance}", flush=True)
    print(flush=True)

    if args.bot == "mm":
        asyncio.run(run_mm(args))
    else:
        asyncio.run(run_directional(args))


if __name__ == "__main__":
    main()
