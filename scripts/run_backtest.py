#!/usr/bin/env python3
"""
Unified Backtest Runner.

Run backtests for different strategies.

Usage:
    # Bond strategy backtest
    python run_backtest.py bond --capital 1000 --days 7
    
    # Flow signal backtest
    python run_backtest.py flow --capital 1000 --days 7
    
    # Save results to JSON
    python run_backtest.py bond --capital 1000 --output results.json
"""

import asyncio
import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def run_bond_backtest(args):
    """Run bond strategy backtest"""
    from polymarket.backtesting.strategies.bond_backtest import BondBacktester
    
    backtester = BondBacktester(
        initial_capital=args.capital,
        days=args.days,
        min_price=args.min_price,
        max_price=args.max_price,
        verbose=args.verbose,
    )
    
    results = await backtester.run()
    results.print_report()
    
    return results


async def run_flow_backtest(args):
    """Run flow signal backtest"""
    from polymarket.backtesting.strategies.flow_backtest import FlowBacktester
    from polymarket.trading.components.hedge_monitor import HedgeConfig
    
    # Build hedge config from args
    hedge_config = HedgeConfig(
        stop_loss_pct=getattr(args, 'stop_loss', 0.15),
        price_drop_trigger_pct=getattr(args, 'hedge_trigger', 0.05),
    )
    
    backtester = FlowBacktester(
        initial_capital=args.capital,
        days=args.days,
        min_trade_size=args.min_trade_size,
        verbose=args.verbose,
        optimize_params=getattr(args, 'optimize', False),
        max_markets=getattr(args, 'max_markets', 200),
        enable_hedging=not getattr(args, 'no_hedge', False),
        hedge_config=hedge_config,
    )
    
    results = await backtester.run()
    results.print_report()
    
    # Print detailed trade breakdown if verbose
    if args.verbose and results.trades:
        results.print_trade_details(limit=20)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py bond --capital 1000 --days 7
  python run_backtest.py flow --capital 1000 --days 14
  python run_backtest.py bond --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="strategy", help="Strategy to backtest")
    
    # Bond strategy
    bond_parser = subparsers.add_parser("bond", help="Expiring market strategy")
    bond_parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    bond_parser.add_argument("--days", type=int, default=3, help="Days to backtest")
    bond_parser.add_argument("--min-price", type=float, default=0.95, help="Minimum price")
    bond_parser.add_argument("--max-price", type=float, default=0.98, help="Maximum price")
    bond_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Flow strategy
    flow_parser = subparsers.add_parser("flow", help="Flow signal analysis")
    flow_parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    flow_parser.add_argument("--days", type=int, default=3, help="Days to backtest")
    flow_parser.add_argument("--min-trade-size", type=float, default=100.0, help="Min trade size")
    flow_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    flow_parser.add_argument("--optimize", action="store_true", help="Run parameter optimization")
    flow_parser.add_argument("--max-markets", type=int, default=200, help="Max markets to analyze (default: 200)")
    flow_parser.add_argument("--no-hedge", action="store_true", help="Disable hedging")
    flow_parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop-loss threshold (default 15%%)")
    flow_parser.add_argument("--hedge-trigger", type=float, default=0.05, help="Hedge trigger threshold (default 5%%)")
    
    # Global args
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file")
    parser.add_argument("--log-level", default="WARNING", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level")
    
    args = parser.parse_args()
    
    if not args.strategy:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Run backtest
    if args.strategy == "bond":
        results = asyncio.run(run_bond_backtest(args))
    elif args.strategy == "flow":
        results = asyncio.run(run_flow_backtest(args))
    else:
        parser.print_help()
        sys.exit(1)
    
    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

