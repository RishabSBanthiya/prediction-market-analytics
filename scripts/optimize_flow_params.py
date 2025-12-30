#!/usr/bin/env python3
"""
Flow Strategy Parameter Optimizer.

Uses Bayesian optimization (Optuna TPE sampler) to find optimal parameters
for the flow copy strategy, exploring:
- Signal detection thresholds
- Price movement thresholds
- Spread tolerances
- Position sizing parameters

Optimization target: Maximum total return (configurable).
"""

import asyncio
import argparse
import logging
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polymarket.backtesting.strategies.flow_backtest import FlowBacktester
from polymarket.backtesting.data.trade_fetcher import TradeFetcher
from polymarket.backtesting.wallet_profiler import BacktestWalletProfiler
from polymarket.trading.components.hedge_monitor import HedgeConfig
from polymarket.trading.components.exit_strategies import ExitConfig

logger = logging.getLogger(__name__)


class FlowParameterOptimizer:
    """
    Bayesian optimizer for flow strategy parameters.
    
    Uses Optuna with TPE (Tree-structured Parzen Estimator) sampler
    for efficient hyperparameter search.
    
    Caches market data to avoid re-fetching for every trial.
    """
    
    def __init__(
        self,
        days: int = 3,
        initial_capital: float = 1000.0,
        metric: str = "return",
        enable_hedging: bool = True,
        enable_trade_signals: bool = True,
        max_markets: int = 200,
        verbose: bool = False,
    ):
        self.days = days
        self.initial_capital = initial_capital
        self.metric = metric
        self.enable_hedging = enable_hedging
        self.enable_trade_signals = enable_trade_signals
        self.max_markets = max_markets
        self.verbose = verbose
        
        # Cache for market data (fetch once, reuse)
        self._cached_price_history: Optional[Dict] = None
        self._cached_token_liquidity: Optional[Dict] = None
        self._cached_market_info: Optional[Dict] = None
        self._cached_markets: Optional[List] = None
        self._cached_trades: Optional[Dict] = None
        self._wallet_profiler: Optional[BacktestWalletProfiler] = None
        
        # Track best results
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float('-inf')
        self.all_results: List[Dict] = []
    
    async def prefetch_data(self):
        """Pre-fetch all market data before optimization."""
        from polymarket.core.api import PolymarketAPI
        from polymarket.core.config import get_config
        from polymarket.core.models import Market, Token
        import aiohttp
        
        logger.info("Pre-fetching market data for flow optimization...")
        
        config = get_config()
        api = PolymarketAPI(config)
        await api.connect()
        
        try:
            # Initialize CLOB client for price history
            from py_clob_client.client import ClobClient
            
            clob_client = ClobClient(
                config.clob_host,
                key=config.private_key,
                chain_id=config.chain_id,
                signature_type=2,
                funder=config.proxy_address
            )
            clob_client.set_api_creds(clob_client.create_or_derive_api_creds())
            logger.info("CLOB client initialized")
            
            # Get all sampling markets (active, liquid markets) with pagination
            logger.info("Fetching active markets...")
            markets_data = []
            cursor = 'MA=='
            
            while cursor:
                try:
                    result = clob_client.get_sampling_simplified_markets(next_cursor=cursor)
                    data = result.get('data', [])
                    markets_data.extend(data)
                    cursor = result.get('next_cursor')
                    
                    if not data or not cursor:
                        break
                except Exception as e:
                    logger.debug(f"Pagination ended: {e}")
                    break
            
            logger.info(f"Found {len(markets_data)} sampling markets")
            
            # Fetch historical price data, spreads, and liquidity for each token
            self._cached_price_history = {}
            self._cached_market_info = {}
            self._cached_token_liquidity = {}
            
            # Limit markets for reasonable optimization time
            num_markets = min(len(markets_data), self.max_markets)
            logger.info(f"Fetching historical price data for {num_markets} markets...")
            
            async with aiohttp.ClientSession() as session:
                for i, market in enumerate(markets_data[:num_markets]):
                    if (i + 1) % 50 == 0:
                        logger.info(f"  Fetching data for market {i+1}/{num_markets}...")
                    
                    condition_id = market.get('condition_id')
                    tokens = market.get('tokens', [])
                    rewards = market.get('rewards', {})
                    
                    # Extract liquidity info from rewards config
                    min_size = rewards.get('min_size', 50)
                    max_spread = rewards.get('max_spread', 5.0)  # Percentage
                    
                    self._cached_market_info[condition_id] = {
                        "tokens": tokens,
                        "active": market.get('active'),
                        "min_size": min_size,
                        "max_spread": max_spread,
                    }
                    
                    for token in tokens:
                        token_id = token.get('token_id')
                        if not token_id:
                            continue
                        
                        # Store token liquidity data
                        token_price = token.get('price', 0.5)
                        self._cached_token_liquidity[token_id] = {
                            "price": token_price,
                            "min_size": min_size,
                            "max_spread_pct": max_spread / 100,  # Convert to decimal
                            # Estimate liquidity from rewards config (markets with rewards are liquid)
                            "estimated_liquidity_usd": min_size * 20 if rewards else 100,
                        }
                        
                        # Fetch price history using the /prices-history endpoint
                        url = f"{config.clob_host}/prices-history?market={token_id}&interval=1h"
                        try:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    data = await resp.json()
                                    history = data.get('history', [])
                                    if history:
                                        self._cached_price_history[token_id] = history
                        except Exception as e:
                            logger.debug(f"Error fetching price history for {token_id}: {e}")
            
            logger.info(f"Fetched price history for {len(self._cached_price_history)} tokens")
            
            # Fetch trade history for wallet profiling (if enabled)
            if self.enable_trade_signals:
                logger.info("Fetching trade history for wallet profiling...")
                async with TradeFetcher(clob_client=clob_client) as fetcher:
                    token_ids = list(self._cached_price_history.keys())
                    
                    def progress_cb(curr, total):
                        if curr % 50 == 0:
                            logger.info(f"  Fetching trades: {curr}/{total} tokens...")
                    
                    self._cached_trades = await fetcher.fetch_trades_for_tokens(
                        token_ids,
                        limit_per_token=100,  # Fewer trades for speed
                        progress_callback=progress_cb,
                    )
                    
                    # Build wallet profiles
                    all_trades = []
                    for trades in self._cached_trades.values():
                        all_trades.extend(trades)
                    
                    if all_trades:
                        self._wallet_profiler = BacktestWalletProfiler()
                        self._wallet_profiler.build_profiles_from_trades(all_trades)
                        logger.info(f"Built wallet profiles from {len(all_trades)} trades")
            
            # Convert to market list format
            from datetime import timedelta
            self._cached_markets = []
            for market in markets_data[:num_markets]:
                tokens = market.get('tokens', [])
                if tokens:
                    condition_id = market.get('condition_id', '')
                    m = Market(
                        condition_id=condition_id,
                        question=f"Market {condition_id[:20]}...",
                        slug=f"market-{condition_id[:16]}",
                        tokens=[Token(
                            token_id=t.get('token_id'),
                            outcome=t.get('outcome', 'Unknown'),
                            price=t.get('price', 0.5)
                        ) for t in tokens],
                        end_date=datetime.now() + timedelta(days=30),
                    )
                    self._cached_markets.append(m)
            
            logger.info(f"Prepared {len(self._cached_markets)} markets for optimization")
            
        finally:
            await api.close()
    
    def _extract_metric(self, results) -> float:
        """Extract the optimization metric from backtest results."""
        if self.metric == "return":
            return results.return_pct
        elif self.metric == "sharpe":
            sharpe = results.sharpe_ratio
            return sharpe if sharpe is not None else -10.0
        elif self.metric == "win_rate":
            return results.win_rate
        elif self.metric == "profit_factor":
            pf = results.profit_factor
            return pf if pf != float('inf') else 10.0
        elif self.metric == "drawdown":
            # Minimize drawdown (return negative for maximization)
            return -results.max_drawdown_pct
        elif self.metric == "trades":
            # Maximize number of trades (for finding more active parameter sets)
            return results.total_trades
        else:
            return results.return_pct
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function (synchronous wrapper for async backtest).
        
        Suggests parameters and runs backtest using FlowBacktester.
        """
        # Signal detection parameters
        price_threshold = trial.suggest_float("price_threshold", 0.01, 0.10, step=0.01)
        volume_multiplier = trial.suggest_float("volume_multiplier", 1.5, 5.0, step=0.5)
        
        # Predictive threshold parameters
        predictive_threshold = trial.suggest_float("predictive_threshold", 0.001, 0.01, step=0.001)
        
        # Spread tolerance
        max_spread_pct = trial.suggest_float("max_spread_pct", 0.02, 0.08, step=0.01)
        
        # Position sizing
        base_position_pct = trial.suggest_float("base_position_pct", 0.01, 0.05, step=0.005)
        max_multiplier = trial.suggest_float("max_multiplier", 1.5, 4.0, step=0.5)
        
        # Hedge parameters (if enabled)
        stop_loss_pct = 0.15
        hedge_trigger_pct = 0.05
        if self.enable_hedging:
            stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.10, 0.25, step=0.05)
            hedge_trigger_pct = trial.suggest_float("hedge_trigger_pct", 0.03, 0.10, step=0.01)
        
        # Exit strategy parameters
        take_profit_pct = trial.suggest_float("take_profit_pct", 0.02, 0.08, step=0.01)
        trailing_activation = trial.suggest_float("trailing_activation", 0.01, 0.05, step=0.01)
        trailing_distance = trial.suggest_float("trailing_distance", 0.005, 0.02, step=0.005)
        max_hold_minutes = trial.suggest_int("max_hold_minutes", 15, 120, step=15)
        
        # Run backtest with suggested parameters
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self._run_backtest(
                    price_threshold=price_threshold,
                    volume_multiplier=volume_multiplier,
                    predictive_threshold=predictive_threshold,
                    max_spread_pct=max_spread_pct,
                    base_position_pct=base_position_pct,
                    max_multiplier=max_multiplier,
                    stop_loss_pct=stop_loss_pct,
                    hedge_trigger_pct=hedge_trigger_pct,
                    take_profit_pct=take_profit_pct,
                    trailing_activation=trailing_activation,
                    trailing_distance=trailing_distance,
                    max_hold_minutes=max_hold_minutes,
                ))
            finally:
                loop.close()
            
            # Extract metric
            value = self._extract_metric(results)
            
            # Store results
            result_dict = {
                "trial": trial.number,
                "price_threshold": price_threshold,
                "volume_multiplier": volume_multiplier,
                "predictive_threshold": predictive_threshold,
                "max_spread_pct": max_spread_pct,
                "base_position_pct": base_position_pct,
                "max_multiplier": max_multiplier,
                "stop_loss_pct": stop_loss_pct,
                "hedge_trigger_pct": hedge_trigger_pct,
                "take_profit_pct": take_profit_pct,
                "trailing_activation": trailing_activation,
                "trailing_distance": trailing_distance,
                "max_hold_minutes": max_hold_minutes,
                "return_pct": results.return_pct,
                "total_pnl": results.total_pnl,
                "win_rate": results.win_rate,
                "total_trades": results.total_trades,
                "sharpe": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown_pct,
                "profit_factor": results.profit_factor,
                "metric_value": value,
            }
            self.all_results.append(result_dict)
            
            # Update best if improved
            if value > self.best_value:
                self.best_value = value
                self.best_params = result_dict.copy()
            
            # Log progress
            if self.verbose or trial.number % 5 == 0:
                logger.info(
                    f"Trial {trial.number}: {self.metric}={value:.4f} "
                    f"(price_thresh: {price_threshold:.2f}, vol_mult: {volume_multiplier:.1f}, "
                    f"spread: {max_spread_pct:.0%}, trades: {results.total_trades})"
                )
            
            return value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            raise optuna.TrialPruned()
    
    async def _run_backtest(
        self,
        price_threshold: float,
        volume_multiplier: float,
        predictive_threshold: float,
        max_spread_pct: float,
        base_position_pct: float,
        max_multiplier: float,
        stop_loss_pct: float,
        hedge_trigger_pct: float,
        take_profit_pct: float,
        trailing_activation: float,
        trailing_distance: float,
        max_hold_minutes: int,
    ):
        """Run backtest with given parameters using FlowBacktester and cached data."""
        hedge_config = HedgeConfig(
            stop_loss_pct=stop_loss_pct,
            price_drop_trigger_pct=hedge_trigger_pct,
        )
        
        # Build exit config
        exit_config = ExitConfig(
            take_profit_pct=take_profit_pct,
            trailing_stop_enabled=True,
            trailing_stop_activation_pct=trailing_activation,
            trailing_stop_distance_pct=trailing_distance,
            max_hold_minutes=max_hold_minutes,
            stop_loss_pct=stop_loss_pct,
        )
        
        backtester = FlowBacktester(
            initial_capital=self.initial_capital,
            days=self.days,
            max_spread_pct=max_spread_pct,
            enable_hedging=self.enable_hedging,
            hedge_config=hedge_config,
            exit_config=exit_config,
            enable_trade_signals=self.enable_trade_signals,
            max_markets=self.max_markets,
            verbose=False,
        )
        
        # Inject cached data
        backtester._price_history = self._cached_price_history
        backtester._token_liquidity = self._cached_token_liquidity
        backtester._market_info = self._cached_market_info
        
        # Inject trade data and wallet profiler (if enabled)
        if self.enable_trade_signals and self._cached_trades:
            backtester._cached_trades = self._cached_trades
            backtester.wallet_profiler = self._wallet_profiler
        
        # Set optimization parameters
        backtester._opt_price_threshold = price_threshold
        backtester._opt_volume_multiplier = volume_multiplier
        backtester._opt_predictive_threshold = predictive_threshold
        backtester._opt_base_position_pct = base_position_pct
        backtester._opt_max_multiplier = max_multiplier
        
        # Run strategy with cached markets
        return await backtester.run_with_cached_data(self._cached_markets)
    
    def create_study(
        self,
        study_name: Optional[str] = None,
        n_startup_trials: int = 10,
    ) -> optuna.Study:
        """Create an Optuna study with TPE sampler."""
        sampler = TPESampler(
            n_startup_trials=n_startup_trials,
            seed=42,
        )
        
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
        )
        
        study = optuna.create_study(
            study_name=study_name or f"flow_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        
        return study
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the optimization.
        
        Args:
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
        
        Returns:
            Dictionary with best parameters and results
        """
        # Pre-fetch markets before optimization starts
        logger.info("Pre-fetching market data for all trials...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.prefetch_data())
        finally:
            loop.close()
        
        if not self._cached_markets:
            logger.error("No markets fetched! Cannot optimize.")
            return {
                "best_params": None,
                "best_value": 0,
                "study": None,
                "all_results": [],
            }
        
        logger.info(f"Using {len(self._cached_markets)} pre-fetched markets for optimization")
        logger.info(f"Price history available for {len(self._cached_price_history)} tokens")
        
        study = self.create_study()
        
        logger.info(f"\nRunning {n_trials} optimization trials...")
        
        # Use Optuna's standard optimize() method with synchronous objective
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose,
        )
        
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "study": study,
            "all_results": self.all_results,
        }
    
    def print_report(self):
        """Print optimization results report."""
        print("\n" + "=" * 70)
        print("FLOW STRATEGY PARAMETER OPTIMIZATION RESULTS")
        print("=" * 70)
        
        if not self.best_params:
            print("\nNo successful trials completed.")
            return
        
        bp = self.best_params
        
        print(f"\n📊 OPTIMIZATION SETTINGS")
        print(f"   Days Backtested:    {self.days}")
        print(f"   Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"   Metric Optimized:   {self.metric}")
        print(f"   Trials Completed:   {len(self.all_results)}")
        print(f"   Hedging Enabled:    {self.enable_hedging}")
        print(f"   Max Markets:        {self.max_markets}")
        
        print(f"\n🏆 BEST PARAMETERS FOUND")
        print("-" * 40)
        print(f"   Price Threshold:    {bp['price_threshold']:.2%}")
        print(f"   Volume Multiplier:  {bp['volume_multiplier']:.1f}x")
        print(f"   Predictive Thresh:  {bp['predictive_threshold']:.3%}")
        print(f"   Max Spread:         {bp['max_spread_pct']:.0%}")
        print(f"   Base Position:      {bp['base_position_pct']:.1%}")
        print(f"   Max Multiplier:     {bp['max_multiplier']:.1f}x")
        
        if self.enable_hedging:
            print(f"   Stop-Loss:          {bp['stop_loss_pct']:.0%}")
            print(f"   Hedge Trigger:      {bp['hedge_trigger_pct']:.0%}")
        
        print(f"\n🚪 EXIT STRATEGY")
        print("-" * 40)
        print(f"   Take-Profit:        {bp.get('take_profit_pct', 0.03):.0%}")
        print(f"   Trailing Activation:{bp.get('trailing_activation', 0.02):.0%}")
        print(f"   Trailing Distance:  {bp.get('trailing_distance', 0.01):.1%}")
        print(f"   Max Hold Time:      {bp.get('max_hold_minutes', 60)} min")
        
        print(f"\n📈 BEST RESULTS")
        print("-" * 40)
        print(f"   Total Return:       {bp['return_pct']:.2%}")
        print(f"   Total P&L:          ${bp['total_pnl']:.2f}")
        print(f"   Win Rate:           {bp['win_rate']:.1%}")
        print(f"   Total Trades:       {bp['total_trades']}")
        print(f"   Sharpe Ratio:       {bp['sharpe']:.2f}" if bp['sharpe'] else "   Sharpe Ratio:       N/A")
        print(f"   Max Drawdown:       {bp['max_drawdown']:.1%}")
        print(f"   Profit Factor:      {bp['profit_factor']:.2f}")
        
        # Show top 5 configurations
        if len(self.all_results) > 1:
            print(f"\n🔝 TOP 5 CONFIGURATIONS")
            print("-" * 70)
            
            sorted_results = sorted(
                self.all_results,
                key=lambda x: x['metric_value'],
                reverse=True
            )[:5]
            
            for i, r in enumerate(sorted_results, 1):
                print(
                    f"   {i}. Thresh: {r['price_threshold']:.0%} | "
                    f"Vol: {r['volume_multiplier']:.1f}x | "
                    f"Spread: {r['max_spread_pct']:.0%} | "
                    f"Return: {r['return_pct']:.2%} | "
                    f"Trades: {r['total_trades']}"
                )
        
        # Show parameter importance (based on correlation with metric)
        if len(self.all_results) > 10:
            print(f"\n📉 PARAMETER SENSITIVITY")
            print("-" * 40)
            self._print_parameter_importance()
        
        print("\n" + "=" * 70)
        print("💡 To run backtest with best params, update flow_backtest.py constants:")
        print(f"   PRICE_THRESHOLD = {bp['price_threshold']:.2f}")
        print(f"   VOLUME_MULTIPLIER = {bp['volume_multiplier']:.1f}")
        print(f"   PREDICTIVE_THRESHOLD = {bp['predictive_threshold']:.4f}")
        print(f"   MAX_SPREAD_PCT = {bp['max_spread_pct']:.2f}")
        print("=" * 70 + "\n")
    
    def _print_parameter_importance(self):
        """Calculate and print rough parameter importance."""
        if len(self.all_results) < 10:
            return
        
        import statistics
        
        params = [
            'price_threshold', 'volume_multiplier', 'predictive_threshold',
            'max_spread_pct', 'base_position_pct', 'max_multiplier',
            'take_profit_pct', 'trailing_activation', 'trailing_distance', 'max_hold_minutes'
        ]
        if self.enable_hedging:
            params.extend(['stop_loss_pct', 'hedge_trigger_pct'])
        
        metrics = [r['metric_value'] for r in self.all_results]
        mean_metric = statistics.mean(metrics)
        
        for param in params:
            values = [r[param] for r in self.all_results]
            
            # Split into high/low groups
            median_val = statistics.median(values)
            high_group = [r['metric_value'] for r, v in zip(self.all_results, values) if v >= median_val]
            low_group = [r['metric_value'] for r, v in zip(self.all_results, values) if v < median_val]
            
            if high_group and low_group:
                high_mean = statistics.mean(high_group)
                low_mean = statistics.mean(low_group)
                impact = abs(high_mean - low_mean)
                direction = "↑" if high_mean > low_mean else "↓"
                print(f"   {param:22s}: {direction} impact={impact:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize flow strategy parameters using Bayesian optimization"
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--days", type=int, default=3, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument("--max-markets", type=int, default=200, help="Max markets to analyze")
    parser.add_argument(
        "--metric",
        choices=["return", "sharpe", "win_rate", "profit_factor", "drawdown", "trades"],
        default="return",
        help="Metric to optimize"
    )
    parser.add_argument("--no-hedge", action="store_true", help="Disable hedging")
    parser.add_argument("--no-trade-signals", action="store_true", help="Disable trade-based signals")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Suppress optuna logs unless verbose
    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print("\n" + "=" * 70)
    print("🔍 FLOW STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"   Trials:         {args.trials}")
    print(f"   Days:           {args.days}")
    print(f"   Capital:        ${args.capital:,.2f}")
    print(f"   Max Markets:    {args.max_markets}")
    print(f"   Metric:         {args.metric}")
    print(f"   Hedging:        {'Disabled' if args.no_hedge else 'Enabled'}")
    print(f"   Trade Signals:  {'Disabled' if args.no_trade_signals else 'Enabled'}")
    print("=" * 70)
    print("\nStarting optimization...\n")
    
    optimizer = FlowParameterOptimizer(
        days=args.days,
        initial_capital=args.capital,
        metric=args.metric,
        enable_hedging=not args.no_hedge,
        enable_trade_signals=not args.no_trade_signals,
        max_markets=args.max_markets,
        verbose=args.verbose,
    )
    
    results = optimizer.optimize(n_trials=args.trials)
    
    optimizer.print_report()
    
    return results


if __name__ == "__main__":
    main()

