#!/usr/bin/env python3
"""
Bond Strategy Parameter Optimizer.

Uses Bayesian optimization (Optuna TPE sampler) to find optimal parameters
for the bond strategy, exploring:
- Entry price ranges (80-99%)
- Time-to-expiry windows
- Dynamic volatility/volume adjustments

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

from polymarket.backtesting.strategies.bond_backtest import BondBacktester, run_bond_backtest
from polymarket.trading.components.hedge_monitor import HedgeConfig

logger = logging.getLogger(__name__)


class BondParameterOptimizer:
    """
    Bayesian optimizer for bond strategy parameters.
    
    Uses Optuna with TPE (Tree-structured Parzen Estimator) sampler
    for efficient hyperparameter search.
    
    Caches market data to avoid re-fetching for every trial.
    """
    
    def __init__(
        self,
        days: int = 7,
        initial_capital: float = 1000.0,
        metric: str = "return",
        enable_hedging: bool = True,
        use_dynamic_time: bool = True,
        verbose: bool = False,
    ):
        self.days = days
        self.initial_capital = initial_capital
        self.metric = metric
        self.enable_hedging = enable_hedging
        self.use_dynamic_time = use_dynamic_time
        self.verbose = verbose
        
        # Cache for market data (fetch once, reuse)
        self._cached_markets: Optional[List] = None
        
        # Track best results
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: float = float('-inf')
        self.all_results: List[Dict] = []
    
    async def prefetch_data(self):
        """Pre-fetch all market data before optimization."""
        from polymarket.core.api import PolymarketAPI
        from polymarket.core.config import get_config
        
        logger.info("Pre-fetching market data...")
        
        config = get_config()
        api = PolymarketAPI(config)
        await api.connect()
        
        try:
            # Fetch closed markets
            raw_markets = await api.fetch_closed_markets(days=self.days, resolved_only=True)
            logger.info(f"Fetched {len(raw_markets)} raw markets")
            
            # Create a temporary backtester to use its prepare_market method
            # This ensures markets are prepared the same way as in actual backtests
            temp_backtester = BondBacktester(
                initial_capital=self.initial_capital,
                days=self.days,
                verbose=False,
            )
            temp_backtester.api = api
            
            # Prepare markets using the same logic as BaseBacktester
            self._cached_markets = []
            for raw in raw_markets:
                market = await temp_backtester.prepare_market(raw)
                if market:
                    self._cached_markets.append(market)
            
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
        else:
            return results.return_pct
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function (synchronous wrapper for async backtest).
        
        Suggests parameters and runs backtest using BondBacktester.
        """
        # Price range parameters
        min_price = trial.suggest_float("min_price", 0.80, 0.96, step=0.01)
        max_price = trial.suggest_float("max_price", min_price + 0.01, 0.99, step=0.01)
        
        # Time window parameters
        min_seconds = trial.suggest_int("min_seconds", 30, 300, step=30)
        max_seconds = trial.suggest_int("max_seconds", 600, 7200, step=300)
        
        # Dynamic time parameters (if enabled)
        vol_multiplier = 1.0
        vol_threshold = 0.05
        volume_weight = 0.5
        
        if self.use_dynamic_time:
            vol_multiplier = trial.suggest_float("vol_multiplier", 0.5, 2.0, step=0.1)
            vol_threshold = trial.suggest_float("vol_threshold", 0.01, 0.10, step=0.01)
            volume_weight = trial.suggest_float("volume_weight", 0.0, 1.0, step=0.1)
        
        # Run backtest with suggested parameters
        try:
            # Use asyncio.run() to execute async backtest in sync context
            # Create new event loop for each trial to avoid conflicts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self._run_backtest(
                    min_price=min_price,
                    max_price=max_price,
                    min_seconds=min_seconds,
                    max_seconds=max_seconds,
                    vol_multiplier=vol_multiplier,
                    vol_threshold=vol_threshold,
                    volume_weight=volume_weight,
                ))
            finally:
                loop.close()
            
            # Extract metric
            value = self._extract_metric(results)
            
            # Store results
            result_dict = {
                "trial": trial.number,
                "min_price": min_price,
                "max_price": max_price,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
                "vol_multiplier": vol_multiplier,
                "vol_threshold": vol_threshold,
                "volume_weight": volume_weight,
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
            if self.verbose or trial.number % 10 == 0:
                logger.info(
                    f"Trial {trial.number}: {self.metric}={value:.4f} "
                    f"(price: {min_price:.2f}-{max_price:.2f}, "
                    f"time: {min_seconds}-{max_seconds}s, trades: {results.total_trades})"
                )
            
            # Early pruning for clearly bad trials
            if results.total_trades < 3:
                raise optuna.TrialPruned()
            
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
        min_price: float,
        max_price: float,
        min_seconds: int,
        max_seconds: int,
        vol_multiplier: float,
        vol_threshold: float,
        volume_weight: float,
    ):
        """Run backtest with given parameters using BondBacktester and cached markets."""
        backtester = BondBacktester(
            initial_capital=self.initial_capital,
            days=self.days,
            min_price=min_price,
            max_price=max_price,
            min_seconds_left=min_seconds,
            max_seconds_left=max_seconds,
            enable_hedging=self.enable_hedging,
            use_dynamic_time=self.use_dynamic_time,
            vol_multiplier=vol_multiplier,
            vol_threshold=vol_threshold,
            volume_weight=volume_weight,
            verbose=False,
        )
        
        # Use pre-fetched markets if available
        return await backtester.run(pre_fetched_markets=self._cached_markets)
    
    
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
            study_name=study_name or f"bond_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        
        return study
    
    def optimize(
        self,
        n_trials: int = 100,
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
        print("BOND STRATEGY PARAMETER OPTIMIZATION RESULTS")
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
        print(f"   Dynamic Time:       {self.use_dynamic_time}")
        
        print(f"\n🏆 BEST PARAMETERS FOUND")
        print("-" * 40)
        print(f"   Price Range:        {bp['min_price']:.2f} - {bp['max_price']:.2f}")
        print(f"   Time Window:        {bp['min_seconds']}s - {bp['max_seconds']}s")
        
        if self.use_dynamic_time:
            print(f"   Vol Multiplier:     {bp['vol_multiplier']:.2f}")
            print(f"   Vol Threshold:      {bp['vol_threshold']:.2f}")
            print(f"   Volume Weight:      {bp['volume_weight']:.2f}")
        
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
                    f"   {i}. Price: {r['min_price']:.2f}-{r['max_price']:.2f} | "
                    f"Time: {r['min_seconds']}-{r['max_seconds']}s | "
                    f"Return: {r['return_pct']:.2%} | "
                    f"Trades: {r['total_trades']}"
                )
        
        # Show parameter importance (based on correlation with metric)
        if len(self.all_results) > 10:
            print(f"\n📉 PARAMETER SENSITIVITY")
            print("-" * 40)
            self._print_parameter_importance()
        
        print("\n" + "=" * 70)
        print("💡 To run backtest with best params:")
        print(f"   python scripts/run_backtest.py bond --days {self.days} \\")
        print(f"       --min-price {bp['min_price']:.2f} --max-price {bp['max_price']:.2f} \\")
        print(f"       --min-seconds {bp['min_seconds']} --max-seconds {bp['max_seconds']}")
        if self.use_dynamic_time:
            print(f"       --dynamic-time --vol-multiplier {bp['vol_multiplier']:.1f}")
        print("=" * 70 + "\n")
    
    def _print_parameter_importance(self):
        """Calculate and print rough parameter importance."""
        if len(self.all_results) < 10:
            return
        
        import statistics
        
        params = ['min_price', 'max_price', 'min_seconds', 'max_seconds']
        if self.use_dynamic_time:
            params.extend(['vol_multiplier', 'vol_threshold', 'volume_weight'])
        
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
                print(f"   {param:18s}: {direction} impact={impact:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize bond strategy parameters using Bayesian optimization"
    )
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--days", type=int, default=7, help="Days to backtest")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital")
    parser.add_argument(
        "--metric",
        choices=["return", "sharpe", "win_rate", "profit_factor", "drawdown"],
        default="return",
        help="Metric to optimize"
    )
    parser.add_argument("--no-hedge", action="store_true", help="Disable hedging")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic time window")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log-level", default="WARNING", help="Log level")
    
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
    print("🔍 BOND STRATEGY PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"   Trials:         {args.trials}")
    print(f"   Days:           {args.days}")
    print(f"   Capital:        ${args.capital:,.2f}")
    print(f"   Metric:         {args.metric}")
    print(f"   Hedging:        {'Disabled' if args.no_hedge else 'Enabled'}")
    print(f"   Dynamic Time:   {'Disabled' if args.no_dynamic else 'Enabled'}")
    print("=" * 70)
    print("\nStarting optimization...\n")
    
    optimizer = BondParameterOptimizer(
        days=args.days,
        initial_capital=args.capital,
        metric=args.metric,
        enable_hedging=not args.no_hedge,
        use_dynamic_time=not args.no_dynamic,
        verbose=args.verbose,
    )
    
    results = optimizer.optimize(n_trials=args.trials)
    
    optimizer.print_report()
    
    return results


if __name__ == "__main__":
    main()

