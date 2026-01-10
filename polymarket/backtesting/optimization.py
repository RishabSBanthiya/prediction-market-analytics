"""
Simplified Bayesian optimization with anti-overfitting measures.

Key improvements over v2:
- 3-4 parameters only (not 8-9)
- Single objective metric (Sharpe ratio)
- Walk-forward temporal validation (not random market splits)
- L2 regularization toward sensible defaults
- Bootstrap confidence intervals for robustness
- Minimum trade requirements as hard filters
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import os

import numpy as np


try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.callbacks import DeltaYStopper
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not installed. Run: pip install scikit-optimize")

from .results import BacktestResults

logger = logging.getLogger(__name__)


# Sensible defaults for regularization
BOND_DEFAULTS = {
    'entry_price': 0.95,
    'max_spread_pct': 0.03,
    'max_position_pct': 0.10,
}

FLOW_DEFAULTS = {
    # V6 Optimized from real flow alert backtest
    'min_entry_price': 0.50,  # V6: 50c min price
    'max_entry_price': 0.90,  # V6: 90c max price
    'stop_loss_pct': 0.20,  # V6: 20% stop loss
    'take_profit_pct': 0.50,  # V6: 50% take profit
    'max_position_pct': 0.10,
}


@dataclass
class OptimizationConfigV3:
    """Simplified optimization configuration."""
    # Data settings
    total_days: int = 180  # More data for robustness

    # Walk-forward settings
    n_splits: int = 3  # Fewer splits = more data per fold
    holdout_pct: float = 0.25  # Larger holdout

    # Optimization settings
    n_calls: int = 50  # Fewer iterations = less overfitting
    n_random_starts: int = 15
    random_state: int = 42

    # Regularization
    regularization_strength: float = 0.5  # L2 penalty weight

    # Bootstrap
    n_bootstrap: int = 10  # Bootstrap samples for robustness check
    max_cv_coefficient: float = 0.5  # Reject if CV > this

    # Requirements
    min_trades: int = 10  # Hard minimum

    # Capital
    initial_capital: float = 1000.0

    # Output
    reports_dir: str = "reports/optimization_v3"


@dataclass
class SimpleParameterSpace:
    """Minimal parameter space - 3-4 parameters only."""
    name: str
    dimensions: List[Any] = field(default_factory=list)
    dimension_names: List[str] = field(default_factory=list)
    defaults: Dict[str, float] = field(default_factory=dict)

    def to_skopt_space(self) -> List:
        return self.dimensions


def get_bond_parameter_space_v3() -> SimpleParameterSpace:
    """
    Simplified Bond parameter space: 3 parameters only.

    Removed: kelly_multiplier, slippage_pct, position_cutoff, enable_hedging, min_price
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize required")

    return SimpleParameterSpace(
        name="bond",
        dimensions=[
            Real(0.92, 0.97, name="entry_price", prior="uniform"),
            Real(0.01, 0.06, name="max_spread_pct", prior="log-uniform"),
            Real(0.05, 0.20, name="max_position_pct", prior="uniform"),
        ],
        dimension_names=["entry_price", "max_spread_pct", "max_position_pct"],
        defaults=BOND_DEFAULTS,
    )


def get_flow_parameter_space_v3() -> SimpleParameterSpace:
    """
    V6 Flow parameter space - optimized from real flow alert backtest.

    Key parameters:
    - stop_loss_pct: 10-30% (optimal: 20%)
    - take_profit_pct: 30-70% (optimal: 50%)
    - max_position_pct: 5-20% (optimal: 10%)
    """
    if not SKOPT_AVAILABLE:
        raise ImportError("scikit-optimize required")

    return SimpleParameterSpace(
        name="flow",
        dimensions=[
            Real(0.10, 0.30, name="stop_loss_pct", prior="uniform"),
            Real(0.30, 0.70, name="take_profit_pct", prior="uniform"),
            Real(0.05, 0.20, name="max_position_pct", prior="uniform"),
        ],
        dimension_names=["stop_loss_pct", "take_profit_pct", "max_position_pct"],
        defaults=FLOW_DEFAULTS,
    )


def calculate_objective_simple(
    results: BacktestResults,
    min_trades: int = 10,
) -> float:
    """
    Simple objective: Sharpe ratio only.

    No complex multi-metric combinations that can be gamed.
    """
    # Hard filter: minimum trades
    total_trades = results.total_trades or 0
    if total_trades < min_trades:
        return -999.0

    # Use Sharpe ratio as the sole metric
    sharpe = results.sharpe_ratio or 0.0

    # Cap extreme values to avoid lucky outliers
    # Sharpe > 3 is suspicious, < -5 is catastrophic
    return max(-5.0, min(3.0, sharpe))


def calculate_regularization_penalty(
    params: Dict[str, Any],
    defaults: Dict[str, float],
    strength: float = 0.5,
) -> float:
    """
    L2 regularization penalty for deviation from defaults.

    Encourages parameters to stay near known-good values.
    """
    penalty = 0.0

    for name, default in defaults.items():
        if name in params and default != 0:
            value = params[name]
            # Normalized squared deviation
            deviation = (value - default) / default
            penalty += strength * (deviation ** 2)

    return penalty


def calculate_regularized_objective(
    results: BacktestResults,
    params: Dict[str, Any],
    defaults: Dict[str, float],
    config: OptimizationConfigV3,
) -> float:
    """
    Objective with regularization toward defaults.
    """
    base_score = calculate_objective_simple(results, config.min_trades)

    # If base score is penalty (no trades), don't regularize
    if base_score <= -900:
        return base_score

    penalty = calculate_regularization_penalty(
        params, defaults, config.regularization_strength
    )

    return base_score - penalty


@dataclass
class WalkForwardFold:
    """A single walk-forward fold: train on past, test on future."""
    fold_id: int
    train_markets: List[Dict]
    test_markets: List[Dict]
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]


class WalkForwardValidator:
    """
    Walk-forward temporal validation.

    Unlike random market splits, this ensures:
    - Training always on past data
    - Testing always on future data
    - No look-ahead bias possible
    """

    def __init__(
        self,
        markets: List[Dict],
        n_splits: int = 3,
        holdout_pct: float = 0.25,
    ):
        self.markets = markets
        self.n_splits = n_splits
        self.holdout_pct = holdout_pct

        self._prepare_folds()

    def _get_market_date(self, market: Dict) -> datetime:
        """Extract end date from market for sorting."""
        for key in ['end_date', 'endDate', 'closed_time']:
            val = market.get(key)
            if val:
                try:
                    if isinstance(val, str):
                        return datetime.fromisoformat(val.replace('Z', '+00:00'))
                    elif isinstance(val, (int, float)):
                        return datetime.fromtimestamp(val, tz=timezone.utc)
                except (ValueError, TypeError):
                    pass

        # Fallback: use current time
        return datetime.now(timezone.utc)

    def _prepare_folds(self):
        """Prepare walk-forward folds with temporal ordering."""
        # Sort markets by end date (oldest first)
        sorted_markets = sorted(
            self.markets,
            key=lambda m: self._get_market_date(m)
        )

        n_markets = len(sorted_markets)

        # Reserve holdout (most recent markets)
        n_holdout = max(1, int(n_markets * self.holdout_pct))
        self.holdout_markets = sorted_markets[-n_holdout:]
        cv_markets = sorted_markets[:-n_holdout]

        # Create expanding window folds
        # Fold 1: train on 1/3, test on 2/3
        # Fold 2: train on 2/3, test on 3/3
        # etc.
        n_cv = len(cv_markets)
        fold_size = n_cv // (self.n_splits + 1)

        self.folds: List[WalkForwardFold] = []

        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end
            test_end = min((i + 2) * fold_size, n_cv)

            if test_end <= test_start:
                continue

            train_markets = cv_markets[:train_end]
            test_markets = cv_markets[test_start:test_end]

            if train_markets and test_markets:
                self.folds.append(WalkForwardFold(
                    fold_id=i,
                    train_markets=train_markets,
                    test_markets=test_markets,
                    train_period=(
                        self._get_market_date(train_markets[0]),
                        self._get_market_date(train_markets[-1]),
                    ),
                    test_period=(
                        self._get_market_date(test_markets[0]),
                        self._get_market_date(test_markets[-1]),
                    ),
                ))

        logger.info(
            f"Walk-forward: {len(cv_markets)} CV markets in {len(self.folds)} folds, "
            f"{len(self.holdout_markets)} holdout"
        )

    def get_folds(self) -> List[WalkForwardFold]:
        return self.folds

    def get_holdout_markets(self) -> List[Dict]:
        return self.holdout_markets


def bootstrap_evaluate_sync(
    params: Dict[str, Any],
    markets: List[Dict],
    backtest_fn: Callable,  # Synchronous function
    defaults: Dict[str, float],
    config: OptimizationConfigV3,
) -> Tuple[float, float, bool]:
    """
    Bootstrap evaluation for robustness check (SYNC version).

    Returns: (mean_score, std_score, is_stable)
    """
    if not markets:
        return 0.0, float('inf'), False

    scores = []

    for _ in range(config.n_bootstrap):
        # Sample with replacement
        sample = random.choices(markets, k=len(markets))

        try:
            results = backtest_fn(params, sample)
            score = calculate_regularized_objective(results, params, defaults, config)

            if score > -900:  # Valid result
                scores.append(score)
        except Exception as e:
            logger.debug(f"Bootstrap sample failed: {e}")

    if len(scores) < 3:
        return 0.0, float('inf'), False

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    # Check coefficient of variation
    if abs(mean_score) > 0.01:
        cv = std_score / abs(mean_score)
        is_stable = cv <= config.max_cv_coefficient
    else:
        is_stable = std_score < 0.5

    return mean_score, std_score, is_stable


@dataclass
class OptimizationResultV3:
    """Simplified optimization results."""
    strategy_name: str
    best_params: Dict[str, Any]
    best_cv_score: float
    best_holdout_score: float

    # Walk-forward results
    fold_scores: List[float] = field(default_factory=list)

    # Robustness metrics
    cv_std: float = 0.0
    bootstrap_mean: float = 0.0
    bootstrap_std: float = 0.0
    is_robust: bool = False
    overfitting_ratio: float = 0.0

    # Holdout details
    holdout_trades: int = 0
    holdout_return_pct: float = 0.0
    holdout_sharpe: float = 0.0

    # Metadata
    optimization_time_seconds: float = 0.0
    n_iterations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "timestamp": self.timestamp,
            "optimization_time_seconds": self.optimization_time_seconds,
            "n_iterations": self.n_iterations,
            "best_params": self.best_params,
            "best_cv_score": self.best_cv_score,
            "best_holdout_score": self.best_holdout_score,
            "fold_scores": self.fold_scores,
            "cv_std": self.cv_std,
            "bootstrap_mean": self.bootstrap_mean,
            "bootstrap_std": self.bootstrap_std,
            "is_robust": self.is_robust,
            "overfitting_ratio": self.overfitting_ratio,
            "holdout_trades": self.holdout_trades,
            "holdout_return_pct": self.holdout_return_pct,
            "holdout_sharpe": self.holdout_sharpe,
        }


class BayesianOptimizerV3:
    """
    Simplified Bayesian optimizer with anti-overfitting measures.

    Uses SYNCHRONOUS backtest function to avoid async-in-async issues.
    """

    def __init__(
        self,
        strategy_type: str,
        config: OptimizationConfigV3,
        backtest_fn: Callable,  # SYNC function: (params, markets) -> BacktestResults
        markets: List[Dict],
    ):
        if not SKOPT_AVAILABLE:
            raise ImportError("Install scikit-optimize: pip install scikit-optimize")

        self.strategy_type = strategy_type
        self.config = config
        self.backtest_fn = backtest_fn  # Must be synchronous!
        self.markets = markets

        # Get parameter space
        if strategy_type == "bond":
            self.param_space = get_bond_parameter_space_v3()
        elif strategy_type == "flow":
            self.param_space = get_flow_parameter_space_v3()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        # Walk-forward validator
        self.validator = WalkForwardValidator(
            markets=markets,
            n_splits=config.n_splits,
            holdout_pct=config.holdout_pct,
        )

        # Results tracking
        self.all_params: List[Dict] = []
        self.all_scores: List[float] = []

        # Best tracking
        self.best_params: Optional[Dict] = None
        self.best_cv_score: float = float('-inf')

        self._iteration = 0

    def _params_to_dict(self, params: List) -> Dict[str, Any]:
        """Convert parameter list to named dictionary."""
        return {
            name: value
            for name, value in zip(self.param_space.dimension_names, params)
        }

    def _evaluate_fold(
        self,
        params_dict: Dict[str, Any],
        fold: WalkForwardFold,
    ) -> Tuple[float, Optional[BacktestResults]]:
        """
        Evaluate parameters on a single walk-forward fold.

        Tests on TEST markets only (future data).
        """
        try:
            test_markets = fold.test_markets

            if not test_markets:
                logger.warning(f"Fold {fold.fold_id}: No test markets")
                return -999.0, None

            # Run SYNC backtest on test markets
            results = self.backtest_fn(params_dict, test_markets)

            if results is None:
                return -999.0, None

            score = calculate_regularized_objective(
                results, params_dict, self.param_space.defaults, self.config
            )

            logger.debug(
                f"Fold {fold.fold_id}: score={score:.3f}, "
                f"trades={results.total_trades}, sharpe={results.sharpe_ratio:.2f}"
            )

            return score, results

        except Exception as e:
            logger.error(f"Fold {fold.fold_id} failed: {e}")
            return -999.0, None

    def _objective(self, params: List) -> float:
        """
        Objective function with walk-forward validation.
        """
        self._iteration += 1
        params_dict = self._params_to_dict(params)

        # Evaluate across all walk-forward folds
        fold_scores = []
        for fold in self.validator.get_folds():
            score, _ = self._evaluate_fold(params_dict, fold)
            if score > -900:  # Valid result
                fold_scores.append(score)

        if not fold_scores:
            return 100.0  # Penalty for no valid folds

        # Average score across folds
        cv_score = np.mean(fold_scores)

        # Store results
        self.all_params.append(params_dict)
        self.all_scores.append(cv_score)

        # Track best
        if cv_score > self.best_cv_score:
            self.best_cv_score = cv_score
            self.best_params = params_dict.copy()
            logger.info(
                f"[{self._iteration}/{self.config.n_calls}] "
                f"New best: {cv_score:.3f} (params: {params_dict})"
            )

        # Progress log
        if self._iteration % 10 == 0:
            logger.info(f"[{self._iteration}/{self.config.n_calls}] CV score: {cv_score:.3f}")

        # Return negative for minimization
        return -cv_score

    def optimize(self) -> OptimizationResultV3:
        """Run simplified Bayesian optimization (SYNCHRONOUS)."""
        start_time = time.time()

        logger.info(f"Starting V3 optimization for {self.strategy_type}")
        logger.info(f"Parameters: {self.param_space.dimension_names}")
        logger.info(f"Markets: {len(self.markets)}, Folds: {len(self.validator.get_folds())}")
        logger.info(f"Iterations: {self.config.n_calls}")

        # Early stopping
        early_stop = DeltaYStopper(delta=0.1, n_best=10)

        # Run optimization
        result = gp_minimize(
            self._objective,
            self.param_space.to_skopt_space(),
            n_calls=self.config.n_calls,
            n_random_starts=self.config.n_random_starts,
            random_state=self.config.random_state,
            callback=[early_stop],
            verbose=False,
        )

        best_params = self._params_to_dict(result.x)
        best_cv_score = -result.fun

        logger.info(f"Optimization complete. Best CV score: {best_cv_score:.3f}")
        logger.info(f"Best params: {best_params}")

        # Evaluate fold-by-fold for reporting
        fold_scores = []
        for fold in self.validator.get_folds():
            score, _ = self._evaluate_fold(best_params, fold)
            if score > -900:
                fold_scores.append(score)

        cv_std = np.std(fold_scores) if len(fold_scores) > 1 else 0.0

        # Bootstrap robustness check on holdout
        logger.info("Running bootstrap robustness check...")
        holdout_markets = self.validator.get_holdout_markets()

        bootstrap_mean, bootstrap_std, is_robust = bootstrap_evaluate_sync(
            best_params,
            holdout_markets,
            self.backtest_fn,
            self.param_space.defaults,
            self.config,
        )

        # Final holdout evaluation
        logger.info("Evaluating on holdout set...")
        holdout_results = None
        holdout_score = 0.0

        if holdout_markets:
            try:
                holdout_results = self.backtest_fn(best_params, holdout_markets)
                holdout_score = calculate_regularized_objective(
                    holdout_results, best_params, self.param_space.defaults, self.config
                )
                logger.info(f"Holdout score: {holdout_score:.3f}")
            except Exception as e:
                logger.error(f"Holdout evaluation failed: {e}")

        # Calculate overfitting ratio
        if holdout_score > 0.1 and best_cv_score > 0.1:
            overfitting_ratio = best_cv_score / holdout_score
        elif holdout_score <= 0 and best_cv_score > 0:
            overfitting_ratio = 10.0  # Clear overfitting
        else:
            overfitting_ratio = 1.0

        elapsed = time.time() - start_time

        return OptimizationResultV3(
            strategy_name=self.strategy_type,
            best_params=best_params,
            best_cv_score=best_cv_score,
            best_holdout_score=holdout_score,
            fold_scores=fold_scores,
            cv_std=cv_std,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            is_robust=is_robust,
            overfitting_ratio=overfitting_ratio,
            holdout_trades=holdout_results.total_trades if holdout_results else 0,
            holdout_return_pct=holdout_results.return_pct if holdout_results else 0.0,
            holdout_sharpe=holdout_results.sharpe_ratio if holdout_results else 0.0,
            optimization_time_seconds=elapsed,
            n_iterations=len(self.all_params),
        )


def save_optimization_report_v3(
    result: OptimizationResultV3,
    reports_dir: str = "reports/optimization_v3",
) -> str:
    """Save optimization results to JSON."""
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.strategy_name}_v3_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    logger.info(f"Saved report: {filepath}")
    return filepath


def generate_optimization_summary_v3(result: OptimizationResultV3) -> str:
    """Generate human-readable summary."""
    lines = [
        "=" * 60,
        f"OPTIMIZATION REPORT V3: {result.strategy_name.upper()}",
        "=" * 60,
        "",
        f"Duration: {result.optimization_time_seconds:.1f}s",
        f"Iterations: {result.n_iterations}",
        "",
        "--- BEST PARAMETERS (3 only) ---",
    ]

    for name, value in result.best_params.items():
        if isinstance(value, float):
            lines.append(f"  {name}: {value:.4f}")
        else:
            lines.append(f"  {name}: {value}")

    lines.extend([
        "",
        "--- VALIDATION SCORES ---",
        f"CV Score (mean):     {result.best_cv_score:.3f}",
        f"CV Std Dev:          {result.cv_std:.3f}",
        f"Holdout Score:       {result.best_holdout_score:.3f}",
        f"Overfitting Ratio:   {result.overfitting_ratio:.2f}",
        "",
        "Walk-forward fold scores:",
    ])

    for i, score in enumerate(result.fold_scores):
        lines.append(f"  Fold {i}: {score:.3f}")

    lines.extend([
        "",
        "--- ROBUSTNESS CHECK ---",
        f"Bootstrap Mean:      {result.bootstrap_mean:.3f}",
        f"Bootstrap Std:       {result.bootstrap_std:.3f}",
        f"Is Robust:           {'YES' if result.is_robust else 'NO'}",
        "",
        "--- HOLDOUT PERFORMANCE ---",
        f"Trades:              {result.holdout_trades}",
        f"Return:              {result.holdout_return_pct*100:.1f}%",
        f"Sharpe:              {result.holdout_sharpe:.2f}",
        "",
        "--- VERDICT ---",
    ])

    # Provide clear verdict
    if result.overfitting_ratio > 2.0:
        lines.append("  [FAIL] Severe overfitting detected")
    elif result.overfitting_ratio > 1.5:
        lines.append("  [WARN] Moderate overfitting")
    elif not result.is_robust:
        lines.append("  [WARN] Parameters unstable (high bootstrap variance)")
    elif result.holdout_trades < 10:
        lines.append("  [WARN] Too few trades on holdout")
    elif result.holdout_sharpe < 0:
        lines.append("  [FAIL] Negative Sharpe on holdout")
    else:
        lines.append("  [PASS] Parameters appear robust")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
