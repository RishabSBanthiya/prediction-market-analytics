#!/usr/bin/env python3
"""
Fast Bayesian optimization that pre-fetches data once.

This approach:
1. Fetches all historical market data upfront
2. Runs fast backtests on cached data
3. Uses scikit-optimize for Bayesian optimization
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("ERROR: scikit-optimize required. Run: pip install scikit-optimize")
    sys.exit(1)

from polymarket.core.config import get_config
from polymarket.core.api import PolymarketAPI
from polymarket.backtesting.results import BacktestResults
from polymarket.backtesting.execution import SURVIVORSHIP_BIAS_PENALTY, RealisticExecution

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CachedMarket:
    """Pre-fetched market data for fast backtesting."""
    condition_id: str
    question: str
    tokens: List[Dict]
    winning_outcome: Optional[str]
    price_histories: Dict[str, List[Dict]]  # token_id -> price history
    end_timestamp: int


class FastBondOptimizer:
    """
    Fast Bayesian optimizer for Bond strategy.

    Pre-fetches data once, then runs fast in-memory backtests.
    """

    def __init__(
        self,
        total_days: int = 14,
        train_ratio: float = 0.70,
        n_calls: int = 50,
        n_random_starts: int = 10,
        initial_capital: float = 1000.0,
        reports_dir: str = "reports/optimization"
    ):
        self.total_days = total_days
        self.train_days = int(total_days * train_ratio)
        self.test_days = total_days - self.train_days
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.initial_capital = initial_capital
        self.reports_dir = reports_dir

        # Cached data
        self.train_markets: List[CachedMarket] = []
        self.test_markets: List[CachedMarket] = []

        # Results tracking
        self.all_results: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_train_score: float = float('-inf')
        self.best_test_score: float = float('-inf')

    async def fetch_and_cache_data(self, max_markets: int = 200):
        """Pre-fetch market data for backtesting (limited for speed)."""
        logger.info(f"Pre-fetching up to {max_markets} markets from last {self.total_days} days...")

        config = get_config()
        api = PolymarketAPI(config)
        await api.connect()
        import aiohttp

        try:
            # Fetch closed markets using the API method
            raw_markets = await api.fetch_closed_markets(days=self.total_days)
            logger.info(f"Found {len(raw_markets)} closed markets")

            # Shuffle to randomize, but use consistent seed for reproducibility
            import random
            random.seed(42)
            random.shuffle(raw_markets)
            raw_markets = raw_markets[:min(max_markets * 2, len(raw_markets))]  # Get 2x for filtering

            # We'll use index-based split after collecting valid markets
            now = datetime.now(timezone.utc)
            valid_markets = []

            markets_processed = 0
            async with aiohttp.ClientSession() as session:
                for i, raw_market in enumerate(raw_markets):
                    if markets_processed >= max_markets:
                        break

                    market = api.parse_market(raw_market)
                    if not market or not market.winning_outcome:
                        continue

                    # Fetch price histories via direct API call (faster)
                    price_histories = {}
                    for token in market.tokens:
                        url = f"{config.clob_host}/prices-history?market={token.token_id}&interval=1h"
                        try:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    hist_data = await resp.json()
                                    history = hist_data.get('history', [])
                                    if history:
                                        price_histories[token.token_id] = [
                                            {"timestamp": h.get('t', 0), "price": h.get('p', 0.5)}
                                            for h in history
                                        ]
                        except Exception:
                            continue

                    if not price_histories:
                        continue

                    cached = CachedMarket(
                        condition_id=market.condition_id,
                        question=market.question,
                        tokens=[{"token_id": t.token_id, "outcome": t.outcome} for t in market.tokens],
                        winning_outcome=market.winning_outcome,
                        price_histories=price_histories,
                        end_timestamp=int(market.end_date.timestamp()) if market.end_date else 0
                    )

                    valid_markets.append(cached)
                    markets_processed += 1
                    if markets_processed % 20 == 0:
                        logger.info(f"Processed {markets_processed}/{max_markets} markets...")

            # Index-based train/test split (70% train, 30% test)
            train_cutoff = int(len(valid_markets) * 0.70)
            self.train_markets = valid_markets[:train_cutoff]
            self.test_markets = valid_markets[train_cutoff:]

            logger.info(f"Cached {len(self.train_markets)} train markets, {len(self.test_markets)} test markets")

        finally:
            await api.close()

    def run_fast_backtest(
        self,
        markets: List[CachedMarket],
        params: Dict[str, Any]
    ) -> BacktestResults:
        """
        Run a fast in-memory backtest on cached data.

        No API calls - everything is in memory.
        """
        min_price = params.get("min_price", 0.95)
        max_price = params.get("max_price", 0.98)
        max_spread_pct = params.get("max_spread_pct", 0.02)
        kelly_multiplier = params.get("kelly_multiplier", 1.0)
        max_position_pct = params.get("max_position_pct", 0.15)
        slippage_pct = params.get("slippage_pct", 0.015)
        position_cutoff = params.get("position_cutoff", 0.80)

        execution = RealisticExecution(
            buy_slippage_pct=slippage_pct,
            sell_slippage_pct=slippage_pct,
            max_spread_pct=max_spread_pct,
        )

        results = BacktestResults(
            strategy_name="Bond Strategy (Fast)",
            start_date=datetime.now(timezone.utc) - timedelta(days=self.total_days),
            end_date=datetime.now(timezone.utc),
            initial_capital=self.initial_capital,
        )

        cash = self.initial_capital
        trades_count = 0
        winning_trades = 0
        total_pnl = 0.0

        for market in markets:
            for token_info in market.tokens:
                token_id = token_info["token_id"]
                outcome = token_info["outcome"]

                history = market.price_histories.get(token_id, [])
                if len(history) < 10:
                    continue

                # Look for entry opportunities
                for i, point in enumerate(history):
                    price = point["price"]

                    # Check price range
                    if not (min_price <= price <= max_price):
                        continue

                    # Check position in market life
                    position_ratio = i / len(history)
                    if position_ratio < position_cutoff:
                        continue

                    # Estimate spread from price volatility
                    recent_prices = [p["price"] for p in history[max(0, i-10):i+1]]
                    if len(recent_prices) >= 2:
                        price_range = max(recent_prices) - min(recent_prices)
                        estimated_spread = price_range / price if price > 0 else 0.10
                    else:
                        estimated_spread = 0.02

                    if estimated_spread > max_spread_pct:
                        continue

                    # Calculate position size (simplified Kelly)
                    if price < 0.92:
                        kelly_fraction = 0.0
                    else:
                        edge_factor = (price - 0.92) / (0.99 - 0.92)
                        edge_factor = max(0, min(1, edge_factor))
                        true_prob = price + (1 - price) * 0.3 * edge_factor
                        p = true_prob
                        b = (1.0 / price) - 1
                        if b > 0:
                            kelly = (p * b - (1-p)) / b
                            kelly_fraction = kelly * kelly_multiplier
                            kelly_fraction = max(0, min(max_position_pct, kelly_fraction))
                        else:
                            kelly_fraction = 0

                    if kelly_fraction <= 0:
                        continue

                    position_dollars = cash * kelly_fraction * 0.7  # 70% of kelly
                    position_dollars = min(position_dollars, cash * max_position_pct)

                    if position_dollars < 10:  # Min trade
                        continue

                    # Simulate entry
                    exec_price = price * (1 + slippage_pct)
                    shares = position_dollars / exec_price
                    cost = shares * exec_price

                    if cost > cash:
                        continue

                    cash -= cost

                    # Determine exit (look at final prices and resolution)
                    final_price = history[-1]["price"]
                    is_winner = (outcome == market.winning_outcome)

                    if is_winner:
                        exit_price = 0.99  # Resolved to 1
                    else:
                        exit_price = 0.01  # Resolved to 0

                    proceeds = shares * exit_price
                    cash += proceeds

                    pnl = proceeds - cost
                    total_pnl += pnl
                    trades_count += 1

                    if pnl > 0:
                        winning_trades += 1

                    break  # One trade per token

        results.total_trades = trades_count
        results.winning_trades = winning_trades
        results.losing_trades = trades_count - winning_trades
        results.total_pnl = total_pnl
        results.final_capital = self.initial_capital + total_pnl
        results.markets_traded = trades_count
        results.markets_analyzed = len(markets)

        return results

    def calculate_score(self, results: BacktestResults) -> float:
        """Calculate optimization objective score."""
        if results.total_trades < 3:
            return -1000.0

        adjusted_return = results.return_pct * (1 - SURVIVORSHIP_BIAS_PENALTY)
        win_rate = results.win_rate

        # Profit factor
        if results.losing_trades > 0:
            avg_win = results.total_pnl / results.winning_trades if results.winning_trades > 0 else 0
            avg_loss = abs(results.total_pnl) / results.losing_trades if results.losing_trades > 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            profit_factor = 5.0 if results.winning_trades > 0 else 0
        profit_factor = min(profit_factor, 5.0)

        # Trade count bonus
        trade_bonus = min(results.total_trades / 30, 1.0)

        # Combined score
        score = (
            adjusted_return * 100 * 0.40 +  # 40% weight on returns
            win_rate * 100 * 0.30 +          # 30% weight on win rate
            profit_factor * 10 * 0.20 +      # 20% weight on profit factor
            trade_bonus * 10 * 0.10          # 10% bonus for trade count
        )

        return score

    def _objective(self, params_list: List) -> float:
        """Objective function for optimization (minimizes, so return negative)."""
        param_names = [
            "min_price", "max_price", "max_spread_pct", "kelly_multiplier",
            "max_position_pct", "slippage_pct", "position_cutoff"
        ]
        params = {name: val for name, val in zip(param_names, params_list)}

        # Run on training data
        results = self.run_fast_backtest(self.train_markets, params)
        score = self.calculate_score(results)

        # Track results
        self.all_results.append({
            "params": params,
            "train_score": score,
            "train_return": results.return_pct,
            "train_trades": results.total_trades,
            "train_win_rate": results.win_rate,
        })

        if score > self.best_train_score:
            self.best_train_score = score
            self.best_params = params.copy()
            logger.info(f"New best: score={score:.2f}, return={results.return_pct:.1%}, "
                       f"trades={results.total_trades}, win_rate={results.win_rate:.1%}")

        return -score  # Minimize

    async def optimize(self) -> Dict[str, Any]:
        """Run the optimization pipeline (data must be pre-fetched)."""
        start_time = time.time()

        if not self.train_markets:
            logger.error("No training data available")
            return {}

        logger.info(f"Starting Bayesian optimization with {self.n_calls} iterations...")

        # Define parameter space
        space = [
            Real(0.93, 0.97, name="min_price"),
            Real(0.96, 0.99, name="max_price"),
            Real(0.01, 0.04, name="max_spread_pct"),
            Real(0.5, 1.0, name="kelly_multiplier"),
            Real(0.10, 0.20, name="max_position_pct"),
            Real(0.005, 0.025, name="slippage_pct"),
            Real(0.70, 0.90, name="position_cutoff"),
        ]

        # Run optimization
        result = gp_minimize(
            self._objective,
            space,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            random_state=42,
            verbose=False
        )

        # Validate on test data
        logger.info("Validating on test data...")
        test_results = self.run_fast_backtest(self.test_markets, self.best_params)
        self.best_test_score = self.calculate_score(test_results)

        elapsed = time.time() - start_time

        # Prepare report
        report = {
            "strategy": "bond",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_time_seconds": elapsed,
            "n_iterations": self.n_calls,
            "train_markets": len(self.train_markets),
            "test_markets": len(self.test_markets),
            "best_params": self.best_params,
            "best_train_score": self.best_train_score,
            "best_test_score": self.best_test_score,
            "overfitting_ratio": self.best_train_score / self.best_test_score if self.best_test_score > 0 else float('inf'),
            "test_results": {
                "return_pct": test_results.return_pct,
                "adjusted_return_pct": test_results.return_pct * (1 - SURVIVORSHIP_BIAS_PENALTY),
                "total_trades": test_results.total_trades,
                "win_rate": test_results.win_rate,
            }
        }

        return report


class FastFlowOptimizer:
    """
    Fast Bayesian optimizer for Flow strategy.

    Uses cached price data and simplified signal detection.
    """

    def __init__(
        self,
        total_days: int = 14,
        train_ratio: float = 0.70,
        n_calls: int = 50,
        n_random_starts: int = 10,
        initial_capital: float = 1000.0,
        reports_dir: str = "reports/optimization"
    ):
        self.total_days = total_days
        self.train_days = int(total_days * train_ratio)
        self.test_days = total_days - self.train_days
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.initial_capital = initial_capital
        self.reports_dir = reports_dir

        # Cached data
        self.train_data: List[Dict] = []
        self.test_data: List[Dict] = []

        # Results tracking
        self.all_results: List[Dict] = []
        self.best_params: Optional[Dict] = None
        self.best_train_score: float = float('-inf')
        self.best_test_score: float = float('-inf')

    async def fetch_and_cache_data(self):
        """Pre-fetch market data for flow backtesting."""
        logger.info(f"Pre-fetching {self.total_days} days of market data for flow...")

        config = get_config()

        # Use CLOB client for active markets
        from py_clob_client.client import ClobClient
        import aiohttp

        clob_client = ClobClient(
            config.clob_host,
            key=config.private_key,
            chain_id=config.chain_id,
            signature_type=2,
            funder=config.proxy_address
        )
        clob_client.set_api_creds(clob_client.create_or_derive_api_creds())

        # Get sampling markets
        markets_data = []
        cursor = 'MA=='
        while cursor and len(markets_data) < 100:  # Limit for speed
            try:
                result = clob_client.get_sampling_simplified_markets(next_cursor=cursor)
                data = result.get('data', [])
                markets_data.extend(data)
                cursor = result.get('next_cursor')
                if not data or not cursor:
                    break
            except Exception as e:
                break

        logger.info(f"Found {len(markets_data)} markets")

        # Fetch price history for each token
        async with aiohttp.ClientSession() as session:
            now = datetime.now(timezone.utc)
            test_cutoff = now - timedelta(days=self.test_days)

            for i, market in enumerate(markets_data[:100]):
                tokens = market.get('tokens', [])

                for token in tokens:
                    token_id = token.get('token_id')
                    if not token_id:
                        continue

                    # Fetch price history
                    url = f"{config.clob_host}/prices-history?market={token_id}&interval=1h"
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                history = data.get('history', [])

                                if len(history) < 20:
                                    continue

                                # Parse into train/test using index-based split
                                # First 70% of history = train, last 30% = test
                                train_cutoff_idx = int(len(history) * 0.70)

                                for j, point in enumerate(history):
                                    ts = point.get('t', 0)
                                    price = point.get('p', 0.5)

                                    # Need at least 5 future points for evaluation
                                    if j >= len(history) - 5:
                                        continue

                                    entry = {
                                        "token_id": token_id,
                                        "timestamp": ts,
                                        "price": price,
                                        "history_idx": j,
                                        "history_len": len(history),
                                        "outcome": token.get('outcome', 'Unknown'),
                                        "future_prices": [h.get('p', price) for h in history[j+1:j+16]],
                                    }

                                    # Index-based split: earlier data = train, later = test
                                    if j < train_cutoff_idx:
                                        self.train_data.append(entry)
                                    else:
                                        self.test_data.append(entry)
                    except Exception as e:
                        continue

                if (i + 1) % 20 == 0:
                    logger.info(f"Processed {i+1}/{len(markets_data)} markets")

        logger.info(f"Cached {len(self.train_data)} train points, {len(self.test_data)} test points")

    def detect_signal(self, entry: Dict, params: Dict) -> Tuple[bool, float]:
        """
        Detect if this price point is a flow signal.

        Returns (is_signal, signal_score)
        """
        price = entry["price"]
        future_prices = entry["future_prices"]

        min_price = params.get("min_price", 0.20)
        max_price = params.get("max_price", 0.80)
        price_threshold = params.get("price_threshold", 0.02)

        # Price filter
        if price < min_price or price > max_price:
            return False, 0.0

        if len(future_prices) < 5:
            return False, 0.0

        # Simple signal: sudden price movement
        # Look at 5-point window for momentum
        recent_move = abs(future_prices[0] - price) if future_prices else 0

        if recent_move > price_threshold:
            # Calculate signal score based on move size
            score = recent_move * 100
            return True, score

        return False, 0.0

    def run_fast_backtest(
        self,
        data: List[Dict],
        params: Dict
    ) -> BacktestResults:
        """Run fast in-memory backtest on cached data."""
        min_price = params.get("min_price", 0.20)
        max_price = params.get("max_price", 0.80)
        min_score = params.get("min_score", 30.0)
        take_profit_pct = params.get("take_profit_pct", 0.05)
        stop_loss_pct = params.get("stop_loss_pct", 0.10)
        kelly_multiplier = params.get("kelly_multiplier", 1.0)
        max_position_pct = params.get("max_position_pct", 0.15)
        slippage_pct = params.get("slippage_pct", 0.015)

        results = BacktestResults(
            strategy_name="Flow Strategy (Fast)",
            start_date=datetime.now(timezone.utc) - timedelta(days=self.total_days),
            end_date=datetime.now(timezone.utc),
            initial_capital=self.initial_capital,
        )

        cash = self.initial_capital
        trades_count = 0
        winning_trades = 0
        total_pnl = 0.0

        for entry in data:
            is_signal, score = self.detect_signal(entry, params)

            if not is_signal or score < min_score:
                continue

            price = entry["price"]
            future_prices = entry["future_prices"]

            if len(future_prices) < 3:
                continue

            # Calculate position size
            base_pct = 0.05 * kelly_multiplier
            position_dollars = min(cash * base_pct, cash * max_position_pct)

            if position_dollars < 10:
                continue

            # Entry
            exec_price = price * (1 + slippage_pct)
            shares = position_dollars / exec_price
            cost = shares * exec_price

            if cost > cash:
                continue

            cash -= cost

            # Simulate exit using future prices
            exit_price = exec_price
            for fp in future_prices:
                pnl_pct = (fp - exec_price) / exec_price

                if pnl_pct >= take_profit_pct:
                    exit_price = fp * (1 - slippage_pct)  # Take profit
                    break
                elif pnl_pct <= -stop_loss_pct:
                    exit_price = fp * (1 - slippage_pct)  # Stop loss
                    break
            else:
                # Use last future price
                exit_price = future_prices[-1] * (1 - slippage_pct) if future_prices else exec_price

            proceeds = shares * exit_price
            cash += proceeds

            pnl = proceeds - cost
            total_pnl += pnl
            trades_count += 1

            if pnl > 0:
                winning_trades += 1

        results.total_trades = trades_count
        results.winning_trades = winning_trades
        results.losing_trades = trades_count - winning_trades
        results.total_pnl = total_pnl
        results.final_capital = self.initial_capital + total_pnl
        results.markets_traded = trades_count
        results.markets_analyzed = len(data)

        return results

    def calculate_score(self, results: BacktestResults) -> float:
        """Calculate optimization objective score."""
        if results.total_trades < 5:
            return -1000.0

        adjusted_return = results.return_pct * (1 - SURVIVORSHIP_BIAS_PENALTY)
        win_rate = results.win_rate

        # Trade count bonus
        trade_bonus = min(results.total_trades / 50, 1.0)

        score = (
            adjusted_return * 100 * 0.50 +
            win_rate * 100 * 0.30 +
            trade_bonus * 20 * 0.20
        )

        return score

    def _objective(self, params_list: List) -> float:
        """Objective function for optimization."""
        param_names = [
            "min_price", "max_price", "min_score", "price_threshold",
            "take_profit_pct", "stop_loss_pct", "kelly_multiplier",
            "max_position_pct", "slippage_pct"
        ]
        params = {name: val for name, val in zip(param_names, params_list)}

        results = self.run_fast_backtest(self.train_data, params)
        score = self.calculate_score(results)

        self.all_results.append({
            "params": params,
            "train_score": score,
            "train_return": results.return_pct,
            "train_trades": results.total_trades,
            "train_win_rate": results.win_rate,
        })

        if score > self.best_train_score:
            self.best_train_score = score
            self.best_params = params.copy()
            logger.info(f"New best: score={score:.2f}, return={results.return_pct:.1%}, "
                       f"trades={results.total_trades}, win_rate={results.win_rate:.1%}")

        return -score

    async def optimize(self) -> Dict[str, Any]:
        """Run the full optimization pipeline."""
        start_time = time.time()

        await self.fetch_and_cache_data()

        if not self.train_data:
            logger.error("No training data available")
            return {}

        logger.info(f"Starting Bayesian optimization with {self.n_calls} iterations...")

        space = [
            Real(0.15, 0.40, name="min_price"),
            Real(0.60, 0.85, name="max_price"),
            Real(15.0, 60.0, name="min_score"),
            Real(0.01, 0.05, name="price_threshold"),
            Real(0.03, 0.12, name="take_profit_pct"),
            Real(0.05, 0.20, name="stop_loss_pct"),
            Real(0.5, 1.0, name="kelly_multiplier"),
            Real(0.10, 0.20, name="max_position_pct"),
            Real(0.005, 0.025, name="slippage_pct"),
        ]

        result = gp_minimize(
            self._objective,
            space,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            random_state=42,
            verbose=False
        )

        # Validate on test data
        logger.info("Validating on test data...")
        test_results = self.run_fast_backtest(self.test_data, self.best_params)
        self.best_test_score = self.calculate_score(test_results)

        elapsed = time.time() - start_time

        report = {
            "strategy": "flow",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "optimization_time_seconds": elapsed,
            "n_iterations": self.n_calls,
            "train_datapoints": len(self.train_data),
            "test_datapoints": len(self.test_data),
            "best_params": self.best_params,
            "best_train_score": self.best_train_score,
            "best_test_score": self.best_test_score,
            "overfitting_ratio": self.best_train_score / self.best_test_score if self.best_test_score > 0 else float('inf'),
            "test_results": {
                "return_pct": test_results.return_pct,
                "adjusted_return_pct": test_results.return_pct * (1 - SURVIVORSHIP_BIAS_PENALTY),
                "total_trades": test_results.total_trades,
                "win_rate": test_results.win_rate,
            }
        }

        return report


def save_report(report: Dict, reports_dir: str = "reports/optimization") -> str:
    """Save optimization report to JSON file."""
    Path(reports_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = report.get("strategy", "unknown")
    filename = f"{strategy}_optimization_{timestamp}.json"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved: {filepath}")
    return filepath


def print_summary(report: Dict):
    """Print optimization summary."""
    print("\n" + "=" * 70)
    print(f"OPTIMIZATION RESULTS: {report.get('strategy', 'UNKNOWN').upper()}")
    print("=" * 70)
    print(f"\nTime: {report.get('optimization_time_seconds', 0):.1f} seconds")
    print(f"Iterations: {report.get('n_iterations', 0)}")

    print("\n--- BEST PARAMETERS ---")
    for name, value in report.get("best_params", {}).items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("\n--- PERFORMANCE ---")
    print(f"Train Score: {report.get('best_train_score', 0):.2f}")
    print(f"Test Score:  {report.get('best_test_score', 0):.2f}")
    print(f"Overfitting Ratio: {report.get('overfitting_ratio', 0):.2f} (ideal ~1.0)")

    test_results = report.get("test_results", {})
    print(f"\nTest Return: {test_results.get('return_pct', 0)*100:.1f}%")
    print(f"Adjusted Return: {test_results.get('adjusted_return_pct', 0)*100:.1f}%")
    print(f"Win Rate: {test_results.get('win_rate', 0)*100:.1f}%")
    print(f"Trades: {test_results.get('total_trades', 0)}")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(description="Fast Bayesian optimization")
    parser.add_argument("strategy", choices=["bond", "flow", "all"])
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--random-starts", type=int, default=10)
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--reports-dir", type=str, default="reports/optimization")
    parser.add_argument("--max-markets", type=int, default=150,
                        help="Max markets to fetch for bond optimization (default: 150)")

    args = parser.parse_args()

    reports = {}

    if args.strategy in ("bond", "all"):
        logger.info("Optimizing Bond strategy...")
        optimizer = FastBondOptimizer(
            total_days=args.days,
            n_calls=args.iterations,
            n_random_starts=args.random_starts,
            initial_capital=args.capital,
            reports_dir=args.reports_dir,
        )
        await optimizer.fetch_and_cache_data(max_markets=args.max_markets)
        report = await optimizer.optimize()
        if report:
            save_report(report, args.reports_dir)
            print_summary(report)
            reports["bond"] = report

    if args.strategy in ("flow", "all"):
        logger.info("Optimizing Flow strategy...")
        optimizer = FastFlowOptimizer(
            total_days=args.days,
            n_calls=args.iterations,
            n_random_starts=args.random_starts,
            initial_capital=args.capital,
            reports_dir=args.reports_dir,
        )
        report = await optimizer.optimize()
        if report:
            save_report(report, args.reports_dir)
            print_summary(report)
            reports["flow"] = report

    # Save combined optimized params
    if reports:
        combined_path = os.path.join(args.reports_dir, "optimized_params.json")
        combined = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bond": reports.get("bond", {}).get("best_params"),
            "flow": reports.get("flow", {}).get("best_params"),
        }
        with open(combined_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nCombined params saved to: {combined_path}")


if __name__ == "__main__":
    asyncio.run(main())
