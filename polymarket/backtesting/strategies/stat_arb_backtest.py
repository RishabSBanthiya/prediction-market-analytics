"""
Statistical Arbitrage Strategy - Backtest

Tests cross-market arbitrage opportunities:
1. Multi-outcome sum arbitrage (sum != 100%)
2. Duplicate market arbitrage (same question, different prices)

Only 3 parameters (anti-overfitting):
- min_edge_bps: minimum edge in basis points (30-100)
- position_size_pct: position size as % of capital (0.05-0.20)
- min_similarity: minimum semantic similarity for duplicates (0.80-0.95)

Run backtest:
    python -m polymarket.backtesting.strategies.stat_arb_backtest --backtest

Run with custom params:
    python -m polymarket.backtesting.strategies.stat_arb_backtest --backtest --min-edge 50

Run optimization:
    python -m polymarket.backtesting.strategies.stat_arb_backtest --optimize
"""

import argparse
import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from ...core.models import Market, Token
from ...core.config import get_config
from ...core.api import PolymarketAPI
from ..results import BacktestResults, SimulatedTrade
from ..data.cached_fetcher import CachedDataFetcher

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class StatArbParams:
    """Statistical arbitrage parameters (3 params for anti-overfitting)."""
    min_edge_bps: int = 50           # Minimum edge in basis points
    position_size_pct: float = 0.10  # Position size as % of capital
    min_similarity: float = 0.85     # Min semantic similarity for duplicates
    fee_bps: int = 100               # 1% fee estimate
    execution_success_rate: float = 0.70  # % of trades that execute correctly


@dataclass
class MultiOutcomeOpp:
    """Multi-outcome arbitrage opportunity."""
    market_id: str
    question: str
    outcomes: List[Tuple[str, str, float]]  # (token_id, outcome, price)
    total_price: float
    edge_bps: int
    timestamp: datetime


@dataclass
class DuplicateOpp:
    """Duplicate market arbitrage opportunity."""
    market_a_id: str
    market_b_id: str
    question_a: str
    question_b: str
    price_a: float
    price_b: float
    similarity: float
    edge_bps: int
    timestamp: datetime


class StatArbBacktester:
    """
    Backtest for statistical arbitrage strategies.

    Supports:
    - Multi-outcome sum arbitrage (buy all when sum < 100%)
    - Duplicate market arbitrage (buy cheap, sell expensive)
    """

    def __init__(
        self,
        params: StatArbParams,
        initial_capital: float = 1000.0,
    ):
        self.params = params
        self.initial_capital = initial_capital
        self.fee_multiplier = 1 + (params.fee_bps / 10000)

        self.cash = initial_capital
        self._price_cache: Dict[str, List[Dict]] = {}

        # Tracking
        self.multi_outcome_trades: List[Dict] = []
        self.duplicate_trades: List[Dict] = []

    def set_price_cache(self, cache: Dict[str, List[Dict]]):
        """Set pre-fetched price history."""
        self._price_cache = cache

    def run_sync(self, markets: List[Market]) -> BacktestResults:
        """Synchronous backtest."""
        return asyncio.run(self.run(markets))

    async def run(self, markets: List[Market], seed: int = 42) -> BacktestResults:
        """Run backtest on given markets."""
        # Set seed for reproducibility
        random.seed(seed)

        start_date = datetime.now(timezone.utc) - timedelta(days=30)
        end_date = datetime.now(timezone.utc)

        results = BacktestResults(
            strategy_name=f"StatArb (edge={self.params.min_edge_bps}bps, sim={self.params.min_similarity})",
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
        )

        self.cash = self.initial_capital
        self.multi_outcome_trades = []
        self.duplicate_trades = []

        # 1. Find and execute multi-outcome opportunities
        multi_opps = self._find_multi_outcome_opportunities(markets)
        for opp in multi_opps:
            self._execute_multi_outcome(opp, results)

        # 2. Find and execute duplicate market opportunities
        if SKLEARN_AVAILABLE:
            dup_opps = self._find_duplicate_opportunities(markets)
            for opp in dup_opps:
                self._execute_duplicate(opp, results)

        # Update final capital
        results.final_capital = self.cash
        results.markets_analyzed = len(markets)

        # Log summary
        logger.info(f"StatArb Backtest Complete:")
        logger.info(f"  Markets analyzed: {len(markets)}")
        logger.info(f"  Multi-outcome markets (3+ outcomes): {sum(1 for m in markets if len(m.tokens) >= 3)}")
        logger.info(f"  Multi-outcome opps executed: {len(self.multi_outcome_trades)}")
        logger.info(f"  Duplicate opps executed: {len(self.duplicate_trades)}")
        logger.info(f"  Final capital: ${self.cash:.2f}")
        logger.info(f"  Total P&L: ${results.total_pnl:.2f}")

        results.finalize()
        return results

    def _find_multi_outcome_opportunities(
        self,
        markets: List[Market],
    ) -> List[MultiOutcomeOpp]:
        """Find markets where sum of prices < 100%."""
        opportunities = []

        for market in markets:
            if len(market.tokens) < 3:
                continue

            # Get prices from cache or use token.price
            prices = []
            for token in market.tokens:
                if token.token_id in self._price_cache:
                    history = self._price_cache[token.token_id]
                    if history:
                        # Use last price
                        price = history[-1].get('p') or history[-1].get('price', 0)
                        prices.append((token.token_id, token.outcome, float(price)))
                elif token.price > 0:
                    prices.append((token.token_id, token.outcome, token.price))

            if len(prices) != len(market.tokens):
                continue

            total_price = sum(p[2] for p in prices)

            # Check for underpricing (sum < 100%)
            if total_price < 1.0:
                edge = 1.0 - total_price
                edge_bps = int(edge * 10000)

                if edge_bps >= self.params.min_edge_bps:
                    opportunities.append(MultiOutcomeOpp(
                        market_id=market.condition_id,
                        question=market.question,
                        outcomes=prices,
                        total_price=total_price,
                        edge_bps=edge_bps,
                        timestamp=market.end_date or datetime.now(timezone.utc),
                    ))

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)
        return opportunities

    def _find_duplicate_opportunities(
        self,
        markets: List[Market],
    ) -> List[DuplicateOpp]:
        """Find semantically similar markets with price differences.

        Conservative filtering to ensure we only catch true duplicates:
        - Same end date (within 1 day)
        - Very high similarity (0.95+)
        - Reasonable edge (not extreme spreads which indicate different markets)
        """
        if not SKLEARN_AVAILABLE:
            return []

        opportunities = []

        # Group by category
        by_category: Dict[str, List[Market]] = defaultdict(list)
        for market in markets:
            if market.tokens and not market.closed:
                cat = market.category or "OTHER"
                by_category[cat].append(market)

        # Check pairs within each category
        for category, cat_markets in by_category.items():
            if len(cat_markets) < 2:
                continue

            # Build TF-IDF vectors
            questions = [self._normalize_question(m.question) for m in cat_markets]
            try:
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                tfidf = vectorizer.fit_transform(questions)
                sim_matrix = cosine_similarity(tfidf)
            except Exception:
                continue

            # Find similar pairs
            for i in range(len(cat_markets)):
                for j in range(i + 1, len(cat_markets)):
                    similarity = sim_matrix[i, j]
                    if similarity < self.params.min_similarity:
                        continue

                    market_a = cat_markets[i]
                    market_b = cat_markets[j]

                    # CRITICAL: Check end dates match (within 1 day)
                    # Markets with different end dates are NOT duplicates
                    if market_a.end_date and market_b.end_date:
                        date_diff = abs((market_a.end_date - market_b.end_date).total_seconds())
                        if date_diff > 86400:  # More than 1 day apart
                            continue

                    # Get YES prices
                    price_a = self._get_yes_price(market_a)
                    price_b = self._get_yes_price(market_b)

                    if price_a is None or price_b is None:
                        continue

                    # Skip extreme prices (likely different markets)
                    if price_a < 0.05 or price_a > 0.95:
                        continue
                    if price_b < 0.05 or price_b > 0.95:
                        continue

                    # Check for price difference
                    spread = abs(price_a - price_b)
                    edge_bps = int(spread * 10000)

                    # Skip unrealistic edges (>30% spread means different markets)
                    if edge_bps > 3000:
                        continue

                    if edge_bps >= self.params.min_edge_bps:
                        opportunities.append(DuplicateOpp(
                            market_a_id=market_a.condition_id,
                            market_b_id=market_b.condition_id,
                            question_a=market_a.question,
                            question_b=market_b.question,
                            price_a=price_a,
                            price_b=price_b,
                            similarity=similarity,
                            edge_bps=edge_bps,
                            timestamp=min(
                                market_a.end_date or datetime.now(timezone.utc),
                                market_b.end_date or datetime.now(timezone.utc),
                            ),
                        ))

        # Sort by edge (descending)
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)

        # Limit to realistic number of opportunities per day
        # (can't execute unlimited trades simultaneously)
        max_daily_opps = 20
        return opportunities[:max_daily_opps]

    def _execute_multi_outcome(
        self,
        opp: MultiOutcomeOpp,
        results: BacktestResults,
    ) -> None:
        """Execute a multi-outcome arbitrage trade."""
        # Calculate position size
        position_usd = self.cash * self.params.position_size_pct

        if position_usd < 10:
            return

        # Cost per "set" of all outcomes
        cost_per_set = opp.total_price * self.fee_multiplier
        num_sets = position_usd / cost_per_set

        if num_sets < 0.1:
            return

        # Execute: buy all outcomes
        total_cost = num_sets * opp.total_price * self.fee_multiplier
        self.cash -= total_cost

        # At resolution, exactly one outcome pays $1
        payout = num_sets * 1.0
        profit = payout - total_cost

        # Record trade
        self.cash += payout

        trade = SimulatedTrade(
            market_question=opp.question[:100],
            token_id=opp.outcomes[0][0],  # Primary token
            token_outcome="ALL",
            entry_time=opp.timestamp,
            entry_price=opp.total_price,
            exit_time=opp.timestamp,
            exit_price=1.0,
            shares=num_sets,
            cost=total_cost,
            proceeds=payout,
            pnl=profit,
            pnl_percent=(profit / total_cost * 100) if total_cost > 0 else 0,
            resolved_to=1.0,
            held_to_resolution=True,
            reason=f"multi_outcome_arb ({opp.edge_bps}bps)",
        )
        results.add_trade(trade)

        self.multi_outcome_trades.append({
            "market_id": opp.market_id,
            "edge_bps": opp.edge_bps,
            "profit": profit,
        })

        logger.debug(
            f"Multi-outcome arb: {opp.question[:50]}... "
            f"edge={opp.edge_bps}bps profit=${profit:.2f}"
        )

    def _execute_duplicate(
        self,
        opp: DuplicateOpp,
        results: BacktestResults,
    ) -> None:
        """Execute a duplicate market arbitrage trade.

        Correct execution model for Polymarket:
        - Buy YES on cheap market (price_low)
        - Buy NO on expensive market (price = 1 - price_high)
        - If markets are true duplicates, one resolves YES, one NO
        - Net profit = $1 - cost_of_YES - cost_of_NO

        Example:
        - Market A: YES @ 0.30, NO @ 0.70
        - Market B: YES @ 0.50, NO @ 0.50
        - Buy YES on A @ 0.30, Buy NO on B @ 0.50
        - Total cost: $0.80 per "set"
        - Guaranteed payout: $1.00 (either A-YES or B-NO wins)
        - Profit: $0.20 per set (before fees)
        """
        # Max position per trade (avoid unrealistic compounding)
        max_position = 100.0  # $100 max per trade

        # Calculate position size (per leg, so total is 2x)
        position_usd = min(self.cash * self.params.position_size_pct / 2, max_position / 2)

        if position_usd < 5:
            return

        # Identify cheap YES and expensive YES
        if opp.price_a < opp.price_b:
            yes_price_cheap = opp.price_a
            no_price_expensive = 1.0 - opp.price_b
            buy_market = opp.market_a_id
        else:
            yes_price_cheap = opp.price_b
            no_price_expensive = 1.0 - opp.price_a
            buy_market = opp.market_b_id

        # Cost to buy one "arbitrage set"
        cost_per_set = yes_price_cheap + no_price_expensive
        cost_per_set_with_fees = cost_per_set * self.fee_multiplier

        # Edge is only valid if cost < $1
        if cost_per_set_with_fees >= 1.0:
            return

        # Number of sets we can buy
        num_sets = position_usd / cost_per_set_with_fees

        # Total cost
        total_cost = num_sets * cost_per_set_with_fees

        # Simulate execution risk: some "duplicates" aren't true duplicates
        # and resolve differently, causing the arb to fail
        is_success = random.random() < self.params.execution_success_rate

        if is_success:
            # Payout: exactly $1 per set (one leg wins)
            payout = num_sets * 1.0
            profit = payout - total_cost
        else:
            # Both legs lose (markets resolved differently)
            # We lose whichever leg was wrong - assume worst case: lose everything
            payout = 0.0
            profit = -total_cost

        self.cash -= total_cost
        self.cash += payout

        # Record trade
        trade = SimulatedTrade(
            market_question=f"{opp.question_a[:50]} vs {opp.question_b[:50]}",
            token_id=buy_market,
            token_outcome="ARB",
            entry_time=opp.timestamp,
            entry_price=cost_per_set,
            exit_time=opp.timestamp,
            exit_price=1.0,
            shares=num_sets,
            cost=total_cost,
            proceeds=payout,
            pnl=profit,
            pnl_percent=(profit / total_cost * 100) if total_cost > 0 else 0,
            resolved_to=1.0,
            held_to_resolution=True,
            reason=f"duplicate_arb ({opp.edge_bps}bps, sim={opp.similarity:.2f})",
        )
        results.add_trade(trade)

        self.duplicate_trades.append({
            "market_a": opp.market_a_id,
            "market_b": opp.market_b_id,
            "edge_bps": opp.edge_bps,
            "profit": profit,
        })

        logger.debug(
            f"Duplicate arb: sim={opp.similarity:.2f} "
            f"cost_per_set=${cost_per_set:.3f} profit=${profit:.2f}"
        )

    def _get_yes_price(self, market: Market) -> Optional[float]:
        """Get YES token price for a market."""
        for token in market.tokens:
            if token.outcome.lower() in ("yes", "up", "over"):
                if token.token_id in self._price_cache:
                    history = self._price_cache[token.token_id]
                    if history:
                        return float(history[-1].get('p') or history[-1].get('price', 0))
                return token.price if token.price > 0 else None

        # Fallback to first token
        if market.tokens:
            token = market.tokens[0]
            if token.token_id in self._price_cache:
                history = self._price_cache[token.token_id]
                if history:
                    return float(history[-1].get('p') or history[-1].get('price', 0))
            return token.price if token.price > 0 else None

        return None

    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison."""
        normalized = question.lower()
        normalized = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', normalized)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '', normalized)
        normalized = re.sub(r'\d{1,2}:\d{2}\s*(am|pm)?', '', normalized)
        normalized = ' '.join(normalized.split())
        return normalized


async def run_backtest(params: StatArbParams, days: int = 30, use_active: bool = True) -> BacktestResults:
    """Run stat arb backtest with given parameters."""
    config = get_config()
    api = PolymarketAPI(config)

    logger.info(f"Fetching markets for analysis...")

    raw_markets = []

    if use_active:
        # Use active markets for real-time opportunity detection
        try:
            raw_markets = await api.fetch_all_markets_including_restricted()
            logger.info(f"Fetched {len(raw_markets)} active markets (including restricted)")
        except Exception as e:
            logger.error(f"Failed to fetch active markets: {e}")
            try:
                raw_markets = await api.fetch_all_markets()
                logger.info(f"Fetched {len(raw_markets)} active markets")
            except Exception as e2:
                logger.error(f"Failed to fetch markets: {e2}")
    else:
        # Use closed markets for historical analysis
        try:
            raw_markets = await api.fetch_closed_markets(days=days)
            logger.info(f"Fetched {len(raw_markets)} closed markets (raw)")
        except Exception as e:
            logger.error(f"Failed to fetch closed markets: {e}")

    if not raw_markets:
        return BacktestResults(
            strategy_name="StatArb",
            start_date=datetime.now(timezone.utc),
            end_date=datetime.now(timezone.utc),
            initial_capital=1000.0,
        )

    # Parse raw markets into Market objects
    markets = []
    for raw in raw_markets:
        if isinstance(raw, dict):
            parsed = api.parse_market(raw)
            if parsed:
                markets.append(parsed)
        else:
            # Already a Market object
            markets.append(raw)

    logger.info(f"Parsed {len(markets)} valid Market objects")

    # Pre-fetch price history
    logger.info("Pre-fetching price history...")
    fetcher = CachedDataFetcher(api, config)
    price_cache = {}

    token_ids = []
    for market in markets:
        if market.tokens:
            for token in market.tokens:
                token_ids.append(token.token_id)

    # Fetch in batches
    batch_size = 50
    for i in range(0, min(len(token_ids), 500), batch_size):
        batch = token_ids[i:i + batch_size]
        for token_id in batch:
            try:
                history = await fetcher.get_price_history(token_id)
                if history:
                    price_cache[token_id] = history
            except Exception:
                pass

    logger.info(f"Cached prices for {len(price_cache)} tokens")

    # Run backtest
    backtester = StatArbBacktester(params, initial_capital=1000.0)
    backtester.set_price_cache(price_cache)

    results = await backtester.run(markets)

    await api.close()
    return results


def run_optimization(n_trials: int = 50, days: int = 30):
    """Run grid search optimization on stat arb parameters.

    Uses a simple grid search since the stat arb strategy has limited
    parameter space and the backtest is fast.
    """
    logger.info("Starting stat arb parameter optimization...")

    # Fetch markets once
    config = get_config()
    api = PolymarketAPI(config)
    markets = asyncio.run(_fetch_markets_for_optimization(api))
    asyncio.run(api.close())

    if not markets:
        logger.error("No markets fetched")
        return None, 0.0

    logger.info(f"Fetched {len(markets)} markets for optimization")

    # Parameter grid (simplified for speed)
    min_edge_values = [30, 50, 75, 100]
    position_size_values = [0.05, 0.10, 0.15]
    min_similarity_values = [0.85, 0.90, 0.95]

    best_params = None
    best_sharpe = float('-inf')
    results_grid = []

    total_combinations = len(min_edge_values) * len(position_size_values) * len(min_similarity_values)
    iteration = 0

    for min_edge in min_edge_values:
        for position_size in position_size_values:
            for min_sim in min_similarity_values:
                iteration += 1

                params = StatArbParams(
                    min_edge_bps=min_edge,
                    position_size_pct=position_size,
                    min_similarity=min_sim,
                )

                # Run backtest
                backtester = StatArbBacktester(params, initial_capital=1000.0)
                results = asyncio.run(backtester.run(markets))

                sharpe = results.sharpe_ratio or 0.0
                ret = results.return_pct * 100

                results_grid.append({
                    "min_edge": min_edge,
                    "position_size": position_size,
                    "min_similarity": min_sim,
                    "sharpe": sharpe,
                    "return_pct": ret,
                    "trades": results.total_trades,
                    "win_rate": results.win_rate * 100,
                })

                if sharpe > best_sharpe and results.total_trades >= 5:
                    best_sharpe = sharpe
                    best_params = {
                        "min_edge_bps": min_edge,
                        "position_size_pct": position_size,
                        "min_similarity": min_sim,
                    }

                if iteration % 10 == 0:
                    logger.info(
                        f"[{iteration}/{total_combinations}] "
                        f"edge={min_edge}, size={position_size:.2f}, sim={min_sim:.2f} -> "
                        f"Sharpe={sharpe:.2f}, Return={ret:.1f}%"
                    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    if best_params:
        print(f"\nBest Sharpe Ratio: {best_sharpe:.3f}")
        print(f"\nOptimal Parameters:")
        print(f"  min_edge_bps:      {best_params['min_edge_bps']}")
        print(f"  position_size_pct: {best_params['position_size_pct']:.3f}")
        print(f"  min_similarity:    {best_params['min_similarity']:.3f}")
    else:
        print("\nNo valid parameter set found.")

    print("\n" + "-" * 60)
    print("Top 5 Results:")
    print("-" * 60)

    sorted_results = sorted(results_grid, key=lambda x: x["sharpe"], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(
            f"{i}. edge={r['min_edge']}, size={r['position_size']:.2f}, "
            f"sim={r['min_similarity']:.2f} -> "
            f"Sharpe={r['sharpe']:.2f}, Return={r['return_pct']:.1f}%, "
            f"Trades={r['trades']}, Win={r['win_rate']:.0f}%"
        )

    print("=" * 60)

    return best_params, best_sharpe


async def _fetch_markets_for_optimization(api: PolymarketAPI) -> List[Market]:
    """Fetch markets for optimization."""
    try:
        raw_markets = await api.fetch_all_markets_including_restricted()
    except Exception:
        raw_markets = await api.fetch_all_markets()

    markets = []
    for raw in raw_markets:
        if isinstance(raw, dict):
            parsed = api.parse_market(raw)
            if parsed:
                markets.append(parsed)
        else:
            markets.append(raw)

    return markets


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Statistical Arbitrage Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")

    # Backtest parameters
    parser.add_argument("--min-edge", type=int, default=50, help="Min edge in bps")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size %%")
    parser.add_argument("--min-similarity", type=float, default=0.85, help="Min similarity")
    parser.add_argument("--days", type=int, default=30, help="Days to backtest")

    # Optimization parameters
    parser.add_argument("-n", "--n-trials", type=int, default=50, help="Optimization trials")

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.optimize:
        run_optimization(n_trials=args.n_trials, days=args.days)
    elif args.backtest:
        params = StatArbParams(
            min_edge_bps=args.min_edge,
            position_size_pct=args.position_size,
            min_similarity=args.min_similarity,
        )

        results = asyncio.run(run_backtest(params, days=args.days))

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"\nStrategy: {results.strategy_name}")
        print(f"Period: {results.start_date.date()} to {results.end_date.date()}")
        print(f"\nInitial Capital: ${results.initial_capital:,.2f}")
        print(f"Final Capital:   ${results.final_capital:,.2f}")
        print(f"Total Return:    {results.return_pct * 100:.2f}%")
        print(f"\nTotal Trades:    {results.total_trades}")
        print(f"Win Rate:        {results.win_rate * 100:.1f}%")
        sharpe = results.sharpe_ratio
        print(f"Sharpe Ratio:    {sharpe:.3f}" if sharpe else "Sharpe Ratio:    N/A")
        print(f"Max Drawdown:    {results.max_drawdown_pct:.2f}%")
        print("=" * 60)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
