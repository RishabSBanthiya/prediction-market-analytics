"""
Duplicate Market Arbitrage Scanner.

Detects markets with semantically identical questions but different prices.

Strategy: Buy the cheaper version, sell the expensive version.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Tuple
from itertools import combinations
import uuid

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market

from ..models import (
    ArbType,
    StatArbOpportunity,
    ArbLeg,
    MarketCluster,
)
from ..config import DuplicateConfig
from ..correlation_engine import CorrelationEngine

logger = logging.getLogger(__name__)


class DuplicateMarketScanner:
    """
    Scanner for duplicate market arbitrage.

    Finds markets with:
    - Very similar questions (semantic similarity >= threshold)
    - Similar resolution dates
    - Different prices

    Creates opportunities to buy cheap / sell expensive.
    """

    # Limit concurrent API calls to avoid overwhelming the rate limiter
    MAX_CONCURRENT = 10
    # Limit pairs to check per scan to avoid long blocking
    MAX_PAIRS_PER_SCAN = 50

    def __init__(
        self,
        api: PolymarketAPI,
        config: DuplicateConfig,
        correlation_engine: CorrelationEngine,
    ):
        self.api = api
        self.config = config
        self.correlation_engine = correlation_engine
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    async def scan(self, markets: List[Market]) -> List[StatArbOpportunity]:
        """
        Scan for duplicate market arbitrage opportunities.

        Returns list of opportunities sorted by edge.
        """
        opportunities: List[StatArbOpportunity] = []

        # Filter to active markets
        active_markets = [m for m in markets if not m.closed and m.tokens]

        if len(active_markets) < 2:
            return []

        logger.debug(f"Scanning {len(active_markets)} markets for duplicates")

        # Get semantic clusters
        clusters = self.correlation_engine.cluster_markets_by_similarity(
            active_markets,
            threshold=self.config.min_similarity,
        )

        logger.debug(f"Found {len(clusters)} semantic clusters")

        # Collect all pairs to check
        pairs_to_check = []
        for cluster in clusters:
            if len(cluster.market_ids) < 2:
                continue

            # Get markets in this cluster
            cluster_markets = [
                m for m in active_markets
                if m.condition_id in cluster.market_ids
            ]

            # Add all pairs within cluster
            for market_a, market_b in combinations(cluster_markets, 2):
                pairs_to_check.append((market_a, market_b))

        # Limit pairs to avoid long blocking scans
        if len(pairs_to_check) > self.MAX_PAIRS_PER_SCAN:
            logger.info(f"Limiting duplicate scan from {len(pairs_to_check)} to {self.MAX_PAIRS_PER_SCAN} pairs")
            pairs_to_check = pairs_to_check[:self.MAX_PAIRS_PER_SCAN]

        logger.info(f"Checking {len(pairs_to_check)} duplicate market pairs (max {self.MAX_CONCURRENT} concurrent)")

        # Check pairs with concurrency limit
        async def check_with_semaphore(pair):
            async with self._semaphore:
                return await self._check_pair(pair[0], pair[1])

        tasks = [check_with_semaphore(pair) for pair in pairs_to_check]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Pair check failed: {result}")
                continue
            if result:
                opportunities.append(result)

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)

        if opportunities:
            logger.info(
                f"Found {len(opportunities)} duplicate market opportunities "
                f"(best edge: {opportunities[0].edge_bps} bps)"
            )

        return opportunities

    async def _check_pair(
        self,
        market_a: Market,
        market_b: Market,
    ) -> Optional[StatArbOpportunity]:
        """Check a pair of similar markets for arbitrage."""
        # Verify category match if required
        if self.config.require_same_category:
            if (market_a.category or "OTHER") != (market_b.category or "OTHER"):
                return None

        # Verify end date proximity
        if market_a.end_date and market_b.end_date:
            diff_hours = abs((market_a.end_date - market_b.end_date).total_seconds()) / 3600
            if diff_hours > self.config.max_end_date_diff_hours:
                logger.debug(
                    f"End dates too different: {diff_hours:.1f} hours apart"
                )
                return None

        # Get YES token prices for both markets
        price_a = await self._get_yes_price(market_a)
        price_b = await self._get_yes_price(market_b)

        if price_a is None or price_b is None:
            return None

        # For duplicate arb: buy YES in cheap market, buy NO in expensive market
        # Arb exists if YES_cheap + NO_expensive < $1
        # Since NO ~= 1 - YES, if YES_A is cheap and YES_B is expensive,
        # then NO_B is cheap. We buy YES_A + NO_B.

        if price_a["ask"] < price_b["ask"]:
            # A has cheaper YES, B has expensive YES (so B has cheaper NO)
            return await self._create_opportunity(
                buy_market=market_a,
                sell_market=market_b,
                buy_price=price_a,
                sell_price=price_b,
            )
        else:
            # B has cheaper YES, A has expensive YES (so A has cheaper NO)
            return await self._create_opportunity(
                buy_market=market_b,
                sell_market=market_a,
                buy_price=price_b,
                sell_price=price_a,
            )

    async def _get_yes_price(self, market: Market) -> Optional[Dict[str, float]]:
        """Get YES token prices (bid/ask) for a market."""
        yes_token = None
        for token in market.tokens:
            if token.outcome.lower() in ("yes", "up", "over"):
                yes_token = token
                break

        if not yes_token:
            yes_token = market.tokens[0] if market.tokens else None

        if not yes_token:
            return None

        try:
            orderbook = await self.api.fetch_orderbook(yes_token.token_id)
            if not orderbook or not orderbook.best_ask or not orderbook.best_bid:
                return None

            # Calculate available liquidity at best prices
            ask_liquidity = sum(
                float(level.size) for level in (orderbook.asks or [])[:3]
            ) if orderbook.asks else 0
            bid_liquidity = sum(
                float(level.size) for level in (orderbook.bids or [])[:3]
            ) if orderbook.bids else 0

            return {
                "token_id": yes_token.token_id,
                "outcome": yes_token.outcome,
                "ask": orderbook.best_ask,
                "bid": orderbook.best_bid,
                "spread": orderbook.best_ask - orderbook.best_bid,
                "ask_liquidity": ask_liquidity,
                "bid_liquidity": bid_liquidity,
            }
        except Exception as e:
            logger.debug(f"Failed to get price for {market.condition_id}: {e}")
            return None

    async def _get_no_price(self, market: Market) -> Optional[Dict[str, float]]:
        """Get NO token prices (bid/ask) for a market."""
        no_token = None
        for token in market.tokens:
            if token.outcome.lower() in ("no", "down", "under"):
                no_token = token
                break

        # If no explicit NO token, use the second token (binary markets)
        if not no_token and len(market.tokens) >= 2:
            # Find token that's not YES
            for token in market.tokens:
                if token.outcome.lower() not in ("yes", "up", "over"):
                    no_token = token
                    break

        if not no_token:
            return None

        try:
            orderbook = await self.api.fetch_orderbook(no_token.token_id)
            if not orderbook or not orderbook.best_ask or not orderbook.best_bid:
                return None

            # Calculate available liquidity at best prices
            ask_liquidity = sum(
                float(level.size) for level in (orderbook.asks or [])[:3]
            ) if orderbook.asks else 0
            bid_liquidity = sum(
                float(level.size) for level in (orderbook.bids or [])[:3]
            ) if orderbook.bids else 0

            return {
                "token_id": no_token.token_id,
                "outcome": no_token.outcome,
                "ask": orderbook.best_ask,
                "bid": orderbook.best_bid,
                "spread": orderbook.best_ask - orderbook.best_bid,
                "ask_liquidity": ask_liquidity,
                "bid_liquidity": bid_liquidity,
            }
        except Exception as e:
            logger.debug(f"Failed to get NO price for {market.condition_id}: {e}")
            return None

    async def _create_opportunity(
        self,
        buy_market: Market,
        sell_market: Market,
        buy_price: Dict[str, float],
        sell_price: Dict[str, float],
    ) -> Optional[StatArbOpportunity]:
        """Create a duplicate market arbitrage opportunity.

        For Polymarket, we can't sell tokens we don't own. Instead:
        - BUY YES in cheap YES market
        - BUY NO in expensive YES market (NO is cheap when YES is expensive)

        At resolution, one of the tokens pays $1, guaranteeing profit if:
        cost(YES_cheap) + cost(NO_expensive) < $1
        """
        # Get NO token from the expensive market
        no_price = await self._get_no_price(sell_market)
        if no_price is None:
            return None

        # Check minimum liquidity - need at least $50 worth available
        MIN_LIQUIDITY_USD = 50.0
        yes_liquidity_usd = buy_price.get("ask_liquidity", 0) * buy_price["ask"]
        no_liquidity_usd = no_price.get("ask_liquidity", 0) * no_price["ask"]

        if yes_liquidity_usd < MIN_LIQUIDITY_USD or no_liquidity_usd < MIN_LIQUIDITY_USD:
            logger.debug(
                f"Insufficient liquidity: YES=${yes_liquidity_usd:.2f}, NO=${no_liquidity_usd:.2f}"
            )
            return None

        # Edge = $1 - (YES_cheap + NO_expensive)
        total_cost_per_set = buy_price["ask"] + no_price["ask"]
        if total_cost_per_set >= 1.0:
            return None  # No arb opportunity

        edge = 1.0 - total_cost_per_set
        edge_bps = int(edge * 10000)

        if edge_bps < self.config.min_edge_bps:
            return None

        # Size based on available liquidity (take smaller of the two)
        # Use only 50% of available liquidity to reduce slippage
        max_shares_yes = buy_price.get("ask_liquidity", 0) * 0.5
        max_shares_no = no_price.get("ask_liquidity", 0) * 0.5
        max_shares = min(max_shares_yes, max_shares_no)

        # Also cap at reasonable position size ($100 worth)
        max_position_usd = 100.0
        max_shares_by_capital = max_position_usd / total_cost_per_set

        shares = min(max_shares, max_shares_by_capital)

        if shares < 1:  # Minimum 1 share
            return None

        legs = [
            ArbLeg(
                token_id=buy_price["token_id"],
                market_id=buy_market.condition_id,
                outcome=buy_price["outcome"],
                side="BUY",
                target_price=buy_price["ask"],
                target_shares=shares,
            ),
            ArbLeg(
                token_id=no_price["token_id"],
                market_id=sell_market.condition_id,
                outcome=no_price["outcome"],
                side="BUY",  # BUY NO, not SELL YES
                target_price=no_price["ask"],
                target_shares=shares,
            ),
        ]

        total_cost = shares * total_cost_per_set
        expected_profit = shares * edge  # At resolution, get $1 per share

        # Compute similarity for confidence
        similarity = self.correlation_engine.compute_semantic_similarity(
            buy_market.question,
            sell_market.question,
        )

        return StatArbOpportunity(
            opportunity_id=f"dup_{buy_market.condition_id[:8]}_{sell_market.condition_id[:8]}_{uuid.uuid4().hex[:4]}",
            arb_type=ArbType.DUPLICATE_MARKET,
            detected_at=datetime.now(timezone.utc),
            market_ids=[buy_market.condition_id, sell_market.condition_id],
            token_ids=[buy_price["token_id"], sell_price["token_id"]],
            questions=[buy_market.question, sell_market.question],
            edge_bps=edge_bps,
            z_score=0.0,  # N/A
            confidence=similarity,
            legs=legs,
            total_cost=total_cost,
            expected_profit=expected_profit,
            metadata={
                "buy_market_question": buy_market.question,
                "sell_market_question": sell_market.question,
                "similarity": similarity,
                "buy_ask": buy_price["ask"],
                "sell_bid": sell_price["bid"],
            },
        )
