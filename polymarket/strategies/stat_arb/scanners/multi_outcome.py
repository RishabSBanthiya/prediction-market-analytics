"""
Multi-Outcome Sum Arbitrage Scanner.

Detects markets where the sum of outcome prices != 100%.

If sum < 100%: Buy all outcomes, guaranteed profit
If sum > 100%: Sell all outcomes (if holding), guaranteed profit
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional
import uuid

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market

from ..models import (
    ArbType,
    StatArbOpportunity,
    ArbLeg,
    MultiOutcomeMarket,
)
from ..config import MultiOutcomeConfig

logger = logging.getLogger(__name__)


class MultiOutcomeScanner:
    """
    Scanner for multi-outcome sum arbitrage.

    Looks for markets with 3+ outcomes where:
    - Sum of ask prices < 1.0 (buy all for guaranteed profit)
    - Sum of bid prices > 1.0 (sell all for guaranteed profit)
    """

    # Limit concurrent API calls to avoid overwhelming the rate limiter
    MAX_CONCURRENT = 10

    def __init__(
        self,
        api: PolymarketAPI,
        config: MultiOutcomeConfig,
    ):
        self.api = api
        self.config = config
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    async def scan(self, markets: List[Market]) -> List[StatArbOpportunity]:
        """
        Scan markets for multi-outcome arbitrage opportunities.

        Returns list of opportunities sorted by edge (highest first).
        """
        opportunities: List[StatArbOpportunity] = []

        # Filter to multi-outcome markets
        multi_markets = [
            m for m in markets
            if not m.closed
            and len(m.tokens) >= self.config.min_outcomes
            and len(m.tokens) <= self.config.max_outcomes
        ]

        logger.info(f"Scanning {len(multi_markets)} multi-outcome markets (max {self.MAX_CONCURRENT} concurrent)")

        # Check each market with concurrency limit
        async def check_with_semaphore(market):
            async with self._semaphore:
                return await self._check_market(market)

        tasks = [check_with_semaphore(m) for m in multi_markets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Market check failed: {result}")
                continue
            if result:
                opportunities.append(result)

        # Sort by edge (highest first)
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)

        if opportunities:
            logger.info(
                f"Found {len(opportunities)} multi-outcome opportunities "
                f"(best edge: {opportunities[0].edge_bps} bps)"
            )

        return opportunities

    async def _check_market(self, market: Market) -> Optional[StatArbOpportunity]:
        """Check a single market for sum arbitrage."""
        # Fetch orderbooks for all outcomes
        outcomes = []
        total_ask = 0.0
        total_bid = 0.0

        for token in market.tokens:
            try:
                orderbook = await self.api.fetch_orderbook(token.token_id)
            except Exception as e:
                logger.debug(f"Failed to fetch orderbook for {token.token_id}: {e}")
                return None

            if not orderbook or not orderbook.best_ask or not orderbook.best_bid:
                return None

            # Check spread
            spread_bps = int((orderbook.best_ask - orderbook.best_bid) * 10000)
            if spread_bps > self.config.max_spread_bps:
                logger.debug(
                    f"Spread too wide for {token.outcome}: {spread_bps} bps"
                )
                return None

            # Check liquidity
            ask_depth = sum(size for _, size in orderbook.ask_depth[:3])
            ask_liquidity = ask_depth * orderbook.best_ask
            if ask_liquidity < self.config.min_liquidity_usd:
                logger.debug(
                    f"Insufficient liquidity for {token.outcome}: ${ask_liquidity:.2f}"
                )
                return None

            outcomes.append({
                "token_id": token.token_id,
                "outcome": token.outcome,
                "ask_price": orderbook.best_ask,
                "bid_price": orderbook.best_bid,
                "ask_depth": ask_depth,
            })
            total_ask += orderbook.best_ask
            total_bid += orderbook.best_bid

        # Check for buy arbitrage (sum of asks < 1.0)
        if total_ask < 1.0:
            edge_bps = int((1.0 - total_ask) * 10000)
            if edge_bps >= self.config.min_edge_bps:
                return self._create_buy_opportunity(market, outcomes, total_ask, edge_bps)

        # Check for sell arbitrage (sum of bids > 1.0)
        # Note: This requires already holding shares in all outcomes
        if total_bid > 1.0:
            edge_bps = int((total_bid - 1.0) * 10000)
            if edge_bps >= self.config.min_edge_bps:
                return self._create_sell_opportunity(market, outcomes, total_bid, edge_bps)

        return None

    def _create_buy_opportunity(
        self,
        market: Market,
        outcomes: List[dict],
        total_ask: float,
        edge_bps: int,
    ) -> StatArbOpportunity:
        """Create a buy-all opportunity."""
        # Calculate position size per outcome
        # Equal dollar amount per outcome
        per_outcome_usd = self.config.max_position_pct * 1000 / len(outcomes)  # Placeholder

        legs = []
        total_cost = 0.0

        for outcome in outcomes:
            shares = per_outcome_usd / outcome["ask_price"]
            leg = ArbLeg(
                token_id=outcome["token_id"],
                market_id=market.condition_id,
                outcome=outcome["outcome"],
                side="BUY",
                target_price=outcome["ask_price"],
                target_shares=shares,
            )
            legs.append(leg)
            total_cost += shares * outcome["ask_price"]

        # At resolution, exactly one outcome pays $1
        # We own all outcomes, so we get $1 per "set"
        min_shares = min(leg.target_shares for leg in legs)
        expected_payout = min_shares * 1.0
        expected_profit = expected_payout - (min_shares * total_ask)

        return StatArbOpportunity(
            opportunity_id=f"multi_buy_{market.condition_id}_{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.MULTI_OUTCOME_SUM,
            detected_at=datetime.now(timezone.utc),
            market_ids=[market.condition_id],
            token_ids=[o["token_id"] for o in outcomes],
            questions=[market.question],
            edge_bps=edge_bps,
            z_score=0.0,  # N/A for this type
            confidence=min(edge_bps / 100, 1.0),
            legs=legs,
            total_cost=total_cost,
            expected_profit=expected_profit,
            metadata={
                "arb_direction": "buy_all",
                "total_ask": total_ask,
                "outcome_count": len(outcomes),
            },
        )

    def _create_sell_opportunity(
        self,
        market: Market,
        outcomes: List[dict],
        total_bid: float,
        edge_bps: int,
    ) -> StatArbOpportunity:
        """Create a sell-all opportunity (requires holding shares)."""
        # This is less common - requires already having shares
        # Usually from a previous buy or market making

        legs = []
        for outcome in outcomes:
            leg = ArbLeg(
                token_id=outcome["token_id"],
                market_id=market.condition_id,
                outcome=outcome["outcome"],
                side="SELL",
                target_price=outcome["bid_price"],
                target_shares=0.0,  # Depends on current holdings
            )
            legs.append(leg)

        return StatArbOpportunity(
            opportunity_id=f"multi_sell_{market.condition_id}_{uuid.uuid4().hex[:8]}",
            arb_type=ArbType.MULTI_OUTCOME_SUM,
            detected_at=datetime.now(timezone.utc),
            market_ids=[market.condition_id],
            token_ids=[o["token_id"] for o in outcomes],
            questions=[market.question],
            edge_bps=edge_bps,
            z_score=0.0,
            confidence=min(edge_bps / 100, 1.0),
            legs=legs,
            total_cost=0.0,  # Would need to check holdings
            expected_profit=0.0,  # Depends on holdings
            metadata={
                "arb_direction": "sell_all",
                "total_bid": total_bid,
                "outcome_count": len(outcomes),
                "requires_holdings": True,
            },
        )

    def analyze_market(self, market: Market, outcomes: List[dict]) -> MultiOutcomeMarket:
        """Analyze a multi-outcome market for display/logging."""
        return MultiOutcomeMarket(
            condition_id=market.condition_id,
            question=market.question,
            outcomes=outcomes,
            total_ask=sum(o["ask_price"] for o in outcomes),
            total_bid=sum(o["bid_price"] for o in outcomes),
        )
