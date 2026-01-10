"""
Conditional Probability Arbitrage Scanner.

Detects mispricings in conditional/hierarchical market relationships.

Examples:
- P(Win General) should be <= P(Win Primary) since winning general requires primary
- P(A and B) should equal P(A) * P(B|A)
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Tuple
import uuid

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market

from ..models import (
    ArbType,
    StatArbOpportunity,
    ArbLeg,
    MarketCluster,
)
from ..config import ConditionalConfig
from ..correlation_engine import CorrelationEngine

logger = logging.getLogger(__name__)


# Patterns for detecting conditional relationships
CONDITIONAL_PATTERNS = [
    # Political hierarchy
    (r"win.*primary", r"win.*general", "B_requires_A"),
    (r"win.*nomination", r"win.*election", "B_requires_A"),
    (r"win.*state.*primary", r"win.*national.*primary", "B_requires_A"),

    # Sports hierarchy
    (r"make.*playoffs", r"win.*championship", "B_requires_A"),
    (r"win.*division", r"win.*conference", "B_requires_A"),
    (r"win.*conference", r"win.*championship", "B_requires_A"),

    # General patterns
    (r"advance.*round.*(\d+)", r"advance.*round.*(\d+)", "ROUND_PROGRESSION"),
    (r"qualify", r"win", "B_requires_A"),
]


class ConditionalProbScanner:
    """
    Scanner for conditional probability arbitrage.

    Finds markets where:
    - Market B logically requires Market A (B implies A)
    - P(B) > P(A) - which is a logical impossibility
    - Or P(A and B) != P(A) * P(B|A)
    """

    # Limit concurrent API calls to avoid overwhelming the rate limiter
    MAX_CONCURRENT = 10
    # Limit pairs to check per category to avoid long blocking
    MAX_PAIRS_PER_CATEGORY = 100
    # Limit markets per category to avoid O(n^2) explosion
    MAX_MARKETS_PER_CATEGORY = 200

    def __init__(
        self,
        api: PolymarketAPI,
        config: ConditionalConfig,
        correlation_engine: Optional[CorrelationEngine] = None,
    ):
        self.api = api
        self.config = config
        self.correlation_engine = correlation_engine
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    async def scan(
        self,
        markets: List[Market],
        clusters: Optional[List[MarketCluster]] = None,
    ) -> List[StatArbOpportunity]:
        """
        Scan for conditional probability mispricings.

        Args:
            markets: All markets to scan
            clusters: Optional pre-computed market clusters

        Returns:
            List of opportunities sorted by edge.
        """
        opportunities: List[StatArbOpportunity] = []

        # Filter active markets
        active_markets = [m for m in markets if not m.closed and m.tokens]

        if len(active_markets) < 2:
            return []

        # Group by category if required
        if self.config.require_same_category:
            by_category = self._group_by_category(active_markets)
        else:
            by_category = {"ALL": active_markets}

        total_categories = len(by_category)
        logger.info(f"Scanning {len(active_markets)} markets across {total_categories} categories for conditional relationships")

        # Check each category group
        for idx, (category, cat_markets) in enumerate(by_category.items()):
            if len(cat_markets) < 2:
                continue
            cat_opps = await self._scan_category(cat_markets)
            opportunities.extend(cat_opps)
            if (idx + 1) % 10 == 0:
                logger.debug(f"Conditional scanner: processed {idx + 1}/{total_categories} categories")

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)

        if opportunities:
            logger.info(
                f"Found {len(opportunities)} conditional probability opportunities "
                f"(best edge: {opportunities[0].edge_bps} bps)"
            )

        return opportunities

    async def _scan_category(self, markets: List[Market]) -> List[StatArbOpportunity]:
        """Scan markets within a single category."""
        # Limit markets to avoid O(n^2) explosion
        if len(markets) > self.MAX_MARKETS_PER_CATEGORY:
            logger.debug(f"Limiting conditional scan from {len(markets)} to {self.MAX_MARKETS_PER_CATEGORY} markets")
            markets = markets[:self.MAX_MARKETS_PER_CATEGORY]

        # First pass: find pairs with conditional relationships (no API calls)
        pairs_with_rel = []
        for i, market_a in enumerate(markets):
            for market_b in markets[i + 1:]:
                rel = self._detect_relationship(market_a.question, market_b.question)
                if rel:
                    pairs_with_rel.append((market_a, market_b, rel))

        if not pairs_with_rel:
            return []

        # Limit pairs to avoid long blocking scans
        if len(pairs_with_rel) > self.MAX_PAIRS_PER_CATEGORY:
            logger.debug(f"Limiting conditional scan from {len(pairs_with_rel)} to {self.MAX_PAIRS_PER_CATEGORY} pairs")
            pairs_with_rel = pairs_with_rel[:self.MAX_PAIRS_PER_CATEGORY]

        logger.info(f"Checking {len(pairs_with_rel)} conditional pairs (max {self.MAX_CONCURRENT} concurrent)")

        # Second pass: check pairs with API calls (with concurrency limit)
        async def check_with_semaphore(pair_data):
            async with self._semaphore:
                market_a, market_b, rel = pair_data
                return await self._check_pair(market_a, market_b, rel)

        tasks = [check_with_semaphore(p) for p in pairs_with_rel]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        opportunities = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Conditional pair check failed: {result}")
                continue
            if result:
                opportunities.append(result)

        return opportunities

    def _detect_relationship(
        self,
        question_a: str,
        question_b: str,
    ) -> Optional[Tuple[str, str]]:
        """
        Detect if there's a conditional relationship between questions.

        Returns: (relationship_type, direction) or None
        - direction: "A_requires_B" or "B_requires_A"
        """
        q_a_lower = question_a.lower()
        q_b_lower = question_b.lower()

        for pattern_a, pattern_b, rel_type in CONDITIONAL_PATTERNS:
            # Check A -> B relationship
            if re.search(pattern_a, q_a_lower) and re.search(pattern_b, q_b_lower):
                # Verify they're about same entity
                if self._same_entity(question_a, question_b):
                    return (rel_type, "B_requires_A")

            # Check B -> A relationship
            if re.search(pattern_b, q_a_lower) and re.search(pattern_a, q_b_lower):
                if self._same_entity(question_a, question_b):
                    return (rel_type, "A_requires_B")

            # Check round progression
            if rel_type == "ROUND_PROGRESSION":
                round_a = self._extract_round(q_a_lower)
                round_b = self._extract_round(q_b_lower)
                if round_a is not None and round_b is not None:
                    if self._same_entity(question_a, question_b):
                        if round_a < round_b:
                            return ("ROUND_PROGRESSION", "B_requires_A")
                        elif round_b < round_a:
                            return ("ROUND_PROGRESSION", "A_requires_B")

        return None

    def _same_entity(self, question_a: str, question_b: str) -> bool:
        """Check if questions are about the same entity (person, team, etc.)."""
        # Extract proper nouns (simplified)
        # In production, you'd want NER here

        # Simple heuristic: check for common capitalized words
        words_a = set(re.findall(r'\b[A-Z][a-z]+\b', question_a))
        words_b = set(re.findall(r'\b[A-Z][a-z]+\b', question_b))

        # Need at least one common proper noun
        common = words_a & words_b

        # Filter out common words that aren't entities
        common = {w for w in common if w.lower() not in {
            "will", "the", "who", "what", "when", "where", "how",
            "yes", "no", "win", "primary", "general", "election",
        }}

        return len(common) >= 1

    def _extract_round(self, question: str) -> Optional[int]:
        """Extract round number from question."""
        match = re.search(r'round\s*(\d+)', question)
        if match:
            return int(match.group(1))
        return None

    async def _check_pair(
        self,
        market_a: Market,
        market_b: Market,
        relationship: Tuple[str, str],
    ) -> Optional[StatArbOpportunity]:
        """Check a pair with known relationship for mispricing."""
        rel_type, direction = relationship

        # Get prices
        price_a = await self._get_yes_price(market_a)
        price_b = await self._get_yes_price(market_b)

        if price_a is None or price_b is None:
            return None

        # Determine which is the "required" (prerequisite) market
        if direction == "B_requires_A":
            prereq_market = market_a
            dependent_market = market_b
            prereq_price = price_a
            dependent_price = price_b
        else:
            prereq_market = market_b
            dependent_market = market_a
            prereq_price = price_b
            dependent_price = price_a

        # Check for violation: P(dependent) should be <= P(prereq)
        if dependent_price["mid"] <= prereq_price["mid"]:
            return None  # No violation

        # Calculate edge
        edge = dependent_price["mid"] - prereq_price["mid"]
        edge_bps = int(edge * 10000)

        if edge_bps < self.config.min_edge_bps:
            return None

        # Create opportunity: buy prereq, sell dependent
        return self._create_opportunity(
            prereq_market=prereq_market,
            dependent_market=dependent_market,
            prereq_price=prereq_price,
            dependent_price=dependent_price,
            relationship=relationship,
            edge_bps=edge_bps,
        )

    async def _get_yes_price(self, market: Market) -> Optional[Dict[str, float]]:
        """Get YES token price for a market."""
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

            return {
                "token_id": yes_token.token_id,
                "outcome": yes_token.outcome,
                "bid": orderbook.best_bid,
                "ask": orderbook.best_ask,
                "mid": (orderbook.best_bid + orderbook.best_ask) / 2,
            }
        except Exception as e:
            logger.debug(f"Failed to get price for {market.condition_id}: {e}")
            return None

    def _create_opportunity(
        self,
        prereq_market: Market,
        dependent_market: Market,
        prereq_price: Dict[str, float],
        dependent_price: Dict[str, float],
        relationship: Tuple[str, str],
        edge_bps: int,
    ) -> StatArbOpportunity:
        """Create a conditional probability arbitrage opportunity."""
        # Position: Buy the prerequisite (underpriced), sell the dependent (overpriced)
        position_usd = 100.0  # Placeholder

        buy_shares = position_usd / prereq_price["ask"]
        sell_shares = position_usd / dependent_price["bid"]

        shares = min(buy_shares, sell_shares)

        legs = [
            ArbLeg(
                token_id=prereq_price["token_id"],
                market_id=prereq_market.condition_id,
                outcome=prereq_price["outcome"],
                side="BUY",
                target_price=prereq_price["ask"],
                target_shares=shares,
            ),
            ArbLeg(
                token_id=dependent_price["token_id"],
                market_id=dependent_market.condition_id,
                outcome=dependent_price["outcome"],
                side="SELL",
                target_price=dependent_price["bid"],
                target_shares=shares,
            ),
        ]

        # Expected profit comes from the logical constraint
        # At resolution:
        # - If prereq = No: dependent = No (we win on both)
        # - If prereq = Yes, dependent = No: we lose on prereq, win on dependent
        # - If prereq = Yes, dependent = Yes: we lose on both
        # Net expectation is positive when P(dependent) > P(prereq)

        total_cost = shares * prereq_price["ask"]
        expected_profit = shares * (dependent_price["bid"] - prereq_price["ask"])

        rel_type, direction = relationship

        return StatArbOpportunity(
            opportunity_id=f"cond_{prereq_market.condition_id[:8]}_{dependent_market.condition_id[:8]}_{uuid.uuid4().hex[:4]}",
            arb_type=ArbType.CONDITIONAL_PROB,
            detected_at=datetime.now(timezone.utc),
            market_ids=[prereq_market.condition_id, dependent_market.condition_id],
            token_ids=[prereq_price["token_id"], dependent_price["token_id"]],
            questions=[prereq_market.question, dependent_market.question],
            edge_bps=edge_bps,
            z_score=0.0,  # N/A
            confidence=self.config.min_confidence,
            legs=legs,
            total_cost=total_cost,
            expected_profit=expected_profit,
            metadata={
                "relationship_type": rel_type,
                "direction": direction,
                "prereq_question": prereq_market.question,
                "dependent_question": dependent_market.question,
                "prereq_price": prereq_price["mid"],
                "dependent_price": dependent_price["mid"],
                "violation": f"P({dependent_market.question[:30]}...) > P({prereq_market.question[:30]}...)",
            },
        )

    def _group_by_category(self, markets: List[Market]) -> Dict[str, List[Market]]:
        """Group markets by category."""
        groups: Dict[str, List[Market]] = {}
        for market in markets:
            cat = market.category or "OTHER"
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(market)
        return groups
