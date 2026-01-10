"""
Pair Spread (Mean Reversion) Scanner.

Detects correlated market pairs where the spread has deviated
significantly from its historical mean.

Strategy: Trade the spread expecting mean reversion.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict
import uuid

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market

from ..models import (
    ArbType,
    StatArbOpportunity,
    ArbLeg,
    MarketPair,
)
from ..config import PairTradingConfig
from ..correlation_engine import CorrelationEngine

logger = logging.getLogger(__name__)


class PairSpreadScanner:
    """
    Scanner for pair spread (statistical arbitrage) opportunities.

    Finds correlated pairs where:
    - Historical correlation is high
    - Current spread deviates significantly from mean (high z-score)
    - Expected mean reversion creates profit opportunity
    """

    # Limit concurrent API calls to avoid overwhelming the rate limiter
    MAX_CONCURRENT = 10

    def __init__(
        self,
        api: PolymarketAPI,
        config: PairTradingConfig,
        correlation_engine: CorrelationEngine,
    ):
        self.api = api
        self.config = config
        self.correlation_engine = correlation_engine
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)

    async def scan(
        self,
        pairs: Optional[List[MarketPair]] = None,
    ) -> List[StatArbOpportunity]:
        """
        Scan correlated pairs for spread trading opportunities.

        Args:
            pairs: Optional list of pairs to scan. If None, uses all
                   pairs from correlation engine.

        Returns:
            List of opportunities sorted by z-score magnitude.
        """
        opportunities: List[StatArbOpportunity] = []

        # Get pairs to scan
        if pairs is None:
            pairs = self.correlation_engine.get_correlated_pairs(
                min_correlation=self.config.min_correlation
            )

        if not pairs:
            logger.debug("No correlated pairs to scan")
            return []

        logger.info(f"Scanning {len(pairs)} correlated pairs (max {self.MAX_CONCURRENT} concurrent)")

        # Check each pair with concurrency limit
        async def check_with_semaphore(pair):
            async with self._semaphore:
                return await self._check_pair(pair)

        tasks = [check_with_semaphore(pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Pair check failed: {result}")
                continue
            if result:
                opportunities.append(result)

        # Sort by z-score magnitude (highest deviation first)
        opportunities.sort(key=lambda x: abs(x.z_score), reverse=True)

        if opportunities:
            logger.info(
                f"Found {len(opportunities)} pair spread opportunities "
                f"(best z-score: {opportunities[0].z_score:.2f})"
            )

        return opportunities

    async def _check_pair(self, pair: MarketPair) -> Optional[StatArbOpportunity]:
        """Check a single pair for spread trading opportunity."""
        # Validate pair data
        if not pair.is_valid:
            logger.debug(f"Pair data stale: {pair.market_a_id[:8]}_{pair.market_b_id[:8]}")
            return None

        if pair.spread_std <= 0:
            logger.debug(f"Invalid spread std for pair: {pair.spread_std}")
            return None

        # Get current prices
        price_a = await self._get_price(pair.token_a_id)
        price_b = await self._get_price(pair.token_b_id)

        if price_a is None or price_b is None:
            return None

        # Calculate current spread and z-score
        current_spread = price_a["mid"] - price_b["mid"]
        z_score = self.correlation_engine.get_z_score(pair, current_spread)

        # Check if spread is extended enough
        if abs(z_score) < self.config.entry_z_score:
            return None

        # Calculate expected edge (spread to mean)
        expected_move = abs(current_spread - pair.spread_mean)
        edge_bps = int(expected_move * 10000)

        if edge_bps < self.config.min_edge_bps:
            return None

        # Determine trade direction
        if z_score > 0:
            # Spread is wide (A expensive relative to B)
            # Sell A, buy B
            return self._create_opportunity(
                pair=pair,
                sell_token=pair.token_a_id,
                buy_token=pair.token_b_id,
                sell_price=price_a,
                buy_price=price_b,
                z_score=z_score,
                current_spread=current_spread,
            )
        else:
            # Spread is narrow (B expensive relative to A)
            # Buy A, sell B
            return self._create_opportunity(
                pair=pair,
                sell_token=pair.token_b_id,
                buy_token=pair.token_a_id,
                sell_price=price_b,
                buy_price=price_a,
                z_score=z_score,
                current_spread=current_spread,
            )

    async def _get_price(self, token_id: str) -> Optional[Dict[str, float]]:
        """Get current price data for a token."""
        try:
            orderbook = await self.api.fetch_orderbook(token_id)
            if not orderbook or not orderbook.best_ask or not orderbook.best_bid:
                return None

            return {
                "token_id": token_id,
                "bid": orderbook.best_bid,
                "ask": orderbook.best_ask,
                "mid": (orderbook.best_bid + orderbook.best_ask) / 2,
                "spread": orderbook.best_ask - orderbook.best_bid,
            }
        except Exception as e:
            logger.debug(f"Failed to get price for {token_id}: {e}")
            return None

    def _create_opportunity(
        self,
        pair: MarketPair,
        sell_token: str,
        buy_token: str,
        sell_price: Dict[str, float],
        buy_price: Dict[str, float],
        z_score: float,
        current_spread: float,
    ) -> StatArbOpportunity:
        """Create a pair spread opportunity."""
        # Position sizing: equal dollar amounts on each leg
        position_usd = 100.0  # Placeholder, will be sized by position manager

        buy_shares = position_usd / buy_price["ask"]
        sell_shares = position_usd / sell_price["bid"]

        # Use minimum to stay balanced
        shares = min(buy_shares, sell_shares)

        # Determine which market is which
        is_buy_a = buy_token == pair.token_a_id
        buy_market_id = pair.market_a_id if is_buy_a else pair.market_b_id
        sell_market_id = pair.market_b_id if is_buy_a else pair.market_a_id

        legs = [
            ArbLeg(
                token_id=buy_token,
                market_id=buy_market_id,
                outcome="YES",  # Assuming YES tokens
                side="BUY",
                target_price=buy_price["ask"],
                target_shares=shares,
            ),
            ArbLeg(
                token_id=sell_token,
                market_id=sell_market_id,
                outcome="YES",
                side="SELL",
                target_price=sell_price["bid"],
                target_shares=shares,
            ),
        ]

        # Expected profit when spread returns to mean
        expected_profit = abs(current_spread - pair.spread_mean) * shares

        # Calculate targets for exit
        target_spread = pair.spread_mean + (
            self.config.exit_z_score * pair.spread_std * (1 if z_score > 0 else -1)
        )
        stop_spread = current_spread + (
            self.config.stop_z_score * pair.spread_std * (1 if z_score > 0 else -1)
        )

        edge_bps = int(abs(current_spread - pair.spread_mean) * 10000)

        return StatArbOpportunity(
            opportunity_id=f"pair_{pair.market_a_id[:8]}_{pair.market_b_id[:8]}_{uuid.uuid4().hex[:4]}",
            arb_type=ArbType.PAIR_SPREAD,
            detected_at=datetime.now(timezone.utc),
            market_ids=[pair.market_a_id, pair.market_b_id],
            token_ids=[pair.token_a_id, pair.token_b_id],
            questions=[pair.market_a_question, pair.market_b_question],
            edge_bps=edge_bps,
            z_score=z_score,
            confidence=abs(pair.correlation),
            legs=legs,
            total_cost=shares * buy_price["ask"],
            expected_profit=expected_profit,
            stop_loss_pct=(self.config.stop_z_score - abs(z_score)) / abs(z_score),
            take_profit_pct=(abs(z_score) - self.config.exit_z_score) / abs(z_score),
            metadata={
                "correlation": pair.correlation,
                "spread_mean": pair.spread_mean,
                "spread_std": pair.spread_std,
                "current_spread": current_spread,
                "target_spread": target_spread,
                "stop_spread": stop_spread,
                "half_life_hours": pair.half_life_hours,
                "buy_token": buy_token,
                "sell_token": sell_token,
            },
        )
