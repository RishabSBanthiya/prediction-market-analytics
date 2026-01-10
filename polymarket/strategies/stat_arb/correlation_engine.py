"""
Correlation Engine for Statistical Arbitrage.

Detects related markets using:
1. Historical price correlation (Pearson)
2. Semantic similarity (TF-IDF cosine)
3. Manual groupings (config-based)

Maintains cached correlation data with periodic updates.
"""

import asyncio
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Set
import math

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Market

from .models import MarketPair, MarketCluster, CorrelationType
from .config import CorrelationConfig

logger = logging.getLogger(__name__)

# Optional imports for NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Semantic similarity disabled.")

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Price correlation disabled.")


class CorrelationEngine:
    """
    Engine for computing and caching market correlations.

    Supports multiple correlation types:
    - Price: Historical price movement correlation
    - Semantic: Question text similarity
    - Manual: Explicit market groupings
    """

    def __init__(
        self,
        api: PolymarketAPI,
        config: CorrelationConfig,
    ):
        self.api = api
        self.config = config

        # Cached data
        self._pairs: Dict[str, MarketPair] = {}  # pair_key -> MarketPair
        self._clusters: Dict[str, MarketCluster] = {}  # cluster_id -> MarketCluster
        self._price_cache: Dict[str, List[Tuple[datetime, float]]] = {}  # token_id -> prices
        self._last_update: Optional[datetime] = None

        # TF-IDF vectorizer (lazy init)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._question_vectors = None
        self._question_index: Dict[str, int] = {}  # market_id -> vector index

    async def initialize(self, markets: List[Market]) -> None:
        """Initialize correlation data from market list."""
        logger.info(f"Initializing correlation engine with {len(markets)} markets")

        # Build semantic clusters
        if SKLEARN_AVAILABLE:
            await self._build_semantic_clusters(markets)

        # Build price correlations for active markets
        if SCIPY_AVAILABLE:
            await self._build_price_correlations(markets)

        self._last_update = datetime.now(timezone.utc)
        logger.info(
            f"Correlation engine initialized: {len(self._pairs)} pairs, "
            f"{len(self._clusters)} clusters"
        )

    async def update_if_stale(self, markets: List[Market]) -> None:
        """Update correlations if they're stale."""
        if self._last_update is None:
            await self.initialize(markets)
            return

        age_hours = (datetime.now(timezone.utc) - self._last_update).total_seconds() / 3600
        if age_hours >= self.config.update_interval_hours:
            logger.info("Correlation data stale, updating...")
            await self.initialize(markets)

    def get_correlated_pairs(
        self,
        min_correlation: Optional[float] = None,
    ) -> List[MarketPair]:
        """Get all correlated market pairs."""
        threshold = min_correlation or self.config.min_price_correlation
        return [
            pair for pair in self._pairs.values()
            if abs(pair.correlation) >= threshold and pair.is_valid
        ]

    def get_clusters(self) -> List[MarketCluster]:
        """Get all market clusters."""
        return list(self._clusters.values())

    def get_pair(self, market_a_id: str, market_b_id: str) -> Optional[MarketPair]:
        """Get a specific market pair."""
        key = self._pair_key(market_a_id, market_b_id)
        return self._pairs.get(key)

    async def compute_price_correlation(
        self,
        token_a_id: str,
        token_b_id: str,
        lookback_days: Optional[int] = None,
    ) -> Optional[float]:
        """
        Compute Pearson correlation between two token price series.

        Returns None if insufficient data or scipy not available.
        """
        if not SCIPY_AVAILABLE:
            return None

        lookback = lookback_days or self.config.lookback_days

        # Fetch price histories
        history_a = await self._get_price_history(token_a_id, lookback)
        history_b = await self._get_price_history(token_b_id, lookback)

        if not history_a or not history_b:
            return None

        # Align timestamps (within 60-second windows)
        aligned = self._align_price_series(history_a, history_b)

        if len(aligned) < self.config.min_data_points:
            return None

        # Extract aligned prices
        prices_a = [p[0] for p in aligned]
        prices_b = [p[1] for p in aligned]

        # Check for constant series (would cause division by zero)
        if len(set(prices_a)) < 2 or len(set(prices_b)) < 2:
            return None

        try:
            correlation, _ = pearsonr(prices_a, prices_b)
            return correlation if not math.isnan(correlation) else None
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return None

    def compute_semantic_similarity(
        self,
        question_a: str,
        question_b: str,
    ) -> float:
        """
        Compute TF-IDF cosine similarity between two questions.

        Returns 0.0 if sklearn not available.
        """
        if not SKLEARN_AVAILABLE:
            return 0.0

        # Normalize questions
        norm_a = self._normalize_question(question_a)
        norm_b = self._normalize_question(question_b)

        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform([norm_a, norm_b])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity failed: {e}")
            return 0.0

    def compute_spread_stats(
        self,
        prices_a: List[Tuple[datetime, float]],
        prices_b: List[Tuple[datetime, float]],
    ) -> Tuple[float, float, float]:
        """
        Compute spread statistics for pair trading.

        Returns: (mean, std, half_life_hours)
        """
        aligned = self._align_price_series(prices_a, prices_b)

        if len(aligned) < self.config.min_spread_observations:
            return 0.0, 0.0, 0.0

        spreads = [p[0] - p[1] for p in aligned]

        # Mean and std
        mean = sum(spreads) / len(spreads)
        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        std = math.sqrt(variance) if variance > 0 else 0.0

        # Half-life estimation (simplified AR(1))
        # In practice, you'd want a more sophisticated approach
        half_life = self._estimate_half_life(spreads)

        return mean, std, half_life

    def get_z_score(
        self,
        pair: MarketPair,
        current_spread: float,
    ) -> float:
        """Calculate z-score (deviation from mean in std devs)."""
        if pair.spread_std <= 0:
            return 0.0
        return (current_spread - pair.spread_mean) / pair.spread_std

    def cluster_markets_by_similarity(
        self,
        markets: List[Market],
        threshold: Optional[float] = None,
    ) -> List[MarketCluster]:
        """
        Cluster markets by semantic similarity.

        Uses single-linkage clustering with TF-IDF similarity.
        """
        if not SKLEARN_AVAILABLE or len(markets) < 2:
            return []

        threshold = threshold or self.config.min_semantic_similarity
        clusters: List[MarketCluster] = []

        # Build question vectors
        questions = [m.question for m in markets]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform([self._normalize_question(q) for q in questions])
            sim_matrix = cosine_similarity(tfidf)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            return []

        # Simple greedy clustering
        assigned: Set[int] = set()
        for i, market in enumerate(markets):
            if i in assigned:
                continue

            # Find all markets similar to this one
            cluster_indices = [i]
            for j in range(i + 1, len(markets)):
                if j not in assigned and sim_matrix[i, j] >= threshold:
                    cluster_indices.append(j)

            if len(cluster_indices) >= 2:
                cluster = MarketCluster(
                    cluster_id=f"cluster_{market.condition_id[:8]}",
                    name=f"Related to: {market.question[:50]}...",
                    market_ids=[markets[idx].condition_id for idx in cluster_indices],
                    questions=[markets[idx].question for idx in cluster_indices],
                    category=market.category or "OTHER",
                    similarity_threshold=threshold,
                )
                clusters.append(cluster)
                assigned.update(cluster_indices)

        return clusters

    async def _build_semantic_clusters(self, markets: List[Market]) -> None:
        """Build semantic clusters from markets."""
        # Group by category first
        by_category: Dict[str, List[Market]] = defaultdict(list)
        for market in markets:
            cat = market.category or "OTHER"
            by_category[cat].append(market)

        # Cluster within each category
        self._clusters.clear()
        for category, cat_markets in by_category.items():
            if len(cat_markets) < 2:
                continue

            clusters = self.cluster_markets_by_similarity(cat_markets)
            for cluster in clusters:
                self._clusters[cluster.cluster_id] = cluster

        logger.info(f"Built {len(self._clusters)} semantic clusters")

    async def _build_price_correlations(self, markets: List[Market]) -> None:
        """Build price correlations for tradeable pairs."""
        # Focus on markets with sufficient volume/activity
        active_markets = [m for m in markets if not m.closed and len(m.tokens) >= 2]

        if len(active_markets) < 2:
            return

        # Build pairs within same category
        by_category: Dict[str, List[Market]] = defaultdict(list)
        for market in active_markets:
            cat = market.category or "OTHER"
            by_category[cat].append(market)

        self._pairs.clear()
        pairs_checked = 0
        pairs_added = 0

        for category, cat_markets in by_category.items():
            # Limit pairs per category to avoid explosion
            for i, market_a in enumerate(cat_markets[:20]):  # Max 20 markets per category
                for market_b in cat_markets[i + 1:20]:
                    pairs_checked += 1

                    # Get YES tokens for both markets
                    token_a = self._get_yes_token(market_a)
                    token_b = self._get_yes_token(market_b)

                    if not token_a or not token_b:
                        continue

                    # Compute correlation
                    correlation = await self.compute_price_correlation(
                        token_a.token_id,
                        token_b.token_id,
                    )

                    if correlation is None:
                        continue

                    if abs(correlation) >= self.config.min_price_correlation:
                        # Compute spread stats
                        history_a = await self._get_price_history(
                            token_a.token_id,
                            self.config.spread_lookback_days,
                        )
                        history_b = await self._get_price_history(
                            token_b.token_id,
                            self.config.spread_lookback_days,
                        )

                        mean, std, half_life = self.compute_spread_stats(
                            history_a or [],
                            history_b or [],
                        )

                        pair = MarketPair(
                            market_a_id=market_a.condition_id,
                            market_b_id=market_b.condition_id,
                            market_a_question=market_a.question,
                            market_b_question=market_b.question,
                            token_a_id=token_a.token_id,
                            token_b_id=token_b.token_id,
                            correlation=correlation,
                            correlation_type=CorrelationType.PRICE,
                            lookback_days=self.config.lookback_days,
                            last_updated=datetime.now(timezone.utc),
                            spread_mean=mean,
                            spread_std=std,
                            half_life_hours=half_life,
                        )

                        key = self._pair_key(market_a.condition_id, market_b.condition_id)
                        self._pairs[key] = pair
                        pairs_added += 1

        logger.info(
            f"Price correlations: checked {pairs_checked} pairs, "
            f"found {pairs_added} correlated pairs"
        )

    async def _get_price_history(
        self,
        token_id: str,
        lookback_days: int,
    ) -> Optional[List[Tuple[datetime, float]]]:
        """Fetch price history with caching."""
        if token_id in self._price_cache:
            return self._price_cache[token_id]

        try:
            history = await self.api.fetch_price_history(token_id)
            if history:
                # Filter to lookback period
                cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
                filtered = []
                for item in history:
                    # Handle HistoricalPrice objects or raw dicts
                    if hasattr(item, 'datetime'):
                        ts = item.datetime
                        price = item.price
                    elif hasattr(item, 'timestamp'):
                        ts = datetime.fromtimestamp(item.timestamp, tz=timezone.utc)
                        price = item.price
                    elif isinstance(item, dict):
                        ts_val = item.get('t') or item.get('timestamp', 0)
                        ts = datetime.fromtimestamp(ts_val, tz=timezone.utc) if ts_val else None
                        price = item.get('p') or item.get('price', 0)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        ts, price = item[0], item[1]
                        if isinstance(ts, (int, float)):
                            ts = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        continue

                    if ts and ts >= cutoff:
                        filtered.append((ts, price))

                self._price_cache[token_id] = filtered
                return filtered
        except Exception as e:
            logger.warning(f"Failed to fetch price history for {token_id}: {e}")

        return None

    def _align_price_series(
        self,
        series_a: List[Tuple[datetime, float]],
        series_b: List[Tuple[datetime, float]],
        window_seconds: int = 60,
    ) -> List[Tuple[float, float]]:
        """Align two price series by timestamp."""
        if not series_a or not series_b:
            return []

        # Sort both by timestamp
        sorted_a = sorted(series_a, key=lambda x: x[0])
        sorted_b = sorted(series_b, key=lambda x: x[0])

        aligned = []
        j = 0

        for ts_a, price_a in sorted_a:
            # Find closest timestamp in B within window
            while j < len(sorted_b) - 1:
                diff_current = abs((sorted_b[j][0] - ts_a).total_seconds())
                diff_next = abs((sorted_b[j + 1][0] - ts_a).total_seconds())
                if diff_next < diff_current:
                    j += 1
                else:
                    break

            if j < len(sorted_b):
                diff = abs((sorted_b[j][0] - ts_a).total_seconds())
                if diff <= window_seconds:
                    aligned.append((price_a, sorted_b[j][1]))

        return aligned

    def _normalize_question(self, question: str) -> str:
        """Normalize question for comparison."""
        # Lowercase
        normalized = question.lower()

        # Remove dates (various formats)
        normalized = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', normalized)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '', normalized)
        normalized = re.sub(
            r'(january|february|march|april|may|june|july|august|'
            r'september|october|november|december)\s+\d{1,2}(,?\s*\d{4})?',
            '',
            normalized,
        )

        # Remove specific times
        normalized = re.sub(r'\d{1,2}:\d{2}\s*(am|pm|AM|PM)?', '', normalized)

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def _estimate_half_life(self, spreads: List[float]) -> float:
        """Estimate mean reversion half-life from spread series."""
        if len(spreads) < 10:
            return 24.0  # Default 24 hours

        # Simple AR(1) coefficient estimation
        # half_life = -log(2) / log(phi)
        # where phi is AR(1) coefficient

        mean = sum(spreads) / len(spreads)
        deviations = [s - mean for s in spreads]

        if len(deviations) < 2:
            return 24.0

        # Estimate AR(1) coefficient
        numerator = sum(
            deviations[i] * deviations[i - 1]
            for i in range(1, len(deviations))
        )
        denominator = sum(d ** 2 for d in deviations[:-1])

        if denominator == 0:
            return 24.0

        phi = numerator / denominator

        # Clamp phi to valid range
        phi = max(min(phi, 0.99), 0.01)

        # Convert to half-life in hours
        # Assuming ~hourly observations
        half_life_periods = -math.log(2) / math.log(phi) if phi < 1 else 24.0

        return max(1.0, min(168.0, half_life_periods))  # 1 hour to 1 week

    def _pair_key(self, market_a_id: str, market_b_id: str) -> str:
        """Generate consistent key for market pair."""
        return f"{min(market_a_id, market_b_id)}_{max(market_a_id, market_b_id)}"

    def _get_yes_token(self, market: Market):
        """Get the YES/primary token for a market."""
        if not market.tokens:
            return None

        for token in market.tokens:
            if token.outcome.lower() in ("yes", "up", "over"):
                return token

        # Default to first token
        return market.tokens[0]
