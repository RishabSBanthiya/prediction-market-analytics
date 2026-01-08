"""
Base backtester class with common functionality.

Provides:
- Historical data fetching
- Market preparation
- Kelly criterion sizing
- Liquidity estimation
- Result tracking
"""

import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple

from ..core.models import Market, Token, HistoricalPrice, OrderbookSnapshot
from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from .results import BacktestResults, SimulatedTrade
from .execution import SimulatedExecution

logger = logging.getLogger(__name__)


class BaseBacktester(ABC):
    """
    Abstract base class for backtesting strategies.
    
    Subclasses implement the specific strategy logic in run_strategy().
    """
    
    def __init__(
        self,
        initial_capital: float = 1000.0,
        days: int = 3,
        config: Optional[Config] = None,
        verbose: bool = False,
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital in USD
            days: Number of days to backtest
            config: Configuration (uses default if not provided)
            verbose: Enable verbose logging
        """
        self.initial_capital = initial_capital
        self.days = days
        self.config = config or get_config()
        self.verbose = verbose
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Tuple[float, float, datetime]] = {}  # token_id -> (shares, price, time)
        
        # Execution simulator
        self.execution = SimulatedExecution()
        
        # API client
        self.api: Optional[PolymarketAPI] = None
        
        # Results
        self.results: Optional[BacktestResults] = None
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the strategy being backtested"""
        pass
    
    @abstractmethod
    async def run_strategy(self, markets: List[Market]) -> BacktestResults:
        """
        Run the backtest strategy.
        
        Args:
            markets: List of markets to backtest on
        
        Returns:
            BacktestResults with all trade details
        """
        pass
    
    async def run(self, pre_fetched_markets: Optional[List[Market]] = None) -> BacktestResults:
        """
        Main entry point for running backtest.
        
        Fetches data and runs the strategy.
        
        Args:
            pre_fetched_markets: Optional list of pre-fetched markets to use.
                                 If provided, skips market fetching.
        """
        logger.info(f"Starting backtest: {self.strategy_name}")
        logger.info(f"Capital: ${self.initial_capital:,.2f}, Days: {self.days}")
        
        # Initialize API
        self.api = PolymarketAPI(self.config)
        await self.api.connect()
        
        try:
            if pre_fetched_markets is not None:
                # Use pre-fetched markets
                markets = pre_fetched_markets
                logger.info(f"Using {len(markets)} pre-fetched markets")
            else:
                # Fetch closed markets
                logger.info("Fetching closed markets...")
                raw_markets = await self.api.fetch_closed_markets(days=self.days)
                logger.info(f"Found {len(raw_markets)} closed markets")
                
                # Prepare markets
                markets = []
                for raw in raw_markets:
                    market = await self.prepare_market(raw)
                    if market:
                        markets.append(market)
                
                logger.info(f"Prepared {len(markets)} markets for backtesting")
            
            # Run strategy
            self.results = await self.run_strategy(markets)
            self.results.markets_analyzed = len(markets)
            
            return self.results
            
        finally:
            await self.api.close()
    
    async def prepare_market(self, raw_market: dict) -> Optional[Market]:
        """
        Prepare a market for backtesting.
        
        Uses resolution data from API response to determine winner.
        This avoids look-ahead bias by not fetching price history to infer outcomes.
        
        Resolution is detected from outcomePrices (0/1 split) in parse_market(),
        which handles the case where the API's resolved flag lags behind.
        """
        market = self.api.parse_market(raw_market)
        if not market:
            logger.debug(f"Failed to parse market: {raw_market.get('question', 'unknown')[:50]}")
            return None
        
        # Log market status for debugging
        if self.verbose:
            outcome_prices = raw_market.get("outcomePrices", [])
            logger.info(
                f"Market: {market.question[:40]}... | "
                f"closed={market.closed}, resolved={market.resolved}, "
                f"prices={outcome_prices}, winner={market.winning_outcome}"
            )
        
        # For backtesting, we need a resolved market with a clear winner
        # parse_market() now detects resolution from prices even if resolved=False in API
        if not market.winning_outcome:
            logger.debug(f"Skipping market with no clear winner: {market.question[:50]}")
            return None
        
        return market
    
    async def fetch_price_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None
    ) -> List[HistoricalPrice]:
        """Fetch price history for a token"""
        if start_ts and end_ts:
            return await self.api.fetch_price_history_range(token_id, start_ts, end_ts)
        else:
            return await self.api.fetch_price_history(token_id)
    
    async def fetch_orderbook(self, token_id: str) -> Optional[OrderbookSnapshot]:
        """
        Fetch orderbook for a token.
        
        Note: For closed markets, orderbook will be empty.
        """
        return await self.api.fetch_orderbook(token_id)
    
    def calculate_kelly_fraction(
        self,
        price: float,
        kelly_multiplier: float = 1.0,
        max_fraction: float = 0.15
    ) -> float:
        """
        Calculate Kelly Criterion fraction.

        For expiring market strategy where we believe high prices
        have high probability of resolving to $1.

        Args:
            price: Current market price (0-1)
            kelly_multiplier: Fraction of Kelly to use (1.0 = full, 0.5 = half)
            max_fraction: Maximum position size as fraction of capital

        IMPORTANT NOTES:
        - Market price != true probability (information asymmetry exists)
        - Edge estimate is conservative to avoid overconfidence
        - Resolution risk exists even at 98% prices
        """
        if price <= 0 or price >= 1:
            return 0.0

        # Estimate edge based on price with conservative assumptions
        # Only apply edge factor for very high prices where resolution is likely
        min_price = 0.92  # More conservative threshold
        max_price = 0.99

        if price < min_price:
            # Below threshold, assume no systematic edge
            edge_factor = 0.0
        else:
            edge_factor = (price - min_price) / (max_price - min_price)
            edge_factor = max(0, min(1, edge_factor))

        # Estimated true probability with conservative edge
        # Use 0.3 multiplier instead of 0.5 to avoid overconfidence
        true_prob = price + (1 - price) * 0.3 * edge_factor

        p = true_prob
        q = 1 - p
        b = (1.0 / price) - 1  # Odds

        if b <= 0:
            return 0.0

        # Kelly formula: (p*b - q) / b
        kelly = (p * b - q) / b

        # Apply kelly multiplier (full kelly for aggressive, half for moderate)
        kelly = kelly * kelly_multiplier

        # Cap at max_fraction (15% for aggressive risk tolerance)
        return max(0.0, min(max_fraction, kelly))
    
    def calculate_position_size(
        self,
        price: float,
        available_cash: float,
        kelly_multiplier: float = 1.0,
        max_position_pct: float = 0.15
    ) -> Tuple[float, float]:
        """
        Calculate position size in dollars.

        Args:
            price: Current market price
            available_cash: Available capital
            kelly_multiplier: Kelly fraction (1.0 = full, 0.5 = half)
            max_position_pct: Max position as % of capital (15% for aggressive)

        Returns: (position_dollars, kelly_fraction)
        """
        kelly = self.calculate_kelly_fraction(price, kelly_multiplier, max_position_pct)

        if kelly <= 0:
            return 0.0, 0.0

        min_price = 0.90
        max_price = 0.98

        # Scale by price proximity to max
        price_scale = (price - min_price) / (max_price - min_price)
        price_scale = max(0, min(1, price_scale))

        # Less aggressive scaling (0.6 base + 0.4 scaled instead of 0.5/0.5)
        adjusted_fraction = kelly * (0.6 + 0.4 * price_scale)

        position_dollars = available_cash * adjusted_fraction

        if position_dollars < self.config.risk.min_trade_value_usd:
            return 0.0, 0.0

        return position_dollars, adjusted_fraction
    
    def estimate_liquidity(
        self,
        price_history: List[HistoricalPrice],
        target_price: float,
        max_slippage: float = 0.01
    ) -> float:
        """
        Estimate liquidity from price history using multiple heuristics.
        
        Uses:
        - Price volatility (lower volatility = higher liquidity)
        - Trade frequency (gaps in timestamps = low liquidity)
        - Multi-window analysis (1min, 5min, 15min stability)
        - Recent data weighted more heavily
        
        Returns:
            Estimated liquidity in USD
        """
        if len(price_history) < 10:
            return 100.0  # Base estimate for insufficient data
        
        # Get recent prices with timestamps
        recent = price_history[-60:]  # Last 60 data points
        prices = [p.price for p in recent]
        timestamps = [p.timestamp for p in recent]
        
        if not prices:
            return 100.0
        
        # 1. Volatility-based estimate (existing logic, improved)
        avg_price = sum(prices) / len(prices)
        volatility = sum(abs(p - avg_price) for p in prices) / len(prices)
        
        if volatility > 0:
            volatility_liquidity = 500.0 / (volatility * 50)  # Scaled estimate
        else:
            volatility_liquidity = 2000.0
        
        # 2. Trade frequency analysis (gaps indicate low liquidity)
        if len(timestamps) >= 2:
            gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_gap = sum(gaps) / len(gaps)
            
            # If average gap > 5 minutes (300s), reduce liquidity estimate
            # If average gap < 1 minute (60s), increase liquidity estimate
            if avg_gap > 300:
                frequency_factor = 0.3  # Low frequency = low liquidity
            elif avg_gap > 60:
                frequency_factor = 0.6
            elif avg_gap > 10:
                frequency_factor = 1.0
            else:
                frequency_factor = 1.5  # High frequency = high liquidity
        else:
            frequency_factor = 1.0
        
        # 3. Multi-window stability analysis
        # Check if price is stable across different time windows
        stability_scores = []
        
        for window_size in [5, 15, 30]:
            if len(prices) >= window_size:
                window_prices = prices[-window_size:]
                window_range = max(window_prices) - min(window_prices)
                # Smaller range = more stable = higher liquidity
                window_stability = 1.0 / (1.0 + window_range * 20)
                stability_scores.append(window_stability)
        
        if stability_scores:
            # Weight recent windows more heavily
            weights = [0.5, 0.3, 0.2][:len(stability_scores)]
            total_weight = sum(weights)
            stability_factor = sum(s * w for s, w in zip(stability_scores, weights)) / total_weight
        else:
            stability_factor = 0.5
        
        # 4. Recent data weighting
        # Use last 10 prices for recent volatility, weight it 2x
        if len(prices) >= 10:
            recent_prices = prices[-10:]
            recent_avg = sum(recent_prices) / len(recent_prices)
            recent_vol = sum(abs(p - recent_avg) for p in recent_prices) / len(recent_prices)
            
            if recent_vol > 0:
                recent_liquidity = 300.0 / (recent_vol * 30)
            else:
                recent_liquidity = 1500.0
        else:
            recent_liquidity = volatility_liquidity
        
        # Combine all factors
        base_estimate = (volatility_liquidity * 0.3 + recent_liquidity * 0.4) * frequency_factor
        final_estimate = base_estimate * (0.5 + stability_factor)
        
        return max(10.0, min(10000.0, final_estimate))
    
    def check_spread(
        self,
        orderbook: Optional[OrderbookSnapshot]
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Get bid, ask, and spread from orderbook.
        
        Returns: (best_bid, best_ask, spread_pct)
        """
        if not orderbook:
            return None, None, 0.0
        
        best_bid = orderbook.best_bid
        best_ask = orderbook.best_ask
        
        if not best_bid or not best_ask:
            return best_bid, best_ask, 0.0
        
        spread_pct = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.0
        
        return best_bid, best_ask, spread_pct
    
    def record_trade(
        self,
        results: BacktestResults,
        market: Market,
        token: Token,
        entry_time: datetime,
        entry_price: float,
        shares: float,
        cost: float,
        exit_time: Optional[datetime],
        exit_price: Optional[float],
        reason: str
    ):
        """Record a trade in results"""
        pnl = None
        pnl_pct = None
        proceeds = None
        resolved_to = None
        held_to_resolution = False
        
        if exit_price is not None and shares > 0:
            proceeds = shares * exit_price
            pnl = proceeds - cost
            pnl_pct = pnl / cost if cost > 0 else 0.0
            
            if exit_price > 0.9:
                resolved_to = 1.0
                held_to_resolution = True
            elif exit_price < 0.1:
                resolved_to = 0.0
                held_to_resolution = True
        
        trade = SimulatedTrade(
            market_question=market.question[:100],
            token_id=token.token_id,
            token_outcome=token.outcome,
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=exit_time,
            exit_price=exit_price,
            shares=shares,
            cost=cost,
            proceeds=proceeds,
            pnl=pnl,
            pnl_percent=pnl_pct,
            resolved_to=resolved_to,
            held_to_resolution=held_to_resolution,
            reason=reason
        )
        
        results.add_trade(trade)

