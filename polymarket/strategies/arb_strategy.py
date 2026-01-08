"""
Delta-Neutral Arbitrage Strategy for Polymarket 15-minute crypto markets.

This strategy exploits pricing inefficiencies in binary markets where:
- UP price + DOWN price < 1.0 (after fees)
- Buying both sides guarantees profit at resolution

Key insight: Without latency advantage, we use PATIENT limit orders:
1. Place bids on BOTH outcomes at prices that guarantee profit IF both fill
2. Wait for fills (could take minutes)
3. Only lock in profit when BOTH sides fill
4. Handle partial fills gracefully

Delta neutral = no directional exposure once both sides fill.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum

from polymarket.core.models import Market, Token, Side, OrderbookSnapshot
from polymarket.core.api import PolymarketAPI
from polymarket.core.config import Config

logger = logging.getLogger(__name__)


class ArbPositionStatus(Enum):
    """Status of an arbitrage position."""
    PENDING = "pending"  # Orders placed, waiting for fills
    PARTIAL = "partial"  # One side filled, waiting for other
    LOCKED = "locked"    # Both sides filled, profit locked
    EXPIRED = "expired"  # Market expired before completion
    CANCELLED = "cancelled"  # Manually cancelled


@dataclass
class ArbOpportunity:
    """An arbitrage opportunity in a binary market."""
    market_id: str
    question: str
    up_token_id: str
    down_token_id: str
    up_price: float  # Best ask (for buy arb) or best bid (for sell arb)
    down_price: float
    total_cost: float  # Sum of prices
    edge_bps: int  # Edge in basis points
    guaranteed_profit_pct: float
    market_end: datetime
    arb_type: str = "buy"  # "buy" = buy both for < $1, "sell" = sell both for > $1
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Additional market data for context
    up_ask: float = 0.0
    down_ask: float = 0.0
    up_bid: float = 0.0
    down_bid: float = 0.0
    spread_up: float = 0.0  # bid-ask spread
    spread_down: float = 0.0

    @property
    def seconds_to_expiry(self) -> float:
        """Seconds until market expires."""
        now = datetime.now(timezone.utc)
        return max(0, (self.market_end - now).total_seconds())

    @property
    def is_valid(self) -> bool:
        """Check if opportunity is still valid (market not expired)."""
        return self.seconds_to_expiry > 60  # Need at least 1 min


@dataclass
class ArbOrder:
    """A single order in an arb position."""
    order_id: str
    token_id: str
    side: str  # "UP" or "DOWN"
    price: float
    size: float  # shares
    placed_at: datetime
    filled: bool = False
    fill_price: float = 0.0
    fill_time: Optional[datetime] = None


@dataclass
class ArbPosition:
    """A delta-neutral arbitrage position."""
    position_id: str
    market_id: str
    question: str
    up_order: Optional[ArbOrder] = None
    down_order: Optional[ArbOrder] = None
    status: ArbPositionStatus = ArbPositionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    target_edge_bps: int = 0  # Target edge when position was opened

    @property
    def up_filled(self) -> bool:
        return self.up_order is not None and self.up_order.filled

    @property
    def down_filled(self) -> bool:
        return self.down_order is not None and self.down_order.filled

    @property
    def is_locked(self) -> bool:
        """Both sides filled = profit locked."""
        return self.up_filled and self.down_filled

    @property
    def total_cost(self) -> float:
        """Total cost to acquire both sides."""
        cost = 0.0
        if self.up_order and self.up_order.filled:
            cost += self.up_order.size * self.up_order.fill_price
        if self.down_order and self.down_order.filled:
            cost += self.down_order.size * self.down_order.fill_price
        return cost

    @property
    def guaranteed_payout(self) -> float:
        """Guaranteed payout at resolution (if both filled)."""
        if not self.is_locked:
            return 0.0
        # Payout = min shares (we can only redeem paired shares)
        up_shares = self.up_order.size if self.up_order else 0
        down_shares = self.down_order.size if self.down_order else 0
        return min(up_shares, down_shares) * 1.0  # $1 per share pair

    @property
    def locked_profit(self) -> float:
        """Guaranteed profit if position is locked."""
        if not self.is_locked:
            return 0.0
        return self.guaranteed_payout - self.total_cost


@dataclass
class ArbConfig:
    """Configuration for arbitrage strategy."""
    # Minimum edge to consider (in basis points)
    # 50 bps = 0.5% = $5 profit per $1000 invested
    min_edge_bps: int = 50

    # Order size in USD per side
    order_size_usd: float = 20.0

    # Maximum positions open at once
    max_positions: int = 5

    # Time to wait for second fill before adjusting (seconds)
    fill_patience_seconds: int = 60

    # How much to improve price if second side doesn't fill (bps)
    price_improvement_bps: int = 10

    # Minimum time to market end to open new position (seconds)
    min_time_to_expiry: int = 300  # 5 minutes

    # Fee assumption (taker fee in bps)
    fee_bps: int = 10


@dataclass
class MarketStatus:
    """Current status of a market for monitoring."""
    market_id: str
    question: str
    up_ask: float
    down_ask: float
    up_bid: float
    down_bid: float
    buy_cost: float  # Cost to buy both (after fees)
    sell_proceeds: float  # Proceeds from selling both (after fees)
    buy_edge_bps: int  # Negative = cost > $1
    sell_edge_bps: int  # Negative = proceeds < $1
    spread_up_bps: int
    spread_down_bps: int
    seconds_left: float


class ArbScanner:
    """Scans markets for arbitrage opportunities."""

    def __init__(self, api: PolymarketAPI, config: ArbConfig):
        self.api = api
        self.config = config

    async def scan_for_opportunities(
        self,
        markets: List[Market],
    ) -> List[ArbOpportunity]:
        """Scan markets for arbitrage opportunities."""
        opportunities = []

        for market in markets:
            opp = await self._check_market(market)
            if opp and opp.edge_bps >= self.config.min_edge_bps:
                opportunities.append(opp)

        # Sort by edge (best first)
        opportunities.sort(key=lambda x: x.edge_bps, reverse=True)

        return opportunities

    async def get_market_statuses(
        self,
        markets: List[Market],
    ) -> List[MarketStatus]:
        """Get status of all markets (for monitoring even without arb)."""
        statuses = []

        fee_multiplier = 1 + (self.config.fee_bps / 10000)
        fee_reduction = 1 - (self.config.fee_bps / 10000)

        for market in markets:
            if len(market.tokens) != 2:
                continue

            up_token = market.tokens[0]
            down_token = market.tokens[1]

            up_book = await self.api.fetch_orderbook(up_token.token_id)
            down_book = await self.api.fetch_orderbook(down_token.token_id)

            if not up_book or not down_book:
                continue

            up_ask = up_book.best_ask or 0
            down_ask = down_book.best_ask or 0
            up_bid = up_book.best_bid or 0
            down_bid = down_book.best_bid or 0

            buy_cost = (up_ask + down_ask) * fee_multiplier if up_ask and down_ask else 999
            sell_proceeds = (up_bid + down_bid) * fee_reduction if up_bid and down_bid else 0

            buy_edge = int((1.0 - buy_cost) * 10000)
            sell_edge = int((sell_proceeds - 1.0) * 10000)

            spread_up = int((up_ask - up_bid) / up_ask * 10000) if up_ask and up_bid else 0
            spread_down = int((down_ask - down_bid) / down_ask * 10000) if down_ask and down_bid else 0

            statuses.append(MarketStatus(
                market_id=market.condition_id,
                question=market.question,
                up_ask=up_ask,
                down_ask=down_ask,
                up_bid=up_bid,
                down_bid=down_bid,
                buy_cost=buy_cost,
                sell_proceeds=sell_proceeds,
                buy_edge_bps=buy_edge,
                sell_edge_bps=sell_edge,
                spread_up_bps=spread_up,
                spread_down_bps=spread_down,
                seconds_left=market.seconds_left,
            ))

        # Sort by best opportunity (closest to arb)
        statuses.sort(key=lambda x: max(x.buy_edge_bps, x.sell_edge_bps), reverse=True)

        return statuses

    async def _check_market(self, market: Market) -> Optional[ArbOpportunity]:
        """Check a single market for arb opportunity (buy or sell)."""
        if len(market.tokens) != 2:
            return None

        # Need enough time to execute
        if market.seconds_left < self.config.min_time_to_expiry:
            return None

        # Get orderbooks for both tokens
        up_token = market.tokens[0]
        down_token = market.tokens[1]

        up_book = await self.api.fetch_orderbook(up_token.token_id)
        down_book = await self.api.fetch_orderbook(down_token.token_id)

        if not up_book or not down_book:
            return None

        up_ask = up_book.best_ask or 0
        down_ask = down_book.best_ask or 0
        up_bid = up_book.best_bid or 0
        down_bid = down_book.best_bid or 0

        # Calculate spreads
        spread_up = (up_ask - up_bid) if up_ask and up_bid else 0
        spread_down = (down_ask - down_bid) if down_ask and down_bid else 0

        # Fee estimate
        fee_multiplier = 1 + (self.config.fee_bps / 10000)
        fee_reduction = 1 - (self.config.fee_bps / 10000)

        # Check BUY arbitrage: buy both asks for < $1
        buy_arb = None
        if up_ask and down_ask:
            buy_cost = (up_ask + down_ask) * fee_multiplier
            if buy_cost < 1.0:
                edge = 1.0 - buy_cost
                buy_arb = ArbOpportunity(
                    market_id=market.condition_id,
                    question=market.question,
                    up_token_id=up_token.token_id,
                    down_token_id=down_token.token_id,
                    up_price=up_ask,
                    down_price=down_ask,
                    total_cost=buy_cost,
                    edge_bps=int(edge * 10000),
                    guaranteed_profit_pct=(1.0 / buy_cost - 1) * 100,
                    market_end=market.end_date,
                    arb_type="buy",
                    up_ask=up_ask,
                    down_ask=down_ask,
                    up_bid=up_bid,
                    down_bid=down_bid,
                    spread_up=spread_up,
                    spread_down=spread_down,
                )

        # Check SELL arbitrage: sell both bids for > $1 (need to own shares)
        sell_arb = None
        if up_bid and down_bid:
            sell_proceeds = (up_bid + down_bid) * fee_reduction
            if sell_proceeds > 1.0:
                edge = sell_proceeds - 1.0
                sell_arb = ArbOpportunity(
                    market_id=market.condition_id,
                    question=market.question,
                    up_token_id=up_token.token_id,
                    down_token_id=down_token.token_id,
                    up_price=up_bid,
                    down_price=down_bid,
                    total_cost=sell_proceeds,  # Actually proceeds, not cost
                    edge_bps=int(edge * 10000),
                    guaranteed_profit_pct=edge * 100,
                    market_end=market.end_date,
                    arb_type="sell",
                    up_ask=up_ask,
                    down_ask=down_ask,
                    up_bid=up_bid,
                    down_bid=down_bid,
                    spread_up=spread_up,
                    spread_down=spread_down,
                )

        # Return best opportunity (or the one that exists)
        if buy_arb and sell_arb:
            return buy_arb if buy_arb.edge_bps >= sell_arb.edge_bps else sell_arb
        return buy_arb or sell_arb


class ArbPositionManager:
    """Manages arbitrage positions."""

    def __init__(
        self,
        api: PolymarketAPI,
        config: ArbConfig,
        dry_run: bool = True,
    ):
        self.api = api
        self.config = config
        self.dry_run = dry_run
        self.positions: Dict[str, ArbPosition] = {}
        self.client = None  # CLOB client for live trading
        self._position_counter = 0

    def set_client(self, client):
        """Set the CLOB client for live trading."""
        self.client = client

    @property
    def open_positions(self) -> List[ArbPosition]:
        """Get all non-completed positions."""
        return [
            p for p in self.positions.values()
            if p.status in (ArbPositionStatus.PENDING, ArbPositionStatus.PARTIAL)
        ]

    @property
    def locked_positions(self) -> List[ArbPosition]:
        """Get all locked (profitable) positions."""
        return [
            p for p in self.positions.values()
            if p.status == ArbPositionStatus.LOCKED
        ]

    def get_stats(self) -> Dict:
        """Get position statistics."""
        locked = self.locked_positions
        total_locked_profit = sum(p.locked_profit for p in locked)
        total_invested = sum(p.total_cost for p in locked)

        return {
            "open_positions": len(self.open_positions),
            "locked_positions": len(locked),
            "total_locked_profit": total_locked_profit,
            "total_invested": total_invested,
            "avg_edge_bps": (
                sum(p.target_edge_bps for p in locked) / len(locked)
                if locked else 0
            ),
        }

    async def open_position(
        self,
        opportunity: ArbOpportunity,
    ) -> Optional[ArbPosition]:
        """Open a new arbitrage position."""
        if len(self.open_positions) >= self.config.max_positions:
            logger.warning("Max positions reached, skipping opportunity")
            return None

        self._position_counter += 1
        position_id = f"arb_{self._position_counter}_{datetime.now().strftime('%H%M%S')}"

        # Calculate order sizes
        # We want equal dollar amounts on each side
        up_shares = self.config.order_size_usd / opportunity.up_ask
        down_shares = self.config.order_size_usd / opportunity.down_ask

        # Use the smaller share count to ensure we can pair them
        shares = min(up_shares, down_shares)

        position = ArbPosition(
            position_id=position_id,
            market_id=opportunity.market_id,
            question=opportunity.question,
            target_edge_bps=opportunity.edge_bps,
        )

        # Place orders on both sides
        up_order = await self._place_order(
            token_id=opportunity.up_token_id,
            side="UP",
            price=opportunity.up_ask,
            size=shares,
            position_id=position_id,
        )

        down_order = await self._place_order(
            token_id=opportunity.down_token_id,
            side="DOWN",
            price=opportunity.down_ask,
            size=shares,
            position_id=position_id,
        )

        if up_order:
            position.up_order = up_order
        if down_order:
            position.down_order = down_order

        if not up_order and not down_order:
            logger.error(f"Failed to place any orders for {position_id}")
            return None

        self.positions[position_id] = position

        logger.info(
            f"Opened arb position {position_id}: "
            f"{opportunity.question[:40]}... "
            f"edge={opportunity.edge_bps}bps"
        )

        return position

    async def _place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        position_id: str,
    ) -> Optional[ArbOrder]:
        """Place a single order."""
        if self.dry_run:
            order_id = f"dry_{position_id}_{side}"
            logger.info(
                f"[DRY RUN] Would place BUY {size:.4f} {side} shares "
                f"@ ${price:.4f} for {token_id[:16]}..."
            )
            return ArbOrder(
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                placed_at=datetime.now(timezone.utc),
            )

        if not self.client:
            logger.error("No CLOB client available")
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY

            order_args = OrderArgs(
                price=price,
                size=size,
                side=BUY,
                token_id=token_id,
                fee_rate_bps=self.config.fee_bps,
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTC)

            if isinstance(response, dict):
                success = response.get("success", False)
                order_id = response.get("orderID", "")
            else:
                success = getattr(response, "success", False)
                order_id = getattr(response, "orderID", "")

            if not success:
                logger.warning(f"Failed to place {side} order")
                return None

            logger.info(
                f"Placed BUY {size:.4f} {side} shares "
                f"@ ${price:.4f} for {token_id[:16]}..."
            )

            return ArbOrder(
                order_id=order_id,
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                placed_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            return None

    async def update_positions(self):
        """Update status of all open positions."""
        for position in self.open_positions:
            await self._update_position(position)

    async def _update_position(self, position: ArbPosition):
        """Update a single position's status."""
        # In dry run, simulate random fills
        if self.dry_run:
            import random
            if position.up_order and not position.up_order.filled:
                if random.random() < 0.1:  # 10% chance per update
                    position.up_order.filled = True
                    position.up_order.fill_price = position.up_order.price
                    position.up_order.fill_time = datetime.now(timezone.utc)
                    logger.info(f"[DRY RUN] UP order filled for {position.position_id}")

            if position.down_order and not position.down_order.filled:
                if random.random() < 0.1:
                    position.down_order.filled = True
                    position.down_order.fill_price = position.down_order.price
                    position.down_order.fill_time = datetime.now(timezone.utc)
                    logger.info(f"[DRY RUN] DOWN order filled for {position.position_id}")
        else:
            # Live: check order status via API
            # TODO: Implement actual order status checking
            pass

        # Update position status
        if position.is_locked:
            position.status = ArbPositionStatus.LOCKED
            logger.info(
                f"Position {position.position_id} LOCKED! "
                f"Profit: ${position.locked_profit:.4f}"
            )
        elif position.up_filled or position.down_filled:
            position.status = ArbPositionStatus.PARTIAL
            # Check if we've waited too long
            wait_time = (datetime.now(timezone.utc) - position.created_at).total_seconds()
            if wait_time > self.config.fill_patience_seconds:
                logger.warning(
                    f"Position {position.position_id} partial fill timeout - "
                    f"consider adjusting unfilled side"
                )

    async def cancel_position(self, position_id: str) -> bool:
        """Cancel an open position."""
        if position_id not in self.positions:
            return False

        position = self.positions[position_id]

        if position.status not in (ArbPositionStatus.PENDING, ArbPositionStatus.PARTIAL):
            return False

        # Cancel unfilled orders
        if position.up_order and not position.up_order.filled:
            await self._cancel_order(position.up_order)
        if position.down_order and not position.down_order.filled:
            await self._cancel_order(position.down_order)

        position.status = ArbPositionStatus.CANCELLED
        logger.info(f"Cancelled position {position_id}")

        return True

    async def _cancel_order(self, order: ArbOrder):
        """Cancel a single order."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would cancel order {order.order_id}")
            return

        if not self.client:
            return

        try:
            self.client.cancel(order.order_id)
            logger.info(f"Cancelled order {order.order_id}")
        except Exception as e:
            logger.error(f"Error cancelling order {order.order_id}: {e}")


class ArbStrategy:
    """
    Delta-neutral arbitrage strategy.

    Monitors 15-minute crypto markets for pricing inefficiencies
    and opens positions when both outcomes can be bought for < $1.
    """

    def __init__(
        self,
        api: PolymarketAPI,
        config: Optional[ArbConfig] = None,
        dry_run: bool = True,
    ):
        self.api = api
        self.config = config or ArbConfig()
        self.dry_run = dry_run
        self.scanner = ArbScanner(api, self.config)
        self.position_manager = ArbPositionManager(api, self.config, dry_run)
        self._markets: List[Market] = []

    def set_client(self, client):
        """Set CLOB client for live trading."""
        self.position_manager.set_client(client)

    async def update_markets(self):
        """Fetch latest 15-minute crypto markets."""
        raw_markets = await self.api.fetch_15min_markets(
            window_count=8,  # 2 hours ahead
            include_past=1,
            cryptos=["btc", "eth", "sol", "xrp"],
        )

        self._markets = []
        for raw in raw_markets:
            parsed = self.api.parse_market(raw)
            if parsed and parsed.fees_enabled and parsed.enable_order_book:
                self._markets.append(parsed)

        logger.debug(f"Updated markets: {len(self._markets)} active")

    async def scan_and_execute(self) -> List[ArbOpportunity]:
        """Scan for opportunities and open positions."""
        # Find opportunities
        opportunities = await self.scanner.scan_for_opportunities(self._markets)

        if opportunities:
            logger.info(f"Found {len(opportunities)} arb opportunities")
            for opp in opportunities[:3]:  # Log top 3
                logger.info(
                    f"  {opp.question[:40]}... "
                    f"edge={opp.edge_bps}bps "
                    f"({opp.guaranteed_profit_pct:.2f}% profit)"
                )

        # Open positions for best opportunities
        opened = []
        for opp in opportunities:
            if len(self.position_manager.open_positions) >= self.config.max_positions:
                break

            position = await self.position_manager.open_position(opp)
            if position:
                opened.append(opp)

        # Update existing positions
        await self.position_manager.update_positions()

        return opportunities

    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        pm_stats = self.position_manager.get_stats()
        return {
            **pm_stats,
            "active_markets": len(self._markets),
            "dry_run": self.dry_run,
        }
