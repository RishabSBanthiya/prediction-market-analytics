"""
Execution engine components.

Execution engines handle the actual placement of orders on the exchange.
Different engines use different strategies (market, limit, aggressive, etc.)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone

from ...core.models import Signal, ExecutionResult, OrderbookSnapshot, Side
from ...core.config import RiskConfig

if TYPE_CHECKING:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType

logger = logging.getLogger(__name__)


class ExecutionEngine(ABC):
    """
    Abstract base class for execution engines.
    
    Execution engines handle order placement and fill tracking.
    """
    
    @abstractmethod
    async def execute(
        self,
        client: "ClobClient",
        token_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        original_signal_price: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute a trade.
        
        Args:
            client: CLOB client for placing orders
            token_id: Token to trade
            side: BUY or SELL
            size_usd: Target size in USD
            price: Target price
            orderbook: Current orderbook state (optional)
            original_signal_price: Original price from signal/alert (for drift check)
        
        Returns:
            ExecutionResult with fill details
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this execution engine"""
        pass


class AggressiveExecutor(ExecutionEngine):
    """
    Aggressive execution - takes best available price.
    
    For BUY: Takes best ask (immediate fill)
    For SELL: Takes best bid (immediate fill)
    
    Prioritizes fill over price.
    
    Includes safety checks:
    - Max spread check (default 3%): Reject if bid-ask spread is too wide
    - Max price slippage from signal (default 10%): Reject if price moved too much
    """
    
    def __init__(
        self, 
        max_slippage: float = 0.02,
        max_spread: float = 0.03,  # 3% max spread
        max_price_drift: float = 0.10  # 10% max price drift from original signal
    ):
        self.max_slippage = max_slippage
        self.max_spread = max_spread
        self.max_price_drift = max_price_drift
    
    @property
    def name(self) -> str:
        return "aggressive"
    
    def _calculate_spread(self, best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
        """Calculate bid-ask spread as a percentage"""
        if best_bid is None or best_ask is None:
            return None
        if best_bid <= 0:
            return None
        
        # Spread = (ask - bid) / midpoint
        midpoint = (best_ask + best_bid) / 2
        if midpoint <= 0:
            return None
        
        spread = (best_ask - best_bid) / midpoint
        return spread
    
    async def execute(
        self,
        client: "ClobClient",
        token_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        original_signal_price: Optional[float] = None  # Price from original flow alert
    ) -> ExecutionResult:
        """Execute aggressively at best available price"""
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL
        
        try:
            # Get current orderbook if not provided
            if orderbook is None:
                book = client.get_order_book(token_id)
                if book:
                    # OrderBookSummary has .asks and .bids as lists of OrderSummary objects
                    # OrderSummary has .price and .size as string attributes
                    asks = book.asks if book.asks else []
                    bids = book.bids if book.bids else []
                    
                    # Parse and sort properly:
                    # - Bids: highest first (best bid = highest price buyer)
                    # - Asks: lowest first (best ask = lowest price seller)
                    bid_prices = sorted([float(b.price) for b in bids], reverse=True)
                    ask_prices = sorted([float(a.price) for a in asks], reverse=False)
                    
                    best_bid = bid_prices[0] if bid_prices else None
                    best_ask = ask_prices[0] if ask_prices else None
                else:
                    best_ask = None
                    best_bid = None
            else:
                best_ask = orderbook.best_ask
                best_bid = orderbook.best_bid
            
            # ============ SPREAD CHECK ============
            # Reject if bid-ask spread is too wide (poor liquidity)
            # NOTE: Only check spread for BUY orders - SELLs should always be able to exit
            # to reduce exposure, even with wide spreads (limit orders will be placed)
            spread = self._calculate_spread(best_bid, best_ask)
            if side == Side.BUY and spread is not None and spread > self.max_spread:
                logger.warning(
                    f"Spread too wide: {spread:.1%} > {self.max_spread:.1%} max. "
                    f"Bid: ${best_bid:.4f}, Ask: ${best_ask:.4f}"
                )
                return ExecutionResult(
                    success=False,
                    error_message=f"Spread too wide: {spread:.1%} (max {self.max_spread:.1%})"
                )
            elif side == Side.SELL and spread is not None and spread > self.max_spread:
                # Log but allow SELLs to proceed - reducing exposure is priority
                bid_str = f"${best_bid:.4f}" if best_bid else "N/A"
                logger.info(
                    f"Wide spread ({spread:.1%}) but allowing SELL to reduce exposure. "
                    f"Bid: {bid_str}"
                )
            
            # ============ PRICE DRIFT CHECK ============
            # Reject if price has moved too much from original signal
            if original_signal_price is not None and original_signal_price > 0:
                current_price = best_ask if side == Side.BUY else best_bid
                if current_price:
                    price_drift = abs(current_price - original_signal_price) / original_signal_price
                    if price_drift > self.max_price_drift:
                        logger.warning(
                            f"Price drifted too much from signal: {price_drift:.1%} > {self.max_price_drift:.1%} max. "
                            f"Original: ${original_signal_price:.4f}, Current: ${current_price:.4f}"
                        )
                        return ExecutionResult(
                            success=False,
                            requested_price=original_signal_price,
                            error_message=f"Price drifted {price_drift:.1%} from signal (max {self.max_price_drift:.1%})"
                        )
            
            # Determine execution price
            if side == Side.BUY:
                if best_ask is None:
                    return ExecutionResult(
                        success=False,
                        error_message="No ask available"
                    )
                exec_price = best_ask
                clob_side = BUY
            else:
                if best_bid is None:
                    return ExecutionResult(
                        success=False,
                        error_message="No bid available"
                    )
                exec_price = best_bid
                clob_side = SELL
            
            # Check slippage from target price
            if price > 0:
                slippage = abs(exec_price - price) / price
                if slippage > self.max_slippage:
                    return ExecutionResult(
                        success=False,
                        requested_price=price,
                        error_message=f"Slippage too high: {slippage:.1%}"
                    )
            
            # Calculate shares
            shares = size_usd / exec_price if exec_price > 0 else 0
            
            if shares <= 0:
                return ExecutionResult(
                    success=False,
                    error_message="Invalid share calculation"
                )
            
            # Create and place order
            order_args = OrderArgs(
                price=exec_price,
                size=shares,
                side=clob_side,
                token_id=token_id
            )
            
            signed_order = client.create_order(order_args)
            response = client.post_order(signed_order, OrderType.GTC)
            
            # Handle response - can be dict or object
            def get_response_field(field: str, default=None):
                """Get field from response, handling both dict and object types."""
                if isinstance(response, dict):
                    return response.get(field, default)
                return getattr(response, field, default)
            
            success = get_response_field("success")
            if not success:
                error_msg = get_response_field("errorMsg") or get_response_field("error_msg") or "Order failed"
                return ExecutionResult(
                    success=False,
                    requested_shares=shares,
                    requested_price=exec_price,
                    error_message=error_msg
                )
            
            # Determine fill
            order_status = str(get_response_field("status", "") or "").lower()
            order_id = get_response_field("orderID") or get_response_field("order_id") or ""
            
            if order_status in ["matched", "filled"]:
                # Immediate fill
                filled_shares = shares
                try:
                    taking_amount = get_response_field("takingAmount") or get_response_field("taking_amount")
                    if taking_amount:
                        filled_shares = float(taking_amount)
                except:
                    pass
                
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    filled_shares=filled_shares,
                    filled_price=exec_price,
                    requested_shares=shares,
                    requested_price=exec_price
                )
            else:
                # Order placed but not filled yet
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    filled_shares=0.0,
                    filled_price=exec_price,
                    requested_shares=shares,
                    requested_price=exec_price
                )
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )


class LimitOrderExecutor(ExecutionEngine):
    """
    Limit order execution - places order at specified price.
    
    May not fill immediately; order sits on book until matched.
    """
    
    def __init__(self, price_offset: float = 0.001):
        """
        Args:
            price_offset: Offset from best price to place limit order
                         Positive = more aggressive (higher bid, lower ask)
        """
        self.price_offset = price_offset
    
    @property
    def name(self) -> str:
        return "limit"
    
    async def execute(
        self,
        client: "ClobClient",
        token_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        original_signal_price: Optional[float] = None
    ) -> ExecutionResult:
        """Execute with limit order at or near target price"""
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL
        
        try:
            # Get current orderbook if not provided
            if orderbook is None:
                book = client.get_order_book(token_id)
                if book:
                    # OrderBookSummary has .asks and .bids as lists of OrderSummary objects
                    # OrderSummary has .price and .size as string attributes
                    asks = book.asks if book.asks else []
                    bids = book.bids if book.bids else []
                    
                    # Parse and sort properly:
                    # - Bids: highest first (best bid = highest price buyer)
                    # - Asks: lowest first (best ask = lowest price seller)
                    bid_prices = sorted([float(b.price) for b in bids], reverse=True)
                    ask_prices = sorted([float(a.price) for a in asks], reverse=False)
                    
                    best_bid = bid_prices[0] if bid_prices else None
                    best_ask = ask_prices[0] if ask_prices else None
                else:
                    best_ask = None
                    best_bid = None
            else:
                best_ask = orderbook.best_ask
                best_bid = orderbook.best_bid
            
            # Determine limit price
            if side == Side.BUY:
                if best_bid:
                    # Place slightly above best bid
                    exec_price = min(price, best_bid + self.price_offset)
                else:
                    exec_price = price
                clob_side = BUY
            else:
                if best_ask:
                    # Place slightly below best ask
                    exec_price = max(price, best_ask - self.price_offset)
                else:
                    exec_price = price
                clob_side = SELL
            
            # Calculate shares
            shares = size_usd / exec_price if exec_price > 0 else 0
            
            if shares <= 0:
                return ExecutionResult(
                    success=False,
                    error_message="Invalid share calculation"
                )
            
            # Create and place order
            order_args = OrderArgs(
                price=exec_price,
                size=shares,
                side=clob_side,
                token_id=token_id
            )
            
            signed_order = client.create_order(order_args)
            response = client.post_order(signed_order, OrderType.GTC)
            
            # Handle response - can be dict or object
            def get_response_field(field: str, default=None):
                """Get field from response, handling both dict and object types."""
                if isinstance(response, dict):
                    return response.get(field, default)
                return getattr(response, field, default)
            
            success = get_response_field("success")
            if not success:
                error_msg = get_response_field("errorMsg") or get_response_field("error_msg") or "Order failed"
                return ExecutionResult(
                    success=False,
                    requested_shares=shares,
                    requested_price=exec_price,
                    error_message=error_msg
                )
            
            order_id = get_response_field("orderID") or get_response_field("order_id") or ""
            order_status = str(get_response_field("status", "") or "").lower()
            
            # Check for immediate fill
            if order_status in ["matched", "filled"]:
                filled_shares = shares
                try:
                    taking_amount = get_response_field("takingAmount") or get_response_field("taking_amount")
                    if taking_amount:
                        filled_shares = float(taking_amount)
                except:
                    pass
                
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    filled_shares=filled_shares,
                    filled_price=exec_price,
                    requested_shares=shares,
                    requested_price=exec_price
                )
            else:
                # Limit order on book
                return ExecutionResult(
                    success=True,
                    order_id=order_id,
                    filled_shares=0.0,
                    filled_price=exec_price,
                    requested_shares=shares,
                    requested_price=exec_price
                )
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return ExecutionResult(
                success=False,
                error_message=str(e)
            )


class DryRunExecutor(ExecutionEngine):
    """
    Dry run executor - simulates execution without placing orders.
    
    Useful for testing and paper trading.
    """
    
    def __init__(self, fill_probability: float = 0.95):
        self.fill_probability = fill_probability
    
    @property
    def name(self) -> str:
        return "dry_run"
    
    async def execute(
        self,
        client: "ClobClient",
        token_id: str,
        side: Side,
        size_usd: float,
        price: float,
        orderbook: Optional[OrderbookSnapshot] = None,
        original_signal_price: Optional[float] = None
    ) -> ExecutionResult:
        """Simulate execution without placing real orders"""
        import random
        
        shares = size_usd / price if price > 0 else 0
        
        # Simulate fill with some probability
        if random.random() < self.fill_probability:
            # Simulate some slippage
            slippage = random.uniform(0, 0.01)
            if side == Side.BUY:
                filled_price = price * (1 + slippage)
            else:
                filled_price = price * (1 - slippage)
            
            logger.info(
                f"[DRY RUN] {side.value} {shares:.2f} shares @ ${filled_price:.4f}"
            )
            
            return ExecutionResult(
                success=True,
                order_id=f"dry_run_{datetime.now().timestamp()}",
                filled_shares=shares,
                filled_price=filled_price,
                requested_shares=shares,
                requested_price=price
            )
        else:
            logger.info(f"[DRY RUN] Order not filled (simulated)")
            return ExecutionResult(
                success=False,
                requested_shares=shares,
                requested_price=price,
                error_message="Simulated: order not filled"
            )


