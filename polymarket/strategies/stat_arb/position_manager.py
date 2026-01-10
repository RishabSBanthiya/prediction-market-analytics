"""
Multi-Leg Position Manager for Statistical Arbitrage.

Handles atomic execution of multi-leg trades with:
- Capital reservation via RiskCoordinator
- Sequential leg execution with rollback on failure
- Position monitoring and exit execution
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid

from polymarket.core.api import PolymarketAPI
from polymarket.core.models import Side
from polymarket.trading.risk_coordinator import RiskCoordinator

from .models import (
    StatArbOpportunity,
    StatArbPosition,
    StatArbPositionStatus,
    ArbLeg,
    ArbType,
)
from .config import StatArbConfig

logger = logging.getLogger(__name__)


class StatArbPositionManager:
    """
    Manages multi-leg statistical arbitrage positions.

    Key responsibilities:
    - Atomic capital reservation for all legs
    - Sequential execution with rollback on partial failure
    - Position tracking and P&L calculation
    - Exit execution (take profit, stop loss, mean reversion)
    """

    def __init__(
        self,
        api: PolymarketAPI,
        config: StatArbConfig,
        risk_coordinator: Optional[RiskCoordinator] = None,
        dry_run: bool = True,
    ):
        self.api = api
        self.config = config
        self.risk_coordinator = risk_coordinator
        self.dry_run = dry_run

        # Active positions
        self.positions: Dict[str, StatArbPosition] = {}

        # CLOB client for live trading (set externally)
        self.clob_client = None

    def set_clob_client(self, client) -> None:
        """Set the CLOB client for live order execution."""
        self.clob_client = client

    async def open_position(
        self,
        opportunity: StatArbOpportunity,
        agent_id: str,
        position_size_usd: float,
    ) -> Optional[StatArbPosition]:
        """
        Open a multi-leg position atomically.

        Args:
            opportunity: The arbitrage opportunity to execute
            agent_id: Agent opening the position
            position_size_usd: Total position size in USD

        Returns:
            StatArbPosition if successful, None otherwise
        """
        # Check position limits
        if len(self.positions) >= self.config.max_total_positions:
            logger.warning(f"Max positions reached: {len(self.positions)}")
            return None

        # Scale legs to desired position size
        scaled_opportunity = self._scale_opportunity(opportunity, position_size_usd)

        # Calculate total capital needed
        total_required = sum(
            leg.target_shares * leg.target_price
            for leg in scaled_opportunity.legs
            if leg.side == "BUY"
        )

        # Reserve capital atomically
        reservation_id = None
        if self.risk_coordinator:
            reservation_id = await self._reserve_capital(
                agent_id=agent_id,
                market_id=scaled_opportunity.market_ids[0],
                amount=total_required,
            )
            if not reservation_id:
                logger.warning("Failed to reserve capital")
                return None

        try:
            # Create position
            position = StatArbPosition.from_opportunity(scaled_opportunity, agent_id)
            position.total_entry_cost = total_required

            # Execute all legs
            filled_legs = []
            for leg in position.legs:
                result = await self._execute_leg(leg)

                if not result["success"]:
                    # Unwind filled legs
                    logger.warning(f"Leg execution failed: {result.get('error')}")
                    await self._unwind_legs(filled_legs)

                    if reservation_id and self.risk_coordinator:
                        await self._release_reservation(reservation_id)

                    return None

                # Update leg with fill info
                leg.filled = True
                leg.fill_price = result["fill_price"]
                leg.fill_shares = result["fill_shares"]
                leg.fill_time = datetime.now(timezone.utc)
                leg.order_id = result.get("order_id")

                filled_legs.append(leg)

            # All legs filled - position is active
            position.status = StatArbPositionStatus.FILLED

            # Confirm execution with risk coordinator
            if reservation_id and self.risk_coordinator:
                await self._confirm_execution(
                    reservation_id,
                    position.total_fill_cost,
                )

            # Store position
            self.positions[position.position_id] = position

            logger.info(
                f"Opened {position.arb_type.value} position {position.position_id[:8]} "
                f"with {len(position.legs)} legs, cost=${position.total_fill_cost:.2f}"
            )

            return position

        except Exception as e:
            logger.error(f"Position open failed: {e}")

            if reservation_id and self.risk_coordinator:
                await self._release_reservation(reservation_id)

            raise

    async def close_position(
        self,
        position_id: str,
        reason: str = "manual",
    ) -> Optional[float]:
        """
        Close all legs of a position.

        Args:
            position_id: Position to close
            reason: Reason for closing (manual, take_profit, stop_loss, mean_reversion)

        Returns:
            Realized P&L if successful, None otherwise
        """
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None

        if position.status not in (StatArbPositionStatus.FILLED,):
            logger.warning(f"Cannot close position in status: {position.status}")
            return None

        position.status = StatArbPositionStatus.CLOSING
        total_pnl = 0.0

        try:
            for leg in position.legs:
                if not leg.filled:
                    continue

                # Reverse the leg
                close_side = "SELL" if leg.side == "BUY" else "BUY"

                result = await self._execute_leg(ArbLeg(
                    token_id=leg.token_id,
                    market_id=leg.market_id,
                    outcome=leg.outcome,
                    side=close_side,
                    target_price=0.0,  # Market order
                    target_shares=leg.fill_shares,
                ))

                if result["success"]:
                    # Calculate P&L for this leg
                    entry_value = leg.fill_shares * leg.fill_price
                    exit_value = result["fill_shares"] * result["fill_price"]

                    if leg.side == "BUY":
                        # Bought at entry, sold at exit
                        leg_pnl = exit_value - entry_value
                    else:
                        # Sold at entry, bought back at exit
                        leg_pnl = entry_value - exit_value

                    total_pnl += leg_pnl

            position.status = StatArbPositionStatus.CLOSED
            position.closed_at = datetime.now(timezone.utc)
            position.realized_pnl = total_pnl
            position.metadata["close_reason"] = reason

            logger.info(
                f"Closed position {position_id[:8]} ({reason}), "
                f"P&L=${total_pnl:.2f}"
            )

            return total_pnl

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            position.status = StatArbPositionStatus.FILLED  # Revert status
            return None

    async def update_positions(self) -> Dict[str, Dict]:
        """
        Update all active positions with current prices.

        Returns dict of position updates including any that need closing.
        """
        updates = {}

        for position_id, position in self.positions.items():
            if position.status != StatArbPositionStatus.FILLED:
                continue

            # Get current prices for all tokens
            prices = {}
            for leg in position.legs:
                try:
                    orderbook = await self.api.fetch_orderbook(leg.token_id)
                    if orderbook:
                        prices[leg.token_id] = (orderbook.best_bid + orderbook.best_ask) / 2
                except Exception:
                    pass

            if not prices:
                continue

            # Update position metrics
            position.update_from_prices(prices)

            # Check exit conditions
            should_close, reason = self._check_exit_conditions(position, prices)

            updates[position_id] = {
                "unrealized_pnl": position.unrealized_pnl,
                "current_spread": position.current_spread,
                "current_z_score": position.current_z_score,
                "should_close": should_close,
                "close_reason": reason,
            }

        return updates

    def _check_exit_conditions(
        self,
        position: StatArbPosition,
        prices: Dict[str, float],
    ) -> tuple[bool, str]:
        """Check if position should be closed."""
        if position.arb_type == ArbType.PAIR_SPREAD:
            # Calculate current spread
            if len(position.legs) >= 2:
                leg_a = position.legs[0]
                leg_b = position.legs[1]

                price_a = prices.get(leg_a.token_id, 0)
                price_b = prices.get(leg_b.token_id, 0)

                # Spread direction depends on how we entered
                if leg_a.side == "BUY":
                    current_spread = price_a - price_b
                else:
                    current_spread = price_b - price_a

                position.current_spread = current_spread

                # Check mean reversion target
                if position.target_spread != 0:
                    if position.entry_z_score > 0:
                        # We shorted the spread, target is lower
                        if current_spread <= position.target_spread:
                            return True, "mean_reversion"
                    else:
                        # We longed the spread, target is higher
                        if current_spread >= position.target_spread:
                            return True, "mean_reversion"

                # Check stop loss
                if position.stop_spread != 0:
                    if position.entry_z_score > 0:
                        if current_spread >= position.stop_spread:
                            return True, "stop_loss"
                    else:
                        if current_spread <= position.stop_spread:
                            return True, "stop_loss"

        # Check time-based exit
        if position.target_close_at:
            if datetime.now(timezone.utc) >= position.target_close_at:
                return True, "time_exit"

        # Check max holding time
        age_hours = (datetime.now(timezone.utc) - position.opened_at).total_seconds() / 3600
        if age_hours >= self.config.pair_trading.max_holding_hours:
            return True, "max_hold_time"

        return False, ""

    async def _execute_leg(self, leg: ArbLeg) -> Dict:
        """Execute a single leg of the trade."""
        # Handle case where leg is a dict instead of ArbLeg
        if isinstance(leg, dict):
            leg = ArbLeg(
                token_id=leg["token_id"],
                market_id=leg["market_id"],
                outcome=leg.get("outcome", ""),
                side=leg["side"],
                target_price=leg["target_price"],
                target_shares=leg["target_shares"],
            )

        if self.dry_run:
            return await self._execute_leg_dry_run(leg)
        else:
            return await self._execute_leg_live(leg)

    async def _execute_leg_dry_run(self, leg: ArbLeg) -> Dict:
        """Simulate leg execution in dry run mode."""
        logger.info(
            f"[DRY RUN] {leg.side} {leg.target_shares:.4f} shares of "
            f"{leg.outcome} @ ${leg.target_price:.4f}"
        )

        return {
            "success": True,
            "order_id": f"dry_run_{uuid.uuid4().hex[:8]}",
            "fill_price": leg.target_price,
            "fill_shares": leg.target_shares,
        }

    async def _execute_leg_live(self, leg: ArbLeg) -> Dict:
        """Execute leg via CLOB client."""
        # Ensure leg is an ArbLeg object
        if isinstance(leg, dict):
            leg = ArbLeg(
                token_id=leg["token_id"],
                market_id=leg["market_id"],
                outcome=leg.get("outcome", ""),
                side=leg["side"],
                target_price=leg["target_price"],
                target_shares=leg["target_shares"],
            )

        if not self.clob_client:
            return {"success": False, "error": "CLOB client not configured"}

        try:
            # Use py_clob_client
            from py_clob_client.order_builder.constants import BUY, SELL
            from py_clob_client.clob_types import OrderArgs

            # Create OrderArgs object (not a dict)
            order_args = OrderArgs(
                token_id=leg.token_id,
                price=leg.target_price,
                size=leg.target_shares,
                side=BUY if leg.side == "BUY" else SELL,
            )

            # Create and sign order
            signed_order = self.clob_client.create_order(order_args)
            response = self.clob_client.post_order(signed_order, "GTC")

            if response and response.get("orderID"):
                return {
                    "success": True,
                    "order_id": response["orderID"],
                    "fill_price": leg.target_price,
                    "fill_shares": leg.target_shares,
                }
            else:
                return {
                    "success": False,
                    "error": f"Order failed: {response}",
                }

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def _unwind_legs(self, filled_legs: List[ArbLeg]) -> None:
        """Unwind filled legs after a failure."""
        logger.warning(f"Unwinding {len(filled_legs)} filled legs")

        for leg in filled_legs:
            try:
                reverse_side = "SELL" if leg.side == "BUY" else "BUY"
                await self._execute_leg(ArbLeg(
                    token_id=leg.token_id,
                    market_id=leg.market_id,
                    outcome=leg.outcome,
                    side=reverse_side,
                    target_price=0.0,  # Market order
                    target_shares=leg.fill_shares,
                ))
            except Exception as e:
                logger.error(f"Failed to unwind leg: {e}")

    async def _reserve_capital(
        self,
        agent_id: str,
        market_id: str,
        amount: float,
    ) -> Optional[str]:
        """Reserve capital via RiskCoordinator."""
        if not self.risk_coordinator:
            return "no_coordinator"

        try:
            reservation = self.risk_coordinator.atomic_reserve(
                agent_id=agent_id,
                market_id=market_id,
                token_id="",
                amount_usd=amount,
            )
            return reservation
        except Exception as e:
            logger.error(f"Capital reservation failed: {e}")
            return None

    async def _confirm_execution(
        self,
        reservation_id: str,
        actual_amount: float,
    ) -> None:
        """Confirm execution with RiskCoordinator."""
        if not self.risk_coordinator or reservation_id == "no_coordinator":
            return

        try:
            self.risk_coordinator.confirm_execution(
                reservation_id=reservation_id,
                filled_shares=0,
                filled_price=0,
                requested_shares=0,
            )
        except Exception as e:
            logger.error(f"Execution confirmation failed: {e}")

    async def _release_reservation(self, reservation_id: str) -> None:
        """Release capital reservation."""
        if not self.risk_coordinator or reservation_id == "no_coordinator":
            return

        try:
            self.risk_coordinator.release_reservation(reservation_id)
        except Exception as e:
            logger.error(f"Reservation release failed: {e}")

    def _scale_opportunity(
        self,
        opportunity: StatArbOpportunity,
        target_size_usd: float,
    ) -> StatArbOpportunity:
        """Scale opportunity legs to target position size."""
        # Calculate current total cost
        current_cost = sum(
            leg.target_shares * leg.target_price
            for leg in opportunity.legs
            if leg.side == "BUY"
        )

        if current_cost <= 0:
            return opportunity

        # Scale factor
        scale = target_size_usd / current_cost

        # Scale all legs
        for leg in opportunity.legs:
            leg.target_shares *= scale

        # Update totals
        opportunity.total_cost *= scale
        opportunity.expected_profit *= scale

        return opportunity

    def get_stats(self) -> Dict:
        """Get position manager statistics."""
        open_positions = sum(
            1 for p in self.positions.values()
            if p.status == StatArbPositionStatus.FILLED
        )
        total_pnl = sum(
            p.realized_pnl for p in self.positions.values()
            if p.status == StatArbPositionStatus.CLOSED
        )

        return {
            "total_positions": len(self.positions),
            "open_positions": open_positions,
            "total_realized_pnl": total_pnl,
            "dry_run": self.dry_run,
        }
