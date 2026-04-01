"""
Polymarket data adapter.

Converts between Polymarket API formats and unified omnitrade models.
Handles:
- Gamma API market data -> Instrument
- CLOB orderbook -> OrderbookSnapshot
- CLOB order responses -> OrderResult
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from ...core.enums import ExchangeId, InstrumentType, Side, OrderStatus
from ...core.models import (
    Instrument, OrderbookSnapshot, OrderbookLevel,
    OrderResult, OpenOrder, AccountBalance, ExchangePosition,
)

logger = logging.getLogger(__name__)


class PolymarketAdapter:
    """Converts Polymarket API responses to unified models."""

    @staticmethod
    def market_to_instruments(market_data: dict) -> list[Instrument]:
        """Convert a Gamma API market response to Instrument list (one per token)."""
        instruments = []

        # Handle both camelCase (live API) and snake_case (tests) field names
        condition_id = market_data.get("conditionId") or market_data.get("condition_id", "")
        question = market_data.get("question", "")
        end_date_str = market_data.get("endDateIso") or market_data.get("end_date_iso", "")
        closed = market_data.get("closed", False)
        active = market_data.get("active", True) and not closed

        expiry = None
        if end_date_str:
            try:
                expiry = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        # Prefer explicit tokens array (used in tests), fall back to
        # constructing from clobTokenIds + outcomes + outcomePrices (live API)
        tokens = market_data.get("tokens", [])
        if not tokens:
            import json
            raw_ids = market_data.get("clobTokenIds", "[]")
            token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
            outcomes = market_data.get("outcomes", [])
            prices = market_data.get("outcomePrices", [])
            if isinstance(prices, str):
                prices = json.loads(prices)
            for i, tid in enumerate(token_ids):
                tokens.append({
                    "token_id": tid,
                    "outcome": outcomes[i] if i < len(outcomes) else "",
                    "price": float(prices[i]) if i < len(prices) else 0.0,
                })

        for token in tokens:
            token_id = token.get("token_id", "")
            outcome = token.get("outcome", "")
            price = float(token.get("price", 0))

            # Validate critical fields before creating Instrument
            if not token_id:
                logger.warning(
                    "Skipping Polymarket token with empty instrument_id "
                    "(market=%s, outcome=%s)", condition_id, outcome,
                )
                continue

            if not (0.0 <= price <= 1.0):
                logger.warning(
                    "Skipping Polymarket token with out-of-range price %.4f "
                    "(instrument_id=%s)", price, token_id,
                )
                continue

            instruments.append(Instrument(
                instrument_id=token_id,
                exchange=ExchangeId.POLYMARKET,
                instrument_type=InstrumentType.BINARY_OUTCOME,
                name=f"{question} - {outcome}",
                price=price,
                market_id=condition_id,
                outcome=outcome,
                active=active,
                closed=closed,
                expiry=expiry,
                tick_size=0.01,
                min_order_size=1.0,
                raw=market_data,
            ))

        return instruments

    @staticmethod
    def orderbook_to_snapshot(token_id: str, book_data) -> OrderbookSnapshot:
        """Convert CLOB orderbook response to OrderbookSnapshot."""
        bids = []
        asks = []

        if hasattr(book_data, 'bids') and book_data.bids:
            for b in book_data.bids:
                bids.append(OrderbookLevel(
                    price=float(b.price),
                    size=float(b.size),
                ))

        if hasattr(book_data, 'asks') and book_data.asks:
            for a in book_data.asks:
                asks.append(OrderbookLevel(
                    price=float(a.price),
                    size=float(a.size),
                ))

        # Sort: bids highest first, asks lowest first
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderbookSnapshot(
            instrument_id=token_id,
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def order_response_to_result(response, requested_size: float = 0, requested_price: float = 0) -> OrderResult:
        """Convert CLOB order response to OrderResult."""
        def _get(field, default=None):
            if isinstance(response, dict):
                return response.get(field, default)
            return getattr(response, field, default)

        success = _get("success", False)
        if not success:
            error = _get("errorMsg") or _get("error_msg") or "Order failed"
            return OrderResult(
                success=False,
                requested_size=requested_size,
                requested_price=requested_price,
                error_message=str(error),
            )

        order_id = _get("orderID") or _get("order_id") or ""
        status_str = str(_get("status", "") or "").lower()

        if status_str in ("matched", "filled"):
            order_status = OrderStatus.FILLED
            filled_size = requested_size
            try:
                taking = _get("takingAmount") or _get("taking_amount")
                if taking:
                    filled_size = float(taking)
            except (ValueError, TypeError):
                pass
        else:
            order_status = OrderStatus.OPEN
            filled_size = 0.0

        return OrderResult(
            success=True,
            order_id=str(order_id),
            status=order_status,
            filled_size=filled_size,
            filled_price=requested_price,
            requested_size=requested_size,
            requested_price=requested_price,
        )

    @staticmethod
    def position_to_exchange_position(pos_data: dict) -> ExchangePosition:
        """Convert Polymarket API position to ExchangePosition."""
        size = float(pos_data.get("size", 0))
        avg_price = float(pos_data.get("avgPrice", pos_data.get("avg_price", 0)))
        cur_price = float(pos_data.get("curPrice", pos_data.get("cur_price", avg_price)))

        return ExchangePosition(
            instrument_id=pos_data.get("asset", pos_data.get("token_id", "")),
            exchange=ExchangeId.POLYMARKET,
            side=Side.BUY,  # Polymarket positions are always long
            size=size,
            entry_price=avg_price,
            current_price=cur_price,
            unrealized_pnl=(cur_price - avg_price) * size if cur_price and avg_price else 0,
        )
