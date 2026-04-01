"""
Kalshi data adapter.

Key conversion: Kalshi prices are in cents (0-99), omnitrade uses 0-1.
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


def cents_to_normalized(cents: float) -> float:
    """Convert Kalshi cents (0-99) to normalized price (0-1)."""
    return cents / 100.0


def normalized_to_cents(price: float) -> int:
    """Convert normalized price (0-1) to Kalshi cents (0-99)."""
    return max(1, min(99, round(price * 100)))


class KalshiAdapter:
    """Converts Kalshi API responses to unified models."""

    @staticmethod
    def event_to_instruments(event_data: dict) -> list[Instrument]:
        """Convert Kalshi event/market to Instruments."""
        instruments = []

        markets = event_data.get("markets", [event_data])
        for market in markets:
            ticker = market.get("ticker", "")
            title = market.get("title", market.get("subtitle", ""))
            event_ticker = market.get("event_ticker", "")

            # v3 API uses dollar fields (already 0-1); fall back to
            # legacy cents fields for backwards compatibility.
            if "yes_ask_dollars" in market:
                yes_price = float(market.get("yes_ask_dollars") or 0)
                no_price = float(market.get("no_ask_dollars") or 0)
                yes_bid = float(market.get("yes_bid_dollars") or 0)
                no_bid = float(market.get("no_bid_dollars") or 0)
            else:
                yes_price = cents_to_normalized(float(market.get("yes_ask", 50)))
                no_price = cents_to_normalized(float(market.get("no_ask", 50)))
                yes_bid = cents_to_normalized(float(market.get("yes_bid", 0)))
                no_bid = cents_to_normalized(float(market.get("no_bid", 0)))

            close_time = market.get("close_time", "")
            expiry = None
            if close_time:
                try:
                    expiry = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            status = market.get("status", "")
            active = status in ("open", "active")
            closed = status in ("closed", "settled")

            # YES contract
            instruments.append(Instrument(
                instrument_id=f"{ticker}-YES",
                exchange=ExchangeId.KALSHI,
                instrument_type=InstrumentType.EVENT_CONTRACT,
                name=f"{title} - Yes",
                price=yes_price,
                bid=yes_bid,
                ask=yes_price,
                market_id=event_ticker or ticker,
                outcome="YES",
                active=active,
                closed=closed,
                expiry=expiry,
                tick_size=0.01,
                min_order_size=1.0,
                raw=market,
            ))

            # NO contract
            instruments.append(Instrument(
                instrument_id=f"{ticker}-NO",
                exchange=ExchangeId.KALSHI,
                instrument_type=InstrumentType.EVENT_CONTRACT,
                name=f"{title} - No",
                price=no_price,
                bid=no_bid,
                ask=no_price,
                market_id=event_ticker or ticker,
                outcome="NO",
                active=active,
                closed=closed,
                expiry=expiry,
                tick_size=0.01,
                min_order_size=1.0,
                raw=market,
            ))

        return instruments

    @staticmethod
    def orderbook_to_snapshot(ticker: str, book_data: dict) -> OrderbookSnapshot:
        """Convert Kalshi orderbook to OrderbookSnapshot (0-1).

        Supports both legacy cents format and v3 dollar format.
        v3 format uses ``yes_dollars`` / ``no_dollars`` lists of [price, size].
        For a YES instrument: bids come from ``yes_dollars``, asks are derived
        as (1 - no_price) from ``no_dollars``.
        """
        bids = []
        asks = []

        # v3 dollar format (orderbook_fp container already stripped by caller,
        # but handle both shapes)
        yes_levels = book_data.get("yes_dollars", [])
        no_levels = book_data.get("no_dollars", [])

        if yes_levels or no_levels:
            # YES bids: people willing to buy YES at these prices
            for level in yes_levels:
                bids.append(OrderbookLevel(
                    price=float(level[0]),
                    size=float(level[1]),
                ))
            # YES asks: derived from NO bids (ask = 1 - no_bid)
            for level in no_levels:
                asks.append(OrderbookLevel(
                    price=1.0 - float(level[0]),
                    size=float(level[1]),
                ))
        else:
            # Legacy cents format
            for level in book_data.get("yes", {}).get("bids", []):
                bids.append(OrderbookLevel(
                    price=cents_to_normalized(float(level[0])),
                    size=float(level[1]),
                ))
            for level in book_data.get("yes", {}).get("asks", []):
                asks.append(OrderbookLevel(
                    price=cents_to_normalized(float(level[0])),
                    size=float(level[1]),
                ))

        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderbookSnapshot(instrument_id=ticker, bids=bids, asks=asks)

    @staticmethod
    def order_response_to_result(response: dict, requested_size: float = 0, requested_price: float = 0) -> OrderResult:
        """Convert Kalshi order response to OrderResult.

        Kalshi responses include ``remaining_count`` (contracts still unfilled)
        and ``count`` (total order size).  When ``remaining_count`` is missing
        from the response, we must NOT assume it is 0 (fully filled) — a
        resting order with no ``remaining_count`` field is still unfilled.
        We use the order ``status`` field as the source of truth.
        """
        order = response.get("order", response)
        order_id = order.get("order_id", "")
        status = order.get("status", "").lower()

        total = int(order.get("count", requested_size) or requested_size)

        # Determine filled count from status + remaining_count.
        # Only trust remaining_count arithmetic when the field is actually present.
        has_remaining = "remaining_count" in order
        if has_remaining:
            remaining = int(order["remaining_count"])
            filled = max(0, total - remaining)
        elif status == "executed":
            # Fully executed — all contracts filled
            filled = total
        elif status == "resting":
            # Sitting on the book — nothing filled yet
            filled = 0
        else:
            filled = 0

        if status == "resting":
            order_status = OrderStatus.OPEN
        elif status == "executed" or (has_remaining and remaining == 0):
            order_status = OrderStatus.FILLED
        elif filled > 0:
            order_status = OrderStatus.PARTIALLY_FILLED
        else:
            order_status = OrderStatus.PENDING

        avg_price_cents = float(order.get("avg_fill_price", 0) or 0)
        avg_price = cents_to_normalized(avg_price_cents) if avg_price_cents else requested_price

        return OrderResult(
            success=True,
            order_id=order_id,
            status=order_status,
            filled_size=float(filled),
            filled_price=avg_price,
            requested_size=requested_size,
            requested_price=requested_price,
        )
