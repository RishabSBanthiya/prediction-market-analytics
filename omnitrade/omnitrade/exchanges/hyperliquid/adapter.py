"""
Hyperliquid data adapter.

Converts between Hyperliquid API formats and unified models.
Handles perpetual-specific fields: leverage, funding, margin, liquidation.
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


class HyperliquidAdapter:
    """Converts Hyperliquid API responses to unified models."""

    @staticmethod
    def meta_to_instruments(meta: dict, asset_ctxs: list) -> list[Instrument]:
        """Convert meta + asset contexts to Instruments."""
        instruments = []
        universe = meta.get("universe", [])

        for i, asset in enumerate(universe):
            symbol = asset.get("name", "")
            max_leverage = int(asset.get("maxLeverage", 1))

            # Get current price/funding from asset context
            ctx = asset_ctxs[i] if i < len(asset_ctxs) else {}
            mark_price = float(ctx.get("markPx", 0))
            funding_rate = float(ctx.get("funding", 0))

            instruments.append(Instrument(
                instrument_id=symbol,
                exchange=ExchangeId.HYPERLIQUID,
                instrument_type=InstrumentType.PERPETUAL,
                name=f"{symbol}-PERP",
                price=mark_price,
                market_id=symbol,
                active=True,
                max_leverage=float(max_leverage),
                funding_rate=funding_rate,
                tick_size=float(asset.get("szDecimals", 0)),
                raw={"asset": asset, "ctx": ctx},
            ))

        return instruments

    @staticmethod
    def l2_to_snapshot(symbol: str, l2_data: dict) -> OrderbookSnapshot:
        """Convert L2 book to OrderbookSnapshot."""
        bids = []
        asks = []

        levels = l2_data.get("levels", [[], []])

        # Bids
        for level in levels[0] if len(levels) > 0 else []:
            bids.append(OrderbookLevel(
                price=float(level.get("px", 0)),
                size=float(level.get("sz", 0)),
            ))

        # Asks
        for level in levels[1] if len(levels) > 1 else []:
            asks.append(OrderbookLevel(
                price=float(level.get("px", 0)),
                size=float(level.get("sz", 0)),
            ))

        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderbookSnapshot(instrument_id=symbol, bids=bids, asks=asks)

    @staticmethod
    def order_response_to_result(response: dict, requested_size: float = 0, requested_price: float = 0) -> OrderResult:
        """Convert SDK order response to OrderResult."""
        status = response.get("status", "")

        if status == "ok":
            data = response.get("response", {}).get("data", {})
            statuses = data.get("statuses", [{}])

            if statuses:
                s = statuses[0]
                if "filled" in s:
                    filled = s["filled"]
                    return OrderResult(
                        success=True,
                        order_id=str(filled.get("oid", "")),
                        status=OrderStatus.FILLED,
                        filled_size=float(filled.get("totalSz", requested_size)),
                        filled_price=float(filled.get("avgPx", requested_price)),
                        requested_size=requested_size,
                        requested_price=requested_price,
                    )
                elif "resting" in s:
                    resting = s["resting"]
                    return OrderResult(
                        success=True,
                        order_id=str(resting.get("oid", "")),
                        status=OrderStatus.OPEN,
                        requested_size=requested_size,
                        requested_price=requested_price,
                    )
                elif "error" in s:
                    return OrderResult(
                        success=False,
                        error_message=s["error"],
                        requested_size=requested_size,
                        requested_price=requested_price,
                    )

        return OrderResult(
            success=False,
            error_message=response.get("response", str(response)),
            requested_size=requested_size,
            requested_price=requested_price,
        )

    @staticmethod
    def user_state_to_positions(state: dict) -> list[ExchangePosition]:
        """Convert user state to ExchangePositions."""
        positions = []

        for pos in state.get("assetPositions", []):
            p = pos.get("position", {})
            symbol = p.get("coin", "")
            size = abs(float(p.get("szi", 0)))

            if size == 0:
                continue

            entry = float(p.get("entryPx", 0))
            unrealized = float(p.get("unrealizedPnl", 0))
            liq_price = p.get("liquidationPx")
            leverage_val = p.get("leverage", {})
            lev = float(leverage_val.get("value", 1)) if isinstance(leverage_val, dict) else float(leverage_val or 1)

            positions.append(ExchangePosition(
                instrument_id=symbol,
                exchange=ExchangeId.HYPERLIQUID,
                side=Side.BUY if float(p.get("szi", 0)) > 0 else Side.SELL,
                size=size,
                entry_price=entry,
                unrealized_pnl=unrealized,
                liquidation_price=float(liq_price) if liq_price else None,
                leverage=lev,
            ))

        return positions

    @staticmethod
    def user_state_to_balance(state: dict) -> AccountBalance:
        """Convert user state to AccountBalance."""
        margin = state.get("marginSummary", state.get("crossMarginSummary", {}))

        return AccountBalance(
            exchange=ExchangeId.HYPERLIQUID,
            total_equity=float(margin.get("accountValue", 0)),
            available_balance=float(margin.get("totalRawUsd", 0)),
            unrealized_pnl=float(margin.get("totalNtlPos", 0)),
            currency="USD",
        )
