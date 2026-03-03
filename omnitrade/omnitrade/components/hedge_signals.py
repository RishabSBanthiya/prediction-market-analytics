"""
Cross-exchange hedge signal sources.

Generates multi-leg signals for hedged strategies across platforms.
Example: Buy "BTC Up" binary on Polymarket + Short BTC perp on Hyperliquid.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import ExchangeId, SignalDirection, InstrumentType
from ..core.models import MultiLegSignal, SignalLeg, Instrument
from ..exchanges.base import ExchangeClient
from ..bots.cross_exchange import CrossExchangeSignalSource

logger = logging.getLogger(__name__)

# Map Polymarket binary market keywords to Hyperliquid perp symbols
ASSET_KEYWORD_MAP = {
    "BTC": ["bitcoin", "btc"],
    "ETH": ["ethereum", "eth"],
    "SOL": ["solana", "sol"],
    "DOGE": ["dogecoin", "doge"],
    "XRP": ["xrp", "ripple"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "ARB": ["arbitrum", "arb"],
    "OP": ["optimism"],
    "LINK": ["chainlink", "link"],
}


def match_perp_symbol(market_name: str) -> Optional[str]:
    """
    Try to match a Polymarket/Kalshi market name to a Hyperliquid perp symbol.

    Examples:
        "Will Bitcoin reach $100k?" -> "BTC"
        "ETH price above $4000 by March?" -> "ETH"
        "Solana ATH in Q1?" -> "SOL"
    """
    name_lower = market_name.lower()
    for symbol, keywords in ASSET_KEYWORD_MAP.items():
        for kw in keywords:
            if kw in name_lower:
                return symbol
    return None


def detect_direction(market_name: str) -> Optional[SignalDirection]:
    """
    Detect directional intent from a binary market name.

    "Will BTC go above $100k?" -> LONG (betting on price going up)
    "Will BTC drop below $50k?" -> SHORT (betting on price going down)
    """
    name_lower = market_name.lower()

    up_keywords = ["above", "reach", "exceed", "hit", "ath", "high", "rise", "up",
                   "over", "surpass", "break"]
    down_keywords = ["below", "drop", "fall", "crash", "low", "under", "decline",
                     "down", "sink"]

    up_count = sum(1 for kw in up_keywords if kw in name_lower)
    down_count = sum(1 for kw in down_keywords if kw in name_lower)

    if up_count > down_count:
        return SignalDirection.LONG
    elif down_count > up_count:
        return SignalDirection.SHORT
    return None


class BinaryPerpHedgeSignal(CrossExchangeSignalSource):
    """
    Generates hedged signals: binary outcome + perp hedge.

    Strategy logic:
    1. Scan Polymarket/Kalshi for crypto price binary markets
    2. If a binary is near resolution (e.g., YES token at 0.90+),
       buy the binary AND short the perp as a delta hedge
    3. If the binary resolves YES, we profit on the binary and close the perp
    4. If it resolves NO, the perp hedge limits downside

    The hedge ratio is configurable (default: 50% of binary notional).
    """

    def __init__(
        self,
        binary_exchange: ExchangeId = ExchangeId.POLYMARKET,
        hedge_exchange: ExchangeId = ExchangeId.HYPERLIQUID,
        min_binary_price: float = 0.70,
        max_binary_price: float = 0.95,
        min_score: float = 30.0,
        hedge_ratio: float = 0.5,
        hedge_leverage: float = 2.0,
    ):
        self.binary_exchange = binary_exchange
        self.hedge_exchange = hedge_exchange
        self.min_binary_price = min_binary_price
        self.max_binary_price = max_binary_price
        self.min_score = min_score
        self.hedge_ratio = hedge_ratio
        self.hedge_leverage = hedge_leverage

    @property
    def name(self) -> str:
        return "binary_perp_hedge"

    async def generate(
        self, clients: dict[ExchangeId, ExchangeClient]
    ) -> list[MultiLegSignal]:
        binary_client = clients.get(self.binary_exchange)
        hedge_client = clients.get(self.hedge_exchange)

        if not binary_client or not hedge_client:
            return []

        signals: list[MultiLegSignal] = []

        # Get binary instruments
        try:
            instruments = await binary_client.get_instruments(active_only=True)
        except Exception as e:
            logger.warning(f"Failed to fetch binary instruments: {e}")
            return []

        # Get available perps for matching
        try:
            perps = await hedge_client.get_instruments(active_only=True)
            perp_symbols = {p.instrument_id for p in perps}
        except Exception as e:
            logger.warning(f"Failed to fetch perp instruments: {e}")
            return []

        for inst in instruments:
            if inst.instrument_type not in (
                InstrumentType.BINARY_OUTCOME,
                InstrumentType.EVENT_CONTRACT,
            ):
                continue

            # Price filter
            if not (self.min_binary_price <= inst.price <= self.max_binary_price):
                continue

            # Try to match to a perp
            perp_symbol = match_perp_symbol(inst.name)
            if perp_symbol is None or perp_symbol not in perp_symbols:
                continue

            # Detect direction
            direction = detect_direction(inst.name)
            if direction is None:
                continue

            # Score: higher binary price = higher conviction = higher score
            score = inst.price * 100  # 70-95 range

            if score < self.min_score:
                continue

            # Calculate edge: binary at 0.90 pays $1 if resolved YES
            # Edge = (1 - price) - expected_loss_from_hedge
            edge_bps = (1.0 - inst.price) * 10000 * 0.5  # Conservative 50% of raw edge

            # Determine hedge direction (opposite of the binary bet)
            if direction == SignalDirection.LONG:
                # Binary bets on price going UP -> hedge by shorting perp
                hedge_direction = SignalDirection.SHORT
            else:
                # Binary bets on price going DOWN -> hedge by longing perp
                hedge_direction = SignalDirection.LONG

            # Get perp price for the hedge leg
            try:
                perp_mid = await hedge_client.get_midpoint(perp_symbol)
            except Exception:
                perp_mid = 0.0

            signal = MultiLegSignal(
                legs=[
                    SignalLeg(
                        exchange=self.binary_exchange,
                        instrument_id=inst.instrument_id,
                        direction=SignalDirection.LONG,  # Always buy the binary
                        weight=1.0,
                        price=inst.price,
                        metadata={"market_name": inst.name, "outcome": inst.outcome},
                    ),
                    SignalLeg(
                        exchange=self.hedge_exchange,
                        instrument_id=perp_symbol,
                        direction=hedge_direction,
                        weight=self.hedge_ratio,
                        price=perp_mid or 0.0,
                        leverage=self.hedge_leverage,
                        metadata={"hedge_type": "delta"},
                    ),
                ],
                strategy_type="binary_perp_hedge",
                score=score,
                source=self.name,
                edge_bps=edge_bps,
                metadata={
                    "binary_price": inst.price,
                    "perp_symbol": perp_symbol,
                    "direction": direction.value,
                },
            )
            signals.append(signal)

        # Sort by score descending
        signals.sort(key=lambda s: s.score, reverse=True)
        logger.info(f"BinaryPerpHedge: found {len(signals)} opportunities")
        return signals


class CrossExchangeArbSignal(CrossExchangeSignalSource):
    """
    Cross-exchange arbitrage signal.

    Scans for the same event priced differently on Polymarket vs Kalshi.
    If combined YES probabilities sum to < 1.0, there's an arb opportunity.
    """

    def __init__(
        self,
        min_edge_bps: float = 50.0,
        max_price: float = 0.95,
    ):
        self.min_edge_bps = min_edge_bps
        self.max_price = max_price

    @property
    def name(self) -> str:
        return "cross_exchange_arb"

    async def generate(
        self, clients: dict[ExchangeId, ExchangeClient]
    ) -> list[MultiLegSignal]:
        poly_client = clients.get(ExchangeId.POLYMARKET)
        kalshi_client = clients.get(ExchangeId.KALSHI)

        if not poly_client or not kalshi_client:
            return []

        signals: list[MultiLegSignal] = []

        try:
            poly_instruments = await poly_client.get_instruments(active_only=True)
            kalshi_instruments = await kalshi_client.get_instruments(active_only=True)
        except Exception as e:
            logger.warning(f"Failed to fetch instruments for arb scan: {e}")
            return []

        # Build name-based index for matching
        # This is a simplistic matcher - production would use fuzzy matching
        kalshi_by_name: dict[str, Instrument] = {}
        for ki in kalshi_instruments:
            # Normalize name for matching
            clean = re.sub(r"[^a-z0-9 ]", "", ki.name.lower().split(" - ")[0])
            kalshi_by_name[clean] = ki

        for pi in poly_instruments:
            clean = re.sub(r"[^a-z0-9 ]", "", pi.name.lower().split(" - ")[0])
            ki = kalshi_by_name.get(clean)
            if ki is None:
                continue

            # Check for price discrepancy
            if pi.outcome != ki.outcome:
                continue

            price_diff = abs(pi.price - ki.price)
            edge_bps = price_diff * 10000

            if edge_bps < self.min_edge_bps:
                continue

            # Buy cheap, sell expensive
            if pi.price < ki.price:
                buy_exchange, buy_inst = ExchangeId.POLYMARKET, pi
                sell_exchange, sell_inst = ExchangeId.KALSHI, ki
            else:
                buy_exchange, buy_inst = ExchangeId.KALSHI, ki
                sell_exchange, sell_inst = ExchangeId.POLYMARKET, pi

            signal = MultiLegSignal(
                legs=[
                    SignalLeg(
                        exchange=buy_exchange,
                        instrument_id=buy_inst.instrument_id,
                        direction=SignalDirection.LONG,
                        weight=1.0,
                        price=buy_inst.price,
                    ),
                    SignalLeg(
                        exchange=sell_exchange,
                        instrument_id=sell_inst.instrument_id,
                        direction=SignalDirection.SHORT,
                        weight=1.0,
                        price=sell_inst.price,
                    ),
                ],
                strategy_type="cross_exchange_arb",
                score=edge_bps,
                source=self.name,
                edge_bps=edge_bps,
                metadata={
                    "buy_price": buy_inst.price,
                    "sell_price": sell_inst.price,
                    "market_name": pi.name,
                },
            )
            signals.append(signal)

        signals.sort(key=lambda s: s.score, reverse=True)
        logger.info(f"CrossExchangeArb: found {len(signals)} opportunities")
        return signals
