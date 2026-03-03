"""
Signal sources for trading bots.

Each signal source implements the SignalSource ABC and produces
Signal objects that the bot acts on.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..core.models import Signal, OrderbookSnapshot
from ..core.enums import SignalDirection
from ..exchanges.base import ExchangeClient


class SignalSource(ABC):
    """Abstract base for signal generation."""

    @abstractmethod
    async def generate(self, client: ExchangeClient) -> list[Signal]:
        """Generate trading signals. Called each bot iteration."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class MidpointDeviationSignal(SignalSource):
    """
    Simple signal: go long when price is below fair value,
    short when above. Useful for mean-reversion on binary markets.
    """

    def __init__(self, fair_value: float = 0.5, min_deviation: float = 0.05):
        self.fair_value = fair_value
        self.min_deviation = min_deviation

    @property
    def name(self) -> str:
        return "midpoint_deviation"

    async def generate(self, client: ExchangeClient) -> list[Signal]:
        signals = []
        instruments = await client.get_instruments(active_only=True)
        for inst in instruments:
            mid = await client.get_midpoint(inst.instrument_id)
            if mid is None:
                continue
            deviation = mid - self.fair_value
            if abs(deviation) < self.min_deviation:
                continue
            direction = SignalDirection.SHORT if deviation > 0 else SignalDirection.LONG
            score = abs(deviation) * 100  # Scale to 0-50 range
            signals.append(Signal(
                instrument_id=inst.instrument_id,
                direction=direction,
                score=score,
                source=self.name,
                price=mid,
                market_id=inst.market_id,
                exchange=client.exchange_id,
            ))
        return signals
