"""
Position sizing components.

Different sizing strategies control capital allocation per trade.
All sizers work with any exchange through the unified Signal model.
"""

import logging
from abc import ABC, abstractmethod

from ..core.models import Signal

logger = logging.getLogger(__name__)


class PositionSizer(ABC):
    """Abstract base for position sizing strategies."""

    @abstractmethod
    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        """Calculate position size in USD. Returns 0 to skip."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FixedSizer(PositionSizer):
    """Fixed USD amount per trade."""

    def __init__(self, amount_usd: float = 50.0):
        self.amount = amount_usd

    @property
    def name(self) -> str:
        return "fixed"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        return min(self.amount, available_capital)


class PercentageSizer(PositionSizer):
    """Fixed percentage of capital per trade."""

    def __init__(self, percentage: float = 0.02):
        if not 0 < percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        self.percentage = percentage

    @property
    def name(self) -> str:
        return "percentage"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        return available_capital * self.percentage


class FixedFractionSizer(PositionSizer):
    """Fraction of capital with min/max constraints."""

    def __init__(self, fraction: float = 0.10, min_usd: float = 10.0, max_usd: float = 100.0):
        if not 0 < fraction <= 1:
            raise ValueError("Fraction must be between 0 and 1")
        self.fraction = fraction
        self.min_usd = min_usd
        self.max_usd = max_usd

    @property
    def name(self) -> str:
        return "fixed_fraction"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        base = available_capital * self.fraction
        size = max(self.min_usd, min(self.max_usd, base))
        return min(size, available_capital)


class KellySizer(PositionSizer):
    """
    Kelly Criterion sizing for binary outcomes.

    Uses fractional Kelly (default half) for safety.
    """

    def __init__(self, kelly_fraction: float = 0.5, min_edge: float = 0.02, max_kelly: float = 0.25):
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_kelly = max_kelly

    @property
    def name(self) -> str:
        return "kelly"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        if current_price <= 0 or current_price >= 1:
            return 0.0
        # Estimate edge from signal score
        edge = min(signal.score / 100, 0.5) * 0.1  # Conservative edge estimate
        if edge < self.min_edge:
            return 0.0
        # Kelly: f* = (p*b - q) / b where b = (1/price) - 1
        p = current_price + edge
        q = 1 - p
        b = (1.0 / current_price) - 1
        if b <= 0:
            return 0.0
        kelly = max(0, (p * b - q) / b)
        adjusted = min(kelly * self.kelly_fraction, self.max_kelly)
        return available_capital * adjusted


class SignalScaledSizer(PositionSizer):
    """Size scaled by signal strength."""

    def __init__(
        self, base_fraction: float = 0.02, reference_score: float = 50.0,
        scale_factor: float = 1.0, min_score: float = 20.0, max_multiplier: float = 3.0,
    ):
        self.base_fraction = base_fraction
        self.reference_score = reference_score
        self.scale_factor = scale_factor
        self.min_score = min_score
        self.max_multiplier = max_multiplier

    @property
    def name(self) -> str:
        return "signal_scaled"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        if signal.score < self.min_score:
            return 0.0
        multiplier = min((signal.score / self.reference_score) * self.scale_factor, self.max_multiplier)
        return available_capital * self.base_fraction * multiplier


class CompositeSizer(PositionSizer):
    """Takes the minimum of multiple sizers (safety)."""

    def __init__(self, sizers: list[PositionSizer]):
        if not sizers:
            raise ValueError("Need at least one sizer")
        self._sizers = sizers

    @property
    def name(self) -> str:
        return "composite"

    def calculate_size(self, signal: Signal, available_capital: float, current_price: float) -> float:
        sizes = [s.calculate_size(signal, available_capital, current_price) for s in self._sizers]
        return min(sizes) if sizes else 0.0
