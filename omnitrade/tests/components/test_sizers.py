"""Tests for position sizers."""

import pytest
from omnitrade.core.enums import SignalDirection, ExchangeId
from omnitrade.core.models import Signal
from omnitrade.components.sizers import (
    FixedSizer, PercentageSizer, FixedFractionSizer,
    KellySizer, SignalScaledSizer, CompositeSizer,
)


@pytest.fixture
def signal():
    return Signal(
        instrument_id="test",
        direction=SignalDirection.LONG,
        score=50.0,
        source="test",
        price=0.60,
    )


class TestFixedSizer:
    def test_basic(self, signal):
        s = FixedSizer(amount_usd=100.0)
        assert s.calculate_size(signal, 1000.0, 0.60) == 100.0

    def test_capped_by_capital(self, signal):
        s = FixedSizer(amount_usd=100.0)
        assert s.calculate_size(signal, 50.0, 0.60) == 50.0

    def test_name(self):
        assert FixedSizer().name == "fixed"

    def test_default_amount(self, signal):
        s = FixedSizer()
        assert s.calculate_size(signal, 1000.0, 0.60) == 50.0

    def test_zero_capital(self, signal):
        s = FixedSizer(amount_usd=100.0)
        assert s.calculate_size(signal, 0.0, 0.60) == 0.0


class TestPercentageSizer:
    def test_basic(self, signal):
        s = PercentageSizer(percentage=0.10)
        assert s.calculate_size(signal, 1000.0, 0.60) == 100.0

    def test_invalid_percentage_above(self):
        with pytest.raises(ValueError):
            PercentageSizer(percentage=1.5)

    def test_invalid_percentage_zero(self):
        with pytest.raises(ValueError):
            PercentageSizer(percentage=0.0)

    def test_invalid_percentage_negative(self):
        with pytest.raises(ValueError):
            PercentageSizer(percentage=-0.1)

    def test_name(self):
        assert PercentageSizer().name == "percentage"

    def test_full_percentage(self, signal):
        s = PercentageSizer(percentage=1.0)
        assert s.calculate_size(signal, 500.0, 0.60) == 500.0


class TestFixedFractionSizer:
    def test_within_bounds(self, signal):
        s = FixedFractionSizer(fraction=0.10, min_usd=10.0, max_usd=200.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert 10.0 <= result <= 200.0

    def test_min_floor(self, signal):
        s = FixedFractionSizer(fraction=0.001, min_usd=10.0, max_usd=200.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 10.0

    def test_max_cap(self, signal):
        s = FixedFractionSizer(fraction=0.50, min_usd=10.0, max_usd=100.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 100.0

    def test_capped_by_available_capital(self, signal):
        s = FixedFractionSizer(fraction=0.50, min_usd=10.0, max_usd=200.0)
        result = s.calculate_size(signal, 30.0, 0.60)
        # fraction=0.50 * 30 = 15, max(10, min(200, 15)) = 15, min(15, 30) = 15
        assert result == 15.0

    def test_name(self):
        assert FixedFractionSizer().name == "fixed_fraction"

    def test_invalid_fraction(self):
        with pytest.raises(ValueError):
            FixedFractionSizer(fraction=0.0)


class TestKellySizer:
    def test_name(self):
        assert KellySizer().name == "kelly"

    def test_zero_price(self, signal):
        signal.price = 0.0
        s = KellySizer()
        assert s.calculate_size(signal, 1000.0, 0.0) == 0.0

    def test_one_price(self, signal):
        s = KellySizer()
        assert s.calculate_size(signal, 1000.0, 1.0) == 0.0

    def test_positive_edge_returns_nonzero(self, signal):
        signal.score = 80.0
        s = KellySizer(kelly_fraction=0.5, min_edge=0.01)
        result = s.calculate_size(signal, 1000.0, 0.50)
        assert result > 0

    def test_low_score_below_min_edge(self, signal):
        signal.score = 1.0
        s = KellySizer(min_edge=0.05)
        result = s.calculate_size(signal, 1000.0, 0.50)
        assert result == 0.0

    def test_max_kelly_cap(self, signal):
        signal.score = 100.0
        s = KellySizer(kelly_fraction=1.0, min_edge=0.01, max_kelly=0.05)
        result = s.calculate_size(signal, 10000.0, 0.50)
        assert result <= 10000.0 * 0.05


class TestSignalScaledSizer:
    def test_reference_score(self, signal):
        s = SignalScaledSizer(base_fraction=0.02, reference_score=50.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 20.0  # 2% * 1x

    def test_below_min_score(self, signal):
        signal.score = 5.0
        s = SignalScaledSizer(min_score=20.0)
        assert s.calculate_size(signal, 1000.0, 0.60) == 0.0

    def test_max_multiplier(self, signal):
        signal.score = 500.0  # Way above reference
        s = SignalScaledSizer(base_fraction=0.02, reference_score=50.0, max_multiplier=3.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 60.0  # 2% * 3x (capped)

    def test_name(self):
        assert SignalScaledSizer().name == "signal_scaled"

    def test_double_reference_score(self, signal):
        signal.score = 100.0
        s = SignalScaledSizer(base_fraction=0.02, reference_score=50.0, max_multiplier=5.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 40.0  # 2% * 2x

    def test_scale_factor(self, signal):
        signal.score = 50.0
        s = SignalScaledSizer(base_fraction=0.02, reference_score=50.0, scale_factor=2.0, max_multiplier=5.0)
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 40.0  # 2% * (50/50 * 2.0) = 2% * 2


class TestCompositeSizer:
    def test_takes_minimum(self, signal):
        s = CompositeSizer([
            FixedSizer(amount_usd=100.0),
            PercentageSizer(percentage=0.05),  # 50
        ])
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 50.0  # min(100, 50)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            CompositeSizer([])

    def test_name(self):
        s = CompositeSizer([FixedSizer()])
        assert s.name == "composite"

    def test_single_sizer(self, signal):
        s = CompositeSizer([FixedSizer(amount_usd=75.0)])
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 75.0

    def test_three_sizers(self, signal):
        s = CompositeSizer([
            FixedSizer(amount_usd=200.0),
            PercentageSizer(percentage=0.10),  # 100
            FixedFractionSizer(fraction=0.05, min_usd=10.0, max_usd=80.0),  # 50, capped at 80 -> 50
        ])
        result = s.calculate_size(signal, 1000.0, 0.60)
        assert result == 50.0  # min(200, 100, 50)
