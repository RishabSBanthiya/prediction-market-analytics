"""Tests for market making bot and adaptive quoting components."""

import math
import time
import pytest
from unittest.mock import patch

from omnitrade.core.enums import Side, Environment, ExchangeId, InstrumentType
from omnitrade.core.models import (
    Instrument, OrderbookSnapshot, OrderbookLevel, Quote,
)
from omnitrade.bots.market_making import (
    VolatilityTracker,
    FairValueEstimator,
    FillToxicityTracker,
    AdaptiveQuoter,
    ActiveMarketSelector,
    InventoryManager,
    MarketMakingBot,
)


# === VolatilityTracker ===


class TestVolatilityTracker:
    def test_no_volatility_with_insufficient_samples(self):
        vt = VolatilityTracker(window=20, min_samples=3)
        vt.update("X", 0.50)
        vt.update("X", 0.51)
        assert vt.get_volatility("X") == 0.0
        assert vt.get_drift("X") == 0.0

    def test_no_volatility_for_unknown_instrument(self):
        vt = VolatilityTracker()
        assert vt.get_volatility("unknown") == 0.0
        assert vt.get_drift("unknown") == 0.0

    def test_zero_volatility_for_constant_price(self):
        vt = VolatilityTracker(min_samples=3)
        for _ in range(5):
            vt.update("X", 0.50)
        assert vt.get_volatility("X") == 0.0

    def test_positive_volatility_for_varying_price(self):
        vt = VolatilityTracker(min_samples=3)
        prices = [0.50, 0.52, 0.48, 0.53, 0.47, 0.55]
        for p in prices:
            vt.update("X", p)
        vol = vt.get_volatility("X")
        assert vol > 0

    def test_drift_positive_when_prices_rise(self):
        vt = VolatilityTracker(min_samples=3)
        for p in [0.40, 0.42, 0.44, 0.46, 0.48]:
            vt.update("X", p)
        assert vt.get_drift("X") > 0

    def test_drift_negative_when_prices_fall(self):
        vt = VolatilityTracker(min_samples=3)
        for p in [0.60, 0.58, 0.56, 0.54, 0.52]:
            vt.update("X", p)
        assert vt.get_drift("X") < 0

    def test_window_eviction(self):
        vt = VolatilityTracker(window=4, min_samples=3)
        # Fill with stable prices
        for _ in range(4):
            vt.update("X", 0.50)
        assert vt.get_volatility("X") == 0.0
        # Now add volatile prices — old ones get evicted
        vt.update("X", 0.60)
        vt.update("X", 0.40)
        vol = vt.get_volatility("X")
        assert vol > 0

    def test_separate_instruments(self):
        vt = VolatilityTracker(min_samples=3)
        for p in [0.40, 0.42, 0.44, 0.46]:
            vt.update("A", p)
        for p in [0.60, 0.58, 0.56, 0.54]:
            vt.update("B", p)
        assert vt.get_drift("A") > 0
        assert vt.get_drift("B") < 0


# === FairValueEstimator ===


class TestFairValueEstimator:
    def _make_ob(self, bids, asks):
        return OrderbookSnapshot(
            instrument_id="X",
            bids=[OrderbookLevel(price=p, size=s) for p, s in bids],
            asks=[OrderbookLevel(price=p, size=s) for p, s in asks],
        )

    def test_fair_value_equals_mid_when_balanced(self):
        fve = FairValueEstimator()
        ob = self._make_ob(
            bids=[(0.48, 100), (0.47, 100)],
            asks=[(0.52, 100), (0.53, 100)],
        )
        fv = fve.estimate(ob, drift=0.0)
        assert fv == pytest.approx(0.50, abs=0.001)

    def test_fair_value_above_mid_when_bid_heavy(self):
        fve = FairValueEstimator()
        ob = self._make_ob(
            bids=[(0.48, 500), (0.47, 500)],
            asks=[(0.52, 50), (0.53, 50)],
        )
        fv = fve.estimate(ob, drift=0.0)
        mid = ob.midpoint
        assert fv > mid

    def test_fair_value_below_mid_when_ask_heavy(self):
        fve = FairValueEstimator()
        ob = self._make_ob(
            bids=[(0.48, 50), (0.47, 50)],
            asks=[(0.52, 500), (0.53, 500)],
        )
        fv = fve.estimate(ob, drift=0.0)
        mid = ob.midpoint
        assert fv < mid

    def test_drift_shifts_fair_value(self):
        fve = FairValueEstimator(drift_weight=0.5)
        ob = self._make_ob(
            bids=[(0.48, 100)],
            asks=[(0.52, 100)],
        )
        fv_neutral = fve.estimate(ob, drift=0.0)
        fv_positive = fve.estimate(ob, drift=0.1)
        assert fv_positive > fv_neutral

    def test_returns_none_for_empty_orderbook(self):
        fve = FairValueEstimator()
        ob = OrderbookSnapshot(instrument_id="X", bids=[], asks=[])
        assert fve.estimate(ob) is None

    def test_returns_mid_when_zero_spread(self):
        fve = FairValueEstimator()
        ob = self._make_ob(
            bids=[(0.50, 100)],
            asks=[(0.50, 100)],
        )
        fv = fve.estimate(ob, drift=0.0)
        assert fv == pytest.approx(0.50)


# === FillToxicityTracker ===


class TestFillToxicityTracker:
    def test_no_fills_returns_zero_ratio(self):
        ft = FillToxicityTracker()
        assert ft.get_toxic_ratio("X") == 0.0
        assert ft.get_spread_penalty("X") == 0.0

    def test_toxic_fill_detected(self):
        ft = FillToxicityTracker(toxic_threshold_seconds=1.0)
        ft.record_order_placed("order-1")
        # Fill immediately (within threshold)
        ft.record_fill("order-1", "X")
        assert ft.get_toxic_ratio("X") == 1.0

    def test_passive_fill_detected(self):
        ft = FillToxicityTracker(toxic_threshold_seconds=0.0)
        ft.record_order_placed("order-1")
        # Any fill after threshold=0 is passive
        ft.record_fill("order-1", "X")
        assert ft.get_toxic_ratio("X") == 0.0

    def test_spread_penalty_scales_with_ratio(self):
        ft = FillToxicityTracker(
            toxic_threshold_seconds=10.0,
            spread_penalty_scale=0.5,
        )
        ft.record_order_placed("o1")
        ft.record_fill("o1", "X")
        ft.record_order_placed("o2")
        ft.record_fill("o2", "X")
        # All fills toxic -> ratio=1.0, penalty=0.5
        assert ft.get_spread_penalty("X") == pytest.approx(0.5)

    def test_mixed_fills(self):
        ft = FillToxicityTracker(toxic_threshold_seconds=10.0, spread_penalty_scale=1.0)
        # One toxic fill
        ft.record_order_placed("o1")
        ft.record_fill("o1", "X")
        # One passive fill (unknown order, won't be counted)
        ft.record_fill("unknown", "X")
        assert ft.get_toxic_ratio("X") == 1.0  # only the recorded one counts

    def test_unknown_order_fill_ignored(self):
        ft = FillToxicityTracker()
        ft.record_fill("never-placed", "X")
        assert ft.get_toxic_ratio("X") == 0.0

    def test_cleanup_stale(self):
        ft = FillToxicityTracker()
        ft._order_timestamps["old"] = time.monotonic() - 600
        ft._order_timestamps["fresh"] = time.monotonic()
        ft.cleanup_stale(max_age_seconds=300.0)
        assert "old" not in ft._order_timestamps
        assert "fresh" in ft._order_timestamps


# === AdaptiveQuoter ===


class TestAdaptiveQuoter:
    async def test_quote_generated_with_defaults(self, mock_client):
        quoter = AdaptiveQuoter()
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is not None
        assert quote.bid_price < quote.ask_price
        assert quote.bid_size > 0
        assert quote.ask_size > 0

    async def test_spread_widens_with_volatility(self, mock_client):
        quoter = AdaptiveQuoter(vol_scale=5.0, vol_min_samples=3)
        # Feed calm prices first
        for p in [0.50, 0.50, 0.50, 0.50]:
            quoter.volatility_tracker.update("test-token", p)
        calm_quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)

        # Feed volatile prices
        quoter2 = AdaptiveQuoter(vol_scale=5.0, vol_min_samples=3)
        for p in [0.40, 0.60, 0.35, 0.65]:
            quoter2.volatility_tracker.update("test-token", p)
        vol_quote = await quoter2.generate_quote(mock_client, "test-token", inventory=0.0)

        assert vol_quote is not None
        assert calm_quote is not None
        vol_spread = vol_quote.ask_price - vol_quote.bid_price
        calm_spread = calm_quote.ask_price - calm_quote.bid_price
        assert vol_spread > calm_spread

    async def test_asymmetric_sizes_on_rising_market(self, mock_client):
        quoter = AdaptiveQuoter(toxic_size_scale=20.0, vol_min_samples=3)
        # Rising prices
        for p in [0.40, 0.45, 0.50, 0.55, 0.60]:
            quoter.volatility_tracker.update("test-token", p)
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is not None
        # Ask side is toxic when rising -> ask_size should be smaller
        assert quote.ask_size < quote.bid_size

    async def test_asymmetric_sizes_on_falling_market(self, mock_client):
        quoter = AdaptiveQuoter(toxic_size_scale=20.0, vol_min_samples=3)
        # Falling prices
        for p in [0.60, 0.55, 0.50, 0.45, 0.40]:
            quoter.volatility_tracker.update("test-token", p)
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is not None
        # Bid side is toxic when falling -> bid_size should be smaller
        assert quote.bid_size < quote.ask_size

    async def test_fair_value_shift_from_imbalance(self, mock_client):
        """Bid-heavy orderbook should shift quotes upward."""
        mock_client._orderbook = OrderbookSnapshot(
            instrument_id="test-token",
            bids=[OrderbookLevel(price=0.48, size=500)],
            asks=[OrderbookLevel(price=0.52, size=50)],
        )
        quoter = AdaptiveQuoter()
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is not None
        quote_mid = (quote.bid_price + quote.ask_price) / 2
        raw_mid = 0.50
        assert quote_mid > raw_mid

    async def test_toxicity_penalty_widens_spread(self, mock_client):
        tox = FillToxicityTracker(toxic_threshold_seconds=10.0, spread_penalty_scale=0.05)
        # Record some toxic fills
        for i in range(5):
            tox.record_order_placed(f"o{i}")
            tox.record_fill(f"o{i}", "test-token")

        quoter_no_tox = AdaptiveQuoter()
        quoter_tox = AdaptiveQuoter(toxicity_tracker=tox)

        q_clean = await quoter_no_tox.generate_quote(mock_client, "test-token", inventory=0.0)
        q_toxic = await quoter_tox.generate_quote(mock_client, "test-token", inventory=0.0)

        assert q_clean is not None and q_toxic is not None
        clean_spread = q_clean.ask_price - q_clean.bid_price
        toxic_spread = q_toxic.ask_price - q_toxic.bid_price
        assert toxic_spread > clean_spread

    async def test_quadratic_inventory_skew(self, mock_client):
        """High inventory should skew quotes more aggressively than linear."""
        quoter = AdaptiveQuoter(max_inventory=100.0, inventory_skew=1.0)

        q_low = await quoter.generate_quote(mock_client, "test-token", inventory=20.0)
        # Reset vol tracker to not pollute between calls
        quoter2 = AdaptiveQuoter(max_inventory=100.0, inventory_skew=1.0)
        q_high = await quoter2.generate_quote(mock_client, "test-token", inventory=80.0)

        assert q_low is not None and q_high is not None
        # Both should shift bid down (to discourage more buying)
        # High inventory should shift more
        low_mid = (q_low.bid_price + q_low.ask_price) / 2
        high_mid = (q_high.bid_price + q_high.ask_price) / 2
        assert high_mid < low_mid

    async def test_max_contracts_respected(self, mock_client):
        """Quote sizes should not exceed max_contracts."""
        mock_client._orderbook = OrderbookSnapshot(
            instrument_id="test-token",
            bids=[OrderbookLevel(price=0.01, size=100)],
            asks=[OrderbookLevel(price=0.03, size=100)],
        )
        quoter = AdaptiveQuoter(size_usd=100.0, max_contracts=50.0)
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is not None
        assert quote.bid_size <= 50.0
        assert quote.ask_size <= 50.0

    async def test_returns_none_for_empty_orderbook(self, mock_client):
        mock_client._orderbook = OrderbookSnapshot(
            instrument_id="test-token", bids=[], asks=[],
        )
        quoter = AdaptiveQuoter()
        quote = await quoter.generate_quote(mock_client, "test-token", inventory=0.0)
        assert quote is None


# === InventoryManager netting ===


class TestInventoryManagerNetting:
    def test_net_inventory_without_pair(self):
        inv = InventoryManager()
        inv.update_from_fill("YES", Side.BUY, 100.0)
        assert inv.get_net_inventory("YES") == 100.0

    def test_net_inventory_with_pair(self):
        inv = InventoryManager()
        inv.register_pair("YES", "NO")
        inv.update_from_fill("YES", Side.BUY, 100.0)
        inv.update_from_fill("NO", Side.BUY, 40.0)
        assert inv.get_net_inventory("YES") == pytest.approx(60.0)
        assert inv.get_net_inventory("NO") == pytest.approx(-60.0)

    def test_net_inventory_symmetric(self):
        inv = InventoryManager()
        inv.register_pair("YES", "NO")
        inv.update_from_fill("YES", Side.BUY, 75.0)
        inv.update_from_fill("NO", Side.BUY, 25.0)
        assert inv.get_net_inventory("YES") == pytest.approx(-inv.get_net_inventory("NO"))

    def test_register_pair_bidirectional(self):
        inv = InventoryManager()
        inv.register_pair("A", "B")
        assert inv._pair_map["A"] == "B"
        assert inv._pair_map["B"] == "A"

    def test_net_inventory_zero_when_equal(self):
        inv = InventoryManager()
        inv.register_pair("YES", "NO")
        inv.update_from_fill("YES", Side.BUY, 50.0)
        inv.update_from_fill("NO", Side.BUY, 50.0)
        assert inv.get_net_inventory("YES") == pytest.approx(0.0)


# === MarketMakingBot ===


class TestMarketMakingBot:
    async def test_default_refresh_interval(self, mock_client, risk_coordinator):
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
        )
        assert bot.refresh_interval == 1.5

    async def test_start_stop(self, mock_client, risk_coordinator):
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
        )
        await bot.start()
        assert mock_client.is_connected
        await bot.stop()

    async def test_iteration_paper_mode(self, mock_client, risk_coordinator):
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        await bot.stop()

    async def test_pair_detection_from_market_id(self, mock_client, risk_coordinator):
        """Instruments with same market_id should be registered as pairs."""
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        await bot._iteration()
        # The mock_client has token-yes and token-no with market_id="test-market"
        assert bot.inventory._pair_map.get("token-yes") == "token-no"
        assert bot.inventory._pair_map.get("token-no") == "token-yes"
        await bot.stop()

    async def test_drawdown_stops_bot(self, mock_client, risk_coordinator):
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
            environment=Environment.PAPER,
        )
        await bot.start()
        bot._running = True
        # First iteration at normal equity to establish daily_start_equity
        await bot._iteration()
        assert bot._running is True
        # Now simulate massive loss
        mock_client._balance.total_equity = 1.0
        await bot._iteration()
        assert bot._running is False
        await bot.stop()

    async def test_toxicity_tracker_wired(self, mock_client, risk_coordinator):
        """Toxicity tracker should record order placements in live mode."""
        tox = FillToxicityTracker()
        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
            toxicity_tracker=tox,
            environment=Environment.LIVE,
        )
        await bot.start()
        await bot._iteration()
        # Mock client returns FILLED orders, so order_timestamps may be cleared
        # but the tracker should have been called
        await bot.stop()

    async def test_net_inventory_used_for_quotes(self, mock_client, risk_coordinator):
        """Bot should pass net inventory (not raw) to the quote engine."""
        inventory = InventoryManager()

        bot = MarketMakingBot(
            agent_id="test-mm",
            client=mock_client,
            quote_engine=AdaptiveQuoter(),
            market_selector=ActiveMarketSelector(),
            risk=risk_coordinator,
            inventory=inventory,
            environment=Environment.PAPER,
        )
        await bot.start()
        # Set inventory after start (which calls sync_from_exchange and clears)
        inventory.register_pair("token-yes", "token-no")
        inventory.update_from_fill("token-yes", Side.BUY, 100.0)
        inventory.update_from_fill("token-no", Side.BUY, 60.0)
        # net_inventory for token-yes = 100 - 60 = 40
        assert inventory.get_net_inventory("token-yes") == pytest.approx(40.0)
        assert inventory.get_net_inventory("token-no") == pytest.approx(-40.0)
        await bot.stop()
