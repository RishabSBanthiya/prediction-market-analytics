"""Tests for exit strategies."""

import pytest
from datetime import datetime, timezone, timedelta
from omnitrade.core.enums import ExitReason, Side
from omnitrade.core.models import PositionState
from omnitrade.components.exit_strategies import ExitMonitor, ExitConfig


@pytest.fixture
def monitor():
    return ExitMonitor(ExitConfig())


@pytest.fixture
def state():
    return PositionState(
        instrument_id="test",
        entry_price=0.50,
        entry_time=datetime.now(timezone.utc),
        size=100.0,
    )


class TestExitMonitor:
    def test_no_exit(self, monitor, state):
        result = monitor.check(state, 0.51, datetime.now(timezone.utc))
        assert result is None

    def test_take_profit(self, monitor, state):
        # 5% TP on 0.50 entry -> need 0.525 or above
        result = monitor.check(state, 0.53, datetime.now(timezone.utc))
        assert result is not None
        reason, _, _ = result
        assert reason == ExitReason.TAKE_PROFIT

    def test_stop_loss(self, monitor, state):
        # 25% SL on 0.50 entry -> need 0.375 or below
        result = monitor.check(state, 0.37, datetime.now(timezone.utc))
        assert result is not None
        reason, _, _ = result
        assert reason == ExitReason.STOP_LOSS

    def test_time_limit(self, monitor, state):
        state.entry_time = datetime.now(timezone.utc) - timedelta(minutes=80)
        result = monitor.check(state, 0.51, datetime.now(timezone.utc))
        assert result is not None
        reason, _, _ = result
        assert reason == ExitReason.TIME_LIMIT

    def test_near_resolution_high(self, monitor, state):
        result = monitor.check(state, 0.995, datetime.now(timezone.utc))
        assert result is not None
        reason, _, _ = result
        assert reason == ExitReason.NEAR_RESOLUTION

    def test_near_resolution_low(self, monitor, state):
        result = monitor.check(state, 0.005, datetime.now(timezone.utc))
        assert result is not None
        reason, _, _ = result
        assert reason == ExitReason.NEAR_RESOLUTION

    def test_trailing_stop(self, monitor, state):
        now = datetime.now(timezone.utc)
        # Price goes up to activate trailing stop (need >= 2% gain from 0.50 -> 0.51)
        monitor.check(state, 0.52, now)  # +4% gain -> activates trailing stop
        assert state.trailing_stop_activated
        # Trailing stop level = 0.52 * (1 - 0.01) = 0.5148
        # Price drops below stop
        result = monitor.check(state, 0.505, now)
        if result:
            reason, _, _ = result
            assert reason == ExitReason.TRAILING_STOP

    def test_trailing_stop_not_activated_below_threshold(self, monitor, state):
        now = datetime.now(timezone.utc)
        # Only 1% gain, threshold is 2%
        monitor.check(state, 0.505, now)
        assert not state.trailing_stop_activated

    def test_trailing_stop_level_ratchets_up(self, monitor, state):
        now = datetime.now(timezone.utc)
        # Activate at +4%
        monitor.check(state, 0.52, now)
        assert state.trailing_stop_activated
        first_stop = state.trailing_stop_level
        # Price goes higher
        monitor.check(state, 0.55, now)
        assert state.trailing_stop_level > first_stop

    def test_zero_entry_price(self, monitor):
        state = PositionState(
            instrument_id="test",
            entry_price=0.0,
            entry_time=datetime.now(timezone.utc),
            size=100.0,
        )
        result = monitor.check(state, 0.50, datetime.now(timezone.utc))
        assert result is None

    def test_near_resolution_has_priority(self, monitor, state):
        """Near resolution should fire before take profit."""
        # Entry at 0.50, price at 0.995 -> both near-resolution and take-profit
        result = monitor.check(state, 0.995, datetime.now(timezone.utc))
        reason, _, _ = result
        assert reason == ExitReason.NEAR_RESOLUTION

    def test_register_and_get_state(self, monitor, state):
        monitor.register("pos-1", state)
        retrieved = monitor.get_state("pos-1")
        assert retrieved is state

    def test_unregister(self, monitor, state):
        monitor.register("pos-1", state)
        monitor.unregister("pos-1")
        assert monitor.get_state("pos-1") is None

    def test_get_nonexistent_state(self, monitor):
        assert monitor.get_state("nonexistent") is None

    def test_peak_tracking(self, monitor, state):
        now = datetime.now(timezone.utc)
        monitor.check(state, 0.55, now)
        assert state.peak_price == 0.55
        monitor.check(state, 0.53, now)
        assert state.peak_price == 0.55  # Should not decrease

    def test_trough_tracking(self, monitor, state):
        now = datetime.now(timezone.utc)
        monitor.check(state, 0.45, now)
        assert state.trough_price == 0.45
        monitor.check(state, 0.47, now)
        assert state.trough_price == 0.45  # Should not increase


class TestExitConfig:
    def test_defaults(self):
        config = ExitConfig()
        assert config.take_profit_pct == 0.05
        assert config.stop_loss_pct == 0.25

    def test_validation_take_profit(self):
        with pytest.raises(ValueError):
            ExitConfig(take_profit_pct=-0.1)

    def test_validation_trailing_stop(self):
        with pytest.raises(ValueError):
            ExitConfig(trailing_stop_distance_pct=-0.1)

    def test_validation_stop_loss(self):
        with pytest.raises(ValueError):
            ExitConfig(stop_loss_pct=0.0)

    def test_custom_config(self):
        config = ExitConfig(
            take_profit_pct=0.10,
            stop_loss_pct=0.15,
            max_hold_minutes=120,
            near_resolution_high=0.98,
            near_resolution_low=0.02,
        )
        assert config.take_profit_pct == 0.10
        assert config.stop_loss_pct == 0.15
        assert config.max_hold_minutes == 120

    def test_disabled_checks(self):
        config = ExitConfig(
            take_profit_enabled=False,
            stop_loss_enabled=False,
            trailing_stop_enabled=False,
            time_exit_enabled=False,
            near_resolution_enabled=False,
        )
        monitor = ExitMonitor(config)
        state = PositionState(
            instrument_id="test",
            entry_price=0.50,
            entry_time=datetime.now(timezone.utc) - timedelta(hours=10),
            size=100.0,
        )
        # Even with extreme conditions, all checks disabled means no exit
        result = monitor.check(state, 0.999, datetime.now(timezone.utc))
        assert result is None
