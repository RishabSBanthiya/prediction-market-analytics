# Directional Bot

## Overview

Signal-driven directional trading bot. Takes one-sided positions based on signals from a pluggable signal source. Works with any exchange through the `ExchangeClient` interface.

## How It Works

The bot runs a continuous loop at a configurable interval (default 30s):

1. **Update equity** — fetch balance, update risk coordinator for drawdown tracking
2. **Monitor exits** — check all open positions against exit conditions (stop-loss, take-profit, trailing stop, time-based)
3. **Reconcile** — every 10 iterations, compare DB positions with exchange positions and log discrepancies
4. **Generate signals** — ask the signal source for new trade ideas
5. **Process best signal** — size the position, reserve capital through risk, execute, and register for exit monitoring

Only one new trade is opened per iteration to avoid overcommitting.

## Components

All components are pluggable (composition over inheritance):

| Component | Interface | Role |
|-----------|-----------|------|
| `SignalSource` | `components.signals` | Generates `Signal` objects with direction, price, score |
| `PositionSizer` | `components.sizers` | Determines USD size given signal, balance, price |
| `Executor` | `components.executors` | Places orders on the exchange (auto-wrapped with `DryRunExecutor` in paper mode) |
| `ExitMonitor` | `components.exit_strategies` | Tracks stop-loss, take-profit, trailing stops, time expiry |
| `RiskCoordinator` | `risk.coordinator` | Capital reservation, drawdown limits, failure tracking |

## Configuration

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier for this bot instance |
| `client` | `ExchangeClient` | required | Exchange connection |
| `signal_source` | `SignalSource` | required | Signal generator |
| `sizer` | `PositionSizer` | required | Position sizing logic |
| `executor` | `Executor` | required | Order execution (wrapped in paper mode) |
| `risk` | `RiskCoordinator` | required | Shared risk coordinator |
| `exit_config` | `ExitConfig` | `None` | Exit strategy thresholds |
| `environment` | `Environment` | `PAPER` | Paper or live trading |
| `max_positions` | `int` | `10` | Maximum concurrent open positions |
| `min_price` | `float` | `0.05` | Minimum price filter (normalized 0-1) |
| `max_price` | `float` | `0.95` | Maximum price filter (normalized 0-1) |

## CLI Usage

```bash
# Paper trading (default)
python scripts/run_bot.py directional --exchange polymarket

# Live trading
python scripts/run_bot.py directional --exchange kalshi --live
```

## Risk Management

- **Capital reservation** — `risk.atomic_reserve()` locks capital before execution; released on failure
- **Drawdown tracking** — equity updates every iteration feed the risk coordinator's drawdown monitor
- **Position limits** — `max_positions` cap prevents overexposure
- **Price filters** — `min_price`/`max_price` avoid extreme-probability contracts
- **Failure circuit breaker** — repeated errors recorded via `risk.record_failure()`
- **Reconciliation** — periodic DB-vs-exchange comparison catches state drift
- **Exit monitoring** — persistent exit state survives restarts via `_restore_exit_states()`
