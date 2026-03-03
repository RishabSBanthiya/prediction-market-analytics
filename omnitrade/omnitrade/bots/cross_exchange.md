# Cross-Exchange Bot

## Overview

Multi-leg strategy bot that executes coordinated trades across different exchanges simultaneously. Key use cases include hedged positions (binary outcome + perp hedge) and cross-exchange arbitrage (same event priced differently on two platforms).

## How It Works

The bot runs a continuous loop at a configurable interval (default 30s):

1. **Update balances** — fetch balances from all connected exchanges
2. **Monitor strategies** — check active multi-leg positions for exit conditions
3. **Monitor restored positions** — check individually-restored positions from DB that weren't part of a known strategy
4. **Reconcile** — every 10 iterations, compare DB positions with exchange positions across all exchanges
5. **Generate signals** — ask the cross-exchange signal source for multi-leg trade ideas
6. **Execute strategy** — execute legs sequentially with rollback on failure

Only one new strategy is opened per iteration.

### Leg Execution & Rollback

Legs are executed sequentially, each with its own capital reservation:

- If any leg fails after previous legs succeeded, the bot **rolls back** completed legs by placing opposing orders
- All reservations are released on failure
- Each leg is tracked as a separate position under a sub-agent ID (`{agent_id}-{exchange_id}`)

## Components

| Component | Interface | Role |
|-----------|-----------|------|
| `CrossExchangeSignalSource` | defined in `bots.cross_exchange` | Generates `MultiLegSignal` objects with per-leg exchange, direction, price, weight |
| `Executor` (per exchange) | `components.executors` | Places orders on each exchange (all wrapped with `DryRunExecutor` in paper mode) |
| `ExitMonitor` | `components.exit_strategies` | Tracks exit conditions on the primary leg |
| `RiskCoordinator` | `risk.coordinator` | Capital reservation, drawdown limits, failure tracking across all exchanges |

## Configuration

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier for this bot instance |
| `clients` | `dict[ExchangeId, ExchangeClient]` | required | Map of exchange connections |
| `signal_source` | `CrossExchangeSignalSource` | required | Multi-leg signal generator |
| `executors` | `dict[ExchangeId, Executor]` | required | Per-exchange order executors |
| `risk` | `RiskCoordinator` | required | Shared risk coordinator |
| `exit_config` | `ExitConfig` | `None` | Exit strategy thresholds |
| `environment` | `Environment` | `PAPER` | Paper or live trading |
| `base_size_usd` | `float` | `50.0` | Base position size in USD (scaled by leg weight) |
| `max_strategies` | `int` | `5` | Maximum concurrent active strategies |

## CLI Usage

```bash
# Hedge strategy: binary on Polymarket + perp hedge on Hyperliquid
python scripts/run_bot.py hedge --exchange polymarket --exchange hyperliquid

# Cross-exchange arbitrage: Polymarket vs Kalshi
python scripts/run_bot.py cross-arb --exchange polymarket --exchange kalshi
```

## Risk Management

- **Per-leg capital reservation** — each leg independently reserves capital via `risk.atomic_reserve()`
- **Sequential execution with rollback** — if leg N fails, legs 0..N-1 are unwound with opposing orders
- **Sub-agent isolation** — each exchange leg runs under its own sub-agent ID for independent risk tracking
- **Drawdown tracking** — balances from all exchanges update the risk coordinator every iteration
- **Strategy limits** — `max_strategies` cap prevents overexposure
- **Failure circuit breaker** — repeated errors recorded via `risk.record_failure()`
- **Reconciliation** — periodic DB-vs-exchange comparison on every connected exchange
- **Restart recovery** — `_restore_exit_states()` rebuilds exit monitoring from DB on restart
