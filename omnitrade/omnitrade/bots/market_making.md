# Market Making Bot

## Overview

Two-sided quoting bot that provides liquidity by placing bid and ask orders. Manages inventory to control directional exposure. Works with any exchange through the `ExchangeClient` interface.

## How It Works

The bot runs a continuous loop at a configurable interval (default 10s):

1. **Update equity** — fetch balance, update risk coordinator for drawdown tracking
2. **Select markets** — ask the market selector for the best instruments to quote
3. **Cancel stale orders** — cancel all existing orders for each selected instrument
4. **Generate quotes** — the quote engine produces bid/ask prices and sizes, adjusted for current inventory
5. **Place orders** — submit bid and ask limit orders (in paper mode, just logs the quotes)
6. **Track fills** — update inventory manager when orders fill

## Components

All components are pluggable (composition over inheritance):

| Component | Interface | Role |
|-----------|-----------|------|
| `QuoteEngine` | `components.quote_engines` | Generates `Quote` objects with bid/ask price and size |
| `MarketSelector` | `components.market_selectors` | Chooses which instruments to quote |
| `InventoryManager` | `components.inventory` | Tracks net exposure per instrument, syncs from exchange |
| `RiskCoordinator` | `risk.coordinator` | Drawdown limits, failure tracking, heartbeat |

## Configuration

Constructor parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_id` | `str` | required | Unique identifier for this bot instance |
| `client` | `ExchangeClient` | required | Exchange connection |
| `quote_engine` | `QuoteEngine` | required | Quote generation logic |
| `market_selector` | `MarketSelector` | required | Instrument selection logic |
| `risk` | `RiskCoordinator` | required | Shared risk coordinator |
| `inventory` | `InventoryManager` | `None` | Inventory tracker (default created if omitted) |
| `environment` | `Environment` | `PAPER` | Paper or live trading |
| `max_instruments` | `int` | `5` | Maximum instruments to quote simultaneously |
| `refresh_interval` | `float` | `10.0` | Seconds between quote refresh cycles |

## CLI Usage

```bash
# Paper trading (default)
python scripts/run_bot.py mm --exchange polymarket

# Live trading
python scripts/run_bot.py mm --exchange kalshi --live
```

## Risk Management

- **Inventory management** — `InventoryManager` tracks net exposure; quote engine skews prices to reduce imbalance
- **Order cancellation on shutdown** — `stop()` cancels all outstanding orders before exiting
- **Drawdown tracking** — equity updates every iteration feed the risk coordinator
- **Instrument limits** — `max_instruments` cap prevents spreading liquidity too thin
- **Failure circuit breaker** — repeated errors recorded via `risk.record_failure()`
- **Exchange sync** — inventory syncs from exchange state on startup via `sync_from_exchange()`
