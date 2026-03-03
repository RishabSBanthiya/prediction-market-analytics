# OmniTrade

Multi-platform autonomous trading system supporting Polymarket, Kalshi, and Hyperliquid.

## Architecture

- **`omnitrade/core/`** - Platform-agnostic foundation (models, config, enums, errors)
- **`omnitrade/exchanges/`** - Exchange clients implementing `ExchangeClient` ABC
- **`omnitrade/bots/`** - Bot types (directional, market making) - exchange-agnostic
- **`omnitrade/components/`** - Pluggable bot components (signals, sizers, executors, exits)
- **`omnitrade/risk/`** - Risk coordinator and safety modules
- **`omnitrade/storage/`** - SQLite WAL persistence
- **`omnitrade/utils/`** - Rate limiter, logging

## Key Design Rules

1. **Bots never import platform-specific code** - only `ExchangeClient` interface
2. **Prices normalized to 0-1** for binary/event instruments (Kalshi cents converted at adapter)
3. **SignalDirection.LONG/SHORT** for intent, executor maps to BUY/SELL
4. **Composition over inheritance** - bots are assembled from pluggable components
5. **Paper mode by default** - must explicitly opt into live trading

## Running

```bash
# Paper trading (default)
python scripts/run_bot.py directional --exchange polymarket
python scripts/run_bot.py mm --exchange kalshi

# Live trading (requires confirmation)
python scripts/run_bot.py directional --exchange hyperliquid --live

# Tests
pytest tests/
```

## Environment

- Python 3.11+
- Async throughout (aiohttp, asyncio.to_thread for sync SDKs)
- SQLite WAL for storage
