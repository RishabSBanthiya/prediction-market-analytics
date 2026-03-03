# OmniTrade

Multi-platform autonomous trading system for prediction markets and perpetual futures. Runs directional, market-making, and cross-exchange hedge strategies across **Polymarket**, **Kalshi**, and **Hyperliquid** through a unified interface.

```
58 Python files | 7,500+ lines | 190 tests | 3 exchanges | 5 bot modes
```

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Bot Types](#bot-types)
- [Exchanges](#exchanges)
- [Components](#components)
- [Risk Management](#risk-management)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Design Principles](#design-principles)

---

## Architecture

```
                          +-----------------+
                          |   CLI / Script  |
                          +--------+--------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
   +--------v--------+   +--------v--------+   +---------v--------+
   | DirectionalBot  |   | MarketMakingBot |   | CrossExchangeBot |
   +---------+-------+   +--------+--------+   +---------+--------+
             |                     |                      |
   +---------v---------------------v----------------------v--------+
   |                     Components Layer                          |
   |  SignalSource | Sizer | Executor | ExitMonitor | QuoteEngine  |
   +----------------------------+----------------------------------+
                                |
                     +----------v----------+
                     |   ExchangeClient    |   <-- abstract interface
                     +----------+----------+
                                |
            +-------------------+-------------------+
            |                   |                   |
   +--------v------+  +--------v------+  +---------v--------+
   | Polymarket    |  |    Kalshi     |  |   Hyperliquid    |
   | (binary/CLOB) |  | (event/REST)  |  |  (perps/SDK)     |
   +---------------+  +---------------+  +------------------+

   Cross-cutting: RiskCoordinator | SQLiteStorage | RateLimiter
```

Bots are **exchange-agnostic** -- they never import platform-specific code. Each exchange implements the `ExchangeClient` ABC, and adapters normalize all data to unified models (prices in 0-1 for binary/event instruments, USD for perpetuals).

---

## Quick Start

### 1. Install

```bash
cd omnitrade
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (see Configuration section)
```

### 3. Run (paper mode)

```bash
# Directional bot on Polymarket
python scripts/run_bot.py directional --exchange polymarket

# Market making on Kalshi
python scripts/run_bot.py mm --exchange kalshi

# Cross-exchange hedge: binary on Polymarket + perp on Hyperliquid
python scripts/run_bot.py hedge --exchange polymarket

# Cross-exchange arbitrage: Polymarket vs Kalshi
python scripts/run_bot.py cross-arb
```

Paper mode is the default -- all orders are simulated via `DryRunExecutor`. No real money at risk.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Bot Types

### Directional Bot

Signal-driven trading on a single exchange. Generates signals, sizes positions, executes, and monitors exits.

```
Signal Source  -->  Position Sizer  -->  Risk Check  -->  Executor  -->  Exit Monitor
```

**Loop (every N seconds):**
1. Fetch balance, update equity tracking
2. Monitor open positions for exit conditions (take-profit, stop-loss, trailing stop, time limit, near-resolution)
3. Generate signals from the configured `SignalSource`
4. Filter by price range (0.05-0.95) and position limits
5. Size using the configured `PositionSizer` (Kelly, fixed, signal-scaled, etc.)
6. Reserve capital atomically through `RiskCoordinator`
7. Execute via the configured `Executor`
8. Register position for exit monitoring

```bash
python scripts/run_bot.py directional --exchange polymarket --interval 15
```

### Market Making Bot

Two-sided quoting on binary/event markets with inventory management.

**Loop (every N seconds):**
1. Select liquid markets via `MarketSelector`
2. For each market: cancel stale orders, generate bid/ask quotes via `QuoteEngine`
3. Place orders on both sides with inventory-based skew
4. Track exposure via `InventoryManager`

```bash
python scripts/run_bot.py mm --exchange kalshi
```

### Cross-Exchange Bot

Multi-leg strategies across different exchanges with atomic execution and rollback.

**Strategies:**

| Strategy | Description | Exchanges |
|----------|-------------|-----------|
| **Binary+Perp Hedge** | Buy binary outcome near resolution + hedge with perp | Polymarket/Kalshi + Hyperliquid |
| **Cross-Exchange Arb** | Same event priced differently across platforms | Polymarket + Kalshi |

**Hedge example:** "Will BTC reach $100k?" is trading at 0.85 on Polymarket. The bot buys the YES token and simultaneously shorts BTC-PERP on Hyperliquid as a delta hedge.

**Execution flow:**
1. Generate multi-leg signals (scan binary markets, match to perps by keyword)
2. For each leg: reserve capital per-exchange, execute sequentially
3. If any leg fails: **roll back** all completed legs with opposing orders
4. Monitor primary leg for exit conditions, unwind all legs on exit

```bash
# Binary + perp hedge
python scripts/run_bot.py hedge --exchange polymarket --hedge-exchange hyperliquid

# Cross-platform arbitrage
python scripts/run_bot.py cross-arb
```

---

## Exchanges

All three exchanges implement the same `ExchangeClient` interface:

```python
class ExchangeClient(ABC):
    # Lifecycle
    async def connect() -> None
    async def close() -> None

    # Market data
    async def get_instruments(active_only=True) -> list[Instrument]
    async def get_orderbook(instrument_id, depth=10) -> OrderbookSnapshot
    async def get_midpoint(instrument_id) -> Optional[float]

    # Trading
    async def place_order(request: OrderRequest) -> OrderResult
    async def cancel_order(order_id, instrument_id) -> bool
    async def cancel_all_orders(instrument_id=None) -> int
    async def get_open_orders(instrument_id=None) -> list[OpenOrder]

    # Account
    async def get_balance() -> AccountBalance
    async def get_positions() -> list[ExchangePosition]
```

### Polymarket

| Feature | Detail |
|---------|--------|
| Type | Binary prediction market (Polygon L2) |
| Auth | Private key + proxy wallet (EIP-191) |
| SDK | `py-clob-client` (sync, wrapped with `asyncio.to_thread`) |
| Instruments | YES/NO tokens per market, prices 0-1 |
| Data source | Gamma API (markets), CLOB API (orderbook, trading) |

### Kalshi

| Feature | Detail |
|---------|--------|
| Type | Regulated event contracts |
| Auth | API key + RSA-PSS signature |
| SDK | Native async via `aiohttp` |
| Instruments | Event contracts, prices in cents (0-99) |
| Price conversion | Adapter normalizes cents to 0-1 at the boundary |

### Hyperliquid

| Feature | Detail |
|---------|--------|
| Type | Perpetual futures DEX |
| Auth | Private key (eth-account) |
| SDK | `hyperliquid-python-sdk` (sync, wrapped with `asyncio.to_thread`) |
| Instruments | Perps with leverage, funding rates, liquidation prices |
| Extras | `set_leverage()` for cross/isolated margin |

### Exchange Registry

Clients register via decorator and are instantiated through a factory:

```python
from omnitrade.exchanges.registry import create_client
from omnitrade.core.enums import ExchangeId

client = create_client(ExchangeId.POLYMARKET, config)
async with client:
    instruments = await client.get_instruments()
```

---

## Components

Bots are assembled from pluggable, interchangeable components. Each component is an ABC with concrete implementations.

### Signal Sources

| Source | Description |
|--------|-------------|
| `MidpointDeviationSignal` | Mean-reversion: LONG below fair value, SHORT above |
| `BinaryPerpHedgeSignal` | Scans binary markets, matches to perps, generates hedged multi-leg signals |
| `CrossExchangeArbSignal` | Finds same-event price discrepancies across Polymarket and Kalshi |

### Position Sizers

| Sizer | Strategy | Default |
|-------|----------|---------|
| `FixedSizer` | Fixed USD per trade | $50 |
| `PercentageSizer` | Percentage of capital | 2% |
| `FixedFractionSizer` | Fraction with min/max bounds | 10%, $10-$200 |
| `KellySizer` | Fractional Kelly Criterion | 0.5x Kelly, 5% max |
| `SignalScaledSizer` | Base size scaled by signal strength | 2% base, 3x max |
| `CompositeSizer` | Takes minimum of N sizers | Safety overlay |

### Executors

| Executor | Mode | Behavior |
|----------|------|----------|
| `DryRunExecutor` | Paper | Simulates fills with configurable slippage |
| `AggressiveExecutor` | Live | Takes best price with spread (3% max) and slippage (2% max) checks |
| `LimitExecutor` | Live | Places limit orders with optional price offset |

### Exit Strategies

`ExitMonitor` checks positions against exit conditions in priority order:

| Priority | Condition | Default |
|----------|-----------|---------|
| 1 | Near-resolution (binary expiry) | Price >= 0.99 or <= 0.01 |
| 2 | Take-profit | Return >= 5% |
| 3 | Trailing stop | Activates at 2% gain, 1% trail distance |
| 4 | Stop-loss | Loss >= 25% |
| 5 | Time limit | Hold <= 75 minutes |

### Market Making Components

| Component | Purpose |
|-----------|---------|
| `SimpleSpreadQuoter` | Fixed-spread quotes around midpoint with inventory skew |
| `ActiveMarketSelector` | Selects most liquid markets by spread tightness |
| `InventoryManager` | Tracks signed USD exposure per instrument |

---

## Risk Management

### RiskCoordinator

Central authority for capital allocation. Every trade passes through `atomic_reserve()` which performs six sequential checks:

```
Trading Halt? --> Circuit Breaker? --> Drawdown? --> Trade Size? --> Exposure? --> Balance?
     |                  |                |              |              |            |
   BLOCK             BLOCK            BLOCK          BLOCK          BLOCK        BLOCK
```

**Exposure Limits (defaults):**

| Limit | Default | Scope |
|-------|---------|-------|
| Wallet exposure | 60% of equity | Total across all positions |
| Per-agent exposure | 30% of equity | Single bot instance |
| Per-instrument exposure | 10% of equity | Single market/contract |
| Min trade size | $10 | Minimum order value |
| Max trade size | $500 | Maximum order value |

### Circuit Breaker

Prevents trading after consecutive failures:

```
CLOSED (normal) --[N failures]--> OPEN (halted) --[timeout]--> HALF-OPEN (test trade)
     ^                                                              |
     +------------------[success]-----------------------------------+
```

- **Failure threshold:** 3 consecutive failures to trip
- **Reset timeout:** 600 seconds before allowing a test trade
- **Recovery:** Single successful trade returns to CLOSED

### Drawdown Protection

| Check | Default | Resets |
|-------|---------|--------|
| Daily drawdown | 5% | Midnight UTC |
| Total drawdown | 15% | Never (from all-time peak) |

### Trading Halt

Composite halt mechanism supporting multiple independent reasons. Trading resumes only when **all** halt reasons are cleared. Used for manual intervention, exchange outages, or cascading risk events.

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

**General:**
```env
OMNITRADE_ENV=paper          # paper | live
OMNITRADE_DB_PATH=data/omnitrade.db
LOG_LEVEL=INFO
```

**Polymarket:**
```env
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...
POLYMARKET_CLOB_HOST=https://clob.polymarket.com
POLYMARKET_CHAIN_ID=137
```

**Kalshi:**
```env
KALSHI_API_KEY=your-api-key
KALSHI_RSA_KEY_PATH=./kalshi_private_key.pem
KALSHI_API_BASE=https://api.elections.kalshi.com/trade-api/v2
```

**Hyperliquid:**
```env
HYPERLIQUID_PRIVATE_KEY=0x...
HYPERLIQUID_API_BASE=https://api.hyperliquid.xyz
HYPERLIQUID_WS_URL=wss://api.hyperliquid.xyz/ws
```

**Risk Limits:**
```env
MAX_WALLET_EXPOSURE_PCT=0.60
MAX_PER_AGENT_EXPOSURE_PCT=0.30
MAX_PER_MARKET_EXPOSURE_PCT=0.10
MIN_TRADE_VALUE_USD=10.0
MAX_TRADE_VALUE_USD=500.0
MAX_DAILY_DRAWDOWN_PCT=0.05
MAX_TOTAL_DRAWDOWN_PCT=0.15
```

### Config API

```python
from omnitrade.core.config import Config

config = Config.from_env()
config.enabled_exchanges()          # [ExchangeId.POLYMARKET, ...]
config.get_exchange_config(exchange) # ExchangeConfig for specific exchange
```

Exchanges auto-enable when their primary auth credential is set.

---

## CLI Reference

```
python scripts/run_bot.py <bot_type> [options]
```

### Bot Types

| Type | Description | Requires `--exchange` |
|------|-------------|----------------------|
| `directional` | Signal-driven trading | Yes |
| `mm` / `market-making` | Two-sided quoting | Yes |
| `hedge` | Binary + perp hedge | Optional (default: polymarket) |
| `cross-arb` | Polymarket vs Kalshi arb | No |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--exchange`, `-e` | - | Exchange: `polymarket`, `kalshi`, `hyperliquid` |
| `--hedge-exchange` | `hyperliquid` | Hedge leg exchange (for `hedge` mode) |
| `--paper` | `true` | Paper trading mode (simulated fills) |
| `--live` | `false` | Live trading (requires "yes" confirmation) |
| `--interval`, `-i` | `30` | Loop interval in seconds |
| `--agent-id` | auto | Custom bot agent identifier |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

```bash
# Paper trading (default, safe)
python scripts/run_bot.py directional --exchange polymarket
python scripts/run_bot.py mm --exchange kalshi --interval 10
python scripts/run_bot.py hedge --exchange polymarket
python scripts/run_bot.py cross-arb

# Live trading (real money)
python scripts/run_bot.py directional --exchange hyperliquid --live
# WARNING: LIVE TRADING MODE - Real money at risk!
# Type 'yes' to confirm:

# Debug logging
python scripts/run_bot.py mm --exchange polymarket --log-level DEBUG

# Custom agent ID and interval
python scripts/run_bot.py directional --exchange kalshi --agent-id my-bot --interval 60
```

---

## Testing

```bash
# Run all 190 tests
pytest tests/ -v

# Run by module
pytest tests/risk/ -v              # Risk coordinator + safety
pytest tests/components/ -v         # Sizers, executors, exits
pytest tests/exchanges/ -v          # Exchange adapters
pytest tests/bots/ -v               # Bot lifecycle
pytest tests/test_storage.py -v     # SQLite storage
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Risk: Safety (circuit breaker, drawdown, halt) | 24 | State transitions, edge cases, timing |
| Risk: Coordinator | 19 | All 6 reservation checks, exposure limits, lifecycle |
| Components: Sizers | 27 | All 6 sizers with boundary conditions |
| Components: Exit Strategies | 18 | All 5 exit conditions, priority ordering |
| Components: Executors | 18 | All 3 executors, spread/slippage validation |
| Exchanges: Adapters | 26 | All 3 adapters, price normalization |
| Storage: SQLite | 27 | Full CRUD, exposure calculations, cleanup |
| Bots: Directional | 11 | Lifecycle, signal processing, paper/live mode |
| **Total** | **190** | |

All tests use mocks and temporary databases -- no network calls or real exchanges.

---

## Project Structure

```
omnitrade/
├── pyproject.toml                  # Build config, dependencies
├── .env.example                    # Environment variable template
├── CLAUDE.md                       # Project rules for AI assistants
├── README.md                       # This file
│
├── omnitrade/
│   ├── __init__.py                 # Package root (v0.1.0)
│   │
│   ├── core/                       # Platform-agnostic foundation
│   │   ├── enums.py                # ExchangeId, Side, SignalDirection, OrderType, etc.
│   │   ├── models.py               # Instrument, Signal, OrderResult, MultiLegSignal, etc.
│   │   ├── config.py               # Config, ExchangeConfig, RiskConfig
│   │   └── errors.py               # Exception hierarchy
│   │
│   ├── exchanges/                  # Exchange abstraction layer
│   │   ├── base.py                 # ExchangeClient ABC, ExchangeAuth ABC
│   │   ├── registry.py             # @register_exchange decorator, create_client factory
│   │   ├── polymarket/
│   │   │   ├── auth.py             # Wallet + proxy auth via py-clob-client
│   │   │   ├── adapter.py          # Gamma/CLOB API -> unified models
│   │   │   └── client.py           # PolymarketClient(ExchangeClient)
│   │   ├── kalshi/
│   │   │   ├── auth.py             # RSA-PSS request signing
│   │   │   ├── adapter.py          # Cents (0-99) -> normalized (0-1)
│   │   │   └── client.py           # KalshiClient(ExchangeClient)
│   │   └── hyperliquid/
│   │       ├── auth.py             # eth-account wallet auth
│   │       ├── adapter.py          # Perps: leverage, funding, margin
│   │       └── client.py           # HyperliquidClient(ExchangeClient)
│   │
│   ├── bots/                       # Exchange-agnostic bot implementations
│   │   ├── directional.py          # Signal -> size -> execute -> exit
│   │   ├── market_making.py        # Select -> quote -> place/cancel
│   │   └── cross_exchange.py       # Multi-leg across exchanges + rollback
│   │
│   ├── components/                 # Pluggable bot components
│   │   ├── signals.py              # SignalSource ABC + MidpointDeviationSignal
│   │   ├── sizers.py               # PositionSizer ABC + 6 implementations
│   │   ├── executors.py            # Executor ABC + DryRun, Aggressive, Limit
│   │   ├── exit_strategies.py      # ExitConfig + ExitMonitor (5 exit conditions)
│   │   ├── hedge_signals.py        # BinaryPerpHedgeSignal, CrossExchangeArbSignal
│   │   ├── quote_engines.py        # QuoteEngine ABC + SimpleSpreadQuoter
│   │   ├── market_selectors.py     # MarketSelector ABC + ActiveMarketSelector
│   │   └── inventory.py            # InventoryManager for market making
│   │
│   ├── risk/                       # Risk management
│   │   ├── coordinator.py          # RiskCoordinator: atomic reserves, exposure limits
│   │   └── safety.py               # CircuitBreaker, DrawdownLimit, TradingHalt
│   │
│   ├── storage/                    # Persistence
│   │   ├── base.py                 # StorageBackend ABC
│   │   └── sqlite.py              # SQLite WAL with indexes, multi-exchange support
│   │
│   └── utils/                      # Utilities
│       ├── rate_limiter.py         # RateLimiter + EndpointRateLimiter
│       └── logging.py             # setup_logging, get_logger
│
├── scripts/
│   └── run_bot.py                  # Unified CLI entry point
│
└── tests/
    ├── conftest.py                 # MockExchangeClient, fixtures
    ├── test_storage.py             # SQLite CRUD, exposure, lifecycle
    ├── bots/
    │   └── test_directional.py     # Bot lifecycle, signal processing
    ├── components/
    │   ├── test_sizers.py          # All 6 sizers
    │   ├── test_executors.py       # All 3 executors
    │   └── test_exit_strategies.py # All 5 exit conditions
    ├── exchanges/
    │   └── test_adapters.py        # All 3 exchange adapters
    └── risk/
        ├── test_safety.py          # CircuitBreaker, DrawdownLimit, TradingHalt
        └── test_coordinator.py     # Reservation guards, exposure limits
```

---

## Design Principles

1. **Bots never import platform-specific code.** They depend only on the `ExchangeClient` interface. Exchange adapters handle all normalization.

2. **Composition over inheritance.** Bots are assembled from interchangeable components (signals, sizers, executors). Swap a `FixedSizer` for a `KellySizer` without changing anything else.

3. **Paper mode by default.** The `--live` flag requires explicit confirmation. Paper mode wraps all executors with `DryRunExecutor` automatically.

4. **Atomic risk management.** Every trade passes through `RiskCoordinator.atomic_reserve()` which checks 6 conditions in sequence. Capital is reserved before execution and confirmed after fill.

5. **Normalized pricing.** Binary/event instruments use 0-1 range everywhere. Kalshi's cents (0-99) are converted at the adapter boundary. Sizers, signals, and bots never deal with platform-specific price formats.

6. **Async throughout.** All I/O is async. Synchronous SDKs (py-clob-client, hyperliquid-python-sdk) are wrapped with `asyncio.to_thread` for non-blocking operation.

7. **Multi-exchange by design.** Config, risk, storage, and bots all support multiple exchanges natively. Cross-exchange strategies are first-class citizens with per-leg risk checks and rollback.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `aiohttp` | Async HTTP (Kalshi client, Polymarket Gamma API) |
| `python-dotenv` | Environment variable loading |
| `py-clob-client` | Polymarket CLOB trading SDK |
| `web3` | Polymarket wallet operations |
| `cryptography` | Kalshi RSA-PSS request signing |
| `hyperliquid-python-sdk` | Hyperliquid trading SDK |
| `eth-account` | Hyperliquid wallet authentication |
| `pytest` | Testing framework |
| `pytest-asyncio` | Async test support |

---

## License

Private. Not for redistribution.
