# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymarket Analytics is a multi-agent trading infrastructure for Polymarket prediction markets. It features atomic capital reservation, composable trading bots with pluggable components, real-time flow detection via WebSocket, and state synchronization with on-chain data via Polygon RPC.

**Language**: Python 3.10+ (async-first)

## Common Commands

```bash
# Run trading bot (dry-run mode)
python scripts/run_bot.py bond --dry-run --agent-id bond-1

# Run with live trading
python scripts/run_bot.py bond --agent-id bond-1 --interval 60

# Run flow strategy
python scripts/run_bot.py flow --agent-id flow-1 --dry-run

# Run market maker (maker rebates)
python scripts/run_market_maker.py --dry-run
python scripts/run_market_maker.py --spread 50 --size 20 --max-inventory 200

# Risk monitoring CLI
python scripts/risk_monitor.py status      # Wallet and risk status
python scripts/risk_monitor.py agents      # Active agents
python scripts/risk_monitor.py positions   # Open positions
python scripts/risk_monitor.py sync        # Force chain reconciliation

# Run backtests (3-parameter simplified strategies)
python -m polymarket.backtesting.strategies.bond_backtest --backtest
python -m polymarket.backtesting.strategies.flow_backtest --backtest
python -m polymarket.backtesting.strategies.bond_backtest --backtest --entry-price 0.96
python -m polymarket.backtesting.strategies.flow_backtest --backtest --take-profit 0.08

# Parameter optimization (anti-overfitting)
python -m polymarket.backtesting.strategies.bond_backtest --optimize
python -m polymarket.backtesting.strategies.flow_backtest --optimize
python -m polymarket.backtesting.strategies.bond_backtest --optimize -n 30  # Quick (30 iterations)

# Run web dashboard
python scripts/run_webapp.py

# Run tests (pytest, NOT unittest)
pytest tests/
pytest tests/test_risk_engine_integration.py -v
```

## Architecture

### Core Components

```
polymarket/
├── core/                    # Shared infrastructure
│   ├── models.py           # All dataclasses (Market, Position, Signal, etc.)
│   ├── config.py           # Configuration with validation
│   ├── api.py              # Async Polymarket API client with rate limiting
│   └── rate_limiter.py     # Sliding window rate limiter
│
├── trading/                 # Live trading infrastructure
│   ├── bot.py              # TradingBot (composition-based, pluggable components)
│   ├── market_maker.py     # MarketMakerBot (maker rebates on 15-min crypto)
│   ├── risk_coordinator.py # Multi-agent risk management, atomic reservations
│   ├── chain_sync.py       # On-chain transaction syncing (source of truth)
│   ├── safety.py           # Circuit breakers, drawdown limits
│   ├── storage/sqlite.py   # SQLite persistence (WAL mode)
│   └── components/         # Pluggable bot components
│       ├── signals.py      # Signal sources (expiring markets, flow alerts)
│       ├── sizers.py       # Position sizers (Kelly, fixed, signal-scaled)
│       ├── executors.py    # Execution engines (dry-run, aggressive)
│       └── exit_strategies.py  # Exit monitors (TP, SL, trailing)
│
├── strategies/              # Strategy implementations
│   ├── bond_strategy.py    # Expiring market strategy (95-98c -> $1)
│   ├── flow_strategy.py    # Flow copy strategy (smart money signals)
│   └── maker_strategy.py   # Market maker strategy (maker rebates)
│
├── flow_detector.py        # Real-time unusual flow detection (WebSocket)
│
└── backtesting/            # Backtesting framework with bias warnings
    ├── base.py             # BaseBacktester class
    ├── results.py          # BacktestResults dataclass
    ├── execution.py        # Simulated execution with fees/slippage
    ├── optimization.py     # Anti-overfitting Bayesian optimizer
    ├── strategies/
    │   ├── bond_backtest.py   # Bond strategy (3 params)
    │   └── flow_backtest.py   # Flow strategy (3 params)
    └── data/               # Data caching infrastructure
        ├── cache.py           # SQLite cache for prices/trades/orderbooks
        ├── cached_fetcher.py  # Cached data fetcher (avoids re-fetching)
        └── trade_fetcher.py   # Trade history with wallet addresses
```

### Composition Pattern

`TradingBot` uses composition instead of inheritance:

```python
bot = TradingBot(
    agent_id="bond-1",
    signal_source=ExpiringMarketSignals(...),  # Where signals come from
    position_sizer=KellyPositionSizer(...),    # How to size positions
    executor=AggressiveExecutor(...),           # How to execute trades
    exit_config=ExitConfig(...)                 # When/how to exit
)
```

### Risk Management

**RiskCoordinator** provides:
- Atomic capital reservation (no race conditions between agents)
- State reconciliation with on-chain data on startup
- Exposure limits: wallet (80%), per-agent (40%), per-market (15%)
- Circuit breaker after 5 consecutive failures
- Daily (10%) and total (25%) drawdown limits
- Agent heartbeats (120s crash detection threshold)

### Data Flow

1. **Chain Sync** fetches on-chain transactions from Polygon RPC (source of truth)
2. **Storage** maintains positions, reservations, agents in SQLite (WAL mode)
3. **RiskCoordinator** reconciles DB state with chain on startup
4. **Signals** flow from detectors to bots via consistent `Signal` dataclass
5. All I/O is async (aiohttp, asyncio)

### Key Data Models (`polymarket/core/models.py`)

All dataclasses are centralized here to avoid circular imports:
- `Market`, `Token`, `Position`, `Trade`, `Signal`
- `Reservation`, `WalletState`, `ExecutionResult`, `AgentInfo`
- Enums: `Side`, `SignalDirection`, `PositionStatus`, `ReservationStatus`, `AgentStatus`

## Key Files by Purpose

| Purpose | Files |
|---------|-------|
| Bot entry point | `scripts/run_bot.py` |
| Market maker entry | `scripts/run_market_maker.py` |
| Trading strategies | `polymarket/strategies/bond_strategy.py`, `flow_strategy.py`, `maker_strategy.py` |
| Risk management | `polymarket/trading/risk_coordinator.py`, `safety.py` |
| Chain sync | `polymarket/trading/chain_sync.py` |
| Persistence | `polymarket/trading/storage/sqlite.py` |
| API client | `polymarket/core/api.py` |
| Flow detection | `polymarket/flow_detector.py` |
| Monitoring CLI | `scripts/risk_monitor.py` |
| Backtesting | `polymarket/backtesting/strategies/bond_backtest.py`, `flow_backtest.py` |
| Optimization | `polymarket/backtesting/optimization.py` |

## Development Guidelines

- **No emojis** in code or comments (per Cursor rules)
- **Full typing annotations** on all functions
- **pytest** for testing (not unittest)
- **Docstrings** following PEP 257
- **Async/await** for all I/O operations
- **Database transactions** for atomic operations
- **Chain sync is source of truth** for positions

## Configuration

Environment variables in `.env`:
```
# API credentials
POLYMARKET_API_KEY=
POLYMARKET_API_SECRET=
POLYMARKET_PASSPHRASE=
POLYGON_RPC_URL=

# Risk limits (defaults shown)
MAX_WALLET_EXPOSURE_PCT=0.80
MAX_PER_AGENT_EXPOSURE_PCT=0.40
MAX_PER_MARKET_EXPOSURE_PCT=0.15
MAX_DAILY_DRAWDOWN_PCT=0.10
MAX_TOTAL_DRAWDOWN_PCT=0.25
CIRCUIT_BREAKER_FAILURES=5
```

## Debugging

1. Check logs: `logs/{agent_id}.log` and `logs/{agent_id}_trades.log`
2. Run `python scripts/risk_monitor.py status` for wallet state
3. Inspect `data/risk_state.db` with SQLite client
4. Force reconciliation: `python scripts/risk_monitor.py sync`

## Backtesting & Optimization

The optimization system prevents overfitting through simplification:

### Anti-Overfitting Measures
- **3 parameters only**: Bond (entry_price, max_spread, max_position), Flow (take_profit, stop_loss, max_position)
- **Single objective**: Sharpe ratio only (no complex multi-metric combinations)
- **Walk-forward validation**: Train on past, test on future (not random market splits)
- **L2 regularization**: Penalty for deviation from sensible defaults
- **Bootstrap confidence**: Reject unstable parameter sets
- **180 days default**: More data = less overfitting

### Parameter Spaces

**Bond Strategy (3 params):**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| entry_price | 0.92-0.97 | 0.95 | Minimum price to enter |
| max_spread_pct | 0.01-0.06 | 0.03 | Maximum acceptable spread |
| max_position_pct | 0.05-0.20 | 0.10 | Position size as % of capital |

**Flow Strategy (3 params):**
| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| take_profit_pct | 0.03-0.15 | 0.06 | Exit at this profit |
| stop_loss_pct | 0.04-0.20 | 0.08 | Exit at this loss |
| max_position_pct | 0.05-0.20 | 0.10 | Position size as % of capital |

### Usage
```bash
# Run bond optimization (50 iterations, 180 days)
python -m polymarket.backtesting.strategies.bond_backtest --optimize

# Run flow optimization
python -m polymarket.backtesting.strategies.flow_backtest --optimize

# Quick test (30 iterations)
python -m polymarket.backtesting.strategies.bond_backtest --optimize -n 30

# Custom parameters for backtest
python -m polymarket.backtesting.strategies.bond_backtest --backtest --entry-price 0.96
python -m polymarket.backtesting.strategies.flow_backtest --backtest --take-profit 0.08 --stop-loss 0.10
```

### Interpreting Results
- **CV Score**: Average Sharpe across walk-forward folds
- **Holdout Score**: Sharpe on unseen future data
- **Overfitting Ratio**: CV/Holdout (ideal ~1.0, >1.5 = overfitting, >2.0 = severe)
- **Is Robust**: Bootstrap variance check (YES = parameters are stable)
- **Verdict**: PASS/WARN/FAIL summary

## Market Maker Bot (Maker Rebates)

A specialized bot for earning Polymarket maker rebates on 15-minute crypto markets.

### Overview

Polymarket's Maker Rebates Program distributes 100% of taker fees to makers (until Jan 9, 2026).
The market maker bot earns these rebates by providing liquidity on eligible markets.

**Docs**: https://docs.polymarket.com/developers/market-makers/maker-rebates-program

### How It Works

1. **Finds eligible markets**: Scans for 15-minute crypto markets (BTC, ETH, SOL)
2. **Places passive quotes**: Bid and ask orders around mid-price
3. **Earns rebates**: When takers fill maker orders, bot receives rebates
4. **Manages inventory**: Skews quotes to stay market-neutral

### Usage

```bash
# Dry run mode (recommended first)
python scripts/run_market_maker.py --dry-run

# Live trading with defaults
python scripts/run_market_maker.py

# Custom configuration
python scripts/run_market_maker.py --spread 50 --size 25 --max-inventory 300

# With custom agent ID
python scripts/run_market_maker.py --agent-id maker-1
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--spread` | 50 | Half spread in basis points (50 = 0.5%) |
| `--size` | 20 | Order size in USD per quote |
| `--max-inventory` | 200 | Max position in one direction (USD) |
| `--max-markets` | 5 | Max markets to trade simultaneously |
| `--interval` | 5 | Quote refresh interval (seconds) |
| `--order-ttl` | 30 | Cancel orders older than this (seconds) |

### Revenue Sources

1. **Maker rebates**: Proportional share of taker fees collected
2. **Spread capture**: Buying low, selling high within the spread

### Risks

1. **Inventory risk**: Price moves against your position
2. **Adverse selection**: Informed traders trading against you
3. **Execution risk**: Orders may not fill or partially fill

### Key Files

| Purpose | File |
|---------|------|
| Run script | `scripts/run_market_maker.py` |
| Market maker bot | `polymarket/trading/market_maker.py` |
| Strategy & signals | `polymarket/strategies/maker_strategy.py` |

### Example Output

```
POLYMARKET MARKET MAKER BOT
  Agent ID:       maker-bot
  Mode:           DRY RUN
  Half Spread:    50 bps (0.50%)
  Order Size:     $20
  Max Inventory:  $200
  Max Markets:    5
  Quote Interval: 5s

WALLET STATE
  USDC Balance:    $1,000.00
  Available:       $1,000.00

New eligible market: BTC above $98000 11:00PM-11:15PM ET? (12.5 min left)
Placed BUY 0.4082 shares @ $0.49 for 5a8b3c...
Placed SELL 0.3846 shares @ $0.52 for 5a8b3c...
```
