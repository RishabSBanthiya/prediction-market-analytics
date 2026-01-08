# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymarket Analytics is a multi-agent trading infrastructure for Polymarket prediction markets. It features atomic capital reservation, composable trading bots with pluggable components, real-time flow detection via WebSocket, and state synchronization with on-chain data via Polygon RPC.

**Language**: Python 3.10+ (async-first)

## Common Commands

```bash
# Run trading bot (dry-run mode)
python scripts/run_bot.py bond --dry-run --agent-id bond-1
python scripts/run_bot.py flow --dry-run --agent-id flow-1

# Run with live trading
python scripts/run_bot.py bond --agent-id bond-1 --interval 60
python scripts/run_bot.py flow --agent-id flow-1 --interval 5

# Run arbitrage bot (delta-neutral)
python scripts/run_arb_bot.py --dry-run
python scripts/run_arb_bot.py --min-edge 0.02 --position-size 50

# Risk monitoring CLI
python scripts/risk_monitor.py status      # Wallet and risk status
python scripts/risk_monitor.py agents      # Active agents
python scripts/risk_monitor.py positions   # Open positions
python scripts/risk_monitor.py drawdown    # Drawdown tracking
python scripts/risk_monitor.py sync        # Force chain reconciliation
python scripts/risk_monitor.py reset-drawdown  # Reset drawdown tracking

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
│   └── flow_strategy.py    # Flow copy strategy (smart money signals)
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
        └── cached_fetcher.py  # Cached data fetcher (avoids re-fetching)
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
| Arbitrage bot | `scripts/run_arb_bot.py` |
| Trading strategies | `polymarket/strategies/bond_strategy.py`, `flow_strategy.py` |
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
PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...
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
5. Reset drawdown: `python scripts/risk_monitor.py reset-drawdown`

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

## Arbitrage Bot (Delta-Neutral)

Monitors 15-minute crypto markets for risk-free arbitrage opportunities.

### How It Works

1. Scans markets where `UP_price + DOWN_price < 1.0` (after fees)
2. Places limit orders on BOTH sides at profitable prices
3. When both fill = guaranteed profit at resolution

### Usage

```bash
# Dry run mode (recommended first)
python scripts/run_arb_bot.py --dry-run

# Live trading
python scripts/run_arb_bot.py

# Custom configuration
python scripts/run_arb_bot.py --min-edge 0.02 --position-size 50 --max-positions 10
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-edge` | 0.01 | Minimum profit margin (1%) |
| `--position-size` | 25 | USD per side |
| `--max-positions` | 10 | Max concurrent arb positions |
| `--dry-run` | false | Simulation mode |
