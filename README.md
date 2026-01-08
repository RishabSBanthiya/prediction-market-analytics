# Polymarket Analytics

A multi-agent trading infrastructure for Polymarket prediction markets, featuring:
- **Multi-agent risk coordination** with atomic capital reservation
- **Composable trading bots** with pluggable components
- **Real-time flow detection** via WebSocket
- **Three trading strategies**: Bond (expiring markets), Flow (copy trading), Arbitrage (delta-neutral)

---

## Quick Start

### Prerequisites

- Python 3.10+
- Polymarket account with trading enabled
- Polygon wallet with USDC

### Installation

```bash
git clone <repo-url>
cd polymarket-analytics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your keys
```

### Running Bots

```bash
# Dry run (no real trades)
python scripts/run_bot.py bond --dry-run
python scripts/run_bot.py flow --dry-run
python scripts/run_arb_bot.py --dry-run

# Live trading
python scripts/run_bot.py bond --agent-id bond-1
python scripts/run_bot.py flow --agent-id flow-1
python scripts/run_arb_bot.py

# Monitor
python scripts/risk_monitor.py status
python scripts/risk_monitor.py agents
```

---

## Trading Strategies

### 1. Bond Strategy (Expiring Markets)

Trades markets near expiration priced 95-98c, betting they resolve to $1. Behaves like short-term bonds.

```bash
python scripts/run_bot.py bond --dry-run --interval 10
python scripts/run_bot.py bond --agent-id bond-1 --min-price 0.94 --max-price 0.99
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-price` | 0.95 | Minimum entry price |
| `--max-price` | 0.98 | Maximum entry price |
| `--interval` | 10 | Scan interval (seconds) |

### 2. Flow Copy Strategy

Copies unusual flow signals from smart money, oversized bets, and coordinated wallets.

```bash
python scripts/run_bot.py flow --dry-run --interval 5
python scripts/run_bot.py flow --agent-id flow-1 --min-score 40
python scripts/run_bot.py flow --category crypto
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-score` | 30 | Minimum signal score to trade |
| `--min-trade-size` | 100 | Minimum trade size in USD |
| `--category` | all | Filter: crypto, sports, politics, other |

**Signal Weights:**
| Signal | Description | Weight |
|--------|-------------|--------|
| SMART_MONEY_ACTIVITY | Wallets with >65% win rate | 30 |
| OVERSIZED_BET | Trades 10x+ avg or >$10k | 25 |
| COORDINATED_WALLETS | Connected wallets trading together | 25 |
| VOLUME_SPIKE | Volume 3x+ baseline | 10 |
| PRICE_ACCELERATION | Momentum building | 10 |

### 3. Arbitrage Strategy (Delta-Neutral)

Monitors 15-minute crypto markets for risk-free arbitrage where both outcomes can be bought for less than $1 total.

```bash
python scripts/run_arb_bot.py --dry-run
python scripts/run_arb_bot.py --min-edge 0.02 --position-size 50
```

**How it works:**
1. Scans for markets where `UP_price + DOWN_price < 1.0` (after fees)
2. Places limit orders on BOTH sides at profitable prices
3. When both fill = guaranteed profit at resolution

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-edge` | 0.01 | Minimum profit margin (1%) |
| `--position-size` | 25 | USD per side |
| `--max-positions` | 10 | Max concurrent arb positions |

---

## Risk Management

The `RiskCoordinator` provides multi-agent risk management:

| Feature | Description |
|---------|-------------|
| **Atomic Reservation** | No race conditions between agents |
| **Exposure Limits** | 80% wallet, 40% per-agent, 15% per-market |
| **Circuit Breaker** | Stops after 5 consecutive failures |
| **Drawdown Limits** | 10% daily, 25% total max drawdown |
| **Agent Heartbeats** | Detects crashed agents (120s threshold) |

### Monitoring Commands

```bash
python scripts/risk_monitor.py status        # Overall wallet/risk status
python scripts/risk_monitor.py agents        # List all agents
python scripts/risk_monitor.py positions     # View open positions
python scripts/risk_monitor.py drawdown      # Drawdown tracking status
python scripts/risk_monitor.py cleanup       # Cleanup stale data
python scripts/risk_monitor.py reset-drawdown # Reset drawdown tracking
python scripts/risk_monitor.py stop-all      # Emergency stop all agents
python scripts/risk_monitor.py sync          # Force chain reconciliation
```

---

## Backtesting

Run backtests directly via Python modules:

```bash
# Bond strategy backtest
python -m polymarket.backtesting.strategies.bond_backtest --backtest
python -m polymarket.backtesting.strategies.bond_backtest --backtest --entry-price 0.96

# Flow strategy backtest
python -m polymarket.backtesting.strategies.flow_backtest --backtest
python -m polymarket.backtesting.strategies.flow_backtest --backtest --take-profit 0.08

# Parameter optimization (Bayesian, anti-overfitting)
python -m polymarket.backtesting.strategies.bond_backtest --optimize -n 50
python -m polymarket.backtesting.strategies.flow_backtest --optimize -n 50
```

**Anti-Overfitting Measures:**
- 3 parameters only per strategy
- Walk-forward validation (train on past, test on future)
- L2 regularization toward sensible defaults
- Bootstrap confidence intervals

---

## Configuration

### Environment Variables (.env)

```bash
# Required for live trading
PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...

# Optional
CHAIN_ID=137
POLYGON_RPC_URL=https://polygon-rpc.com
RISK_DB_PATH=data/risk_state.db
LOG_LEVEL=INFO

# Risk limits
MAX_WALLET_EXPOSURE_PCT=0.80
MAX_PER_AGENT_EXPOSURE_PCT=0.40
MAX_PER_MARKET_EXPOSURE_PCT=0.15
MAX_DAILY_DRAWDOWN_PCT=0.10
MAX_TOTAL_DRAWDOWN_PCT=0.25
CIRCUIT_BREAKER_FAILURES=5
```

---

## Project Structure

```
polymarket-analytics/
├── polymarket/
│   ├── core/                    # Shared infrastructure
│   │   ├── models.py            # Dataclasses (Market, Position, Signal)
│   │   ├── api.py               # Async Polymarket API client
│   │   └── config.py            # Validated configuration
│   │
│   ├── trading/                 # Live trading infrastructure
│   │   ├── bot.py               # Composition-based TradingBot
│   │   ├── risk_coordinator.py  # Multi-agent risk management
│   │   ├── chain_sync.py        # On-chain transaction syncing
│   │   ├── safety.py            # Circuit breakers, drawdown limits
│   │   ├── storage/sqlite.py    # SQLite persistence (WAL mode)
│   │   └── components/          # Pluggable components
│   │       ├── signals.py       # Signal sources
│   │       ├── sizers.py        # Position sizers
│   │       ├── executors.py     # Execution engines
│   │       └── exit_strategies.py
│   │
│   ├── strategies/              # Strategy implementations
│   │   ├── bond_strategy.py     # Expiring market strategy
│   │   └── flow_strategy.py     # Flow copy strategy
│   │
│   ├── backtesting/             # Backtesting framework
│   │   ├── optimization.py      # Bayesian optimizer
│   │   └── strategies/          # Strategy backtesters
│   │
│   └── flow_detector.py         # Real-time flow detection
│
├── scripts/                     # CLI utilities
│   ├── run_bot.py               # Main bot entry (bond/flow)
│   ├── run_arb_bot.py           # Arbitrage bot
│   ├── risk_monitor.py          # Risk monitoring & management
│   ├── run_webapp.py            # Dashboard server
│   ├── check_portfolio.py       # Portfolio viewer
│   └── cleanup_dead_positions.py # Maintenance utility
│
├── webapp/                      # Dashboard (FastAPI)
└── data/                        # SQLite databases
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_bot.py` | Main trading bot (bond/flow) | `python scripts/run_bot.py {bond,flow} [options]` |
| `run_arb_bot.py` | Delta-neutral arbitrage | `python scripts/run_arb_bot.py [--dry-run]` |
| `risk_monitor.py` | Monitoring & management | `python scripts/risk_monitor.py {status,agents,positions,...}` |
| `run_webapp.py` | Start dashboard | `python scripts/run_webapp.py` |
| `check_portfolio.py` | View portfolio | `python scripts/check_portfolio.py` |
| `cleanup_dead_positions.py` | Clean stale positions | `python scripts/cleanup_dead_positions.py [--dry-run]` |

---

## API Reference

| API | Base URL | Purpose | Rate Limit |
|-----|----------|---------|------------|
| **RTDS WebSocket** | `wss://ws-live-data.polymarket.com` | Real-time trades | N/A |
| **Gamma API** | `https://gamma-api.polymarket.com` | Market metadata | 4,000/10s |
| **CLOB API** | `https://clob.polymarket.com` | Orderbook, orders | 9,000/10s |
| **Data API** | `https://data-api.polymarket.com` | Positions, activity | 1,000/10s |

---

## Troubleshooting

### Bot not starting
- Check `.env` has `PRIVATE_KEY` and `POLYMARKET_PROXY_ADDRESS`
- Run `python scripts/check_portfolio.py` to verify credentials

### Rate limit errors
- Increase interval: `--interval 30`
- Check `API_RATE_LIMIT_PER_10S` in config

### Drawdown breached (phantom)
- Run `python scripts/risk_monitor.py reset-drawdown` to reset
- This can happen if positions reconcile during trade execution

### No signals detected
- Flow detector needs time to build market state (~1 min)
- Lower `--min-score` or `--min-trade-size`

### Circuit breaker triggered
```bash
python scripts/risk_monitor.py status   # Check what happened
python scripts/risk_monitor.py cleanup  # Reset circuit breaker
```

---

## Testing

```bash
pytest tests/
pytest tests/test_risk_engine_integration.py -v
pytest -k "test_reservation" -v
```

---

## License

MIT License
