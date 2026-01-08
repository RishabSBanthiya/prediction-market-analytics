# Polymarket Analytics

Multi-agent trading infrastructure for Polymarket prediction markets.

```
Strategies: Bond (expiring markets) | Flow (copy trading) | Arbitrage (delta-neutral)
Features:   Atomic capital reservation | Real-time flow detection | On-chain sync
```

---

## Architecture Overview

```mermaid
flowchart TB
    subgraph External[External APIs]
        RTDS[RTDS WebSocket<br/>Real-time Trades]
        GAMMA[Gamma API<br/>Market Data]
        CLOB[CLOB API<br/>Orders & Orderbook]
        DATA[Data API<br/>Positions]
        POLY[Polygon RPC<br/>USDC Balance]
    end

    subgraph Detection[Signal Detection]
        FLOW_DET[Flow Detector<br/>WebSocket Consumer]
        ALERTS[Flow Alerts<br/>Smart Money, Volume, etc]
    end

    subgraph Agents[Trading Agents]
        BOND[Bond Bot<br/>Expiring Markets]
        FLOW[Flow Bot<br/>Copy Trading]
        ARB[Arb Bot<br/>Delta-Neutral]
    end

    subgraph Core[Core Infrastructure]
        RISK[Risk Coordinator<br/>Capital Management]
        SAFETY[Safety Module<br/>Circuit Breaker, Drawdown]
        STORE[(SQLite WAL<br/>State & History)]
        SYNC[Chain Sync<br/>On-chain Truth]
    end

    RTDS --> FLOW_DET
    FLOW_DET --> ALERTS
    ALERTS --> FLOW

    GAMMA --> BOND
    GAMMA --> FLOW
    GAMMA --> ARB

    CLOB --> BOND
    CLOB --> FLOW
    CLOB --> ARB

    DATA --> RISK
    POLY --> RISK
    POLY --> SYNC

    BOND --> RISK
    FLOW --> RISK
    ARB --> RISK

    RISK --> SAFETY
    RISK --> STORE
    SYNC --> STORE
    RISK --> CLOB
```

---

## Trading Flow

```mermaid
sequenceDiagram
    participant Signal as Signal Source
    participant Bot as Trading Bot
    participant Risk as Risk Coordinator
    participant Safety as Safety Module
    participant Exec as Executor
    participant CLOB as CLOB API
    participant Exit as Exit Monitor

    Signal->>Bot: Generate Signal (score, direction)
    Bot->>Bot: Calculate Position Size
    Bot->>Risk: Request Capital Reservation

    Risk->>Safety: Check Circuit Breaker
    Safety-->>Risk: OK/BLOCKED

    Risk->>Safety: Check Drawdown Limits
    Safety-->>Risk: OK/BLOCKED

    Risk->>Risk: Check Exposure Limits<br/>(Wallet 80%, Agent 40%, Market 15%)

    alt All Checks Pass
        Risk->>Risk: Atomic Reserve (SQLite TX)
        Risk-->>Bot: Reservation ID

        Bot->>Exec: Execute Trade
        Exec->>CLOB: Place IOC Order
        CLOB-->>Exec: Fill Result
        Exec-->>Bot: Execution Result

        Bot->>Risk: Confirm Execution
        Risk->>Risk: Convert Reservation to Position

        Bot->>Exit: Register for Exit Monitoring

        loop Monitor Position
            Exit->>Exit: Check TP/SL/Trailing/MaxHold
            alt Exit Triggered
                Exit->>Exec: Exit Trade
                Exec->>CLOB: Place Exit Order
            end
        end
    else Check Failed
        Risk-->>Bot: Reject (reason)
    end
```

---

## Quick Start

```bash
# Setup
git clone <repo-url> && cd polymarket-analytics
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your keys

# Run bots (dry-run first)
python scripts/run_bot.py bond --dry-run
python scripts/run_bot.py flow --dry-run
python scripts/run_arb_bot.py --dry-run

# Monitor
python scripts/risk_monitor.py status
```

---

## Strategies

### Bond Strategy (Expiring Markets)

Buys markets at 95-98c near expiration, expecting resolution to $1.

```mermaid
flowchart LR
    SCAN[Scan Markets<br/>Expiring 1-30min] --> FILTER[Filter<br/>Price 95-98c]
    FILTER --> SCORE[Score<br/>Time + Return]
    SCORE --> SIZE[Kelly Sizing<br/>Half-Kelly, 25% max]
    SIZE --> EXEC[Execute IOC]
    EXEC --> HEDGE[Hedge Monitor<br/>Adverse Moves]
```

```bash
python scripts/run_bot.py bond --dry-run --interval 10
python scripts/run_bot.py bond --agent-id bond-1 --min-price 0.94 --max-price 0.99
```

### Flow Strategy (Copy Trading)

Copies unusual flow from smart money, oversized bets, coordinated wallets.

```mermaid
flowchart LR
    WS[WebSocket<br/>Real-time Trades] --> DETECT[Flow Detector]
    DETECT --> SCORE[Score Signals<br/>Weighted Sum]

    subgraph Signals[Signal Types]
        SM[Smart Money 30x]
        OB[Oversized Bet 25x]
        CW[Coordinated 25x]
        VS[Volume Spike 10x]
    end

    Signals --> SCORE
    SCORE --> DECAY[Time Decay<br/>Freshness]
    DECAY --> SIZE[Signal-Scaled<br/>Position Size]
    SIZE --> EXEC[Execute]
    EXEC --> EXIT[Exit Monitor<br/>TP/SL/Trail/MaxHold]
```

```bash
python scripts/run_bot.py flow --dry-run --min-score 30
python scripts/run_bot.py flow --agent-id flow-1 --category crypto
```

### Arbitrage Strategy (Delta-Neutral)

Risk-free arbitrage on binary markets where both sides sum to < $1.

```mermaid
flowchart LR
    SCAN[Scan 15-min<br/>Crypto Markets] --> FIND[Find Edge<br/>UP + DOWN < 1.0]
    FIND --> CALC[Calculate<br/>Net Profit]
    CALC --> BUY_BOTH[Buy Both Sides<br/>Limit Orders]
    BUY_BOTH --> WAIT[Wait for<br/>Resolution]
    WAIT --> PROFIT[Guaranteed<br/>Profit]
```

```bash
python scripts/run_arb_bot.py --dry-run
python scripts/run_arb_bot.py --min-edge 0.02 --position-size 50
```

---

## Risk Management

```mermaid
flowchart TB
    REQ[Trade Request] --> CB{Circuit Breaker<br/>< 5 failures?}
    CB -->|OPEN| REJECT[Reject]
    CB -->|CLOSED| DD{Drawdown OK?<br/>Daily < 10%<br/>Total < 25%}
    DD -->|BREACH| REJECT
    DD -->|OK| WALLET{Wallet Limit<br/>< 80% equity?}
    WALLET -->|EXCEED| REJECT
    WALLET -->|OK| AGENT{Agent Limit<br/>< 40% equity?}
    AGENT -->|EXCEED| REJECT
    AGENT -->|OK| MARKET{Market Limit<br/>< 15% equity?}
    MARKET -->|EXCEED| REJECT
    MARKET -->|OK| RESERVE[Atomic Reserve<br/>SQLite Transaction]

    style REJECT fill:#ffcccc
    style RESERVE fill:#ccffcc
```

| Limit | Value | Description |
|-------|-------|-------------|
| Wallet Exposure | 80% | Max total exposure |
| Agent Exposure | 40% | Max per trading agent |
| Market Exposure | 15% | Max per single market |
| Daily Drawdown | 10% | Stop trading for day |
| Total Drawdown | 25% | Stop trading entirely |
| Circuit Breaker | 5 | Consecutive failures |

### Monitoring Commands

```bash
python scripts/risk_monitor.py status         # Wallet & risk overview
python scripts/risk_monitor.py agents         # List agents
python scripts/risk_monitor.py positions      # Open positions
python scripts/risk_monitor.py drawdown       # Drawdown status
python scripts/risk_monitor.py sync           # Force chain sync
python scripts/risk_monitor.py reset-drawdown # Reset DD tracking
python scripts/risk_monitor.py stop-all       # Emergency stop
```

---

## API Reference

### External APIs

| API | Base URL | Purpose | Rate Limit |
|-----|----------|---------|------------|
| **RTDS WebSocket** | `wss://ws-live-data.polymarket.com` | Real-time trades stream | N/A |
| **Gamma API** | `https://gamma-api.polymarket.com` | Market metadata, resolution | 4,000/10s |
| **CLOB API** | `https://clob.polymarket.com` | Orderbook, place/cancel orders | 9,000/10s |
| **Data API** | `https://data-api.polymarket.com` | Positions, trade history | 1,000/10s |
| **Polygon RPC** | Configurable | USDC balance, on-chain data | Varies |

### Key Endpoints

```
# CLOB API
GET  /book?token_id={id}           # Orderbook snapshot
GET  /price?token_id={id}          # Current mid price
POST /order                         # Place order
DELETE /order/{id}                  # Cancel order
GET  /orders?market={id}           # List orders

# Gamma API
GET  /markets                       # All markets
GET  /markets/{id}                  # Market details
GET  /markets?closed=false          # Active markets

# Data API
GET  /positions?user={addr}         # User positions
GET  /activity?user={addr}          # Trade history
```

### Internal APIs (Python)

```python
# Risk Coordinator
coordinator.atomic_reserve(agent_id, market_id, token_id, amount_usd)
coordinator.confirm_execution(reservation_id, filled_shares, filled_price)
coordinator.release_reservation(reservation_id)
coordinator.get_wallet_state() -> WalletState

# Trading Bot
bot = TradingBot(signal_source, position_sizer, executor, exit_config)
await bot.start()
await bot.stop()

# Flow Detector
detector = FlowDetector(on_alert_callback)
await detector.start()
```

---

## Data Models

```mermaid
classDiagram
    class Signal {
        +str market_id
        +str token_id
        +SignalDirection direction
        +float score
        +str source
        +datetime timestamp
    }

    class Position {
        +str agent_id
        +str market_id
        +str token_id
        +float shares
        +float entry_price
        +PositionStatus status
    }

    class Reservation {
        +str id
        +str agent_id
        +str market_id
        +float amount_usd
        +ReservationStatus status
        +datetime expires_at
    }

    class WalletState {
        +str wallet_address
        +float usdc_balance
        +float total_positions_value
        +float total_reserved
        +List~Position~ positions
    }

    Signal --> Position : triggers
    Reservation --> Position : converts to
    WalletState --> Position : contains
```

---

## Project Structure

```
polymarket-analytics/
├── polymarket/
│   ├── core/                      # Shared infrastructure
│   │   ├── api.py                 # Async Polymarket API client
│   │   ├── config.py              # Configuration management
│   │   ├── models.py              # All dataclasses
│   │   └── rate_limiter.py        # Sliding window limiter
│   │
│   ├── trading/                   # Live trading
│   │   ├── bot.py                 # TradingBot (composition-based)
│   │   ├── risk_coordinator.py    # Multi-agent risk management
│   │   ├── chain_sync.py          # On-chain transaction sync
│   │   ├── safety.py              # Circuit breaker, drawdown
│   │   ├── storage/sqlite.py      # SQLite persistence (WAL)
│   │   └── components/            # Pluggable components
│   │       ├── signals.py         # Signal sources
│   │       ├── sizers.py          # Position sizers
│   │       ├── executors.py       # Execution engines
│   │       └── exit_strategies.py # Exit monitors
│   │
│   ├── strategies/                # Strategy implementations
│   │   ├── bond_strategy.py       # Expiring markets
│   │   ├── flow_strategy.py       # Flow copy trading
│   │   └── arb_strategy.py        # Delta-neutral arb
│   │
│   ├── flow_detector.py           # Real-time flow detection
│   │
│   └── backtesting/               # Backtesting framework
│       ├── base.py                # BaseBacktester
│       ├── optimization.py        # Bayesian optimizer
│       └── strategies/            # Strategy backtests
│
├── scripts/                       # CLI tools
│   ├── run_bot.py                 # Main bot entry
│   ├── run_arb_bot.py             # Arbitrage bot
│   └── risk_monitor.py            # Monitoring CLI
│
└── data/                          # SQLite databases
```

---

## Backtesting

```bash
# Run backtests
python -m polymarket.backtesting.strategies.bond_backtest --backtest
python -m polymarket.backtesting.strategies.flow_backtest --backtest
python -m polymarket.backtesting.strategies.arb_backtest --backtest

# Parameter optimization (Bayesian, anti-overfitting)
python -m polymarket.backtesting.strategies.bond_backtest --optimize -n 50
python -m polymarket.backtesting.strategies.flow_backtest --optimize -n 50
```

**Anti-Overfitting:** 3 parameters only, walk-forward validation, L2 regularization, bootstrap confidence.

---

## Configuration

```bash
# .env file
PRIVATE_KEY=0x...
POLYMARKET_PROXY_ADDRESS=0x...
POLYGON_RPC_URL=https://polygon-rpc.com

# Risk limits
MAX_WALLET_EXPOSURE_PCT=0.80
MAX_PER_AGENT_EXPOSURE_PCT=0.40
MAX_PER_MARKET_EXPOSURE_PCT=0.15
MAX_DAILY_DRAWDOWN_PCT=0.10
MAX_TOTAL_DRAWDOWN_PCT=0.25
CIRCUIT_BREAKER_FAILURES=5
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Bot not starting | Check `.env` has `PRIVATE_KEY` and `POLYMARKET_PROXY_ADDRESS` |
| Rate limit errors | Increase `--interval`, check API limits |
| Phantom drawdown | Run `risk_monitor.py reset-drawdown` |
| No signals | Lower `--min-score`, wait for flow detector warmup (~1 min) |
| Circuit breaker | Run `risk_monitor.py cleanup` to reset |

---

## License

MIT
