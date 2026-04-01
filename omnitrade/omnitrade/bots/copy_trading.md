# Copy Trading Bot

## Overview

Tracks target wallet addresses on Polymarket and copies their trades with configurable size scaling and price adjustments. Uses Polymarket's public positions API — no authentication needed to read target accounts.

Kalshi is not supported (no public portfolio API).

## How It Works

The bot runs a continuous loop at a configurable interval (default 30s):

1. **Update equity** — fetch balance, update risk coordinator for drawdown tracking
2. **Poll targets** — fetch current positions for each tracked address via `data-api.polymarket.com`
3. **Diff snapshots** — compare current positions against the previous snapshot to detect new, increased, decreased, or closed positions
4. **First-poll guard** — the initial poll only snapshots existing positions without copying (avoids blindly entering all existing positions)
5. **Filter & adjust** — check price bounds, price deviation tolerance, cooldowns, and minimum delta size
6. **Copy trades** — size the trade (target_size x weight x multiplier), reserve capital through risk, execute
7. **Copy exits** — when a target closes a position, optionally close the mirrored position

## Components

| Component | Module | Role |
|-----------|--------|------|
| `TargetTracker` | `bots.copy_trading` | Polls Polymarket positions API, maintains snapshots, produces `PositionDelta` diffs |
| `TargetAccount` | `bots.copy_trading` | Wallet address + label + weight for a single tracked trader |
| `CopyConfig` | `bots.copy_trading` | Size multiplier, price bounds, deviation tolerance, cooldowns |
| `RiskCoordinator` | `risk.coordinator` | Capital reservation, drawdown limits, failure tracking |

## Configuration

### CopyConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size_multiplier` | `float` | `1.0` | Scale factor applied to target's trade size |
| `min_trade_usd` | `float` | `5.0` | Minimum trade value in USD |
| `max_trade_usd` | `float` | `500.0` | Maximum trade value in USD |
| `max_price_deviation_pct` | `float` | `0.05` | Skip if current price deviates more than 5% from target's entry |
| `min_price` | `float` | `0.05` | Minimum price filter (normalized 0-1) |
| `max_price` | `float` | `0.95` | Maximum price filter (normalized 0-1) |
| `copy_exits` | `bool` | `True` | Whether to mirror position closes |
| `min_delta_size` | `float` | `1.0` | Minimum position change (shares) to trigger a copy |
| `cooldown_seconds` | `float` | `60.0` | Wait time after copying before copying the same instrument again |

### TargetAccount

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `address` | `str` | required | Polymarket wallet address (e.g., `0xABC...`) |
| `label` | `str` | `""` | Human-readable label for logging |
| `weight` | `float` | `1.0` | Relative sizing weight (0-1) — e.g., 0.5 means copy at half size |

## CLI Usage

```bash
# Copy a single address (paper mode, --exchange not needed)
python scripts/run_bot.py copy -t 0xADDRESS

# Copy multiple addresses with labels and weights
python scripts/run_bot.py copy \
  -t 0xAAA:whale:1.0 \
  -t 0xBBB:smart-money:0.5

# Use a targets file
python scripts/run_bot.py copy --targets-file targets.json

# Adjust size multiplier and disable exit copying
python scripts/run_bot.py copy -t 0xABC --size-multiplier 0.25 --no-copy-exits

# Live mode
python scripts/run_bot.py copy -t 0xABC --live
```

### Targets file format (`targets.json`)

```json
[
  {"address": "0xAAA", "label": "whale", "weight": 1.0},
  {"address": "0xBBB", "label": "degen", "weight": 0.5},
  "0xCCC"
]
```

## Data Source

| Exchange | Endpoint | Auth |
|----------|----------|------|
| Polymarket | `https://data-api.polymarket.com/positions?user=<address>` | None (public) |

## Risk Management

- **Target validation** — on startup, each target address is checked against the API; invalid targets are dropped
- **Capital reservation** — `risk.atomic_reserve()` locks capital before execution; released on failure
- **Drawdown tracking** — equity updates every iteration feed the risk coordinator's drawdown monitor
- **Price filters** — `min_price`/`max_price` avoid extreme-probability contracts
- **Price deviation guard** — skips copy if market has moved too far since the target traded
- **Cooldown** — prevents rapid-fire copies of the same instrument
- **First-poll guard** — initial poll only snapshots, doesn't copy existing positions
- **Size capping** — trades capped by `max_trade_usd` and available balance
