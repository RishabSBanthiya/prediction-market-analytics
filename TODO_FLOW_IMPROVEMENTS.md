# Flow Strategy Improvements

Status: **COMPLETED** - V4 Enhanced Flow Strategy implemented

## Problem Summary

The simple flow backtest uses **price momentum only** (2% move = signal), but the live strategy uses **rich flow detection** (whale trades, wallet profiles, coordinated activity). This mismatch causes backtest failure.

## Solution

All 7 improvements have been implemented in:
- `polymarket/trading/components/flow_enhancements.py` - Core enhancement components
- `polymarket/strategies/flow_strategy.py` - Integrated `EnhancedFlowCopySignalSource` class

### Usage

Run the enhanced V4 flow strategy:
```bash
# Run with all V4 enhancements enabled
python -m polymarket.strategies.flow_strategy --enhanced --dry-run

# Disable specific features
python -m polymarket.strategies.flow_strategy --enhanced --no-contrarian --dry-run
```

---

## Priority 1: Wallet Reputation Scoring (High Impact, Medium Effort)

Track wallet historical accuracy before copying trades.

```python
@dataclass
class WalletScore:
    address: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def is_smart_money(self) -> bool:
        return self.trades >= 10 and self.win_rate >= 0.55 and self.total_pnl > 0

def should_copy_trade(wallet: WalletScore, trade_size: float) -> bool:
    if not wallet.is_smart_money:
        return False
    min_size = 1000 if wallet.trades >= 50 else 5000
    return trade_size >= min_size
```

---

## Priority 2: Multi-Signal Confirmation (High Impact, Low Effort)

Require multiple confirming signals before trading.

```python
@dataclass
class SignalCluster:
    token_id: str
    direction: str
    signals: List[str]  # ["whale_buy", "smart_money", "momentum"]
    total_score: float

    @property
    def is_strong(self) -> bool:
        unique_types = set(s.split("_")[0] for s in self.signals)
        return len(unique_types) >= 2 and self.total_score >= 50
```

---

## Priority 3: Information Timing Filter (High Impact, Medium Effort)

Only trade on EARLY signals, not stale ones.

```python
def is_early_signal(trade: TradeEvent, market_state: MarketState) -> bool:
    recent_trades = list(market_state.recent_trades)

    # First large trade in this direction?
    same_direction_large = [
        t for t in recent_trades[-20:]
        if t.side == trade.side and t.value_usd >= 2000 and t != trade
    ]
    if len(same_direction_large) > 2:
        return False

    # Price hasn't moved much?
    price_change = market_state.get_price_change(window_seconds=3600)
    if price_change and abs(price_change) > 0.03:
        return False

    # Low recent volume?
    hour_volume = sum(t.value_usd for t in recent_trades[-30:])
    if hour_volume > 50000:
        return False

    return True
```

---

## Priority 4: Contrarian Signal - Fade the Crowd (Medium Impact, Medium Effort)

Detect when retail is wrong and fade them.

```python
def detect_contrarian_signal(market_state: MarketState) -> Optional[Signal]:
    recent_trades = list(market_state.recent_trades)[-50:]

    small_buys = sum(1 for t in recent_trades if t.side == "BUY" and t.value_usd < 500)
    small_sells = sum(1 for t in recent_trades if t.side == "SELL" and t.value_usd < 500)
    large_buys = sum(1 for t in recent_trades if t.side == "BUY" and t.value_usd >= 5000)
    large_sells = sum(1 for t in recent_trades if t.side == "SELL" and t.value_usd >= 5000)

    # Retail FOMO, whales selling -> SELL
    if small_buys > small_sells * 3 and large_sells > large_buys * 2:
        return Signal(direction="SELL", reason="fade_retail_fomo")

    # Retail panic, whales buying -> BUY
    if small_sells > small_buys * 3 and large_buys > large_sells * 2:
        return Signal(direction="BUY", reason="fade_retail_panic")

    return None
```

---

## Priority 5: Market Context Filters (Medium Impact, Low Effort)

Adjust signal weight based on market characteristics.

```python
def calculate_context_multiplier(market: MarketState) -> float:
    multiplier = 1.0

    # Time to resolution
    if market.end_date:
        hours_left = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
        if hours_left < 24:
            multiplier *= 1.5  # Near-term = higher confidence
        elif hours_left > 720:
            multiplier *= 0.5  # Far out = more uncertainty

    # Liquidity (spread)
    spread = (market.best_ask - market.best_bid) / market.best_bid if market.best_bid else 0.1
    if spread > 0.05:
        multiplier *= 0.5
    elif spread < 0.02:
        multiplier *= 1.2

    # Price extremes
    if market.current_price < 0.10 or market.current_price > 0.90:
        multiplier *= 0.7

    return multiplier
```

---

## Priority 6: Dynamic Position Sizing (Medium Impact, Low Effort)

Scale position with signal confidence.

```python
def calculate_position_size(
    signal: SignalCluster,
    wallet_score: Optional[WalletScore],
    context_mult: float,
    base_pct: float = 0.05,
    max_pct: float = 0.15,
) -> float:
    size_pct = base_pct

    if signal.is_strong:
        size_pct *= 1.5

    if wallet_score and wallet_score.is_smart_money:
        size_pct *= (1 + wallet_score.win_rate - 0.5)

    size_pct *= context_mult

    return min(size_pct, max_pct)
```

---

## Priority 7: Adaptive Exits (Low Impact, Low Effort)

Exit parameters based on market dynamics.

```python
def get_exit_params(signal: SignalCluster, market: MarketState) -> ExitConfig:
    hours_left = (market.end_date - datetime.now(timezone.utc)).total_seconds() / 3600

    if hours_left < 24:
        # Near resolution: hold for outcome
        return ExitConfig(take_profit_pct=0.20, stop_loss_pct=0.05, max_hold_minutes=None)
    elif hours_left < 168:
        return ExitConfig(take_profit_pct=0.08, stop_loss_pct=0.06, max_hold_minutes=240)
    else:
        # Long-term: quick scalp
        return ExitConfig(take_profit_pct=0.05, stop_loss_pct=0.08, max_hold_minutes=60)
```

---

## Implementation Order

| # | Improvement | Impact | Effort | Status |
|---|-------------|--------|--------|--------|
| 1 | Wallet reputation scoring | High | Medium | DONE |
| 2 | Multi-signal confirmation | High | Low | DONE |
| 3 | Information timing filter | High | Medium | DONE |
| 4 | Contrarian signal | Medium | Medium | DONE |
| 5 | Market context filters | Medium | Low | DONE |
| 6 | Dynamic position sizing | Medium | Low | DONE |
| 7 | Adaptive exits | Low | Low | DONE |

All improvements implemented in V4 Enhanced Flow Strategy.
