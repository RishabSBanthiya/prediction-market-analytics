"""
Flow Strategy Backtest with Historical Trade Data.

Uses actual trade flow data (wallet addresses, trade sizes) to backtest
the flow copy strategy more accurately than momentum-based approaches.

Key difference from flow_backtest.py:
- Uses historical trades with wallet addresses
- Identifies "smart money" wallets from historical win rates
- Simulates following smart money trades

Run backtest:
    python -m polymarket.backtesting.strategies.flow_trade_backtest --backtest

Note: Requires fetching trade data first (slow due to API rate limits).
"""

import argparse
import asyncio
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set, Tuple

from ...core.config import get_config
from ...core.api import PolymarketAPI
from ..results import BacktestResults, SimulatedTrade

logger = logging.getLogger(__name__)

# Smart money detection thresholds
MIN_TRADES_FOR_SMART_MONEY = 10
MIN_WIN_RATE_FOR_SMART_MONEY = 0.55
MIN_TRADE_SIZE_USD = 500  # Only track trades >= $500

# Sports keywords to filter
SPORTS_KEYWORDS = [
    "nba", "nfl", "nhl", "mlb", "cfb", "cbb", "mls", "ufc", "pga",
    "premier league", "champions league", " vs ", " v ",
    "game ", "match ", "fight ",
]


@dataclass
class WalletStats:
    """Track wallet performance for smart money detection."""
    address: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    markets_traded: Set[str] = field(default_factory=set)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0

    @property
    def is_smart_money(self) -> bool:
        return (
            self.trades >= MIN_TRADES_FOR_SMART_MONEY and
            self.win_rate >= MIN_WIN_RATE_FOR_SMART_MONEY and
            self.total_pnl > 0
        )


@dataclass
class HistoricalTrade:
    """A historical trade with wallet info."""
    trade_id: str
    market_id: str
    token_id: str
    wallet: str
    side: str  # BUY or SELL
    price: float
    size: float
    value_usd: float
    timestamp: datetime
    market_question: str = ""
    outcome: str = ""


class FlowTradeBacktester:
    """
    Backtest flow copy strategy using historical trade data.

    Process:
    1. Fetch historical trades for resolved markets
    2. Build wallet profiles from early trades
    3. Identify smart money wallets
    4. Simulate copying their later trades
    5. Evaluate P&L based on market resolutions
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.10,
        min_trade_to_copy: float = 500.0,
    ):
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.min_trade_to_copy = min_trade_to_copy

        self.wallet_stats: Dict[str, WalletStats] = {}
        self.trades_by_market: Dict[str, List[HistoricalTrade]] = defaultdict(list)
        self.market_resolutions: Dict[str, Dict[str, float]] = {}  # market_id -> {token_id: resolution_price}

    def _get_wallet(self, address: str) -> WalletStats:
        """Get or create wallet stats."""
        if address not in self.wallet_stats:
            self.wallet_stats[address] = WalletStats(address=address)
        return self.wallet_stats[address]

    def _is_sports_market(self, question: str) -> bool:
        """Check if market is sports-related."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in SPORTS_KEYWORDS)

    def add_trade(self, trade: HistoricalTrade):
        """Add a historical trade to the dataset."""
        self.trades_by_market[trade.market_id].append(trade)

    def set_resolution(self, market_id: str, token_id: str, resolution_price: float):
        """Set market resolution (0 or 1 typically)."""
        if market_id not in self.market_resolutions:
            self.market_resolutions[market_id] = {}
        self.market_resolutions[market_id][token_id] = resolution_price

    def _calculate_trade_pnl(
        self,
        trade: HistoricalTrade,
    ) -> Optional[float]:
        """Calculate P&L for a trade based on resolution."""
        resolution = self.market_resolutions.get(trade.market_id, {})
        resolution_price = resolution.get(trade.token_id)

        if resolution_price is None:
            return None  # Market not resolved

        if trade.side == "BUY":
            # Bought shares, profit if resolved to $1
            pnl = (resolution_price - trade.price) * trade.size
        else:
            # Sold shares, profit if resolved to $0
            pnl = (trade.price - resolution_price) * trade.size

        return pnl

    def build_wallet_profiles(self, training_cutoff: datetime):
        """
        Build wallet profiles from trades before the cutoff.

        This simulates having historical data to identify smart money
        before we start copying their trades.
        """
        for market_id, trades in self.trades_by_market.items():
            for trade in trades:
                if trade.timestamp >= training_cutoff:
                    continue  # Only use training data

                if trade.value_usd < MIN_TRADE_SIZE_USD:
                    continue  # Skip small trades

                wallet = self._get_wallet(trade.wallet)
                wallet.trades += 1
                wallet.markets_traded.add(market_id)

                pnl = self._calculate_trade_pnl(trade)
                if pnl is not None:
                    wallet.total_pnl += pnl
                    if pnl > 0:
                        wallet.wins += 1
                    else:
                        wallet.losses += 1

        # Count smart money wallets
        smart_money_count = sum(1 for w in self.wallet_stats.values() if w.is_smart_money)
        logger.info(f"Identified {smart_money_count} smart money wallets from {len(self.wallet_stats)} total")

    def run_backtest(
        self,
        test_start: datetime,
        test_end: datetime,
    ) -> BacktestResults:
        """
        Run backtest by simulating copying smart money trades.

        Args:
            test_start: Start of test period (after training)
            test_end: End of test period
        """
        results = BacktestResults(
            strategy_name="Flow Copy (Trade-Based)",
            start_date=test_start,
            end_date=test_end,
            initial_capital=self.initial_capital,
        )

        cash = self.initial_capital
        positions: Dict[str, Dict] = {}  # token_id -> {shares, entry_price, market_id}

        # Collect all trades in test period, sorted by time
        test_trades = []
        for market_id, trades in self.trades_by_market.items():
            for trade in trades:
                if test_start <= trade.timestamp <= test_end:
                    test_trades.append(trade)

        test_trades.sort(key=lambda t: t.timestamp)

        # Process trades chronologically
        trades_copied = 0
        trades_skipped_no_smart = 0
        trades_skipped_sports = 0
        trades_skipped_small = 0

        for trade in test_trades:
            # Skip sports markets
            if self._is_sports_market(trade.market_question):
                trades_skipped_sports += 1
                continue

            # Skip small trades
            if trade.value_usd < self.min_trade_to_copy:
                trades_skipped_small += 1
                continue

            # Check if wallet is smart money
            wallet = self.wallet_stats.get(trade.wallet)
            if not wallet or not wallet.is_smart_money:
                trades_skipped_no_smart += 1
                continue

            # Only copy BUY signals for simplicity
            if trade.side != "BUY":
                continue

            # Skip if we already have a position in this token
            if trade.token_id in positions:
                continue

            # Skip if price is too extreme (likely resolved)
            if trade.price >= 0.95 or trade.price <= 0.05:
                continue

            # Position sizing
            position_size = min(
                cash * self.max_position_pct,
                trade.value_usd * 0.5,  # Don't copy more than 50% of their size
            )

            if position_size < 10:
                continue

            # Execute entry
            shares = position_size / trade.price
            cost = shares * trade.price

            if cost > cash:
                continue

            cash -= cost
            positions[trade.token_id] = {
                "shares": shares,
                "entry_price": trade.price,
                "market_id": trade.market_id,
                "entry_time": trade.timestamp,
                "question": trade.market_question,
                "outcome": trade.outcome,
                "wallet": trade.wallet,
                "wallet_win_rate": wallet.win_rate,
            }
            trades_copied += 1

        # Resolve all positions
        for token_id, pos in positions.items():
            resolution = self.market_resolutions.get(pos["market_id"], {})
            exit_price = resolution.get(token_id)

            if exit_price is None:
                # Market not resolved, assume exit at entry (no P&L)
                exit_price = pos["entry_price"]

            proceeds = pos["shares"] * exit_price
            cash += proceeds
            pnl = proceeds - (pos["shares"] * pos["entry_price"])

            results.add_trade(SimulatedTrade(
                market_question=pos["question"][:80],
                token_id=token_id,
                token_outcome=pos["outcome"],
                entry_time=pos["entry_time"],
                entry_price=pos["entry_price"],
                exit_time=test_end,
                exit_price=exit_price,
                shares=pos["shares"],
                cost=pos["shares"] * pos["entry_price"],
                proceeds=proceeds,
                pnl=pnl,
                pnl_percent=pnl / (pos["shares"] * pos["entry_price"]) if pos["entry_price"] > 0 else 0,
                resolved_to=exit_price,
                held_to_resolution=True,
                reason=f"Copy smart money ({pos['wallet'][:10]}..., {pos['wallet_win_rate']:.0%} WR)",
            ))

        results.finalize()

        # Print stats
        print(f"\n--- FLOW COPY STATS ---")
        print(f"Trades copied:          {trades_copied}")
        print(f"Skipped (not smart):    {trades_skipped_no_smart}")
        print(f"Skipped (sports):       {trades_skipped_sports}")
        print(f"Skipped (too small):    {trades_skipped_small}")
        print(f"Smart money wallets:    {sum(1 for w in self.wallet_stats.values() if w.is_smart_money)}")

        return results


async def fetch_and_cache_trades(
    api: PolymarketAPI,
    markets: List[dict],
    cache_path: str = "data/trade_cache.db",
) -> int:
    """
    Fetch trades for markets and cache to SQLite.

    Uses authenticated CLOB client for full trade data with wallet addresses.
    Returns number of trades fetched.
    """
    import os
    from py_clob_client.client import ClobClient

    # Create cache directory
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    conn = sqlite3.connect(cache_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            market_id TEXT,
            token_id TEXT,
            wallet TEXT,
            side TEXT,
            price REAL,
            size REAL,
            value_usd REAL,
            timestamp TEXT,
            market_question TEXT,
            outcome TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_market ON trades(market_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_wallet ON trades(wallet)")
    conn.commit()

    # Initialize authenticated CLOB client with L2 credentials
    try:
        from py_clob_client.clob_types import ApiCreds

        api_key = os.getenv("POLYMARKET_API_KEY")
        api_secret = os.getenv("POLYMARKET_API_SECRET")
        api_passphrase = os.getenv("POLYMARKET_API_PASSPHRASE")

        if not all([api_key, api_secret, api_passphrase]):
            print("Missing L2 API credentials. Run this to generate:")
            print("  python -c \"from py_clob_client.client import ClobClient; ...")
            print("  client.create_or_derive_api_creds()\"")
            return 0

        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )

        clob_client = ClobClient(
            host="https://clob.polymarket.com",
            key=api.config.private_key,
            chain_id=137,
            creds=creds,
        )
        print("Using authenticated CLOB client with L2 credentials")
    except Exception as e:
        print(f"Failed to init CLOB client: {e}")
        print("Trade fetching requires L2 API credentials in .env")
        return 0

    total_trades = 0

    for i, market in enumerate(markets):
        if (i + 1) % 10 == 0:
            print(f"Fetching trades: {i+1}/{len(markets)}... ({total_trades} trades)")

        market_id = market.get("condition_id", "")
        question = market.get("question", "")

        for token in market.get("tokens", []):
            token_id = token.get("token_id", "")
            outcome = token.get("outcome", "")

            try:
                # Use authenticated CLOB client to get trades with wallet addresses
                from py_clob_client.clob_types import TradeParams
                params = TradeParams(asset_id=token_id)
                trades_data = clob_client.get_trades(params)

                if not trades_data:
                    continue

                for trade in trades_data:
                    # Extract wallet (taker is the active party)
                    wallet = trade.get("taker") or trade.get("maker") or ""
                    if not wallet:
                        continue

                    price = float(trade.get("price", 0))
                    size = float(trade.get("size", 0))

                    # Parse timestamp
                    ts_str = trade.get("created_at") or trade.get("timestamp", "")
                    try:
                        if isinstance(ts_str, (int, float)):
                            ts = datetime.fromtimestamp(ts_str, tz=timezone.utc)
                        else:
                            ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                    except:
                        ts = datetime.now(timezone.utc)

                    trade_id = trade.get("id") or f"{token_id}_{ts.timestamp()}"

                    conn.execute("""
                        INSERT OR IGNORE INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        trade_id,
                        market_id,
                        token_id,
                        wallet.lower(),
                        trade.get("side", "BUY").upper(),
                        price,
                        size,
                        price * size,
                        ts.isoformat(),
                        question,
                        outcome,
                    ))
                    total_trades += 1

                conn.commit()
                await asyncio.sleep(0.15)  # Rate limit

            except Exception as e:
                logger.debug(f"Error fetching trades for {token_id}: {e}")

    conn.close()
    return total_trades


def load_trades_from_cache(cache_path: str = "data/trade_cache.db") -> List[HistoricalTrade]:
    """Load trades from SQLite cache."""
    import os

    if not os.path.exists(cache_path):
        return []

    conn = sqlite3.connect(cache_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT * FROM trades").fetchall()
    conn.close()

    trades = []
    for row in rows:
        trades.append(HistoricalTrade(
            trade_id=row["trade_id"],
            market_id=row["market_id"],
            token_id=row["token_id"],
            wallet=row["wallet"],
            side=row["side"],
            price=row["price"],
            size=row["size"],
            value_usd=row["value_usd"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            market_question=row["market_question"],
            outcome=row["outcome"],
        ))

    return trades


async def run_trade_flow_backtest(
    days: int = 90,
    capital: float = 1000.0,
    fetch_trades: bool = False,
    max_markets: int = 100,
) -> BacktestResults:
    """
    Run flow copy backtest with historical trade data.

    Args:
        days: Days of history
        capital: Initial capital
        fetch_trades: If True, fetch new trades (slow). If False, use cache.
        max_markets: Max markets to process
    """
    config = get_config()
    api = PolymarketAPI(config)
    await api.connect()

    try:
        print(f"\n{'='*60}")
        print("FLOW COPY BACKTEST (Trade-Based)")
        print(f"{'='*60}")

        # Fetch resolved markets
        print(f"Fetching resolved markets (last {days} days)...")
        raw_markets = await api.fetch_closed_markets(days=days)

        markets = []
        resolutions = {}

        for raw in raw_markets[:max_markets]:
            m = api.parse_market(raw)
            if not m:
                continue

            markets.append({
                "condition_id": m.condition_id,
                "question": m.question,
                "tokens": [{"token_id": t.token_id, "outcome": t.outcome, "price": t.price} for t in m.tokens],
            })

            # Store resolutions
            for t in m.tokens:
                if m.condition_id not in resolutions:
                    resolutions[m.condition_id] = {}
                # Price at resolution (0 or 1 for resolved markets)
                resolutions[m.condition_id][t.token_id] = t.price

        print(f"Found {len(markets)} resolved markets")

        # Fetch or load trades
        cache_path = "data/trade_cache.db"

        if fetch_trades:
            print("Fetching trades (this may take a while)...")
            total = await fetch_and_cache_trades(api, markets, cache_path)
            print(f"Fetched {total} trades")

        trades = load_trades_from_cache(cache_path)
        print(f"Loaded {len(trades)} trades from cache")

        if not trades:
            print("\nNo trades in cache. Run with --fetch-trades to download trade data.")
            return None

        # Create backtester
        backtester = FlowTradeBacktester(
            initial_capital=capital,
            max_position_pct=0.10,
            min_trade_to_copy=500.0,
        )

        # Add trades and resolutions
        for trade in trades:
            backtester.add_trade(trade)

        for market_id, token_resolutions in resolutions.items():
            for token_id, price in token_resolutions.items():
                backtester.set_resolution(market_id, token_id, price)

        # Split into training (first 60%) and test (last 40%)
        all_times = [t.timestamp for t in trades]
        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            training_cutoff = min_time + (max_time - min_time) * 0.6

            print(f"\nTraining period: {min_time.date()} to {training_cutoff.date()}")
            print(f"Test period:     {training_cutoff.date()} to {max_time.date()}")

            # Build wallet profiles from training data
            backtester.build_wallet_profiles(training_cutoff)

            # Run backtest on test period
            results = backtester.run_backtest(training_cutoff, max_time)
            results.print_report()

            return results

        return None

    finally:
        await api.close()


def main():
    parser = argparse.ArgumentParser(description="Flow Copy Backtest (Trade-Based)")
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--fetch-trades', action='store_true', help='Fetch new trade data (slow)')
    parser.add_argument('--capital', type=float, default=1000.0, help='Initial capital')
    parser.add_argument('--days', type=int, default=90, help='Days of history')
    parser.add_argument('--max-markets', type=int, default=100, help='Max markets to process')

    args = parser.parse_args()

    if args.backtest or args.fetch_trades:
        asyncio.run(run_trade_flow_backtest(
            days=args.days,
            capital=args.capital,
            fetch_trades=args.fetch_trades,
            max_markets=args.max_markets,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
