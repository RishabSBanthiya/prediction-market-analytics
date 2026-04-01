"""Tests for data_loader: OrderbookReconstructor, NormalizedTrade, parse_ctf_fill, BlockTimestampLookup."""

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from omnitrade.backtest.data_loader import (
    NormalizedTrade,
    MarketInfo,
    OrderbookReconstructor,
    PolymarketDataLoader,
    parse_ctf_fill,
    BlockTimestampLookup,
)
from omnitrade.backtest.engine import BacktestRunner, BacktestResult
from omnitrade.core.models import OrderbookSnapshot
from omnitrade.components.signals import MidpointDeviationSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trades(
    n: int = 20,
    base_price: float = 0.55,
    spread: float = 0.02,
    start: datetime | None = None,
    interval_seconds: int = 5,
) -> list[NormalizedTrade]:
    """Create a list of synthetic NormalizedTrades for testing."""
    if start is None:
        start = datetime(2024, 6, 1, tzinfo=timezone.utc)

    trades = []
    for i in range(n):
        side = "BUY" if i % 2 == 0 else "SELL"
        offset = spread / 2 if side == "BUY" else -spread / 2
        trades.append(NormalizedTrade(
            asset_id="TOKEN-001",
            side=side,
            price=round(base_price + offset, 4),
            size=round(10.0 + i * 0.5, 2),
            timestamp=start + timedelta(seconds=i * interval_seconds),
            condition_id="CID-001",
        ))
    return trades


# ---------------------------------------------------------------------------
# NormalizedTrade
# ---------------------------------------------------------------------------

class TestNormalizedTrade:
    def test_basic_construction(self):
        t = NormalizedTrade(
            asset_id="abc",
            side="BUY",
            price=0.65,
            size=100.0,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            condition_id="cid-1",
        )
        assert t.asset_id == "abc"
        assert t.side == "BUY"
        assert t.price == 0.65
        assert t.size == 100.0

    def test_defaults(self):
        t = NormalizedTrade(
            asset_id="x",
            side="SELL",
            price=0.5,
            size=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert t.condition_id == ""


# ---------------------------------------------------------------------------
# OrderbookReconstructor
# ---------------------------------------------------------------------------

class TestOrderbookReconstructor:
    def test_empty_trades_returns_empty(self):
        recon = OrderbookReconstructor()
        result = recon.reconstruct([])
        assert result == []

    def test_basic_reconstruction(self):
        trades = _make_trades(n=10)
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades, "TEST-001")

        assert len(snapshots) > 0
        for snap in snapshots:
            assert snap.instrument_id == "TEST-001"
            assert len(snap.bids) > 0
            assert len(snap.asks) > 0

    def test_bid_less_than_ask(self):
        """Best bid must always be below best ask."""
        trades = _make_trades(n=40, base_price=0.6, spread=0.04)
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades)

        for snap in snapshots:
            if snap.best_bid is not None and snap.best_ask is not None:
                assert snap.best_bid < snap.best_ask, (
                    f"bid={snap.best_bid} >= ask={snap.best_ask}"
                )

    def test_monotonic_timestamps(self):
        """Snapshot timestamps must be non-decreasing."""
        trades = _make_trades(n=30)
        recon = OrderbookReconstructor(window_seconds=15)
        snapshots = recon.reconstruct(trades)

        for i in range(1, len(snapshots)):
            assert snapshots[i].timestamp >= snapshots[i - 1].timestamp

    def test_carry_forward_for_gaps(self):
        """Empty windows should carry forward previous snapshot."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Trades at t=0s and t=120s, gap in between
        trades = [
            NormalizedTrade("T1", "BUY", 0.60, 50, start, "C1"),
            NormalizedTrade("T1", "SELL", 0.58, 50, start, "C1"),
            NormalizedTrade("T1", "BUY", 0.62, 50, start + timedelta(seconds=120), "C1"),
            NormalizedTrade("T1", "SELL", 0.59, 50, start + timedelta(seconds=120), "C1"),
        ]
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades)

        # Should have >2 snapshots due to the gap being filled
        assert len(snapshots) >= 3
        # Middle snapshots should have valid bids/asks (carried forward)
        for snap in snapshots:
            assert len(snap.bids) > 0
            assert len(snap.asks) > 0

    def test_single_sided_trades_get_synthetic_opposite(self):
        """If only buys or only sells, the other side is synthesized."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        # Only buy trades
        trades = [
            NormalizedTrade("T1", "BUY", 0.55, 100, start + timedelta(seconds=i), "C1")
            for i in range(5)
        ]
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades)

        assert len(snapshots) >= 1
        snap = snapshots[0]
        assert len(snap.bids) > 0, "Bids should be synthesized"
        assert len(snap.asks) > 0, "Asks should come from buy trades"

    def test_configurable_depth(self):
        """Snapshots should respect depth_levels setting."""
        trades = _make_trades(n=20)
        for depth in [3, 5, 8]:
            recon = OrderbookReconstructor(depth_levels=depth)
            snapshots = recon.reconstruct(trades)
            for snap in snapshots:
                assert len(snap.bids) <= depth
                assert len(snap.asks) <= depth

    def test_prices_in_valid_range(self):
        """All prices must be in (0, 1)."""
        # Include extreme prices
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        trades = [
            NormalizedTrade("T1", "BUY", 0.99, 50, start, "C1"),
            NormalizedTrade("T1", "SELL", 0.01, 50, start + timedelta(seconds=1), "C1"),
            NormalizedTrade("T1", "BUY", 0.50, 50, start + timedelta(seconds=2), "C1"),
        ]
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades)

        for snap in snapshots:
            for level in snap.bids + snap.asks:
                assert 0 < level.price < 1, f"Price {level.price} out of range"

    def test_window_size_affects_snapshot_count(self):
        """Larger windows should produce fewer snapshots."""
        trades = _make_trades(n=60, interval_seconds=5)  # 300s of trades
        snap_small = OrderbookReconstructor(window_seconds=15).reconstruct(trades)
        snap_large = OrderbookReconstructor(window_seconds=60).reconstruct(trades)
        assert len(snap_small) > len(snap_large)


# ---------------------------------------------------------------------------
# Integration: BacktestRunner with external snapshots
# ---------------------------------------------------------------------------

class TestExternalSnapshots:
    def test_runner_accepts_external_snapshots(self):
        """BacktestRunner should work with pre-built snapshots."""
        trades = _make_trades(n=40, base_price=0.55)
        recon = OrderbookReconstructor(window_seconds=30)
        snapshots = recon.reconstruct(trades, "EXT-001")

        signal = MidpointDeviationSignal(fair_value=0.5, min_deviation=0.03)
        runner = BacktestRunner(
            signal_source=signal,
            snapshots=snapshots,
            instrument_id="EXT-001",
            scenario_name="real_data_test",
        )
        result = asyncio.run(runner.run())

        assert isinstance(result, BacktestResult)
        assert result.scenario_name == "real_data_test"
        assert result.signal_name == "midpoint_deviation"
        assert result.final_equity > 0
        assert len(result.equity_curve) > 0


# ---------------------------------------------------------------------------
# CTF fill parser
# ---------------------------------------------------------------------------

class TestParseCTFFill:
    def test_taker_buys_tokens(self):
        """taker_asset_id=0 means taker gave USDC -> BUY."""
        result = parse_ctf_fill(
            maker_amount=500_000,   # maker gives 0.5 token (6 decimals)
            taker_amount=300_000,   # taker gives 0.3 USDC (6 decimals)
            maker_asset_id="12345",
            taker_asset_id="0",
        )
        assert result is not None
        asset_id, side, price, size, fee = result
        assert side == "BUY"
        assert asset_id == "12345"
        assert abs(price - 0.6) < 0.001  # 0.3 USDC / 0.5 tokens
        assert abs(size - 0.5) < 0.001

    def test_taker_sells_tokens(self):
        """maker_asset_id=0 means maker gave USDC -> taker SELL."""
        result = parse_ctf_fill(
            maker_amount=400_000,   # maker gives 0.4 USDC
            taker_amount=800_000,   # taker gives 0.8 tokens
            maker_asset_id="0",
            taker_asset_id="99999",
        )
        assert result is not None
        asset_id, side, price, size, fee = result
        assert side == "SELL"
        assert asset_id == "99999"
        assert abs(price - 0.5) < 0.001  # 0.4 / 0.8

    def test_neither_side_usdc_returns_none(self):
        """Token-for-token swap should be skipped."""
        result = parse_ctf_fill(
            maker_amount=100_000,
            taker_amount=200_000,
            maker_asset_id="111",
            taker_asset_id="222",
        )
        assert result is None

    def test_zero_amounts_returns_none(self):
        result = parse_ctf_fill(0, 0, "0", "123")
        assert result is None

    def test_zero_token_amount_returns_none(self):
        """Avoid division by zero."""
        result = parse_ctf_fill(
            maker_amount=100_000,
            taker_amount=0,
            maker_asset_id="0",
            taker_asset_id="123",
        )
        assert result is None

    def test_price_clamped_high(self):
        """Price > 0.999 should be clamped."""
        result = parse_ctf_fill(
            maker_amount=1_000_000,   # 1.0 tokens
            taker_amount=999_000,     # 0.999 USDC
            maker_asset_id="123",
            taker_asset_id="0",
        )
        assert result is not None
        _, _, price, _, _ = result
        assert price <= 0.999

    def test_price_clamped_low(self):
        """Very low price should be clamped to 0.001."""
        result = parse_ctf_fill(
            maker_amount=1_000_000,   # 1.0 USDC
            taker_amount=2_000_000_000,  # huge tokens
            maker_asset_id="0",
            taker_asset_id="123",
        )
        assert result is not None
        _, _, price, _, _ = result
        assert price >= 0.001

    def test_fee_parsed(self):
        result = parse_ctf_fill(
            maker_amount=500_000,
            taker_amount=250_000,
            maker_asset_id="0",
            taker_asset_id="777",
            fee=5_000,
        )
        assert result is not None
        _, _, _, _, fee = result
        assert abs(fee - 0.005) < 0.0001


# ---------------------------------------------------------------------------
# Block timestamp lookup
# ---------------------------------------------------------------------------

class TestBlockTimestampLookup:
    def test_no_data_uses_fallback(self, tmp_path):
        """Empty blocks dir -> interpolate returns fallback."""
        blocks_dir = tmp_path / "blocks"
        blocks_dir.mkdir()
        lookup = BlockTimestampLookup(str(blocks_dir))
        ts = lookup.interpolate(1_000_000)
        # Fallback: 1672531200 + 1_000_000 * 2
        assert abs(ts - (1672531200 + 1_000_000 * 2)) < 1.0

    def test_nonexistent_dir_uses_fallback(self, tmp_path):
        lookup = BlockTimestampLookup(str(tmp_path / "nope"))
        assert not lookup.has_data
        ts = lookup.interpolate(100)
        assert abs(ts - (1672531200 + 100 * 2)) < 1.0

    # -- batch_interpolate --------------------------------------------------

    def test_batch_interpolate_no_data_uses_fallback(self, tmp_path):
        """batch_interpolate with no block data falls back same as scalar."""
        blocks_dir = tmp_path / "blocks"
        blocks_dir.mkdir()
        lookup = BlockTimestampLookup(str(blocks_dir))
        blocks = [100, 200, 300]
        result = lookup.batch_interpolate(blocks)
        expected = [1672531200 + b * 2 for b in blocks]
        assert result == expected

    def test_batch_interpolate_empty_input(self, tmp_path):
        """Empty input list returns empty output."""
        blocks_dir = tmp_path / "blocks"
        blocks_dir.mkdir()
        lookup = BlockTimestampLookup(str(blocks_dir))
        assert lookup.batch_interpolate([]) == []

    def test_batch_interpolate_matches_scalar(self, tmp_path):
        """batch_interpolate results must match per-element interpolate."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        blocks_dir = tmp_path / "blocks"
        blocks_dir.mkdir()

        # Write a small block parquet file with known data
        block_nums = [100, 200, 300, 400, 500]
        timestamps = [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:03:20Z",
            "2024-01-01T00:06:40Z",
            "2024-01-01T00:10:00Z",
            "2024-01-01T00:13:20Z",
        ]
        table = pa.table({
            "block_number": block_nums,
            "timestamp": timestamps,
        })
        pq.write_table(table, str(blocks_dir / "blocks.parquet"))

        lookup = BlockTimestampLookup(str(blocks_dir))

        # Test exact hits, interpolated values, and boundary clamps
        test_blocks = [50, 100, 150, 250, 400, 500, 600]
        batch_result = lookup.batch_interpolate(test_blocks)
        scalar_result = [lookup.interpolate(b) for b in test_blocks]

        assert len(batch_result) == len(test_blocks)
        for i, (batch_val, scalar_val) in enumerate(zip(batch_result, scalar_result)):
            assert abs(batch_val - scalar_val) < 0.01, (
                f"block={test_blocks[i]}: batch={batch_val} != scalar={scalar_val}"
            )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_normalized_trade_new_fields_default(self):
        """New fields should have sensible defaults."""
        t = NormalizedTrade(
            asset_id="x",
            side="BUY",
            price=0.5,
            size=10.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert t.exchange == ""
        assert t.trade_id == ""
        assert t.fee == 0.0
        assert t.condition_id == ""

    def test_market_info_new_fields_default(self):
        """New MarketInfo fields should have sensible defaults."""
        m = MarketInfo(condition_id="cid", question="Will X happen?")
        assert m.exchange == ""
        assert m.slug == ""
        assert m.liquidity == 0.0
        assert m.outcome_prices == []
        assert m.end_date is None
        assert m.created_at is None
        assert m.event_ticker == ""
        assert m.yes_bid is None
        assert m.result == ""

    def test_normalized_trade_positional_args(self):
        """Positional construction with original args still works."""
        t = NormalizedTrade("abc", "BUY", 0.65, 100.0, datetime.now(timezone.utc), "cid-1")
        assert t.asset_id == "abc"
        assert t.condition_id == "cid-1"


# ---------------------------------------------------------------------------
# Vectorized vs scalar trade parsing
# ---------------------------------------------------------------------------

def _make_ctf_table(
    n: int = 10,
    token_id: str = "12345",
    include_fee: bool = True,
    include_tx: bool = True,
):
    """Build a PyArrow table mimicking a Polymarket CTF trades parquet file.

    Alternates BUY/SELL rows so both code paths are exercised.
    """
    import pyarrow as pa

    maker_amounts = []
    taker_amounts = []
    maker_asset_ids = []
    taker_asset_ids = []
    block_numbers = []
    tx_hashes = []
    log_indices = []
    fees = []

    for i in range(n):
        block_numbers.append(1000 + i)
        if i % 2 == 0:
            # BUY: taker_asset_id == "0" (taker gives USDC)
            maker_amounts.append(500_000)   # 0.5 tokens from maker
            taker_amounts.append(300_000)   # 0.3 USDC from taker
            maker_asset_ids.append(token_id)
            taker_asset_ids.append("0")
        else:
            # SELL: maker_asset_id == "0" (maker gives USDC)
            maker_amounts.append(400_000)   # 0.4 USDC from maker
            taker_amounts.append(800_000)   # 0.8 tokens from taker
            maker_asset_ids.append("0")
            taker_asset_ids.append(token_id)
        tx_hashes.append(f"0xabc{i:04d}")
        log_indices.append(str(i))
        fees.append(5_000)

    cols = {
        "maker_amount": maker_amounts,
        "taker_amount": taker_amounts,
        "maker_asset_id": maker_asset_ids,
        "taker_asset_id": taker_asset_ids,
        "block_number": block_numbers,
    }
    if include_tx:
        cols["transaction_hash"] = tx_hashes
        cols["log_index"] = log_indices
    if include_fee:
        cols["fee"] = fees

    return pa.table(cols)


def _make_block_lookup_stub(tmp_path):
    """Create a BlockTimestampLookup backed by an empty dir (fallback mode)."""
    blocks_dir = tmp_path / "blocks"
    blocks_dir.mkdir(exist_ok=True)
    return BlockTimestampLookup(str(blocks_dir))


def _make_loader_stub(tmp_path):
    """Create a minimal PolymarketDataLoader with empty dirs."""
    data_dir = tmp_path / "data"
    (data_dir / "trades").mkdir(parents=True)
    (data_dir / "markets").mkdir(parents=True)
    return PolymarketDataLoader(str(data_dir))


class TestParseTradesVectorized:
    """Tests for _parse_trades_vectorized (numpy path)."""

    def test_basic_buy_sell(self, tmp_path):
        """Vectorized path produces correct sides, prices, and sizes."""
        table = _make_ctf_table(n=4, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        assert len(trades) == 4

        buys = [t for t in trades if t.side == "BUY"]
        sells = [t for t in trades if t.side == "SELL"]
        assert len(buys) == 2
        assert len(sells) == 2

        # BUY: price = 0.3 / 0.5 = 0.6, size = 0.5
        for t in buys:
            assert abs(t.price - 0.6) < 0.001
            assert abs(t.size - 0.5) < 0.001

        # SELL: price = 0.4 / 0.8 = 0.5, size = 0.8
        for t in sells:
            assert abs(t.price - 0.5) < 0.001
            assert abs(t.size - 0.8) < 0.001

    def test_token_filter(self, tmp_path):
        """Only trades matching target_token_ids are returned."""
        table = _make_ctf_table(n=6, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        # Filter for a token not in the data
        trades = loader._parse_trades_vectorized(table, None, {"OTHER"}, bl)
        assert trades == []

        # Filter for the correct token
        trades = loader._parse_trades_vectorized(table, None, {"T1"}, bl)
        assert len(trades) == 6

    def test_fee_and_trade_id(self, tmp_path):
        """Fee and trade_id are correctly populated."""
        table = _make_ctf_table(n=2, token_id="T1", include_fee=True, include_tx=True)
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        assert len(trades) == 2
        for t in trades:
            assert t.fee > 0
            assert t.trade_id != ""
            assert "-" in t.trade_id  # "txhash-logindex"

    def test_no_fee_column(self, tmp_path):
        """Missing fee column defaults to 0."""
        table = _make_ctf_table(n=2, include_fee=False)
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        for t in trades:
            assert t.fee == 0.0

    def test_no_tx_columns(self, tmp_path):
        """Missing transaction_hash/log_index columns -> empty trade_id."""
        table = _make_ctf_table(n=2, include_tx=False)
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        for t in trades:
            assert t.trade_id == ""

    def test_all_invalid_rows_returns_empty(self, tmp_path):
        """Table where no row has maker_asset_id or taker_asset_id == '0'."""
        import pyarrow as pa

        table = pa.table({
            "maker_amount": [100_000, 200_000],
            "taker_amount": [100_000, 200_000],
            "maker_asset_id": ["AAA", "BBB"],
            "taker_asset_id": ["CCC", "DDD"],
            "block_number": [100, 200],
        })
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        assert trades == []

    def test_zero_token_rows_filtered(self, tmp_path):
        """Rows with zero maker or taker amount are filtered out."""
        import pyarrow as pa

        table = pa.table({
            "maker_amount": [0, 500_000],
            "taker_amount": [300_000, 300_000],
            "maker_asset_id": ["T1", "T1"],
            "taker_asset_id": ["0", "0"],
            "block_number": [100, 200],
        })
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        # First row has maker_amount=0, should be invalid
        assert len(trades) == 1
        assert trades[0].side == "BUY"

    def test_exchange_field(self, tmp_path):
        """All trades should have exchange='polymarket'."""
        table = _make_ctf_table(n=2)
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_vectorized(table, None, set(), bl)
        for t in trades:
            assert t.exchange == "polymarket"


class TestParseTradesScalar:
    """Tests for _parse_trades_scalar (no-numpy fallback)."""

    def test_basic_buy_sell(self, tmp_path):
        """Scalar path produces correct sides, prices, and sizes."""
        table = _make_ctf_table(n=4, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_scalar(table, None, set(), bl)
        assert len(trades) == 4

        buys = [t for t in trades if t.side == "BUY"]
        sells = [t for t in trades if t.side == "SELL"]
        assert len(buys) == 2
        assert len(sells) == 2

    def test_token_filter(self, tmp_path):
        """Only trades matching target_token_ids are returned."""
        table = _make_ctf_table(n=6, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        trades = loader._parse_trades_scalar(table, None, {"OTHER"}, bl)
        assert trades == []

        trades = loader._parse_trades_scalar(table, None, {"T1"}, bl)
        assert len(trades) == 6


class TestVectorizedScalarParity:
    """Ensure vectorized and scalar paths produce identical results."""

    def test_outputs_match(self, tmp_path):
        """Both paths must produce the same trades for the same input."""
        table = _make_ctf_table(n=20, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        vec = loader._parse_trades_vectorized(table, None, set(), bl)
        sca = loader._parse_trades_scalar(table, None, set(), bl)

        assert len(vec) == len(sca)
        for v, s in zip(vec, sca):
            assert v.asset_id == s.asset_id
            assert v.side == s.side
            assert abs(v.price - s.price) < 1e-9
            assert abs(v.size - s.size) < 1e-9
            assert v.timestamp == s.timestamp
            assert v.exchange == s.exchange
            assert v.trade_id == s.trade_id
            assert abs(v.fee - s.fee) < 1e-9

    def test_outputs_match_with_token_filter(self, tmp_path):
        """Parity holds when filtering by token ID."""
        table = _make_ctf_table(n=10, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        vec = loader._parse_trades_vectorized(table, None, {"T1"}, bl)
        sca = loader._parse_trades_scalar(table, None, {"T1"}, bl)

        assert len(vec) == len(sca)
        for v, s in zip(vec, sca):
            assert v.side == s.side
            assert abs(v.price - s.price) < 1e-9

    def test_fallback_when_numpy_unavailable(self, tmp_path):
        """_load_trades_from_file falls back to scalar when numpy is missing."""
        import pyarrow.parquet as pq

        table = _make_ctf_table(n=4, token_id="T1")
        loader = _make_loader_stub(tmp_path)
        bl = _make_block_lookup_stub(tmp_path)

        # Write table to a parquet file
        fpath = str(tmp_path / "data" / "trades" / "test.parquet")
        pq.write_table(table, fpath)

        # Patch numpy import to fail inside _parse_trades_vectorized
        import builtins
        real_import = builtins.__import__

        def _block_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy not available")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_block_numpy):
            trades = loader._load_trades_from_file(fpath, None, set(), bl)

        assert len(trades) == 4
