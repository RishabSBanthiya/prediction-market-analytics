"""Tests for Polymarket client balance tracking.

Verifies that get_balance() reflects trade activity instead of returning
a hardcoded value. Covers paper mode (tracked balance) and live mode
(on-chain query with fallback).
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from omnitrade.core.enums import ExchangeId, Side, OrderStatus, OrderType
from omnitrade.core.config import ExchangeConfig
from omnitrade.core.models import (
    OrderRequest, OrderResult, AccountBalance,
)
from omnitrade.exchanges.polymarket.client import PolymarketClient


@pytest.fixture
def poly_config() -> ExchangeConfig:
    """Polymarket config with no real credentials (paper mode)."""
    return ExchangeConfig(
        exchange=ExchangeId.POLYMARKET,
        rate_limit_per_window=9000,
        rate_limit_window_seconds=10,
    )


@pytest.fixture
def poly_client(poly_config) -> PolymarketClient:
    """PolymarketClient with mocked auth (unauthenticated)."""
    client = PolymarketClient(poly_config)
    # Auth returns False for is_authenticated since no keys configured
    client._auth = MagicMock()
    client._auth.is_authenticated.return_value = False
    return client


class TestPaperBalance:
    """Balance tracking in paper mode (no API credentials)."""

    @pytest.mark.asyncio
    async def test_initial_balance(self, poly_client: PolymarketClient):
        """Starting balance should be 10_000 USDC."""
        balance = await poly_client.get_balance()
        assert balance.total_equity == 10_000.0
        assert balance.available_balance == 10_000.0
        assert balance.currency == "USDC"
        assert balance.exchange == ExchangeId.POLYMARKET

    @pytest.mark.asyncio
    async def test_buy_reduces_balance(self, poly_client: PolymarketClient):
        """A successful buy should deduct cost from balance."""
        result = OrderResult(
            success=True,
            order_id="ORDER-1",
            status=OrderStatus.FILLED,
            filled_size=100.0,
            filled_price=0.65,
            requested_size=100.0,
            requested_price=0.65,
        )
        poly_client._apply_balance_delta(result, Side.BUY)

        balance = await poly_client.get_balance()
        expected = 10_000.0 - (100.0 * 0.65)
        assert balance.total_equity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_sell_increases_balance(self, poly_client: PolymarketClient):
        """A successful sell should add proceeds to balance."""
        result = OrderResult(
            success=True,
            order_id="ORDER-2",
            status=OrderStatus.FILLED,
            filled_size=50.0,
            filled_price=0.80,
            requested_size=50.0,
            requested_price=0.80,
        )
        poly_client._apply_balance_delta(result, Side.SELL)

        balance = await poly_client.get_balance()
        expected = 10_000.0 + (50.0 * 0.80)
        assert balance.total_equity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_multiple_trades_accumulate(self, poly_client: PolymarketClient):
        """Balance should reflect cumulative trade activity."""
        # Buy 100 shares at 0.50
        buy_result = OrderResult(
            success=True, order_id="O1", status=OrderStatus.FILLED,
            filled_size=100.0, filled_price=0.50,
        )
        poly_client._apply_balance_delta(buy_result, Side.BUY)

        # Sell 100 shares at 0.70
        sell_result = OrderResult(
            success=True, order_id="O2", status=OrderStatus.FILLED,
            filled_size=100.0, filled_price=0.70,
        )
        poly_client._apply_balance_delta(sell_result, Side.SELL)

        balance = await poly_client.get_balance()
        # Net: -50 + 70 = +20 profit
        expected = 10_000.0 - 50.0 + 70.0
        assert balance.total_equity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_fees_deducted(self, poly_client: PolymarketClient):
        """Fees should be subtracted from balance."""
        result = OrderResult(
            success=True, order_id="O3", status=OrderStatus.FILLED,
            filled_size=100.0, filled_price=0.60, fees=1.50,
        )
        poly_client._apply_balance_delta(result, Side.BUY)

        balance = await poly_client.get_balance()
        expected = 10_000.0 - (100.0 * 0.60) - 1.50
        assert balance.total_equity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_failed_order_no_change(self, poly_client: PolymarketClient):
        """Failed orders should not affect balance."""
        result = OrderResult(
            success=False, error_message="Insufficient liquidity",
        )
        poly_client._apply_balance_delta(result, Side.BUY)

        balance = await poly_client.get_balance()
        assert balance.total_equity == 10_000.0

    @pytest.mark.asyncio
    async def test_zero_fill_no_change(self, poly_client: PolymarketClient):
        """Orders with zero fill size should not affect balance."""
        result = OrderResult(
            success=True, order_id="O4", status=OrderStatus.OPEN,
            filled_size=0.0, filled_price=0.50,
        )
        poly_client._apply_balance_delta(result, Side.BUY)

        balance = await poly_client.get_balance()
        assert balance.total_equity == 10_000.0

    @pytest.mark.asyncio
    async def test_zero_price_no_change(self, poly_client: PolymarketClient):
        """Orders with zero price should not affect balance."""
        result = OrderResult(
            success=True, order_id="O5", status=OrderStatus.FILLED,
            filled_size=100.0, filled_price=0.0,
        )
        poly_client._apply_balance_delta(result, Side.BUY)

        balance = await poly_client.get_balance()
        assert balance.total_equity == 10_000.0


class TestLiveBalance:
    """Balance fetching in live mode (authenticated with CLOB client)."""

    @pytest.fixture
    def live_client(self, poly_config) -> PolymarketClient:
        """PolymarketClient that appears authenticated."""
        client = PolymarketClient(poly_config)
        client._auth = MagicMock()
        client._auth.is_authenticated.return_value = True

        mock_clob = MagicMock()
        # USDC uses 6 decimals: 5000 USDC = 5_000_000_000
        mock_clob.get_bal_allowance.return_value = {
            "balance": "5000000000",
        }
        client._auth.client = mock_clob
        return client

    @pytest.mark.asyncio
    async def test_live_fetches_on_chain_balance(self, live_client: PolymarketClient):
        """Authenticated client should query CLOB for on-chain USDC balance."""
        balance = await live_client.get_balance()
        assert balance.total_equity == pytest.approx(5000.0)
        assert balance.currency == "USDC"

    @pytest.mark.asyncio
    async def test_live_falls_back_to_tracked_on_error(self, live_client: PolymarketClient):
        """If CLOB balance call fails, fall back to tracked balance."""
        live_client._auth.client.get_bal_allowance.side_effect = Exception("RPC error")

        balance = await live_client.get_balance()
        # Should fall back to initial paper balance
        assert balance.total_equity == 10_000.0

    @pytest.mark.asyncio
    async def test_live_fallback_reflects_trades(self, live_client: PolymarketClient):
        """Fallback tracked balance should still reflect trade activity."""
        live_client._auth.client.get_bal_allowance.side_effect = Exception("RPC error")

        result = OrderResult(
            success=True, order_id="O1", status=OrderStatus.FILLED,
            filled_size=200.0, filled_price=0.40,
        )
        live_client._apply_balance_delta(result, Side.BUY)

        balance = await live_client.get_balance()
        expected = 10_000.0 - (200.0 * 0.40)
        assert balance.total_equity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_live_non_dict_response_falls_back(self, live_client: PolymarketClient):
        """If CLOB returns unexpected type, fall back to tracked balance."""
        live_client._auth.client.get_bal_allowance.return_value = "unexpected"

        balance = await live_client.get_balance()
        assert balance.total_equity == 10_000.0
