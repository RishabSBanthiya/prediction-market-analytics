"""Tests for CancelResult model and cancel_orders() return type."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from omnitrade.core.models import CancelResult, CancelDetail


class TestCancelResultModel:
    """Tests for the CancelResult and CancelDetail dataclasses."""

    def test_empty_cancel_result(self):
        result = CancelResult()
        assert result.cancelled == 0
        assert result.failed == 0
        assert result.already_filled == 0
        assert result.details == []
        assert result.total == 0
        assert result.failed_order_ids == []

    def test_cancel_result_total(self):
        result = CancelResult(cancelled=3, failed=1, already_filled=2)
        assert result.total == 6

    def test_failed_order_ids_excludes_not_found(self):
        result = CancelResult(
            failed=2,
            already_filled=1,
            details=[
                CancelDetail(order_id="o1", success=False, error_code="rate_limit"),
                CancelDetail(order_id="o2", success=False, error_code="not_found"),
                CancelDetail(order_id="o3", success=False, error_code="unknown"),
                CancelDetail(order_id="o4", success=True),
            ],
        )
        failed = result.failed_order_ids
        assert "o1" in failed
        assert "o3" in failed
        assert "o2" not in failed  # not_found = already filled, not retryable
        assert "o4" not in failed  # success

    def test_cancel_detail_defaults(self):
        detail = CancelDetail(order_id="abc", success=True)
        assert detail.error_code == ""
        assert detail.error_message == ""


class TestBaseCancelOrders:
    """Tests for the ExchangeClient base class cancel_orders() default impl."""

    @pytest.mark.asyncio
    async def test_base_cancel_orders_all_succeed(self):
        from omnitrade.exchanges.base import ExchangeClient

        client = AsyncMock(spec=ExchangeClient)
        client.cancel_order = AsyncMock(return_value=True)

        # Call the real base implementation
        result = await ExchangeClient.cancel_orders(client, ["o1", "o2", "o3"])

        assert result.cancelled == 3
        assert result.failed == 0
        assert len(result.details) == 3
        assert all(d.success for d in result.details)

    @pytest.mark.asyncio
    async def test_base_cancel_orders_partial_failure(self):
        from omnitrade.exchanges.base import ExchangeClient

        client = AsyncMock(spec=ExchangeClient)
        client.cancel_order = AsyncMock(side_effect=[True, False, True])

        result = await ExchangeClient.cancel_orders(client, ["o1", "o2", "o3"])

        assert result.cancelled == 2
        assert result.failed == 1
        assert result.details[1].order_id == "o2"
        assert not result.details[1].success

    @pytest.mark.asyncio
    async def test_base_cancel_orders_empty(self):
        from omnitrade.exchanges.base import ExchangeClient

        client = AsyncMock(spec=ExchangeClient)
        result = await ExchangeClient.cancel_orders(client, [])
        assert result.cancelled == 0
        assert result.total == 0


class TestKalshiCancelOrders:
    """Tests for KalshiClient.cancel_orders() structured return."""

    @pytest.fixture
    def kalshi_client(self):
        from omnitrade.exchanges.kalshi.client import KalshiClient
        from omnitrade.core.config import ExchangeConfig

        config = ExchangeConfig(
            exchange="kalshi",
            api_key="test",
            api_secret="test",
            api_base="https://test.kalshi.com",
        )
        client = KalshiClient(config)
        return client

    @pytest.mark.asyncio
    async def test_cancel_orders_empty(self, kalshi_client):
        result = await kalshi_client.cancel_orders([])
        assert isinstance(result, CancelResult)
        assert result.cancelled == 0
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_cancel_orders_all_success(self, kalshi_client):
        api_response = {
            "orders": [
                {"order_id": "o1"},
                {"order_id": "o2"},
            ]
        }
        kalshi_client._request = AsyncMock(return_value=api_response)

        result = await kalshi_client.cancel_orders(["o1", "o2"])

        assert result.cancelled == 2
        assert result.failed == 0
        assert result.already_filled == 0
        assert len(result.details) == 2
        assert all(d.success for d in result.details)

    @pytest.mark.asyncio
    async def test_cancel_orders_with_not_found(self, kalshi_client):
        api_response = {
            "orders": [
                {"order_id": "o1"},
                {"order_id": "o2", "error": {"code": "not_found", "message": "Order already filled"}},
            ]
        }
        kalshi_client._request = AsyncMock(return_value=api_response)

        result = await kalshi_client.cancel_orders(["o1", "o2"])

        assert result.cancelled == 1
        assert result.already_filled == 1
        assert result.failed == 0
        # not_found should not appear in failed_order_ids (not retryable)
        assert result.failed_order_ids == []
        # But should appear in details
        detail_o2 = [d for d in result.details if d.order_id == "o2"][0]
        assert detail_o2.error_code == "not_found"

    @pytest.mark.asyncio
    async def test_cancel_orders_with_real_failure(self, kalshi_client):
        api_response = {
            "orders": [
                {"order_id": "o1"},
                {"order_id": "o2", "error": {"code": "rate_limited", "message": "Too many requests"}},
            ]
        }
        kalshi_client._request = AsyncMock(return_value=api_response)

        result = await kalshi_client.cancel_orders(["o1", "o2"])

        assert result.cancelled == 1
        assert result.failed == 1
        assert result.failed_order_ids == ["o2"]
        detail_o2 = [d for d in result.details if d.order_id == "o2"][0]
        assert detail_o2.error_code == "rate_limited"
        assert "Too many requests" in detail_o2.error_message

    @pytest.mark.asyncio
    async def test_cancel_orders_batch_fallback(self, kalshi_client):
        """When batch API fails, individual cancels are attempted."""
        kalshi_client._request = AsyncMock(side_effect=Exception("API down"))
        kalshi_client.cancel_order = AsyncMock(side_effect=[True, False])

        result = await kalshi_client.cancel_orders(["o1", "o2"])

        assert result.cancelled == 1
        assert result.failed == 1
        failed_detail = [d for d in result.details if d.order_id == "o2"][0]
        assert failed_detail.error_code == "individual_fallback"

    @pytest.mark.asyncio
    async def test_cancel_orders_batching(self, kalshi_client):
        """Orders are batched in groups of 20."""
        order_ids = [f"o{i}" for i in range(25)]

        call_count = 0

        async def mock_request(method, path, json=None):
            nonlocal call_count
            call_count += 1
            return {"orders": [{"order_id": oid} for oid in json["ids"]]}

        kalshi_client._request = mock_request

        result = await kalshi_client.cancel_orders(order_ids)

        assert call_count == 2  # 20 + 5
        assert result.cancelled == 25
        assert len(result.details) == 25
