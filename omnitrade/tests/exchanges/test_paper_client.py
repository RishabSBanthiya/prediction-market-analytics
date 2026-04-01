"""Tests for PaperClient order ID generation."""

import asyncio

import pytest

from omnitrade.core.enums import ExchangeId, OrderType, Side
from omnitrade.core.models import OrderRequest
from omnitrade.exchanges.base import PaperClient


@pytest.mark.asyncio
async def test_paper_order_ids_are_unique(mock_client):
    """Concurrent place_order calls must never produce duplicate order IDs."""
    paper = PaperClient(mock_client)

    request = OrderRequest(
        instrument_id="token-yes",
        side=Side.BUY,
        size=10.0,
        price=0.65,
        order_type=OrderType.LIMIT,
    )

    results = await asyncio.gather(*(paper.place_order(request) for _ in range(50)))
    order_ids = [r.order_id for r in results]

    assert len(order_ids) == len(set(order_ids)), (
        f"Duplicate order IDs detected: {[oid for oid in order_ids if order_ids.count(oid) > 1]}"
    )


@pytest.mark.asyncio
async def test_paper_order_ids_sequential(mock_client):
    """Order IDs should be sequentially numbered starting from 1."""
    paper = PaperClient(mock_client)

    request = OrderRequest(
        instrument_id="token-yes",
        side=Side.BUY,
        size=10.0,
        price=0.65,
        order_type=OrderType.LIMIT,
    )

    r1 = await paper.place_order(request)
    r2 = await paper.place_order(request)

    assert r1.order_id == "PAPER-000001"
    assert r2.order_id == "PAPER-000002"
