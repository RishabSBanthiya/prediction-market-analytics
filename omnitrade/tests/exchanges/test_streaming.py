"""Tests for WebSocket streaming interface and Hyperliquid WebSocket implementation."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from omnitrade.core.config import ExchangeConfig
from omnitrade.core.enums import ExchangeId
from omnitrade.core.models import OrderbookLevel, OrderbookSnapshot
from omnitrade.exchanges.base import ExchangeClient, MarketDataUpdate


class TestMarketDataUpdate:
    """Tests for the MarketDataUpdate wrapper."""

    def test_create_with_defaults(self):
        snapshot = OrderbookSnapshot(
            instrument_id="BTC",
            bids=[OrderbookLevel(price=100.0, size=1.0)],
            asks=[OrderbookLevel(price=101.0, size=1.0)],
        )
        update = MarketDataUpdate(snapshot)
        assert update.snapshot is snapshot
        assert update.source == "ws"

    def test_create_with_rest_source(self):
        snapshot = OrderbookSnapshot(instrument_id="ETH")
        update = MarketDataUpdate(snapshot, source="rest")
        assert update.source == "rest"

    def test_repr(self):
        snapshot = OrderbookSnapshot(
            instrument_id="BTC",
            bids=[OrderbookLevel(price=100.0, size=1.0)],
            asks=[OrderbookLevel(price=101.0, size=1.0)],
        )
        update = MarketDataUpdate(snapshot)
        r = repr(update)
        assert "BTC" in r
        assert "ws" in r


class TestExchangeClientStreamingDefaults:
    """Tests for the default REST-polling fallback on ExchangeClient."""

    @pytest.fixture
    def client(self, mock_client):
        return mock_client

    def test_supports_streaming_default_false(self, client):
        assert client.supports_streaming is False

    @pytest.mark.asyncio
    async def test_subscribe_starts_rest_poll(self, client):
        """subscribe_orderbook should start a background REST polling task."""
        updates = []
        await client.subscribe_orderbook("test-token", updates.append)

        # Give the poll loop one iteration
        await asyncio.sleep(0.05)

        assert len(updates) >= 1
        assert isinstance(updates[0], MarketDataUpdate)
        assert updates[0].source == "rest"
        assert updates[0].snapshot.instrument_id == "test-token"

        # Clean up
        await client.unsubscribe_orderbook("test-token")

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_polling(self, client):
        updates = []
        await client.subscribe_orderbook("test-token", updates.append)
        await asyncio.sleep(0.05)

        await client.unsubscribe_orderbook("test-token")
        count_after_unsub = len(updates)

        # Wait to confirm no more updates
        await asyncio.sleep(0.1)
        assert len(updates) == count_after_unsub

    @pytest.mark.asyncio
    async def test_active_subscriptions(self, client):
        updates = []
        assert client.active_subscriptions == set()

        await client.subscribe_orderbook("BTC", updates.append)
        assert "BTC" in client.active_subscriptions

        await client.unsubscribe_orderbook("BTC")
        assert "BTC" not in client.active_subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, client):
        updates = []
        await client.subscribe_orderbook("BTC", updates.append)
        await client.subscribe_orderbook("ETH", updates.append)
        assert len(client.active_subscriptions) == 2

        await client.unsubscribe_all()
        assert client.active_subscriptions == set()

    @pytest.mark.asyncio
    async def test_duplicate_subscribe_ignored(self, client):
        updates = []
        await client.subscribe_orderbook("BTC", updates.append)
        # Should not create a second poll task
        await client.subscribe_orderbook("BTC", updates.append)
        assert len(client.active_subscriptions) == 1
        await client.unsubscribe_all()

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_is_noop(self, client):
        """Unsubscribing from a non-existent subscription should not raise."""
        await client.unsubscribe_orderbook("DOESNOTEXIST")


class TestPaperClientStreamingDelegation:
    """Tests that PaperClient delegates streaming to the wrapped client."""

    @pytest.mark.asyncio
    async def test_paper_client_delegates_subscribe(self, mock_client):
        from omnitrade.exchanges.base import PaperClient

        paper = PaperClient(mock_client)

        updates = []
        await paper.subscribe_orderbook("test-token", updates.append)
        await asyncio.sleep(0.05)

        assert len(updates) >= 1
        assert updates[0].source == "rest"

        # Subscription is tracked on the inner client
        assert "test-token" in paper.active_subscriptions

        await paper.unsubscribe_all()

    def test_paper_client_supports_streaming(self, mock_client):
        from omnitrade.exchanges.base import PaperClient

        paper = PaperClient(mock_client)
        assert paper.supports_streaming == mock_client.supports_streaming


class TestHyperliquidWebSocket:
    """Tests for HyperliquidWebSocket message handling."""

    def test_handle_l2_message(self):
        from omnitrade.exchanges.hyperliquid.websocket import HyperliquidWebSocket

        ws = HyperliquidWebSocket()
        received = []

        ws._subscriptions["BTC"] = received.append

        msg = {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "50000", "sz": "1.5"}, {"px": "49999", "sz": "2.0"}],
                    [{"px": "50001", "sz": "1.0"}, {"px": "50002", "sz": "0.5"}],
                ],
            },
        }

        ws._handle_message(msg)

        assert len(received) == 1
        update = received[0]
        assert isinstance(update, MarketDataUpdate)
        assert update.source == "ws"
        assert update.snapshot.instrument_id == "BTC"
        assert update.snapshot.best_bid == 50000.0
        assert update.snapshot.best_ask == 50001.0

    def test_handle_irrelevant_channel_ignored(self):
        from omnitrade.exchanges.hyperliquid.websocket import HyperliquidWebSocket

        ws = HyperliquidWebSocket()
        received = []
        ws._subscriptions["BTC"] = received.append

        ws._handle_message({"channel": "trades", "data": {}})
        assert len(received) == 0

    def test_handle_unsubscribed_coin_ignored(self):
        from omnitrade.exchanges.hyperliquid.websocket import HyperliquidWebSocket

        ws = HyperliquidWebSocket()
        received = []
        ws._subscriptions["ETH"] = received.append

        msg = {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [[], []],
            },
        }

        ws._handle_message(msg)
        assert len(received) == 0

    def test_subscribed_instruments_property(self):
        from omnitrade.exchanges.hyperliquid.websocket import HyperliquidWebSocket

        ws = HyperliquidWebSocket()
        assert ws.subscribed_instruments == set()

        ws._subscriptions["BTC"] = lambda x: None
        ws._subscriptions["ETH"] = lambda x: None
        assert ws.subscribed_instruments == {"BTC", "ETH"}


class TestHyperliquidClientStreaming:
    """Tests that HyperliquidClient correctly reports streaming support."""

    def test_supports_streaming_true(self):
        from omnitrade.exchanges.hyperliquid.client import HyperliquidClient

        config = ExchangeConfig(exchange=ExchangeId.HYPERLIQUID)
        client = HyperliquidClient(config)
        assert client.supports_streaming is True

    def test_active_subscriptions_empty_initially(self):
        from omnitrade.exchanges.hyperliquid.client import HyperliquidClient

        config = ExchangeConfig(exchange=ExchangeId.HYPERLIQUID)
        client = HyperliquidClient(config)
        assert client.active_subscriptions == set()
