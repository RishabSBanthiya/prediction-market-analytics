"""Tests for exchange adapters."""

import pytest
from omnitrade.core.enums import ExchangeId, InstrumentType, OrderStatus, Side


class TestPolymarketAdapter:
    def test_market_to_instruments(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        market = {
            "condition_id": "0xabc",
            "question": "Will it rain?",
            "tokens": [
                {"token_id": "tok-yes", "outcome": "Yes", "price": 0.65},
                {"token_id": "tok-no", "outcome": "No", "price": 0.35},
            ],
            "active": True,
            "closed": False,
        }

        instruments = PolymarketAdapter.market_to_instruments(market)
        assert len(instruments) == 2
        assert instruments[0].instrument_id == "tok-yes"
        assert instruments[0].price == 0.65
        assert instruments[0].exchange == ExchangeId.POLYMARKET
        assert instruments[0].instrument_type == InstrumentType.BINARY_OUTCOME
        assert instruments[1].outcome == "No"

    def test_market_to_instruments_closed(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        market = {
            "condition_id": "0xdef",
            "question": "Closed market?",
            "tokens": [
                {"token_id": "tok-1", "outcome": "Yes", "price": 0.99},
            ],
            "active": True,
            "closed": True,
        }
        instruments = PolymarketAdapter.market_to_instruments(market)
        assert len(instruments) == 1
        assert instruments[0].active is False
        assert instruments[0].closed is True

    def test_market_to_instruments_empty_tokens(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        market = {
            "condition_id": "0xabc",
            "question": "No tokens",
            "tokens": [],
        }
        instruments = PolymarketAdapter.market_to_instruments(market)
        assert len(instruments) == 0

    def test_market_to_instruments_with_expiry(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        market = {
            "condition_id": "0xabc",
            "question": "With expiry?",
            "tokens": [
                {"token_id": "tok-1", "outcome": "Yes", "price": 0.50},
            ],
            "end_date_iso": "2026-06-01T00:00:00Z",
            "active": True,
            "closed": False,
        }
        instruments = PolymarketAdapter.market_to_instruments(market)
        assert instruments[0].expiry is not None

    def test_market_id_set_to_condition_id(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        market = {
            "condition_id": "0xcondition123",
            "question": "Test",
            "tokens": [{"token_id": "tok-1", "outcome": "Yes", "price": 0.50}],
            "active": True,
            "closed": False,
        }
        instruments = PolymarketAdapter.market_to_instruments(market)
        assert instruments[0].market_id == "0xcondition123"

    def test_order_response_success(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        response = {
            "success": True,
            "orderID": "order-123",
            "status": "matched",
            "takingAmount": 50.0,
        }
        result = PolymarketAdapter.order_response_to_result(response, 50.0, 0.50)
        assert result.success
        assert result.order_id == "order-123"
        assert result.status == OrderStatus.FILLED

    def test_order_response_failure(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        response = {
            "success": False,
            "errorMsg": "Insufficient funds",
        }
        result = PolymarketAdapter.order_response_to_result(response, 50.0, 0.50)
        assert not result.success
        assert "Insufficient" in result.error_message

    def test_position_to_exchange_position(self):
        from omnitrade.exchanges.polymarket.adapter import PolymarketAdapter

        pos_data = {
            "asset": "tok-yes",
            "size": 100.0,
            "avgPrice": 0.50,
            "curPrice": 0.60,
        }
        pos = PolymarketAdapter.position_to_exchange_position(pos_data)
        assert pos.instrument_id == "tok-yes"
        assert pos.side == Side.BUY
        assert pos.size == 100.0
        assert pos.entry_price == 0.50
        assert pos.current_price == 0.60
        assert pos.unrealized_pnl == pytest.approx(10.0)


class TestKalshiAdapter:
    def test_cents_conversion(self):
        from omnitrade.exchanges.kalshi.adapter import cents_to_normalized, normalized_to_cents

        assert cents_to_normalized(65) == 0.65
        assert cents_to_normalized(0) == 0.0
        assert cents_to_normalized(99) == 0.99
        assert normalized_to_cents(0.65) == 65
        assert normalized_to_cents(0.0) == 1  # Clamped to min 1
        assert normalized_to_cents(1.0) == 99  # Clamped to max 99

    def test_cents_boundary_values(self):
        from omnitrade.exchanges.kalshi.adapter import cents_to_normalized, normalized_to_cents

        assert cents_to_normalized(1) == 0.01
        assert cents_to_normalized(50) == 0.50
        assert normalized_to_cents(0.01) == 1
        assert normalized_to_cents(0.50) == 50
        assert normalized_to_cents(0.99) == 99

    def test_event_to_instruments(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        market = {
            "ticker": "RAIN-23",
            "title": "Will it rain tomorrow?",
            "event_ticker": "RAIN",
            "yes_ask": 65,
            "no_ask": 35,
            "yes_bid": 63,
            "no_bid": 33,
            "status": "open",
        }

        instruments = KalshiAdapter.event_to_instruments(market)
        assert len(instruments) == 2

        yes_inst = [i for i in instruments if i.outcome == "YES"][0]
        assert yes_inst.instrument_id == "RAIN-23-YES"
        assert abs(yes_inst.price - 0.65) < 0.01
        assert yes_inst.exchange == ExchangeId.KALSHI

    def test_event_to_instruments_inactive(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        market = {
            "ticker": "SETTLED-1",
            "title": "Settled market",
            "event_ticker": "SETTLED",
            "yes_ask": 99,
            "no_ask": 1,
            "yes_bid": 99,
            "no_bid": 1,
            "status": "settled",
        }
        instruments = KalshiAdapter.event_to_instruments(market)
        assert all(not i.active for i in instruments)
        assert all(i.closed for i in instruments)

    def test_event_to_instruments_market_id(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        market = {
            "ticker": "RAIN-23",
            "title": "Test",
            "event_ticker": "RAIN",
            "yes_ask": 50,
            "no_ask": 50,
            "yes_bid": 48,
            "no_bid": 48,
            "status": "open",
        }
        instruments = KalshiAdapter.event_to_instruments(market)
        assert instruments[0].market_id == "RAIN"

    def test_event_to_instruments_type(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        market = {
            "ticker": "TEST-1",
            "title": "Test",
            "yes_ask": 50,
            "no_ask": 50,
            "yes_bid": 48,
            "no_bid": 48,
            "status": "open",
        }
        instruments = KalshiAdapter.event_to_instruments(market)
        assert all(i.instrument_type == InstrumentType.EVENT_CONTRACT for i in instruments)

    def test_orderbook_to_snapshot(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        book_data = {
            "yes": {
                "bids": [[45, 10], [44, 20]],
                "asks": [[55, 15], [56, 25]],
            }
        }
        snapshot = KalshiAdapter.orderbook_to_snapshot("TEST-YES", book_data)
        assert snapshot.instrument_id == "TEST-YES"
        assert snapshot.best_bid == pytest.approx(0.45)
        assert snapshot.best_ask == pytest.approx(0.55)

    def test_order_response_filled(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        response = {
            "order": {
                "order_id": "ord-123",
                "status": "executed",
                "count": 10,
                "remaining_count": 0,
                "avg_fill_price": 65,
            }
        }
        result = KalshiAdapter.order_response_to_result(response, 10.0, 0.65)
        assert result.success
        assert result.status == OrderStatus.FILLED
        assert result.filled_size == 10.0

    def test_order_response_resting(self):
        from omnitrade.exchanges.kalshi.adapter import KalshiAdapter

        response = {
            "order": {
                "order_id": "ord-456",
                "status": "resting",
                "count": 10,
                "remaining_count": 10,
            }
        }
        result = KalshiAdapter.order_response_to_result(response, 10.0, 0.50)
        assert result.success
        assert result.status == OrderStatus.OPEN


class TestHyperliquidAdapter:
    def test_meta_to_instruments(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        meta = {
            "universe": [
                {"name": "BTC", "maxLeverage": 50, "szDecimals": 3},
                {"name": "ETH", "maxLeverage": 25, "szDecimals": 2},
            ]
        }
        ctxs = [
            {"markPx": "65000.0", "funding": "0.0001"},
            {"markPx": "3500.0", "funding": "-0.0002"},
        ]

        instruments = HyperliquidAdapter.meta_to_instruments(meta, ctxs)
        assert len(instruments) == 2

        btc = instruments[0]
        assert btc.instrument_id == "BTC"
        assert btc.instrument_type == InstrumentType.PERPETUAL
        assert btc.max_leverage == 50.0
        assert btc.price == 65000.0
        assert btc.funding_rate == 0.0001

    def test_meta_to_instruments_names(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        meta = {"universe": [{"name": "SOL", "maxLeverage": 20, "szDecimals": 1}]}
        ctxs = [{"markPx": "150.0", "funding": "0.0"}]
        instruments = HyperliquidAdapter.meta_to_instruments(meta, ctxs)
        assert instruments[0].name == "SOL-PERP"
        assert instruments[0].exchange == ExchangeId.HYPERLIQUID

    def test_meta_to_instruments_empty(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        meta = {"universe": []}
        instruments = HyperliquidAdapter.meta_to_instruments(meta, [])
        assert len(instruments) == 0

    def test_l2_to_snapshot(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        l2_data = {
            "levels": [
                [{"px": "65000.0", "sz": "1.5"}, {"px": "64999.0", "sz": "2.0"}],
                [{"px": "65001.0", "sz": "1.0"}, {"px": "65002.0", "sz": "3.0"}],
            ]
        }
        snapshot = HyperliquidAdapter.l2_to_snapshot("BTC", l2_data)
        assert snapshot.instrument_id == "BTC"
        assert snapshot.best_bid == 65000.0
        assert snapshot.best_ask == 65001.0
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2

    def test_order_response_filled(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        response = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"oid": 12345, "totalSz": "1.5", "avgPx": "65000.0"}}
                    ]
                }
            },
        }
        result = HyperliquidAdapter.order_response_to_result(response, 1.5, 65000.0)
        assert result.success
        assert result.status == OrderStatus.FILLED
        assert result.filled_size == 1.5
        assert result.filled_price == 65000.0

    def test_order_response_resting(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        response = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"resting": {"oid": 12346}}
                    ]
                }
            },
        }
        result = HyperliquidAdapter.order_response_to_result(response, 1.0, 3500.0)
        assert result.success
        assert result.status == OrderStatus.OPEN

    def test_order_response_error(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        response = {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"error": "Insufficient margin"}
                    ]
                }
            },
        }
        result = HyperliquidAdapter.order_response_to_result(response, 1.0, 65000.0)
        assert not result.success
        assert "Insufficient" in result.error_message

    def test_order_response_bad_status(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        response = {"status": "error", "response": "Something went wrong"}
        result = HyperliquidAdapter.order_response_to_result(response, 1.0, 100.0)
        assert not result.success

    def test_user_state_to_positions(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        state = {
            "assetPositions": [
                {
                    "position": {
                        "coin": "BTC",
                        "szi": "0.5",
                        "entryPx": "65000.0",
                        "unrealizedPnl": "500.0",
                        "liquidationPx": "60000.0",
                        "leverage": {"value": 10},
                    }
                },
                {
                    "position": {
                        "coin": "ETH",
                        "szi": "-2.0",
                        "entryPx": "3500.0",
                        "unrealizedPnl": "-100.0",
                        "liquidationPx": "4000.0",
                        "leverage": {"value": 5},
                    }
                },
            ]
        }
        positions = HyperliquidAdapter.user_state_to_positions(state)
        assert len(positions) == 2

        btc = [p for p in positions if p.instrument_id == "BTC"][0]
        assert btc.side == Side.BUY
        assert btc.size == 0.5
        assert btc.leverage == 10.0

        eth = [p for p in positions if p.instrument_id == "ETH"][0]
        assert eth.side == Side.SELL
        assert eth.size == 2.0

    def test_user_state_to_positions_filters_zero(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        state = {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0", "entryPx": "0"}}
            ]
        }
        positions = HyperliquidAdapter.user_state_to_positions(state)
        assert len(positions) == 0

    def test_user_state_to_balance(self):
        from omnitrade.exchanges.hyperliquid.adapter import HyperliquidAdapter

        state = {
            "marginSummary": {
                "accountValue": "10000.0",
                "totalRawUsd": "5000.0",
                "totalNtlPos": "3000.0",
            }
        }
        balance = HyperliquidAdapter.user_state_to_balance(state)
        assert balance.exchange == ExchangeId.HYPERLIQUID
        assert balance.total_equity == 10000.0
        assert balance.available_balance == 5000.0
