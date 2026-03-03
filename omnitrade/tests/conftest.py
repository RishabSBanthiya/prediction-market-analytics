"""Shared test fixtures for omnitrade."""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from omnitrade.core.enums import ExchangeId, Side, SignalDirection, InstrumentType, OrderStatus, OrderType
from omnitrade.core.models import (
    Instrument, OrderbookSnapshot, OrderbookLevel,
    OrderRequest, OrderResult, Signal, AccountBalance,
    ExchangePosition, OpenOrder,
)
from omnitrade.core.config import Config, RiskConfig, ExchangeConfig
from omnitrade.exchanges.base import ExchangeClient
from omnitrade.storage.sqlite import SQLiteStorage
from omnitrade.risk.coordinator import RiskCoordinator


class MockExchangeClient(ExchangeClient):
    """Mock exchange client for testing."""

    def __init__(self, exchange_id: ExchangeId = ExchangeId.POLYMARKET):
        config = ExchangeConfig(exchange=exchange_id)
        super().__init__(config)
        self._exchange_id = exchange_id
        self._instruments: list[Instrument] = []
        self._balance = AccountBalance(
            exchange=exchange_id,
            total_equity=1000.0,
            available_balance=800.0,
        )
        self._positions: list[ExchangePosition] = []
        self._orderbook = OrderbookSnapshot(
            instrument_id="test-token",
            bids=[OrderbookLevel(price=0.50, size=100)],
            asks=[OrderbookLevel(price=0.52, size=100)],
        )
        self._order_counter = 0

    @property
    def exchange_id(self) -> ExchangeId:
        return self._exchange_id

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def get_instruments(self, active_only=True, **filters) -> list[Instrument]:
        return self._instruments

    async def get_instrument(self, instrument_id: str):
        for i in self._instruments:
            if i.instrument_id == instrument_id:
                return i
        return None

    async def get_orderbook(self, instrument_id: str, depth=10) -> OrderbookSnapshot:
        return self._orderbook

    async def get_midpoint(self, instrument_id: str):
        return self._orderbook.midpoint

    async def place_order(self, request: OrderRequest) -> OrderResult:
        self._order_counter += 1
        return OrderResult(
            success=True,
            order_id=f"MOCK-{self._order_counter}",
            status=OrderStatus.FILLED,
            filled_size=request.size,
            filled_price=request.price,
            requested_size=request.size,
            requested_price=request.price,
        )

    async def cancel_order(self, order_id: str, instrument_id: str = "") -> bool:
        return True

    async def cancel_all_orders(self, instrument_id=None) -> int:
        return 0

    async def get_open_orders(self, instrument_id=None) -> list[OpenOrder]:
        return []

    async def get_balance(self) -> AccountBalance:
        return self._balance

    async def get_positions(self) -> list[ExchangePosition]:
        return self._positions


@pytest.fixture
def mock_client():
    """Create a mock exchange client."""
    client = MockExchangeClient()
    client._instruments = [
        Instrument(
            instrument_id="token-yes",
            exchange=ExchangeId.POLYMARKET,
            instrument_type=InstrumentType.BINARY_OUTCOME,
            name="Test Market - Yes",
            price=0.65,
            bid=0.64,
            ask=0.66,
            market_id="test-market",
            outcome="YES",
            active=True,
        ),
        Instrument(
            instrument_id="token-no",
            exchange=ExchangeId.POLYMARKET,
            instrument_type=InstrumentType.BINARY_OUTCOME,
            name="Test Market - No",
            price=0.35,
            bid=0.34,
            ask=0.36,
            market_id="test-market",
            outcome="NO",
            active=True,
        ),
    ]
    return client


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    storage = SQLiteStorage(db_path)
    storage.initialize()
    yield storage
    storage.close()
    os.unlink(db_path)


@pytest.fixture
def risk_config():
    """Create a test risk config."""
    return RiskConfig(
        max_wallet_exposure_pct=0.60,
        max_per_agent_exposure_pct=0.30,
        max_per_market_exposure_pct=0.10,
        min_trade_value_usd=5.0,
        max_trade_value_usd=500.0,
    )


@pytest.fixture
def risk_coordinator(tmp_db, risk_config):
    """Create a risk coordinator with temp storage."""
    coord = RiskCoordinator(tmp_db, risk_config)
    coord.register_account(ExchangeId.POLYMARKET, "test-wallet")
    tmp_db.update_balance("polymarket", "test-wallet", 1000.0)
    return coord


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return Signal(
        instrument_id="token-yes",
        direction=SignalDirection.LONG,
        score=75.0,
        source="test",
        price=0.65,
        market_id="test-market",
        exchange=ExchangeId.POLYMARKET,
    )
