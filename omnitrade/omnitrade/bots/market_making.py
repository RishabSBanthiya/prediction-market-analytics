"""
Market making bot.

Loop: select markets -> generate quotes -> place/cancel orders -> manage inventory
Works with any ExchangeClient through the unified interface.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import Side, Environment, OrderType
from ..core.models import OrderRequest, Quote
from ..core.errors import OmniTradeError
from ..exchanges.base import ExchangeClient
from ..components.quote_engines import QuoteEngine
from ..components.market_selectors import MarketSelector
from ..components.inventory import InventoryManager
from ..components.executors import DryRunExecutor
from ..risk.coordinator import RiskCoordinator

logger = logging.getLogger(__name__)


class MarketMakingBot:
    """
    Market making bot.

    Generates two-sided quotes and manages inventory.
    Exchange-agnostic through ExchangeClient interface.
    """

    def __init__(
        self,
        agent_id: str,
        client: ExchangeClient,
        quote_engine: QuoteEngine,
        market_selector: MarketSelector,
        risk: RiskCoordinator,
        inventory: Optional[InventoryManager] = None,
        environment: Environment = Environment.PAPER,
        max_instruments: int = 5,
        refresh_interval: float = 10.0,
    ):
        self.agent_id = agent_id
        self.client = client
        self.quote_engine = quote_engine
        self.market_selector = market_selector
        self.risk = risk
        self.inventory = inventory or InventoryManager()
        self.environment = environment
        self.max_instruments = max_instruments
        self.refresh_interval = refresh_interval
        self._running = False
        self._active_orders: dict[str, list[str]] = {}  # instrument_id -> [order_ids]

    async def start(self) -> None:
        if not self.client.is_connected:
            await self.client.connect()
        self.risk.startup(self.agent_id, "market_making", self.client.exchange_id)
        await self.inventory.sync_from_exchange(self.client)
        logger.info(f"MarketMakingBot '{self.agent_id}' started on {self.client.exchange_id.value}")

    async def stop(self) -> None:
        self._running = False
        # Cancel all outstanding orders
        for instrument_id, order_ids in self._active_orders.items():
            for oid in order_ids:
                await self.client.cancel_order(oid, instrument_id)
        self._active_orders.clear()
        self.risk.shutdown(self.agent_id)
        logger.info(f"MarketMakingBot '{self.agent_id}' stopped")

    async def run(self, interval_seconds: Optional[float] = None) -> None:
        """Main loop."""
        interval = interval_seconds or self.refresh_interval
        self._running = True
        await self.start()

        try:
            while self._running:
                try:
                    await self._iteration()
                except OmniTradeError as e:
                    logger.error(f"MM error: {e}")
                    self.risk.record_failure()
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    self.risk.record_failure()

                self.risk.heartbeat(self.agent_id)
                await asyncio.sleep(interval)
        finally:
            await self.stop()

    async def _iteration(self) -> None:
        """Single MM iteration: select -> cancel old -> quote -> place."""
        # Update balance
        balance = await self.client.get_balance()
        self.risk.update_equity(balance.total_equity)

        # Select markets
        instruments = await self.market_selector.select(self.client)
        instruments = instruments[:self.max_instruments]

        for inst in instruments:
            instrument_id = inst.instrument_id

            # Cancel existing orders for this instrument
            old_orders = self._active_orders.get(instrument_id, [])
            for oid in old_orders:
                await self.client.cancel_order(oid, instrument_id)
            self._active_orders[instrument_id] = []

            # Get current inventory
            inv = self.inventory.get_inventory(instrument_id)

            # Generate quote
            quote = await self.quote_engine.generate_quote(
                self.client, instrument_id, inv
            )
            if quote is None:
                continue

            # Place orders (paper mode just logs)
            if self.environment == Environment.PAPER:
                logger.info(
                    f"[DRY RUN] MM {instrument_id}: "
                    f"BID {quote.bid_size:.2f} @ ${quote.bid_price:.4f} | "
                    f"ASK {quote.ask_size:.2f} @ ${quote.ask_price:.4f} "
                    f"(spread={quote.spread:.2%})"
                )
                continue

            # Place bid
            bid_result = await self.client.place_order(OrderRequest(
                instrument_id=instrument_id,
                side=Side.BUY,
                size=quote.bid_size,
                price=quote.bid_price,
                order_type=OrderType.LIMIT,
            ))
            if bid_result.success and bid_result.order_id:
                self._active_orders.setdefault(instrument_id, []).append(bid_result.order_id)

            # Place ask
            ask_result = await self.client.place_order(OrderRequest(
                instrument_id=instrument_id,
                side=Side.SELL,
                size=quote.ask_size,
                price=quote.ask_price,
                order_type=OrderType.LIMIT,
            ))
            if ask_result.success and ask_result.order_id:
                self._active_orders.setdefault(instrument_id, []).append(ask_result.order_id)

            # Update inventory if filled
            if bid_result.filled_size > 0:
                self.inventory.update_from_fill(instrument_id, Side.BUY, bid_result.filled_size * bid_result.filled_price)
            if ask_result.filled_size > 0:
                self.inventory.update_from_fill(instrument_id, Side.SELL, ask_result.filled_size * ask_result.filled_price)
