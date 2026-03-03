"""
Hyperliquid wallet-based authentication.

Uses eth-account for signing. The hyperliquid-python-sdk handles
signing internally, but we manage the wallet here.
"""

import logging
from typing import Optional

from ...core.config import ExchangeConfig
from ...core.errors import AuthError
from ..base import ExchangeAuth

logger = logging.getLogger(__name__)


class HyperliquidAuth(ExchangeAuth):
    """
    Hyperliquid wallet authentication.

    Requires HYPERLIQUID_PRIVATE_KEY env var.
    """

    def __init__(self, config: ExchangeConfig):
        self._config = config
        self._authenticated = False
        self._address: Optional[str] = None
        self._info = None   # hyperliquid Info API
        self._exchange = None  # hyperliquid Exchange API

    async def authenticate(self) -> None:
        if not self._config.private_key:
            raise AuthError("HYPERLIQUID_PRIVATE_KEY not set")

        try:
            from eth_account import Account
            import asyncio

            account = Account.from_key(self._config.private_key)
            self._address = account.address

            # Initialize SDK
            from hyperliquid.info import Info
            from hyperliquid.exchange import Exchange

            def _init():
                info = Info(self._config.api_base, skip_ws=True)
                exchange = Exchange(account, self._config.api_base)
                return info, exchange

            self._info, self._exchange = await asyncio.to_thread(_init)
            self._authenticated = True
            logger.info(f"Hyperliquid auth successful: {self._address}")

        except ImportError as e:
            raise AuthError(f"Missing dependency: {e}. Run: pip install hyperliquid-python-sdk eth-account")
        except Exception as e:
            raise AuthError(f"Hyperliquid auth failed: {e}")

    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def address(self) -> str:
        if not self._address:
            raise AuthError("Not authenticated")
        return self._address

    @property
    def info(self):
        if not self._authenticated:
            raise AuthError("Not authenticated")
        return self._info

    @property
    def exchange(self):
        if not self._authenticated:
            raise AuthError("Not authenticated")
        return self._exchange
