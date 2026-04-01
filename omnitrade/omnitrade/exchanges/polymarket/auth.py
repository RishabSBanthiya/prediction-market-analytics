"""Polymarket wallet-based authentication."""

import logging
from typing import Optional

from ...core.config import ExchangeConfig
from ...core.errors import AuthError
from ..base import ExchangeAuth

logger = logging.getLogger(__name__)


class PolymarketAuth(ExchangeAuth):
    """
    Polymarket uses wallet-based auth via py-clob-client.

    Requires:
    - POLYMARKET_PRIVATE_KEY: Ethereum private key
    - POLYMARKET_PROXY_ADDRESS: Polymarket proxy wallet address
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__()
        self._config = config
        self._authenticated = False
        self._client = None  # py-clob-client ClobClient instance

    async def authenticate(self) -> None:
        if not self._config.private_key:
            raise AuthError("POLYMARKET_PRIVATE_KEY not set")
        if not self._config.proxy_address:
            raise AuthError("POLYMARKET_PROXY_ADDRESS not set")

        try:
            from py_clob_client.client import ClobClient
            import asyncio

            def _create_client():
                return ClobClient(
                    host=self._config.api_base,
                    key=self._config.private_key,
                    chain_id=self._config.chain_id,
                    funder=self._config.proxy_address,
                )

            self._client = await asyncio.to_thread(_create_client)

            # Derive API creds
            def _derive_creds():
                self._client.set_api_creds(self._client.derive_api_key())

            await asyncio.to_thread(_derive_creds)
            self._authenticated = True
            self._auth_count += 1
            logger.info("Polymarket authentication successful (auth_count=%d)", self._auth_count)

        except ImportError:
            raise AuthError("py-clob-client not installed. Run: pip install py-clob-client")
        except Exception as e:
            raise AuthError(f"Polymarket auth failed: {e}")

    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def client(self):
        """Get the underlying py-clob-client ClobClient."""
        if not self._authenticated or self._client is None:
            raise AuthError("Not authenticated. Call authenticate() first.")
        return self._client
