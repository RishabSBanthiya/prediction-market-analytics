"""
Kalshi RSA-PSS authentication.

Kalshi API uses RSA-PSS key signing for authentication.
Each request is signed with the private key.
"""

import base64
import hashlib
import logging
import time
from typing import Optional
from pathlib import Path

from ...core.config import ExchangeConfig
from ...core.errors import AuthError
from ..base import ExchangeAuth

logger = logging.getLogger(__name__)


class KalshiAuth(ExchangeAuth):
    """
    Kalshi RSA-PSS authentication.

    Signs requests using an RSA private key with PSS padding.
    Requires: KALSHI_API_KEY and KALSHI_RSA_KEY_PATH env vars.
    """

    def __init__(self, config: ExchangeConfig):
        super().__init__()
        self._config = config
        self._authenticated = False
        self._private_key = None
        self._api_key = config.api_key

    async def authenticate(self) -> None:
        if not self._config.api_key:
            raise AuthError("KALSHI_API_KEY not set")
        if not self._config.rsa_key_path:
            raise AuthError("KALSHI_RSA_KEY_PATH not set")

        key_path = Path(self._config.rsa_key_path)
        if not key_path.exists():
            raise AuthError(f"RSA key file not found: {key_path}")

        try:
            from cryptography.hazmat.primitives import serialization

            key_data = key_path.read_bytes()
            self._private_key = serialization.load_pem_private_key(key_data, password=None)
            self._authenticated = True
            self._auth_count += 1
            logger.info("Kalshi authentication successful (auth_count=%d)", self._auth_count)
        except ImportError:
            raise AuthError("cryptography package not installed. Run: pip install cryptography")
        except Exception as e:
            raise AuthError(f"Failed to load RSA key: {e}")

    def is_authenticated(self) -> bool:
        return self._authenticated

    def sign_request(self, method: str, path: str, timestamp_ms: Optional[int] = None) -> dict:
        """
        Sign a request and return auth headers.

        Returns dict with headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE, KALSHI-ACCESS-TIMESTAMP
        """
        if not self._authenticated or self._private_key is None:
            raise AuthError("Not authenticated")

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        ts = timestamp_ms or int(time.time() * 1000)
        ts_str = str(ts)

        # Message to sign: timestamp + method + path
        message = ts_str + method.upper() + path

        signature = self._private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts_str,
        }
