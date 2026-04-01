"""
Auth retry decorator for exchange clients.

Wraps async methods so that when an API call fails with an authentication
error (HTTP 401/403, or known auth exception patterns), the client
re-authenticates once and retries the call.  If the retry also fails the
exception propagates normally.

Usage inside an ExchangeClient subclass::

    @with_auth_retry
    async def _request(self, method: str, path: str, ...) -> dict:
        ...

The decorated method's ``self`` must expose:
- ``self._auth``  - an :class:`ExchangeAuth` instance with ``authenticate()``
"""

import asyncio
import functools
import logging
from typing import TypeVar, Callable, Any

import aiohttp

from ..core.errors import AuthError

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# HTTP status codes that indicate an authentication/authorization problem.
AUTH_HTTP_CODES = {401, 403}

# Substrings in exception messages that hint at auth failures across SDKs.
AUTH_ERROR_PATTERNS = (
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "auth failed",
    "not authenticated",
    "token expired",
    "session expired",
    "invalid signature",
    "api key",
)


def _is_auth_error(exc: BaseException) -> bool:
    """Determine whether *exc* represents a recoverable auth failure."""
    # aiohttp raises ClientResponseError for non-2xx when raise_for_status is called
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in AUTH_HTTP_CODES

    # Our own AuthError
    if isinstance(exc, AuthError):
        return True

    # Catch generic exceptions whose message hints at auth issues
    msg = str(exc).lower()
    return any(pattern in msg for pattern in AUTH_ERROR_PATTERNS)


def with_auth_retry(fn: F) -> F:
    """Decorator that retries an async method once after re-authenticating.

    The owning class must have an ``_auth`` (ExchangeAuth) attribute.
    """

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # If credentials are known-expired, refresh proactively before the call.
        if hasattr(self, "_auth") and self._auth.is_expired:
            logger.info(
                "%s: credentials expired, refreshing before request",
                type(self).__name__,
            )
            await _reauthenticate(self)

        try:
            return await fn(self, *args, **kwargs)
        except Exception as exc:
            if not _is_auth_error(exc):
                raise

            logger.warning(
                "%s: auth error on %s, attempting re-authentication: %s",
                type(self).__name__,
                fn.__name__,
                exc,
            )

            await _reauthenticate(self)

            # Retry once after re-auth
            return await fn(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


async def _reauthenticate(client: Any) -> None:
    """Re-run authentication on a client's auth object.

    Uses ``client._auth_lock`` (created lazily) to prevent concurrent
    re-authentication attempts from racing each other.

    Raises AuthError if re-authentication itself fails.
    """
    auth = getattr(client, "_auth", None)
    if auth is None:
        raise AuthError("Client has no _auth attribute; cannot re-authenticate")

    # Lazily create the lock so existing clients don't need __init__ changes.
    if not hasattr(client, "_auth_lock"):
        client._auth_lock = asyncio.Lock()

    async with client._auth_lock:
        try:
            await auth.authenticate()
            logger.info(
                "%s: re-authentication successful (auth_count=%d)",
                type(client).__name__,
                auth.auth_count,
            )
        except Exception as exc:
            logger.error(
                "%s: re-authentication failed: %s",
                type(client).__name__,
                exc,
            )
            raise AuthError(f"Re-authentication failed: {exc}") from exc
