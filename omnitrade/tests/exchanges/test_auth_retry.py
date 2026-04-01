"""Tests for auth retry decorator and credential expiry tracking."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import aiohttp

from omnitrade.core.errors import AuthError
from omnitrade.exchanges.base import ExchangeAuth
from omnitrade.exchanges.auth_retry import (
    with_auth_retry,
    _is_auth_error,
    AUTH_HTTP_CODES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAuth(ExchangeAuth):
    """Minimal ExchangeAuth for testing."""

    def __init__(self, expires_at=None):
        super().__init__()
        self._authenticated = False
        self._expires_at = expires_at

    async def authenticate(self) -> None:
        self._authenticated = True
        self._auth_count += 1

    def is_authenticated(self) -> bool:
        return self._authenticated


class FakeClient:
    """Minimal object that looks like an ExchangeClient with _auth."""

    def __init__(self, auth: FakeAuth):
        self._auth = auth
        self._call_count = 0
        self._fail_first_n = 0  # how many initial calls should raise auth error

    @with_auth_retry
    async def do_request(self, value: str) -> str:
        """Simulates an API method."""
        self._call_count += 1
        if self._call_count <= self._fail_first_n:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=401,
                message="Unauthorized",
            )
        return f"ok-{value}"

    @with_auth_retry
    async def do_request_generic_error(self) -> str:
        """Simulates a generic auth message error from SDK."""
        self._call_count += 1
        if self._call_count <= self._fail_first_n:
            raise Exception("unauthorized: invalid api key")
        return "ok"

    @with_auth_retry
    async def do_request_non_auth_error(self) -> str:
        """Always raises a non-auth error."""
        raise ValueError("some random error")


# ---------------------------------------------------------------------------
# ExchangeAuth expiry tracking
# ---------------------------------------------------------------------------

class TestExchangeAuthExpiry:
    def test_no_expiry_by_default(self):
        auth = FakeAuth()
        assert auth.expires_at is None
        assert auth.is_expired is False

    def test_expired_when_in_past(self):
        auth = FakeAuth(expires_at=time.time() - 10)
        assert auth.is_expired is True

    def test_not_expired_when_in_future(self):
        auth = FakeAuth(expires_at=time.time() + 3600)
        assert auth.is_expired is False

    def test_auth_count_starts_at_zero(self):
        auth = FakeAuth()
        assert auth.auth_count == 0

    @pytest.mark.asyncio
    async def test_auth_count_increments(self):
        auth = FakeAuth()
        await auth.authenticate()
        assert auth.auth_count == 1
        await auth.authenticate()
        assert auth.auth_count == 2


# ---------------------------------------------------------------------------
# _is_auth_error detection
# ---------------------------------------------------------------------------

class TestIsAuthError:
    def test_aiohttp_401(self):
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=401, message="Unauthorized"
        )
        assert _is_auth_error(exc) is True

    def test_aiohttp_403(self):
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=403, message="Forbidden"
        )
        assert _is_auth_error(exc) is True

    def test_aiohttp_500_not_auth(self):
        exc = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500, message="Server Error"
        )
        assert _is_auth_error(exc) is False

    def test_auth_error_class(self):
        assert _is_auth_error(AuthError("bad key")) is True

    def test_generic_exception_with_auth_message(self):
        assert _is_auth_error(Exception("unauthorized request")) is True
        assert _is_auth_error(Exception("invalid api key for user")) is True
        assert _is_auth_error(Exception("token expired")) is True

    def test_generic_non_auth_exception(self):
        assert _is_auth_error(ValueError("timeout")) is False
        assert _is_auth_error(RuntimeError("network down")) is False


# ---------------------------------------------------------------------------
# with_auth_retry decorator
# ---------------------------------------------------------------------------

class TestWithAuthRetry:
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Happy path -- no auth error, no retry needed."""
        auth = FakeAuth()
        client = FakeClient(auth)
        result = await client.do_request("hello")
        assert result == "ok-hello"
        assert client._call_count == 1
        assert auth.auth_count == 0  # no re-auth triggered

    @pytest.mark.asyncio
    async def test_retry_on_401(self):
        """First call gets 401, re-authenticates, second call succeeds."""
        auth = FakeAuth()
        client = FakeClient(auth)
        client._fail_first_n = 1

        result = await client.do_request("world")
        assert result == "ok-world"
        assert client._call_count == 2  # first failed, second succeeded
        assert auth.auth_count == 1  # re-auth happened once

    @pytest.mark.asyncio
    async def test_retry_on_generic_auth_message(self):
        """Retry on SDK exception containing auth-related message."""
        auth = FakeAuth()
        client = FakeClient(auth)
        client._fail_first_n = 1

        result = await client.do_request_generic_error()
        assert result == "ok"
        assert auth.auth_count == 1

    @pytest.mark.asyncio
    async def test_non_auth_error_not_retried(self):
        """Non-auth errors propagate immediately without retry."""
        auth = FakeAuth()
        client = FakeClient(auth)

        with pytest.raises(ValueError, match="some random error"):
            await client.do_request_non_auth_error()
        assert auth.auth_count == 0  # no re-auth attempted

    @pytest.mark.asyncio
    async def test_both_attempts_fail_propagates(self):
        """When retry also fails with auth error, it propagates."""
        auth = FakeAuth()
        client = FakeClient(auth)
        client._fail_first_n = 999  # always fail

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await client.do_request("fail")
        assert exc_info.value.status == 401
        assert client._call_count == 2  # original + one retry
        assert auth.auth_count == 1  # re-auth happened but retry still failed

    @pytest.mark.asyncio
    async def test_proactive_refresh_on_expired_creds(self):
        """If credentials are expired, refresh BEFORE the first call."""
        auth = FakeAuth(expires_at=time.time() - 10)  # already expired
        client = FakeClient(auth)

        result = await client.do_request("proactive")
        assert result == "ok-proactive"
        # Auth was refreshed proactively before the call
        assert auth.auth_count == 1
        assert client._call_count == 1  # no retry needed

    @pytest.mark.asyncio
    async def test_no_proactive_refresh_when_not_expired(self):
        """No proactive refresh when credentials are still valid."""
        auth = FakeAuth(expires_at=time.time() + 3600)
        client = FakeClient(auth)

        result = await client.do_request("valid")
        assert result == "ok-valid"
        assert auth.auth_count == 0

    @pytest.mark.asyncio
    async def test_reauth_failure_raises_auth_error(self):
        """If re-authentication itself fails, raise AuthError."""
        auth = FakeAuth()

        # Make authenticate() fail on re-auth
        original_authenticate = auth.authenticate

        async def failing_authenticate():
            raise Exception("key revoked permanently")

        client = FakeClient(auth)
        client._fail_first_n = 1  # first call will trigger re-auth

        auth.authenticate = failing_authenticate

        with pytest.raises(AuthError, match="Re-authentication failed"):
            await client.do_request("doomed")

    @pytest.mark.asyncio
    async def test_preserves_function_arguments(self):
        """Verify that arguments are correctly forwarded."""
        auth = FakeAuth()
        client = FakeClient(auth)
        client._fail_first_n = 1

        result = await client.do_request("preserved-arg")
        assert result == "ok-preserved-arg"
