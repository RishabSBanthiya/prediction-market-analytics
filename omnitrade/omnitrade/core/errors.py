"""Exception hierarchy for omnitrade."""


class OmniTradeError(Exception):
    """Base exception for all omnitrade errors."""
    pass


class ConfigError(OmniTradeError):
    """Invalid configuration."""
    pass


class AuthError(OmniTradeError):
    """Authentication/authorization failure."""
    pass


class ExchangeError(OmniTradeError):
    """Base for exchange-level errors."""
    def __init__(self, exchange: str, message: str):
        self.exchange = exchange
        super().__init__(f"[{exchange}] {message}")


class ConnectionError(ExchangeError):
    """Cannot connect to exchange."""
    pass


class RateLimitError(ExchangeError):
    """Rate limit exceeded."""
    def __init__(self, exchange: str, retry_after: float | None = None):
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f" (retry after {retry_after:.1f}s)"
        super().__init__(exchange, msg)


class OrderError(ExchangeError):
    """Order placement/management failure."""
    pass


class InsufficientBalanceError(ExchangeError):
    """Not enough funds."""
    def __init__(self, exchange: str, required: float, available: float):
        self.required = required
        self.available = available
        super().__init__(exchange, f"Need ${required:.2f}, have ${available:.2f}")


class InstrumentNotFoundError(ExchangeError):
    """Instrument/market not found."""
    def __init__(self, exchange: str, instrument_id: str):
        self.instrument_id = instrument_id
        super().__init__(exchange, f"Instrument not found: {instrument_id}")


class RiskLimitError(OmniTradeError):
    """Risk limit breached."""
    def __init__(self, limit_type: str, message: str):
        self.limit_type = limit_type
        super().__init__(f"Risk limit [{limit_type}]: {message}")


class StorageError(OmniTradeError):
    """Storage/persistence failure."""
    pass
