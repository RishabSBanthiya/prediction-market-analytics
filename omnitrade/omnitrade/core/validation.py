"""
Startup configuration validation with clear, actionable error messages.

Validates all required config values, types, and ranges before the bot
starts, catching misconfigurations early rather than letting them surface
as cryptic errors deep in exchange client initialization.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .config import Config, ExchangeConfig, RiskConfig
from .enums import Environment, ExchangeId
from .errors import ConfigError

logger = logging.getLogger(__name__)

# Valid log levels accepted by Python logging
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass
class ValidationError:
    """A single configuration validation error."""

    field: str
    message: str
    hint: str = ""

    def __str__(self) -> str:
        s = f"  - {self.field}: {self.message}"
        if self.hint:
            s += f"\n    Hint: {self.hint}"
        return s


@dataclass
class ValidationResult:
    """Aggregated result of configuration validation."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Configuration is valid if there are no errors (warnings are ok)."""
        return len(self.errors) == 0

    def add_error(self, field_name: str, message: str, hint: str = "") -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field_name, message=message, hint=hint))

    def add_warning(self, field_name: str, message: str, hint: str = "") -> None:
        """Add a non-fatal validation warning."""
        self.warnings.append(ValidationError(field=field_name, message=message, hint=hint))

    def format_report(self) -> str:
        """Format a human-readable validation report."""
        lines: list[str] = []
        if self.errors:
            lines.append(f"Configuration validation failed ({len(self.errors)} error(s)):\n")
            for err in self.errors:
                lines.append(str(err))
        if self.warnings:
            if lines:
                lines.append("")
            lines.append(f"Warnings ({len(self.warnings)}):\n")
            for warn in self.warnings:
                lines.append(str(warn))
        return "\n".join(lines)


def validate_exchange_config(
    exchange_config: ExchangeConfig,
    environment: Environment,
    result: ValidationResult,
) -> None:
    """Validate a single exchange's configuration.

    Args:
        exchange_config: The exchange config to validate.
        environment: Current environment (PAPER/LIVE).
        result: ValidationResult to accumulate errors/warnings into.
    """
    ex_name = exchange_config.exchange.value
    prefix = f"{ex_name}"

    if not exchange_config.enabled:
        return

    # Validate API base URL
    if exchange_config.api_base:
        parsed = urlparse(exchange_config.api_base)
        if parsed.scheme not in ("http", "https"):
            result.add_error(
                f"{prefix}.api_base",
                f"Invalid URL scheme '{parsed.scheme}' in '{exchange_config.api_base}'",
                hint="Must start with http:// or https://",
            )
    else:
        result.add_error(
            f"{prefix}.api_base",
            "API base URL is empty",
            hint=f"Set {ex_name.upper()}_API_BASE or use the default",
        )

    # Validate WebSocket URL if present
    if exchange_config.ws_url:
        parsed = urlparse(exchange_config.ws_url)
        if parsed.scheme not in ("ws", "wss"):
            result.add_error(
                f"{prefix}.ws_url",
                f"Invalid WebSocket URL scheme '{parsed.scheme}'",
                hint="Must start with ws:// or wss://",
            )

    # Rate limit validation
    if exchange_config.rate_limit_per_window <= 0:
        result.add_error(
            f"{prefix}.rate_limit_per_window",
            f"Must be positive, got {exchange_config.rate_limit_per_window}",
        )
    if exchange_config.rate_limit_window_seconds <= 0:
        result.add_error(
            f"{prefix}.rate_limit_window_seconds",
            f"Must be positive, got {exchange_config.rate_limit_window_seconds}",
        )

    # Exchange-specific credential validation
    if exchange_config.exchange == ExchangeId.POLYMARKET:
        _validate_polymarket_creds(exchange_config, environment, result)
    elif exchange_config.exchange == ExchangeId.KALSHI:
        _validate_kalshi_creds(exchange_config, environment, result)
    elif exchange_config.exchange == ExchangeId.HYPERLIQUID:
        _validate_hyperliquid_creds(exchange_config, environment, result)


def _validate_polymarket_creds(
    config: ExchangeConfig,
    environment: Environment,
    result: ValidationResult,
) -> None:
    """Validate Polymarket-specific configuration."""
    if not config.private_key:
        result.add_error(
            "polymarket.private_key",
            "POLYMARKET_PRIVATE_KEY is required when Polymarket is enabled",
            hint="Set POLYMARKET_PRIVATE_KEY in your .env file",
        )
    elif not config.private_key.startswith("0x") and len(config.private_key) != 64:
        # Ethereum private keys are 32 bytes = 64 hex chars, optionally prefixed with 0x
        if not (config.private_key.startswith("0x") and len(config.private_key) == 66):
            result.add_warning(
                "polymarket.private_key",
                "Private key format looks unusual (expected 64 hex chars or 0x-prefixed 66 chars)",
            )

    if not config.proxy_address:
        result.add_error(
            "polymarket.proxy_address",
            "POLYMARKET_PROXY_ADDRESS is required when Polymarket is enabled",
            hint="Set POLYMARKET_PROXY_ADDRESS in your .env file",
        )
    elif not config.proxy_address.startswith("0x"):
        result.add_warning(
            "polymarket.proxy_address",
            "Proxy address should be an Ethereum address starting with 0x",
        )

    if config.chain_id not in (137, 80001, 80002):
        result.add_warning(
            "polymarket.chain_id",
            f"Unusual chain ID {config.chain_id} (expected 137 for Polygon mainnet)",
        )


def _validate_kalshi_creds(
    config: ExchangeConfig,
    environment: Environment,
    result: ValidationResult,
) -> None:
    """Validate Kalshi-specific configuration."""
    if not config.api_key:
        result.add_error(
            "kalshi.api_key",
            "KALSHI_API_KEY is required when Kalshi is enabled",
            hint="Set KALSHI_API_KEY in your .env file",
        )

    if not config.rsa_key_path:
        result.add_error(
            "kalshi.rsa_key_path",
            "KALSHI_RSA_KEY_PATH is required when Kalshi is enabled",
            hint="Set KALSHI_RSA_KEY_PATH to the path of your RSA private key file",
        )
    else:
        key_path = Path(config.rsa_key_path)
        if not key_path.exists():
            result.add_error(
                "kalshi.rsa_key_path",
                f"RSA key file not found: {config.rsa_key_path}",
                hint="Check that the file path is correct and the file exists",
            )
        elif not key_path.is_file():
            result.add_error(
                "kalshi.rsa_key_path",
                f"RSA key path is not a file: {config.rsa_key_path}",
            )


def _validate_hyperliquid_creds(
    config: ExchangeConfig,
    environment: Environment,
    result: ValidationResult,
) -> None:
    """Validate Hyperliquid-specific configuration."""
    if not config.private_key:
        result.add_error(
            "hyperliquid.private_key",
            "HYPERLIQUID_PRIVATE_KEY is required when Hyperliquid is enabled",
            hint="Set HYPERLIQUID_PRIVATE_KEY in your .env file",
        )
    elif not config.private_key.startswith("0x") and len(config.private_key) != 64:
        if not (config.private_key.startswith("0x") and len(config.private_key) == 66):
            result.add_warning(
                "hyperliquid.private_key",
                "Private key format looks unusual (expected 64 hex chars or 0x-prefixed 66 chars)",
            )

    if config.ws_url and not config.ws_url.startswith(("ws://", "wss://")):
        result.add_error(
            "hyperliquid.ws_url",
            f"Invalid WebSocket URL: {config.ws_url}",
            hint="Must start with ws:// or wss://",
        )


def validate_risk_config(risk: RiskConfig, result: ValidationResult) -> None:
    """Validate risk configuration values and cross-field constraints.

    Note: RiskConfig.__post_init__ already validates ranges. This function
    catches type/parse errors that happen *before* __post_init__ runs
    (e.g., non-numeric env var values) and adds cross-field warnings.

    Args:
        risk: The risk config to validate.
        result: ValidationResult to accumulate errors/warnings into.
    """
    # Cross-field consistency (these are warnings since RiskConfig.__post_init__
    # handles the hard constraints)
    if risk.max_daily_drawdown_pct >= risk.max_total_drawdown_pct:
        result.add_warning(
            "risk.max_daily_drawdown_pct",
            f"Daily drawdown limit ({risk.max_daily_drawdown_pct}) >= total drawdown limit "
            f"({risk.max_total_drawdown_pct}), daily limit will never trigger first",
        )

    if risk.circuit_breaker_failures <= 0:
        result.add_error(
            "risk.circuit_breaker_failures",
            f"Must be positive, got {risk.circuit_breaker_failures}",
        )

    if risk.circuit_breaker_reset_seconds <= 0:
        result.add_error(
            "risk.circuit_breaker_reset_seconds",
            f"Must be positive, got {risk.circuit_breaker_reset_seconds}",
        )


def validate_config(config: Config) -> ValidationResult:
    """Run full validation on a Config instance.

    Validates all fields, types, ranges, and cross-field constraints.
    Returns a ValidationResult with all errors and warnings.

    Args:
        config: The application config to validate.

    Returns:
        ValidationResult with accumulated errors and warnings.
    """
    result = ValidationResult()

    # Environment validation
    if not isinstance(config.environment, Environment):
        result.add_error(
            "environment",
            f"Invalid environment: {config.environment}",
            hint="Must be 'paper' or 'live'",
        )

    # Log level validation
    if config.log_level.upper() not in _VALID_LOG_LEVELS:
        result.add_error(
            "log_level",
            f"Invalid log level '{config.log_level}'",
            hint=f"Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}",
        )

    # DB path validation
    db_dir = Path(config.db_path).parent
    if not db_dir.exists():
        result.add_warning(
            "db_path",
            f"Database directory does not exist: {db_dir}",
            hint="It will be created at startup, but ensure the path is correct",
        )

    # Check that at least one exchange is enabled
    enabled = config.enabled_exchanges()
    if not enabled:
        result.add_warning(
            "exchanges",
            "No exchanges are enabled. Set API keys to enable an exchange.",
            hint="e.g., set POLYMARKET_PRIVATE_KEY in your .env file",
        )

    # Validate each exchange config
    for ex_config in (config.polymarket, config.kalshi, config.hyperliquid):
        validate_exchange_config(ex_config, config.environment, result)

    # Validate risk config
    validate_risk_config(config.risk, result)

    # Live mode additional checks
    if config.environment == Environment.LIVE:
        if not enabled:
            result.add_error(
                "environment",
                "LIVE mode requires at least one enabled exchange",
                hint="Set API credentials for at least one exchange",
            )

    return result


def validate_config_for_exchange(
    config: Config,
    exchange: ExchangeId,
) -> ValidationResult:
    """Validate config specifically for running on a given exchange.

    This is a targeted validation that checks the exchange is enabled and
    properly configured, plus general config. Used by run_bot.py before
    starting a bot on a specific exchange.

    Args:
        config: The application config.
        exchange: The exchange the bot will run on.

    Returns:
        ValidationResult with accumulated errors and warnings.
    """
    result = validate_config(config)

    ex_config = config.get_exchange_config(exchange)
    if not ex_config.enabled:
        result.add_error(
            f"{exchange.value}.enabled",
            f"Exchange {exchange.value} is not enabled (missing API credentials)",
            hint=f"Set the required environment variables for {exchange.value}",
        )

    return result


def validate_env_types() -> ValidationResult:
    """Validate environment variable types before constructing Config.

    This catches malformed env vars (e.g., POLYMARKET_CHAIN_ID=abc) that
    would cause int()/float() to raise ValueError during Config.from_env().

    Should be called before Config.from_env() to give clear error messages
    instead of cryptic ValueErrors.

    Returns:
        ValidationResult with type-checking errors.
    """
    result = ValidationResult()

    int_vars = {
        "POLYMARKET_CHAIN_ID": "Polymarket chain ID",
        "POLYMARKET_RATE_LIMIT": "Polymarket rate limit",
        "KALSHI_RATE_LIMIT": "Kalshi rate limit",
        "HYPERLIQUID_RATE_LIMIT": "Hyperliquid rate limit",
        "CIRCUIT_BREAKER_FAILURES": "Circuit breaker failure count",
        "CIRCUIT_BREAKER_RESET_SECONDS": "Circuit breaker reset period",
        "RESERVATION_TTL_SECONDS": "Reservation TTL",
        "HEARTBEAT_INTERVAL_SECONDS": "Heartbeat interval",
        "STALE_AGENT_THRESHOLD_SECONDS": "Stale agent threshold",
    }

    float_vars = {
        "MAX_WALLET_EXPOSURE_PCT": "Max wallet exposure",
        "MAX_PER_AGENT_EXPOSURE_PCT": "Max per-agent exposure",
        "MAX_PER_MARKET_EXPOSURE_PCT": "Max per-market exposure",
        "MIN_TRADE_VALUE_USD": "Min trade value",
        "MAX_TRADE_VALUE_USD": "Max trade value",
        "MAX_SPREAD_PCT": "Max spread",
        "MAX_SLIPPAGE_PCT": "Max slippage",
        "MAX_DAILY_DRAWDOWN_PCT": "Max daily drawdown",
        "MAX_TOTAL_DRAWDOWN_PCT": "Max total drawdown",
    }

    for var, description in int_vars.items():
        val = os.getenv(var)
        if val is not None:
            try:
                int(val)
            except ValueError:
                result.add_error(
                    var,
                    f"{description} must be an integer, got '{val}'",
                    hint=f"Set {var} to a valid integer in your .env file",
                )

    for var, description in float_vars.items():
        val = os.getenv(var)
        if val is not None:
            try:
                float(val)
            except ValueError:
                result.add_error(
                    var,
                    f"{description} must be a number, got '{val}'",
                    hint=f"Set {var} to a valid number in your .env file",
                )

    return result


def validate_startup(
    config: Optional[Config] = None,
    exchange: Optional[ExchangeId] = None,
) -> Config:
    """Full startup validation: env types, config construction, and field validation.

    This is the main entry point for startup validation. It:
    1. Validates env var types (catches parse errors early)
    2. Constructs Config from environment
    3. Validates all config fields, types, and ranges
    4. Optionally validates for a specific exchange

    Raises ConfigError with a clear, multi-line report if validation fails.
    Logs warnings for non-fatal issues.

    Args:
        config: Optional pre-built config (skips env type checking and construction).
        exchange: Optional exchange to validate for specifically.

    Returns:
        The validated Config instance.

    Raises:
        ConfigError: If validation fails with actionable error messages.
    """
    # Step 1: Validate env var types (only if building from env)
    if config is None:
        env_result = validate_env_types()
        if not env_result.is_valid:
            raise ConfigError(
                "Environment variable type errors:\n" + env_result.format_report()
            )

    # Step 2: Build config
    if config is None:
        try:
            config = Config.from_env()
        except ValueError as e:
            raise ConfigError(f"Failed to build configuration: {e}") from e

    # Step 3: Validate config
    if exchange is not None:
        result = validate_config_for_exchange(config, exchange)
    else:
        result = validate_config(config)

    # Step 4: Log warnings
    for warning in result.warnings:
        logger.warning("Config warning: %s", warning)

    # Step 5: Raise on errors
    if not result.is_valid:
        raise ConfigError(result.format_report())

    logger.info("Configuration validated successfully")
    return config
