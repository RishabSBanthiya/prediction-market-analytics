"""
Multi-exchange configuration with validation.

Supports per-exchange settings, risk limits, and paper/live environments.
All configuration is loaded from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from .enums import ExchangeId, Environment

load_dotenv()


@dataclass
class ExchangeConfig:
    """Per-exchange configuration."""
    exchange: ExchangeId
    enabled: bool = False

    # API endpoints (defaults set per-exchange)
    api_base: str = ""
    ws_url: str = ""

    # Auth (loaded from env)
    api_key: str = ""
    api_secret: str = ""
    private_key: str = ""       # For wallet-based auth (Polymarket, Hyperliquid)

    # Rate limiting
    rate_limit_per_window: int = 100
    rate_limit_window_seconds: int = 10

    # Exchange-specific
    proxy_address: str = ""      # Polymarket proxy wallet
    chain_id: int = 137          # Polygon for Polymarket
    rsa_key_path: str = ""       # Kalshi RSA-PSS key file

    @classmethod
    def polymarket(cls) -> "ExchangeConfig":
        return cls(
            exchange=ExchangeId.POLYMARKET,
            enabled=bool(os.getenv("POLYMARKET_PRIVATE_KEY")),
            api_base=os.getenv("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com"),
            private_key=os.getenv("POLYMARKET_PRIVATE_KEY", ""),
            proxy_address=os.getenv("POLYMARKET_PROXY_ADDRESS", ""),
            chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            rate_limit_per_window=int(os.getenv("POLYMARKET_RATE_LIMIT", "9000")),
            rate_limit_window_seconds=10,
        )

    @classmethod
    def kalshi(cls) -> "ExchangeConfig":
        return cls(
            exchange=ExchangeId.KALSHI,
            enabled=bool(os.getenv("KALSHI_API_KEY")),
            api_base=os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2"),
            api_key=os.getenv("KALSHI_API_KEY", ""),
            rsa_key_path=os.getenv("KALSHI_RSA_KEY_PATH", ""),
            rate_limit_per_window=int(os.getenv("KALSHI_RATE_LIMIT", "100")),
            rate_limit_window_seconds=10,
        )

    @classmethod
    def hyperliquid(cls) -> "ExchangeConfig":
        return cls(
            exchange=ExchangeId.HYPERLIQUID,
            enabled=bool(os.getenv("HYPERLIQUID_PRIVATE_KEY")),
            api_base=os.getenv("HYPERLIQUID_API_BASE", "https://api.hyperliquid.xyz"),
            ws_url=os.getenv("HYPERLIQUID_WS_URL", "wss://api.hyperliquid.xyz/ws"),
            private_key=os.getenv("HYPERLIQUID_PRIVATE_KEY", ""),
            rate_limit_per_window=int(os.getenv("HYPERLIQUID_RATE_LIMIT", "1200")),
            rate_limit_window_seconds=60,
        )


@dataclass
class RiskConfig:
    """Risk management configuration - validated on creation."""

    # Wallet-level limits (per exchange)
    max_wallet_exposure_pct: float = 0.60
    max_per_agent_exposure_pct: float = 0.30
    max_per_market_exposure_pct: float = 0.10

    # Per-trade limits
    min_trade_value_usd: float = 10.0
    max_trade_value_usd: float = 500.0
    max_spread_pct: float = 0.03
    max_slippage_pct: float = 0.02

    # Safety limits
    max_daily_drawdown_pct: float = 0.05
    max_total_drawdown_pct: float = 0.15
    circuit_breaker_failures: int = 3
    circuit_breaker_reset_seconds: int = 600

    # Timing
    reservation_ttl_seconds: int = 60
    heartbeat_interval_seconds: int = 30
    stale_agent_threshold_seconds: int = 120

    def __post_init__(self):
        errors = []
        if not 0 < self.max_wallet_exposure_pct <= 1.0:
            errors.append(f"max_wallet_exposure_pct must be in (0, 1], got {self.max_wallet_exposure_pct}")
        if not 0 < self.max_per_agent_exposure_pct <= self.max_wallet_exposure_pct:
            errors.append(f"max_per_agent_exposure_pct must be in (0, {self.max_wallet_exposure_pct}]")
        if not 0 < self.max_per_market_exposure_pct <= self.max_per_agent_exposure_pct:
            errors.append(f"max_per_market_exposure_pct must be in (0, {self.max_per_agent_exposure_pct}]")
        if self.min_trade_value_usd <= 0:
            errors.append("min_trade_value_usd must be positive")
        if self.max_trade_value_usd <= self.min_trade_value_usd:
            errors.append("max_trade_value_usd must be > min_trade_value_usd")
        if not 0 < self.max_spread_pct < 1.0:
            errors.append(f"max_spread_pct must be in (0, 1), got {self.max_spread_pct}")
        if not 0 < self.max_slippage_pct < 1.0:
            errors.append(f"max_slippage_pct must be in (0, 1), got {self.max_slippage_pct}")
        if not 0 < self.max_daily_drawdown_pct < 1.0:
            errors.append(f"max_daily_drawdown_pct must be in (0, 1)")
        if not 0 < self.max_total_drawdown_pct < 1.0:
            errors.append(f"max_total_drawdown_pct must be in (0, 1)")
        if self.reservation_ttl_seconds <= 0:
            errors.append("reservation_ttl_seconds must be positive")
        if self.heartbeat_interval_seconds <= 0:
            errors.append("heartbeat_interval_seconds must be positive")
        if self.stale_agent_threshold_seconds <= self.heartbeat_interval_seconds:
            errors.append("stale_agent_threshold_seconds must be > heartbeat_interval_seconds")
        if errors:
            raise ValueError("Invalid RiskConfig:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            max_wallet_exposure_pct=float(os.getenv("MAX_WALLET_EXPOSURE_PCT", "0.60")),
            max_per_agent_exposure_pct=float(os.getenv("MAX_PER_AGENT_EXPOSURE_PCT", "0.30")),
            max_per_market_exposure_pct=float(os.getenv("MAX_PER_MARKET_EXPOSURE_PCT", "0.10")),
            min_trade_value_usd=float(os.getenv("MIN_TRADE_VALUE_USD", "10.0")),
            max_trade_value_usd=float(os.getenv("MAX_TRADE_VALUE_USD", "500.0")),
            max_spread_pct=float(os.getenv("MAX_SPREAD_PCT", "0.03")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.02")),
            max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05")),
            max_total_drawdown_pct=float(os.getenv("MAX_TOTAL_DRAWDOWN_PCT", "0.15")),
            circuit_breaker_failures=int(os.getenv("CIRCUIT_BREAKER_FAILURES", "3")),
            circuit_breaker_reset_seconds=int(os.getenv("CIRCUIT_BREAKER_RESET_SECONDS", "600")),
            reservation_ttl_seconds=int(os.getenv("RESERVATION_TTL_SECONDS", "60")),
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30")),
            stale_agent_threshold_seconds=int(os.getenv("STALE_AGENT_THRESHOLD_SECONDS", "120")),
        )


@dataclass
class Config:
    """Main application configuration."""

    # Environment
    environment: Environment = Environment.PAPER

    # Exchange configs
    polymarket: ExchangeConfig = field(default_factory=ExchangeConfig.polymarket)
    kalshi: ExchangeConfig = field(default_factory=ExchangeConfig.kalshi)
    hyperliquid: ExchangeConfig = field(default_factory=ExchangeConfig.hyperliquid)

    # Risk
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Storage
    db_path: str = "data/omnitrade.db"

    # Logging
    log_level: str = "INFO"

    def __post_init__(self):
        if not os.path.isabs(self.db_path):
            import pathlib
            project_root = pathlib.Path(__file__).resolve().parent.parent.parent
            self.db_path = str(project_root / self.db_path)

    def get_exchange_config(self, exchange: ExchangeId) -> ExchangeConfig:
        mapping = {
            ExchangeId.POLYMARKET: self.polymarket,
            ExchangeId.KALSHI: self.kalshi,
            ExchangeId.HYPERLIQUID: self.hyperliquid,
        }
        config = mapping.get(exchange)
        if config is None:
            raise ValueError(f"Unknown exchange: {exchange}")
        return config

    def enabled_exchanges(self) -> list[ExchangeId]:
        return [
            ex.exchange for ex in [self.polymarket, self.kalshi, self.hyperliquid]
            if ex.enabled
        ]

    @classmethod
    def from_env(cls) -> "Config":
        env_str = os.getenv("OMNITRADE_ENV", "paper").lower()
        environment = Environment.LIVE if env_str == "live" else Environment.PAPER
        return cls(
            environment=environment,
            polymarket=ExchangeConfig.polymarket(),
            kalshi=ExchangeConfig.kalshi(),
            hyperliquid=ExchangeConfig.hyperliquid(),
            risk=RiskConfig.from_env(),
            db_path=os.getenv("OMNITRADE_DB_PATH", "data/omnitrade.db"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


_default_config: Optional[Config] = None

def get_config() -> Config:
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config

def set_config(config: Config):
    global _default_config
    _default_config = config
