"""
Centralized configuration with validation.

All configuration is loaded from environment variables with sensible defaults.
Configuration is validated on instantiation to fail fast on invalid values.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class RiskConfig:
    """Risk management configuration - validated on creation"""
    
    # Wallet-level limits
    max_wallet_exposure_pct: float = 0.80
    max_per_agent_exposure_pct: float = 0.40
    max_per_market_exposure_pct: float = 0.15
    
    # Per-trade limits
    min_trade_value_usd: float = 5.0
    max_trade_value_usd: float = 1000.0
    max_spread_pct: float = 0.03
    max_slippage_pct: float = 0.01
    
    # Safety limits
    max_daily_drawdown_pct: float = 0.10
    max_total_drawdown_pct: float = 0.25
    circuit_breaker_failures: int = 5
    circuit_breaker_reset_seconds: int = 300
    
    # Timing
    reservation_ttl_seconds: int = 60
    heartbeat_interval_seconds: int = 30
    stale_agent_threshold_seconds: int = 120
    
    # Rate limiting (Polymarket allows 9,000 requests per 10 seconds for CLOB API)
    api_rate_limit_per_10s: int = 9000
    api_rate_limit_window_seconds: int = 10
    
    def __post_init__(self):
        """Validate configuration on creation"""
        errors = []
        
        # Wallet exposure validation
        if not 0 < self.max_wallet_exposure_pct <= 1.0:
            errors.append(f"max_wallet_exposure_pct must be in (0, 1], got {self.max_wallet_exposure_pct}")
        
        if not 0 < self.max_per_agent_exposure_pct <= self.max_wallet_exposure_pct:
            errors.append(
                f"max_per_agent_exposure_pct must be in (0, {self.max_wallet_exposure_pct}], "
                f"got {self.max_per_agent_exposure_pct}"
            )
        
        if not 0 < self.max_per_market_exposure_pct <= self.max_per_agent_exposure_pct:
            errors.append(
                f"max_per_market_exposure_pct must be in (0, {self.max_per_agent_exposure_pct}], "
                f"got {self.max_per_market_exposure_pct}"
            )
        
        # Trade value validation
        if self.min_trade_value_usd <= 0:
            errors.append(f"min_trade_value_usd must be positive, got {self.min_trade_value_usd}")
        
        if self.max_trade_value_usd <= self.min_trade_value_usd:
            errors.append(
                f"max_trade_value_usd must be > min_trade_value_usd, "
                f"got {self.max_trade_value_usd} <= {self.min_trade_value_usd}"
            )
        
        # Spread and slippage
        if not 0 < self.max_spread_pct < 1.0:
            errors.append(f"max_spread_pct must be in (0, 1), got {self.max_spread_pct}")
        
        if not 0 < self.max_slippage_pct < 1.0:
            errors.append(f"max_slippage_pct must be in (0, 1), got {self.max_slippage_pct}")
        
        # Drawdown limits
        if not 0 < self.max_daily_drawdown_pct < 1.0:
            errors.append(f"max_daily_drawdown_pct must be in (0, 1), got {self.max_daily_drawdown_pct}")
        
        if not 0 < self.max_total_drawdown_pct < 1.0:
            errors.append(f"max_total_drawdown_pct must be in (0, 1), got {self.max_total_drawdown_pct}")
        
        # Timing validation
        if self.reservation_ttl_seconds <= 0:
            errors.append(f"reservation_ttl_seconds must be positive, got {self.reservation_ttl_seconds}")
        
        if self.heartbeat_interval_seconds <= 0:
            errors.append(f"heartbeat_interval_seconds must be positive, got {self.heartbeat_interval_seconds}")
        
        if self.stale_agent_threshold_seconds <= self.heartbeat_interval_seconds:
            errors.append(
                f"stale_agent_threshold_seconds must be > heartbeat_interval_seconds, "
                f"got {self.stale_agent_threshold_seconds} <= {self.heartbeat_interval_seconds}"
            )
        
        # Rate limiting
        if self.api_rate_limit_per_10s <= 0:
            errors.append(f"api_rate_limit_per_10s must be positive, got {self.api_rate_limit_per_10s}")
        if self.api_rate_limit_window_seconds <= 0:
            errors.append(f"api_rate_limit_window_seconds must be positive, got {self.api_rate_limit_window_seconds}")
        
        if errors:
            raise ValueError("Invalid RiskConfig:\n" + "\n".join(f"  - {e}" for e in errors))
    
    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Load configuration from environment variables"""
        return cls(
            # V5: Tightened exposure limits for better risk management
            max_wallet_exposure_pct=float(os.getenv("MAX_WALLET_EXPOSURE_PCT", "0.60")),  # 60% (was 80%)
            max_per_agent_exposure_pct=float(os.getenv("MAX_PER_AGENT_EXPOSURE_PCT", "0.30")),  # 30% (was 40%)
            max_per_market_exposure_pct=float(os.getenv("MAX_PER_MARKET_EXPOSURE_PCT", "0.10")),  # 10% (was 15%)
            min_trade_value_usd=float(os.getenv("MIN_TRADE_VALUE_USD", "10.0")),  # $10 (was $5)
            max_trade_value_usd=float(os.getenv("MAX_TRADE_VALUE_USD", "500.0")),  # $500 (was $1000)
            max_spread_pct=float(os.getenv("MAX_SPREAD_PCT", "0.03")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.01")),
            # V5: Tightened drawdown limits
            max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "0.05")),  # 5% (was 10%)
            max_total_drawdown_pct=float(os.getenv("MAX_TOTAL_DRAWDOWN_PCT", "0.15")),  # 15% (was 25%)
            circuit_breaker_failures=int(os.getenv("CIRCUIT_BREAKER_FAILURES", "3")),  # 3 (was 5)
            circuit_breaker_reset_seconds=int(os.getenv("CIRCUIT_BREAKER_RESET_SECONDS", "600")),  # 10min (was 5min)
            reservation_ttl_seconds=int(os.getenv("RESERVATION_TTL_SECONDS", "60")),
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL_SECONDS", "30")),
            stale_agent_threshold_seconds=int(os.getenv("STALE_AGENT_THRESHOLD_SECONDS", "120")),
            api_rate_limit_per_10s=int(os.getenv("API_RATE_LIMIT_PER_10S", "9000")),
            api_rate_limit_window_seconds=int(os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "10")),
        )


@dataclass
class ChainSyncConfig:
    """Chain synchronization configuration"""
    
    # Contract addresses (Polygon mainnet)
    ctf_contract_address: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Polymarket CTF
    neg_risk_ctf_exchange: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # NegRisk CTF Exchange
    neg_risk_adapter: str = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"  # NegRisk Adapter
    
    # Sync settings
    # Note: Most public RPCs limit getLogs to ~100-200 blocks for busy contracts
    batch_size: int = 100  # Blocks per batch when fetching events
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Initial sync - set to a block near when Polymarket CTF was active
    # The Polymarket CTF contract was deployed around block 39,410,580
    # Polymarket prediction markets became active around block 48,000,000 (mid-2023)
    # Most 2024+ activity starts around block 55,000,000+
    # For most users, setting this ~1 week before their first trade is optimal
    # Use CHAIN_SYNC_INITIAL_BLOCK env var to customize
    initial_sync_block: int = 65000000  # Covers most 2024+ activity
    
    @classmethod
    def from_env(cls) -> "ChainSyncConfig":
        """Load from environment variables"""
        return cls(
            ctf_contract_address=os.getenv(
                "CTF_CONTRACT_ADDRESS",
                "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            ),
            neg_risk_ctf_exchange=os.getenv(
                "NEG_RISK_CTF_EXCHANGE",
                "0xC5d563A36AE78145C45a50134d48A1215220f80a"
            ),
            neg_risk_adapter=os.getenv(
                "NEG_RISK_ADAPTER",
                "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
            ),
            batch_size=int(os.getenv("CHAIN_SYNC_BATCH_SIZE", "2000")),
            max_retries=int(os.getenv("CHAIN_SYNC_MAX_RETRIES", "3")),
            retry_delay_seconds=float(os.getenv("CHAIN_SYNC_RETRY_DELAY", "1.0")),
            initial_sync_block=int(os.getenv("CHAIN_SYNC_INITIAL_BLOCK", "65000000")),
        )


@dataclass
class Config:
    """Main application configuration"""
    
    # API endpoints
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    data_api_base: str = "https://data-api.polymarket.com"
    
    # Blockchain
    polygon_rpc_url: str = "https://polygon-rpc.com"
    usdc_contract_address: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    chain_id: int = 137
    
    # Credentials (loaded from env)
    private_key: Optional[str] = None
    proxy_address: Optional[str] = None
    
    # Storage
    db_path: str = "data/risk_state.db"
    
    # Risk configuration
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Chain sync configuration
    chain_sync: ChainSyncConfig = field(default_factory=ChainSyncConfig)
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Load credentials from environment and resolve paths"""
        if self.private_key is None:
            self.private_key = os.getenv("PRIVATE_KEY")
        if self.proxy_address is None:
            self.proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
        
        # Make db_path absolute if it's relative
        if not os.path.isabs(self.db_path):
            # Find project root (where .env file would be)
            import pathlib
            current_file = pathlib.Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            self.db_path = str(project_root / self.db_path)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load full configuration from environment variables"""
        return cls(
            gamma_api_base=os.getenv("GAMMA_API_BASE", "https://gamma-api.polymarket.com"),
            clob_host=os.getenv("CLOB_HOST", "https://clob.polymarket.com"),
            data_api_base=os.getenv("DATA_API_BASE", "https://data-api.polymarket.com"),
            polygon_rpc_url=os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
            chain_id=int(os.getenv("CHAIN_ID", "137")),
            db_path=os.getenv("RISK_DB_PATH", "data/risk_state.db"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            risk=RiskConfig.from_env(),
            chain_sync=ChainSyncConfig.from_env(),
        )
    
    def validate_credentials(self) -> bool:
        """Check if required credentials are present"""
        return bool(self.private_key and self.proxy_address)
    
    def require_credentials(self):
        """Raise error if credentials are missing"""
        if not self.validate_credentials():
            raise ValueError(
                "Missing credentials. Please create a .env file with:\n"
                "PRIVATE_KEY=0x...\n"
                "POLYMARKET_PROXY_ADDRESS=0x..."
            )


# Global default configuration
_default_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance"""
    global _default_config
    if _default_config is None:
        _default_config = Config.from_env()
    return _default_config


def set_config(config: Config):
    """Set the global configuration instance"""
    global _default_config
    _default_config = config


