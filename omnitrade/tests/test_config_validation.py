"""Tests for startup configuration validation."""

import os
import pytest
from unittest.mock import patch

from omnitrade.core.config import Config, ExchangeConfig, RiskConfig
from omnitrade.core.enums import Environment, ExchangeId
from omnitrade.core.errors import ConfigError
from omnitrade.core.validation import (
    ValidationResult,
    validate_config,
    validate_config_for_exchange,
    validate_env_types,
    validate_exchange_config,
    validate_risk_config,
    validate_startup,
)


class TestValidationResult:
    def test_empty_result_is_valid(self):
        result = ValidationResult()
        assert result.is_valid

    def test_result_with_errors_is_invalid(self):
        result = ValidationResult()
        result.add_error("field", "message")
        assert not result.is_valid

    def test_result_with_only_warnings_is_valid(self):
        result = ValidationResult()
        result.add_warning("field", "message")
        assert result.is_valid

    def test_format_report_includes_errors(self):
        result = ValidationResult()
        result.add_error("api_key", "Missing API key", hint="Set API_KEY env var")
        report = result.format_report()
        assert "api_key" in report
        assert "Missing API key" in report
        assert "Set API_KEY env var" in report

    def test_format_report_includes_warnings(self):
        result = ValidationResult()
        result.add_warning("chain_id", "Unusual chain")
        report = result.format_report()
        assert "chain_id" in report
        assert "Unusual chain" in report


class TestExchangeConfigValidation:
    def test_disabled_exchange_skips_validation(self):
        config = ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False)
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_polymarket_missing_private_key(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="https://clob.polymarket.com",
            private_key="",
            proxy_address="0xABC123",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("private_key" in e.field for e in result.errors)

    def test_polymarket_missing_proxy_address(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="https://clob.polymarket.com",
            private_key="0x" + "a" * 64,
            proxy_address="",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("proxy_address" in e.field for e in result.errors)

    def test_polymarket_valid_config(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="https://clob.polymarket.com",
            private_key="0x" + "a" * 64,
            proxy_address="0x" + "b" * 40,
            chain_id=137,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert result.is_valid

    def test_kalshi_missing_api_key(self):
        config = ExchangeConfig(
            exchange=ExchangeId.KALSHI,
            enabled=True,
            api_base="https://api.elections.kalshi.com/trade-api/v2",
            api_key="",
            rsa_key_path="/some/path",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("api_key" in e.field for e in result.errors)

    def test_kalshi_missing_rsa_key_path(self):
        config = ExchangeConfig(
            exchange=ExchangeId.KALSHI,
            enabled=True,
            api_base="https://api.elections.kalshi.com/trade-api/v2",
            api_key="my-key",
            rsa_key_path="",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("rsa_key_path" in e.field for e in result.errors)

    def test_kalshi_nonexistent_rsa_file(self):
        config = ExchangeConfig(
            exchange=ExchangeId.KALSHI,
            enabled=True,
            api_base="https://api.elections.kalshi.com/trade-api/v2",
            api_key="my-key",
            rsa_key_path="/nonexistent/key.pem",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("not found" in e.message for e in result.errors)

    def test_hyperliquid_missing_private_key(self):
        config = ExchangeConfig(
            exchange=ExchangeId.HYPERLIQUID,
            enabled=True,
            api_base="https://api.hyperliquid.xyz",
            private_key="",
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("private_key" in e.field for e in result.errors)

    def test_hyperliquid_valid_config(self):
        config = ExchangeConfig(
            exchange=ExchangeId.HYPERLIQUID,
            enabled=True,
            api_base="https://api.hyperliquid.xyz",
            ws_url="wss://api.hyperliquid.xyz/ws",
            private_key="0x" + "c" * 64,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert result.is_valid

    def test_invalid_api_base_url(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="ftp://invalid.com",
            private_key="0x" + "a" * 64,
            proxy_address="0x" + "b" * 40,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("api_base" in e.field for e in result.errors)

    def test_empty_api_base_url(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="",
            private_key="0x" + "a" * 64,
            proxy_address="0x" + "b" * 40,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("api_base" in e.field for e in result.errors)

    def test_invalid_ws_url(self):
        config = ExchangeConfig(
            exchange=ExchangeId.HYPERLIQUID,
            enabled=True,
            api_base="https://api.hyperliquid.xyz",
            ws_url="http://not-a-ws-url",
            private_key="0x" + "c" * 64,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("ws_url" in e.field for e in result.errors)

    def test_negative_rate_limit(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="https://clob.polymarket.com",
            private_key="0x" + "a" * 64,
            proxy_address="0x" + "b" * 40,
            rate_limit_per_window=-1,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        assert not result.is_valid
        assert any("rate_limit_per_window" in e.field for e in result.errors)

    def test_unusual_chain_id_warns(self):
        config = ExchangeConfig(
            exchange=ExchangeId.POLYMARKET,
            enabled=True,
            api_base="https://clob.polymarket.com",
            private_key="0x" + "a" * 64,
            proxy_address="0x" + "b" * 40,
            chain_id=999,
        )
        result = ValidationResult()
        validate_exchange_config(config, Environment.PAPER, result)
        # Unusual chain_id is a warning, not an error
        assert result.is_valid
        assert any("chain_id" in w.field for w in result.warnings)


class TestRiskConfigValidation:
    def test_valid_risk_config(self):
        risk = RiskConfig()
        result = ValidationResult()
        validate_risk_config(risk, result)
        assert result.is_valid

    def test_daily_drawdown_exceeds_total_warns(self):
        risk = RiskConfig(
            max_daily_drawdown_pct=0.20,
            max_total_drawdown_pct=0.15,
        )
        result = ValidationResult()
        validate_risk_config(risk, result)
        assert any("daily drawdown" in w.message.lower() for w in result.warnings)

    def test_zero_circuit_breaker_failures(self):
        risk = RiskConfig(circuit_breaker_failures=0)
        result = ValidationResult()
        validate_risk_config(risk, result)
        assert not result.is_valid

    def test_negative_circuit_breaker_reset(self):
        risk = RiskConfig(circuit_breaker_reset_seconds=-10)
        result = ValidationResult()
        validate_risk_config(risk, result)
        assert not result.is_valid


class TestValidateConfig:
    def _make_config(self, **overrides) -> Config:
        """Create a Config with no enabled exchanges (avoids credential checks)."""
        return Config(
            environment=overrides.get("environment", Environment.PAPER),
            polymarket=ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False),
            kalshi=ExchangeConfig(exchange=ExchangeId.KALSHI, enabled=False),
            hyperliquid=ExchangeConfig(exchange=ExchangeId.HYPERLIQUID, enabled=False),
            risk=overrides.get("risk", RiskConfig()),
            log_level=overrides.get("log_level", "INFO"),
            db_path=overrides.get("db_path", "/tmp/test.db"),
        )

    def test_valid_config_passes(self):
        config = self._make_config()
        result = validate_config(config)
        assert result.is_valid

    def test_invalid_log_level(self):
        config = self._make_config(log_level="VERBOSE")
        result = validate_config(config)
        assert not result.is_valid
        assert any("log_level" in e.field for e in result.errors)

    def test_no_enabled_exchanges_warns(self):
        config = self._make_config()
        result = validate_config(config)
        assert any("exchanges" in w.field for w in result.warnings)

    def test_live_mode_no_exchanges_errors(self):
        config = self._make_config(environment=Environment.LIVE)
        result = validate_config(config)
        assert not result.is_valid
        assert any("LIVE" in e.message for e in result.errors)


class TestValidateConfigForExchange:
    def test_disabled_exchange_errors(self):
        config = Config(
            polymarket=ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False),
            kalshi=ExchangeConfig(exchange=ExchangeId.KALSHI, enabled=False),
            hyperliquid=ExchangeConfig(exchange=ExchangeId.HYPERLIQUID, enabled=False),
            db_path="/tmp/test.db",
        )
        result = validate_config_for_exchange(config, ExchangeId.POLYMARKET)
        assert not result.is_valid
        assert any("not enabled" in e.message for e in result.errors)


class TestValidateEnvTypes:
    def test_valid_env_vars(self):
        env = {"POLYMARKET_CHAIN_ID": "137", "MAX_WALLET_EXPOSURE_PCT": "0.6"}
        with patch.dict(os.environ, env, clear=False):
            result = validate_env_types()
            assert result.is_valid

    def test_invalid_int_env_var(self):
        with patch.dict(os.environ, {"POLYMARKET_CHAIN_ID": "not_a_number"}, clear=False):
            result = validate_env_types()
            assert not result.is_valid
            assert any("POLYMARKET_CHAIN_ID" in e.field for e in result.errors)

    def test_invalid_float_env_var(self):
        with patch.dict(os.environ, {"MAX_WALLET_EXPOSURE_PCT": "sixty"}, clear=False):
            result = validate_env_types()
            assert not result.is_valid
            assert any("MAX_WALLET_EXPOSURE_PCT" in e.field for e in result.errors)

    def test_unset_env_vars_are_fine(self):
        """Unset env vars are OK -- defaults are used."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_env_types()
            assert result.is_valid


class TestValidateStartup:
    def test_valid_config_returns_config(self):
        config = Config(
            polymarket=ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False),
            kalshi=ExchangeConfig(exchange=ExchangeId.KALSHI, enabled=False),
            hyperliquid=ExchangeConfig(exchange=ExchangeId.HYPERLIQUID, enabled=False),
            db_path="/tmp/test.db",
        )
        result = validate_startup(config=config)
        assert isinstance(result, Config)

    def test_invalid_config_raises_config_error(self):
        config = Config(
            polymarket=ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False),
            kalshi=ExchangeConfig(exchange=ExchangeId.KALSHI, enabled=False),
            hyperliquid=ExchangeConfig(exchange=ExchangeId.HYPERLIQUID, enabled=False),
            log_level="INVALID_LEVEL",
            db_path="/tmp/test.db",
        )
        with pytest.raises(ConfigError):
            validate_startup(config=config)

    def test_invalid_env_types_raises_config_error(self):
        with patch.dict(os.environ, {"POLYMARKET_CHAIN_ID": "abc"}, clear=False):
            with pytest.raises(ConfigError, match="POLYMARKET_CHAIN_ID"):
                validate_startup()

    def test_exchange_specific_validation(self):
        config = Config(
            polymarket=ExchangeConfig(exchange=ExchangeId.POLYMARKET, enabled=False),
            kalshi=ExchangeConfig(exchange=ExchangeId.KALSHI, enabled=False),
            hyperliquid=ExchangeConfig(exchange=ExchangeId.HYPERLIQUID, enabled=False),
            db_path="/tmp/test.db",
        )
        with pytest.raises(ConfigError, match="not enabled"):
            validate_startup(config=config, exchange=ExchangeId.POLYMARKET)
