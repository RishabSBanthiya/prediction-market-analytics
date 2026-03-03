"""Exchange client registry - lookup and creation."""

from typing import Optional

from ..core.enums import ExchangeId
from ..core.config import Config, ExchangeConfig
from .base import ExchangeClient


_registry: dict[ExchangeId, type[ExchangeClient]] = {}


def register_exchange(exchange_id: ExchangeId):
    """Decorator to register an exchange client class."""
    def decorator(cls: type[ExchangeClient]):
        _registry[exchange_id] = cls
        return cls
    return decorator


def create_client(exchange: ExchangeId, config: Optional[Config] = None) -> ExchangeClient:
    """Create an exchange client instance."""
    if config is None:
        from ..core.config import get_config
        config = get_config()

    exchange_config = config.get_exchange_config(exchange)

    if exchange not in _registry:
        raise ValueError(
            f"No client registered for {exchange.value}. "
            f"Available: {[e.value for e in _registry]}"
        )

    client_class = _registry[exchange]
    return client_class(exchange_config)


def available_exchanges() -> list[ExchangeId]:
    """List registered exchanges."""
    return list(_registry.keys())
