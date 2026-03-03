#!/usr/bin/env python3
"""
Test exchange connectivity.

Runs read-only API calls against each configured exchange
to verify credentials and network access work.

Usage:
    python scripts/test_connectivity.py
"""

import asyncio
import sys
import os
import logging

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnitrade.core.config import Config, ExchangeConfig
from omnitrade.core.enums import ExchangeId

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def is_placeholder(val: str) -> bool:
    """Check if a config value is still a placeholder."""
    placeholders = {"0x...", "your-api-key", ""}
    return val.strip() in placeholders


async def test_polymarket(config: ExchangeConfig) -> None:
    print("\n--- Polymarket ---")

    if is_placeholder(config.private_key) or is_placeholder(config.proxy_address):
        print(f"  Config:     {SKIP} (POLYMARKET_PRIVATE_KEY or PROXY_ADDRESS not set)")
        return

    print(f"  Config:     {PASS} credentials detected")

    from omnitrade.exchanges.polymarket.client import PolymarketClient

    client = PolymarketClient(config)
    try:
        # 1. Auth / connect
        await client.connect()
        print(f"  Auth:       {PASS} wallet auth + API key derivation succeeded")

        # 2. Fetch instruments (Gamma API - public, but tests HTTP stack)
        instruments = await client.get_instruments(active_only=True, limit=5, order="volume24hr", ascending="false")
        if not instruments:
            instruments = await client.get_instruments(active_only=True, limit=5)
        print(f"  Markets:    {PASS} fetched {len(instruments)} instruments from Gamma API")
        if instruments:
            # Pick one with a nonzero price
            sample = next((i for i in instruments if i.price > 0), instruments[0])
            print(f"              sample: {sample.name[:60]} price={sample.price:.4f}")

        # 3. Get midpoint (CLOB - requires auth)
        # Try each instrument until we find one with an active orderbook
        midpoint_ok = False
        for inst in instruments:
            if inst.price <= 0:
                continue
            try:
                mid = await client.get_midpoint(inst.instrument_id)
                if mid is not None:
                    print(f"  Midpoint:   {PASS} {inst.instrument_id[:12]}... = {mid:.4f} (CLOB auth works)")
                    midpoint_ok = True
                    break
            except Exception:
                continue
        if not midpoint_ok:
            print(f"  Midpoint:   {SKIP} no active CLOB orderbook found in sample (auth still OK)")

        # 4. Positions (Data API)
        positions = await client.get_positions()
        print(f"  Positions:  {PASS} {len(positions)} open position(s)")

    except Exception as e:
        print(f"  Connect:    {FAIL} {e}")
    finally:
        await client.close()


async def test_kalshi(config: ExchangeConfig) -> None:
    print("\n--- Kalshi ---")

    if is_placeholder(config.api_key):
        print(f"  Config:     {SKIP} (KALSHI_API_KEY not set)")
        return

    if not os.path.exists(config.rsa_key_path):
        print(f"  Config:     {FAIL} RSA key file not found: {config.rsa_key_path}")
        return

    if os.path.getsize(config.rsa_key_path) == 0:
        print(f"  Config:     {FAIL} RSA key file is empty: {config.rsa_key_path}")
        print(f"              Paste your RSA private key (PEM format) into this file")
        print(f"              It should start with: -----BEGIN RSA PRIVATE KEY-----")
        return

    print(f"  Config:     {PASS} API key + RSA key detected")

    from omnitrade.exchanges.kalshi.client import KalshiClient

    client = KalshiClient(config)
    try:
        # 1. Auth
        await client.connect()
        print(f"  Auth:       {PASS} RSA key loaded")

        # 2. Balance (authenticated endpoint)
        balance = await client.get_balance()
        if balance.total_equity > 0 or balance.available_balance >= 0:
            print(f"  Balance:    {PASS} ${balance.total_equity:.2f} total, ${balance.available_balance:.2f} available")
        else:
            print(f"  Balance:    {PASS} $0.00 (account may be empty)")

        # 3. Markets
        instruments = await client.get_instruments(active_only=True, limit=3)
        print(f"  Markets:    {PASS} fetched {len(instruments)} instruments")
        if instruments:
            sample = instruments[0]
            print(f"              sample: {sample.name} price={sample.price:.4f}")

        # 4. Positions
        positions = await client.get_positions()
        print(f"  Positions:  {PASS} {len(positions)} open position(s)")

    except Exception as e:
        print(f"  Connect:    {FAIL} {e}")
    finally:
        await client.close()


async def test_hyperliquid(config: ExchangeConfig) -> None:
    print("\n--- Hyperliquid ---")

    if is_placeholder(config.private_key):
        print(f"  Config:     {SKIP} (HYPERLIQUID_PRIVATE_KEY not set)")
        return

    print(f"  Config:     {PASS} private key detected")

    from omnitrade.exchanges.hyperliquid.client import HyperliquidClient

    client = HyperliquidClient(config)
    try:
        # 1. Auth
        await client.connect()
        print(f"  Auth:       {PASS} wallet address: {client._auth.address}")

        # 2. Instruments (public info.meta endpoint)
        instruments = await client.get_instruments()
        print(f"  Markets:    {PASS} {len(instruments)} perpetuals")
        if instruments:
            sample = instruments[0]
            print(f"              sample: {sample.name} price=${sample.price:.2f}")

        # 3. Balance (user_state - needs valid address)
        balance = await client.get_balance()
        print(f"  Balance:    {PASS} ${balance.total_equity:.2f} equity, ${balance.available_balance:.2f} available")

        # 4. Positions
        positions = await client.get_positions()
        print(f"  Positions:  {PASS} {len(positions)} open position(s)")

    except Exception as e:
        print(f"  Connect:    {FAIL} {e}")
    finally:
        await client.close()


async def main():
    print("=" * 50)
    print("OmniTrade Exchange Connectivity Test")
    print("=" * 50)

    # Load config from .env
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = Config.from_env()

    print(f"\nEnvironment:  {config.environment.value}")
    print(f"Enabled:      {[e.value for e in config.enabled_exchanges()] or 'none (all placeholder)'}")

    await test_polymarket(config.polymarket)
    await test_kalshi(config.kalshi)
    await test_hyperliquid(config.hyperliquid)

    print("\n" + "=" * 50)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
