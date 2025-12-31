#!/usr/bin/env python3
"""
Cleanup dead/stale positions from the database.

This script finds positions in the database that reference markets/tokens 
whose orderbooks no longer exist (404 errors), indicating the market has
been closed or resolved.

Usage:
    python scripts/cleanup_dead_positions.py [--dry-run]
    
Options:
    --dry-run    Show what would be cleaned up without making changes
"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polymarket.core.config import get_config
from polymarket.core.api import PolymarketAPI
from polymarket.trading.storage.sqlite import SQLiteStorage
from polymarket.core.models import PositionStatus


async def check_orderbook_exists(api: PolymarketAPI, token_id: str) -> bool:
    """Check if an orderbook exists for a token."""
    try:
        bid, ask, spread = await api.get_spread(token_id)
        return bid is not None or ask is not None
    except Exception as e:
        print(f"  Error checking {token_id[:20]}...: {e}")
        return False


async def cleanup_dead_positions(dry_run: bool = True):
    """Find and cleanup positions with dead orderbooks."""
    config = get_config()
    storage = SQLiteStorage(config.db_path)
    
    print("=" * 60)
    print("🔍 SCANNING FOR DEAD POSITIONS")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if dry_run else '⚠️  LIVE - Will make changes!'}")
    print(f"Database: {config.db_path}")
    print()
    
    async with PolymarketAPI(config) as api:
        # Get all positions (open and orphan)
        with storage.transaction() as txn:
            open_positions = txn.get_all_positions(
                config.proxy_address, 
                PositionStatus.OPEN
            )
            orphan_positions = txn.get_all_positions(
                config.proxy_address, 
                PositionStatus.ORPHAN
            )
        
        all_positions = open_positions + orphan_positions
        print(f"📊 Found {len(all_positions)} active positions to check")
        print()
        
        dead_positions = []
        
        for pos in all_positions:
            print(f"Checking {pos.token_id[:32]}...")
            exists = await check_orderbook_exists(api, pos.token_id)
            
            if exists:
                print(f"  ✅ Orderbook exists")
            else:
                print(f"  ❌ NO ORDERBOOK - market likely closed/resolved")
                dead_positions.append(pos)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.2)
        
        print()
        print("=" * 60)
        print("📋 RESULTS")
        print("=" * 60)
        
        if not dead_positions:
            print("✅ No dead positions found!")
            return
        
        print(f"Found {len(dead_positions)} dead position(s):")
        print()
        
        for pos in dead_positions:
            print(f"  Token: {pos.token_id}")
            print(f"  Market: {pos.market_id}")
            print(f"  Shares: {pos.shares:.4f}")
            print(f"  Entry: ${pos.entry_price:.4f}")
            print(f"  Current: ${pos.current_price:.4f}" if pos.current_price else "  Current: N/A")
            print(f"  Status: {pos.status.value}")
            print(f"  ID: {pos.id}")
            print()
        
        if dry_run:
            print("=" * 60)
            print("🔄 DRY RUN - No changes made")
            print("Run with --live to actually cleanup these positions")
            print("=" * 60)
        else:
            print("=" * 60)
            print("🗑️ CLEANING UP...")
            print("=" * 60)
            
            with storage.transaction() as txn:
                for pos in dead_positions:
                    txn.mark_position_closed(pos.id)
                    print(f"  ✅ Marked position {pos.id} as closed")
            
            print()
            print(f"✅ Cleaned up {len(dead_positions)} dead position(s)")
            print("The bot should now operate normally.")


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup dead/stale positions from the database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be cleaned up without making changes (default)"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually cleanup the dead positions"
    )
    
    args = parser.parse_args()
    
    # If --live is specified, don't do dry run
    dry_run = not args.live
    
    asyncio.run(cleanup_dead_positions(dry_run))


if __name__ == "__main__":
    main()

