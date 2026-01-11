#!/usr/bin/env python3
"""
Risk Monitor CLI.

Monitor and manage multi-agent risk coordinator for all trading bots.

Supported Strategies (via unified entry point):
    python scripts/run_bot.py bond --dry-run      # Expiring markets
    python scripts/run_bot.py flow --dry-run      # Flow copy trading
    python scripts/run_bot.py arb --dry-run       # Delta-neutral arbitrage
    python scripts/run_bot.py stat-arb --dry-run  # Statistical arbitrage
    python scripts/run_bot.py sports --dry-run    # Sports portfolio

Usage:
    # View current status (with live sync)
    python scripts/risk_monitor.py status

    # View drawdown status
    python scripts/risk_monitor.py drawdown

    # View all agents
    python scripts/risk_monitor.py agents

    # Force sync with on-chain data
    python scripts/risk_monitor.py sync

    # Cleanup stale data
    python scripts/risk_monitor.py cleanup

    # Emergency stop (mark all agents as stopped)
    python scripts/risk_monitor.py stop-all

Reconciliation Commands:
    # Show reconciliation status (API vs computed positions)
    python scripts/risk_monitor.py recon

    # Run full verification and log discrepancies
    python scripts/risk_monitor.py verify

    # Show unresolved reconciliation issues
    python scripts/risk_monitor.py issues
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from polymarket.core.config import Config, get_config
from polymarket.core.models import AgentStatus
from polymarket.trading.storage.sqlite import SQLiteStorage
from polymarket.trading.chain_sync import ChainSyncService, fast_sync_from_api


def get_storage(config: Optional[Config] = None) -> SQLiteStorage:
    """Get storage instance"""
    config = config or get_config()
    return SQLiteStorage(config.db_path)


async def do_sync(wallet: str, config: Config, verbose: bool = True) -> bool:
    """Perform incremental sync with on-chain data"""
    try:
        sync_service = ChainSyncService(config)
        storage = SQLiteStorage(config.db_path)
        
        # Check if we have sync state
        with storage.transaction() as txn:
            sync_state = txn.get_chain_sync_state(wallet)
        
        if sync_state:
            if verbose:
                print(f"📡 Syncing from block {sync_state['last_synced_block']:,}...")
            result = await sync_service.incremental_sync(wallet)
        else:
            if verbose:
                print("📡 First sync - using fast API sync...")
            result = await fast_sync_from_api(wallet, config)
        
        # Verify and fix discrepancies
        is_valid, discrepancies = await sync_service.verify_sync_integrity(wallet)
        if not is_valid:
            if verbose:
                print(f"⚠️  Found {len(discrepancies)} discrepancies, fixing...")

        # Always call fix_discrepancies - it refreshes USDC balance from chain
        # and is idempotent (0 fixes if no position issues)
        await sync_service.fix_discrepancies(wallet)

        await sync_service.close()
        return True
    except Exception as e:
        if verbose:
            print(f"❌ Sync error: {e}")
        return False


def cmd_status(args):
    """Show overall status"""
    config = get_config()
    wallet = config.proxy_address or "unknown"
    
    # Perform sync first (unless --no-sync)
    if not getattr(args, 'no_sync', False):
        print("\n🔄 Syncing with on-chain data...")
        asyncio.run(do_sync(wallet, config, verbose=True))
    
    storage = get_storage()
    
    with storage.transaction() as txn:
        wallet_state = txn.get_wallet_state(wallet)
        # Also get all agents (not filtered by wallet) to see the full picture
        all_agents = txn.get_all_agents()
    
    print("\n" + "="*60)
    print("RISK MONITOR STATUS")
    print("="*60)
    
    print(f"\nWallet: {wallet[:10]}...{wallet[-6:] if len(wallet) > 16 else ''}")
    print(f"USDC Balance: ${wallet_state.usdc_balance:,.2f}")
    print(f"Positions Value: ${wallet_state.total_positions_value:,.2f}")
    print(f"Reserved: ${wallet_state.total_reserved:,.2f}")
    print(f"Available: ${wallet_state.available_capital:,.2f}")
    print(f"Total Exposure: ${wallet_state.total_exposure:,.2f} ({wallet_state.exposure_pct:.1%})")
    
    # Show agents - both for this wallet and total
    wallet_active = len([a for a in wallet_state.agents if a.status == AgentStatus.ACTIVE])
    all_active = len([a for a in all_agents if a.status == AgentStatus.ACTIVE])
    
    if wallet_active != all_active:
        print(f"\nActive Agents (this wallet): {wallet_active}")
        print(f"Active Agents (total): {all_active}")
        if all_active > wallet_active:
            print(f"  ⚠️  {all_active - wallet_active} agent(s) registered with different wallet")
    else:
        print(f"\nActive Agents: {all_active}")
    
    print(f"Open Positions: {len([p for p in wallet_state.positions])}")
    print(f"Active Reservations: {len([r for r in wallet_state.reservations if r.is_active])}")
    
    # Show agent breakdown if there are multiple agents
    if len(all_agents) > 0:
        print(f"\nAgent Breakdown:")
        for agent in all_agents:
            status_emoji = {
                AgentStatus.ACTIVE: "🟢",
                AgentStatus.STOPPED: "⚪",
                AgentStatus.CRASHED: "🔴",
            }.get(agent.status, "❓")
            print(f"  {status_emoji} {agent.agent_id} ({agent.agent_type}) - {agent.status.value}")
    
    print("="*60 + "\n")


def cmd_agents(args):
    """Show all agents"""
    storage = get_storage()
    
    with storage.transaction() as txn:
        agents = txn.get_all_agents()
    
    print("\n" + "="*60)
    print("REGISTERED AGENTS")
    print("="*60)
    
    if not agents:
        print("\nNo agents registered.")
    else:
        print(f"\n{'ID':<20} {'Type':<10} {'Status':<10} {'Last Heartbeat':<25}")
        print("-"*65)
        
        for agent in agents:
            hb_ago = agent.seconds_since_heartbeat
            if hb_ago < 60:
                hb_str = f"{hb_ago:.0f}s ago"
            elif hb_ago < 3600:
                hb_str = f"{hb_ago/60:.0f}m ago"
            else:
                hb_str = f"{hb_ago/3600:.1f}h ago"
            
            status_emoji = {
                AgentStatus.ACTIVE: "🟢",
                AgentStatus.STOPPED: "⚪",
                AgentStatus.CRASHED: "🔴",
            }.get(agent.status, "❓")
            
            print(f"{agent.agent_id:<20} {agent.agent_type:<10} {status_emoji} {agent.status.value:<8} {hb_str:<25}")
    
    print("="*60 + "\n")


def cmd_positions(args):
    """Show all positions"""
    config = get_config()
    wallet = config.proxy_address or "unknown"
    
    # Sync first
    if not getattr(args, 'no_sync', False):
        print("\n🔄 Syncing with on-chain data...")
        asyncio.run(do_sync(wallet, config, verbose=False))
    
    storage = get_storage()
    
    with storage.transaction() as txn:
        positions = txn.get_all_positions(wallet)
    
    print("\n" + "="*60)
    print("POSITIONS")
    print("="*60)
    
    if not positions:
        print("\nNo positions found.")
    else:
        print(f"\n{'Agent':<15} {'Token':<15} {'Shares':<12} {'Entry':<10} {'Current':<10} {'P&L':<12} {'Status':<8}")
        print("-"*85)
        
        for pos in positions:
            pnl = pos.unrealized_pnl
            pnl_str = f"${pnl:+.2f}" if pnl else "N/A"
            
            print(
                f"{pos.agent_id:<15} "
                f"{pos.token_id[:12]+'...':<15} "
                f"{pos.shares:<12.2f} "
                f"${pos.entry_price:<9.4f} "
                f"${(pos.current_price or 0):<9.4f} "
                f"{pnl_str:<12} "
                f"{pos.status.value:<8}"
            )
    
    print("="*60 + "\n")


def cmd_reservations(args):
    """Show active reservations"""
    storage = get_storage()
    config = get_config()
    wallet = config.proxy_address or "unknown"
    
    with storage.transaction() as txn:
        reservations = txn.get_all_reservations(wallet)
    
    print("\n" + "="*60)
    print("RESERVATIONS")
    print("="*60)
    
    active = [r for r in reservations if r.is_active]
    
    if not active:
        print("\nNo active reservations.")
    else:
        print(f"\n{'Agent':<15} {'Amount':<12} {'Market':<15} {'Expires':<20} {'Status':<10}")
        print("-"*75)
        
        for res in active:
            expires_in = (res.expires_at - datetime.now(timezone.utc)).total_seconds()
            expires_str = f"{expires_in:.0f}s" if expires_in > 0 else "EXPIRED"
            
            print(
                f"{res.agent_id:<15} "
                f"${res.amount_usd:<11.2f} "
                f"{res.market_id[:12]+'...':<15} "
                f"{expires_str:<20} "
                f"{res.status.value:<10}"
            )
    
    print(f"\nTotal reserved: ${sum(r.amount_usd for r in active):,.2f}")
    print("="*60 + "\n")


def cmd_sync(args):
    """Force sync with on-chain data"""
    config = get_config()
    wallet = config.proxy_address or "unknown"
    
    print("\n" + "="*60)
    print("CHAIN SYNC")
    print("="*60)
    
    print(f"\nWallet: {wallet[:10]}...{wallet[-6:]}")
    
    success = asyncio.run(do_sync(wallet, config, verbose=True))
    
    if success:
        print("\n✅ Sync complete!")
        
        # Show updated stats
        storage = get_storage()
        with storage.transaction() as txn:
            wallet_state = txn.get_wallet_state(wallet)
            computed = txn.get_computed_positions(wallet.lower())
        
        print(f"\nUpdated Stats:")
        print(f"  Positions: {len(computed)}")
        print(f"  Total Value: ${wallet_state.total_positions_value:,.2f}")
        print(f"  USDC Balance: ${wallet_state.usdc_balance:,.2f}")
    else:
        print("\n❌ Sync failed!")
    
    print("="*60 + "\n")


def cmd_drawdown(args):
    """Show drawdown status"""
    storage = get_storage()
    config = get_config()
    wallet = config.proxy_address or "unknown"
    
    # Sync first to get accurate data
    if not getattr(args, 'no_sync', False):
        print("\n🔄 Syncing with on-chain data...")
        asyncio.run(do_sync(wallet, config, verbose=False))
    
    with storage.transaction() as txn:
        wallet_state = txn.get_wallet_state(wallet)
        # Get transaction summary to show P&L
        summary = txn.get_transaction_summary(wallet.lower())
        # Get persisted drawdown state
        drawdown_state = txn.get_drawdown_state(wallet)
    
    total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value
    
    print("\n" + "="*60)
    print("DRAWDOWN STATUS")
    print("="*60)
    
    print(f"\nCurrent Equity: ${total_equity:,.2f}")
    print(f"Positions Value: ${wallet_state.total_positions_value:,.2f}")
    print(f"USDC Balance: ${wallet_state.usdc_balance:,.2f}")
    
    print(f"\nLimits from config:")
    print(f"  Max Daily Drawdown: {config.risk.max_daily_drawdown_pct:.1%}")
    print(f"  Max Total Drawdown: {config.risk.max_total_drawdown_pct:.1%}")
    
    # Show persisted drawdown tracking
    if drawdown_state:
        peak = drawdown_state["peak_equity"]
        daily_start = drawdown_state["daily_start_equity"]
        
        # Calculate current drawdowns
        daily_dd = (daily_start - total_equity) / daily_start if daily_start > 0 else 0
        total_dd = (peak - total_equity) / peak if peak > 0 else 0
        
        print(f"\nDrawdown Tracking (persisted):")
        print(f"  Peak Equity: ${peak:,.2f}")
        print(f"  Daily Start: ${daily_start:,.2f} (as of {drawdown_state['daily_start_date'].strftime('%Y-%m-%d') if drawdown_state['daily_start_date'] else 'N/A'})")
        print(f"  Current Daily DD: {daily_dd:.1%} (limit: {config.risk.max_daily_drawdown_pct:.1%})")
        print(f"  Current Total DD: {total_dd:.1%} (limit: {config.risk.max_total_drawdown_pct:.1%})")
        
        if drawdown_state["is_breached"]:
            print(f"\n  🚨 BREACHED: {drawdown_state['breach_reason']}")
        else:
            daily_remaining = max(0, config.risk.max_daily_drawdown_pct - daily_dd)
            total_remaining = max(0, config.risk.max_total_drawdown_pct - total_dd)
            print(f"\n  ✅ Within limits")
            print(f"     Daily headroom: {daily_remaining:.1%} (${daily_start * daily_remaining:,.2f})")
            print(f"     Total headroom: {total_remaining:.1%} (${peak * total_remaining:,.2f})")
    else:
        print(f"\n⚠️  No drawdown state saved yet - run the bot to start tracking")
    
    # Show transaction summary
    if summary:
        buy = summary.get('buy', {})
        sell = summary.get('sell', {})
        claim = summary.get('claim', {})
        print(f"\nTransaction Summary:")
        print(f"  Total Buys: {buy.get('count', 0)} (${buy.get('total_usdc', 0):,.2f})")
        print(f"  Total Sells: {sell.get('count', 0)} (${sell.get('total_usdc', 0):,.2f})")
        print(f"  Claims/Redemptions: {claim.get('count', 0)} (${claim.get('total_usdc', 0):,.2f})")
    
    print("="*60 + "\n")


def cmd_cleanup(args):
    """Cleanup stale data"""
    storage = get_storage()
    config = get_config()
    
    with storage.transaction() as txn:
        # Cleanup expired reservations
        expired_res = txn.cleanup_expired_reservations()
        
        # Cleanup stale agents
        stale_agents = txn.cleanup_stale_agents(
            config.risk.stale_agent_threshold_seconds
        )
    
    print("\n" + "="*60)
    print("CLEANUP RESULTS")
    print("="*60)
    
    print(f"\nExpired reservations cleaned: {expired_res}")
    print(f"Stale agents marked crashed: {stale_agents}")
    
    print("\n✅ Cleanup complete")
    print("="*60 + "\n")


def cmd_reconcile(args):
    """Show reconciliation status between API and computed positions"""
    config = get_config()
    storage = get_storage(config)
    wallet = config.proxy_address

    if not wallet:
        print("Error: PROXY_ADDRESS not configured")
        return

    # Sync first unless skipped
    if not args.no_sync:
        asyncio.run(do_sync(wallet, config))

    with storage.transaction() as txn:
        api_positions = txn.get_api_positions(wallet)
        computed_positions = txn.get_computed_positions(wallet)
        issues = txn.get_unresolved_issues(wallet)
        issue_summary = txn.get_issue_summary(wallet)
        gaps = txn.get_unresolved_gaps(wallet)

    print("\n" + "="*60)
    print("RECONCILIATION STATUS")
    print("="*60)

    # API positions (source of truth)
    print(f"\n📊 API POSITIONS (Source of Truth):")
    if api_positions:
        total_api_value = sum(p["shares"] * (p.get("current_price") or p.get("avg_price", 0)) for p in api_positions)
        print(f"   Total: {len(api_positions)} positions, ${total_api_value:,.2f} value")
        for pos in api_positions[:5]:  # Show first 5
            price = pos.get("current_price") or pos.get("avg_price", 0)
            value = pos["shares"] * price
            print(f"   - {pos['token_id'][:16]}... : {pos['shares']:.4f} shares @ ${price:.4f} = ${value:.2f}")
        if len(api_positions) > 5:
            print(f"   ... and {len(api_positions) - 5} more")
    else:
        print("   No positions")

    # Computed positions (from transactions)
    print(f"\n📝 COMPUTED POSITIONS (From Transactions):")
    if computed_positions:
        total_computed_value = sum(
            p["shares"] * p.get("avg_entry_price", 0)
            for p in computed_positions
        )
        print(f"   Total: {len(computed_positions)} positions, ${total_computed_value:,.2f} value")
        for pos in computed_positions[:5]:
            value = pos["shares"] * pos.get("avg_entry_price", 0)
            print(f"   - {pos['token_id'][:16]}... : {pos['shares']:.4f} shares @ ${pos.get('avg_entry_price', 0):.4f} = ${value:.2f}")
        if len(computed_positions) > 5:
            print(f"   ... and {len(computed_positions) - 5} more")
    else:
        print("   No positions")

    # Compare and show discrepancies
    api_by_token = {p["token_id"]: p for p in api_positions}
    computed_by_token = {p["token_id"]: p for p in computed_positions}
    all_tokens = set(api_by_token.keys()) | set(computed_by_token.keys())

    discrepancies = []
    for token_id in all_tokens:
        api_shares = api_by_token.get(token_id, {}).get("shares", 0)
        computed_shares = computed_by_token.get(token_id, {}).get("shares", 0)
        if abs(api_shares - computed_shares) > 0.001:
            discrepancies.append({
                "token_id": token_id,
                "api_shares": api_shares,
                "computed_shares": computed_shares,
                "diff": api_shares - computed_shares
            })

    print(f"\n⚖️  DISCREPANCIES:")
    if discrepancies:
        print(f"   Found {len(discrepancies)} discrepancies:")
        for d in discrepancies[:10]:
            status = "MISSING_TX" if d["api_shares"] > 0 and d["computed_shares"] == 0 else \
                    "EXTRA_TX" if d["api_shares"] == 0 and d["computed_shares"] > 0 else \
                    "MISMATCH"
            print(f"   - [{status}] {d['token_id'][:16]}...")
            print(f"     API: {d['api_shares']:.4f}, Computed: {d['computed_shares']:.4f}, Diff: {d['diff']:.4f}")
        if len(discrepancies) > 10:
            print(f"   ... and {len(discrepancies) - 10} more")
    else:
        print("   None - API and computed positions match!")

    # Show issue summary
    print(f"\n📋 LOGGED ISSUES:")
    if issue_summary:
        for issue_type, counts in issue_summary.items():
            print(f"   {issue_type}: {counts['unresolved']} unresolved / {counts['total']} total ({counts['auto_fixed']} auto-fixed)")
    else:
        print("   No issues logged")

    # Show sync gaps
    print(f"\n🔍 SYNC GAPS:")
    if gaps:
        print(f"   Found {len(gaps)} unresolved gaps:")
        for gap in gaps[:5]:
            print(f"   - Blocks {gap['from_block']:,} - {gap['to_block']:,} (retries: {gap['retry_count']})")
        if len(gaps) > 5:
            print(f"   ... and {len(gaps) - 5} more")
    else:
        print("   No sync gaps")

    print("="*60 + "\n")


def cmd_issues(args):
    """Show unresolved reconciliation issues"""
    config = get_config()
    storage = get_storage(config)
    wallet = config.proxy_address

    if not wallet:
        print("Error: PROXY_ADDRESS not configured")
        return

    with storage.transaction() as txn:
        issues = txn.get_unresolved_issues(wallet, limit=args.limit)

    print("\n" + "="*60)
    print("UNRESOLVED RECONCILIATION ISSUES")
    print("="*60)

    if not issues:
        print("\nNo unresolved issues found!")
    else:
        print(f"\nFound {len(issues)} unresolved issues:\n")
        for issue in issues:
            print(f"  [{issue['issue_type'].upper()}] Token: {issue.get('token_id', 'N/A')[:20]}...")
            print(f"    API Value: {issue.get('api_value', 'N/A')}")
            print(f"    Computed Value: {issue.get('computed_value', 'N/A')}")
            print(f"    Difference: {issue.get('difference', 'N/A')}")
            print(f"    Details: {issue.get('details', 'N/A')}")
            print(f"    Detected: {issue.get('detected_at', 'N/A')}")
            print()

    print("="*60 + "\n")


def cmd_verify(args):
    """Run full verification and force re-sync of API positions"""
    config = get_config()
    storage = get_storage(config)
    wallet = config.proxy_address

    if not wallet:
        print("Error: PROXY_ADDRESS not configured")
        return

    print("\n" + "="*60)
    print("RUNNING FULL VERIFICATION")
    print("="*60)

    async def run_verification():
        from polymarket.core.api import PolymarketAPI
        from polymarket.trading.risk_coordinator import RiskCoordinator

        api = PolymarketAPI(config)
        await api.connect()

        try:
            print("\n[1/4] Fetching API positions (source of truth)...")
            api_positions = await api.fetch_positions(wallet)
            print(f"       Found {len(api_positions)} positions")

            print("\n[2/4] Storing API positions...")
            with storage.transaction() as txn:
                count = txn.upsert_api_positions(wallet, api_positions)
                print(f"       Stored {count} positions")

            print("\n[3/4] Fetching USDC balance...")
            balance = await api.fetch_usdc_balance(wallet)
            print(f"       Balance: ${balance:,.2f}")

            print("\n[4/4] Comparing with transaction history...")
            with storage.transaction() as txn:
                computed = txn.get_computed_positions(wallet)
                txn.update_usdc_balance(wallet, balance)

            # Compare
            api_by_token = {p.token_id: p for p in api_positions}
            computed_by_token = {p["token_id"]: p for p in computed}
            all_tokens = set(api_by_token.keys()) | set(computed_by_token.keys())

            issues_found = 0
            for token_id in all_tokens:
                api_shares = api_by_token.get(token_id).shares if token_id in api_by_token else 0
                computed_shares = computed_by_token.get(token_id, {}).get("shares", 0)

                if abs(api_shares - computed_shares) > 0.001:
                    issues_found += 1
                    if api_shares > 0 and computed_shares == 0:
                        issue_type = "missing_tx"
                    elif api_shares == 0 and computed_shares > 0:
                        issue_type = "extra_tx"
                    else:
                        issue_type = "share_mismatch"

                    with storage.transaction() as txn:
                        txn.log_reconciliation_issue(
                            wallet_address=wallet,
                            issue_type=issue_type,
                            api_value=api_shares,
                            computed_value=computed_shares,
                            token_id=token_id,
                            details=f"API: {api_shares:.4f}, Computed: {computed_shares:.4f}"
                        )

            print(f"\nVerification complete:")
            print(f"  API Positions: {len(api_positions)}")
            print(f"  Computed Positions: {len(computed)}")
            print(f"  USDC Balance: ${balance:,.2f}")
            if issues_found > 0:
                print(f"  Issues Found: {issues_found} (logged to reconciliation_issues)")
            else:
                print(f"  Issues Found: None - everything matches!")

        finally:
            await api.close()

    asyncio.run(run_verification())
    print("="*60 + "\n")


def cmd_reset(args):
    """Reset risk monitoring to fresh state from API"""
    if not args.yes:
        confirm = input("\n  This will clear all transaction history and sync fresh from API. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    config = get_config()
    storage = get_storage(config)
    wallet = config.proxy_address

    if not wallet:
        print("Error: PROXY_ADDRESS not configured")
        return

    print("\n" + "="*60)
    print("RESETTING RISK MONITOR")
    print("="*60)

    # Step 1: Clear all stale data
    print("\n[1/4] Clearing stale data...")
    with storage.transaction() as txn:
        txn._execute("DELETE FROM transactions")
        txn._execute("DELETE FROM positions")
        txn._execute("DELETE FROM api_positions")
        txn._execute("DELETE FROM reconciliation_issues")
        txn._execute("DELETE FROM chain_sync_state")
        print("       Cleared: transactions, positions, api_positions, reconciliation_issues, chain_sync_state")

    # Step 2: Fetch fresh from API
    async def fetch_fresh():
        from polymarket.core.api import PolymarketAPI

        api = PolymarketAPI(config)
        await api.connect()

        try:
            print("\n[2/4] Fetching positions from Polymarket API...")
            positions = await api.fetch_positions(wallet)
            print(f"       Found {len(positions)} active positions")

            print("\n[3/4] Fetching USDC balance...")
            balance = await api.fetch_usdc_balance(wallet)
            print(f"       Balance: ${balance:,.2f}")

            print("\n[4/4] Storing fresh state...")
            with storage.transaction() as txn:
                # Store API positions
                if positions:
                    count = txn.upsert_api_positions(wallet, positions)
                    print(f"       Stored {count} positions")
                else:
                    print("       No positions to store")

                # Update USDC balance
                txn.update_usdc_balance(wallet, balance)

            return len(positions), balance

        finally:
            await api.close()

    n_positions, balance = asyncio.run(fetch_fresh())

    # Calculate position value
    with storage.transaction() as txn:
        wallet_state = txn.get_wallet_state(wallet)

    print("\n" + "="*60)
    print("RESET COMPLETE")
    print("="*60)
    print(f"\nWallet: {wallet[:10]}...{wallet[-6:]}")
    print(f"USDC Balance: ${balance:,.2f}")
    print(f"Positions: {n_positions}")
    print(f"Positions Value: ${wallet_state.total_positions_value:,.2f}")
    print(f"Total Equity: ${balance + wallet_state.total_positions_value:,.2f}")
    print("\n" + "="*60 + "\n")


def cmd_reset_drawdown(args):
    """Reset drawdown tracking to current equity"""
    config = get_config()
    storage = get_storage(config)
    wallet = config.proxy_address

    if not wallet:
        print("Error: PROXY_ADDRESS not configured")
        return

    # Sync first to get accurate wallet state
    print("Syncing wallet state...")
    asyncio.run(do_sync(wallet, config, verbose=False))

    with storage.transaction() as txn:
        wallet_state = txn.get_wallet_state(wallet)
        old_state = txn.get_drawdown_state(wallet)

    total_equity = wallet_state.usdc_balance + wallet_state.total_positions_value

    if not args.yes:
        print("\n" + "="*60)
        print("RESET DRAWDOWN")
        print("="*60)
        print(f"\nCurrent equity: ${total_equity:,.2f}")
        if old_state:
            print(f"Current peak: ${old_state['peak_equity']:,.2f}")
            print(f"Current daily start: ${old_state['daily_start_equity']:,.2f}")
            if old_state['is_breached']:
                print(f"Breach status: BREACHED ({old_state['breach_reason']})")
        print(f"\nThis will:")
        print(f"  - Set peak equity to ${total_equity:,.2f}")
        print(f"  - Set daily start equity to ${total_equity:,.2f}")
        print(f"  - Clear any breach status")
        print(f"  - Running bots will pick up the reset within 30 seconds")

        confirm = input("\nProceed with reset? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    # Reset drawdown state
    now = datetime.now(timezone.utc)
    with storage.transaction() as txn:
        txn.update_drawdown_state(
            wallet_address=wallet,
            peak_equity=total_equity,
            daily_start_equity=total_equity,
            daily_start_date=now,
            is_breached=False,
            breach_reason=None
        )

    print("\n" + "="*60)
    print("DRAWDOWN RESET COMPLETE")
    print("="*60)
    print(f"\nNew peak equity: ${total_equity:,.2f}")
    print(f"New daily start: ${total_equity:,.2f}")
    print(f"Breach status: Cleared")
    print(f"\nRunning bots will respect this reset within 30 seconds.")
    print("="*60 + "\n")


def cmd_stop_all(args):
    """Emergency stop all agents"""
    if not args.yes:
        confirm = input("\n  This will mark ALL agents as STOPPED. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    storage = get_storage()

    with storage.transaction() as txn:
        agents = txn.get_all_agents()
        stopped = 0
        
        for agent in agents:
            if agent.status == AgentStatus.ACTIVE:
                txn.update_agent_status(agent.agent_id, AgentStatus.STOPPED)
                stopped += 1
        
        # Release all reservations
        released = txn.release_all_reservations()
    
    print("\n" + "="*60)
    print("EMERGENCY STOP")
    print("="*60)
    
    print(f"\n🛑 Agents stopped: {stopped}")
    print(f"🔓 Reservations released: {released}")
    
    print("\n✅ All agents stopped")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Risk Monitor CLI - Monitor all trading bots (bond, flow, arb, stat-arb, sports)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status         Show overall status (syncs first)
  sync           Force sync with on-chain data
  agents         List all agents (bond-bot, flow-bot, arb-bot, stat-arb-bot, sports-bot)
  positions      Show all positions
  reservations   Show active reservations
  drawdown       Show drawdown status
  cleanup        Cleanup stale data
  reset          Reset to fresh state from API (clears all history)
  reset-drawdown Reset drawdown tracking (running bots will respect this)
  stop-all       Emergency stop all agents (running bots will stop within 10s)

Reconciliation Commands:
  recon        Show reconciliation status (API vs computed positions)
  verify       Run full verification and log discrepancies
  issues       Show unresolved reconciliation issues

Bot Entry Point:
  All bots use unified entry: python scripts/run_bot.py <strategy> [options]
  Strategies: bond, flow, arb, stat-arb, sports
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Status
    status_parser = subparsers.add_parser("status", help="Show overall status")
    status_parser.add_argument("--no-sync", action="store_true", help="Skip sync (use cached data)")

    # Sync
    subparsers.add_parser("sync", help="Force sync with on-chain data")

    # Agents
    subparsers.add_parser("agents", help="List all agents")

    # Positions
    positions_parser = subparsers.add_parser("positions", help="Show all positions")
    positions_parser.add_argument("--no-sync", action="store_true", help="Skip sync (use cached data)")

    # Reservations
    subparsers.add_parser("reservations", help="Show active reservations")

    # Drawdown
    drawdown_parser = subparsers.add_parser("drawdown", help="Show drawdown status")
    drawdown_parser.add_argument("--no-sync", action="store_true", help="Skip sync (use cached data)")

    # Cleanup
    subparsers.add_parser("cleanup", help="Cleanup stale data")

    # Reset
    reset_parser = subparsers.add_parser("reset", help="Reset to fresh state from API")
    reset_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # Reset drawdown
    reset_dd_parser = subparsers.add_parser("reset-drawdown", help="Reset drawdown tracking to current equity")
    reset_dd_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # Stop all
    stop_parser = subparsers.add_parser("stop-all", help="Emergency stop all agents")
    stop_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # Reconciliation commands
    recon_parser = subparsers.add_parser("recon", help="Show reconciliation status")
    recon_parser.add_argument("--no-sync", action="store_true", help="Skip sync (use cached data)")

    subparsers.add_parser("verify", help="Run full verification and log discrepancies")

    issues_parser = subparsers.add_parser("issues", help="Show unresolved reconciliation issues")
    issues_parser.add_argument("--limit", type=int, default=50, help="Max issues to show")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    logging.basicConfig(level=logging.WARNING)

    # Run command
    commands = {
        "status": cmd_status,
        "sync": cmd_sync,
        "agents": cmd_agents,
        "positions": cmd_positions,
        "reservations": cmd_reservations,
        "drawdown": cmd_drawdown,
        "cleanup": cmd_cleanup,
        "reset": cmd_reset,
        "reset-drawdown": cmd_reset_drawdown,
        "stop-all": cmd_stop_all,
        "recon": cmd_reconcile,
        "verify": cmd_verify,
        "issues": cmd_issues,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

