#!/usr/bin/env python3
"""Quick script to check agent statuses"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from polymarket.core.config import get_config
from polymarket.core.models import AgentStatus
from polymarket.trading.storage.sqlite import SQLiteStorage
from datetime import datetime, timezone

config = get_config()
storage = SQLiteStorage(config.db_path)

with storage.transaction() as txn:
    agents = txn.get_all_agents()

print("\n" + "="*70)
print("AGENT STATUS CHECK")
print("="*70)

if not agents:
    print("\nNo agents registered.")
else:
    now = datetime.now(timezone.utc)
    print(f"\nCurrent time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"\n{'Agent ID':<20} {'Type':<10} {'Status':<10} {'Last Heartbeat':<25} {'Age':<10}")
    print("-"*70)
    
    for agent in agents:
        hb_age = (now - agent.last_heartbeat).total_seconds()
        
        if hb_age < 60:
            age_str = f"{hb_age:.0f}s"
        elif hb_age < 3600:
            age_str = f"{hb_age/60:.0f}m"
        else:
            age_str = f"{hb_age/3600:.1f}h"
        
        status_emoji = {
            AgentStatus.ACTIVE: "🟢",
            AgentStatus.STOPPED: "⚪",
            AgentStatus.CRASHED: "🔴",
        }.get(agent.status, "❓")
        
        # Check if agent should be marked as stale
        stale_threshold = config.risk.stale_agent_threshold_seconds
        is_stale = hb_age > stale_threshold and agent.status == AgentStatus.ACTIVE
        
        status_display = f"{status_emoji} {agent.status.value}"
        if is_stale:
            status_display += " ⚠️ STALE"
        
        print(
            f"{agent.agent_id:<20} "
            f"{agent.agent_type:<10} "
            f"{status_display:<20} "
            f"{agent.last_heartbeat.strftime('%Y-%m-%d %H:%M:%S UTC'):<25} "
            f"{age_str:<10}"
        )

print("\n" + "="*70)
print(f"\nHeartbeat interval: {config.risk.heartbeat_interval_seconds}s")
print(f"Stale threshold: {config.risk.stale_agent_threshold_seconds}s")
print("="*70 + "\n")



