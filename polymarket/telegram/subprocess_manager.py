"""
Subprocess manager for trading bots.

Handles starting, stopping, and monitoring trading bot processes.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..core.models import AgentStatus
from ..trading.storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


@dataclass
class BotProcess:
    """Information about a running bot process."""
    agent_id: str
    bot_type: str
    pid: int
    started_at: str
    command: List[str]
    dry_run: bool = False


class SubprocessManager:
    """
    Manages trading bot subprocesses.

    Tracks PIDs in a JSON file for persistence across restarts.
    Uses database status updates for graceful shutdown.
    """

    BOT_SCRIPTS = {
        "bond": "scripts/run_bot.py",
        "flow": "scripts/run_bot.py",
        "arb": "scripts/run_arb_bot.py",
        "stat_arb": "scripts/run_stat_arb_bot.py",
        "sports": "scripts/run_bot.py",
    }

    def __init__(
        self,
        pid_file: str = "data/bot_pids.json",
        db_path: str = "data/risk_state.db",
        project_root: Optional[Path] = None,
    ):
        self.pid_file = Path(pid_file)
        self.storage = SQLiteStorage(db_path)
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.processes: Dict[str, BotProcess] = {}
        self._load_pids()

    def _load_pids(self) -> None:
        """Load PID file from disk."""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, "r") as f:
                    data = json.load(f)
                    for agent_id, info in data.items():
                        self.processes[agent_id] = BotProcess(**info)
                logger.info(f"Loaded {len(self.processes)} bot PIDs from {self.pid_file}")
            except Exception as e:
                logger.warning(f"Failed to load PIDs: {e}")
                self.processes = {}

    def _save_pids(self) -> None:
        """Save PID file to disk."""
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            agent_id: asdict(proc)
            for agent_id, proc in self.processes.items()
        }
        with open(self.pid_file, "w") as f:
            json.dump(data, f, indent=2)

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def start_bot(
        self,
        bot_type: str,
        agent_id: Optional[str] = None,
        dry_run: bool = False,
        extra_args: Optional[List[str]] = None,
    ) -> Optional[BotProcess]:
        """
        Start a trading bot subprocess.

        Args:
            bot_type: Type of bot (bond, flow, arb, stat_arb)
            agent_id: Optional agent ID (default: {bot_type}-bot)
            dry_run: Run in dry-run mode
            extra_args: Additional command-line arguments

        Returns:
            BotProcess info if started successfully, None otherwise
        """
        if bot_type not in self.BOT_SCRIPTS:
            logger.error(f"Unknown bot type: {bot_type}")
            return None

        agent_id = agent_id or f"{bot_type}-bot"

        # Check if already running
        if agent_id in self.processes:
            proc = self.processes[agent_id]
            if self._is_process_running(proc.pid):
                logger.warning(f"Bot {agent_id} already running (PID {proc.pid})")
                return None
            else:
                # Stale entry, remove it
                del self.processes[agent_id]

        # Build command
        script_path = self.project_root / self.BOT_SCRIPTS[bot_type]

        if bot_type in ("bond", "flow", "sports"):
            cmd = [
                sys.executable,
                str(script_path),
                bot_type,
                "--agent-id", agent_id,
            ]
            if dry_run:
                cmd.append("--dry-run")
        else:
            cmd = [
                sys.executable,
                str(script_path),
                "--agent-id", agent_id,
            ]
            if dry_run:
                cmd.append("--dry-run")

        if extra_args:
            cmd.extend(extra_args)

        # Start subprocess
        try:
            # Redirect stdout/stderr to log files
            log_dir = self.project_root / "logs"
            log_dir.mkdir(exist_ok=True)

            stdout_file = open(log_dir / f"{agent_id}.stdout.log", "a")
            stderr_file = open(log_dir / f"{agent_id}.stderr.log", "a")

            process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,  # Detach from parent
            )

            bot_proc = BotProcess(
                agent_id=agent_id,
                bot_type=bot_type,
                pid=process.pid,
                started_at=datetime.now(timezone.utc).isoformat(),
                command=cmd,
                dry_run=dry_run,
            )

            self.processes[agent_id] = bot_proc
            self._save_pids()

            logger.info(f"Started {bot_type} bot: {agent_id} (PID {process.pid})")
            return bot_proc

        except Exception as e:
            logger.error(f"Failed to start bot {agent_id}: {e}")
            return None

    async def stop_bot(
        self,
        agent_id: str,
        force: bool = False,
        timeout: float = 15.0,
    ) -> bool:
        """
        Stop a trading bot.

        First tries graceful shutdown via database status update.
        Falls back to SIGTERM if graceful shutdown times out.

        Args:
            agent_id: Agent ID to stop
            force: Skip graceful shutdown, send SIGTERM immediately
            timeout: Seconds to wait for graceful shutdown

        Returns:
            True if bot was stopped successfully
        """
        proc = self.processes.get(agent_id)

        if not force:
            # Try graceful shutdown via database
            try:
                with self.storage.transaction() as txn:
                    txn.update_agent_status(agent_id, AgentStatus.STOPPED)
                logger.info(f"Set {agent_id} status to STOPPED in database")
            except Exception as e:
                logger.warning(f"Failed to update database status: {e}")

        if proc and self._is_process_running(proc.pid):
            if not force:
                # Wait for graceful shutdown
                logger.info(f"Waiting up to {timeout}s for {agent_id} to stop...")
                start = asyncio.get_event_loop().time()
                while asyncio.get_event_loop().time() - start < timeout:
                    if not self._is_process_running(proc.pid):
                        break
                    await asyncio.sleep(1)

            # If still running, send SIGTERM
            if self._is_process_running(proc.pid):
                logger.info(f"Sending SIGTERM to {agent_id} (PID {proc.pid})")
                try:
                    os.kill(proc.pid, 15)  # SIGTERM
                    # Wait a bit more
                    await asyncio.sleep(2)
                    if self._is_process_running(proc.pid):
                        logger.warning(f"Process still running, sending SIGKILL")
                        os.kill(proc.pid, 9)  # SIGKILL
                except OSError as e:
                    logger.error(f"Failed to kill process: {e}")

        # Clean up
        if agent_id in self.processes:
            del self.processes[agent_id]
            self._save_pids()

        logger.info(f"Bot {agent_id} stopped")
        return True

    def get_running_bots(self) -> List[BotProcess]:
        """Get list of running bot processes."""
        running = []
        stale = []

        for agent_id, proc in self.processes.items():
            if self._is_process_running(proc.pid):
                running.append(proc)
            else:
                stale.append(agent_id)

        # Clean up stale entries
        for agent_id in stale:
            del self.processes[agent_id]

        if stale:
            self._save_pids()

        return running

    def get_all_agents(self) -> List[dict]:
        """
        Get all agents from database with their status.

        Returns combined info from database and subprocess tracking.
        """
        agents = []

        with self.storage.transaction() as txn:
            db_agents = txn.get_all_agents()

        for agent in db_agents:
            proc = self.processes.get(agent.agent_id)
            is_running = proc and self._is_process_running(proc.pid) if proc else False

            agents.append({
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "status": agent.status.value,
                "started_at": agent.started_at.isoformat() if agent.started_at else None,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                "pid": proc.pid if proc else None,
                "is_running": is_running,
            })

        return agents

    def cleanup_stale(self) -> int:
        """Clean up stale agent entries in database."""
        with self.storage.transaction() as txn:
            return txn.cleanup_stale_agents(stale_threshold_seconds=300)
