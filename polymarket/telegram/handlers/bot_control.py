"""
Bot control command handlers for Telegram.

Commands:
    /status - Show wallet and bot status
    /bots - List all bots with their status
    /start_bot <type> [--dry-run] - Start a trading bot
    /stop_bot <agent_id> [--force] - Stop a trading bot
    /help - Show available commands
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

if TYPE_CHECKING:
    from ..bot import TelegramControlBot

logger = logging.getLogger(__name__)


def truncate_error(error: Exception, max_length: int = 200) -> str:
    """Truncate error message for Telegram display."""
    error_str = str(error)
    if "<html" in error_str.lower() or "<!doctype" in error_str.lower():
        error_str = "API error (blocked by Cloudflare or server error)"
    if len(error_str) > max_length:
        error_str = error_str[:max_length] + "..."
    return error_str


def is_authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check if the update is from an authorized chat."""
    bot: "TelegramControlBot" = context.bot_data.get("control_bot")
    if not bot:
        return False
    chat_id = str(update.effective_chat.id)
    # Allow if chat_id is "0" (setup mode) or matches authorized chat
    return bot.chat_id == "0" or chat_id == bot.chat_id


async def check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Check authorization and send error message if not authorized."""
    if not is_authorized(update, context):
        logger.warning(f"Unauthorized access attempt from chat {update.effective_chat.id}")
        await update.message.reply_text("Unauthorized. Use /id to get your chat ID.")
        return False
    return True


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help message with available commands."""
    if not await check_auth(update, context):
        return

    help_text = """
<b>Polymarket Control Bot</b>

<b>Bot Control:</b>
/status - Wallet summary and bot statuses
/bots - List all registered bots
/start_bot &lt;type&gt; - Start a bot (bond/flow/arb/stat_arb/sports)
/stop_bot &lt;agent_id&gt; - Stop a running bot

<b>Trading:</b>
/search &lt;query&gt; - Search markets by name
/positions - Show open positions
/cancel - Cancel pending trade

<b>Options:</b>
/start_bot bond --dry-run
/stop_bot bond-bot --force

<b>Examples:</b>
<code>/start_bot bond</code>
<code>/start_bot flow --dry-run</code>
<code>/stop_bot flow-bot</code>
<code>/search manchester united</code>
"""
    await update.message.reply_text(help_text, parse_mode="HTML")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show wallet and bot status summary."""
    if not await check_auth(update, context):
        return

    bot: "TelegramControlBot" = context.bot_data["control_bot"]

    try:
        # Get wallet state
        wallet_state = await bot.get_wallet_state()

        # Get running bots
        running_bots = bot.subprocess_manager.get_running_bots()
        agents = bot.subprocess_manager.get_all_agents()

        # Format message
        lines = ["<b>Wallet Status</b>"]
        lines.append(f"USDC Balance: ${wallet_state.get('usdc_balance', 0):.2f}")
        lines.append(f"Positions Value: ${wallet_state.get('positions_value', 0):.2f}")
        lines.append(f"Total Equity: ${wallet_state.get('total_equity', 0):.2f}")
        lines.append(f"Reserved: ${wallet_state.get('reserved', 0):.2f}")
        lines.append(f"Available: ${wallet_state.get('available', 0):.2f}")

        lines.append("")
        lines.append(f"<b>Bots ({len(running_bots)} running)</b>")

        if agents:
            for agent in agents:
                status_emoji = "🟢" if agent["is_running"] else "🔴"
                status = agent["status"]
                lines.append(
                    f"{status_emoji} {agent['agent_id']} ({agent['agent_type']}) - {status}"
                )
        else:
            lines.append("No bots registered")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        await update.message.reply_text(f"Error: {truncate_error(e)}")


async def cmd_bots(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all registered bots with detailed status."""
    if not await check_auth(update, context):
        return

    bot: "TelegramControlBot" = context.bot_data["control_bot"]

    try:
        agents = bot.subprocess_manager.get_all_agents()

        if not agents:
            await update.message.reply_text("No bots registered.")
            return

        lines = ["<b>Registered Bots</b>", ""]

        for agent in agents:
            status_emoji = "🟢" if agent["is_running"] else "🔴"
            pid_str = f"PID {agent['pid']}" if agent["pid"] else "No PID"

            # Calculate uptime if running
            uptime_str = ""
            if agent["is_running"] and agent["started_at"]:
                try:
                    started = datetime.fromisoformat(agent["started_at"].replace("Z", "+00:00"))
                    uptime = datetime.now(timezone.utc) - started
                    hours = int(uptime.total_seconds() // 3600)
                    minutes = int((uptime.total_seconds() % 3600) // 60)
                    uptime_str = f" (up {hours}h {minutes}m)"
                except:
                    pass

            # Last heartbeat
            hb_str = ""
            if agent["last_heartbeat"]:
                try:
                    hb = datetime.fromisoformat(agent["last_heartbeat"].replace("Z", "+00:00"))
                    ago = datetime.now(timezone.utc) - hb
                    secs = int(ago.total_seconds())
                    if secs < 60:
                        hb_str = f"heartbeat {secs}s ago"
                    else:
                        hb_str = f"heartbeat {secs // 60}m ago"
                except:
                    pass

            lines.append(f"{status_emoji} <b>{agent['agent_id']}</b>")
            lines.append(f"   Type: {agent['agent_type']}")
            lines.append(f"   Status: {agent['status']}{uptime_str}")
            lines.append(f"   {pid_str}")
            if hb_str:
                lines.append(f"   {hb_str}")
            lines.append("")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    except Exception as e:
        logger.error(f"Error listing bots: {e}")
        await update.message.reply_text(f"Error: {truncate_error(e)}")


async def cmd_start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a trading bot."""
    if not await check_auth(update, context):
        return

    bot: "TelegramControlBot" = context.bot_data["control_bot"]

    if not context.args:
        await update.message.reply_text(
            "Usage: /start_bot <type> [--dry-run]\n"
            "Types: bond, flow, arb, stat_arb"
        )
        return

    bot_type = context.args[0].lower()
    dry_run = "--dry-run" in context.args

    valid_types = ["bond", "flow", "arb", "stat_arb", "sports"]
    if bot_type not in valid_types:
        await update.message.reply_text(
            f"Unknown bot type: {bot_type}\n"
            f"Valid types: {', '.join(valid_types)}"
        )
        return

    # Generate agent_id
    agent_id = f"{bot_type}-bot"

    # Check if already running
    running = bot.subprocess_manager.get_running_bots()
    for proc in running:
        if proc.agent_id == agent_id:
            await update.message.reply_text(
                f"Bot {agent_id} is already running (PID {proc.pid})"
            )
            return

    await update.message.reply_text(
        f"Starting {bot_type} bot{' (dry-run)' if dry_run else ''}..."
    )

    try:
        proc = bot.subprocess_manager.start_bot(
            bot_type=bot_type,
            agent_id=agent_id,
            dry_run=dry_run,
        )

        if proc:
            await update.message.reply_text(
                f"Started {bot_type} bot\n"
                f"Agent ID: {agent_id}\n"
                f"PID: {proc.pid}\n"
                f"Mode: {'DRY RUN' if dry_run else 'LIVE'}"
            )
        else:
            await update.message.reply_text(f"Failed to start {bot_type} bot")

    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        await update.message.reply_text(f"Error: {truncate_error(e)}")


async def cmd_stop_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop a trading bot."""
    if not await check_auth(update, context):
        return

    bot: "TelegramControlBot" = context.bot_data["control_bot"]

    if not context.args:
        # List running bots
        running = bot.subprocess_manager.get_running_bots()
        if running:
            lines = ["Usage: /stop_bot <agent_id> [--force]", "", "Running bots:"]
            for proc in running:
                lines.append(f"  - {proc.agent_id}")
            await update.message.reply_text("\n".join(lines))
        else:
            await update.message.reply_text(
                "Usage: /stop_bot <agent_id> [--force]\n"
                "No bots currently running."
            )
        return

    agent_id = context.args[0]
    force = "--force" in context.args

    await update.message.reply_text(
        f"Stopping {agent_id}{'(force)' if force else ''}..."
    )

    try:
        success = await bot.subprocess_manager.stop_bot(agent_id, force=force)

        if success:
            await update.message.reply_text(f"Bot {agent_id} stopped.")
        else:
            await update.message.reply_text(f"Failed to stop {agent_id}")

    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        await update.message.reply_text(f"Error: {truncate_error(e)}")


def register_bot_control_handlers(app) -> None:
    """Register bot control command handlers."""
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("start", cmd_help))  # Default start command
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("bots", cmd_bots))
    app.add_handler(CommandHandler("start_bot", cmd_start_bot))
    app.add_handler(CommandHandler("stop_bot", cmd_stop_bot))
