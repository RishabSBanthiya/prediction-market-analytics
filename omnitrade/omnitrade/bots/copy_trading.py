"""
Copy trading bot.

Tracks target accounts on Polymarket/Kalshi, detects new trades,
and copies them with configurable size scaling and price adjustments.

Loop: poll target positions -> diff against snapshot -> size & adjust -> execute
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiohttp

from ..core.enums import Side
from ..core.errors import RiskLimitError, OmniTradeError
from ..core.models import OrderRequest, ExchangePosition
from ..exchanges.base import ExchangeClient
from ..risk.coordinator import RiskCoordinator

logger = logging.getLogger(__name__)


def _short_addr(address: str) -> str:
    """Shorten address for logging. Only truncate long hex addresses."""
    if len(address) > 20 and address.startswith("0x"):
        return f"{address[:6]}...{address[-4:]}"
    return address


def _target_label(target: 'TargetAccount') -> str:
    """Get display label for a target account."""
    return target.label or _short_addr(target.address)


# ==================== TARGET POSITION TRACKING ====================


@dataclass
class TargetPosition:
    """A position held by a target account we're tracking."""
    instrument_id: str
    side: Side
    size: float
    price: float  # Approximate entry/avg cost (normalized 0-1)
    market_name: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class TargetAccount:
    """An account/address to copy trade from."""
    address: str  # Wallet address (Polymarket) or user ID (Kalshi)
    label: str = ""  # Human-readable label
    weight: float = 1.0  # Relative weighting for this trader (0-1)


@dataclass
class PositionDelta:
    """A detected change in a target's positions."""
    target: TargetAccount
    instrument_id: str
    side: Side
    size_delta: float  # Positive = new/increased, negative = reduced/closed
    current_price: float
    market_name: str = ""

    @property
    def is_new_position(self) -> bool:
        return self.size_delta > 0

    @property
    def is_close(self) -> bool:
        return self.size_delta < 0


class TargetTracker:
    """
    Polls target Polymarket accounts and detects position changes.

    Maintains a snapshot of each target's positions and diffs
    against fresh data each cycle to detect new/changed/closed positions.

    Only Polymarket is supported — it exposes positions via a public API
    (data-api.polymarket.com). Kalshi has no public portfolio API.
    """

    def __init__(self):
        # {address: {instrument_id: TargetPosition}}
        self._snapshots: dict[str, dict[str, TargetPosition]] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def poll_and_diff(self, target: TargetAccount) -> list[PositionDelta]:
        """Fetch current positions for a target and return deltas since last poll."""
        current = await self._fetch_positions(target.address)

        prev = self._snapshots.get(target.address, {})
        deltas: list[PositionDelta] = []

        # Detect new and increased positions
        for iid, pos in current.items():
            old = prev.get(iid)
            if old is None:
                # Entirely new position
                deltas.append(PositionDelta(
                    target=target,
                    instrument_id=iid,
                    side=pos.side,
                    size_delta=pos.size,
                    current_price=pos.price,
                    market_name=pos.market_name,
                ))
            elif pos.size != old.size or pos.side != old.side:
                # Position changed
                if pos.side == old.side:
                    delta = pos.size - old.size
                else:
                    # Side flipped — treat as close + new
                    deltas.append(PositionDelta(
                        target=target,
                        instrument_id=iid,
                        side=old.side,
                        size_delta=-old.size,
                        current_price=pos.price,
                        market_name=pos.market_name,
                    ))
                    delta = pos.size

                if abs(delta) > 0.001:
                    deltas.append(PositionDelta(
                        target=target,
                        instrument_id=iid,
                        side=pos.side,
                        size_delta=delta,
                        current_price=pos.price,
                        market_name=pos.market_name,
                    ))

        # Detect closed positions
        for iid, old_pos in prev.items():
            if iid not in current:
                deltas.append(PositionDelta(
                    target=target,
                    instrument_id=iid,
                    side=old_pos.side,
                    size_delta=-old_pos.size,
                    current_price=old_pos.price,
                    market_name=old_pos.market_name,
                ))

        # Update snapshot
        self._snapshots[target.address] = current
        return deltas

    async def _fetch_positions(self, address: str) -> dict[str, TargetPosition]:
        """Fetch positions for a target address from Polymarket's public API."""
        if self._session is None:
            return {}

        positions: dict[str, TargetPosition] = {}
        try:
            url = "https://data-api.polymarket.com/positions"
            async with self._session.get(url, params={"user": address}) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Polymarket positions API returned %d for %s",
                        resp.status, _short_addr(address),
                    )
                    return {}
                data = await resp.json()

            for p in (data if isinstance(data, list) else []):
                asset = p.get("asset", "")
                if not asset:
                    continue
                size = float(p.get("size", 0))
                if abs(size) < 0.001:
                    continue
                avg_price = float(p.get("avgPrice", 0) or p.get("avg_price", 0))
                cur_price = float(p.get("curPrice", 0) or p.get("cur_price", avg_price))
                title = p.get("title", "") or p.get("market", "")
                outcome = p.get("outcome", "")
                side = Side.BUY if size > 0 else Side.SELL

                positions[asset] = TargetPosition(
                    instrument_id=asset,
                    side=side,
                    size=abs(size),
                    price=cur_price if cur_price > 0 else avg_price,
                    market_name=f"{title} ({outcome})" if outcome else title,
                    raw=p,
                )
        except Exception as e:
            logger.error("Failed to fetch Polymarket positions for %s: %s", _short_addr(address), e)

        return positions

    async def validate_target(self, target: TargetAccount) -> tuple[bool, str, int]:
        """
        Validate that a target account is reachable and returns data.

        Returns (is_valid, message, position_count).
        """
        label = _target_label(target)
        try:
            positions = await self._fetch_positions(target.address)
            if positions is None:
                return False, f"{label}: API returned null (invalid address?)", 0
            return True, f"{label}: OK, {len(positions)} open positions", len(positions)
        except Exception as e:
            return False, f"{label}: failed to fetch — {e}", 0

    @property
    def is_first_poll(self) -> bool:
        """True if we haven't polled any targets yet."""
        return len(self._snapshots) == 0


# ==================== COPY TRADING BOT ====================


@dataclass
class CopyConfig:
    """Configuration for copy trading behavior."""
    # Size scaling: our_size = target_size * size_multiplier, capped by max/min
    size_multiplier: float = 1.0
    min_trade_usd: float = 5.0
    max_trade_usd: float = 500.0

    # Price tolerance: max deviation from current mid before skipping
    max_price_deviation_pct: float = 0.05

    # Only copy positions within these price bounds (normalized 0-1)
    min_price: float = 0.05
    max_price: float = 0.95

    # If True, also copy position closes (sell when target sells)
    copy_exits: bool = True

    # Minimum position size change (in shares) to trigger a copy
    min_delta_size: float = 1.0

    # Cooldown: seconds to wait after copying before copying same instrument again
    cooldown_seconds: float = 60.0


class CopyTradingBot:
    """
    Copy trading bot.

    Polls target accounts' positions, detects changes, and mirrors them
    on our own account with configurable size and price adjustments.

    The bot never imports platform-specific code — it uses ExchangeClient
    for trading and TargetTracker for polling public APIs.
    """

    def __init__(
        self,
        agent_id: str,
        client: ExchangeClient,
        tracker: TargetTracker,
        targets: list[TargetAccount],
        risk: RiskCoordinator,
        config: Optional[CopyConfig] = None,
        data_dir: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.client = client
        self.tracker = tracker
        self.targets = targets
        self.risk = risk
        self.config = config or CopyConfig()

        self._running = False
        self._iteration_count = 0

        # Resolve sidecar file path for persisting cooldowns across restarts
        base = Path(data_dir) if data_dir else Path(".")
        self._cooldown_path: Path = base / f".copy_cooldowns_{agent_id}.json"

        # Track what we've copied: {instrument_id: last_copy_timestamp}
        self._copy_cooldowns: dict[str, float] = {}
        # Track our own mirrored positions: {instrument_id: (side, size)}
        self._mirrored: dict[str, tuple[Side, float]] = {}

    def _load_cooldowns(self) -> None:
        """Load persisted cooldowns from the JSON sidecar file.

        Discards any entries that have already expired so the file
        doesn't grow without bound.  Silently ignores missing or
        corrupt files so a fresh start always works.
        """
        if not self._cooldown_path.exists():
            return
        try:
            raw = json.loads(self._cooldown_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                logger.warning("Cooldown file %s has unexpected format, ignoring", self._cooldown_path)
                return
            now = time.time()
            loaded = 0
            for instrument_id, ts in raw.items():
                if isinstance(ts, (int, float)) and now - ts < self.config.cooldown_seconds:
                    self._copy_cooldowns[instrument_id] = float(ts)
                    loaded += 1
            if loaded:
                logger.info("Loaded %d active cooldown(s) from %s", loaded, self._cooldown_path)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load cooldown file %s: %s", self._cooldown_path, e)

    def _save_cooldowns(self) -> None:
        """Persist current cooldowns to the JSON sidecar file.

        Only writes entries that are still within the cooldown window
        to keep the file small.  Errors are logged but never raised —
        cooldown persistence is best-effort.
        """
        now = time.time()
        active = {
            iid: ts
            for iid, ts in self._copy_cooldowns.items()
            if now - ts < self.config.cooldown_seconds
        }
        try:
            self._cooldown_path.parent.mkdir(parents=True, exist_ok=True)
            self._cooldown_path.write_text(
                json.dumps(active, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Could not save cooldown file %s: %s", self._cooldown_path, e)

    async def start(self) -> None:
        """Initialize, validate targets, and register with risk coordinator."""
        if not self.client.is_connected:
            await self.client.connect()
        await self.tracker.connect()

        # Validate all targets before starting
        valid_targets = await self._validate_targets()
        if not valid_targets:
            raise OmniTradeError(
                "No valid targets found. All target accounts failed validation. "
                "Check addresses/IDs and ensure profiles are public."
            )
        self.targets = valid_targets

        # Restore cooldowns persisted before a restart (issue #14)
        self._load_cooldowns()

        self.risk.startup(self.agent_id, "copy", self.client.exchange_id)

        target_labels = ", ".join(
            _target_label(t) for t in self.targets
        )
        logger.info(
            "CopyTradingBot '%s' started on %s, tracking: %s",
            self.agent_id, self.client.exchange_id.value, target_labels,
        )

    async def _validate_targets(self) -> list[TargetAccount]:
        """
        Validate all target accounts are reachable and return data.

        Logs status for each target. Returns the list of valid targets.
        Targets that fail validation are removed with a warning.
        """
        valid: list[TargetAccount] = []
        logger.info("Validating %d target account(s)...", len(self.targets))

        for target in self.targets:
            label = _target_label(target)
            is_valid, message, pos_count = await self.tracker.validate_target(target)

            if is_valid:
                logger.info("  [OK] %s", message)
                valid.append(target)
            else:
                logger.warning("  [FAIL] %s", message)

        logger.info(
            "Target validation complete: %d/%d valid",
            len(valid), len(self.targets),
        )
        return valid

    async def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self.tracker.close()
        self.risk.shutdown(self.agent_id)
        logger.info(f"CopyTradingBot '{self.agent_id}' stopped")

    async def run(self, interval_seconds: float = 30.0) -> None:
        """Main loop."""
        self._running = True
        await self.start()

        try:
            while self._running:
                try:
                    await self._iteration()
                except OmniTradeError as e:
                    logger.error(f"Trading error: {e}")
                    self.risk.record_failure()
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    self.risk.record_failure()

                self.risk.heartbeat(self.agent_id)
                await asyncio.sleep(interval_seconds)
        finally:
            await self.stop()

    async def _iteration(self) -> None:
        """Single copy trading iteration."""
        self._iteration_count += 1
        is_verbose = self._iteration_count % 10 == 1
        is_first = self.tracker.is_first_poll

        # Update balance for drawdown tracking
        balance = await self.client.get_balance()
        self.risk.update_equity(balance.total_equity)

        if is_verbose:
            logger.info(
                "[iter %d] balance=$%.2f, tracking %d targets",
                self._iteration_count, balance.total_equity, len(self.targets),
            )

        # Poll all targets and collect deltas
        all_deltas: list[PositionDelta] = []
        for target in self.targets:
            try:
                deltas = await self.tracker.poll_and_diff(target)
                all_deltas.extend(deltas)
            except Exception as e:
                logger.warning(
                    "Failed to poll target %s: %s",
                    _target_label(target), e,
                )

        # On first poll, just snapshot — don't copy existing positions
        if is_first:
            total_positions = sum(
                len(self.tracker._snapshots.get(t.address, {}))
                for t in self.targets
            )
            logger.info(
                "[iter %d] initial snapshot: %d positions across %d targets (not copying existing)",
                self._iteration_count, total_positions, len(self.targets),
            )
            return

        if not all_deltas:
            if is_verbose:
                logger.info("[iter %d] no position changes detected", self._iteration_count)
            return

        logger.info(
            "[iter %d] detected %d position changes",
            self._iteration_count, len(all_deltas),
        )

        # Process each delta
        for delta in all_deltas:
            await self._process_delta(delta, balance.available_balance)

    async def _process_delta(self, delta: PositionDelta, available: float) -> None:
        """Process a single position change from a target."""
        label = _target_label(delta.target)
        action = "NEW" if delta.is_new_position else ("CLOSE" if delta.is_close else "CHANGE")

        logger.info(
            "DETECTED [%s] %s %s %.2f shares of %s @ %.4f (%s)",
            label, action, delta.side.value, abs(delta.size_delta),
            delta.instrument_id[:30], delta.current_price,
            delta.market_name[:40],
        )

        # Skip closes if not configured to copy them
        if delta.is_close and not self.config.copy_exits:
            logger.info("SKIP close (copy_exits=False)")
            return

        # Check cooldown
        now = time.time()
        last_copy = self._copy_cooldowns.get(delta.instrument_id, 0)
        if now - last_copy < self.config.cooldown_seconds:
            logger.info(
                "SKIP %s: cooldown (%.0fs remaining)",
                delta.instrument_id[:20],
                self.config.cooldown_seconds - (now - last_copy),
            )
            return

        # Size delta too small?
        if abs(delta.size_delta) < self.config.min_delta_size:
            logger.info(
                "SKIP %s: delta %.2f < min %.2f",
                delta.instrument_id[:20], abs(delta.size_delta), self.config.min_delta_size,
            )
            return

        # Get current market price
        mid = await self.client.get_midpoint(delta.instrument_id)
        if mid is None:
            logger.warning("SKIP %s: no market data", delta.instrument_id[:20])
            return

        # Price bounds check
        if mid < self.config.min_price or mid > self.config.max_price:
            logger.info(
                "SKIP %s: price %.4f outside [%.2f, %.2f]",
                delta.instrument_id[:20], mid, self.config.min_price, self.config.max_price,
            )
            return

        # Price deviation check — is the current price too far from where the target traded?
        if delta.current_price > 0:
            deviation = abs(mid - delta.current_price) / delta.current_price
            if deviation > self.config.max_price_deviation_pct:
                logger.info(
                    "SKIP %s: price moved too much (mid=%.4f vs target=%.4f, dev=%.1f%%)",
                    delta.instrument_id[:20], mid, delta.current_price,
                    deviation * 100,
                )
                return

        if delta.is_close:
            await self._execute_close(delta, mid)
        else:
            await self._execute_open(delta, mid, available)

    async def _execute_open(
        self, delta: PositionDelta, mid: float, available: float,
    ) -> None:
        """Copy a new/increased position."""
        # Calculate our trade size
        target_value = abs(delta.size_delta) * mid
        our_value = target_value * delta.target.weight * self.config.size_multiplier
        our_value = max(self.config.min_trade_usd, min(self.config.max_trade_usd, our_value))
        our_value = min(our_value, available)

        if our_value < self.config.min_trade_usd:
            logger.info(
                "SKIP %s: sized to $%.2f < min $%.2f",
                delta.instrument_id[:20], our_value, self.config.min_trade_usd,
            )
            return

        shares = our_value / mid if mid > 0 else 0
        if shares <= 0:
            return

        label = _target_label(delta.target)
        logger.info(
            "COPY [%s] %s %.2f shares @ %.4f ($%.2f) of %s",
            label, delta.side.value, shares, mid, our_value,
            delta.instrument_id[:30],
        )

        # Reserve capital
        try:
            reservation_id = self.risk.atomic_reserve(
                agent_id=self.agent_id,
                exchange=self.client.exchange_id,
                instrument_id=delta.instrument_id,
                amount_usd=our_value,
            )
        except (RiskLimitError, Exception) as e:
            logger.info("BLOCKED %s: risk -> %s", delta.instrument_id[:20], e)
            return

        # Execute
        try:
            result = await self.client.place_order(OrderRequest(
                instrument_id=delta.instrument_id,
                side=delta.side,
                size=shares,
                price=mid,
            ))

            if result.success and result.filled_size > 0:
                self.risk.confirm_execution(
                    reservation_id=reservation_id,
                    agent_id=self.agent_id,
                    exchange=self.client.exchange_id,
                    instrument_id=delta.instrument_id,
                    side=delta.side.value,
                    size=result.filled_size,
                    price=result.filled_price,
                    order_id=result.order_id,
                    fees=result.fees,
                )

                self._mirrored[delta.instrument_id] = (delta.side, result.filled_size)
                self._copy_cooldowns[delta.instrument_id] = time.time()
                self._save_cooldowns()

                logger.info(
                    "COPIED %s %.4f @ $%.4f (order=%s)",
                    delta.side.value, result.filled_size,
                    result.filled_price, result.order_id,
                )
            else:
                self.risk.release_reservation(reservation_id)
                if not result.is_rejection:
                    self.risk.record_failure()
                logger.info(
                    "FAILED copy %s: %s",
                    delta.instrument_id[:20], result.error_message or "no fill",
                )
        except Exception:
            self.risk.release_reservation(reservation_id)
            self.risk.record_failure()
            raise

    async def _execute_close(self, delta: PositionDelta, mid: float) -> None:
        """Copy a position close/reduction."""
        mirrored = self._mirrored.get(delta.instrument_id)
        if mirrored is None:
            # Check if we have a DB position
            positions = self.risk.storage.get_agent_positions(self.agent_id, "open")
            for pos in positions:
                if pos["instrument_id"] == delta.instrument_id:
                    mirrored = (
                        Side.BUY if pos["side"] == "BUY" else Side.SELL,
                        pos["size"],
                    )
                    break

        if mirrored is None:
            logger.info(
                "SKIP close %s: we have no mirrored position",
                delta.instrument_id[:20],
            )
            return

        exit_side = Side.SELL if mirrored[0] == Side.BUY else Side.BUY
        exit_size = mirrored[1]  # Close full position

        label = _target_label(delta.target)
        logger.info(
            "COPY EXIT [%s] %s %.2f shares @ %.4f of %s",
            label, exit_side.value, exit_size, mid,
            delta.instrument_id[:30],
        )

        result = await self.client.place_order(OrderRequest(
            instrument_id=delta.instrument_id,
            side=exit_side,
            size=exit_size,
            price=mid,
        ))

        if result.success:
            # Close position in storage
            positions = self.risk.storage.get_agent_positions(self.agent_id, "open")
            for pos in positions:
                if pos["instrument_id"] == delta.instrument_id:
                    self.risk.storage.close_position(
                        pos["position_id"], result.filled_price, "copy_exit"
                    )
                    break

            self._mirrored.pop(delta.instrument_id, None)
            self._copy_cooldowns[delta.instrument_id] = time.time()
            self._save_cooldowns()

            logger.info(
                "CLOSED COPY %s @ %.4f (order=%s)",
                delta.instrument_id[:20], result.filled_price, result.order_id,
            )
        else:
            logger.warning(
                "FAILED close %s: %s",
                delta.instrument_id[:20], result.error_message or "no fill",
            )
