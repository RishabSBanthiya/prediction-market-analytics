"""
Chain Sync Service - Synchronizes on-chain transactions to SQL database.

This module fetches all on-chain activity (buys, sells, claims, deposits, withdrawals)
from Polygon and stores them in the transactions table, which is the source of truth.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from ..core.config import Config, get_config
from ..core.api import PolymarketAPI
from .storage.base import StorageBackend
from .storage.sqlite import SQLiteStorage

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of a chain sync operation"""
    success: bool
    transactions_synced: int
    from_block: int
    to_block: int
    errors: List[str]
    duration_seconds: float


class ChainSyncService:
    """
    Service for synchronizing on-chain transactions to SQL database.
    
    This is the core component that ensures the SQL database reflects
    the true on-chain state. It fetches:
    - CTF token transfers (buys, sells, claims)
    - USDC transfers (deposits, withdrawals)
    
    Usage:
        sync_service = ChainSyncService(config)
        result = await sync_service.full_sync(wallet_address)
        # or for incremental sync:
        result = await sync_service.incremental_sync(wallet_address)
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        storage: Optional[StorageBackend] = None,
        api: Optional[PolymarketAPI] = None
    ):
        self.config = config or get_config()
        self.storage = storage or SQLiteStorage(self.config.db_path)
        self.api = api
        self._block_timestamp_cache: Dict[int, datetime] = {}
    
    async def _ensure_api(self):
        """Ensure API is initialized"""
        if self.api is None:
            self.api = PolymarketAPI(self.config)
            await self.api.connect()
    
    async def close(self):
        """Close API connection"""
        if self.api:
            await self.api.close()

    async def _fetch_all_trades_paginated(
        self,
        wallet_address: str,
        max_pages: int = 20
    ) -> List[dict]:
        """
        Fetch all trades with pagination.

        Continues fetching until no more trades are returned or max_pages reached.
        This ensures we don't miss any trades due to the 500 limit.

        Args:
            wallet_address: Wallet to fetch trades for
            max_pages: Maximum number of pages to fetch (safety limit)

        Returns:
            List of all trade dicts
        """
        all_trades = []
        page = 0
        page_size = 500

        while page < max_pages:
            try:
                # Fetch a page of trades
                trades = await self.api.fetch_user_trades(
                    wallet_address,
                    limit=page_size,
                    offset=page * page_size
                )

                if not trades:
                    # No more trades
                    break

                all_trades.extend(trades)
                logger.debug(f"Fetched page {page + 1}: {len(trades)} trades")

                if len(trades) < page_size:
                    # Last page (not full)
                    break

                page += 1

            except Exception as e:
                logger.warning(f"Error fetching trades page {page}: {e}")
                break

        logger.info(f"Fetched {len(all_trades)} total trades across {page + 1} pages")
        return all_trades

    async def _fetch_all_activity_paginated(
        self,
        wallet_address: str,
        max_pages: int = 10
    ) -> List[dict]:
        """
        Fetch all activity (claims, deposits, withdrawals) with pagination.

        Args:
            wallet_address: Wallet to fetch activity for
            max_pages: Maximum number of pages to fetch

        Returns:
            List of all activity dicts
        """
        all_activity = []
        page = 0
        page_size = 500

        while page < max_pages:
            try:
                activity = await self.api.fetch_activity(
                    limit=page_size,
                    user=wallet_address,
                    offset=page * page_size
                )

                if not activity:
                    break

                all_activity.extend(activity)
                logger.debug(f"Fetched activity page {page + 1}: {len(activity)} items")

                if len(activity) < page_size:
                    break

                page += 1

            except Exception as e:
                logger.debug(f"Activity fetch page {page} failed (non-critical): {e}")
                break

        return all_activity

    async def full_sync(
        self,
        wallet_address: str,
        from_block: Optional[int] = None,
        to_block: Optional[int] = None,
        match_existing_executions: bool = True
    ) -> SyncResult:
        """
        Perform a full sync of all on-chain transactions for a wallet.
        
        This fetches ALL historical transactions from the blockchain
        and stores them in the transactions table.
        
        Args:
            wallet_address: The wallet address to sync
            from_block: Starting block (defaults to config or 0)
            to_block: Ending block (defaults to current block)
            match_existing_executions: Whether to match with existing executions for agent attribution
        
        Returns:
            SyncResult with details of the sync operation
        """
        start_time = datetime.now(timezone.utc)
        errors = []
        total_synced = 0
        
        await self._ensure_api()
        
        # Determine block range
        if from_block is None:
            from_block = self.config.chain_sync.initial_sync_block
        
        if to_block is None:
            to_block = await self.api.get_current_block()
            if to_block == 0:
                errors.append("Failed to get current block number")
                return SyncResult(
                    success=False,
                    transactions_synced=0,
                    from_block=from_block,
                    to_block=0,
                    errors=errors,
                    duration_seconds=(datetime.now(timezone.utc) - start_time).total_seconds()
                )
        
        logger.info(f"Starting full chain sync for {wallet_address}")
        logger.info(f"Block range: {from_block:,} to {to_block:,} ({to_block - from_block:,} blocks)")
        
        # Fetch existing executions for agent attribution matching
        existing_executions = []
        if match_existing_executions:
            with self.storage.transaction() as txn:
                existing_executions = txn.get_executions(wallet_address=wallet_address)
            logger.info(f"Found {len(existing_executions)} existing executions for matching")
        
        # Sync in batches - start with configured batch size but adapt if RPC rejects
        batch_size = self.config.chain_sync.batch_size
        current_block = from_block
        consecutive_failures = 0
        
        while current_block <= to_block:
            batch_end = min(current_block + batch_size - 1, to_block)
            
            try:
                batch_result = await self._sync_block_range(
                    wallet_address,
                    current_block,
                    batch_end,
                    existing_executions
                )
                total_synced += batch_result
                consecutive_failures = 0  # Reset on success
                
                # Progress logging
                progress = (current_block - from_block) / (to_block - from_block) * 100 if to_block > from_block else 100
                if progress > 0 and (int(progress) % 10 == 0 or total_synced > 0):
                    logger.info(f"Sync progress: {progress:.1f}% (block {current_block:,}, {total_synced} transactions)")
                
                current_block = batch_end + 1
                
            except Exception as e:
                error_str = str(e)
                consecutive_failures += 1
                
                # If RPC says range too large, reduce batch size
                if "too large" in error_str.lower() or "range" in error_str.lower():
                    old_batch = batch_size
                    batch_size = max(10, batch_size // 2)  # Halve batch size, minimum 10
                    logger.warning(f"RPC rejected range, reducing batch size from {old_batch} to {batch_size}")
                    continue  # Retry with smaller batch
                
                error_msg = f"Error syncing blocks {current_block}-{batch_end}: {e}"
                logger.error(error_msg)
                
                # Retry logic with exponential backoff
                if consecutive_failures <= self.config.chain_sync.max_retries:
                    delay = self.config.chain_sync.retry_delay_seconds * (2 ** (consecutive_failures - 1))
                    logger.info(f"Retrying in {delay:.1f}s (attempt {consecutive_failures})")
                    await asyncio.sleep(delay)
                else:
                    errors.append(error_msg)
                    # Track this failed range as a gap for later retry
                    with self.storage.transaction() as txn:
                        txn.add_sync_gap(
                            wallet_address=wallet_address.lower(),
                            from_block=current_block,
                            to_block=batch_end,
                            error=error_str[:500]  # Truncate error message
                        )
                    logger.warning(f"Recorded sync gap for blocks {current_block}-{batch_end}")
                    # Move on to next batch
                    current_block = batch_end + 1
                    consecutive_failures = 0
        
        # Update sync state (normalize wallet address to lowercase for consistency)
        with self.storage.transaction() as txn:
            total_tx_count = txn.count_transactions(wallet_address.lower())
            txn.update_chain_sync_state(wallet_address.lower(), to_block, total_tx_count)
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        logger.info(f"Full sync complete: {total_synced} transactions in {duration:.1f}s")
        
        return SyncResult(
            success=len(errors) == 0,
            transactions_synced=total_synced,
            from_block=from_block,
            to_block=to_block,
            errors=errors,
            duration_seconds=duration
        )
    
    async def incremental_sync(self, wallet_address: str) -> SyncResult:
        """
        Perform an incremental sync using the Polymarket API.

        This fetches recent trades/activity from the API which is more reliable
        than scanning RPC blocks (Polymarket trades don't emit direct events).

        Now with pagination to fetch ALL trades since last sync, not just 500.
        """
        start_time = datetime.now(timezone.utc)

        await self._ensure_api()

        wallet_lower = wallet_address.lower()
        errors = []
        total_synced = 0

        try:
            # Fetch ALL trades with pagination (not just 500)
            raw_trades = await self._fetch_all_trades_paginated(wallet_address)

            # Fetch activity with pagination
            activity = await self._fetch_all_activity_paginated(wallet_address)
            
            logger.info(f"Incremental sync: {len(raw_trades)} trades, {len(activity)} activity items")
            
            with self.storage.transaction() as txn:
                # Get existing transaction hashes to avoid duplicates
                existing_txs = txn.get_transactions(wallet_lower, limit=10000)
                existing_hashes = set()
                for tx in existing_txs:
                    # Store both tx_hash and a fingerprint (token_id + shares + timestamp_truncated)
                    existing_hashes.add(tx.get("tx_hash", ""))
                    if tx.get("token_id") and tx.get("shares"):
                        ts = tx.get("block_timestamp")
                        ts_key = ts.strftime("%Y%m%d%H%M") if isinstance(ts, datetime) else str(ts)[:12]
                        fingerprint = f"{tx['token_id']}_{tx['shares']:.4f}_{ts_key}"
                        existing_hashes.add(fingerprint)
                
                new_trades_count = 0
                
                # Process trades (from fetch_user_trades - returns dicts)
                for trade in raw_trades:
                    try:
                        # fetch_user_trades returns normalized dicts
                        trade_id = trade.get("trade_id") or ""
                        token_id = trade.get("token_id")
                        shares = float(trade.get("shares", 0))
                        price = float(trade.get("price", 0))
                        side = trade.get("side", "").upper()
                        timestamp = trade.get("timestamp")
                        if not isinstance(timestamp, datetime):
                            timestamp = datetime.now(timezone.utc)
                        market_id = trade.get("market_id")
                        outcome = trade.get("outcome")
                        
                        # Create fingerprint to check for duplicates
                        ts_key = timestamp.strftime("%Y%m%d%H%M") if isinstance(timestamp, datetime) else str(timestamp)[:12]
                        fingerprint = f"{token_id}_{shares:.4f}_{ts_key}"
                        
                        # Skip if we've already seen this trade
                        if trade_id in existing_hashes or fingerprint in existing_hashes:
                            continue
                        
                        tx_hash = trade_id or f"trade_{timestamp.isoformat()}_{token_id}"
                        
                        if not token_id or shares <= 0:
                            continue
                        
                        tx_type = "buy" if side == "BUY" else "sell"
                        usdc_amount = shares * price
                        
                        txn.upsert_transaction(
                            tx_hash=tx_hash,
                            log_index=0,
                            block_number=0,
                            block_timestamp=timestamp,
                            transaction_type=tx_type,
                            wallet_address=wallet_lower,
                            token_id=token_id,
                            market_id=market_id,
                            outcome=outcome,
                            shares=shares,
                            price_per_share=price,
                            usdc_amount=usdc_amount,
                            agent_id=None,
                            raw_event=json.dumps({"source": "incremental_api", "trade_id": trade_id})
                        )
                        new_trades_count += 1
                        existing_hashes.add(fingerprint)  # Track this trade
                    except Exception as e:
                        logger.debug(f"Error processing trade: {e}")
                
                total_synced += new_trades_count
                if new_trades_count > 0:
                    logger.info(f"Added {new_trades_count} new trades")
                
                # Process activity for claims/deposits/withdrawals
                tx_hash_log_index = {}
                for item in activity:
                    item_type = item.get("type", "").lower()
                    
                    if item_type in ["redeem", "reward"]:
                        try:
                            shares = float(item.get("size", 0))
                            usdc_amount = float(item.get("usdcSize", 0))
                            
                            if shares <= 0 and usdc_amount <= 0:
                                continue
                            
                            market_id = item.get("conditionId")
                            tx_hash = item.get("transactionHash") or f"claim_{item.get('timestamp', '')}_{market_id}"
                            
                            # Track log_index per tx_hash
                            if tx_hash not in tx_hash_log_index:
                                tx_hash_log_index[tx_hash] = 0
                            else:
                                tx_hash_log_index[tx_hash] += 1
                            log_index = tx_hash_log_index[tx_hash]
                            
                            token_ids = item.get("assets", [])
                            if not token_ids and item.get("asset"):
                                token_ids = [item.get("asset")]
                            if not token_ids:
                                token_ids = [f"claim_market_{market_id}"]
                            
                            token_id = token_ids[0] if token_ids else None
                            outcome = item.get("outcome")
                            
                            timestamp = datetime.now(timezone.utc)
                            ts = item.get("timestamp")
                            if ts:
                                try:
                                    if isinstance(ts, (int, float)):
                                        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                                    else:
                                        timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                                except:
                                    pass
                            
                            price_per_share = usdc_amount / shares if shares > 0 else 1.0
                            
                            txn.upsert_transaction(
                                tx_hash=tx_hash,
                                log_index=log_index,
                                block_number=0,
                                block_timestamp=timestamp,
                                transaction_type="claim",
                                wallet_address=wallet_lower,
                                token_id=token_id,
                                market_id=market_id,
                                outcome=outcome,
                                shares=shares,
                                price_per_share=price_per_share,
                                usdc_amount=usdc_amount,
                                agent_id=None,
                                raw_event=json.dumps({"source": "incremental_api", "type": item_type})
                            )
                            total_synced += 1
                        except Exception as e:
                            logger.debug(f"Error processing claim: {e}")
                
                # Update sync state
                current_block = await self.api.get_current_block() or 0
                txn.update_chain_sync_state(wallet_lower, current_block, total_synced)
            
            logger.info(f"Incremental sync complete: {total_synced} transactions")
            
        except Exception as e:
            error_msg = f"Incremental sync error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return SyncResult(
            success=len(errors) == 0,
            transactions_synced=total_synced,
            from_block=0,
            to_block=0,
            errors=errors,
            duration_seconds=duration
        )
    
    async def _sync_block_range(
        self,
        wallet_address: str,
        from_block: int,
        to_block: int,
        existing_executions: List[dict]
    ) -> int:
        """Sync a specific block range. Returns number of transactions synced."""
        transactions_synced = 0
        
        # Fetch CTF transfer events (buys, sells, claims)
        ctf_events = await self.api.fetch_ctf_transfer_events(
            wallet_address,
            from_block,
            to_block
        )
        
        # Fetch USDC transfer events (deposits, withdrawals)
        usdc_events = await self.api.fetch_usdc_transfer_events(
            wallet_address,
            from_block,
            to_block
        )
        
        # Process and store events
        with self.storage.transaction() as txn:
            # Process CTF events
            for event in ctf_events:
                # Get block timestamp
                block_timestamp = await self._get_block_timestamp(event["block_number"])
                if not block_timestamp:
                    block_timestamp = datetime.now(timezone.utc)
                
                # Try to match with existing execution for agent attribution
                agent_id = self._find_matching_agent(
                    event,
                    block_timestamp,
                    existing_executions
                )
                
                # Calculate USDC amount (shares * price estimate)
                # For now, we don't have the exact price from the event
                # This will be enriched later or left as None
                usdc_amount = None
                
                txn.upsert_transaction(
                    tx_hash=event["tx_hash"],
                    log_index=event["log_index"],
                    block_number=event["block_number"],
                    block_timestamp=block_timestamp,
                    transaction_type=event["transaction_type"],
                    wallet_address=wallet_address,
                    token_id=event.get("token_id"),
                    market_id=None,  # Will be enriched later
                    outcome=None,  # Will be enriched later
                    shares=event.get("shares"),
                    price_per_share=None,  # Not available from transfer events
                    usdc_amount=usdc_amount,
                    agent_id=agent_id,
                    raw_event=json.dumps(event.get("raw_log", {}))
                )
                transactions_synced += 1
            
            # Process USDC events
            for event in usdc_events:
                block_timestamp = await self._get_block_timestamp(event["block_number"])
                if not block_timestamp:
                    block_timestamp = datetime.now(timezone.utc)
                
                txn.upsert_transaction(
                    tx_hash=event["tx_hash"],
                    log_index=event["log_index"],
                    block_number=event["block_number"],
                    block_timestamp=block_timestamp,
                    transaction_type=event["transaction_type"],
                    wallet_address=wallet_address,
                    token_id=None,
                    market_id=None,
                    outcome=None,
                    shares=None,
                    price_per_share=None,
                    usdc_amount=event.get("usdc_amount"),
                    agent_id=None,  # USDC transfers don't have agent attribution
                    raw_event=json.dumps(event.get("raw_log", {}))
                )
                transactions_synced += 1
        
        return transactions_synced
    
    async def _get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """Get block timestamp with caching"""
        if block_number in self._block_timestamp_cache:
            return self._block_timestamp_cache[block_number]
        
        timestamp = await self.api.get_block_timestamp(block_number)
        if timestamp:
            self._block_timestamp_cache[block_number] = timestamp
            
            # Limit cache size
            if len(self._block_timestamp_cache) > 10000:
                # Remove oldest entries
                oldest_blocks = sorted(self._block_timestamp_cache.keys())[:5000]
                for block in oldest_blocks:
                    del self._block_timestamp_cache[block]
        
        return timestamp
    
    def _find_matching_agent(
        self,
        event: dict,
        event_timestamp: datetime,
        existing_executions: List[dict]
    ) -> Optional[str]:
        """
        Try to match an on-chain event with an existing execution for agent attribution.
        
        Matching criteria:
        - Same token_id
        - Same side (buy/sell)
        - Timestamp within 5 minutes
        - Shares within 5% tolerance
        """
        if not existing_executions:
            return None
        
        event_type = event.get("transaction_type", "")
        event_token = event.get("token_id", "")
        event_shares = event.get("shares", 0) or 0
        
        # Map transaction type to execution side
        if event_type == "buy":
            expected_side = "BUY"
        elif event_type == "sell":
            expected_side = "SELL"
        else:
            return None  # Claims don't match to executions
        
        for execution in existing_executions:
            # Check token match
            if execution.get("token_id") != event_token:
                continue
            
            # Check side match
            if execution.get("side") != expected_side:
                continue
            
            # Check timestamp within 5 minutes
            exec_timestamp = execution.get("timestamp")
            if exec_timestamp:
                time_diff = abs((event_timestamp - exec_timestamp).total_seconds())
                if time_diff > 300:  # 5 minutes
                    continue
            
            # Check shares within 5% tolerance
            exec_shares = execution.get("shares", 0) or 0
            if exec_shares > 0 and event_shares > 0:
                shares_diff = abs(event_shares - exec_shares) / exec_shares
                if shares_diff > 0.05:  # 5% tolerance
                    continue
            
            # Match found!
            return execution.get("agent_id")
        
        return None
    
    async def enrich_transaction_metadata(
        self,
        wallet_address: str,
        token_to_market_map: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> int:
        """
        Enrich transactions with market_id and outcome information.
        
        This should be called after syncing to add metadata that isn't
        available from the on-chain events directly.
        
        Args:
            wallet_address: Wallet to enrich transactions for
            token_to_market_map: Optional dict of token_id -> (market_id, outcome)
        
        Returns:
            Number of transactions enriched
        """
        enriched = 0
        
        with self.storage.transaction() as txn:
            # Get transactions missing market_id
            transactions = txn.get_transactions(
                wallet_address=wallet_address,
                limit=10000
            )
            
            for tx in transactions:
                if tx.get("market_id") or not tx.get("token_id"):
                    continue
                
                token_id = tx["token_id"]
                
                if token_to_market_map and token_id in token_to_market_map:
                    market_id, outcome = token_to_market_map[token_id]
                    # Update transaction with market info
                    # This would require adding an update method
                    enriched += 1
        
        return enriched
    
    async def verify_sync_integrity(
        self,
        wallet_address: str
    ) -> Tuple[bool, List[str]]:
        """
        Verify that the synced transactions match on-chain state.
        
        Compares:
        - Computed positions vs Data API positions
        - Transaction counts
        
        Returns:
            (is_valid, list of discrepancies)
        """
        await self._ensure_api()
        
        discrepancies = []
        
        # Get computed positions from transactions
        with self.storage.transaction() as txn:
            computed_positions = txn.get_computed_positions(wallet_address)
        
        # Get actual positions from Data API
        actual_positions = await self.api.fetch_positions(wallet_address)
        
        # Build lookup of actual positions
        actual_by_token = {p.token_id: p for p in actual_positions}
        computed_by_token = {p["token_id"]: p for p in computed_positions}
        
        # Check each actual position exists in computed
        for token_id, actual in actual_by_token.items():
            if token_id not in computed_by_token:
                discrepancies.append(f"Position {token_id[:20]}... exists on-chain but not in computed")
                continue
            
            computed = computed_by_token[token_id]
            
            # Compare shares (with small tolerance for rounding)
            actual_shares = actual.shares
            computed_shares = computed["shares"]
            
            if abs(actual_shares - computed_shares) > 0.0001:
                discrepancies.append(
                    f"Share mismatch for {token_id[:20]}...: "
                    f"actual={actual_shares:.6f}, computed={computed_shares:.6f}"
                )
        
        # Check for ghost positions in computed (exist in computed but not on-chain)
        for token_id, computed in computed_by_token.items():
            if token_id not in actual_by_token and computed["shares"] > 0.0001:
                discrepancies.append(
                    f"Ghost position {token_id[:20]}... in computed "
                    f"({computed['shares']:.6f} shares) but not on-chain"
                )
        
        is_valid = len(discrepancies) == 0
        
        if is_valid:
            logger.info("Sync integrity verified: all positions match")
        else:
            logger.warning(f"Sync integrity issues found: {len(discrepancies)} discrepancies")
            for d in discrepancies:
                logger.warning(f"  - {d}")
        
        return is_valid, discrepancies

    async def fix_discrepancies(self, wallet_address: str) -> int:
        """
        Fix discrepancies between computed and actual positions.
        
        Creates reconciliation entries for:
        - Ghost positions (in computed but not on-chain)
        - Share mismatches (different share counts)
        - Missing positions (on-chain but not in computed)
        
        Returns:
            Number of reconciliation entries created
        """
        await self._ensure_api()
        
        wallet_lower = wallet_address.lower()
        
        # Get computed positions from transactions
        with self.storage.transaction() as txn:
            computed_positions = txn.get_computed_positions(wallet_lower)
        
        # Get actual positions from Data API
        actual_positions = await self.api.fetch_positions(wallet_address)
        
        # Build lookups
        actual_by_token = {p.token_id: p for p in actual_positions}
        computed_by_token = {p["token_id"]: p for p in computed_positions}
        
        fixes = 0
        
        with self.storage.transaction() as txn:
            # Fix 1: Ghost positions (computed but not on-chain)
            for token_id, computed in computed_by_token.items():
                computed_shares = computed.get("shares", 0)
                
                if token_id not in actual_by_token and computed_shares > 0.001:
                    # Position doesn't exist on-chain - claim at $0
                    txn.upsert_transaction(
                        tx_hash=f"reconcile_ghost_{wallet_lower}_{token_id}",
                        log_index=0,
                        block_number=0,
                        block_timestamp=datetime.now(timezone.utc),
                        transaction_type="claim",
                        wallet_address=wallet_lower,
                        token_id=token_id,
                        market_id=computed.get("market_id"),
                        outcome=computed.get("outcome"),
                        shares=computed_shares,
                        price_per_share=0.0,
                        usdc_amount=0.0,
                        agent_id=computed.get("agent_id"),
                        raw_event=json.dumps({"source": "reconciliation", "reason": "ghost_position"})
                    )
                    fixes += 1
                    logger.info(f"Fixed ghost position: {token_id[:20]}... ({computed_shares:.4f} shares -> 0)")
                    continue
                
                # Fix 2: Share mismatch (actual != computed)
                if token_id in actual_by_token:
                    actual_shares = actual_by_token[token_id].shares
                    
                    if abs(actual_shares - computed_shares) > 0.001:
                        # Need to adjust shares to match
                        diff = computed_shares - actual_shares
                        
                        if diff > 0:
                            # Computed has more shares than actual - sell/claim the difference
                            txn.upsert_transaction(
                                tx_hash=f"reconcile_excess_{wallet_lower}_{token_id}",
                                log_index=0,
                                block_number=0,
                                block_timestamp=datetime.now(timezone.utc),
                                transaction_type="sell",
                                wallet_address=wallet_lower,
                                token_id=token_id,
                                market_id=computed.get("market_id"),
                                outcome=computed.get("outcome"),
                                shares=diff,
                                price_per_share=0.0,
                                usdc_amount=0.0,
                                agent_id=computed.get("agent_id"),
                                raw_event=json.dumps({"source": "reconciliation", "reason": "excess_shares"})
                            )
                            fixes += 1
                            logger.info(f"Fixed excess shares: {token_id[:20]}... ({computed_shares:.4f} -> {actual_shares:.4f})")
                        else:
                            # Computed has fewer shares - add a buy adjustment
                            missing = abs(diff)
                            current_price = actual_by_token[token_id].current_price or 0.5
                            txn.upsert_transaction(
                                tx_hash=f"reconcile_missing_{wallet_lower}_{token_id}",
                                log_index=0,
                                block_number=0,
                                block_timestamp=datetime.now(timezone.utc),
                                transaction_type="buy",
                                wallet_address=wallet_lower,
                                token_id=token_id,
                                market_id=computed.get("market_id"),
                                outcome=computed.get("outcome"),
                                shares=missing,
                                price_per_share=current_price,
                                usdc_amount=missing * current_price,
                                agent_id=computed.get("agent_id"),
                                raw_event=json.dumps({"source": "reconciliation", "reason": "missing_shares"})
                            )
                            fixes += 1
                            logger.info(f"Fixed missing shares: {token_id[:20]}... ({computed_shares:.4f} -> {actual_shares:.4f})")
            
            # Fix 3: Missing positions (on-chain but not in computed at all)
            for token_id, actual in actual_by_token.items():
                if token_id not in computed_by_token:
                    # Position exists on-chain but not in computed - add it
                    current_price = actual.current_price or actual.entry_price or 0.5
                    txn.upsert_transaction(
                        tx_hash=f"reconcile_new_{wallet_lower}_{token_id}",
                        log_index=0,
                        block_number=0,
                        block_timestamp=datetime.now(timezone.utc),
                        transaction_type="buy",
                        wallet_address=wallet_lower,
                        token_id=token_id,
                        market_id=None,  # Unknown
                        outcome=actual.outcome,
                        shares=actual.shares,
                        price_per_share=current_price,
                        usdc_amount=actual.shares * current_price,
                        agent_id=None,  # Orphan
                        raw_event=json.dumps({"source": "reconciliation", "reason": "new_position"})
                    )
                    fixes += 1
                    logger.info(f"Added missing position: {token_id[:20]}... ({actual.shares:.4f} shares)")
        
        if fixes > 0:
            logger.info(f"Fixed {fixes} discrepancies between computed and actual positions")

        # Always refresh USDC balance from chain during sync
        try:
            actual_balance = await self.api.fetch_usdc_balance(wallet_address)
            with self.storage.transaction() as txn:
                txn.update_usdc_balance(wallet_lower, actual_balance)
            logger.debug(f"Updated USDC balance from chain: ${actual_balance:.2f}")
        except Exception as e:
            logger.warning(f"Failed to refresh USDC balance: {e}")

        return fixes


async def run_initial_sync(wallet_address: str, config: Optional[Config] = None):
    """
    Convenience function to run initial full sync for a wallet.
    
    Usage:
        from polymarket.trading.chain_sync import run_initial_sync
        result = await run_initial_sync("0x...")
    """
    sync_service = ChainSyncService(config)
    try:
        result = await sync_service.full_sync(wallet_address)
        
        if result.success:
            logger.info(f"Initial sync complete: {result.transactions_synced} transactions")
        else:
            logger.error(f"Initial sync had errors: {result.errors}")
        
        # Verify integrity
        is_valid, discrepancies = await sync_service.verify_sync_integrity(wallet_address)
        
        return result
    finally:
        await sync_service.close()


async def fast_sync_from_api(wallet_address: str, config: Optional[Config] = None) -> SyncResult:
    """
    Fast sync using Polymarket Data API instead of scanning blockchain.
    
    This is much faster than full_sync() and works with any RPC.
    It fetches positions and activity from the Data API and stores them
    in the transactions table.
    
    Usage:
        from polymarket.trading.chain_sync import fast_sync_from_api
        result = await fast_sync_from_api("0x...")
    """
    from ..core.api import PolymarketAPI
    
    start_time = datetime.now(timezone.utc)
    errors = []
    total_synced = 0
    config = config or get_config()
    storage = SQLiteStorage(config.db_path)
    api = None
    
    try:
        api = PolymarketAPI(config)
        await api.connect()
        
        logger.info(f"Fast sync from API for {wallet_address}")
        
        # Step 1: Fetch current positions
        logger.info("Fetching current positions from API...")
        positions = await api.fetch_positions(wallet_address)
        logger.info(f"Found {len(positions)} positions")
        
        # Step 2: Fetch activity/trade history
        logger.info("Fetching activity history from API...")
        activity = await api.fetch_activity(limit=1000, user=wallet_address)
        logger.info(f"Found {len(activity)} activity items")
        
        # Step 3: Fetch user trades
        logger.info("Fetching trade history from API...")
        trades = await api.fetch_user_trades(wallet_address, limit=500)
        logger.info(f"Found {len(trades)} trades")
        
        # Step 4: Store in transactions table
        with storage.transaction() as txn:
            # Process trades (may be Trade objects or dicts depending on API response)
            logger.info(f"Processing {len(trades)} trades for storage...")
            for i, trade in enumerate(trades):
                try:
                    # Handle both Trade objects and dict responses
                    if hasattr(trade, 'side'):
                        # Trade object
                        side_val = trade.side.value if hasattr(trade.side, 'value') else str(trade.side)
                        tx_type = "buy" if side_val == "BUY" else "sell"
                        token_id = trade.token_id
                        market_id = trade.market_id
                        outcome = trade.outcome
                        shares = trade.shares
                        price = trade.price
                        order_id = trade.order_id
                        timestamp = trade.timestamp
                    else:
                        # Dict from API
                        side_val = str(trade.get("side", "")).upper()
                        tx_type = "buy" if side_val == "BUY" else "sell"
                        token_id = trade.get("asset") or trade.get("tokenId") or trade.get("token_id") or trade.get("asset_id", "")
                        market_id = trade.get("market") or trade.get("marketId") or trade.get("market_id")
                        outcome = trade.get("outcome")
                        shares = float(trade.get("size", 0) or trade.get("shares", 0) or trade.get("amount", 0))
                        price = float(trade.get("price", 0) or trade.get("fillPrice", 0))
                        order_id = trade.get("id") or trade.get("orderId") or trade.get("order_id")
                        
                        # Parse timestamp
                        timestamp = datetime.now(timezone.utc)
                        ts_str = trade.get("timestamp") or trade.get("createdAt") or trade.get("matchedTime")
                        if ts_str:
                            try:
                                if isinstance(ts_str, (int, float)):
                                    timestamp = datetime.fromtimestamp(ts_str, tz=timezone.utc)
                                else:
                                    timestamp = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
                            except:
                                pass
                    
                    if not token_id or shares <= 0:
                        continue
                    
                    # Create a pseudo tx_hash from trade data if not available
                    tx_hash = order_id or f"trade_{timestamp.timestamp()}_{token_id[:20]}"
                    
                    txn.upsert_transaction(
                        tx_hash=tx_hash,
                        log_index=i,  # Use index as log_index since API doesn't provide it
                        block_number=0,  # Not available from API
                        block_timestamp=timestamp,
                        transaction_type=tx_type,
                        wallet_address=wallet_address.lower(),
                        token_id=token_id,
                        market_id=market_id,
                        outcome=outcome,
                        shares=shares,
                        price_per_share=price,
                        usdc_amount=shares * price if price else None,
                        agent_id=None,  # Will be matched later
                        raw_event=json.dumps({
                            "source": "data_api",
                            "side": side_val,
                            "order_id": order_id
                        })
                    )
                    total_synced += 1
                except Exception as e:
                    logger.warning(f"Error processing trade {i}: {e}")
            
            # Process activity for deposits/withdrawals/claims
            # Track log_index per tx_hash since multiple claims can share same tx_hash
            tx_hash_log_index = {}
            
            for item in activity:
                item_type = item.get("type", "").lower()
                
                # Handle deposits and withdrawals
                if item_type in ["deposit", "withdrawal", "withdraw"]:
                    try:
                        amount = float(item.get("amount", 0) or item.get("value", 0) or item.get("usdcSize", 0))
                        if amount <= 0:
                            continue
                        
                        tx_type = "deposit" if item_type == "deposit" else "withdrawal"
                        tx_hash = item.get("transactionHash") or item.get("id") or item.get("txHash") or f"{tx_type}_{item.get('timestamp', '')}"
                        
                        timestamp = datetime.now(timezone.utc)
                        ts_value = item.get("timestamp") or item.get("createdAt")
                        if ts_value:
                            try:
                                if isinstance(ts_value, (int, float)):
                                    timestamp = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                                else:
                                    timestamp = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
                            except:
                                pass
                        
                        txn.upsert_transaction(
                            tx_hash=tx_hash,
                            log_index=0,
                            block_number=0,
                            block_timestamp=timestamp,
                            transaction_type=tx_type,
                            wallet_address=wallet_address.lower(),
                            token_id=None,
                            market_id=None,
                            outcome=None,
                            shares=None,
                            price_per_share=None,
                            usdc_amount=amount,
                            agent_id=None,
                            raw_event=json.dumps({"source": "data_api", "type": item_type})
                        )
                        total_synced += 1
                    except Exception as e:
                        logger.debug(f"Error processing deposit/withdrawal: {e}")
                
                # Handle REDEEM and REWARD (claims from resolved markets)
                elif item_type in ["redeem", "reward"]:
                    try:
                        shares = float(item.get("size", 0))
                        usdc_amount = float(item.get("usdcSize", 0))
                        token_id = item.get("asset") or None
                        market_id = item.get("conditionId")
                        
                        if shares <= 0 and usdc_amount <= 0:
                            continue
                        
                        # If no token_id, look up from existing transactions for this market
                        if not token_id and market_id:
                            existing = txn._fetchone(
                                """SELECT token_id FROM transactions 
                                   WHERE market_id = ? AND token_id IS NOT NULL 
                                   AND transaction_type = 'buy' 
                                   LIMIT 1""",
                                (market_id,)
                            )
                            if existing:
                                token_id = existing["token_id"]
                                logger.debug(f"Found token_id {token_id[:20]}... for market {market_id[:20]}...")
                        
                        tx_hash = item.get("transactionHash") or f"claim_{item.get('timestamp', '')}_{market_id}"
                        outcome = item.get("outcome")
                        
                        # Track log_index per tx_hash (multiple claims can share same tx_hash)
                        if tx_hash not in tx_hash_log_index:
                            tx_hash_log_index[tx_hash] = 0
                        else:
                            tx_hash_log_index[tx_hash] += 1
                        log_index = tx_hash_log_index[tx_hash]
                        
                        # Calculate price per share if we have both values
                        price_per_share = usdc_amount / shares if shares > 0 else 1.0
                        
                        timestamp = datetime.now(timezone.utc)
                        ts_value = item.get("timestamp")
                        if ts_value:
                            try:
                                if isinstance(ts_value, (int, float)):
                                    timestamp = datetime.fromtimestamp(ts_value, tz=timezone.utc)
                                else:
                                    timestamp = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
                            except:
                                pass
                        
                        txn.upsert_transaction(
                            tx_hash=tx_hash,
                            log_index=log_index,
                            block_number=0,
                            block_timestamp=timestamp,
                            transaction_type="claim",
                            wallet_address=wallet_address.lower(),
                            token_id=token_id,
                            market_id=market_id,
                            outcome=outcome,
                            shares=shares,
                            price_per_share=price_per_share,
                            usdc_amount=usdc_amount,
                            agent_id=None,
                            raw_event=json.dumps({"source": "data_api", "type": item_type, "raw": item})
                        )
                        total_synced += 1
                        logger.info(f"Recorded claim: {shares:.4f} shares for ${usdc_amount:.2f} (token: {token_id[:20] if token_id else 'unknown'}...)")
                    except Exception as e:
                        logger.warning(f"Error processing claim: {e}")
            
            # Reconcile: create adjustment transactions for positions that disappeared
            # (losing bets that resolved to $0 don't have REDEEM transactions)
            logger.info("Reconciling computed positions with actual positions...")
            
            computed = txn.get_computed_positions(wallet_address.lower())
            actual_token_ids = {p.token_id for p in positions}
            
            reconciliation_count = 0
            for cp in computed:
                token_id = cp.get('token_id')
                net_shares = cp.get('shares', 0)
                
                if token_id and net_shares > 0.001 and token_id not in actual_token_ids:
                    # This position doesn't exist on-chain - create a reconciliation entry
                    # (either lost bet resolved to $0, or sold externally)
                    txn.upsert_transaction(
                        tx_hash=f"reconcile_{wallet_address.lower()}_{token_id}",
                        log_index=0,
                        block_number=0,
                        block_timestamp=datetime.now(timezone.utc),
                        transaction_type="claim",  # Treat as claimed at $0
                        wallet_address=wallet_address.lower(),
                        token_id=token_id,
                        market_id=cp.get('market_id'),
                        outcome=cp.get('outcome'),
                        shares=net_shares,
                        price_per_share=0.0,  # Lost bet = $0 value
                        usdc_amount=0.0,
                        agent_id=cp.get('agent_id'),
                        raw_event=json.dumps({"source": "reconciliation", "reason": "position_not_on_chain"})
                    )
                    reconciliation_count += 1
                    logger.info(f"Reconciled missing position: {token_id[:20]}... ({net_shares:.4f} shares, $0 value)")
            
            if reconciliation_count > 0:
                logger.info(f"Created {reconciliation_count} reconciliation entries for missing positions")
                total_synced += reconciliation_count
            
            # Update sync state
            current_block = await api.get_current_block() or 0
            txn.update_chain_sync_state(wallet_address.lower(), current_block, total_synced)
        
        logger.info(f"Fast sync complete: {total_synced} transactions from API")
        
    except Exception as e:
        error_msg = f"Fast sync error: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
    finally:
        if api:
            await api.close()
    
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    return SyncResult(
        success=len(errors) == 0,
        transactions_synced=total_synced,
        from_block=0,
        to_block=0,
        errors=errors,
        duration_seconds=duration
    )

