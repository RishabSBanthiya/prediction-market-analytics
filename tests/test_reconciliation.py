"""
Test suite for API-first wallet reconciliation.

Tests the new reconciliation architecture where:
- api_positions table is the source of truth (from Polymarket API)
- transactions table is the audit log (from chain sync)
- reconciliation_issues table tracks discrepancies
"""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from polymarket.core.models import Position, PositionStatus
from polymarket.core.config import Config
from polymarket.trading.storage.sqlite import SQLiteStorage, SQLiteTransaction
from polymarket.trading.risk_coordinator import RiskCoordinator


# ==================== FIXTURES ====================


@pytest.fixture
def temp_db_path():
    """Create a temporary database file"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    # Cleanup
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def storage(temp_db_path):
    """Create a SQLite storage instance with temp database"""
    storage = SQLiteStorage(temp_db_path)
    storage.initialize()
    return storage


@pytest.fixture
def wallet_address():
    """Sample wallet address for tests"""
    return "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def sample_api_positions():
    """Sample positions as returned from Polymarket API"""
    return [
        Position(
            id=None,
            agent_id="api",
            market_id="market_1",
            token_id="token_abc123",
            outcome="Yes",
            shares=100.5,
            entry_price=0.65,
            entry_time=None,
            current_price=0.72,
            status=PositionStatus.OPEN,
        ),
        Position(
            id=None,
            agent_id="api",
            market_id="market_2",
            token_id="token_def456",
            outcome="No",
            shares=50.0,
            entry_price=0.30,
            entry_time=None,
            current_price=0.35,
            status=PositionStatus.OPEN,
        ),
    ]


# ==================== API POSITIONS TESTS ====================


class TestAPIPositionsStorage:
    """Tests for api_positions table operations"""

    def test_upsert_api_positions_empty(self, storage, wallet_address):
        """Test upserting empty positions list clears existing"""
        with storage.transaction() as txn:
            # First, add some positions
            txn._execute(
                """
                INSERT INTO api_positions (wallet_address, token_id, shares, last_synced_at)
                VALUES (?, 'token_old', 100, ?)
                """,
                (wallet_address.lower(), datetime.now(timezone.utc).isoformat())
            )

        # Now upsert empty list
        with storage.transaction() as txn:
            count = txn.upsert_api_positions(wallet_address, [])
            positions = txn.get_api_positions(wallet_address)

        assert count == 0
        assert len(positions) == 0

    def test_upsert_api_positions(self, storage, wallet_address, sample_api_positions):
        """Test upserting positions from API"""
        with storage.transaction() as txn:
            count = txn.upsert_api_positions(wallet_address, sample_api_positions)

        assert count == 2

        with storage.transaction() as txn:
            positions = txn.get_api_positions(wallet_address)

        assert len(positions) == 2
        assert positions[0]["token_id"] == "token_abc123"
        assert positions[0]["shares"] == 100.5
        assert positions[0]["current_price"] == 0.72

    def test_upsert_replaces_old_positions(self, storage, wallet_address, sample_api_positions):
        """Test that upsert replaces all old positions"""
        # First sync
        with storage.transaction() as txn:
            txn.upsert_api_positions(wallet_address, sample_api_positions)

        # Second sync with different positions
        new_positions = [
            Position(
                id=None,
                agent_id="api",
                market_id="market_3",
                token_id="token_new",
                outcome="Yes",
                shares=200.0,
                entry_price=0.50,
                entry_time=None,
                current_price=0.55,
                status=PositionStatus.OPEN,
            ),
        ]

        with storage.transaction() as txn:
            txn.upsert_api_positions(wallet_address, new_positions)
            positions = txn.get_api_positions(wallet_address)

        # Should only have the new position
        assert len(positions) == 1
        assert positions[0]["token_id"] == "token_new"

    def test_get_api_positions_value(self, storage, wallet_address, sample_api_positions):
        """Test calculating total value of API positions"""
        with storage.transaction() as txn:
            txn.upsert_api_positions(wallet_address, sample_api_positions)
            value = txn.get_api_positions_value(wallet_address)

        # Expected: 100.5 * 0.72 + 50.0 * 0.35 = 72.36 + 17.5 = 89.86
        assert abs(value - 89.86) < 0.01

    def test_get_api_position_by_token(self, storage, wallet_address, sample_api_positions):
        """Test fetching a specific position by token ID"""
        with storage.transaction() as txn:
            txn.upsert_api_positions(wallet_address, sample_api_positions)
            pos = txn.get_api_position_by_token(wallet_address, "token_abc123")

        assert pos is not None
        assert pos["shares"] == 100.5
        assert pos["market_id"] == "market_1"

    def test_get_api_position_nonexistent(self, storage, wallet_address):
        """Test fetching a position that doesn't exist"""
        with storage.transaction() as txn:
            pos = txn.get_api_position_by_token(wallet_address, "nonexistent")

        assert pos is None


# ==================== RECONCILIATION ISSUES TESTS ====================


class TestReconciliationIssues:
    """Tests for reconciliation_issues table operations"""

    def test_log_reconciliation_issue(self, storage, wallet_address):
        """Test logging a reconciliation issue"""
        with storage.transaction() as txn:
            issue_id = txn.log_reconciliation_issue(
                wallet_address=wallet_address,
                issue_type="share_mismatch",
                api_value=100.0,
                computed_value=95.0,
                token_id="token_abc",
                market_id="market_1",
                details="API shows 100, computed shows 95"
            )

        assert issue_id > 0

        with storage.transaction() as txn:
            issues = txn.get_unresolved_issues(wallet_address)

        assert len(issues) == 1
        assert issues[0]["issue_type"] == "share_mismatch"
        assert issues[0]["difference"] == 5.0

    def test_resolve_issue(self, storage, wallet_address):
        """Test resolving an issue"""
        with storage.transaction() as txn:
            issue_id = txn.log_reconciliation_issue(
                wallet_address=wallet_address,
                issue_type="missing_tx",
                api_value=50.0,
                computed_value=0.0,
            )

        with storage.transaction() as txn:
            txn.resolve_issue(issue_id, "Manual sync fixed it", auto_fixed=False)

        with storage.transaction() as txn:
            issues = txn.get_unresolved_issues(wallet_address)

        assert len(issues) == 0

    def test_get_issue_summary(self, storage, wallet_address):
        """Test getting issue summary"""
        with storage.transaction() as txn:
            # Log multiple issues
            txn.log_reconciliation_issue(
                wallet_address=wallet_address,
                issue_type="missing_tx",
                api_value=100.0,
                computed_value=0.0,
            )
            txn.log_reconciliation_issue(
                wallet_address=wallet_address,
                issue_type="missing_tx",
                api_value=50.0,
                computed_value=0.0,
            )
            issue_id = txn.log_reconciliation_issue(
                wallet_address=wallet_address,
                issue_type="share_mismatch",
                api_value=100.0,
                computed_value=90.0,
            )
            # Resolve one
            txn.resolve_issue(issue_id, "Fixed", auto_fixed=True)

        with storage.transaction() as txn:
            summary = txn.get_issue_summary(wallet_address)

        assert summary["missing_tx"]["total"] == 2
        assert summary["missing_tx"]["unresolved"] == 2
        assert summary["share_mismatch"]["total"] == 1
        assert summary["share_mismatch"]["unresolved"] == 0
        assert summary["share_mismatch"]["auto_fixed"] == 1


# ==================== SYNC GAPS TESTS ====================


class TestSyncGaps:
    """Tests for sync_gaps table operations"""

    def test_add_sync_gap(self, storage, wallet_address):
        """Test adding a sync gap"""
        with storage.transaction() as txn:
            gap_id = txn.add_sync_gap(
                wallet_address=wallet_address,
                from_block=1000000,
                to_block=1001000,
                error="RPC timeout"
            )

        assert gap_id > 0

        with storage.transaction() as txn:
            gaps = txn.get_unresolved_gaps(wallet_address)

        assert len(gaps) == 1
        assert gaps[0]["from_block"] == 1000000
        assert gaps[0]["to_block"] == 1001000

    def test_resolve_gap(self, storage, wallet_address):
        """Test resolving a sync gap"""
        with storage.transaction() as txn:
            gap_id = txn.add_sync_gap(
                wallet_address=wallet_address,
                from_block=1000000,
                to_block=1001000,
            )
            txn.resolve_gap(gap_id)

        with storage.transaction() as txn:
            gaps = txn.get_unresolved_gaps(wallet_address)

        assert len(gaps) == 0

    def test_increment_gap_retry(self, storage, wallet_address):
        """Test incrementing retry count"""
        with storage.transaction() as txn:
            gap_id = txn.add_sync_gap(
                wallet_address=wallet_address,
                from_block=1000000,
                to_block=1001000,
            )
            txn.increment_gap_retry(gap_id, "Retry failed")
            txn.increment_gap_retry(gap_id, "Retry failed again")

        with storage.transaction() as txn:
            gaps = txn.get_unresolved_gaps(wallet_address)

        assert gaps[0]["retry_count"] == 2
        assert gaps[0]["last_error"] == "Retry failed again"


# ==================== WALLET STATE TESTS ====================


class TestWalletState:
    """Tests for wallet state using API positions as source of truth"""

    def test_get_wallet_state_uses_api_positions(
        self, storage, wallet_address, sample_api_positions
    ):
        """Test that get_wallet_state uses api_positions table"""
        # Add API positions
        with storage.transaction() as txn:
            txn.upsert_api_positions(wallet_address, sample_api_positions)
            txn.update_usdc_balance(wallet_address, 1000.0)

        # Get wallet state
        with storage.transaction() as txn:
            state = txn.get_wallet_state(wallet_address)

        assert state.usdc_balance == 1000.0
        assert len(state.positions) == 2
        assert state.positions[0].token_id == "token_abc123"
        assert state.positions[0].agent_id == "api"

    def test_get_wallet_state_empty(self, storage, wallet_address):
        """Test wallet state with no positions"""
        with storage.transaction() as txn:
            state = txn.get_wallet_state(wallet_address)

        assert state.usdc_balance == 0.0
        assert len(state.positions) == 0


# ==================== DISCREPANCY DETECTION TESTS ====================


class TestDiscrepancyDetection:
    """Tests for detecting discrepancies between API and computed positions"""

    def test_no_discrepancy_when_matching(self, storage, wallet_address):
        """Test that no issues are logged when API and computed match"""
        # Add matching API and transaction data
        with storage.transaction() as txn:
            # API position
            txn._execute(
                """
                INSERT INTO api_positions (wallet_address, token_id, market_id, shares, last_synced_at)
                VALUES (?, 'token_1', 'market_1', 100.0, ?)
                """,
                (wallet_address.lower(), datetime.now(timezone.utc).isoformat())
            )

            # Transaction that computes to same position
            txn._execute(
                """
                INSERT INTO transactions (
                    tx_hash, log_index, block_number, block_timestamp,
                    transaction_type, wallet_address, token_id, market_id,
                    shares, price_per_share, usdc_amount, synced_at
                ) VALUES (?, 0, 1000, ?, 'buy', ?, 'token_1', 'market_1', 100.0, 0.5, 50.0, ?)
                """,
                (
                    "tx_1",
                    datetime.now(timezone.utc).isoformat(),
                    wallet_address.lower(),
                    datetime.now(timezone.utc).isoformat()
                )
            )

        # Compare positions
        with storage.transaction() as txn:
            api_positions = txn.get_api_positions(wallet_address)
            computed = txn.get_computed_positions(wallet_address)

        assert len(api_positions) == 1
        assert len(computed) == 1
        assert abs(api_positions[0]["shares"] - computed[0]["shares"]) < 0.001

    def test_detects_share_mismatch(self, storage, wallet_address):
        """Test detection of share count mismatch"""
        with storage.transaction() as txn:
            # API shows 100 shares
            txn._execute(
                """
                INSERT INTO api_positions (wallet_address, token_id, shares, last_synced_at)
                VALUES (?, 'token_1', 100.0, ?)
                """,
                (wallet_address.lower(), datetime.now(timezone.utc).isoformat())
            )

            # Transactions show 90 shares
            txn._execute(
                """
                INSERT INTO transactions (
                    tx_hash, log_index, block_number, block_timestamp,
                    transaction_type, wallet_address, token_id,
                    shares, synced_at
                ) VALUES (?, 0, 1000, ?, 'buy', ?, 'token_1', 90.0, ?)
                """,
                (
                    "tx_1",
                    datetime.now(timezone.utc).isoformat(),
                    wallet_address.lower(),
                    datetime.now(timezone.utc).isoformat()
                )
            )

        with storage.transaction() as txn:
            api_positions = txn.get_api_positions(wallet_address)
            computed = txn.get_computed_positions(wallet_address)

        # Should detect 10 share difference
        diff = api_positions[0]["shares"] - computed[0]["shares"]
        assert abs(diff - 10.0) < 0.001

    def test_detects_missing_transaction(self, storage, wallet_address):
        """Test detection when API shows position but no transactions exist"""
        with storage.transaction() as txn:
            # API shows position
            txn._execute(
                """
                INSERT INTO api_positions (wallet_address, token_id, shares, last_synced_at)
                VALUES (?, 'token_missing', 50.0, ?)
                """,
                (wallet_address.lower(), datetime.now(timezone.utc).isoformat())
            )

        with storage.transaction() as txn:
            api_positions = txn.get_api_positions(wallet_address)
            computed = txn.get_computed_positions(wallet_address)

        # API has position, computed doesn't
        assert len(api_positions) == 1
        assert len(computed) == 0

    def test_detects_extra_transaction(self, storage, wallet_address):
        """Test detection when transactions show position but API doesn't"""
        with storage.transaction() as txn:
            # No API position, but transactions exist
            txn._execute(
                """
                INSERT INTO transactions (
                    tx_hash, log_index, block_number, block_timestamp,
                    transaction_type, wallet_address, token_id,
                    shares, synced_at
                ) VALUES (?, 0, 1000, ?, 'buy', ?, 'token_ghost', 100.0, ?)
                """,
                (
                    "tx_ghost",
                    datetime.now(timezone.utc).isoformat(),
                    wallet_address.lower(),
                    datetime.now(timezone.utc).isoformat()
                )
            )

        with storage.transaction() as txn:
            api_positions = txn.get_api_positions(wallet_address)
            computed = txn.get_computed_positions(wallet_address)

        # Computed has position, API doesn't
        assert len(api_positions) == 0
        assert len(computed) == 1


# ==================== INTEGRATION TESTS ====================


class TestReconciliationIntegration:
    """Integration tests for the full reconciliation flow"""

    @pytest.fixture
    def mock_api(self):
        """Create a mock API client"""
        api = AsyncMock()
        api.fetch_positions = AsyncMock(return_value=[])
        api.fetch_usdc_balance = AsyncMock(return_value=1000.0)
        api.connect = AsyncMock()
        api.close = AsyncMock()
        return api

    @pytest.fixture
    def config(self, temp_db_path):
        """Create a test config"""
        config = MagicMock()
        config.db_path = temp_db_path
        config.proxy_address = "0xtest1234"
        config.risk = MagicMock()
        config.risk.stale_agent_threshold_seconds = 120
        config.risk.heartbeat_interval_seconds = 30
        config.chain_sync = MagicMock()
        config.chain_sync.initial_sync_block = 1000000
        config.chain_sync.batch_size = 1000
        config.chain_sync.max_retries = 3
        config.chain_sync.retry_delay_seconds = 1.0
        return config

    def test_reconcile_state_stores_api_positions(
        self, storage, config, mock_api, sample_api_positions
    ):
        """Test that reconciliation stores API positions as source of truth"""
        mock_api.fetch_positions.return_value = sample_api_positions

        coordinator = RiskCoordinator(
            config=config,
            storage=storage,
            api=mock_api
        )
        coordinator.wallet_address = config.proxy_address

        async def run_test():
            # Mock chain sync
            with patch.object(coordinator, '_run_chain_sync_audit', new_callable=AsyncMock):
                await coordinator._reconcile_state()

        asyncio.run(run_test())

        # Verify API positions were stored
        with storage.transaction() as txn:
            positions = txn.get_api_positions(config.proxy_address)

        assert len(positions) == 2
        assert positions[0]["token_id"] in ["token_abc123", "token_def456"]

    def test_reconcile_logs_discrepancies(
        self, storage, config, mock_api, sample_api_positions
    ):
        """Test that reconciliation logs discrepancies between API and computed"""
        mock_api.fetch_positions.return_value = sample_api_positions

        # Pre-populate transactions with different shares
        with storage.transaction() as txn:
            txn._execute(
                """
                INSERT INTO transactions (
                    tx_hash, log_index, block_number, block_timestamp,
                    transaction_type, wallet_address, token_id, market_id,
                    shares, synced_at
                ) VALUES (?, 0, 1000, ?, 'buy', ?, 'token_abc123', 'market_1', 90.0, ?)
                """,
                (
                    "tx_1",
                    datetime.now(timezone.utc).isoformat(),
                    config.proxy_address.lower(),
                    datetime.now(timezone.utc).isoformat()
                )
            )

        coordinator = RiskCoordinator(
            config=config,
            storage=storage,
            api=mock_api
        )
        coordinator.wallet_address = config.proxy_address

        async def run_test():
            # Mock chain sync
            with patch.object(coordinator, '_run_chain_sync_audit', new_callable=AsyncMock):
                await coordinator._reconcile_state()

        asyncio.run(run_test())

        # Verify discrepancies were logged
        with storage.transaction() as txn:
            issues = txn.get_unresolved_issues(config.proxy_address)

        # Should have issues: token_abc123 mismatch (100.5 vs 90)
        # and token_def456 missing_tx (50 vs 0)
        assert len(issues) >= 1

    def test_validate_wallet_state(self, storage, config, mock_api, sample_api_positions):
        """Test the validate_wallet_state method"""
        mock_api.fetch_positions.return_value = sample_api_positions

        coordinator = RiskCoordinator(
            config=config,
            storage=storage,
            api=mock_api
        )
        coordinator.wallet_address = config.proxy_address

        async def run_test():
            return await coordinator.validate_wallet_state()

        is_valid, issues = asyncio.run(run_test())

        # With no transactions, all API positions should be flagged
        assert not is_valid
        assert len(issues) == 2


# ==================== RUN TESTS ====================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
