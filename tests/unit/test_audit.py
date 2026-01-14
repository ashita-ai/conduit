"""Tests for observability/audit.py module.

Tests cover:
- AuditEntry model validation
- AuditQuery model validation
- InMemoryAuditStore operations
- create_audit_entry helper function
- PostgresAuditStore (mocked database)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.models import QueryFeatures, RoutingDecision
from conduit.observability.audit import (
    AuditEntry,
    AuditQuery,
    InMemoryAuditStore,
    PostgresAuditStore,
    create_audit_entry,
)


class TestAuditEntry:
    """Tests for AuditEntry Pydantic model."""

    def test_create_valid_entry(self):
        """Test creating a valid audit entry."""
        entry = AuditEntry(
            decision_id="decision-123",
            query_id="query-456",
            selected_model="gpt-4o-mini",
            confidence=0.95,
            algorithm_phase="thompson_sampling",
            query_count=1500,
            arm_scores={
                "gpt-4o-mini": {"mean": 0.8, "variance": 0.1, "total": 0.8},
                "gpt-4o": {"mean": 0.9, "variance": 0.05, "total": 0.9},
            },
        )

        assert entry.decision_id == "decision-123"
        assert entry.query_id == "query-456"
        assert entry.selected_model == "gpt-4o-mini"
        assert entry.confidence == 0.95
        assert entry.algorithm_phase == "thompson_sampling"
        assert entry.query_count == 1500
        assert len(entry.arm_scores) == 2
        assert entry.fallback_chain == []
        assert entry.feature_vector is None
        assert entry.constraints_applied == {}

    def test_entry_with_all_fields(self):
        """Test creating entry with all optional fields."""
        now = datetime.now(timezone.utc)
        entry = AuditEntry(
            id=1,
            decision_id="decision-123",
            query_id="query-456",
            selected_model="gpt-4o-mini",
            fallback_chain=["gpt-4o", "claude-3-5-sonnet"],
            confidence=0.85,
            algorithm_phase="linucb",
            query_count=5000,
            arm_scores={
                "gpt-4o-mini": {"mean": 0.7, "uncertainty": 0.15, "total": 0.85}
            },
            feature_vector=[0.1] * 386,
            constraints_applied={"max_cost": 0.01, "constraints_relaxed": True},
            reasoning="Selected for cost efficiency",
            created_at=now,
        )

        assert entry.id == 1
        assert entry.fallback_chain == ["gpt-4o", "claude-3-5-sonnet"]
        assert entry.feature_vector is not None
        assert len(entry.feature_vector) == 386
        assert entry.constraints_applied["max_cost"] == 0.01
        assert entry.reasoning == "Selected for cost efficiency"
        assert entry.created_at == now

    def test_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            AuditEntry(
                decision_id="d1",
                query_id="q1",
                selected_model="gpt-4o-mini",
                confidence=1.5,  # Invalid
                algorithm_phase="thompson_sampling",
                query_count=100,
                arm_scores={},
            )

        with pytest.raises(ValueError):
            AuditEntry(
                decision_id="d1",
                query_id="q1",
                selected_model="gpt-4o-mini",
                confidence=-0.1,  # Invalid
                algorithm_phase="thompson_sampling",
                query_count=100,
                arm_scores={},
            )

    def test_query_count_validation(self):
        """Test query_count must be non-negative."""
        with pytest.raises(ValueError):
            AuditEntry(
                decision_id="d1",
                query_id="q1",
                selected_model="gpt-4o-mini",
                confidence=0.5,
                algorithm_phase="thompson_sampling",
                query_count=-1,  # Invalid
                arm_scores={},
            )


class TestAuditQuery:
    """Tests for AuditQuery Pydantic model."""

    def test_default_values(self):
        """Test default query values."""
        query = AuditQuery()

        assert query.decision_id is None
        assert query.query_id is None
        assert query.selected_model is None
        assert query.algorithm_phase is None
        assert query.start_time is None
        assert query.end_time is None
        assert query.limit == 100
        assert query.offset == 0

    def test_query_with_filters(self):
        """Test query with all filters set."""
        now = datetime.now(timezone.utc)
        query = AuditQuery(
            decision_id="decision-123",
            query_id="query-456",
            selected_model="gpt-4o-mini",
            algorithm_phase="linucb",
            start_time=now - timedelta(hours=1),
            end_time=now,
            limit=50,
            offset=10,
        )

        assert query.decision_id == "decision-123"
        assert query.query_id == "query-456"
        assert query.selected_model == "gpt-4o-mini"
        assert query.algorithm_phase == "linucb"
        assert query.start_time is not None
        assert query.end_time is not None
        assert query.limit == 50
        assert query.offset == 10

    def test_limit_validation(self):
        """Test limit must be between 1 and 1000."""
        with pytest.raises(ValueError):
            AuditQuery(limit=0)

        with pytest.raises(ValueError):
            AuditQuery(limit=1001)

        # Valid limits
        query1 = AuditQuery(limit=1)
        assert query1.limit == 1

        query2 = AuditQuery(limit=1000)
        assert query2.limit == 1000

    def test_offset_validation(self):
        """Test offset must be non-negative."""
        with pytest.raises(ValueError):
            AuditQuery(offset=-1)

        query = AuditQuery(offset=0)
        assert query.offset == 0


class TestInMemoryAuditStore:
    """Tests for InMemoryAuditStore."""

    @pytest.fixture
    def store(self):
        """Create an in-memory audit store."""
        return InMemoryAuditStore(max_entries=100)

    @pytest.fixture
    def sample_entry(self):
        """Create a sample audit entry."""
        return AuditEntry(
            decision_id="decision-123",
            query_id="query-456",
            selected_model="gpt-4o-mini",
            confidence=0.95,
            algorithm_phase="thompson_sampling",
            query_count=1500,
            arm_scores={"gpt-4o-mini": {"mean": 0.8, "total": 0.8}},
        )

    @pytest.mark.asyncio
    async def test_log_decision(self, store, sample_entry):
        """Test logging a decision creates entry with ID."""
        entry_id = await store.log_decision(sample_entry)

        assert entry_id == 1
        assert len(store.entries) == 1
        assert store.entries[0].id == 1
        assert store.entries[0].decision_id == "decision-123"

    @pytest.mark.asyncio
    async def test_log_decision_increments_id(self, store, sample_entry):
        """Test each logged entry gets unique ID."""
        id1 = await store.log_decision(sample_entry)
        id2 = await store.log_decision(sample_entry)
        id3 = await store.log_decision(sample_entry)

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    @pytest.mark.asyncio
    async def test_get_entry(self, store, sample_entry):
        """Test retrieving entry by ID."""
        entry_id = await store.log_decision(sample_entry)
        retrieved = await store.get_entry(entry_id)

        assert retrieved is not None
        assert retrieved.id == entry_id
        assert retrieved.decision_id == "decision-123"

    @pytest.mark.asyncio
    async def test_get_entry_not_found(self, store):
        """Test get_entry returns None for non-existent ID."""
        result = await store.get_entry(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_decision_id(self, store, sample_entry):
        """Test retrieving entry by decision ID."""
        await store.log_decision(sample_entry)
        retrieved = await store.get_by_decision_id("decision-123")

        assert retrieved is not None
        assert retrieved.decision_id == "decision-123"

    @pytest.mark.asyncio
    async def test_get_by_decision_id_not_found(self, store):
        """Test get_by_decision_id returns None for non-existent decision."""
        result = await store.get_by_decision_id("non-existent")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_all(self, store):
        """Test querying all entries."""
        # Add multiple entries
        for i in range(5):
            entry = AuditEntry(
                decision_id=f"decision-{i}",
                query_id=f"query-{i}",
                selected_model="gpt-4o-mini",
                confidence=0.9,
                algorithm_phase="thompson_sampling",
                query_count=i * 100,
                arm_scores={},
            )
            await store.log_decision(entry)

        results = await store.query(AuditQuery())

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_query_by_selected_model(self, store):
        """Test filtering by selected model."""
        # Add entries with different models
        for model in ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]:
            entry = AuditEntry(
                decision_id=f"decision-{model}",
                query_id=f"query-{model}",
                selected_model=model,
                confidence=0.9,
                algorithm_phase="thompson_sampling",
                query_count=100,
                arm_scores={},
            )
            await store.log_decision(entry)

        results = await store.query(AuditQuery(selected_model="gpt-4o-mini"))

        assert len(results) == 2
        assert all(r.selected_model == "gpt-4o-mini" for r in results)

    @pytest.mark.asyncio
    async def test_query_by_algorithm_phase(self, store):
        """Test filtering by algorithm phase."""
        for phase in ["thompson_sampling", "linucb", "thompson_sampling"]:
            entry = AuditEntry(
                decision_id=f"decision-{phase}",
                query_id=f"query-{phase}",
                selected_model="gpt-4o-mini",
                confidence=0.9,
                algorithm_phase=phase,
                query_count=100,
                arm_scores={},
            )
            await store.log_decision(entry)

        results = await store.query(AuditQuery(algorithm_phase="linucb"))

        assert len(results) == 1
        assert results[0].algorithm_phase == "linucb"

    @pytest.mark.asyncio
    async def test_query_with_time_range(self, store):
        """Test filtering by time range."""
        now = datetime.now(timezone.utc)

        # Entry from 2 hours ago
        old_entry = AuditEntry(
            decision_id="old-decision",
            query_id="old-query",
            selected_model="gpt-4o-mini",
            confidence=0.9,
            algorithm_phase="thompson_sampling",
            query_count=100,
            arm_scores={},
            created_at=now - timedelta(hours=2),
        )
        await store.log_decision(old_entry)

        # Entry from 30 minutes ago
        recent_entry = AuditEntry(
            decision_id="recent-decision",
            query_id="recent-query",
            selected_model="gpt-4o-mini",
            confidence=0.9,
            algorithm_phase="thompson_sampling",
            query_count=200,
            arm_scores={},
            created_at=now - timedelta(minutes=30),
        )
        await store.log_decision(recent_entry)

        # Query for entries in last hour
        results = await store.query(AuditQuery(start_time=now - timedelta(hours=1)))

        assert len(results) == 1
        assert results[0].decision_id == "recent-decision"

    @pytest.mark.asyncio
    async def test_query_with_limit_and_offset(self, store):
        """Test pagination with limit and offset."""
        # Add 10 entries
        for i in range(10):
            entry = AuditEntry(
                decision_id=f"decision-{i}",
                query_id=f"query-{i}",
                selected_model="gpt-4o-mini",
                confidence=0.9,
                algorithm_phase="thompson_sampling",
                query_count=i * 100,
                arm_scores={},
            )
            await store.log_decision(entry)

        # Get first 3
        page1 = await store.query(AuditQuery(limit=3, offset=0))
        assert len(page1) == 3

        # Get next 3
        page2 = await store.query(AuditQuery(limit=3, offset=3))
        assert len(page2) == 3

        # Entries should be different
        page1_ids = {e.decision_id for e in page1}
        page2_ids = {e.decision_id for e in page2}
        assert page1_ids.isdisjoint(page2_ids)

    @pytest.mark.asyncio
    async def test_delete_older_than(self, store):
        """Test deleting entries older than cutoff."""
        now = datetime.now(timezone.utc)

        # Add old entry
        old_entry = AuditEntry(
            decision_id="old-decision",
            query_id="old-query",
            selected_model="gpt-4o-mini",
            confidence=0.9,
            algorithm_phase="thompson_sampling",
            query_count=100,
            arm_scores={},
            created_at=now - timedelta(days=100),
        )
        await store.log_decision(old_entry)

        # Add recent entry
        recent_entry = AuditEntry(
            decision_id="recent-decision",
            query_id="recent-query",
            selected_model="gpt-4o-mini",
            confidence=0.9,
            algorithm_phase="thompson_sampling",
            query_count=200,
            arm_scores={},
            created_at=now - timedelta(days=10),
        )
        await store.log_decision(recent_entry)

        # Delete entries older than 90 days
        deleted = await store.delete_older_than(now - timedelta(days=90))

        assert deleted == 1
        assert len(store.entries) == 1
        assert store.entries[0].decision_id == "recent-decision"

    @pytest.mark.asyncio
    async def test_max_entries_limit(self):
        """Test that store trims oldest entries when limit exceeded."""
        store = InMemoryAuditStore(max_entries=5)

        # Add 10 entries
        for i in range(10):
            entry = AuditEntry(
                decision_id=f"decision-{i}",
                query_id=f"query-{i}",
                selected_model="gpt-4o-mini",
                confidence=0.9,
                algorithm_phase="thompson_sampling",
                query_count=i * 100,
                arm_scores={},
            )
            await store.log_decision(entry)

        # Should only have 5 entries (the most recent)
        assert len(store.entries) == 5
        # Most recent entries should be kept
        decision_ids = [e.decision_id for e in store.entries]
        assert "decision-9" in decision_ids
        assert "decision-8" in decision_ids
        assert "decision-0" not in decision_ids


class TestCreateAuditEntry:
    """Tests for create_audit_entry helper function."""

    @pytest.fixture
    def sample_decision(self):
        """Create a sample routing decision."""
        features = QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.7,
            query_text="What is the meaning of life?",
        )
        return RoutingDecision(
            query_id="query-123",
            selected_model="gpt-4o-mini",
            fallback_chain=["gpt-4o", "claude-3-5-sonnet"],
            confidence=0.92,
            features=features,
            reasoning="Selected based on cost efficiency",
        )

    def test_create_entry_from_decision(self, sample_decision):
        """Test creating audit entry from routing decision."""
        arm_scores = {
            "gpt-4o-mini": {"mean": 0.8, "variance": 0.1, "total": 0.8},
            "gpt-4o": {"mean": 0.9, "variance": 0.05, "total": 0.9},
        }

        entry = create_audit_entry(
            decision=sample_decision,
            algorithm_phase="thompson_sampling",
            query_count=1500,
            arm_scores=arm_scores,
        )

        assert entry.decision_id == sample_decision.id
        assert entry.query_id == "query-123"
        assert entry.selected_model == "gpt-4o-mini"
        assert entry.fallback_chain == ["gpt-4o", "claude-3-5-sonnet"]
        assert entry.confidence == 0.92
        assert entry.algorithm_phase == "thompson_sampling"
        assert entry.query_count == 1500
        assert entry.arm_scores == arm_scores
        assert entry.reasoning == "Selected based on cost efficiency"

    def test_create_entry_with_constraints(self, sample_decision):
        """Test creating entry with constraints."""
        arm_scores = {"gpt-4o-mini": {"total": 0.8}}
        constraints = {"max_cost": 0.01, "max_latency": 2.0}

        entry = create_audit_entry(
            decision=sample_decision,
            algorithm_phase="linucb",
            query_count=5000,
            arm_scores=arm_scores,
            constraints_applied=constraints,
        )

        assert entry.constraints_applied == constraints

    def test_create_entry_includes_feature_vector(self, sample_decision):
        """Test that feature vector is extracted from decision."""
        arm_scores = {"gpt-4o-mini": {"total": 0.8}}

        entry = create_audit_entry(
            decision=sample_decision,
            algorithm_phase="linucb",
            query_count=5000,
            arm_scores=arm_scores,
        )

        assert entry.feature_vector is not None
        # 384 embedding dims + token_count + complexity_score = 386
        assert len(entry.feature_vector) == 386


class TestPostgresAuditStore:
    """Tests for PostgresAuditStore with mocked database."""

    @pytest.fixture
    def mock_pool(self):
        """Create a mock asyncpg pool."""
        pool = MagicMock()
        return pool

    @pytest.fixture
    def store(self, mock_pool):
        """Create a PostgresAuditStore with mock pool."""
        return PostgresAuditStore(pool=mock_pool, retention_days=90)

    @pytest.mark.asyncio
    async def test_log_decision(self, store, mock_pool):
        """Test log_decision inserts entry and returns ID."""
        # Setup mock
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value={"id": 42})
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        entry = AuditEntry(
            decision_id="decision-123",
            query_id="query-456",
            selected_model="gpt-4o-mini",
            confidence=0.95,
            algorithm_phase="thompson_sampling",
            query_count=1500,
            arm_scores={"gpt-4o-mini": {"total": 0.8}},
        )

        result = await store.log_decision(entry)

        assert result == 42
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_retention_policy_disabled(self, mock_pool):
        """Test retention policy does nothing when days = 0."""
        store = PostgresAuditStore(pool=mock_pool, retention_days=0)

        deleted = await store.apply_retention_policy()

        assert deleted == 0
        mock_pool.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_retention_policy_enabled(self, store, mock_pool):
        """Test retention policy deletes old entries."""
        # Setup mock
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(return_value="DELETE 5")
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock()

        deleted = await store.apply_retention_policy()

        assert deleted == 5
        mock_conn.execute.assert_called_once()


class TestComputeScoresIntegration:
    """Tests for compute_scores method on bandit algorithms."""

    @pytest.fixture
    def features(self):
        """Create sample query features."""
        return QueryFeatures(
            embedding=[0.1] * 384,
            token_count=50,
            complexity_score=0.7,
            query_text="test query",
        )

    def test_linucb_compute_scores(self, features):
        """Test LinUCB compute_scores returns expected structure."""
        from conduit.engines.bandits import LinUCBBandit
        from conduit.engines.bandits.base import ModelArm

        arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_input_token=0.00015,
                cost_per_output_token=0.0006,
                expected_quality=0.85,
            ),
            ModelArm(
                model_id="gpt-4o",
                provider="openai",
                model_name="gpt-4o",
                cost_per_input_token=0.005,
                cost_per_output_token=0.015,
                expected_quality=0.95,
            ),
        ]

        bandit = LinUCBBandit(arms=arms, alpha=1.0, feature_dim=386)
        scores = bandit.compute_scores(features)

        assert "gpt-4o-mini" in scores
        assert "gpt-4o" in scores

        for model_id, score_dict in scores.items():
            assert "mean" in score_dict
            assert "uncertainty" in score_dict
            assert "total" in score_dict
            assert isinstance(score_dict["mean"], float)
            assert isinstance(score_dict["uncertainty"], float)
            assert isinstance(score_dict["total"], float)

    def test_thompson_sampling_compute_scores(self, features):
        """Test Thompson Sampling compute_scores returns expected structure."""
        from conduit.engines.bandits import ThompsonSamplingBandit
        from conduit.engines.bandits.base import ModelArm

        arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_input_token=0.00015,
                cost_per_output_token=0.0006,
                expected_quality=0.85,
            ),
            ModelArm(
                model_id="gpt-4o",
                provider="openai",
                model_name="gpt-4o",
                cost_per_input_token=0.005,
                cost_per_output_token=0.015,
                expected_quality=0.95,
            ),
        ]

        bandit = ThompsonSamplingBandit(arms=arms)
        scores = bandit.compute_scores(features)

        assert "gpt-4o-mini" in scores
        assert "gpt-4o" in scores

        for model_id, score_dict in scores.items():
            assert "alpha" in score_dict
            assert "beta" in score_dict
            assert "mean" in score_dict
            assert "variance" in score_dict
            assert "total" in score_dict
            assert isinstance(score_dict["alpha"], float)
            assert isinstance(score_dict["beta"], float)
            assert isinstance(score_dict["mean"], float)
