"""Tests for FeedbackCollector and FeedbackStore.

Tests the feedback collection system including:
- Delayed feedback (track → record flow)
- Immediate feedback
- Store persistence
- Adapter registration
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.models import QueryFeatures, RoutingDecision
from conduit.feedback.collector import FeedbackCollector
from conduit.feedback.models import FeedbackEvent, PendingQuery
from conduit.feedback.stores import InMemoryFeedbackStore


@pytest.fixture
def mock_router():
    """Create a mock Router."""
    router = MagicMock()
    router.update = AsyncMock()
    return router


@pytest.fixture
def mock_features():
    """Create mock QueryFeatures."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        query_text="What is Python?",
    )


@pytest.fixture
def mock_decision(mock_features):
    """Create mock RoutingDecision."""
    return RoutingDecision(
        query_id="test-query-123",
        selected_model="gpt-4o-mini",
        confidence=0.85,
        features=mock_features,
        reasoning="Test routing decision",
    )


class TestInMemoryFeedbackStore:
    """Tests for InMemoryFeedbackStore."""

    @pytest.mark.asyncio
    async def test_save_and_get_pending(self):
        """Should save and retrieve pending query."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [0.1], "token_count": 50, "complexity_score": 0.5},
        )

        await store.save_pending(pending)
        retrieved = await store.get_pending("q1")

        assert retrieved is not None
        assert retrieved.query_id == "q1"
        assert retrieved.model_id == "gpt-4o"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Should return None for unknown query_id."""
        store = InMemoryFeedbackStore()
        result = await store.get_pending("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_pending(self):
        """Should delete pending query."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
        )

        await store.save_pending(pending)
        await store.delete_pending("q1")
        result = await store.get_pending("q1")

        assert result is None

    @pytest.mark.asyncio
    async def test_count_pending(self):
        """Should count pending queries."""
        store = InMemoryFeedbackStore()

        assert await store.count_pending() == 0

        for i in range(3):
            pending = PendingQuery(
                query_id=f"q{i}",
                model_id="gpt-4o",
                features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            )
            await store.save_pending(pending)

        assert await store.count_pending() == 3

    @pytest.mark.asyncio
    async def test_expired_query_not_returned(self):
        """Expired queries should not be returned."""
        store = InMemoryFeedbackStore()

        # Create pending with very short TTL
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),  # Created 2 hours ago
            ttl_seconds=60,  # 1 minute TTL
        )

        await store.save_pending(pending)
        result = await store.get_pending("q1")

        assert result is None  # Expired


class TestFeedbackCollector:
    """Tests for FeedbackCollector."""

    @pytest.mark.asyncio
    async def test_default_adapters_registered(self, mock_router):
        """Should register default adapters on init."""
        collector = FeedbackCollector(mock_router)

        assert "thumbs" in collector.registered_signals
        assert "rating" in collector.registered_signals
        assert "task_success" in collector.registered_signals
        assert "quality_score" in collector.registered_signals
        assert "completion_time" in collector.registered_signals

    @pytest.mark.asyncio
    async def test_track_query(self, mock_router, mock_decision):
        """Should track query for delayed feedback."""
        collector = FeedbackCollector(mock_router)

        await collector.track(mock_decision, cost=0.001, latency=0.5)

        count = await collector.get_pending_count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_record_feedback_success(self, mock_router, mock_decision):
        """Should record feedback and update router."""
        collector = FeedbackCollector(mock_router)

        # Track query
        await collector.track(mock_decision, cost=0.001, latency=0.5)

        # Record feedback
        event = FeedbackEvent(
            query_id=mock_decision.query_id,
            signal_type="thumbs",
            payload={"value": "up"},
        )
        success = await collector.record(event)

        assert success is True
        mock_router.update.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_router.update.call_args[1]
        assert call_kwargs["model_id"] == "gpt-4o-mini"
        assert call_kwargs["quality_score"] == 1.0  # Thumbs up
        assert call_kwargs["cost"] == 0.001
        assert call_kwargs["latency"] == 0.5

    @pytest.mark.asyncio
    async def test_record_feedback_unknown_query(self, mock_router):
        """Should return False for unknown query_id."""
        collector = FeedbackCollector(mock_router)

        event = FeedbackEvent(
            query_id="unknown-query",
            signal_type="thumbs",
            payload={"value": "up"},
        )
        success = await collector.record(event)

        assert success is False
        mock_router.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_feedback_unknown_signal(self, mock_router, mock_decision):
        """Should return False for unknown signal type."""
        collector = FeedbackCollector(mock_router)

        await collector.track(mock_decision)

        event = FeedbackEvent(
            query_id=mock_decision.query_id,
            signal_type="unknown_signal_type",
            payload={},
        )
        success = await collector.record(event)

        assert success is False
        mock_router.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_immediate(self, mock_router, mock_features):
        """Should record immediate feedback without tracking."""
        collector = FeedbackCollector(mock_router)

        success = await collector.record_immediate(
            model_id="gpt-4o",
            features=mock_features,
            signal_type="thumbs",
            payload={"value": "down"},
            cost=0.002,
            latency=1.0,
        )

        assert success is True
        mock_router.update.assert_called_once()

        call_kwargs = mock_router.update.call_args[1]
        assert call_kwargs["model_id"] == "gpt-4o"
        assert call_kwargs["quality_score"] == 0.0  # Thumbs down
        assert call_kwargs["cost"] == 0.002

    @pytest.mark.asyncio
    async def test_record_batch(self, mock_router, mock_decision):
        """Should record multiple feedback events."""
        collector = FeedbackCollector(mock_router)

        # Track multiple queries
        for i in range(3):
            decision = RoutingDecision(
                query_id=f"q{i}",
                selected_model="gpt-4o-mini",
                confidence=0.85,
                features=mock_decision.features,
                reasoning="Test",
            )
            await collector.track(decision)

        # Record batch
        events = [
            FeedbackEvent(query_id="q0", signal_type="thumbs", payload={"value": "up"}),
            FeedbackEvent(query_id="q1", signal_type="rating", payload={"rating": 4}),
            FeedbackEvent(query_id="q2", signal_type="task_success", payload={"success": True}),
        ]
        results = await collector.record_batch(events)

        assert results["q0"] is True
        assert results["q1"] is True
        assert results["q2"] is True
        assert mock_router.update.call_count == 3

    @pytest.mark.asyncio
    async def test_pending_cleaned_after_record(self, mock_router, mock_decision):
        """Pending query should be removed after feedback recorded."""
        collector = FeedbackCollector(mock_router)

        await collector.track(mock_decision)
        assert await collector.get_pending_count() == 1

        event = FeedbackEvent(
            query_id=mock_decision.query_id,
            signal_type="thumbs",
            payload={"value": "up"},
        )
        await collector.record(event)

        assert await collector.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_custom_adapter_registration(self, mock_router):
        """Should allow custom adapter registration."""
        from conduit.feedback.adapters import FeedbackAdapter
        from conduit.feedback.models import RewardMapping

        class CustomAdapter(FeedbackAdapter):
            @property
            def signal_type(self):
                return "custom_signal"

            def to_reward(self, event):
                return RewardMapping(reward=0.75, confidence=0.9)

        collector = FeedbackCollector(mock_router)
        collector.register(CustomAdapter())

        assert "custom_signal" in collector.registered_signals


class TestPendingQuery:
    """Tests for PendingQuery model."""

    def test_is_expired_false(self):
        """Fresh query should not be expired."""
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            ttl_seconds=3600,
        )
        assert pending.is_expired() is False

    def test_is_expired_true(self):
        """Old query should be expired."""
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            ttl_seconds=60,  # 1 minute TTL
        )
        assert pending.is_expired() is True


class TestAtomicStoreOperations:
    """Tests for atomic store operations."""

    @pytest.mark.asyncio
    async def test_get_and_delete_pending(self):
        """Should atomically retrieve and delete pending query."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
        )

        await store.save_pending(pending)

        # Atomic get and delete
        retrieved = await store.get_and_delete_pending("q1")
        assert retrieved is not None
        assert retrieved.query_id == "q1"

        # Should be deleted now
        second_retrieve = await store.get_and_delete_pending("q1")
        assert second_retrieve is None

    @pytest.mark.asyncio
    async def test_get_and_delete_nonexistent(self):
        """Should return None for unknown query_id."""
        store = InMemoryFeedbackStore()
        result = await store.get_and_delete_pending("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_and_delete_expired(self):
        """Should return None for expired query."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            ttl_seconds=60,  # 1 minute TTL - expired
        )

        await store.save_pending(pending)
        result = await store.get_and_delete_pending("q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_pending_cost_latency(self):
        """Should update cost and latency for pending query."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            cost=0.0,
            latency=0.0,
        )

        await store.save_pending(pending)

        # Update with actual values
        success = await store.update_pending("q1", cost=0.005, latency=1.2)
        assert success is True

        # Verify updates
        updated = await store.get_pending("q1")
        assert updated.cost == 0.005
        assert updated.latency == 1.2

    @pytest.mark.asyncio
    async def test_update_pending_partial(self):
        """Should update only provided fields."""
        store = InMemoryFeedbackStore()
        pending = PendingQuery(
            query_id="q1",
            model_id="gpt-4o",
            features={"embedding": [], "token_count": 50, "complexity_score": 0.5},
            cost=0.001,
            latency=0.5,
        )

        await store.save_pending(pending)

        # Update only cost
        await store.update_pending("q1", cost=0.002)

        updated = await store.get_pending("q1")
        assert updated.cost == 0.002
        assert updated.latency == 0.5  # Unchanged

    @pytest.mark.asyncio
    async def test_update_pending_nonexistent(self):
        """Should return False for unknown query_id."""
        store = InMemoryFeedbackStore()
        result = await store.update_pending("nonexistent", cost=0.1)
        assert result is False


class TestCollectorUpdateTracked:
    """Tests for FeedbackCollector.update_tracked()."""

    @pytest.mark.asyncio
    async def test_update_tracked_success(self, mock_router, mock_decision):
        """Should update cost/latency for tracked query."""
        collector = FeedbackCollector(mock_router)

        # Track with placeholder values
        await collector.track(mock_decision, cost=0.0, latency=0.0)

        # Update with actual values
        success = await collector.update_tracked(
            mock_decision.query_id,
            cost=0.005,
            latency=1.5,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_update_tracked_nonexistent(self, mock_router):
        """Should return False for unknown query_id."""
        collector = FeedbackCollector(mock_router)

        success = await collector.update_tracked(
            "nonexistent",
            cost=0.005,
        )

        assert success is False


class TestIdempotencyStore:
    """Tests for idempotency tracking in FeedbackStore."""

    @pytest.mark.asyncio
    async def test_was_processed_false_initially(self):
        """New key should not be marked as processed."""
        store = InMemoryFeedbackStore()
        assert await store.was_processed("new-key") is False

    @pytest.mark.asyncio
    async def test_mark_and_check_processed(self):
        """Marked key should be detected as processed."""
        store = InMemoryFeedbackStore()

        await store.mark_processed("test-key", ttl_seconds=3600)
        assert await store.was_processed("test-key") is True

    @pytest.mark.asyncio
    async def test_processed_key_expires(self):
        """Processed key should expire after TTL."""
        store = InMemoryFeedbackStore()

        # Mark with expired TTL (using negative time via datetime manipulation)
        await store.mark_processed("expiring-key", ttl_seconds=1)

        # Manually set expiry to past
        from datetime import datetime, timezone, timedelta
        store._processed["expiring-key"] = datetime.now(timezone.utc) - timedelta(seconds=10)

        assert await store.was_processed("expiring-key") is False

    @pytest.mark.asyncio
    async def test_multiple_keys_independent(self):
        """Different keys should be tracked independently."""
        store = InMemoryFeedbackStore()

        await store.mark_processed("key-1", ttl_seconds=3600)

        assert await store.was_processed("key-1") is True
        assert await store.was_processed("key-2") is False


class TestIdempotencyFeedbackEvent:
    """Tests for FeedbackEvent.get_idempotency_key()."""

    def test_natural_key_generation(self):
        """Should generate natural key from query_id and signal_type."""
        event = FeedbackEvent(
            query_id="q123",
            signal_type="thumbs",
            payload={"value": "up"},
        )
        assert event.get_idempotency_key() == "q123:thumbs"

    def test_explicit_idempotency_key(self):
        """Should use explicit idempotency_key if provided."""
        event = FeedbackEvent(
            query_id="q123",
            signal_type="thumbs",
            payload={"value": "up"},
            idempotency_key="client-uuid-abc123",
        )
        assert event.get_idempotency_key() == "client-uuid-abc123"

    def test_different_signals_different_keys(self):
        """Same query with different signals should have different keys."""
        event1 = FeedbackEvent(query_id="q1", signal_type="thumbs", payload={})
        event2 = FeedbackEvent(query_id="q1", signal_type="rating", payload={})

        assert event1.get_idempotency_key() != event2.get_idempotency_key()


class TestIdempotencyCollector:
    """Tests for idempotency in FeedbackCollector methods."""

    @pytest.mark.asyncio
    async def test_record_idempotent_duplicate(self, mock_router, mock_decision):
        """Duplicate feedback should be ignored (idempotent)."""
        collector = FeedbackCollector(mock_router)

        # Track query
        await collector.track(mock_decision, cost=0.001, latency=0.5)

        # First record
        event = FeedbackEvent(
            query_id=mock_decision.query_id,
            signal_type="thumbs",
            payload={"value": "up"},
        )
        first_result = await collector.record(event)
        assert first_result is True
        assert mock_router.update.call_count == 1

        # Duplicate record (should be ignored but return True)
        second_result = await collector.record(event)
        assert second_result is True  # Idempotent success
        assert mock_router.update.call_count == 1  # NOT called again

    @pytest.mark.asyncio
    async def test_record_different_signals_both_allowed(self, mock_router, mock_decision):
        """Different signal types for same query should both be recorded."""
        collector = FeedbackCollector(mock_router)

        # Need two tracked queries (since first record deletes the pending)
        decision1 = RoutingDecision(
            query_id="q1",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=mock_decision.features,
            reasoning="Test",
        )
        decision2 = RoutingDecision(
            query_id="q2",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=mock_decision.features,
            reasoning="Test",
        )
        await collector.track(decision1)
        await collector.track(decision2)

        # Record different signal types
        event1 = FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"})
        event2 = FeedbackEvent(query_id="q2", signal_type="rating", payload={"rating": 5})

        result1 = await collector.record(event1)
        result2 = await collector.record(event2)

        assert result1 is True
        assert result2 is True
        assert mock_router.update.call_count == 2

    @pytest.mark.asyncio
    async def test_record_with_explicit_idempotency_key(self, mock_router, mock_decision):
        """Should use explicit idempotency_key for deduplication."""
        collector = FeedbackCollector(mock_router)

        await collector.track(mock_decision)

        # Record with explicit key
        event = FeedbackEvent(
            query_id=mock_decision.query_id,
            signal_type="thumbs",
            payload={"value": "up"},
            idempotency_key="my-unique-key",
        )
        first_result = await collector.record(event)
        assert first_result is True

        # Duplicate with same explicit key
        second_result = await collector.record(event)
        assert second_result is True  # Idempotent
        assert mock_router.update.call_count == 1

    @pytest.mark.asyncio
    async def test_record_aggregated_idempotent(self, mock_router, mock_decision):
        """Aggregated feedback should be idempotent."""
        collector = FeedbackCollector(mock_router)

        await collector.track(mock_decision)

        events = [
            FeedbackEvent(query_id=mock_decision.query_id, signal_type="thumbs", payload={"value": "up"}),
            FeedbackEvent(query_id=mock_decision.query_id, signal_type="task_success", payload={"success": True}),
        ]

        first_result = await collector.record_aggregated(mock_decision.query_id, events)
        assert first_result is True
        assert mock_router.update.call_count == 1

        # Track again for second attempt
        await collector.track(mock_decision)

        # Duplicate call
        second_result = await collector.record_aggregated(mock_decision.query_id, events)
        assert second_result is True  # Idempotent
        assert mock_router.update.call_count == 1  # NOT called again

    @pytest.mark.asyncio
    async def test_record_session_feedback_idempotent(self, mock_router, mock_decision):
        """Session feedback should be idempotent."""
        collector = FeedbackCollector(mock_router)

        # Track queries with session
        session_id = "test-session"
        decision1 = RoutingDecision(
            query_id="q1",
            selected_model="gpt-4o-mini",
            confidence=0.85,
            features=mock_decision.features,
            reasoning="Test",
        )
        await collector.track(decision1, session_id=session_id)

        # First session feedback
        first_result = await collector.record_session_feedback(
            session_id=session_id,
            signal_type="rating",
            payload={"rating": 5},
        )
        assert len(first_result) == 1
        assert first_result["q1"] is True
        assert mock_router.update.call_count == 1

        # Track again for second attempt
        await collector.track(decision1, session_id=session_id)

        # Duplicate session feedback (same session + signal)
        second_result = await collector.record_session_feedback(
            session_id=session_id,
            signal_type="rating",
            payload={"rating": 5},
        )
        assert len(second_result) == 0  # Empty - already processed
        assert mock_router.update.call_count == 1  # NOT called again
