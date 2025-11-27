"""Unit tests for query history tracking."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from redis.exceptions import ConnectionError, TimeoutError

from conduit.core.models import QueryFeatures
from conduit.feedback.history import QueryHistoryEntry, QueryHistoryTracker


@pytest.fixture
def sample_features():
    """Create sample QueryFeatures for testing."""
    return QueryFeatures(
        embedding=[0.1, 0.2, 0.3] * 128,  # 384 dimensions
        token_count=50,
        complexity_score=0.5
    )


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    return redis


@pytest.fixture
def history_tracker(mock_redis):
    """Create QueryHistoryTracker with mock Redis."""
    return QueryHistoryTracker(redis=mock_redis, ttl_seconds=300)


class TestQueryHistoryEntry:
    """Test QueryHistoryEntry model."""

    def test_entry_creation(self):
        """Test creating query history entry."""
        entry = QueryHistoryEntry(
            query_id="q123",
            query_text="What is Python?",
            embedding=[0.1, 0.2, 0.3],
            timestamp=100.0,
            user_id="user_abc",
            model_used="gpt-4o-mini")

        assert entry.query_id == "q123"
        assert entry.query_text == "What is Python?"
        assert entry.timestamp == 100.0
        assert entry.user_id == "user_abc"
        assert entry.model_used == "gpt-4o-mini"

    def test_entry_optional_model(self):
        """Test entry with optional model_used."""
        entry = QueryHistoryEntry(
            query_id="q123",
            query_text="Test",
            embedding=[0.1],
            timestamp=100.0,
            user_id="user_abc")

        assert entry.model_used is None


class TestQueryHistoryTrackerInit:
    """Test QueryHistoryTracker initialization."""

    def test_init_with_redis(self, mock_redis):
        """Test initialization with Redis client."""
        tracker = QueryHistoryTracker(redis=mock_redis)
        assert tracker.redis is mock_redis
        assert tracker.enabled is True
        assert tracker.ttl == 300  # Default 5 minutes

    def test_init_without_redis(self):
        """Test initialization without Redis client."""
        tracker = QueryHistoryTracker(redis=None)
        assert tracker.redis is None
        assert tracker.enabled is False

    def test_init_custom_ttl(self, mock_redis):
        """Test initialization with custom TTL."""
        tracker = QueryHistoryTracker(redis=mock_redis, ttl_seconds=600)
        assert tracker.ttl == 600


class TestQueryHistoryTrackerAddQuery:
    """Test QueryHistoryTracker.add_query method."""

    async def test_add_query_success(self, history_tracker, mock_redis, sample_features):
        """Test successfully adding query to history."""
        result = await history_tracker.add_query(
            query_id="q123",
            query_text="What is Python?",
            features=sample_features,
            user_id="user_abc",
            model_used="gpt-4o-mini")

        assert result is True
        # Verify Redis setex was called
        mock_redis.setex.assert_called_once()
        # Verify Redis zadd was called for index
        mock_redis.zadd.assert_called_once()

    async def test_add_query_disabled(self, sample_features):
        """Test add_query when tracking is disabled."""
        tracker = QueryHistoryTracker(redis=None)
        result = await tracker.add_query(
            query_id="q123",
            query_text="Test",
            features=sample_features,
            user_id="user_abc")

        assert result is False

    async def test_add_query_connection_error(
        self, history_tracker, mock_redis, sample_features
    ):
        """Test add_query handles Redis connection error gracefully."""
        mock_redis.setex.side_effect = ConnectionError("Connection failed")

        result = await history_tracker.add_query(
            query_id="q123",
            query_text="Test",
            features=sample_features,
            user_id="user_abc")

        assert result is False

    async def test_add_query_timeout_error(
        self, history_tracker, mock_redis, sample_features
    ):
        """Test add_query handles Redis timeout gracefully."""
        mock_redis.setex.side_effect = TimeoutError("Operation timed out")

        result = await history_tracker.add_query(
            query_id="q123",
            query_text="Test",
            features=sample_features,
            user_id="user_abc")

        assert result is False


class TestQueryHistoryTrackerGetRecent:
    """Test QueryHistoryTracker.get_recent_queries method."""

    async def test_get_recent_disabled(self):
        """Test get_recent_queries when tracking is disabled."""
        tracker = QueryHistoryTracker(redis=None)
        result = await tracker.get_recent_queries("user_abc")

        assert result == []

    async def test_get_recent_no_queries(self, history_tracker, mock_redis):
        """Test get_recent_queries with no history."""
        mock_redis.zrange.return_value = []

        result = await history_tracker.get_recent_queries("user_abc")

        assert result == []
        mock_redis.zrange.assert_called_once()

    async def test_get_recent_with_queries(self, history_tracker, mock_redis):
        """Test get_recent_queries retrieves entries."""
        # Mock Redis responses
        mock_redis.zrange.return_value = [b"q123", b"q456"]

        entry1 = QueryHistoryEntry(
            query_id="q123",
            query_text="Query 1",
            embedding=[0.1],
            timestamp=100.0,
            user_id="user_abc")
        entry2 = QueryHistoryEntry(
            query_id="q456",
            query_text="Query 2",
            embedding=[0.2],
            timestamp=200.0,
            user_id="user_abc")

        mock_redis.get.side_effect = [
            entry2.model_dump_json(),
            entry1.model_dump_json(),
        ]

        result = await history_tracker.get_recent_queries("user_abc", limit=10)

        assert len(result) == 2
        assert result[0].query_id == "q456"  # Most recent first
        assert result[1].query_id == "q123"

    async def test_get_recent_with_limit(self, history_tracker, mock_redis):
        """Test get_recent_queries respects limit."""
        mock_redis.zrange.return_value = [b"q1", b"q2", b"q3"]

        entry = QueryHistoryEntry(
            query_id="q1",
            query_text="Test",
            embedding=[0.1],
            timestamp=100.0,
            user_id="user_abc")
        mock_redis.get.return_value = entry.model_dump_json()

        result = await history_tracker.get_recent_queries("user_abc", limit=2)

        assert len(result) == 2

    async def test_get_recent_connection_error(self, history_tracker, mock_redis):
        """Test get_recent_queries handles Redis error gracefully."""
        mock_redis.zrange.side_effect = ConnectionError("Connection failed")

        result = await history_tracker.get_recent_queries("user_abc")

        assert result == []


class TestQueryHistoryTrackerFindSimilar:
    """Test QueryHistoryTracker.find_similar_query method."""

    async def test_find_similar_disabled(self):
        """Test find_similar_query when tracking is disabled."""
        tracker = QueryHistoryTracker(redis=None)
        result = await tracker.find_similar_query(
            current_embedding=[0.5],
            user_id="user_abc")

        assert result is None

    async def test_find_similar_no_history(self, history_tracker, mock_redis):
        """Test find_similar_query with no history."""
        mock_redis.zrangebyscore.return_value = []

        result = await history_tracker.find_similar_query(
            current_embedding=[0.5],
            user_id="user_abc")

        assert result is None

    async def test_find_similar_below_threshold(self, history_tracker, mock_redis):
        """Test find_similar_query with low similarity."""
        entry = QueryHistoryEntry(
            query_id="q123",
            query_text="Different query",
            embedding=[1.0, 0.0, 0.0],  # Orthogonal to current
            timestamp=100.0,
            user_id="user_abc")

        mock_redis.zrangebyscore.return_value = [b"q123"]
        mock_redis.get.return_value = entry.model_dump_json()

        result = await history_tracker.find_similar_query(
            current_embedding=[0.0, 1.0, 0.0],  # Different direction
            user_id="user_abc",
            similarity_threshold=0.85)

        assert result is None

    async def test_find_similar_above_threshold(self, history_tracker, mock_redis):
        """Test find_similar_query with high similarity."""
        # Create similar embedding (normalized for cosine similarity)
        entry = QueryHistoryEntry(
            query_id="q123",
            query_text="Similar query",
            embedding=[0.9, 0.1, 0.0],
            timestamp=100.0,
            user_id="user_abc")

        mock_redis.zrangebyscore.return_value = [b"q123"]
        mock_redis.get.return_value = entry.model_dump_json()

        result = await history_tracker.find_similar_query(
            current_embedding=[0.95, 0.05, 0.0],  # Very similar
            user_id="user_abc",
            similarity_threshold=0.85)

        assert result is not None
        assert result.query_id == "q123"

    async def test_find_similar_best_match(self, history_tracker, mock_redis):
        """Test find_similar_query returns best match."""
        entry1 = QueryHistoryEntry(
            query_id="q123",
            query_text="Somewhat similar",
            embedding=[0.8, 0.2, 0.0],
            timestamp=100.0,
            user_id="user_abc")
        entry2 = QueryHistoryEntry(
            query_id="q456",
            query_text="Very similar",
            embedding=[0.95, 0.05, 0.0],
            timestamp=200.0,
            user_id="user_abc")

        mock_redis.zrangebyscore.return_value = [b"q123", b"q456"]
        mock_redis.get.side_effect = [
            entry2.model_dump_json(),
            entry1.model_dump_json(),
        ]

        result = await history_tracker.find_similar_query(
            current_embedding=[1.0, 0.0, 0.0],
            user_id="user_abc",
            similarity_threshold=0.7)

        assert result is not None
        assert result.query_id == "q456"  # Better match


class TestQueryHistoryTrackerClear:
    """Test QueryHistoryTracker.clear_user_history method."""

    async def test_clear_disabled(self):
        """Test clear when tracking is disabled."""
        tracker = QueryHistoryTracker(redis=None)
        result = await tracker.clear_user_history("user_abc")

        assert result is False

    async def test_clear_success(self, history_tracker, mock_redis):
        """Test clearing user history."""
        mock_redis.zrange.return_value = [b"q123", b"q456"]
        mock_redis.delete.return_value = 3

        result = await history_tracker.clear_user_history("user_abc")

        assert result is True
        mock_redis.delete.assert_called_once()

    async def test_clear_empty_history(self, history_tracker, mock_redis):
        """Test clearing empty history."""
        mock_redis.zrange.return_value = []

        result = await history_tracker.clear_user_history("user_abc")

        assert result is True
        # Delete is called with just the index key even for empty history
        mock_redis.delete.assert_called_once_with("conduit:history:user_abc:index")

    async def test_clear_connection_error(self, history_tracker, mock_redis):
        """Test clear handles Redis error gracefully."""
        mock_redis.zrange.side_effect = ConnectionError("Connection failed")

        result = await history_tracker.clear_user_history("user_abc")

        assert result is False


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        vec = [0.5, 0.5, 0.5]
        similarity = QueryHistoryTracker._cosine_similarity(vec, vec)

        # Identical normalized vectors should have similarity ~1.0
        assert similarity == pytest.approx(0.75, abs=0.01)

    def test_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = QueryHistoryTracker._cosine_similarity(vec1, vec2)

        # Orthogonal vectors should have similarity 0.0
        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_opposite_vectors(self):
        """Test cosine similarity with opposite vectors."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        similarity = QueryHistoryTracker._cosine_similarity(vec1, vec2)

        # Opposite vectors should have similarity -1.0
        assert similarity == pytest.approx(-1.0, abs=0.01)

    def test_similar_vectors(self):
        """Test cosine similarity with similar vectors."""
        vec1 = [0.9, 0.1]
        vec2 = [0.95, 0.05]
        similarity = QueryHistoryTracker._cosine_similarity(vec1, vec2)

        # Similar vectors should have high similarity
        assert similarity > 0.8
