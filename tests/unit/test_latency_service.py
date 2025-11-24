"""Tests for LatencyService historical latency tracking."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from conduit.core.latency import LatencyService
from conduit.core.models import QueryFeatures
from conduit.core.exceptions import DatabaseError


@pytest.fixture
def mock_pool():
    """Create mock asyncpg pool."""
    pool = MagicMock()
    pool.acquire = MagicMock()
    return pool


@pytest.fixture
def latency_service(mock_pool):
    """Create LatencyService with mock pool."""
    return LatencyService(
        pool=mock_pool,
        window_days=7,
        percentile=0.95,
        min_samples=100,
    )


@pytest.fixture
def query_features():
    """Create sample query features."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )


class TestLatencyServiceInit:
    """Test LatencyService initialization."""

    def test_init_default_params(self, mock_pool):
        """Test initialization with default parameters."""
        service = LatencyService(pool=mock_pool)
        assert service.pool == mock_pool
        assert service.window_days == 7
        assert service.percentile == 0.95
        assert service.min_samples == 100

    def test_init_custom_params(self, mock_pool):
        """Test initialization with custom parameters."""
        service = LatencyService(
            pool=mock_pool,
            window_days=14,
            percentile=0.99,
            min_samples=200,
        )
        assert service.window_days == 14
        assert service.percentile == 0.99
        assert service.min_samples == 200


class TestRecordLatency:
    """Test latency recording."""

    @pytest.mark.asyncio
    async def test_record_latency_success(self, latency_service, query_features):
        """Test successful latency recording."""
        # Setup mock connection
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Record latency
        await latency_service.record_latency(
            model_id="gpt-4o-mini",
            latency=1.23,
            query_features=query_features,
        )

        # Verify execute was called with correct parameters
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO model_latencies" in call_args[0]
        assert call_args[1] == "gpt-4o-mini"
        assert call_args[2] == 1.23
        assert call_args[3] == 50  # token_count
        assert call_args[4] == 0.5  # complexity_score

    @pytest.mark.asyncio
    async def test_record_latency_without_features(self, latency_service):
        """Test latency recording without query features."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        await latency_service.record_latency(
            model_id="gpt-4o-mini",
            latency=2.34,
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert call_args[1] == "gpt-4o-mini"
        assert call_args[2] == 2.34
        assert call_args[3] is None  # token_count
        assert call_args[4] is None  # complexity_score

    @pytest.mark.asyncio
    async def test_record_latency_with_custom_timestamp(self, latency_service):
        """Test latency recording with custom timestamp."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        await latency_service.record_latency(
            model_id="gpt-4o-mini",
            latency=1.5,
            timestamp=custom_time,
        )

        call_args = mock_conn.execute.call_args[0]
        assert call_args[5] == custom_time

    @pytest.mark.asyncio
    async def test_record_latency_database_error(self, latency_service):
        """Test latency recording with database error."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB error"))
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(DatabaseError, match="Failed to record latency"):
            await latency_service.record_latency(
                model_id="gpt-4o-mini",
                latency=1.0,
            )


class TestGetEstimatedLatency:
    """Test latency estimation."""

    @pytest.mark.asyncio
    async def test_estimate_with_sufficient_data(self, latency_service):
        """Test estimation when sufficient historical data exists."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "sample_count": 150,
                "percentile_latency": 1.85,
            }
        )
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        estimate = await latency_service.get_estimated_latency("gpt-4o-mini")

        assert estimate == 1.85
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimate_with_insufficient_data_uses_heuristic(self, latency_service):
        """Test fallback to heuristic when insufficient data."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "sample_count": 50,  # Less than min_samples (100)
                "percentile_latency": 1.5,
            }
        )
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Should fall back to heuristic
        estimate = await latency_service.get_estimated_latency("gpt-4o-mini")

        # OpenAI baseline is 1.5, mini models get 0.7x multiplier = 1.05
        assert 1.0 <= estimate <= 1.2

    @pytest.mark.asyncio
    async def test_estimate_with_no_data_uses_heuristic(self, latency_service):
        """Test heuristic fallback when no historical data."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "sample_count": 0,
                "percentile_latency": None,
            }
        )
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        estimate = await latency_service.get_estimated_latency("claude-3-opus")

        # Anthropic baseline is 1.8, opus gets 1.3x multiplier = 2.34
        assert 2.0 <= estimate <= 2.5

    @pytest.mark.asyncio
    async def test_estimate_handles_database_error(self, latency_service):
        """Test estimation falls back to heuristic on database error."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=Exception("DB error"))
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        # Should fall back to heuristic without raising
        estimate = await latency_service.get_estimated_latency("gpt-4o-mini")

        assert estimate > 0  # Should return valid estimate


class TestGetLatencyStats:
    """Test latency statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, latency_service):
        """Test statistics retrieval with available data."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "sample_count": 1000,
                "p50": 1.2,
                "p95": 2.5,
                "p99": 3.8,
                "mean": 1.5,
                "min": 0.5,
                "max": 5.0,
            }
        )
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        stats = await latency_service.get_latency_stats("gpt-4o-mini")

        assert stats["sample_count"] == 1000
        assert stats["p50"] == 1.2
        assert stats["p95"] == 2.5
        assert stats["p99"] == 3.8
        assert stats["mean"] == 1.5
        assert stats["min"] == 0.5
        assert stats["max"] == 5.0
        assert stats["window_days"] == 7

    @pytest.mark.asyncio
    async def test_get_stats_with_no_data(self, latency_service):
        """Test statistics retrieval with no data."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(
            return_value={
                "sample_count": 0,
                "p50": None,
                "p95": None,
                "p99": None,
                "mean": None,
                "min": None,
                "max": None,
            }
        )
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        stats = await latency_service.get_latency_stats("unknown-model")

        assert stats["sample_count"] == 0
        assert stats["p50"] is None
        assert stats["p95"] is None
        assert stats["p99"] is None
        assert stats["mean"] is None
        assert stats["min"] is None
        assert stats["max"] is None

    @pytest.mark.asyncio
    async def test_get_stats_database_error(self, latency_service):
        """Test statistics retrieval with database error."""
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(side_effect=Exception("DB error"))
        latency_service.pool.acquire.return_value.__aenter__.return_value = mock_conn

        with pytest.raises(DatabaseError, match="Failed to get latency stats"):
            await latency_service.get_latency_stats("gpt-4o-mini")


class TestLatencyHeuristics:
    """Test heuristic-based latency estimation (fallback)."""

    def test_heuristic_openai_models(self, latency_service):
        """Test heuristic for OpenAI models."""
        # GPT-4o (premium)
        estimate = latency_service._estimate_latency_heuristic("gpt-4o")
        assert 1.5 <= estimate <= 2.5  # 1.5 base * 1.3 premium

        # GPT-4o-mini (fast)
        estimate = latency_service._estimate_latency_heuristic("gpt-4o-mini")
        assert 0.8 <= estimate <= 1.3  # 1.5 base * 0.7 mini

    def test_heuristic_anthropic_models(self, latency_service):
        """Test heuristic for Anthropic models."""
        # Claude Opus (premium)
        estimate = latency_service._estimate_latency_heuristic("claude-3-opus")
        assert 2.0 <= estimate <= 2.8  # 1.8 base * 1.3 premium

        # Claude Haiku (fast)
        estimate = latency_service._estimate_latency_heuristic("claude-3-haiku")
        assert 1.0 <= estimate <= 1.6  # 1.8 base * 0.7 mini

    def test_heuristic_google_models(self, latency_service):
        """Test heuristic for Google models."""
        estimate = latency_service._estimate_latency_heuristic("gemini-pro")
        assert 1.0 <= estimate <= 1.5

    def test_heuristic_groq_models(self, latency_service):
        """Test heuristic for Groq models."""
        estimate = latency_service._estimate_latency_heuristic("llama-3-groq")
        assert 0.3 <= estimate <= 0.8

    def test_heuristic_unknown_model(self, latency_service):
        """Test heuristic for unknown model uses default."""
        estimate = latency_service._estimate_latency_heuristic("unknown-model-xyz")
        assert estimate == 2.0  # Default baseline


class TestProviderExtraction:
    """Test provider extraction from model IDs."""

    def test_extract_openai_provider(self, latency_service):
        """Test OpenAI provider extraction."""
        assert latency_service._extract_provider("gpt-4o-mini") == "openai"
        assert latency_service._extract_provider("gpt-4o") == "openai"
        assert latency_service._extract_provider("o1-preview") == "openai"

    def test_extract_anthropic_provider(self, latency_service):
        """Test Anthropic provider extraction."""
        assert latency_service._extract_provider("claude-3-opus") == "anthropic"
        assert latency_service._extract_provider("claude-3-haiku") == "anthropic"

    def test_extract_google_provider(self, latency_service):
        """Test Google provider extraction."""
        assert latency_service._extract_provider("gemini-pro") == "google"
        assert latency_service._extract_provider("palm-2") == "google"

    def test_extract_groq_provider(self, latency_service):
        """Test Groq provider extraction."""
        assert latency_service._extract_provider("llama-3-groq") == "groq"

    def test_extract_unknown_provider(self, latency_service):
        """Test unknown provider extraction."""
        assert latency_service._extract_provider("random-model") == "unknown"
