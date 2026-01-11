"""Edge case testing for production resilience.

Tests scenarios that could break routing in production:
- All models fail
- Embedding service failures
- Concurrent state updates
- Malformed/empty queries
- Invalid feedback values
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from conduit.core.models import Query, QueryFeatures
from conduit.engines import Router
from conduit.engines.bandits import LinUCBBandit
from conduit.engines.bandits.base import BanditFeedback, ModelArm


class TestQueryValidation:
    """Test malformed and invalid query inputs."""

    def test_empty_query_raises_validation_error(self):
        """Empty query text should raise ValidationError."""
        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            Query(text="")

    def test_whitespace_only_query_strips_to_valid(self):
        """Whitespace-only query should be stripped and validated."""
        # After stripping, becomes empty string, which fails min_length validation
        with pytest.raises(ValidationError):
            Query(text="   \n\t   ")

    def test_query_strips_whitespace(self):
        """Query should strip leading/trailing whitespace."""
        query = Query(text="  hello world  ")
        assert query.text == "hello world"

    @pytest.mark.asyncio
    async def test_extremely_long_query(self):
        """Router should handle extremely long queries without crashing."""
        router = Router()

        # Create 1M character query
        long_text = "a" * 1_000_000
        query = Query(text=long_text)

        # Should not crash, may truncate in embedding service
        decision = await router.route(query)

        assert decision is not None
        assert decision.selected_model in router.hybrid_router.models


class _ConfigurableEmbeddingProvider:
    """Test helper: embedding provider that can be configured to fail or succeed.

    Implements the EmbeddingProvider interface without inheriting from ABC
    to avoid import complexity in tests. Duck typing is sufficient here.
    """

    def __init__(self, should_fail: bool = True, dim: int = 384):
        self.should_fail = should_fail
        self._dimension = dim
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "configurable_test"

    async def embed(self, text: str) -> list[float]:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError("Embedding service unavailable")
        return [0.1] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class TestEmbeddingFailureHandling:
    """Test embedding failure fallback behavior."""

    @pytest.mark.asyncio
    async def test_analyzer_returns_zero_vector_on_embedding_failure(self):
        """Analyzer should return zero vector when embedding fails."""
        from conduit.engines.analyzer import QueryAnalyzer

        provider = _ConfigurableEmbeddingProvider(should_fail=True)
        analyzer = QueryAnalyzer(embedding_provider=provider)  # type: ignore[arg-type]
        features = await analyzer.analyze("Test query")

        # Should return zero vector and set embedding_failed flag
        assert features.embedding_failed is True
        assert features.embedding == [0.0] * provider.dimension
        assert len(features.embedding) == provider.dimension
        assert features.token_count > 0
        assert 0.0 <= features.complexity_score <= 1.0

    @pytest.mark.asyncio
    async def test_hybrid_router_phase2_falls_back_on_embedding_failure(self):
        """Router in phase2 should fall back to phase1 algorithm when embedding fails."""
        from conduit.engines.hybrid_router import HybridRouter
        from conduit.engines.analyzer import QueryAnalyzer

        provider = _ConfigurableEmbeddingProvider(should_fail=True)
        analyzer = QueryAnalyzer(embedding_provider=provider)  # type: ignore[arg-type]

        # Create router in phase2 (switch_threshold=0 means start in phase2)
        router = HybridRouter(
            models=["gpt-4o-mini", "gpt-4o"],
            switch_threshold=0,  # Start in phase2 immediately
            analyzer=analyzer,
            feature_dim=provider.dimension + 2,  # embedding + token_count + complexity
        )

        query = Query(text="Test query for embedding failure")
        decision = await router.route(query)

        # Should still return a valid decision
        assert decision is not None
        assert decision.selected_model in ["gpt-4o-mini", "gpt-4o"]

        # Should indicate embedding failure in metadata
        assert decision.metadata.get("embedding_failed") is True
        assert decision.metadata.get("fallback_reason") == "embedding_generation_failed"

        # Features should have embedding_failed flag
        assert decision.features.embedding_failed is True

    @pytest.mark.asyncio
    async def test_hybrid_router_phase1_handles_embedding_failure(self):
        """Router in phase1 should continue routing when embedding fails."""
        from conduit.engines.hybrid_router import HybridRouter
        from conduit.engines.analyzer import QueryAnalyzer

        provider = _ConfigurableEmbeddingProvider(should_fail=True)
        analyzer = QueryAnalyzer(embedding_provider=provider)  # type: ignore[arg-type]

        # Create router in phase1 (high threshold means stay in phase1)
        router = HybridRouter(
            models=["gpt-4o-mini", "gpt-4o"],
            switch_threshold=10000,  # Stay in phase1
            analyzer=analyzer,
            feature_dim=provider.dimension + 2,
        )

        query = Query(text="Test query for embedding failure in phase1")
        decision = await router.route(query)

        # Should still return a valid decision (phase1 doesn't need embeddings)
        assert decision is not None
        assert decision.selected_model in ["gpt-4o-mini", "gpt-4o"]

        # Features should have embedding_failed flag even in phase1
        assert decision.features.embedding_failed is True

    @pytest.mark.asyncio
    async def test_failed_embeddings_not_cached(self):
        """Failed embeddings should not be cached, allowing retry on recovery."""
        from conduit.engines.analyzer import QueryAnalyzer

        provider = _ConfigurableEmbeddingProvider(should_fail=True)
        analyzer = QueryAnalyzer(embedding_provider=provider)  # type: ignore[arg-type]

        # First call fails
        features1 = await analyzer.analyze("Test query")
        assert features1.embedding_failed is True
        assert provider.call_count == 1

        # Provider recovers
        provider.should_fail = False

        # Second call should retry (not use cached failure)
        features2 = await analyzer.analyze("Test query")
        assert features2.embedding_failed is False
        assert features2.embedding == [0.1] * provider.dimension
        assert provider.call_count == 2  # Called again, not cached


class TestFeedbackValidation:
    """Test invalid feedback value handling."""

    @pytest.mark.asyncio
    async def test_negative_cost_raises_validation_error(self):
        """Negative cost should raise ValidationError."""
        test_arms = [
            ModelArm(
                model_id="gpt-4o-mini",
                provider="openai",
                model_name="gpt-4o-mini",
                cost_per_input_token=0.00015,
                cost_per_output_token=0.0006)
        ]

        bandit = LinUCBBandit(test_arms)

        with pytest.raises(ValidationError):
            BanditFeedback(
                model_id="gpt-4o-mini",
                cost=-0.01,  # Invalid: negative cost
                quality_score=0.8,
                latency=1.0)

    @pytest.mark.asyncio
    async def test_quality_score_out_of_range(self):
        """Quality score outside [0.0, 1.0] should raise ValidationError."""
        with pytest.raises(ValidationError):
            BanditFeedback(
                model_id="gpt-4o-mini",
                cost=0.001,
                quality_score=1.5,  # Invalid: > 1.0
                latency=1.0)

        with pytest.raises(ValidationError):
            BanditFeedback(
                model_id="gpt-4o-mini",
                cost=0.001,
                quality_score=-0.2,  # Invalid: < 0.0
                latency=1.0)

    @pytest.mark.asyncio
    async def test_negative_latency_raises_validation_error(self):
        """Negative latency should raise ValidationError."""
        with pytest.raises(ValidationError):
            BanditFeedback(
                model_id="gpt-4o-mini",
                cost=0.001,
                quality_score=0.8,
                latency=-1.0,  # Invalid: negative latency
            )


class TestConcurrentUpdates:
    """Test concurrent state update scenarios.

    Note: Full race condition testing moved to tests/integration/test_concurrent_updates.py
    for real database testing. These tests verify basic thread safety.
    """

    @pytest.mark.asyncio
    async def test_concurrent_routing_requests(self):
        """Multiple concurrent route() calls should not crash or corrupt state."""
        router = Router()

        # Send 50 concurrent routing requests
        queries = [Query(text=f"Query {i}") for i in range(50)]
        tasks = [router.route(q) for q in queries]

        # All should complete successfully
        decisions = await asyncio.gather(*tasks)

        assert len(decisions) == 50
        assert all(d is not None for d in decisions)
        assert all(d.selected_model in router.hybrid_router.models for d in decisions)

    @pytest.mark.asyncio
    async def test_concurrent_feedback_updates(self, test_arms, test_features):
        """Concurrent feedback updates should maintain state consistency."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Create 100 concurrent feedback updates
        feedbacks = [
            BanditFeedback(
                model_id=test_arms[i % len(test_arms)].model_id,
                cost=0.001,
                quality_score=0.8,
                latency=1.0)
            for i in range(100)
        ]

        # Update concurrently
        tasks = [bandit.update(fb, test_features) for fb in feedbacks]
        await asyncio.gather(*tasks)

        # Verify A matrices are still positive definite (no corruption)
        for arm in test_arms:
            A = bandit.A[arm.model_id]

            # A should still be square matrix
            assert A.shape == (386, 386)

            # A should be symmetric (approximately, due to floating point)
            assert np.allclose(A, A.T, rtol=1e-10)

            # All eigenvalues should be positive (positive definite)
            eigenvalues = np.linalg.eigvalsh(A)
            assert np.all(eigenvalues > 0), f"Matrix not positive definite for {arm.model_id}"


class TestModelFailures:
    """Test scenarios where models fail or are unavailable."""

    @pytest.mark.asyncio
    async def test_routing_succeeds_when_no_llm_called(self):
        """Router should return decision even if no actual LLM is called.

        This tests the routing decision logic itself, not LLM execution.
        """
        router = Router()
        query = Query(text="Test query")

        # Route returns a decision (doesn't call LLM, that's executor's job)
        decision = await router.route(query)

        assert decision is not None
        assert decision.selected_model in router.hybrid_router.models
        assert 0.0 <= decision.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_single_model_always_selected(self):
        """With only one model, router should always select it."""
        router = Router(models=["gpt-4o-mini"])

        # Route 10 queries
        for i in range(10):
            query = Query(text=f"Query {i}")
            decision = await router.route(query)

            assert decision.selected_model == "gpt-4o-mini"


class TestStateConsistency:
    """Test state consistency under various conditions."""

    @pytest.mark.asyncio
    async def test_bandit_state_after_many_updates(self, test_arms, test_features):
        """After many updates, bandit state should remain valid."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Perform 1000 updates
        for i in range(1000):
            arm = test_arms[i % len(test_arms)]
            feedback = BanditFeedback(
                model_id=arm.model_id,
                cost=0.001 * (i % 10),
                quality_score=0.5 + (i % 5) * 0.1,
                latency=1.0 + (i % 3) * 0.5)
            await bandit.update(feedback, test_features)

        # State should still be valid
        for arm in test_arms:
            A = bandit.A[arm.model_id]
            b = bandit.b[arm.model_id]

            # A should be positive definite
            eigenvalues = np.linalg.eigvalsh(A)
            assert np.all(eigenvalues > 0)

            # b should not be NaN or Inf
            assert not np.any(np.isnan(b))
            assert not np.any(np.isinf(b))

            # Pull count should match updates
            expected_pulls = sum(1 for j in range(1000) if j % len(test_arms) == test_arms.index(arm))
            assert bandit.arm_pulls[arm.model_id] == expected_pulls

    @pytest.mark.asyncio
    async def test_zero_feature_vector_handling(self, test_arms):
        """Bandit should handle zero feature vector without crashing."""
        bandit = LinUCBBandit(test_arms, feature_dim=386)

        # Create zero feature vector
        zero_features = QueryFeatures(
            embedding=[0.0] * 384,
            token_count=0,
            complexity_score=0.0
        )

        # Should not crash on selection
        arm = await bandit.select_arm(zero_features)
        assert arm in test_arms

        # Should not crash on update
        feedback = BanditFeedback(
            model_id=arm.model_id,
            cost=0.001,
            quality_score=0.8,
            latency=1.0)
        await bandit.update(feedback, zero_features)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_arms() -> list[ModelArm]:
    """Standard test arms for edge case tests."""
    return [
        ModelArm(
            model_id="o4-mini",
            model_name="o4-mini",
            provider="openai",
            cost_per_input_token=0.00011,
            cost_per_output_token=0.00044,
            expected_quality=0.7),
        ModelArm(
            model_id="gpt-5.1",
            model_name="gpt-5.1",
            provider="openai",
            cost_per_input_token=0.002,
            cost_per_output_token=0.008,
            expected_quality=0.9),
        ModelArm(
            model_id="claude-haiku-4-5-20241124",
            model_name="claude-haiku-4-5",
            provider="anthropic",
            cost_per_input_token=0.0008,
            cost_per_output_token=0.004,
            expected_quality=0.75),
    ]


@pytest.fixture
def test_features() -> QueryFeatures:
    """Standard test features for edge case tests."""
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5
    )
