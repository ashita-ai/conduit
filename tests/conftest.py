"""Pytest configuration and fixtures for Conduit tests."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.engines.bandits.base import ModelArm
from conduit.core.models import QueryFeatures

# Note: sentence-transformers and scikit-learn are required dependencies,
# so we don't mock them. Tests will use the real libraries.

# Only mock numpy if it's not actually installed
# (LinUCB needs real numpy for matrix operations)
try:
    import numpy as np  # noqa: F401

    # numpy is available, don't mock it
except ImportError:
    # Mock numpy with proper bool_ type for isinstance checks
    numpy_mock = MagicMock()
    numpy_mock.random = MagicMock()
    numpy_mock.random.beta = MagicMock(return_value=0.5)
    # Create a proper type for bool_ to support isinstance checks
    numpy_mock.bool_ = type("bool_", (object,), {})
    sys.modules["numpy"] = numpy_mock

# sklearn is a required dependency (scikit-learn>=1.3.0), don't mock it
# Tests that need sklearn will import it normally


@pytest.fixture(autouse=True)
def mock_huggingface_embeddings():
    """Mock HuggingFace embedding provider to avoid real API calls in tests.

    HuggingFace deprecated api-inference.huggingface.co and now requires
    authentication for router.huggingface.co. Unit tests should not make
    real API calls, so we mock the embedding provider globally.
    """
    # Create fake 384-dim embedding (all-MiniLM-L6-v2 dimension)
    fake_embedding = [0.1] * 384

    async def mock_embed(self, text: str) -> list[float]:
        return fake_embedding

    async def mock_embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [fake_embedding for _ in texts]

    with patch('conduit.engines.embeddings.huggingface.HuggingFaceEmbeddingProvider.embed', new=mock_embed), \
         patch('conduit.engines.embeddings.huggingface.HuggingFaceEmbeddingProvider.embed_batch', new=mock_embed_batch):
        yield


# =============================================================================
# SHARED BANDIT TEST FIXTURES
# =============================================================================
# These fixtures are shared across all bandit algorithm tests to ensure
# consistency and reduce duplication. Update models here when defaults change.


@pytest.fixture
def test_arms() -> list[ModelArm]:
    """Create canonical test model arms for bandit tests.

    Returns three arms representing different cost/quality tradeoffs:
    - o4-mini: Fast, cheap OpenAI model
    - gpt-5.1: Premium OpenAI flagship
    - claude-haiku-4-5: Cheap Anthropic model

    Used by: test_bandits_linucb.py, test_bandits_ucb.py, test_bandits_thompson.py,
             test_bandits_dueling.py, test_bandits_baselines.py,
             test_bandits_epsilon_greedy.py, test_bandits_contextual_thompson_sampling.py,
             test_bandits_non_stationary.py
    """
    return [
        ModelArm(
            model_id="o4-mini",
            model_name="o4-mini",
            provider="openai",
            cost_per_input_token=0.0000011,  # $1.10/1M tokens
            cost_per_output_token=0.0000044,  # $4.40/1M tokens
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="gpt-5.1",
            model_name="gpt-5.1",
            provider="openai",
            cost_per_input_token=0.000002,  # $2.00/1M tokens
            cost_per_output_token=0.000008,  # $8.00/1M tokens
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-haiku-4-5",
            model_name="claude-haiku-4-5-20241124",
            provider="anthropic",
            cost_per_input_token=0.0000008,  # $0.80/1M tokens
            cost_per_output_token=0.000004,  # $4.00/1M tokens
            expected_quality=0.80,
        ),
    ]


@pytest.fixture
def test_features() -> QueryFeatures:
    """Create canonical test query features for bandit tests.

    Returns a QueryFeatures instance with:
    - 384-dim embedding (all-MiniLM-L6-v2 compatible)
    - Moderate token count (50)
    - Medium complexity (0.5)
    - General domain with reasonable confidence

    Used by: All bandit algorithm test files
    """
    return QueryFeatures(
        embedding=[0.1] * 384,
        token_count=50,
        complexity_score=0.5,
        domain="general",
        domain_confidence=0.8,
    )
