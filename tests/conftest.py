"""Pytest configuration and fixtures for Conduit tests."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
