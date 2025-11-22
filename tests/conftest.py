"""Pytest configuration and fixtures for Conduit tests."""

import sys
from unittest.mock import MagicMock

# Mock ML libraries that may not be installed (require Fortran compilers)
# This allows tests to run even if scipy, scikit-learn, or sentence-transformers aren't available


class MockEmbedding:
    """Mock numpy array with tolist() method."""

    def __init__(self, data: list[float]):
        self._data = data

    def tolist(self) -> list[float]:
        """Convert to list (mimics numpy array behavior)."""
        return self._data


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing without the library installed."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, text: str, convert_to_tensor: bool = False):
        """Return mock embedding."""
        # Return a mock embedding vector of dimension 384 with tolist() method
        return MockEmbedding([0.1] * 384)


# Mock the problematic modules before they're imported
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["sentence_transformers"].SentenceTransformer = MockSentenceTransformer

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

# Mock sklearn
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.feature_extraction"] = MagicMock()
sys.modules["sklearn.feature_extraction.text"] = MagicMock()
