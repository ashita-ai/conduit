"""Pytest configuration and fixtures for Conduit tests."""

import sys
from unittest.mock import MagicMock

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
