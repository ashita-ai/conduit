"""Unit tests for conduit.evaluation.__init__ module.

Tests for lazy import functionality.
"""

import pytest


class TestEvaluationLazyImport:
    """Tests for lazy import of ArbiterEvaluator."""

    def test_getattr_returns_arbiter_evaluator(self):
        """Test __getattr__ returns ArbiterEvaluator when it's available."""
        import conduit.evaluation as evaluation_module

        # ArbiterEvaluator should be accessible via attribute access
        # This will use __getattr__ since TYPE_CHECKING is False at runtime
        ArbiterEvaluator = getattr(evaluation_module, "ArbiterEvaluator")

        # Should be the actual class
        assert ArbiterEvaluator.__name__ == "ArbiterEvaluator"

    def test_getattr_raises_attribute_error_for_unknown(self):
        """Test __getattr__ raises AttributeError for unknown attributes."""
        import conduit.evaluation as evaluation_module

        with pytest.raises(AttributeError, match="has no attribute 'UnknownClass'"):
            getattr(evaluation_module, "UnknownClass")

    def test_all_contains_arbiter_evaluator(self):
        """Test __all__ contains ArbiterEvaluator."""
        import conduit.evaluation as evaluation_module

        assert "ArbiterEvaluator" in evaluation_module.__all__

    def test_direct_import_from_module(self):
        """Test direct import from conduit.evaluation works."""
        # This uses the lazy import mechanism
        from conduit.evaluation import ArbiterEvaluator

        assert ArbiterEvaluator.__name__ == "ArbiterEvaluator"

    def test_multiple_getattr_calls_return_same_class(self):
        """Test multiple calls to __getattr__ return the same class."""
        import conduit.evaluation as evaluation_module

        class1 = getattr(evaluation_module, "ArbiterEvaluator")
        class2 = getattr(evaluation_module, "ArbiterEvaluator")

        assert class1 is class2
