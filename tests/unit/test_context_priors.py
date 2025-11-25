"""Unit tests for context-specific priors loading."""

import pytest
from pathlib import Path

from conduit.core.config import load_context_priors


class TestContextPriors:
    """Tests for context-specific priors loading."""

    def test_load_code_priors(self):
        """Test loading code-specific priors."""
        priors = load_context_priors("code")

        # Should have priors for code-optimized models
        assert isinstance(priors, dict)
        # Check that priors are Beta parameters (alpha, beta tuples)
        for model_id, params in priors.items():
            assert isinstance(params, tuple)
            assert len(params) == 2
            alpha, beta = params
            assert alpha > 0
            assert beta > 0
            # Quality should be in reasonable range (0.0-1.0)
            quality = alpha / (alpha + beta)
            assert 0.0 <= quality <= 1.0

    def test_load_creative_priors(self):
        """Test loading creative-specific priors."""
        priors = load_context_priors("creative")

        assert isinstance(priors, dict)
        for model_id, (alpha, beta) in priors.items():
            assert alpha > 0
            assert beta > 0

    def test_load_analysis_priors(self):
        """Test loading analysis-specific priors."""
        priors = load_context_priors("analysis")

        assert isinstance(priors, dict)
        for model_id, (alpha, beta) in priors.items():
            assert alpha > 0
            assert beta > 0

    def test_load_simple_qa_priors(self):
        """Test loading simple_qa-specific priors."""
        priors = load_context_priors("simple_qa")

        assert isinstance(priors, dict)
        for model_id, (alpha, beta) in priors.items():
            assert alpha > 0
            assert beta > 0

    def test_load_general_priors(self):
        """Test loading general priors."""
        priors = load_context_priors("general")

        assert isinstance(priors, dict)
        for model_id, (alpha, beta) in priors.items():
            assert alpha > 0
            assert beta > 0

    def test_load_unknown_context(self):
        """Test loading unknown context returns empty dict."""
        priors = load_context_priors("unknown_context")

        assert isinstance(priors, dict)
        assert len(priors) == 0

    def test_priors_quality_scores(self):
        """Test that priors convert to reasonable quality scores."""
        contexts = ["code", "creative", "analysis", "simple_qa", "general"]

        for context in contexts:
            priors = load_context_priors(context)
            for model_id, (alpha, beta) in priors.items():
                quality = alpha / (alpha + beta)
                # Quality should be between 0.5 and 1.0 (reasonable range)
                assert 0.5 <= quality <= 1.0, f"{context}/{model_id}: quality={quality:.3f}"

    def test_priors_strong_prior_strength(self):
        """Test that priors are strong (equivalent to ~10,000 samples)."""
        priors = load_context_priors("code")

        for model_id, (alpha, beta) in priors.items():
            total_samples = alpha + beta
            # Should be equivalent to ~10,000 samples (strong prior)
            assert total_samples >= 8000, f"{model_id}: total_samples={total_samples}"


