"""Unit tests for conduit.core.pricing module.

Tests LiteLLM-based pricing lookup and cost computation.
"""

import pytest
from datetime import datetime, timezone

from conduit.core.pricing import (
    ModelPricing,
    get_model_pricing,
    get_all_model_pricing,
    compute_cost,
)


class TestModelPricing:
    """Tests for ModelPricing dataclass."""

    def test_model_pricing_basic(self):
        """Test basic ModelPricing creation."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        )
        assert pricing.model_id == "test-model"
        assert pricing.input_cost_per_million == 3.0
        assert pricing.output_cost_per_million == 15.0

    def test_model_pricing_computed_fields(self):
        """Test computed per-token costs."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        )
        assert pricing.input_cost_per_token == 3.0 / 1_000_000
        assert pricing.output_cost_per_token == 15.0 / 1_000_000

    def test_model_pricing_with_cache_costs(self):
        """Test ModelPricing with cache costs."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
            cached_input_cost_per_million=0.3,
            cache_creation_cost_per_million=3.75,
        )
        assert pricing.cached_input_cost_per_token == 0.3 / 1_000_000
        assert pricing.cache_creation_cost_per_million == 3.75

    def test_model_pricing_none_cache_cost(self):
        """Test cached_input_cost_per_token returns None when not set."""
        pricing = ModelPricing(
            model_id="test-model",
            input_cost_per_million=3.0,
            output_cost_per_million=15.0,
        )
        assert pricing.cached_input_cost_per_token is None


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_get_known_model_pricing(self):
        """Test retrieving pricing for a known model."""
        # gpt-4o-mini should always be in LiteLLM
        pricing = get_model_pricing("gpt-4o-mini")
        assert pricing is not None
        assert pricing.model_id == "gpt-4o-mini"
        assert pricing.input_cost_per_million > 0
        assert pricing.output_cost_per_million > 0
        assert pricing.source == "litellm"
        assert pricing.snapshot_at is not None

    def test_get_claude_model_pricing(self):
        """Test retrieving pricing for Claude model."""
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing is not None
        assert pricing.model_id == "claude-3-5-sonnet-20241022"
        assert pricing.input_cost_per_million > 0
        assert pricing.output_cost_per_million > 0

    def test_get_unknown_model_returns_none(self):
        """Test that unknown models return None."""
        pricing = get_model_pricing("unknown-model-xyz-123")
        assert pricing is None

    def test_get_model_with_cache_pricing(self):
        """Test model with cache pricing support."""
        # Claude models typically have cache pricing
        pricing = get_model_pricing("claude-sonnet-4-5-20250929")
        if pricing is not None:
            # Cache pricing may or may not be available
            # Just verify the structure is correct
            assert pricing.cached_input_cost_per_million is None or pricing.cached_input_cost_per_million >= 0


class TestGetAllModelPricing:
    """Tests for get_all_model_pricing function."""

    def test_get_all_model_pricing_returns_dict(self):
        """Test that get_all_model_pricing returns a dict."""
        all_pricing = get_all_model_pricing()
        assert isinstance(all_pricing, dict)
        assert len(all_pricing) > 100  # Should have many models

    def test_get_all_model_pricing_contains_known_models(self):
        """Test that known models are included."""
        all_pricing = get_all_model_pricing()
        # At least some common models should be present
        known_models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]
        found = sum(1 for m in known_models if m in all_pricing)
        assert found >= 2  # At least 2 of 3 should be present

    def test_get_all_model_pricing_values_are_model_pricing(self):
        """Test that all values are ModelPricing instances."""
        all_pricing = get_all_model_pricing()
        for model_id, pricing in list(all_pricing.items())[:10]:
            assert isinstance(pricing, ModelPricing)
            assert pricing.model_id == model_id


class TestComputeCost:
    """Tests for compute_cost function."""

    def test_compute_cost_basic(self):
        """Test basic cost computation."""
        cost = compute_cost(
            input_tokens=1000,
            output_tokens=500,
            model_id="gpt-4o-mini",
        )
        assert cost > 0
        assert isinstance(cost, float)

    def test_compute_cost_unknown_model_uses_conservative_fallback(self):
        """Test that unknown models use conservative fallback pricing."""
        cost = compute_cost(
            input_tokens=1000,
            output_tokens=500,
            model_id="unknown-model-xyz",
        )
        # Conservative fallback: $10/1M input, $30/1M output (GPT-4 tier)
        # 1000 * $10/1M + 500 * $30/1M = $0.01 + $0.015 = $0.025
        assert cost == 0.025

    def test_compute_cost_zero_tokens(self):
        """Test cost with zero tokens."""
        cost = compute_cost(
            input_tokens=0,
            output_tokens=0,
            model_id="gpt-4o-mini",
        )
        assert cost == 0.0

    def test_compute_cost_with_cache_read_tokens(self):
        """Test cost computation with cache read tokens."""
        # Get a model with cache pricing
        pricing = get_model_pricing("claude-sonnet-4-5-20250929")
        if pricing is not None and pricing.cached_input_cost_per_million is not None:
            # Cost with cache should be different than without
            cost_no_cache = compute_cost(
                input_tokens=1000,
                output_tokens=500,
                model_id="claude-sonnet-4-5-20250929",
            )
            cost_with_cache = compute_cost(
                input_tokens=500,
                output_tokens=500,
                model_id="claude-sonnet-4-5-20250929",
                cache_read_tokens=500,
            )
            # Cache read is cheaper than input, so total should be less
            # (500 input + 500 cache) should cost less than 1000 input
            assert cost_with_cache < cost_no_cache

    def test_compute_cost_proportional_to_tokens(self):
        """Test that cost scales with token count."""
        cost_small = compute_cost(
            input_tokens=100,
            output_tokens=50,
            model_id="gpt-4o-mini",
        )
        cost_large = compute_cost(
            input_tokens=1000,
            output_tokens=500,
            model_id="gpt-4o-mini",
        )
        # 10x tokens should be 10x cost
        assert abs(cost_large / cost_small - 10.0) < 0.01


class TestPricingIntegration:
    """Integration tests for pricing with real LiteLLM data."""

    def test_claude_models_have_pricing(self):
        """Test that Claude models have pricing data."""
        claude_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
        ]
        for model in claude_models:
            pricing = get_model_pricing(model)
            assert pricing is not None, f"No pricing for {model}"
            assert pricing.input_cost_per_million > 0

    def test_openai_models_have_pricing(self):
        """Test that OpenAI models have pricing data."""
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
        ]
        for model in openai_models:
            pricing = get_model_pricing(model)
            assert pricing is not None, f"No pricing for {model}"
            assert pricing.input_cost_per_million > 0

    def test_pricing_relative_order(self):
        """Test that pricing follows expected relative order."""
        # gpt-4o-mini should be cheaper than gpt-4o
        mini = get_model_pricing("gpt-4o-mini")
        full = get_model_pricing("gpt-4o")
        if mini and full:
            assert mini.input_cost_per_million < full.input_cost_per_million

        # claude-3-haiku should be cheaper than claude-3-opus
        haiku = get_model_pricing("claude-3-haiku-20240307")
        opus = get_model_pricing("claude-3-opus-20240229")
        if haiku and opus:
            assert haiku.input_cost_per_million < opus.input_cost_per_million
