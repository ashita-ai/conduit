"""Tests for cost-based model filtering."""

import pytest

from conduit.engines.bandits.base import ModelArm
from conduit.engines.cost_filter import CostEstimate, CostFilter, FilterResult


@pytest.fixture
def sample_arms() -> list[ModelArm]:
    """Create sample model arms with different costs."""
    return [
        ModelArm(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.15,  # $0.15 per 1K input tokens
            cost_per_output_token=0.60,  # $0.60 per 1K output tokens
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            cost_per_input_token=2.50,  # $2.50 per 1K input tokens
            cost_per_output_token=10.00,  # $10.00 per 1K output tokens
            expected_quality=0.95,
        ),
        ModelArm(
            model_id="claude-3-haiku",
            provider="anthropic",
            model_name="claude-3-haiku",
            cost_per_input_token=0.25,  # $0.25 per 1K input tokens
            cost_per_output_token=1.25,  # $1.25 per 1K output tokens
            expected_quality=0.80,
        ),
    ]


@pytest.fixture
def cost_filter() -> CostFilter:
    """Create default cost filter."""
    return CostFilter(output_ratio=1.0, fallback_on_empty=True)


class TestCostFilterInit:
    """Tests for CostFilter initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        filter = CostFilter()
        assert filter.output_ratio == 1.0
        assert filter.fallback_on_empty is True

    def test_custom_output_ratio(self) -> None:
        """Test custom output ratio."""
        filter = CostFilter(output_ratio=2.0)
        assert filter.output_ratio == 2.0

    def test_fallback_disabled(self) -> None:
        """Test fallback disabled."""
        filter = CostFilter(fallback_on_empty=False)
        assert filter.fallback_on_empty is False


class TestEstimateTokens:
    """Tests for token estimation using tiktoken."""

    def test_simple_text(self, cost_filter: CostFilter) -> None:
        """Test token estimation for simple text."""
        tokens = cost_filter.estimate_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Simple text should be few tokens

    def test_empty_text(self, cost_filter: CostFilter) -> None:
        """Test token estimation for empty text."""
        tokens = cost_filter.estimate_tokens("")
        assert tokens == 0

    def test_longer_text(self, cost_filter: CostFilter) -> None:
        """Test token estimation for longer text."""
        text = "This is a longer sentence with more words to tokenize. " * 10
        tokens = cost_filter.estimate_tokens(text)
        assert tokens > 50  # Longer text should have more tokens

    def test_encoding_lazy_init(self) -> None:
        """Test that encoding is lazily initialized."""
        filter = CostFilter()
        assert filter._encoding is None
        # Access encoding property
        _ = filter.encoding
        assert filter._encoding is not None


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_cost_basic(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test basic cost estimation."""
        arm = sample_arms[0]  # gpt-4o-mini
        estimate = cost_filter.estimate_cost(arm, input_tokens=1000)

        assert isinstance(estimate, CostEstimate)
        assert estimate.model_id == "gpt-4o-mini"
        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 1000  # output_ratio=1.0
        assert estimate.total_cost > 0

    def test_estimate_cost_with_output_ratio(self, sample_arms: list[ModelArm]) -> None:
        """Test cost estimation with custom output ratio."""
        filter = CostFilter(output_ratio=2.0)
        arm = sample_arms[0]
        estimate = filter.estimate_cost(arm, input_tokens=1000)

        assert estimate.input_tokens == 1000
        assert estimate.output_tokens == 2000  # 2x output ratio

    def test_estimate_cost_zero_tokens(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test cost estimation with zero tokens."""
        arm = sample_arms[0]
        estimate = cost_filter.estimate_cost(arm, input_tokens=0)

        assert estimate.input_tokens == 0
        assert estimate.output_tokens == 0
        assert estimate.total_cost == 0.0


class TestFilterByBudget:
    """Tests for budget-based filtering."""

    def test_filter_all_within_budget(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test filtering when all models are within budget."""
        result = cost_filter.filter_by_budget(
            models=sample_arms,
            max_cost=100.0,  # Very high budget
            query_text="Hello",
        )

        assert isinstance(result, FilterResult)
        assert len(result.models) == len(sample_arms)
        assert result.was_relaxed is False
        assert result.original_count == len(sample_arms)
        assert result.filtered_count == len(sample_arms)

    def test_filter_some_within_budget(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test filtering when only some models are within budget."""
        # Use models with known different costs to ensure filtering
        arms = [
            ModelArm(
                model_id="cheap-model",
                provider="test",
                model_name="cheap-model",
                cost_per_input_token=0.001,
                cost_per_output_token=0.001,
                expected_quality=0.7,
            ),
            ModelArm(
                model_id="expensive-model",
                provider="test",
                model_name="expensive-model",
                cost_per_input_token=100.0,  # Very expensive
                cost_per_output_token=100.0,
                expected_quality=0.9,
            ),
        ]
        # Budget that allows cheap but not expensive
        result = cost_filter.filter_by_budget(
            models=arms,
            max_cost=0.01,
            query_text="Hi",
        )

        assert len(result.models) == 1
        assert result.was_relaxed is False
        assert result.models[0].model_id == "cheap-model"

    def test_filter_none_within_budget_with_fallback(self) -> None:
        """Test filtering when no models are within budget (fallback enabled)."""
        # Use expensive test models that won't be in LiteLLM
        arms = [
            ModelArm(
                model_id="expensive-test-1",
                provider="test",
                model_name="expensive-test-1",
                cost_per_input_token=1000.0,
                cost_per_output_token=1000.0,
                expected_quality=0.9,
            ),
            ModelArm(
                model_id="expensive-test-2",
                provider="test",
                model_name="expensive-test-2",
                cost_per_input_token=2000.0,
                cost_per_output_token=2000.0,
                expected_quality=0.95,
            ),
        ]
        filter = CostFilter(fallback_on_empty=True)
        result = filter.filter_by_budget(
            models=arms,
            max_cost=0.01,  # Budget too low for these expensive models
            query_text="Hello",
        )

        assert len(result.models) == 1  # Falls back to cheapest
        assert result.was_relaxed is True
        assert result.cheapest_model == "expensive-test-1"

    def test_filter_none_within_budget_no_fallback(self) -> None:
        """Test filtering when no models are within budget (fallback disabled)."""
        # Use expensive test models that won't be in LiteLLM
        arms = [
            ModelArm(
                model_id="expensive-test-1",
                provider="test",
                model_name="expensive-test-1",
                cost_per_input_token=1000.0,
                cost_per_output_token=1000.0,
                expected_quality=0.9,
            ),
        ]
        filter = CostFilter(fallback_on_empty=False)

        with pytest.raises(ValueError) as exc_info:
            filter.filter_by_budget(
                models=arms,
                max_cost=0.01,  # Budget too low
                query_text="Hello",
            )

        assert "No models fit budget" in str(exc_info.value)

    def test_filter_empty_models_list(self, cost_filter: CostFilter) -> None:
        """Test filtering with empty models list."""
        result = cost_filter.filter_by_budget(
            models=[],
            max_cost=10.0,
            query_text="Hello",
        )

        assert result.models == []
        assert result.was_relaxed is False
        assert result.original_count == 0
        assert result.filtered_count == 0
        assert result.cheapest_model is None

    def test_filter_result_metadata(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test filter result metadata."""
        result = cost_filter.filter_by_budget(
            models=sample_arms,
            max_cost=0.01,
            query_text="Hello",
        )

        assert result.budget == 0.01
        assert result.original_count == len(sample_arms)


class TestGetCostBreakdown:
    """Tests for cost breakdown utility."""

    def test_get_cost_breakdown(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test getting cost breakdown for all models."""
        breakdown = cost_filter.get_cost_breakdown(
            models=sample_arms,
            query_text="Hello",
        )

        assert len(breakdown) == len(sample_arms)
        assert all(isinstance(e, CostEstimate) for e in breakdown)

    def test_cost_breakdown_sorted(
        self, cost_filter: CostFilter, sample_arms: list[ModelArm]
    ) -> None:
        """Test that cost breakdown is sorted by total cost."""
        breakdown = cost_filter.get_cost_breakdown(
            models=sample_arms,
            query_text="Hello, this is a test query.",
        )

        costs = [e.total_cost for e in breakdown]
        assert costs == sorted(costs)  # Should be ascending

    def test_cost_breakdown_empty_models(self, cost_filter: CostFilter) -> None:
        """Test cost breakdown with empty models list."""
        breakdown = cost_filter.get_cost_breakdown(models=[], query_text="Hello")
        assert breakdown == []


class TestCostEstimateDataclass:
    """Tests for CostEstimate dataclass."""

    def test_cost_estimate_fields(self) -> None:
        """Test CostEstimate fields."""
        estimate = CostEstimate(
            model_id="test-model",
            input_tokens=100,
            output_tokens=200,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
        )

        assert estimate.model_id == "test-model"
        assert estimate.input_tokens == 100
        assert estimate.output_tokens == 200
        assert estimate.input_cost == 0.01
        assert estimate.output_cost == 0.02
        assert estimate.total_cost == 0.03


class TestFilterResultDataclass:
    """Tests for FilterResult dataclass."""

    def test_filter_result_fields(self, sample_arms: list[ModelArm]) -> None:
        """Test FilterResult fields."""
        result = FilterResult(
            models=sample_arms[:2],
            was_relaxed=False,
            original_count=3,
            filtered_count=2,
            budget=0.05,
            cheapest_model="gpt-4o-mini",
        )

        assert len(result.models) == 2
        assert result.was_relaxed is False
        assert result.original_count == 3
        assert result.filtered_count == 2
        assert result.budget == 0.05
        assert result.cheapest_model == "gpt-4o-mini"


class TestIntegrationWithPricing:
    """Integration tests with LiteLLM pricing."""

    def test_uses_litellm_pricing_when_available(self, cost_filter: CostFilter) -> None:
        """Test that LiteLLM pricing is used when available."""
        # Create arm with known LiteLLM model
        arm = ModelArm(
            model_id="gpt-4o-mini",  # Known in LiteLLM
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=999.0,  # Wrong price (should use LiteLLM)
            cost_per_output_token=999.0,
            expected_quality=0.85,
        )

        estimate = cost_filter.estimate_cost(arm, input_tokens=1000)

        # LiteLLM pricing for gpt-4o-mini is much less than $999/1K tokens
        # If LiteLLM pricing is used, total cost should be reasonable
        assert estimate.total_cost < 1.0  # Should be cents, not dollars

    def test_falls_back_to_arm_pricing(self, cost_filter: CostFilter) -> None:
        """Test fallback to arm pricing when LiteLLM doesn't have model."""
        arm = ModelArm(
            model_id="unknown-model-xyz-123",  # Not in LiteLLM
            provider="unknown",
            model_name="unknown-model-xyz-123",
            cost_per_input_token=1.0,  # $1 per 1K tokens
            cost_per_output_token=2.0,  # $2 per 1K tokens
            expected_quality=0.5,
        )

        estimate = cost_filter.estimate_cost(arm, input_tokens=1000)

        # Should use arm pricing: (1000/1000)*1.0 + (1000/1000)*2.0 = $3
        assert estimate.total_cost == pytest.approx(3.0, rel=0.01)
