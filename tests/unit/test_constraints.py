"""Unit tests for constraint filtering service."""

import pytest

from conduit.core.models import QueryConstraints
from conduit.core.pricing import ModelPricing
from conduit.engines.constraints import ConstraintFilter, FilterResult


class TestFilterResult:
    """Tests for FilterResult dataclass."""

    def test_filter_result_creation(self):
        """Test FilterResult creation with all fields."""
        result = FilterResult(
            eligible_models=["gpt-4o-mini", "claude-3-haiku"],
            relaxed=False,
            excluded_models={"gpt-4o": "Cost too high"},
            original_count=3,
            final_count=2,
        )

        assert result.eligible_models == ["gpt-4o-mini", "claude-3-haiku"]
        assert result.relaxed is False
        assert result.excluded_models == {"gpt-4o": "Cost too high"}
        assert result.original_count == 3
        assert result.final_count == 2

    def test_filter_result_defaults(self):
        """Test FilterResult with default values."""
        result = FilterResult(eligible_models=["gpt-4o-mini"])

        assert result.eligible_models == ["gpt-4o-mini"]
        assert result.relaxed is False
        assert result.excluded_models == {}
        assert result.original_count == 0
        assert result.final_count == 0


class TestConstraintFilterInit:
    """Tests for ConstraintFilter initialization."""

    def test_init_empty(self):
        """Test initialization with no pricing data."""
        filter_service = ConstraintFilter()

        assert filter_service.model_pricing == {}
        assert filter_service.model_metadata == {}

    def test_init_with_pricing(self):
        """Test initialization with pricing data."""
        pricing = {
            "gpt-4o-mini": ModelPricing(
                model_id="gpt-4o-mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
            )
        }
        filter_service = ConstraintFilter(model_pricing=pricing)

        assert "gpt-4o-mini" in filter_service.model_pricing
        assert filter_service.model_pricing["gpt-4o-mini"].model_id == "gpt-4o-mini"

    def test_init_with_metadata(self):
        """Test initialization with model metadata."""
        metadata = {
            "gpt-4o": {"expected_quality": 0.95, "expected_latency": 3.0}
        }
        filter_service = ConstraintFilter(model_metadata=metadata)

        assert "gpt-4o" in filter_service.model_metadata
        assert filter_service.model_metadata["gpt-4o"]["expected_quality"] == 0.95


class TestFilterModels:
    """Tests for filter_models method."""

    @pytest.fixture
    def filter_service(self):
        """Create filter service with test pricing data."""
        pricing = {
            "gpt-4o-mini": ModelPricing(
                model_id="gpt-4o-mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
            ),
            "gpt-4o": ModelPricing(
                model_id="gpt-4o",
                input_cost_per_million=5.0,
                output_cost_per_million=15.0,
            ),
            "claude-3-haiku": ModelPricing(
                model_id="claude-3-haiku",
                input_cost_per_million=0.25,
                output_cost_per_million=1.25,
            ),
        }
        return ConstraintFilter(model_pricing=pricing)

    @pytest.fixture
    def test_models(self):
        """List of test models."""
        return ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]

    def test_no_constraints(self, filter_service, test_models):
        """Test filtering with no constraints returns all models."""
        result = filter_service.filter_models(
            models=test_models,
            constraints=None,
        )

        assert result.eligible_models == test_models
        assert result.relaxed is False
        assert result.excluded_models == {}
        assert result.original_count == 3
        assert result.final_count == 3

    def test_cost_constraint(self, filter_service, test_models):
        """Test filtering by max_cost constraint."""
        # Cost estimate for gpt-4o-mini: (0.15/1M * 1000) + (0.60/1M * 500) = 0.00045
        # Cost estimate for gpt-4o: (5.0/1M * 1000) + (15.0/1M * 500) = 0.0125
        # Cost estimate for claude-3-haiku: (0.25/1M * 1000) + (1.25/1M * 500) = 0.000875
        constraints = QueryConstraints(max_cost=0.001)
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
        )

        # Only gpt-4o-mini should pass (cost ~0.00045)
        assert "gpt-4o-mini" in result.eligible_models
        assert "gpt-4o" not in result.eligible_models
        assert result.relaxed is False
        assert "gpt-4o" in result.excluded_models

    def test_quality_constraint(self, filter_service, test_models):
        """Test filtering by min_quality constraint."""
        constraints = QueryConstraints(min_quality=0.90)
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
        )

        # gpt-4o (0.95) should pass, mini (0.75) and haiku (0.75) should not
        assert "gpt-4o" in result.eligible_models
        assert "gpt-4o-mini" not in result.eligible_models
        assert "claude-3-haiku" not in result.eligible_models

    def test_latency_constraint(self, filter_service, test_models):
        """Test filtering by max_latency constraint."""
        constraints = QueryConstraints(max_latency=3.0)
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
        )

        # mini (~2s) should pass, gpt-4o (~3s) should pass, might filter others
        # Based on heuristics: mini=2.0, gpt-4o=3.0, haiku=2.0
        assert "gpt-4o-mini" in result.eligible_models
        assert "gpt-4o" in result.eligible_models
        assert "claude-3-haiku" in result.eligible_models

    def test_provider_constraint(self, filter_service, test_models):
        """Test filtering by preferred_provider constraint."""
        constraints = QueryConstraints(preferred_provider="openai")
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
        )

        assert "gpt-4o-mini" in result.eligible_models
        assert "gpt-4o" in result.eligible_models
        assert "claude-3-haiku" not in result.eligible_models
        assert "claude-3-haiku" in result.excluded_models

    def test_multiple_constraints(self, filter_service, test_models):
        """Test filtering with multiple constraints."""
        constraints = QueryConstraints(
            max_cost=0.001,
            min_quality=0.70,
            preferred_provider="openai",
        )
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
        )

        # Only gpt-4o-mini passes all: openai + cheap + quality 0.75
        assert result.eligible_models == ["gpt-4o-mini"]

    def test_no_eligible_models_without_relaxation(self, filter_service, test_models):
        """Test no models pass strict constraints without relaxation."""
        constraints = QueryConstraints(max_cost=0.00001)  # Impossibly low
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
            allow_relaxation=False,
        )

        assert result.eligible_models == []
        assert result.relaxed is False
        assert len(result.excluded_models) == 3

    def test_relaxation_when_no_eligible(self, filter_service, test_models):
        """Test constraint relaxation when no models eligible."""
        constraints = QueryConstraints(max_cost=0.00001)  # Impossibly low
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
            allow_relaxation=True,
        )

        # After 20% relaxation, max_cost becomes 0.000012
        # Still might not pass, but relaxed should be True
        assert result.relaxed is True

    def test_relaxation_finds_models(self, filter_service, test_models):
        """Test relaxation successfully finds eligible models."""
        # Set constraint just below gpt-4o-mini's cost
        constraints = QueryConstraints(max_cost=0.0004)  # Just below mini's ~0.00045
        result = filter_service.filter_models(
            models=test_models,
            constraints=constraints,
            allow_relaxation=True,
        )

        # After 20% relaxation: 0.0004 * 1.2 = 0.00048 (should include gpt-4o-mini)
        assert result.relaxed is True
        assert "gpt-4o-mini" in result.eligible_models


class TestRelaxConstraints:
    """Tests for relax_constraints method."""

    @pytest.fixture
    def filter_service(self):
        """Create filter service."""
        return ConstraintFilter()

    def test_relax_cost(self, filter_service):
        """Test relaxing max_cost constraint."""
        constraints = QueryConstraints(max_cost=1.0)
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.max_cost == 1.2

    def test_relax_latency(self, filter_service):
        """Test relaxing max_latency constraint."""
        constraints = QueryConstraints(max_latency=5.0)
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.max_latency == 6.0

    def test_relax_quality(self, filter_service):
        """Test relaxing min_quality constraint."""
        constraints = QueryConstraints(min_quality=0.8)
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.min_quality == 0.6

    def test_relax_quality_floor(self, filter_service):
        """Test min_quality doesn't go below 0.0."""
        constraints = QueryConstraints(min_quality=0.1)
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.min_quality == 0.0  # max(0.0, 0.1 - 0.2)

    def test_relax_removes_provider(self, filter_service):
        """Test relaxation removes preferred_provider constraint."""
        constraints = QueryConstraints(
            max_cost=1.0,
            preferred_provider="openai",
        )
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.max_cost == 1.2
        assert relaxed.preferred_provider is None

    def test_relax_none_values(self, filter_service):
        """Test relaxation preserves None values."""
        constraints = QueryConstraints(max_cost=1.0)
        relaxed = filter_service.relax_constraints(constraints, factor=0.2)

        assert relaxed.max_cost == 1.2
        assert relaxed.max_latency is None
        assert relaxed.min_quality is None
        assert relaxed.preferred_provider is None

    def test_custom_relaxation_factor(self, filter_service):
        """Test custom relaxation factor."""
        constraints = QueryConstraints(max_cost=1.0, max_latency=10.0)
        relaxed = filter_service.relax_constraints(constraints, factor=0.5)

        assert relaxed.max_cost == 1.5
        assert relaxed.max_latency == 15.0


class TestCheckModelEligibility:
    """Tests for check_model_eligibility method."""

    @pytest.fixture
    def filter_service(self):
        """Create filter service with test data."""
        pricing = {
            "gpt-4o-mini": ModelPricing(
                model_id="gpt-4o-mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
            )
        }
        metadata = {
            "gpt-4o-mini": {"expected_quality": 0.75, "expected_latency": 2.0}
        }
        return ConstraintFilter(model_pricing=pricing, model_metadata=metadata)

    def test_all_constraints_satisfied(self, filter_service):
        """Test model passes all constraints."""
        constraints = QueryConstraints(
            max_cost=0.001,
            min_quality=0.70,
            max_latency=3.0,
            preferred_provider="openai",
        )
        eligible, reason = filter_service.check_model_eligibility(
            "gpt-4o-mini", constraints
        )

        assert eligible is True
        assert reason == "All constraints satisfied"

    def test_cost_constraint_fails(self, filter_service):
        """Test model fails cost constraint."""
        constraints = QueryConstraints(max_cost=0.0001)
        eligible, reason = filter_service.check_model_eligibility(
            "gpt-4o-mini", constraints
        )

        assert eligible is False
        assert "cost" in reason.lower()
        assert "max" in reason.lower()

    def test_quality_constraint_fails(self, filter_service):
        """Test model fails quality constraint."""
        constraints = QueryConstraints(min_quality=0.90)
        eligible, reason = filter_service.check_model_eligibility(
            "gpt-4o-mini", constraints
        )

        assert eligible is False
        assert "quality" in reason.lower()
        assert "min" in reason.lower()

    def test_latency_constraint_fails(self, filter_service):
        """Test model fails latency constraint."""
        constraints = QueryConstraints(max_latency=1.0)
        eligible, reason = filter_service.check_model_eligibility(
            "gpt-4o-mini", constraints
        )

        assert eligible is False
        assert "latency" in reason.lower()

    def test_provider_constraint_fails(self, filter_service):
        """Test model fails provider constraint."""
        constraints = QueryConstraints(preferred_provider="anthropic")
        eligible, reason = filter_service.check_model_eligibility(
            "gpt-4o-mini", constraints
        )

        assert eligible is False
        assert "provider" in reason.lower()
        assert "anthropic" in reason.lower()


class TestEstimateCost:
    """Tests for _estimate_cost method."""

    def test_estimate_with_pricing(self):
        """Test cost estimation with pricing data."""
        pricing = {
            "gpt-4o-mini": ModelPricing(
                model_id="gpt-4o-mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
            )
        }
        filter_service = ConstraintFilter(model_pricing=pricing)
        cost = filter_service._estimate_cost("gpt-4o-mini")

        # 1000 input tokens * (0.15/1M) + 500 output tokens * (0.60/1M)
        expected_cost = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
        assert cost == pytest.approx(expected_cost)

    def test_estimate_without_pricing(self):
        """Test cost estimation without pricing data."""
        filter_service = ConstraintFilter()
        cost = filter_service._estimate_cost("unknown-model")

        assert cost is None


class TestEstimateQuality:
    """Tests for _estimate_quality method."""

    def test_quality_from_metadata(self):
        """Test quality estimation from metadata."""
        metadata = {"gpt-4o-mini": {"expected_quality": 0.85}}
        filter_service = ConstraintFilter(model_metadata=metadata)
        quality = filter_service._estimate_quality("gpt-4o-mini")

        assert quality == 0.85

    def test_quality_heuristic_gpt4o(self):
        """Test heuristic quality for gpt-4o."""
        filter_service = ConstraintFilter()
        quality = filter_service._estimate_quality("gpt-4o")

        assert quality == 0.95

    def test_quality_heuristic_gpt4o_mini(self):
        """Test heuristic quality for gpt-4o-mini."""
        filter_service = ConstraintFilter()
        quality = filter_service._estimate_quality("gpt-4o-mini")

        assert quality == 0.75

    def test_quality_heuristic_claude_sonnet(self):
        """Test heuristic quality for Claude Sonnet."""
        filter_service = ConstraintFilter()
        quality = filter_service._estimate_quality("claude-3-5-sonnet")

        assert quality == 0.95

    def test_quality_heuristic_unknown(self):
        """Test heuristic quality for unknown model."""
        filter_service = ConstraintFilter()
        quality = filter_service._estimate_quality("unknown-model")

        assert quality == 0.70


class TestEstimateLatency:
    """Tests for _estimate_latency method."""

    def test_latency_from_metadata(self):
        """Test latency estimation from metadata."""
        metadata = {"gpt-4o-mini": {"expected_latency": 1.5}}
        filter_service = ConstraintFilter(model_metadata=metadata)
        latency = filter_service._estimate_latency("gpt-4o-mini")

        assert latency == 1.5

    def test_latency_heuristic_mini(self):
        """Test heuristic latency for mini models."""
        filter_service = ConstraintFilter()
        latency = filter_service._estimate_latency("gpt-4o-mini")

        assert latency == 2.0

    def test_latency_heuristic_haiku(self):
        """Test heuristic latency for haiku models."""
        filter_service = ConstraintFilter()
        latency = filter_service._estimate_latency("claude-3-haiku")

        assert latency == 2.0

    def test_latency_heuristic_gpt4o(self):
        """Test heuristic latency for gpt-4o."""
        filter_service = ConstraintFilter()
        latency = filter_service._estimate_latency("gpt-4o")

        assert latency == 3.0

    def test_latency_heuristic_unknown(self):
        """Test heuristic latency for unknown model."""
        filter_service = ConstraintFilter()
        latency = filter_service._estimate_latency("unknown-model")

        assert latency is None


class TestInferProvider:
    """Tests for _infer_provider method."""

    def test_infer_openai(self):
        """Test inferring OpenAI provider."""
        filter_service = ConstraintFilter()

        assert filter_service._infer_provider("gpt-4o") == "openai"
        assert filter_service._infer_provider("gpt-4o-mini") == "openai"
        assert filter_service._infer_provider("gpt-3.5-turbo") == "openai"

    def test_infer_anthropic(self):
        """Test inferring Anthropic provider."""
        filter_service = ConstraintFilter()

        assert filter_service._infer_provider("claude-3-5-sonnet") == "anthropic"
        assert filter_service._infer_provider("claude-3-haiku") == "anthropic"

    def test_infer_google(self):
        """Test inferring Google provider."""
        filter_service = ConstraintFilter()

        assert filter_service._infer_provider("gemini-1.5-pro") == "google"

    def test_infer_groq(self):
        """Test inferring Groq provider."""
        filter_service = ConstraintFilter()

        assert filter_service._infer_provider("llama-3") == "groq"
        assert filter_service._infer_provider("mixtral-8x7b") == "groq"

    def test_infer_unknown(self):
        """Test inferring unknown provider."""
        filter_service = ConstraintFilter()

        assert filter_service._infer_provider("mystery-model") == "unknown"
