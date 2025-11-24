"""Constraint filtering service for routing decisions.

This module implements constraint-based model filtering, separating constraint
logic from routing logic to follow Single Responsibility Principle.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from conduit.core.models import QueryConstraints
from conduit.core.pricing import ModelPricing

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of constraint filtering operation.
    
    Attributes:
        eligible_models: List of model IDs that passed constraints
        relaxed: Whether constraints were relaxed to get results
        excluded_models: Map of model_id -> exclusion reason
        original_count: Number of models before filtering
        final_count: Number of models after filtering
    """
    
    eligible_models: list[str]
    relaxed: bool = False
    excluded_models: dict[str, str] = field(default_factory=dict)
    original_count: int = 0
    final_count: int = 0


class ConstraintFilter:
    """Service for filtering models by query constraints.
    
    Handles constraint-based model filtering with optional relaxation
    when no models satisfy strict constraints.
    
    Example:
        >>> pricing = {"gpt-4o-mini": ModelPricing(...), ...}
        >>> filter_service = ConstraintFilter(pricing)
        >>> 
        >>> constraints = QueryConstraints(max_cost=0.001, min_quality=0.7)
        >>> result = filter_service.filter_models(
        ...     models=["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"],
        ...     constraints=constraints
        ... )
        >>> print(result.eligible_models)  # Models meeting constraints
        >>> print(result.excluded_models)  # Excluded models with reasons
    """
    
    def __init__(
        self, 
        model_pricing: dict[str, ModelPricing] | None = None,
        model_metadata: dict[str, dict[str, Any]] | None = None
    ):
        """Initialize constraint filter.
        
        Args:
            model_pricing: Map of model_id -> pricing information
            model_metadata: Map of model_id -> metadata (quality, latency estimates, etc.)
        """
        self.model_pricing = model_pricing or {}
        self.model_metadata = model_metadata or {}
        
    def filter_models(
        self,
        models: list[str],
        constraints: QueryConstraints | None,
        allow_relaxation: bool = True,
    ) -> FilterResult:
        """Filter models by constraints with optional relaxation.
        
        Args:
            models: List of available model IDs
            constraints: Query constraints (None means no filtering)
            allow_relaxation: If True and no models pass, try relaxed constraints
            
        Returns:
            FilterResult with eligible models and metadata
            
        Example:
            >>> # Strict filtering
            >>> result = filter_service.filter_models(
            ...     models=["gpt-4o-mini", "gpt-4o"],
            ...     constraints=QueryConstraints(max_cost=0.001),
            ...     allow_relaxation=False
            ... )
            >>>
            >>> # With auto-relaxation fallback
            >>> result = filter_service.filter_models(
            ...     models=["gpt-4o-mini", "gpt-4o"],
            ...     constraints=QueryConstraints(max_cost=0.00001),
            ...     allow_relaxation=True
            ... )
            >>> # result.relaxed == True if constraints were relaxed
        """
        original_count = len(models)
        
        # No constraints means all models eligible
        if constraints is None:
            return FilterResult(
                eligible_models=models.copy(),
                relaxed=False,
                excluded_models={},
                original_count=original_count,
                final_count=original_count,
            )
        
        # Try strict filtering first
        eligible, excluded = self._apply_constraints(models, constraints)
        
        # If no models pass and relaxation allowed, try relaxed constraints
        if not eligible and allow_relaxation:
            logger.info(
                f"No models passed strict constraints. "
                f"Trying relaxed constraints (20% relaxation)"
            )
            relaxed_constraints = self.relax_constraints(constraints, factor=0.2)
            eligible, excluded = self._apply_constraints(models, relaxed_constraints)
            relaxed = True
        else:
            relaxed = False
            
        return FilterResult(
            eligible_models=eligible,
            relaxed=relaxed,
            excluded_models=excluded,
            original_count=original_count,
            final_count=len(eligible),
        )
    
    def _apply_constraints(
        self,
        models: list[str],
        constraints: QueryConstraints,
    ) -> tuple[list[str], dict[str, str]]:
        """Apply constraints to model list.
        
        Args:
            models: List of model IDs to filter
            constraints: Constraints to apply
            
        Returns:
            Tuple of (eligible_models, excluded_models_with_reasons)
        """
        eligible: list[str] = []
        excluded: dict[str, str] = {}
        
        for model_id in models:
            is_eligible, reason = self.check_model_eligibility(model_id, constraints)
            
            if is_eligible:
                eligible.append(model_id)
            else:
                excluded[model_id] = reason
                
        return eligible, excluded
    
    def relax_constraints(
        self,
        constraints: QueryConstraints,
        factor: float = 0.2,
    ) -> QueryConstraints:
        """Create relaxed version of constraints.
        
        Relaxes numeric constraints by the specified factor (percentage).
        Non-numeric constraints (like preferred_provider) are removed.
        
        Args:
            constraints: Original constraints
            factor: Relaxation factor (0.2 = 20% more lenient)
            
        Returns:
            New QueryConstraints with relaxed values
            
        Example:
            >>> constraints = QueryConstraints(max_cost=0.001, max_latency=2.0)
            >>> relaxed = filter_service.relax_constraints(constraints, factor=0.2)
            >>> # relaxed.max_cost == 0.0012 (20% higher)
            >>> # relaxed.max_latency == 2.4 (20% higher)
        """
        return QueryConstraints(
            max_cost=constraints.max_cost * (1 + factor) if constraints.max_cost else None,
            max_latency=constraints.max_latency * (1 + factor) if constraints.max_latency else None,
            min_quality=max(0.0, constraints.min_quality - factor) if constraints.min_quality else None,
            preferred_provider=None,  # Remove provider preference when relaxing
        )
    
    def check_model_eligibility(
        self,
        model_id: str,
        constraints: QueryConstraints,
    ) -> tuple[bool, str]:
        """Check if single model satisfies constraints.
        
        Args:
            model_id: Model to check
            constraints: Constraints to verify
            
        Returns:
            Tuple of (is_eligible, reason)
            - (True, "All constraints satisfied")
            - (False, "Specific reason for exclusion")
            
        Example:
            >>> eligible, reason = filter_service.check_model_eligibility(
            ...     "gpt-4o",
            ...     QueryConstraints(max_cost=0.001)
            ... )
            >>> if not eligible:
            ...     print(f"Excluded: {reason}")
        """
        # Check preferred provider
        if constraints.preferred_provider:
            model_provider = self._infer_provider(model_id)
            if model_provider != constraints.preferred_provider:
                return False, f"Provider {model_provider} != preferred {constraints.preferred_provider}"
        
        # Check cost constraint
        if constraints.max_cost is not None:
            estimated_cost = self._estimate_cost(model_id)
            if estimated_cost is None:
                logger.warning(f"No pricing data for {model_id}, allowing through cost check")
            elif estimated_cost > constraints.max_cost:
                return False, f"Estimated cost ${estimated_cost:.6f} > max ${constraints.max_cost:.6f}"
        
        # Check quality constraint
        if constraints.min_quality is not None:
            estimated_quality = self._estimate_quality(model_id)
            if estimated_quality < constraints.min_quality:
                return False, f"Estimated quality {estimated_quality:.2f} < min {constraints.min_quality:.2f}"
        
        # Check latency constraint
        if constraints.max_latency is not None:
            estimated_latency = self._estimate_latency(model_id)
            if estimated_latency is None:
                logger.warning(f"No latency data for {model_id}, allowing through latency check")
            elif estimated_latency > constraints.max_latency:
                return False, f"Estimated latency {estimated_latency:.2f}s > max {constraints.max_latency:.2f}s"
        
        return True, "All constraints satisfied"
    
    def _estimate_cost(self, model_id: str) -> float | None:
        """Estimate cost for a typical query with this model.
        
        Uses pricing data to estimate cost for ~1000 input + 500 output tokens.
        
        Args:
            model_id: Model to estimate cost for
            
        Returns:
            Estimated cost in USD, or None if no pricing data
        """
        pricing = self.model_pricing.get(model_id)
        if not pricing:
            return None
        
        # Estimate for typical query: 1000 input tokens, 500 output tokens
        estimated_input_tokens = 1000
        estimated_output_tokens = 500
        
        cost = (
            pricing.input_cost_per_token * estimated_input_tokens +
            pricing.output_cost_per_token * estimated_output_tokens
        )
        return cost
    
    def _estimate_quality(self, model_id: str) -> float:
        """Estimate quality for a model.
        
        Uses metadata if available, otherwise returns conservative estimate.
        
        Args:
            model_id: Model to estimate quality for
            
        Returns:
            Estimated quality score (0.0-1.0)
        """
        metadata = self.model_metadata.get(model_id, {})
        
        # Check for expected_quality in metadata
        if "expected_quality" in metadata:
            return float(metadata["expected_quality"])
        
        # Conservative heuristic based on model name
        model_lower = model_id.lower()
        if "gpt-4o" in model_lower and "mini" not in model_lower:
            return 0.95  # GPT-4o is high quality
        elif "claude-3-5-sonnet" in model_lower or "claude-3-opus" in model_lower:
            return 0.95  # Claude 3.5 Sonnet/Opus is high quality
        elif "gpt-4" in model_lower:
            return 0.90  # GPT-4 variants
        elif "claude-3-sonnet" in model_lower:
            return 0.85  # Claude 3 Sonnet
        elif "gpt-3.5" in model_lower or "mini" in model_lower:
            return 0.75  # Smaller/cheaper models
        elif "claude-3-haiku" in model_lower:
            return 0.75  # Claude 3 Haiku
        else:
            return 0.70  # Unknown models - conservative estimate
    
    def _estimate_latency(self, model_id: str) -> float | None:
        """Estimate latency for a model.
        
        Uses metadata if available, otherwise returns heuristic estimate.
        
        Args:
            model_id: Model to estimate latency for
            
        Returns:
            Estimated latency in seconds, or None if unknown
        """
        metadata = self.model_metadata.get(model_id, {})
        
        # Check for expected_latency in metadata
        if "expected_latency" in metadata:
            return float(metadata["expected_latency"])
        
        # Heuristic based on model size/provider
        model_lower = model_id.lower()
        if "mini" in model_lower or "haiku" in model_lower:
            return 2.0  # Fast models
        elif "gpt-4o" in model_lower:
            return 3.0  # Optimized GPT-4
        elif "gpt-4" in model_lower or "sonnet" in model_lower:
            return 5.0  # Larger models
        elif "opus" in model_lower:
            return 8.0  # Very large models
        else:
            return None  # Unknown - don't filter
    
    def _infer_provider(self, model_id: str) -> str:
        """Infer provider from model ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Provider name (openai, anthropic, google, etc.)
        """
        model_lower = model_id.lower()
        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mixtral" in model_lower:
            return "groq"
        elif "mistral" in model_lower:
            return "mistral"
        elif "command" in model_lower:
            return "cohere"
        else:
            return "unknown"
