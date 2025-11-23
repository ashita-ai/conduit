"""Routing engine for ML-powered model selection."""

import logging
from typing import Any

from conduit.cache import CacheConfig, CacheService
from conduit.core.config import settings
from conduit.core.models import (
    Query,
    QueryConstraints,
    QueryFeatures,
    RoutingDecision,
)
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit
from conduit.engines.bandits.base import ModelArm
from conduit.engines.hybrid_router import HybridRouter
from conduit.models.registry import DEFAULT_REGISTRY, get_model_by_id

logger = logging.getLogger(__name__)


class RoutingEngine:
    """ML-powered model selection engine with fallback strategies."""

    def __init__(
        self,
        bandit: ContextualBandit,
        analyzer: QueryAnalyzer,
        models: list[str],
        circuit_breaker_states: dict[str, bool] | None = None,
    ):
        """Initialize routing engine.

        Args:
            bandit: Thompson Sampling bandit
            analyzer: Query analyzer
            models: Available model IDs
            circuit_breaker_states: Optional circuit breaker states (for testing)
        """
        self.bandit = bandit
        self.analyzer = analyzer
        self.models = models
        self.circuit_breaker_states = circuit_breaker_states or {}

    async def route(
        self,
        query: Query,
        features: QueryFeatures | None = None,
    ) -> RoutingDecision:
        """Route query to optimal model using Thompson Sampling with fallbacks.

        Args:
            query: User query
            features: Optional pre-extracted features (for testing)

        Returns:
            RoutingDecision with selected model and metadata

        Fallback Strategy:
            1. Filter models by constraints
            2. If no eligible models: relax constraints by 20% and retry
            3. If still none: use default model (gpt-4o-mini)
            4. Thompson Sampling on eligible models
            5. Check circuit breaker for selected model
            6. If circuit OPEN: exclude and reselect (max 2 retries)
            7. Return selection with metadata flags
        """
        # Extract features if not provided
        if features is None:
            features = await self.analyzer.analyze(query.text)

        # Filter models by constraints
        eligible_models = self._filter_by_constraints(self.models, query.constraints)
        constraints_relaxed = False

        # Fallback 1: Relax constraints if no eligible models
        if not eligible_models and query.constraints:
            logger.warning("No models satisfy constraints, relaxing by 20%")
            relaxed_constraints = self._relax_constraints(query.constraints, factor=0.2)
            eligible_models = self._filter_by_constraints(
                self.models, relaxed_constraints
            )
            constraints_relaxed = True

        # Fallback 2: Use default model if still no matches
        if not eligible_models:
            logger.error("No models after constraint relaxation, using default")
            return RoutingDecision(
                query_id=query.id,
                selected_model="gpt-4o-mini",
                confidence=0.0,
                features=features,
                reasoning="Default fallback - no models satisfied constraints",
                metadata={"constraints_relaxed": True, "fallback": "default"},
            )

        # Thompson Sampling selection with circuit breaker retry
        max_retries = 2
        for attempt in range(max_retries + 1):
            selected_model = self.bandit.select_model(
                features=features, models=eligible_models
            )

            # Check circuit breaker
            if not self._is_circuit_open(selected_model):
                return RoutingDecision(
                    query_id=query.id,
                    selected_model=selected_model,
                    confidence=self.bandit.get_confidence(selected_model),
                    features=features,
                    reasoning=self._explain_selection(selected_model, features),
                    metadata={
                        "constraints_relaxed": constraints_relaxed,
                        "attempt": attempt,
                    },
                )

            # Circuit open, exclude and retry
            logger.warning(f"Circuit breaker OPEN for {selected_model}, retrying")
            eligible_models = [m for m in eligible_models if m != selected_model]
            if not eligible_models:
                break

        # Fallback 3: All models circuit broken or exhausted retries
        logger.error("All models failed circuit breaker checks, using default")
        return RoutingDecision(
            query_id=query.id,
            selected_model="gpt-4o-mini",
            confidence=0.0,
            features=features,
            reasoning="Default fallback - circuit breakers open",
            metadata={"fallback": "circuit_breaker"},
        )

    def _filter_by_constraints(
        self, models: list[str], constraints: QueryConstraints | None
    ) -> list[str]:
        """Filter models that satisfy constraints using real pricing from llm-prices.com.

        Args:
            models: Available models
            constraints: Optional constraints

        Returns:
            List of eligible model IDs

        Note:
            Uses pricing data from llm-prices.com (71+ models, 24h cache).
            Quality and latency estimates are heuristics until historical data available.
        """
        if constraints is None:
            return models

        eligible = []

        for model in models:
            # Get model from registry (contains real pricing from llm-prices.com)
            model_arm = self._get_model_arm(model)
            if model_arm is None:
                # Model not in registry, skip (shouldn't happen in normal usage)
                logger.warning(f"Model {model} not found in registry, skipping")
                continue

            # Check cost constraint (using real pricing)
            if constraints.max_cost:
                # Average of input/output cost as approximation
                avg_cost = (
                    model_arm.cost_per_input_token + model_arm.cost_per_output_token
                ) / 2
                if avg_cost > constraints.max_cost:
                    continue

            # Check quality constraint (using expected_quality from registry)
            if (
                constraints.min_quality
                and model_arm.expected_quality < constraints.min_quality
            ):
                continue

            # Check latency constraint (TODO: use historical data when available)
            # For now, use rough estimates based on model size/provider
            if constraints.max_latency:
                estimated_latency = self._estimate_latency(model_arm)
                if estimated_latency > constraints.max_latency:
                    continue

            # Check provider preference
            if (
                constraints.preferred_provider
                and constraints.preferred_provider != model_arm.provider
            ):
                continue

            eligible.append(model)

        return eligible

    def _get_model_arm(self, model_id: str) -> ModelArm | None:
        """Get ModelArm from registry for a model ID.

        Args:
            model_id: Model identifier (may or may not have provider prefix)

        Returns:
            ModelArm if found, None otherwise
        """
        # Try with model_id as-is first
        arm = get_model_by_id(model_id, DEFAULT_REGISTRY)
        if arm is not None:
            return arm

        # Try adding common provider prefixes if not found
        for provider in ["openai", "anthropic", "google", "groq", "mistral", "cohere"]:
            prefixed_id = f"{provider}:{model_id}"
            arm = get_model_by_id(prefixed_id, DEFAULT_REGISTRY)
            if arm is not None:
                return arm

        return None

    def _estimate_latency(self, model_arm: ModelArm) -> float:
        """Estimate model latency based on provider and cost.

        Args:
            model_arm: Model to estimate latency for

        Returns:
            Estimated latency in seconds

        Note:
            This is a rough heuristic until we have historical data.
            Higher cost models tend to be larger/slower.
        """
        # Provider-specific baseline latencies
        provider_baselines = {
            "openai": 1.5,
            "anthropic": 1.8,
            "google": 1.2,
            "groq": 0.5,  # Groq is notably faster
            "mistral": 1.5,
            "cohere": 1.6,
        }

        baseline = provider_baselines.get(model_arm.provider, 1.5)

        # Adjust based on cost (higher cost = larger model = slower)
        avg_cost = (
            model_arm.cost_per_input_token + model_arm.cost_per_output_token
        ) / 2

        # Scale: 0.001 cost = 1x, 0.01 cost = 2x
        cost_multiplier = 1 + (avg_cost / 0.001) * 0.5

        return baseline * cost_multiplier

    def _relax_constraints(
        self, constraints: QueryConstraints, factor: float = 0.2
    ) -> QueryConstraints:
        """Relax constraints by percentage factor.

        Args:
            constraints: Original constraints
            factor: Relaxation factor (0.2 = 20% increase)

        Returns:
            Relaxed constraints
        """
        return QueryConstraints(
            max_cost=(
                constraints.max_cost * (1 + factor) if constraints.max_cost else None
            ),
            max_latency=(
                constraints.max_latency * (1 + factor)
                if constraints.max_latency
                else None
            ),
            min_quality=(
                constraints.min_quality * (1 - factor)
                if constraints.min_quality
                else None
            ),
            preferred_provider=constraints.preferred_provider,
        )

    def _is_circuit_open(self, model: str) -> bool:
        """Check if circuit breaker is open for model.

        Args:
            model: Model identifier

        Returns:
            True if circuit is OPEN (model unavailable)

        Note:
            Phase 1 uses simple in-memory state.
            Phase 2+ will use Redis for distributed state.
        """
        return self.circuit_breaker_states.get(model, False)

    def _explain_selection(self, model: str, features: QueryFeatures) -> str:
        """Generate human-readable explanation for selection.

        Args:
            model: Selected model
            features: Query features

        Returns:
            Explanation string
        """
        complexity = features.complexity_score
        domain = features.domain

        if complexity < 0.3:
            complexity_desc = "simple"
        elif complexity < 0.7:
            complexity_desc = "moderate"
        else:
            complexity_desc = "complex"

        state = self.bandit.get_model_state(model)
        success_rate = state.mean_success_rate

        return (
            f"Selected {model} for {complexity_desc} {domain} query. "
            f"Model success rate: {success_rate:.2f} "
            f"(α={state.alpha:.1f}, β={state.beta:.1f})"
        )


class Router:
    """High-level router interface for ML-powered model selection.

    This class provides a simple interface for routing queries to optimal LLM models
    using machine learning. It automatically handles all the complex setup and wiring.

    Example:
        >>> router = Router()
        >>> query = Query(text="What is 2+2?")
        >>> decision = await router.route(query)
        >>> print(f"Selected: {decision.selected_model}")
    """

    def __init__(
        self,
        models: list[str] | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_enabled: bool | None = None,
        use_hybrid: bool | None = None,
    ):
        """Initialize router with default components.

        Args:
            models: List of available model IDs. If None, uses defaults.
            embedding_model: Sentence transformer model for query analysis.
            cache_enabled: Override cache enabled setting. If None, uses config default.
            use_hybrid: Override hybrid routing setting. If None, uses config default.
        """
        # Use default models if not specified
        if models is None:
            models = settings.default_models

        # Initialize cache service if enabled
        cache_service: CacheService | None = None
        if cache_enabled is None:
            cache_enabled = settings.redis_cache_enabled

        if cache_enabled:
            cache_config = CacheConfig(
                enabled=True,
                redis_url=settings.redis_url,
                ttl=settings.redis_cache_ttl,
                max_retries=settings.redis_max_retries,
                timeout=settings.redis_timeout,
                circuit_breaker_threshold=settings.redis_circuit_breaker_threshold,
                circuit_breaker_timeout=settings.redis_circuit_breaker_timeout,
            )
            cache_service = CacheService(cache_config)
            logger.info("Router initialized with caching enabled")
        else:
            logger.info("Router initialized with caching disabled")

        # Initialize analyzer
        self.analyzer = QueryAnalyzer(
            embedding_model=embedding_model,
            cache_service=cache_service,
            use_pca=settings.use_pca,
            pca_dimensions=settings.pca_dimensions,
            pca_model_path=settings.pca_model_path,
        )

        # Determine whether to use hybrid routing
        if use_hybrid is None:
            use_hybrid = settings.use_hybrid_routing

        # Declare attributes with proper types (can be None depending on mode)
        self.hybrid_router: HybridRouter | None
        self.routing_engine: RoutingEngine | None
        self.bandit: ContextualBandit | None

        # Initialize routing components based on mode
        if use_hybrid:
            # Hybrid routing: UCB1→LinUCB warm start
            feature_dim = self.analyzer.feature_dim
            reward_weights = {
                "quality": settings.reward_weight_quality,
                "cost": settings.reward_weight_cost,
                "latency": settings.reward_weight_latency,
            }

            self.hybrid_router = HybridRouter(
                models=models,
                switch_threshold=settings.hybrid_switch_threshold,
                analyzer=self.analyzer,
                feature_dim=feature_dim,
                ucb1_c=settings.hybrid_ucb1_c,
                linucb_alpha=settings.hybrid_linucb_alpha,
                reward_weights=reward_weights,
            )
            self.routing_engine = None  # Not used in hybrid mode
            self.bandit = None  # Not used in hybrid mode
            logger.info(
                f"Router initialized with HYBRID routing "
                f"(switch at {settings.hybrid_switch_threshold} queries, "
                f"feature_dim={feature_dim})"
            )
        else:
            # Standard routing: Thompson Sampling
            self.bandit = ContextualBandit(models=models)
            self.routing_engine = RoutingEngine(
                bandit=self.bandit,
                analyzer=self.analyzer,
                models=models,
            )
            self.hybrid_router = None  # Not used in standard mode
            logger.info("Router initialized with STANDARD routing (Thompson Sampling)")

        self.cache = cache_service
        self.use_hybrid = use_hybrid

    async def route(self, query: Query) -> RoutingDecision:
        """Route a query to the optimal model.

        Args:
            query: The query to route, including text and optional constraints.

        Returns:
            RoutingDecision with the selected model, confidence, and reasoning.

        Example:
            >>> query = Query(text="Explain quantum physics simply")
            >>> decision = await router.route(query)
            >>> print(f"Use {decision.selected_model} (confidence: {decision.confidence:.2f})")
        """
        if self.use_hybrid:
            # Hybrid routing: UCB1→LinUCB
            assert self.hybrid_router is not None  # Type narrowing for mypy
            return await self.hybrid_router.route(query)
        else:
            # Standard routing: Thompson Sampling
            assert self.routing_engine is not None  # Type narrowing for mypy
            return await self.routing_engine.route(query)

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache stats or None if caching disabled

        Example:
            >>> stats = router.get_cache_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1f}%")
        """
        if not self.cache:
            return None

        stats = self.cache.get_stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "errors": stats.errors,
            "hit_rate": stats.hit_rate,
            "circuit_state": stats.circuit_state,
        }

    async def clear_cache(self) -> None:
        """Clear all cached query features (admin operation).

        Warning:
            This is expensive and should only be used for maintenance.
            Normal operation uses TTL-based expiry.
        """
        if self.cache:
            await self.cache.clear()

    async def close(self) -> None:
        """Close resources gracefully (Redis connection, etc.)."""
        if self.cache:
            await self.cache.close()
