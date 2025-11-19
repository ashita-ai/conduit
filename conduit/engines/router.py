"""Routing engine for ML-powered model selection."""

import logging

from conduit.core.models import (
    Query,
    QueryConstraints,
    QueryFeatures,
    RoutingDecision,
)
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandit import ContextualBandit

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
        """Filter models that satisfy constraints.

        Args:
            models: Available models
            constraints: Optional constraints

        Returns:
            List of eligible model IDs

        Note:
            Phase 1 uses approximate cost/latency estimates.
            Phase 2+ will use historical data from database.
        """
        if constraints is None:
            return models

        eligible = []

        # Approximate model characteristics (Phase 1 heuristics)
        model_specs = {
            "gpt-4o-mini": {"cost": 0.0001, "latency": 1.0, "quality": 0.7},
            "gpt-4o": {"cost": 0.001, "latency": 2.0, "quality": 0.9},
            "claude-sonnet-4": {"cost": 0.0005, "latency": 1.5, "quality": 0.85},
            "claude-opus-4": {"cost": 0.002, "latency": 3.0, "quality": 0.95},
        }

        for model in models:
            specs = model_specs.get(
                model, {"cost": 0.001, "latency": 2.0, "quality": 0.8}
            )

            # Check constraints
            if constraints.max_cost and specs["cost"] > constraints.max_cost:
                continue

            if constraints.max_latency and specs["latency"] > constraints.max_latency:
                continue

            if constraints.min_quality and specs["quality"] < constraints.min_quality:
                continue

            if (
                constraints.preferred_provider
                and constraints.preferred_provider not in model.lower()
            ):
                continue

            eligible.append(model)

        return eligible

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
    ):
        """Initialize router with default components.

        Args:
            models: List of available model IDs. If None, uses defaults.
            embedding_model: Sentence transformer model for query analysis.
        """
        # Use default models if not specified
        if models is None:
            models = [
                "gpt-4o-mini",
                "gpt-4o",
                "claude-3-5-sonnet-20241022",
                "claude-3-haiku-20240307",
            ]

        # Initialize components
        self.analyzer = QueryAnalyzer(embedding_model=embedding_model)
        self.bandit = ContextualBandit(models=models)
        self.routing_engine = RoutingEngine(
            bandit=self.bandit,
            analyzer=self.analyzer,
            models=models,
        )

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
        return await self.routing_engine.route(query)
