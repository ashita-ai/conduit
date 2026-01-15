"""Routing engine for ML-powered model selection."""

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

from conduit.cache import CacheConfig, CacheService
from conduit.core.config import load_preference_weights, settings
from conduit.core.models import (
    Query,
    QueryFeatures,
    RoutingDecision,
)
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.cost_filter import CostFilter
from conduit.engines.hybrid_router import HybridRouter

if TYPE_CHECKING:
    from conduit.core.state_store import StateStore
    from conduit.engines.executor import ExecutionResult
    from conduit.observability.audit import AuditStore

logger = logging.getLogger(__name__)


class Router:
    """High-level router interface for ML-powered model selection.

    This class provides a simple interface for routing queries to optimal LLM models
    using machine learning. It automatically handles all the complex setup and wiring.

    Contract Guarantees:
        - Thread Safety: Safe for concurrent route() calls
        - Resource Management: Manages all components lifecycle automatically
        - Graceful Degradation: Works without Redis (caching disabled)
        - Error Handling: Never fails routing (fallback to random selection)
        - Configuration Validation: Validates settings at initialization

    State Management:
        - QueryAnalyzer: Query feature extraction with caching
        - HybridRouter: UCB1→LinUCB routing strategy
        - CacheService: Optional Redis cache with circuit breaker
        - Algorithm state: Auto-persisted to database (if state_store provided)

    State Persistence (NEW):
        - Auto-load on initialization: Resumes from last saved state
        - Save after every update: Weights never lost (≤1 query max)
        - Periodic checkpoint: Every N queries as backup
        - Graceful shutdown: Final save in close()
        - Error handling: Persistence errors don't break routing

    Lifecycle:
        1. Initialize: Setup components (analyzer, cache, hybrid router)
        2. Auto-load state: Resume from database if available
        3. Route: Process queries and learn from feedback
        4. Auto-persist: Save state after weight updates
        5. Close: Final save and cleanup resources

    Performance Design Goals (not enforced):
        - Cold start: Fast (UCB1 phase, no embedding needed)
        - Warm routing: Depends on embedding + bandit computation
        - With cache hit: Much faster (embedding cached)
        - Actual performance varies with deployment environment and load

    Error Handling:
        - Embedding failures: Use zero vector, route with UCB1
        - Cache failures: Bypass cache, log warning, continue routing
        - Model unavailable: Try next best model from candidates
        - Invalid query: Validate and sanitize before processing

    Example:
        >>> router = Router()
        >>> query = Query(text="What is 2+2?")
        >>> decision = await router.route(query)
        >>> print(f"Selected: {decision.selected_model}")
        >>> await router.close()  # Cleanup resources

    Context Manager Example:
        >>> async with Router() as router:
        ...     query = Query(text="What is 2+2?")
        ...     decision = await router.route(query)
        ...     print(f"Selected: {decision.selected_model}")
        ... # Resources automatically cleaned up on exit
    """

    def __init__(
        self,
        models: list[str] | None = None,
        embedding_provider_type: str | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        cache_enabled: bool | None = None,
        algorithm: str = "thompson_sampling",
        state_store: "StateStore | None" = None,
        router_id: str | None = None,
        auto_persist: bool = True,
        checkpoint_interval: int = 100,
        audit_store: "AuditStore | None" = None,
    ):
        """Initialize router with default components.

        Args:
            models: List of available model IDs. If None, uses defaults.
            embedding_provider_type: Embedding provider type (huggingface, openai, cohere, sentence-transformers).
                If None, uses config default (huggingface).
            embedding_model: Embedding model identifier (provider-specific, optional).
            embedding_api_key: API key for embedding provider (if required, optional).
            cache_enabled: Override cache enabled setting. If None, uses config default.
            algorithm: Routing algorithm to use. Default: "thompson_sampling" (non-contextual, no PCA needed).
                Options:
                - "thompson_sampling": Pure Thompson Sampling (non-contextual, recommended default)
                - "linucb": Pure LinUCB (contextual, uses query features)
                - "contextual_thompson_sampling": Contextual Thompson Sampling (uses query features)
                - "ucb1": Pure UCB1 (non-contextual)
                - "epsilon_greedy": Epsilon-Greedy with decaying exploration (non-contextual)
                - "random": Pure random selection baseline (non-contextual, no learning)
                - "dueling": Contextual Dueling Bandit (pairwise comparisons with features)
                - "hybrid_thompson_linucb": Hybrid Thompson → LinUCB (legacy, switch at 2000 queries)
                - "hybrid_ucb1_linucb": Hybrid UCB1 → LinUCB (legacy)

                Note: PCA is controlled by config (use_pca setting), not by algorithm choice.
                Contextual algorithms (linucb, contextual_thompson_sampling, dueling) can work with or without PCA.
            state_store: StateStore for automatic persistence (PostgresStateStore recommended).
                If None, persistence is disabled.
            router_id: Unique identifier for this router (used as persistence key).
                Priority: explicit param > CONDUIT_ROUTER_ID env var > timestamp-based.
                For Kubernetes multi-replica deployments, set CONDUIT_ROUTER_ID env var
                so all replicas share state (e.g., "production-cluster").
            auto_persist: If True, automatically save state after updates and periodically.
                Only applies if state_store is provided. Default: True.
            checkpoint_interval: Save state every N queries as backup.
                Default: 100. Only applies if state_store and auto_persist enabled.
            audit_store: AuditStore for decision audit logging (PostgresAuditStore recommended).
                If None, audit logging is disabled. Audit logs capture detailed decision
                context for compliance, debugging, and analysis.

        Example with persistence:
            >>> from conduit.core.database import Database
            >>> from conduit.core.postgres_state_store import PostgresStateStore
            >>>
            >>> db = Database()
            >>> await db.connect()
            >>> store = PostgresStateStore(db.pool)
            >>>
            >>> router = Router(
            ...     models=["gpt-4o-mini", "gpt-4o"],
            ...     state_store=store,
            ...     router_id="production-router",
            ...     auto_persist=True,  # Auto-save after updates
            ... )
            >>> # Router auto-loads from saved state on initialization
            >>> # Saves after every update() call
            >>> # Saves final state on close()
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

        # Initialize analyzer with embedding provider
        embedding_provider_type = embedding_provider_type or settings.embedding_provider
        embedding_model_config = embedding_model or settings.embedding_model or None
        embedding_api_key_config = (
            embedding_api_key or settings.embedding_api_key or None
        )

        self.analyzer = QueryAnalyzer(
            embedding_provider_type=embedding_provider_type,
            embedding_model=embedding_model_config,
            embedding_api_key=embedding_api_key_config,
            cache_service=cache_service,
            use_pca=settings.use_pca,
            pca_dimensions=settings.pca_dimensions,
            pca_model_path=settings.pca_model_path,
        )

        # Configure routing strategy
        feature_dim = self.analyzer.feature_dim
        reward_weights = {
            "quality": settings.reward_weight_quality,
            "cost": settings.reward_weight_cost,
            "latency": settings.reward_weight_latency,
        }

        # Map algorithm to HybridRouter configuration
        # Single algorithms use switch_threshold=infinity (never switch phases)
        # Hybrid algorithms use default switch_threshold=2000
        algorithm_config = {
            # Non-contextual single algorithms (phase1 only, never switch)
            "thompson_sampling": {
                "phase1_algorithm": "thompson_sampling",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            "ucb1": {
                "phase1_algorithm": "ucb1",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            "epsilon_greedy": {
                "phase1_algorithm": "epsilon_greedy",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            "random": {
                "phase1_algorithm": "random",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            # Baseline algorithms (non-learning, for benchmarking)
            "always_best": {
                "phase1_algorithm": "always_best",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            "always_cheapest": {
                "phase1_algorithm": "always_cheapest",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            "oracle": {
                "phase1_algorithm": "oracle",
                "phase2_algorithm": "linucb",  # Unused (never reaches phase2)
                "switch_threshold": float("inf"),
            },
            # Contextual single algorithms (phase2 only, start immediately)
            "linucb": {
                "phase1_algorithm": "thompson_sampling",  # Unused (starts in phase2)
                "phase2_algorithm": "linucb",
                "switch_threshold": 0,
            },
            "contextual_thompson_sampling": {
                "phase1_algorithm": "thompson_sampling",  # Unused (starts in phase2)
                "phase2_algorithm": "contextual_thompson_sampling",
                "switch_threshold": 0,
            },
            "dueling": {
                "phase1_algorithm": "thompson_sampling",  # Unused (starts in phase2)
                "phase2_algorithm": "dueling",
                "switch_threshold": 0,
            },
            # Hybrid algorithms (transition from phase1 to phase2)
            "hybrid_thompson_linucb": {
                "phase1_algorithm": "thompson_sampling",
                "phase2_algorithm": "linucb",
                "switch_threshold": settings.hybrid_switch_threshold,
            },
            "hybrid_ucb1_linucb": {
                "phase1_algorithm": "ucb1",
                "phase2_algorithm": "linucb",
                "switch_threshold": settings.hybrid_switch_threshold,
            },
        }

        # Validate algorithm choice
        if algorithm not in algorithm_config:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Valid options: {list(algorithm_config.keys())}"
            )

        config = algorithm_config[algorithm]

        # Type narrowing for config values (mypy can't infer from literal dict)
        switch_threshold_val = config["switch_threshold"]
        phase1_algo = config["phase1_algorithm"]
        phase2_algo = config["phase2_algorithm"]

        # Handle int, float (inf), or any numeric type
        if isinstance(switch_threshold_val, float) and switch_threshold_val == float(
            "inf"
        ):
            switch_threshold_int = 999999999
        else:
            switch_threshold_int = int(switch_threshold_val)  # type: ignore[call-overload]

        self.hybrid_router = HybridRouter(
            models=models,
            switch_threshold=switch_threshold_int,
            phase1_algorithm=str(phase1_algo),
            phase2_algorithm=str(phase2_algo),
            analyzer=self.analyzer,
            feature_dim=feature_dim,
            ucb1_c=settings.hybrid_ucb1_c,
            linucb_alpha=settings.hybrid_linucb_alpha,
            reward_weights=reward_weights,
            window_size=settings.bandit_window_size,
        )
        self.algorithm = algorithm

        # State persistence configuration
        self.state_store = state_store
        # Router ID priority: explicit param > CONDUIT_ROUTER_ID env var > timestamp-based
        # Using env var allows Kubernetes replicas to share state via shared router_id
        if router_id is None:
            router_id = os.getenv("CONDUIT_ROUTER_ID")
        if router_id is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            router_id = f"router-{timestamp}"
        self.router_id = router_id
        self.auto_persist = auto_persist and state_store is not None
        self.checkpoint_interval = checkpoint_interval
        self._state_loaded = False

        # Auto-load saved state if available
        if self.state_store and self.auto_persist:
            # Schedule async state loading
            # Note: Can't await in __init__, caller should await _load_initial_state()
            logger.info(
                f"State persistence enabled for router '{router_id}' "
                f"(checkpoint every {checkpoint_interval} queries)"
            )

        self.cache = cache_service

        # Initialize cost filter for budget enforcement
        self.cost_filter = CostFilter(
            output_ratio=settings.cost_output_ratio,
            fallback_on_empty=settings.cost_fallback_on_empty,
        )

        # Audit logging (optional)
        self.audit_store = audit_store
        if audit_store:
            logger.info("Decision audit logging enabled")

        logger.info(
            f"Router initialized with algorithm='{algorithm}' "
            f"(switch_threshold={config['switch_threshold']}, "
            f"feature_dim={feature_dim})"
        )

    async def __aenter__(self) -> "Router":
        """Enter async context manager.

        Initializes resources and loads saved state if persistence is enabled.
        This provides an alternative to manually calling close() for cleanup.

        Returns:
            Self for use in async with statement.

        Example:
            >>> async with Router(state_store=store) as router:
            ...     decision = await router.route(query)
            ...     # Use router...
            ... # close() called automatically
        """
        # Load initial state if persistence is enabled
        if self.auto_persist and not self._state_loaded:
            await self._load_initial_state()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager.

        Saves final state and cleans up resources (cache connections, etc.).
        Exceptions are not suppressed and will propagate after cleanup.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise.
            exc_val: Exception instance if an exception was raised, None otherwise.
            exc_tb: Traceback if an exception was raised, None otherwise.
        """
        await self.close()

    async def route(self, query: Query) -> RoutingDecision:
        """Route a query to the optimal model using hybrid routing (UCB1→LinUCB).

        Contract Guarantees:
            - MUST always return valid RoutingDecision (never None)
            - MUST select from available models only
            - MUST be thread-safe for concurrent calls
            - MUST apply user preferences when provided

        Performance:
            - Designed to be fast (typically completes quickly)
            - Actual latency depends on embedding provider, cache, and network
            - Cold start faster than warm routing (no embedding needed)

        State Modifications:
            - Updates bandit algorithm state (via feedback loops)
            - MAY update cache with query features
            - Increments routing counters
            - Auto-saves state periodically if persistence enabled

        Routing Strategy:
            1. Cold start (queries < switch_threshold): UCB1 (fast, no context)
            2. Warm routing (queries >= switch_threshold): LinUCB (contextual)
            3. User preferences: Applied as reward weight overrides

        Args:
            query: The query to route, including text and optional constraints.

        Returns:
            RoutingDecision with the selected model, confidence, and reasoning.

        Example:
            >>> query = Query(
            ...     text="Explain quantum physics simply",
            ...     preferences=UserPreferences(optimize_for="cost")
            ... )
            >>> decision = await router.route(query)
            >>> assert decision.selected_model in router.hybrid_router.models  # Guaranteed
            >>> print(f"Use {decision.selected_model} (confidence: {decision.confidence:.2f})")
        """
        # Auto-load state on first route call if not already loaded
        if self.auto_persist and not self._state_loaded:
            await self._load_initial_state()

        # Apply user preferences to reward weights
        if query.preferences:
            weights = load_preference_weights(query.preferences.optimize_for)
            self.hybrid_router.ucb1.reward_weights = weights
            self.hybrid_router.linucb.reward_weights = weights

        # Apply cost budget filtering if max_cost constraint is set
        available_arms = None
        constraints_relaxed = False
        if query.constraints and query.constraints.max_cost is not None:
            filter_result = self.cost_filter.filter_by_budget(
                models=self.hybrid_router.arms,
                max_cost=query.constraints.max_cost,
                query_text=query.text,
            )
            available_arms = filter_result.models
            constraints_relaxed = filter_result.was_relaxed

            if filter_result.was_relaxed:
                logger.warning(
                    f"Cost constraint relaxed: budget=${query.constraints.max_cost:.4f}, "
                    f"using cheapest model {filter_result.cheapest_model}"
                )

        # Route query with filtered arms
        decision = await self.hybrid_router.route(query, available_arms=available_arms)

        # Add cost constraint metadata to decision if constraints were applied
        if query.constraints and query.constraints.max_cost is not None:
            decision.metadata["max_cost_budget"] = query.constraints.max_cost
            decision.metadata["constraints_relaxed"] = constraints_relaxed
            if constraints_relaxed:
                decision.metadata["cost_fallback_reason"] = (
                    "No models within budget, using cheapest available"
                )

        # Periodic checkpoint
        if (
            self.auto_persist
            and self.hybrid_router.query_count % self.checkpoint_interval == 0
        ):
            await self._save_state()

        # Audit logging (non-blocking, errors don't affect routing)
        if self.audit_store:
            await self._log_audit_entry(decision, query, constraints_relaxed)

        return decision

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

    async def update(
        self,
        model_id: str,
        cost: float,
        quality_score: float,
        latency: float,
        features: QueryFeatures,
        confidence: float = 1.0,
    ) -> None:
        """Update bandit weights with feedback from model execution.

        This method wraps HybridRouter.update() and adds automatic state persistence.
        State is saved after every update when auto_persist=True (recommended).

        Why save on every update?
        - Updates are when weights actually change (A matrices, b vectors)
        - Database write (~1-5ms) is negligible vs LLM call (~500ms+)
        - Never lose more than 1 query of learning if crash occurs
        - Async write doesn't block routing

        Confidence-Weighted Updates:
            The confidence parameter controls how strongly this feedback affects
            the bandit's learned weights:
            - 1.0: Full weight (explicit user feedback like thumbs up/down)
            - 0.7-0.9: High confidence implicit signal (regeneration, task failure)
            - 0.5: Uncertain signal (time-based proxies)

            Lower confidence causes softer updates, appropriate for signals where
            we're less certain about user intent.

        Args:
            model_id: The model that was used
            cost: Total cost of the query execution
            quality_score: Quality assessment (0.0-1.0)
            latency: Response time in seconds
            features: Query features from the routing decision (required for contextual learning)
            confidence: Confidence in this feedback (0.0-1.0). Default: 1.0.
                Lower values cause softer bandit updates.

        Example:
            >>> decision = await router.route(query)
            >>> response = await execute_llm_call(decision.selected_model, query)
            >>>
            >>> # Explicit feedback (full confidence)
            >>> await router.update(
            ...     model_id=decision.selected_model,
            ...     cost=response.cost,
            ...     quality_score=0.95,  # From arbiter evaluation
            ...     latency=response.latency,
            ...     features=decision.features,
            ...     confidence=1.0,  # Explicit signal
            ... )
            >>>
            >>> # Implicit feedback (partial confidence)
            >>> await router.update(
            ...     model_id=decision.selected_model,
            ...     cost=response.cost,
            ...     quality_score=0.0,  # User regenerated
            ...     latency=response.latency,
            ...     features=decision.features,
            ...     confidence=0.8,  # Implicit signal, less certain
            ... )
        """
        from conduit.engines.bandits.base import BanditFeedback

        # Create feedback object with confidence
        feedback = BanditFeedback(
            model_id=model_id,
            cost=cost,
            quality_score=quality_score,
            latency=latency,
            confidence=confidence,
        )

        # Update hybrid router with real features (critical for contextual learning)
        await self.hybrid_router.update(feedback, features)

        # Auto-save state after update (when weights change)
        if self.auto_persist:
            await self._save_state()

    async def update_with_fallback_attribution(
        self,
        execution_result: "ExecutionResult",
        quality_score: float,
        features: QueryFeatures,
    ) -> None:
        """Update bandit weights with proper attribution for fallback scenarios.

        When a fallback model is used, this method:
        1. Penalizes all failed models (quality=0.0) to reduce future selection
        2. Rewards the successful model (fallback or original) with actual quality

        This ensures the bandit learns from model failures and prefers reliable models.

        Args:
            execution_result: Result from execute_with_fallback() with model tracking
            quality_score: Quality assessment of the response (0.0-1.0)
            features: Query features from the routing decision

        Example:
            >>> decision = await router.route(query)
            >>> result = await executor.execute_with_fallback(
            ...     decision=decision,
            ...     prompt=query.text,
            ...     result_type=MyOutput,
            ... )
            >>> await router.update_with_fallback_attribution(
            ...     execution_result=result,
            ...     quality_score=0.95,  # Quality of successful response
            ...     features=decision.features,
            ... )
        """
        # Penalize all failed models (they were unavailable/errored)
        for failed_model in execution_result.failed_models:
            await self.update(
                model_id=failed_model,
                cost=0.0,  # No cost incurred (failed before completion)
                quality_score=0.0,  # Penalize unavailability
                latency=0.0,  # No meaningful latency
                features=features,
            )
            logger.info(f"Penalized failed model: {failed_model}")

        # Reward the successful model with actual quality
        await self.update(
            model_id=execution_result.model_used,
            cost=execution_result.response.cost,
            quality_score=quality_score,
            latency=execution_result.response.latency,
            features=features,
        )

        if execution_result.was_fallback:
            logger.info(
                f"Fallback attribution complete: penalized {len(execution_result.failed_models)} "
                f"failed model(s), rewarded {execution_result.model_used}"
            )

    async def _load_initial_state(self) -> None:
        """Load saved state from database on initialization.

        Called automatically on first route() if auto_persist=True.
        Errors are logged but don't prevent router from working.
        """
        if not self.state_store or self._state_loaded:
            return

        try:
            loaded = await self.hybrid_router.load_state(
                self.state_store, self.router_id
            )

            if loaded:
                logger.info(
                    f"Resumed router '{self.router_id}' from saved state "
                    f"(query_count={self.hybrid_router.query_count}, "
                    f"phase={self.hybrid_router.current_phase})"
                )
            else:
                logger.info(
                    f"No saved state found for router '{self.router_id}', starting fresh"
                )

            self._state_loaded = True

        except Exception as e:
            logger.error(
                f"Failed to load state for router '{self.router_id}': {e}. "
                f"Starting with fresh state."
            )
            self._state_loaded = True  # Don't try again

    async def _save_state(self) -> None:
        """Save current state to database.

        Called automatically:
        - After every update() (when weights change)
        - Periodically every N queries (checkpoint)
        - On close() (graceful shutdown)

        Errors are logged but don't break routing.
        """
        if not self.state_store:
            return

        try:
            await self.hybrid_router.save_state(self.state_store, self.router_id)
            logger.debug(
                f"Saved state for router '{self.router_id}' "
                f"(query_count={self.hybrid_router.query_count})"
            )

        except Exception as e:
            logger.error(
                f"Failed to save state for router '{self.router_id}': {e}. "
                f"Routing continues normally."
            )

    async def _log_audit_entry(
        self, decision: RoutingDecision, query: Query, constraints_relaxed: bool
    ) -> None:
        """Log routing decision to audit trail.

        Called automatically after every route() when audit_store is configured.
        Errors are logged but don't affect routing.

        Args:
            decision: The routing decision made
            query: Original query
            constraints_relaxed: Whether cost constraints were relaxed
        """
        if not self.audit_store:
            return

        try:
            from conduit.observability.audit import create_audit_entry

            # Get the active bandit to compute scores
            if self.hybrid_router.current_phase == self.hybrid_router.phase1_algorithm:
                active_bandit = self.hybrid_router.phase1_bandit
            else:
                active_bandit = self.hybrid_router.phase2_bandit

            # Compute arm scores for audit context
            arm_scores = active_bandit.compute_scores(decision.features)

            # Build constraints metadata
            constraints_applied: dict[str, Any] = {}
            if query.constraints:
                if query.constraints.max_cost is not None:
                    constraints_applied["max_cost"] = query.constraints.max_cost
                if query.constraints.max_latency is not None:
                    constraints_applied["max_latency"] = query.constraints.max_latency
            if constraints_relaxed:
                constraints_applied["constraints_relaxed"] = True

            # Create and log audit entry
            entry = create_audit_entry(
                decision=decision,
                algorithm_phase=self.hybrid_router.current_phase,
                query_count=self.hybrid_router.query_count,
                arm_scores=arm_scores,
                constraints_applied=constraints_applied,
            )

            await self.audit_store.log_decision(entry)
            logger.debug(f"Logged audit entry for decision {decision.id}")

        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}. Routing continues normally.")

    async def close(self) -> None:
        """Close resources gracefully (Redis connection, etc.).

        Saves final state if persistence is enabled.
        """
        # Save final state before shutdown
        if self.auto_persist:
            logger.info(f"Saving final state for router '{self.router_id}'...")
            await self._save_state()

        # Close cache connection
        if self.cache:
            await self.cache.close()
