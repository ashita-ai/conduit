"""Routing engine for ML-powered model selection."""

import asyncio
import logging
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
from conduit.engines.hybrid_router import HybridRouter

if TYPE_CHECKING:
    from conduit.core.state_store import StateStore

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
    """

    def __init__(
        self,
        models: list[str] | None = None,
        embedding_provider_type: str | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        cache_enabled: bool | None = None,
        use_hybrid_routing: bool = True,
        algorithm: str | None = None,
        state_store: "StateStore | None" = None,
        router_id: str | None = None,
        auto_persist: bool = True,
        checkpoint_interval: int = 100,
    ):
        """Initialize router with default components.

        Args:
            models: List of available model IDs. If None, uses defaults.
            embedding_provider_type: Embedding provider type (huggingface, openai, cohere, sentence-transformers).
                If None, uses config default (huggingface).
            embedding_model: Embedding model identifier (provider-specific, optional).
            embedding_api_key: API key for embedding provider (if required, optional).
            cache_enabled: Override cache enabled setting. If None, uses config default.
            use_hybrid_routing: If True, use hybrid routing (UCB1 → LinUCB). If False, use single algorithm.
                Default: True (recommended for faster cold start).
            algorithm: Algorithm to use when use_hybrid_routing=False. Options: "linucb", "thompson_sampling",
                "ucb1", "epsilon_greedy". Default: "linucb". Ignored if use_hybrid_routing=True.
            state_store: StateStore for automatic persistence (PostgresStateStore recommended).
                If None, persistence is disabled.
            router_id: Unique identifier for this router (used as persistence key).
                Default: timestamp-based ID (e.g., "router-20250226-143052-123456").
            auto_persist: If True, automatically save state after updates and periodically.
                Only applies if state_store is provided. Default: True.
            checkpoint_interval: Save state every N queries as backup.
                Default: 100. Only applies if state_store and auto_persist enabled.

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

        # Determine switch threshold based on routing mode
        if use_hybrid_routing:
            switch_threshold = settings.hybrid_switch_threshold
        else:
            # Start directly with LinUCB (effectively disable hybrid routing)
            switch_threshold = 0

        self.hybrid_router = HybridRouter(
            models=models,
            switch_threshold=switch_threshold,
            analyzer=self.analyzer,
            feature_dim=feature_dim,
            ucb1_c=settings.hybrid_ucb1_c,
            linucb_alpha=settings.hybrid_linucb_alpha,
            reward_weights=reward_weights,
            window_size=settings.bandit_window_size,
        )
        self.use_hybrid_routing = use_hybrid_routing

        # State persistence configuration
        self.state_store = state_store
        # Generate timestamp-based router_id if not provided
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
        logger.info(
            f"Router initialized with hybrid routing "
            f"(switch at {settings.hybrid_switch_threshold} queries, "
            f"feature_dim={feature_dim})"
        )

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

        # Route query
        decision = await self.hybrid_router.route(query)

        # Periodic checkpoint
        if (
            self.auto_persist
            and self.hybrid_router.query_count % self.checkpoint_interval == 0
        ):
            await self._save_state()

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
    ) -> None:
        """Update bandit weights with feedback from model execution.

        This method wraps HybridRouter.update() and adds automatic state persistence.
        State is saved after every update when auto_persist=True (recommended).

        Why save on every update?
        - Updates are when weights actually change (A matrices, b vectors)
        - Database write (~1-5ms) is negligible vs LLM call (~500ms+)
        - Never lose more than 1 query of learning if crash occurs
        - Async write doesn't block routing

        Args:
            model_id: The model that was used
            cost: Total cost of the query execution
            quality_score: Quality assessment (0.0-1.0)
            latency: Response time in seconds
            features: Query features from the routing decision (required for contextual learning)

        Example:
            >>> decision = await router.route(query)
            >>> response = await execute_llm_call(decision.selected_model, query)
            >>> await router.update(
            ...     model_id=decision.selected_model,
            ...     cost=response.cost,
            ...     quality_score=0.95,  # From arbiter evaluation
            ...     latency=response.latency,
            ...     features=decision.features,  # Use real features from routing decision
            ... )
            >>> # State automatically saved to database
        """
        from conduit.engines.bandits.base import BanditFeedback

        # Create feedback object
        feedback = BanditFeedback(
            model_id=model_id,
            cost=cost,
            quality_score=quality_score,
            latency=latency,
        )

        # Update hybrid router with real features (critical for contextual learning)
        await self.hybrid_router.update(feedback, features)

        # Auto-save state after update (when weights change)
        if self.auto_persist:
            await self._save_state()

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
