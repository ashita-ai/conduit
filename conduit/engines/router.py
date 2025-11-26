"""Routing engine for ML-powered model selection."""

import logging
from typing import Any

from conduit.cache import CacheConfig, CacheService
from conduit.core.config import load_preference_weights, settings
from conduit.core.models import (
    Query,
    RoutingDecision,
)
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.hybrid_router import HybridRouter

logger = logging.getLogger(__name__)


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
        embedding_provider_type: str | None = None,
        embedding_model: str | None = None,
        embedding_api_key: str | None = None,
        cache_enabled: bool | None = None,
        use_hybrid_routing: bool = True,
        algorithm: str | None = None,
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

        self.cache = cache_service
        logger.info(
            f"Router initialized with hybrid routing "
            f"(switch at {settings.hybrid_switch_threshold} queries, "
            f"feature_dim={feature_dim})"
        )

    async def route(self, query: Query) -> RoutingDecision:
        """Route a query to the optimal model using hybrid routing (UCB1→LinUCB).

        Args:
            query: The query to route, including text and optional constraints.

        Returns:
            RoutingDecision with the selected model, confidence, and reasoning.

        Example:
            >>> query = Query(text="Explain quantum physics simply")
            >>> decision = await router.route(query)
            >>> print(f"Use {decision.selected_model} (confidence: {decision.confidence:.2f})")
        """
        # Apply user preferences to reward weights
        if query.preferences:
            weights = load_preference_weights(query.preferences.optimize_for)
            self.hybrid_router.ucb1.reward_weights = weights
            self.hybrid_router.linucb.reward_weights = weights

        return await self.hybrid_router.route(query)

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
