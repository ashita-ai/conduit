"""Routing engine for ML-powered model selection."""

import logging
from pathlib import Path
from typing import Any

from conduit.cache import CacheConfig, CacheService
from conduit.core.config import settings
from conduit.core.model_discovery import ModelDiscovery
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
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_enabled: bool | None = None,
        model_config_path: str | Path | None = None,
    ):
        """Initialize router with default components.

        Args:
            models: List of available model IDs. If None, uses programmatic discovery
                based on configured API keys (3 models per provider).
            embedding_model: Sentence transformer model for query analysis.
            cache_enabled: Override cache enabled setting. If None, uses config default.
            model_config_path: Optional path to YAML config file with model list.
                Takes precedence over auto-discovery if provided.

        Example:
            >>> # Auto-discovery (recommended)
            >>> router = Router()  # Uses models from configured providers

            >>> # Explicit model list
            >>> router = Router(models=["gpt-4o-mini", "claude-3-5-sonnet-20241022"])

            >>> # YAML config file
            >>> router = Router(model_config_path="models.yaml")
        """
        # Use model discovery if models not explicitly provided
        if models is None:
            discovery = ModelDiscovery(settings, config_path=model_config_path)
            models = discovery.get_models()
            logger.info(f"Auto-discovered {len(models)} models: {models}")

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

        # Router always uses hybrid routing (UCB1→LinUCB warm start)
        # Non-hybrid mode was removed as old ContextualBandit was deleted
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
