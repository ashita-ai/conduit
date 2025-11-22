"""LiteLLM routing strategy using Conduit's ML-powered model selection."""

import logging
from typing import Any, Dict, List, Optional, Union

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit_litellm.utils import extract_model_ids, validate_litellm_model_list

logger = logging.getLogger(__name__)


try:
    from litellm.router import CustomRoutingStrategyBase

    LITELLM_AVAILABLE = True
except ImportError:
    logger.warning(
        "LiteLLM not installed. Install with: pip install conduit[litellm]"
    )
    LITELLM_AVAILABLE = False

    # Create stub base class for type checking when LiteLLM not installed
    class CustomRoutingStrategyBase:  # type: ignore[no-redef]
        """Stub base class when LiteLLM is not available."""

        pass


class ConduitRoutingStrategy(CustomRoutingStrategyBase):
    """ML-powered routing strategy for LiteLLM using Conduit's contextual bandits.

    This strategy integrates Conduit's machine learning-based model selection
    with LiteLLM's router, enabling intelligent routing across 100+ LLM providers.

    The strategy uses Conduit's bandit algorithms (LinUCB, Thompson Sampling, etc.)
    to learn which models perform best for different types of queries, optimizing
    for quality, cost, and latency.

    Example:
        >>> from litellm import Router
        >>> from conduit_litellm import ConduitRoutingStrategy
        >>>
        >>> router = Router(model_list=[...])
        >>> router.set_custom_routing_strategy(
        ...     ConduitRoutingStrategy(use_hybrid=True)
        ... )
        >>>
        >>> # Now LiteLLM uses Conduit's ML routing
        >>> response = await router.acompletion(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """

    def __init__(
        self,
        conduit_router: Optional[Router] = None,
        **conduit_config: Any
    ):
        """Initialize Conduit routing strategy.

        Args:
            conduit_router: Optional pre-configured Conduit router. If not provided,
                a router will be automatically created from LiteLLM's model_list
                on first request.
            **conduit_config: Additional Conduit configuration options passed to Router.
                - use_hybrid (bool): Enable UCB1→LinUCB warm start (default: False)
                - embedding_model (str): Sentence transformer model (default: all-MiniLM-L6-v2)
                - cache_enabled (bool): Enable Redis caching (default: False)
                - redis_url (str): Redis connection URL (if cache_enabled=True)

        Example:
            >>> # Use hybrid routing with caching
            >>> strategy = ConduitRoutingStrategy(
            ...     use_hybrid=True,
            ...     cache_enabled=True,
            ...     redis_url="redis://localhost:6379"
            ... )
        """
        super().__init__()
        self.conduit_router = conduit_router
        self.conduit_config = conduit_config
        self._initialized = False
        self._router: Optional[Any] = None  # LiteLLM router reference

    async def _initialize_from_litellm(self, router: Any) -> None:
        """Initialize Conduit router from LiteLLM model list on first call.

        This method extracts model IDs from LiteLLM's model_list and creates
        a Conduit Router configured with those models. Initialization is lazy
        (happens on first request) to allow LiteLLM to fully configure the router.

        Args:
            router: LiteLLM router instance with model_list attribute.

        Raises:
            ValueError: If model_list is empty or invalid format.
            AttributeError: If router doesn't have model_list attribute.
        """
        if self._initialized:
            return

        # Store LiteLLM router reference
        self._router = router

        # Validate LiteLLM model_list format
        if not hasattr(router, "model_list"):
            raise AttributeError(
                "LiteLLM router missing 'model_list' attribute. "
                "Ensure router is properly initialized before setting custom strategy."
            )

        model_list = router.model_list
        validate_litellm_model_list(model_list)

        # Extract model IDs from LiteLLM model_list
        model_ids = extract_model_ids(model_list)

        logger.info(
            f"Initializing Conduit routing strategy with {len(model_ids)} models: "
            f"{', '.join(model_ids[:5])}{'...' if len(model_ids) > 5 else ''}"
        )

        # Initialize Conduit router if not provided
        if not self.conduit_router:
            self.conduit_router = Router(
                models=model_ids,
                **self.conduit_config
            )
            logger.info(
                f"Created Conduit router with config: "
                f"use_hybrid={self.conduit_config.get('use_hybrid', False)}, "
                f"cache_enabled={self.conduit_config.get('cache_enabled', False)}"
            )

        self._initialized = True

    async def async_get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List[Any]]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Select optimal LiteLLM deployment using Conduit's ML routing.

        This method is called by LiteLLM to select which deployment to use
        for a given request. Conduit analyzes the query and selects the
        optimal model based on learned performance characteristics.

        Args:
            model: Model group name (e.g., "gpt-4").
            messages: Chat messages for the request (OpenAI format).
            input: Alternative input format (for embeddings, etc.).
            specific_deployment: If True, must return exact deployment.
            request_kwargs: Additional request parameters.

        Returns:
            Selected deployment dictionary from litellm.router.model_list.

        Raises:
            RuntimeError: If router not initialized (shouldn't happen in normal usage).
        """
        # Initialize on first call
        if self._router is None:
            raise RuntimeError(
                "ConduitRoutingStrategy not properly initialized. "
                "Ensure LiteLLM router calls this method after setting the strategy."
            )

        await self._initialize_from_litellm(self._router)

        # Extract query text from messages or input
        query_text = self._extract_query_text(messages, input)

        # Route through Conduit
        query = Query(text=query_text)
        decision = await self.conduit_router.route(query)  # type: ignore[union-attr]

        logger.debug(
            f"Conduit selected {decision.selected_model} "
            f"(confidence: {decision.confidence:.2f})"
        )

        # Find matching deployment in LiteLLM's model_list
        for deployment in self._router.model_list:
            if deployment["model_info"]["id"] == decision.selected_model:
                # TODO: Store routing context for feedback loop (Issue #13)
                return deployment

        # Fallback 1: Find deployment matching model group name
        logger.warning(
            f"Conduit selected {decision.selected_model} but not found in model_list. "
            f"Falling back to model group '{model}'"
        )
        for deployment in self._router.model_list:
            if deployment.get("model_name") == model:
                return deployment

        # Fallback 2: Return first deployment (last resort)
        logger.error(
            f"No deployment found for model '{model}'. "
            f"Returning first deployment: {self._router.model_list[0]['model_info']['id']}"
        )
        return self._router.model_list[0]

    def get_available_deployment(
        self,
        model: str,
        messages: Optional[List[Dict[str, str]]] = None,
        input: Optional[Union[str, List[Any]]] = None,
        specific_deployment: Optional[bool] = False,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synchronous version of get_available_deployment.

        This method provides backward compatibility with LiteLLM's synchronous
        routing interface. It's recommended to use async_get_available_deployment
        for better performance.

        Args:
            model: Model group name (e.g., "gpt-4").
            messages: Chat messages for the request.
            input: Alternative input format.
            specific_deployment: If True, must return exact deployment.
            request_kwargs: Additional request parameters.

        Returns:
            Selected deployment dictionary from litellm.router.model_list.

        Note:
            This method uses asyncio.run() internally, which may not work
            in environments with existing event loops. Use async version when possible.
        """
        import asyncio

        try:
            # Try to get running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            return asyncio.run(
                self.async_get_available_deployment(
                    model, messages, input, specific_deployment, request_kwargs
                )
            )
        else:
            # Event loop already running, create task
            logger.warning(
                "Using sync get_available_deployment in async context. "
                "Consider using async_get_available_deployment instead."
            )
            return loop.run_until_complete(
                self.async_get_available_deployment(
                    model, messages, input, specific_deployment, request_kwargs
                )
            )

    def _extract_query_text(
        self,
        messages: Optional[List[Dict[str, str]]],
        input: Optional[Union[str, List[Any]]]
    ) -> str:
        """Extract query text from LiteLLM request format.

        Args:
            messages: Chat messages (OpenAI format).
            input: Alternative input format.

        Returns:
            Extracted query text for Conduit analysis.
        """
        if messages:
            # Get last user message content
            return messages[-1].get("content", "")
        elif isinstance(input, str):
            return input
        elif isinstance(input, list):
            # Join list elements for batch requests
            return " ".join(str(x) for x in input)
        return ""

    async def record_feedback(
        self,
        deployment_id: str,
        cost: float,
        latency: float,
        quality_score: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """Record feedback to update Conduit's bandit algorithm.

        This method enables Conduit to learn from LiteLLM request outcomes,
        improving routing decisions over time.

        Args:
            deployment_id: ID of deployment that was used.
            cost: Request cost from LiteLLM.
            latency: Request latency from LiteLLM.
            quality_score: Optional explicit quality rating (0-1).
            error: Optional error message if request failed.

        Note:
            Implementation in Issue #13 (feedback collection and learning).
        """
        # TODO: Implement feedback loop in Issue #13
        pass
