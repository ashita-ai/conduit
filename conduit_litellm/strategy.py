"""LiteLLM routing strategy using Conduit's ML-powered model selection."""

import logging
from typing import Any, cast

from conduit.core.models import Query
from conduit.engines.router import Router
from conduit_litellm.feedback import ConduitFeedbackLogger
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

    IMPORTANT: LiteLLM's set_custom_routing_strategy() doesn't provide a clean way
    to pass the router reference to the strategy. You must use the setup_strategy()
    helper method instead:

    Example:
        >>> from litellm import Router
        >>> from conduit_litellm import ConduitRoutingStrategy
        >>>
        >>> router = Router(model_list=[...])
        >>> strategy = ConduitRoutingStrategy(use_hybrid=True)
        >>> ConduitRoutingStrategy.setup_strategy(router, strategy)  # Use helper!
        >>>
        >>> # Now LiteLLM uses Conduit's ML routing
        >>> response = await router.acompletion(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
    """

    def __init__(
        self,
        conduit_router: Router | None = None,
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
                - evaluator (ArbiterEvaluator): Optional LLM-as-judge evaluator for quality assessment

        Example:
            >>> # Use hybrid routing with caching and LLM-as-judge
            >>> from conduit.evaluation import ArbiterEvaluator
            >>> evaluator = ArbiterEvaluator(db, sample_rate=0.1)
            >>> strategy = ConduitRoutingStrategy(
            ...     use_hybrid=True,
            ...     cache_enabled=True,
            ...     evaluator=evaluator  # Enable LLM-as-judge quality measurement
            ... )
        """
        super().__init__()
        self.conduit_router = conduit_router
        # Extract evaluator before passing config to Router
        self.evaluator = conduit_config.pop('evaluator', None)
        self.conduit_config = conduit_config
        self._initialized = False
        self._router: Any | None = None  # LiteLLM router reference
        self.feedback_logger: ConduitFeedbackLogger | None = None  # Feedback integration
        self._feedback_registered = False  # Track if feedback logger is registered

    @staticmethod
    def setup_strategy(router: Any, strategy: "ConduitRoutingStrategy") -> None:
        """Set up strategy on LiteLLM router with proper initialization.

        This is a helper method that works around LiteLLM's design where
        set_custom_routing_strategy() doesn't provide the router reference
        to the strategy instance.

        Args:
            router: LiteLLM Router instance
            strategy: ConduitRoutingStrategy instance to install

        Example:
            >>> router = Router(model_list=[...])
            >>> strategy = ConduitRoutingStrategy(use_hybrid=True)
            >>> ConduitRoutingStrategy.setup_strategy(router, strategy)
        """
        # Store router reference before binding
        strategy._router = router
        # Now set the strategy (LiteLLM will bind methods to router)
        router.set_custom_routing_strategy(strategy)

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

        # Initialize feedback logger for bandit learning
        self._initialize_feedback_logger()

        self._initialized = True

    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list[Any] | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
        # Note: _router should be set via setup_strategy() helper method
        await self._initialize_from_litellm(self._router)

        # Extract query text from messages or input
        query_text = self._extract_query_text(messages, input)

        # Route through Conduit
        query = Query(text=query_text, user_id=None, context=None, constraints=None)
        decision = await self.conduit_router.route(query)  # type: ignore[union-attr]

        logger.debug(
            f"Conduit selected {decision.selected_model} "
            f"(confidence: {decision.confidence:.2f})"
        )

        # Find matching deployment in LiteLLM's model_list
        if self._router is None:
            raise RuntimeError("Router not initialized. Use setup_strategy() to initialize.")

        for deployment in self._router.model_list:
            if deployment["model_info"]["id"] == decision.selected_model:
                # TODO: Store routing context for feedback loop (Issue #13)
                return cast(dict[str, Any], deployment)

        # Fallback 1: Find deployment matching model group name
        logger.warning(
            f"Conduit selected {decision.selected_model} but not found in model_list. "
            f"Falling back to model group '{model}'"
        )
        for deployment in self._router.model_list:
            if deployment.get("model_name") == model:
                return cast(dict[str, Any], deployment)

        # Fallback 2: Return first deployment (last resort)
        logger.error(
            f"No deployment found for model '{model}'. "
            f"Returning first deployment: {self._router.model_list[0]['model_info']['id']}"
        )
        return cast(dict[str, Any], self._router.model_list[0])

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,
        input: str | list[Any] | None = None,
        specific_deployment: bool | None = False,
        request_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            This method handles both sync and async contexts correctly.
            Use async_get_available_deployment when possible for better performance.
        """
        import asyncio
        import concurrent.futures

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
            # Event loop already running - run in thread to avoid RuntimeError
            logger.warning(
                "Using sync get_available_deployment in async context. "
                "Consider using async_get_available_deployment instead."
            )

            # Run async function in a new event loop in a separate thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self.async_get_available_deployment(
                        model, messages, input, specific_deployment, request_kwargs
                    )
                )
                return future.result()

    def _extract_query_text(
        self,
        messages: list[dict[str, str]] | None,
        input: str | list[Any] | None
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

    def _initialize_feedback_logger(self) -> None:
        """Initialize and register feedback logger with LiteLLM.

        Creates ConduitFeedbackLogger and registers it with LiteLLM's callback
        system. The logger captures response metadata (cost, latency) and feeds
        it back to Conduit's bandit algorithms for learning.

        Prevents duplicate registration: If a logger from this strategy is already
        registered, it will be removed before adding the new one.

        Note:
            Requires LiteLLM to be installed. Silently skips if unavailable.
        """
        try:
            # Import litellm to register callbacks
            import litellm

            # Remove any existing logger from this strategy to prevent duplicates
            if self.feedback_logger is not None and self._feedback_registered:
                self._unregister_feedback_logger()

            # Create feedback logger with optional evaluator
            self.feedback_logger = ConduitFeedbackLogger(
                self.conduit_router,
                evaluator=self.evaluator
            )

            # Register with LiteLLM's callback system
            # LiteLLM will call async_log_success_event and async_log_failure_event
            if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
                litellm.callbacks = []

            # Check if this exact logger instance is already registered (shouldn't happen)
            if self.feedback_logger not in litellm.callbacks:
                litellm.callbacks.append(self.feedback_logger)
                self._feedback_registered = True
                logger.info(
                    "Feedback logger registered with LiteLLM - bandit learning enabled"
                )
            else:
                logger.warning("Feedback logger already registered, skipping duplicate")

        except ImportError:
            logger.warning(
                "LiteLLM not available, feedback loop disabled. "
                "Install with: pip install conduit[litellm]"
            )
        except Exception as e:
            logger.error(f"Failed to initialize feedback logger: {e}", exc_info=True)

    def _unregister_feedback_logger(self) -> None:
        """Remove feedback logger from LiteLLM callbacks.

        Internal cleanup method called before re-registering or during cleanup.
        """
        try:
            import litellm

            if (
                self.feedback_logger is not None
                and hasattr(litellm, "callbacks")
                and litellm.callbacks is not None
            ):
                if self.feedback_logger in litellm.callbacks:
                    litellm.callbacks.remove(self.feedback_logger)
                    self._feedback_registered = False
                    logger.debug("Feedback logger unregistered from LiteLLM")
        except Exception as e:
            logger.warning(f"Failed to unregister feedback logger: {e}")

    def cleanup(self) -> None:
        """Clean up resources and unregister callbacks.

        Call this method when done using the strategy to properly release resources
        and remove the feedback logger from LiteLLM's global callback list.

        Example:
            >>> strategy = ConduitRoutingStrategy(use_hybrid=True)
            >>> ConduitRoutingStrategy.setup_strategy(router, strategy)
            >>> try:
            ...     # Use strategy
            ...     await router.acompletion(...)
            ... finally:
            ...     strategy.cleanup()
        """
        self._unregister_feedback_logger()
        logger.info("ConduitRoutingStrategy cleaned up")

    async def record_feedback(
        self,
        deployment_id: str,
        cost: float,
        latency: float,
        quality_score: float | None = None,
        error: str | None = None
    ) -> None:
        """Record manual feedback to update Conduit's bandit algorithm.

        This method allows explicit feedback submission in addition to automatic
        feedback captured by ConduitFeedbackLogger from LiteLLM responses.

        Args:
            deployment_id: ID of deployment that was used.
            cost: Request cost from LiteLLM.
            latency: Request latency from LiteLLM.
            quality_score: Optional explicit quality rating (0-1).
            error: Optional error message if request failed.

        Note:
            Automatic feedback from LiteLLM responses is preferred and happens
            transparently via ConduitFeedbackLogger. Use this method only for
            manual overrides or custom feedback scenarios.
        """
        # Manual feedback implementation deferred - automatic feedback via
        # ConduitFeedbackLogger is now the primary mechanism (Issue #13 complete)
        logger.info(
            f"Manual feedback received but not implemented: "
            f"deployment={deployment_id}, cost={cost}, latency={latency}, "
            f"quality={quality_score}, error={error}"
        )
        logger.info(
            "Automatic feedback via ConduitFeedbackLogger is active and preferred. "
            "Manual feedback support may be added in future if needed."
        )
