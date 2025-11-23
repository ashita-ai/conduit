"""LiteLLM feedback integration for Conduit bandit learning.

This module implements CustomLogger callback to capture LiteLLM response
metadata (cost, latency) and feed it back to Conduit's bandit algorithms,
enabling ML-based learning from actual usage.
"""

import logging
from typing import Any

from conduit.engines.bandits.base import BanditFeedback

logger = logging.getLogger(__name__)

try:
    from litellm.integrations.custom_logger import CustomLogger

    LITELLM_AVAILABLE = True
except ImportError:
    logger.warning(
        "LiteLLM not installed. Install with: pip install conduit[litellm]"
    )
    LITELLM_AVAILABLE = False

    # Stub base class for type checking
    class CustomLogger:  # type: ignore[no-redef]
        """Stub base class when LiteLLM is not available."""

        pass


class ConduitFeedbackLogger(CustomLogger):
    """Captures LiteLLM responses and updates Conduit bandit algorithms.

    This logger integrates with LiteLLM's callback system to:
    1. Extract cost/latency from completed requests
    2. Reconstruct query features from request messages
    3. Create BanditFeedback with quality estimation
    4. Update Conduit's bandit algorithm for learning

    The feedback loop enables Conduit to learn which models perform best
    for different query types, optimizing for quality, cost, and latency.

    Example:
        >>> from conduit.engines.router import Router
        >>> router = Router(use_hybrid=True)
        >>> logger = ConduitFeedbackLogger(router)
        >>> # Logger automatically updates bandit when LiteLLM requests complete
    """

    def __init__(self, conduit_router: Any):
        """Initialize feedback logger with Conduit router reference.

        Args:
            conduit_router: Conduit Router instance (provides analyzer and bandit)
        """
        super().__init__()
        self.router = conduit_router
        logger.info(
            f"ConduitFeedbackLogger initialized "
            f"(hybrid={self.router.hybrid_router is not None})"
        )

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Log successful LiteLLM completion and update bandit.

        Called by LiteLLM after successful completion. Extracts cost/latency
        from response, reconstructs query features, and updates bandit algorithm.

        Args:
            kwargs: LiteLLM request parameters (model, messages, etc.)
            response_obj: LiteLLM response object with cost metadata
            start_time: Request start timestamp (Unix time)
            end_time: Request end timestamp (Unix time)
        """
        try:
            # Extract query text from LiteLLM request
            query_text = self._extract_query_text(kwargs)
            if not query_text:
                logger.warning("No query text found in LiteLLM request, skipping feedback")
                return

            # Generate features using router's analyzer
            features = await self.router.analyzer.analyze(query_text)

            # Extract cost from LiteLLM response metadata
            cost = self._extract_cost(response_obj)
            if cost is None:
                logger.warning(
                    "Cost unavailable in LiteLLM response, skipping feedback "
                    "(prevents corrupting bandit with cost=0.0)"
                )
                return

            # Calculate latency
            latency = end_time - start_time

            # Get model ID and validate
            model_id = kwargs.get("model", "unknown")
            if model_id == "unknown":
                logger.warning("No model ID in LiteLLM response, skipping feedback")
                return

            # Validate model exists in router's arms
            if not self._validate_model_id(model_id):
                logger.warning(
                    f"Model '{model_id}' not in router arms, skipping feedback "
                    f"(available: {self._get_available_model_ids()})"
                )
                return

            # Create feedback with high quality (success = good response)
            feedback = BanditFeedback(
                model_id=model_id,
                cost=cost,
                quality_score=0.9,  # Success implies high quality
                latency=latency,
                success=True,
                metadata={
                    "source": "litellm",
                    "response_id": getattr(response_obj, "id", None),
                },
            )

            # Update appropriate router component
            await self._update_bandit(feedback, features)

            logger.debug(
                f"Feedback recorded: model={model_id}, cost=${cost:.6f}, "
                f"latency={latency:.2f}s, quality=0.9"
            )

        except Exception as e:
            logger.error(f"Error recording success feedback: {e}", exc_info=True)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Log failed LiteLLM completion and update bandit.

        Called by LiteLLM when completion fails. Records low quality score
        to teach bandit to avoid failing models.

        Args:
            kwargs: LiteLLM request parameters
            response_obj: LiteLLM error response
            start_time: Request start timestamp
            end_time: Request end timestamp
        """
        try:
            # Extract query text
            query_text = self._extract_query_text(kwargs)
            if not query_text:
                logger.warning("No query text found in failed request, skipping feedback")
                return

            # Generate features
            features = await self.router.analyzer.analyze(query_text)

            # Extract cost (may be None for failures)
            cost = self._extract_cost(response_obj)
            # For failures, cost might be unavailable - use 0.0 since quality is already low
            if cost is None:
                cost = 0.0

            # Calculate latency
            latency = end_time - start_time

            # Get model ID and validate
            model_id = kwargs.get("model", "unknown")
            if model_id == "unknown":
                logger.warning("No model ID in failed request, skipping feedback")
                return

            # Validate model exists in router's arms
            if not self._validate_model_id(model_id):
                logger.warning(
                    f"Model '{model_id}' not in router arms, skipping failure feedback"
                )
                return

            # Create feedback with low quality (failure = poor response)
            feedback = BanditFeedback(
                model_id=model_id,
                cost=cost,
                quality_score=0.1,  # Failure implies low quality
                latency=latency,
                success=False,
                metadata={
                    "source": "litellm",
                    "error": str(response_obj),
                },
            )

            # Update bandit
            await self._update_bandit(feedback, features)

            logger.debug(
                f"Failure feedback recorded: model={model_id}, latency={latency:.2f}s, "
                f"quality=0.1"
            )

        except Exception as e:
            logger.error(f"Error recording failure feedback: {e}", exc_info=True)

    def _extract_query_text(self, kwargs: dict[str, Any]) -> str:
        """Extract query text from LiteLLM request parameters.

        Args:
            kwargs: LiteLLM request parameters

        Returns:
            Extracted query text or empty string
        """
        # Try messages format (chat completions)
        messages = kwargs.get("messages")
        if messages and isinstance(messages, list) and len(messages) > 0:
            # Get last user message
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                return last_msg.get("content", "")

        # Try input format (embeddings, etc.)
        input_data = kwargs.get("input")
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, list):
            # Join list elements
            return " ".join(str(x) for x in input_data)

        return ""

    def _extract_cost(self, response_obj: Any) -> float | None:
        """Extract cost from LiteLLM response object.

        Args:
            response_obj: LiteLLM response object

        Returns:
            Cost in USD, or None if unavailable (prevents corrupting bandit with 0.0)
        """
        try:
            # LiteLLM stores cost in _hidden_params
            if hasattr(response_obj, "_hidden_params"):
                hidden = response_obj._hidden_params
                if isinstance(hidden, dict):
                    cost = hidden.get("response_cost")
                    if cost is not None:
                        return float(cost)

            # Fallback: Try response_cost attribute
            if hasattr(response_obj, "response_cost"):
                cost = response_obj.response_cost
                if cost is not None:
                    return float(cost)

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to extract cost from response: {e}")

        return None

    def _validate_model_id(self, model_id: str) -> bool:
        """Validate that model_id exists in router's available arms.

        Args:
            model_id: Model identifier to validate

        Returns:
            True if model exists in arms, False otherwise
        """
        # Check hybrid router arms
        if self.router.hybrid_router is not None:
            if hasattr(self.router.hybrid_router, "linucb_bandit"):
                bandit = self.router.hybrid_router.linucb_bandit
                return model_id in bandit.arms if bandit else False
            if hasattr(self.router.hybrid_router, "ucb1_bandit"):
                bandit = self.router.hybrid_router.ucb1_bandit
                return model_id in bandit.arms if bandit else False

        # Check standard bandit arms
        if self.router.bandit is not None:
            return model_id in self.router.bandit.arms

        return False

    def _get_available_model_ids(self) -> list[str]:
        """Get list of available model IDs from router arms.

        Returns:
            List of model IDs available in router
        """
        # Check hybrid router
        if self.router.hybrid_router is not None:
            if hasattr(self.router.hybrid_router, "linucb_bandit"):
                bandit = self.router.hybrid_router.linucb_bandit
                return list(bandit.arms.keys()) if bandit else []
            if hasattr(self.router.hybrid_router, "ucb1_bandit"):
                bandit = self.router.hybrid_router.ucb1_bandit
                return list(bandit.arms.keys()) if bandit else []

        # Check standard bandit
        if self.router.bandit is not None:
            return list(self.router.bandit.arms.keys())

        return []

    async def _update_bandit(self, feedback: BanditFeedback, features: Any) -> None:
        """Update appropriate bandit algorithm (hybrid or standard).

        Args:
            feedback: Bandit feedback to record
            features: Query features for contextual learning
        """
        # Hybrid mode: Update HybridRouter
        if self.router.hybrid_router is not None:
            await self.router.hybrid_router.update(feedback, features)
            logger.debug("Updated HybridRouter with feedback")
            return

        # Standard mode: Update ContextualBandit
        if self.router.bandit is not None:
            await self.router.bandit.update(feedback, features)
            logger.debug("Updated ContextualBandit with feedback")
            return

        # No bandit available (shouldn't happen)
        logger.warning(
            "No bandit algorithm available for feedback update "
            "(neither hybrid_router nor bandit found)"
        )

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Sync wrapper for async_log_success_event (not recommended).

        Note:
            LiteLLM prefers async callbacks when using async methods.
            This sync version is provided for compatibility but may have
            performance implications.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, use asyncio.run()
            asyncio.run(
                self.async_log_success_event(kwargs, response_obj, start_time, end_time)
            )
        else:
            # Loop already running, create task
            loop.create_task(
                self.async_log_success_event(kwargs, response_obj, start_time, end_time)
            )

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Sync wrapper for async_log_failure_event (not recommended).

        Note:
            See log_success_event() for async vs sync notes.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(
                self.async_log_failure_event(kwargs, response_obj, start_time, end_time)
            )
        else:
            loop.create_task(
                self.async_log_failure_event(kwargs, response_obj, start_time, end_time)
            )
