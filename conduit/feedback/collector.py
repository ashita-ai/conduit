"""Feedback collector for production feedback integration.

This module provides the FeedbackCollector class which orchestrates
feedback collection and bandit updates. It supports:
- Delayed feedback (track query, receive feedback later)
- Immediate feedback (no tracking needed)
- Multiple signal types via pluggable adapters
- Persistent storage for production reliability

Usage:
    >>> from conduit.feedback.collector import FeedbackCollector
    >>> from conduit.feedback.stores import InMemoryFeedbackStore
    >>> from conduit.engines.router import Router
    >>>
    >>> # Setup
    >>> router = Router()
    >>> store = InMemoryFeedbackStore()
    >>> collector = FeedbackCollector(router, store=store)
    >>>
    >>> # Route query
    >>> decision = await router.route(query)
    >>>
    >>> # Track for delayed feedback
    >>> await collector.track(decision)
    >>>
    >>> # Later: User provides feedback
    >>> from conduit.feedback.models import FeedbackEvent
    >>> event = FeedbackEvent(
    ...     query_id=decision.query_id,
    ...     signal_type="thumbs",
    ...     payload={"value": "up"}
    ... )
    >>> success = await collector.record(event)
"""

import logging
from typing import TYPE_CHECKING, Any

from conduit.core.models import QueryFeatures, RoutingDecision
from conduit.feedback.adapters import (
    CompletionTimeAdapter,
    FeedbackAdapter,
    QualityScoreAdapter,
    RatingAdapter,
    RegenerationAdapter,
    TaskSuccessAdapter,
    ThumbsAdapter,
)
from conduit.feedback.models import (
    FeedbackEvent,
    PendingQuery,
    RewardMapping,
    SessionFeedback,
)
from conduit.feedback.stores import FeedbackStore, InMemoryFeedbackStore

if TYPE_CHECKING:
    from conduit.engines.router import Router

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Collects and processes feedback for bandit learning.

    The FeedbackCollector is the main entry point for integrating
    user feedback into Conduit's bandit learning system. It supports
    both immediate and delayed feedback scenarios.

    Features:
        - Multiple signal types via adapters (thumbs, rating, task_success, etc.)
        - Delayed feedback (track query, receive signal later)
        - Persistent storage for production reliability
        - Automatic adapter registration for built-in signal types

    Architecture:
        1. Track: Store query metadata (model, features, cost, latency)
        2. Record: Convert feedback signal → reward via adapter
        3. Update: Call Router.update() with reward and features

    Example:
        >>> router = Router()
        >>> collector = FeedbackCollector(router)
        >>>
        >>> # Route and execute query
        >>> decision = await router.route(query)
        >>> response = await execute_llm_call(decision.selected_model)
        >>>
        >>> # Track for delayed feedback
        >>> await collector.track(decision, cost=response.cost, latency=response.latency)
        >>>
        >>> # Later: Record user feedback
        >>> await collector.record(FeedbackEvent(
        ...     query_id=decision.query_id,
        ...     signal_type="thumbs",
        ...     payload={"value": "up"}
        ... ))
    """

    def __init__(
        self,
        router: "Router",
        store: FeedbackStore | None = None,
        default_ttl: int = 3600,
    ):
        """Initialize feedback collector.

        Args:
            router: Router instance for bandit updates
            store: Optional persistent storage for pending queries
                   (defaults to InMemoryFeedbackStore)
            default_ttl: Default TTL for pending queries (default 1 hour)
        """
        self.router = router
        self.store = store or InMemoryFeedbackStore(default_ttl=default_ttl)
        self.default_ttl = default_ttl
        self._adapters: dict[str, FeedbackAdapter] = {}

        # Register default adapters
        self._register_default_adapters()

    def _register_default_adapters(self) -> None:
        """Register built-in feedback adapters."""
        self.register(ThumbsAdapter())
        self.register(RatingAdapter())
        self.register(TaskSuccessAdapter())
        self.register(QualityScoreAdapter())
        self.register(CompletionTimeAdapter())
        self.register(RegenerationAdapter())

    def register(self, adapter: FeedbackAdapter) -> None:
        """Register a feedback adapter.

        Args:
            adapter: FeedbackAdapter instance to register

        Example:
            >>> class CustomAdapter(FeedbackAdapter):
            ...     signal_type = "custom"
            ...     def to_reward(self, event):
            ...         return RewardMapping(reward=0.5, confidence=1.0)
            >>>
            >>> collector.register(CustomAdapter())
        """
        self._adapters[adapter.signal_type] = adapter
        logger.debug(f"Registered feedback adapter: {adapter.signal_type}")

    def get_adapter(self, signal_type: str) -> FeedbackAdapter | None:
        """Get adapter for signal type.

        Args:
            signal_type: Signal type to look up

        Returns:
            FeedbackAdapter if registered, None otherwise
        """
        return self._adapters.get(signal_type)

    @property
    def registered_signals(self) -> list[str]:
        """List of registered signal types."""
        return list(self._adapters.keys())

    async def track(
        self,
        decision: RoutingDecision,
        cost: float = 0.0,
        latency: float = 0.0,
        ttl_seconds: int | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track a routed query for delayed feedback.

        Call this after routing (and optionally after execution) to
        store query metadata. Later, when feedback arrives, call
        record() with the query_id.

        Args:
            decision: RoutingDecision from router.route()
            cost: Actual or estimated cost (can update later)
            latency: Actual latency (can update later)
            ttl_seconds: Optional custom TTL (defaults to default_ttl)
            session_id: Optional session ID for multi-turn tracking.
                Use this to group related queries (e.g., a conversation)
                so session-level feedback can propagate to all queries.

        Example:
            >>> # Single query tracking
            >>> decision = await router.route(query)
            >>> response = await execute_llm_call(decision.selected_model)
            >>> await collector.track(decision, cost=response.cost, latency=response.latency)
            >>>
            >>> # Multi-turn session tracking
            >>> session_id = "conversation_123"
            >>> for query in conversation:
            ...     decision = await router.route(query)
            ...     await collector.track(decision, session_id=session_id)
            >>>
            >>> # Later: Record session-level feedback
            >>> await collector.record_session_feedback(
            ...     session_id=session_id,
            ...     signal_type="rating",
            ...     payload={"rating": 4}
            ... )
        """
        pending = PendingQuery(
            query_id=decision.query_id,
            model_id=decision.selected_model,
            features=decision.features.model_dump(),
            cost=cost,
            latency=latency,
            ttl_seconds=ttl_seconds or self.default_ttl,
            session_id=session_id,
        )

        await self.store.save_pending(pending)
        session_info = f", session={session_id[:8]}..." if session_id else ""
        logger.debug(
            f"Tracking query for feedback: {decision.query_id[:8]}...{session_info}"
        )

    async def record(self, event: FeedbackEvent) -> bool:
        """Record feedback for a tracked query.

        Atomically retrieves and deletes the pending query, converts the
        feedback signal to a reward via the appropriate adapter, and
        updates the router's bandit weights.

        Idempotency:
            This method is idempotent: recording the same feedback multiple
            times has the same effect as recording it once. This prevents
            double-counting from retries or duplicate submissions.

            - If event.idempotency_key is set, it's used for deduplication
            - Otherwise, query_id:signal_type is used as natural key
            - Duplicate calls return True (success, already processed)

        Args:
            event: FeedbackEvent with query_id and signal

        Returns:
            True if feedback was recorded successfully (or already recorded)

        Failure Reasons:
            - Unknown signal_type (no adapter registered)
            - Unknown query_id (not tracked or expired)
            - Router update failed

        Note:
            Uses atomic get_and_delete to prevent race conditions when
            multiple feedback events arrive for the same query.

        Example:
            >>> event = FeedbackEvent(
            ...     query_id="q123",
            ...     signal_type="thumbs",
            ...     payload={"value": "up"}
            ... )
            >>> success = await collector.record(event)
            >>> # Calling again is safe (idempotent)
            >>> success = await collector.record(event)  # Still returns True
        """
        # Check idempotency
        idempotency_key = event.get_idempotency_key()
        if await self.store.was_processed(idempotency_key):
            logger.debug(f"Duplicate feedback ignored: {idempotency_key}")
            return True  # Idempotent success

        # Get adapter for signal type
        adapter = self._adapters.get(event.signal_type)
        if not adapter:
            logger.warning(f"No adapter for signal type: {event.signal_type}")
            return False

        # Atomically get and delete pending query (prevents race conditions)
        pending = await self.store.get_and_delete_pending(event.query_id)
        if not pending:
            logger.warning(f"Unknown or expired query_id: {event.query_id}")
            return False

        # Convert signal to reward
        mapping = adapter.to_reward(event)

        # Reconstruct QueryFeatures from stored dict
        features = QueryFeatures(**pending.features)

        # Update router with confidence-weighted feedback
        try:
            await self.router.update(
                model_id=pending.model_id,
                cost=pending.cost,
                quality_score=mapping.reward,
                latency=pending.latency,
                features=features,
                confidence=mapping.confidence,
            )
        except Exception as e:
            logger.error(f"Failed to update router: {e}")
            return False

        # Mark as processed for idempotency
        await self.store.mark_processed(idempotency_key, ttl_seconds=self.default_ttl)

        logger.info(
            f"Feedback recorded: query={event.query_id[:8]}..., "
            f"signal={event.signal_type}, reward={mapping.reward:.2f}, "
            f"confidence={mapping.confidence:.2f}"
        )
        return True

    async def record_immediate(
        self,
        model_id: str,
        features: QueryFeatures,
        signal_type: str,
        payload: dict[str, Any],
        cost: float = 0.0,
        latency: float = 0.0,
    ) -> bool:
        """Record feedback immediately (no tracking required).

        Use this for synchronous feedback scenarios where feedback
        is available immediately after the response.

        Args:
            model_id: Model that generated the response
            features: QueryFeatures from routing decision
            signal_type: Type of feedback signal
            payload: Signal-specific data
            cost: Query cost
            latency: Response latency

        Returns:
            True if feedback was recorded successfully

        Example:
            >>> # User provides immediate thumbs up
            >>> success = await collector.record_immediate(
            ...     model_id=decision.selected_model,
            ...     features=decision.features,
            ...     signal_type="thumbs",
            ...     payload={"value": "up"},
            ...     cost=0.001,
            ...     latency=0.5,
            ... )
        """
        # Get adapter
        adapter = self._adapters.get(signal_type)
        if not adapter:
            logger.warning(f"No adapter for signal type: {signal_type}")
            return False

        # Create event for adapter
        event = FeedbackEvent(
            query_id="immediate",  # Placeholder for immediate feedback
            signal_type=signal_type,
            payload=payload,
        )

        # Convert to reward
        mapping = adapter.to_reward(event)

        # Update router with confidence-weighted feedback
        try:
            await self.router.update(
                model_id=model_id,
                cost=cost,
                quality_score=mapping.reward,
                latency=latency,
                features=features,
                confidence=mapping.confidence,
            )
        except Exception as e:
            logger.error(f"Failed to update router: {e}")
            return False

        logger.info(
            f"Immediate feedback recorded: model={model_id}, "
            f"signal={signal_type}, reward={mapping.reward:.2f}, "
            f"confidence={mapping.confidence:.2f}"
        )
        return True

    async def record_batch(self, events: list[FeedbackEvent]) -> dict[str, bool]:
        """Record multiple feedback events.

        Args:
            events: List of FeedbackEvent objects

        Returns:
            Dict mapping query_id to success status

        Example:
            >>> events = [
            ...     FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"}),
            ...     FeedbackEvent(query_id="q2", signal_type="rating", payload={"rating": 4}),
            ... ]
            >>> results = await collector.record_batch(events)
            >>> print(results)  # {"q1": True, "q2": True}
        """
        results = {}
        for event in events:
            results[event.query_id] = await self.record(event)
        return results

    async def record_aggregated(
        self,
        query_id: str,
        events: list[FeedbackEvent],
    ) -> bool:
        """Record multiple feedback signals for a single query, aggregated.

        Use this when you have multiple feedback signals for the same query
        (e.g., thumbs up AND task success) and want to combine them into
        a single bandit update.

        Idempotency:
            Uses a composite key: query_id:aggregated:signal1+signal2+...
            Duplicate calls return True (success, already processed).

        Aggregation Method:
            Signals are combined using confidence-weighted averaging:
            - Each signal produces (reward, confidence) via its adapter
            - Final reward = sum(reward_i * confidence_i) / sum(confidence_i)
            - Final confidence = average of individual confidences

            This ensures high-confidence signals dominate while low-confidence
            signals contribute proportionally less.

        Args:
            query_id: Query ID for all events (must match tracked query)
            events: List of FeedbackEvent objects for the same query

        Returns:
            True if feedback was recorded successfully

        Example:
            >>> # User gave thumbs up AND the task succeeded
            >>> events = [
            ...     FeedbackEvent(query_id="q1", signal_type="thumbs", payload={"value": "up"}),
            ...     FeedbackEvent(query_id="q1", signal_type="task_success", payload={"success": True}),
            ... ]
            >>> success = await collector.record_aggregated("q1", events)
            >>> # Single update with aggregated reward

        Signal Combination Examples:
            - thumbs_up(1.0, 1.0) + task_success(1.0, 1.0) = (1.0, 1.0)
            - thumbs_up(1.0, 1.0) + regeneration(0.0, 0.8) = (0.56, 0.9)
            - rating_4/5(0.75, 1.0) + task_success(1.0, 1.0) = (0.875, 1.0)
        """
        if not events:
            logger.warning(f"No events provided for query_id: {query_id}")
            return False

        # Create composite idempotency key
        signal_types_sorted = sorted({e.signal_type for e in events})
        idempotency_key = f"{query_id}:aggregated:{'+'.join(signal_types_sorted)}"

        # Check idempotency
        if await self.store.was_processed(idempotency_key):
            logger.debug(f"Duplicate aggregated feedback ignored: {idempotency_key}")
            return True  # Idempotent success

        # Get pending query (peek without deleting)
        pending = await self.store.get_pending(query_id)
        if not pending:
            logger.warning(f"Unknown or expired query_id: {query_id}")
            return False

        # Convert all events to reward mappings
        mappings: list[RewardMapping] = []
        for event in events:
            adapter = self._adapters.get(event.signal_type)
            if not adapter:
                logger.warning(
                    f"No adapter for signal type: {event.signal_type}, skipping"
                )
                continue
            mappings.append(adapter.to_reward(event))

        if not mappings:
            logger.warning(f"No valid signals for query_id: {query_id}")
            return False

        # Aggregate using confidence-weighted average
        total_weighted_reward = sum(m.reward * m.confidence for m in mappings)
        total_confidence = sum(m.confidence for m in mappings)

        if total_confidence == 0:
            logger.warning(f"Zero total confidence for query_id: {query_id}")
            return False

        aggregated_reward = total_weighted_reward / total_confidence
        aggregated_confidence = total_confidence / len(mappings)

        # Now atomically get and delete
        pending = await self.store.get_and_delete_pending(query_id)
        if not pending:
            # Race condition: another process got it first
            logger.warning(f"Query already processed: {query_id}")
            return False

        # Reconstruct QueryFeatures
        features = QueryFeatures(**pending.features)

        # Update router with aggregated feedback
        try:
            await self.router.update(
                model_id=pending.model_id,
                cost=pending.cost,
                quality_score=aggregated_reward,
                latency=pending.latency,
                features=features,
                confidence=aggregated_confidence,
            )
        except Exception as e:
            logger.error(f"Failed to update router with aggregated feedback: {e}")
            return False

        # Mark as processed for idempotency
        await self.store.mark_processed(idempotency_key, ttl_seconds=self.default_ttl)

        signal_types = [
            e.signal_type for e in events if self._adapters.get(e.signal_type)
        ]
        logger.info(
            f"Aggregated feedback recorded: query={query_id[:8]}..., "
            f"signals={signal_types}, "
            f"aggregated_reward={aggregated_reward:.2f}, "
            f"aggregated_confidence={aggregated_confidence:.2f}"
        )
        return True

    async def get_pending_count(self) -> int:
        """Get count of pending queries awaiting feedback.

        Returns:
            Number of pending queries
        """
        return await self.store.count_pending()

    async def update_tracked(
        self,
        query_id: str,
        cost: float | None = None,
        latency: float | None = None,
    ) -> bool:
        """Update cost/latency for a tracked query after LLM execution.

        Use this when you need to track queries before execution but want
        to record actual cost/latency after the response arrives.

        Args:
            query_id: Query ID to update
            cost: Actual cost from LLM response
            latency: Actual latency from LLM response

        Returns:
            True if updated, False if query_id not found

        Example:
            >>> # Track before execution (with estimate)
            >>> await collector.track(decision, cost=0.0, latency=0.0)
            >>>
            >>> # Execute LLM call
            >>> response = await execute_llm_call(decision.selected_model)
            >>>
            >>> # Update with actual values
            >>> await collector.update_tracked(
            ...     decision.query_id,
            ...     cost=response.usage.total_cost,
            ...     latency=response.elapsed_time,
            ... )
        """
        return await self.store.update_pending(query_id, cost=cost, latency=latency)

    async def cleanup_expired(self) -> int:
        """Remove expired pending queries.

        Call periodically in long-running applications to clean up
        queries that never received feedback.

        Returns:
            Number of expired queries removed
        """
        return await self.store.cleanup_expired()

    async def record_session_feedback(
        self,
        session_id: str,
        signal_type: str,
        payload: dict[str, Any],
        propagation_weight: float = 0.5,
    ) -> dict[str, bool]:
        """Record session-level feedback that propagates to all queries in the session.

        Use this for multi-turn conversations where the user provides overall
        feedback at the end of a session. The feedback is applied to all queries
        in the session with the specified propagation weight.

        Idempotency:
            Uses composite key: session:{session_id}:{signal_type}
            Duplicate calls return empty dict with logged warning.

        Propagation Logic:
            For each query in the session:
            1. Convert session signal to reward via adapter
            2. Apply propagation_weight to confidence (softer update)
            3. Update router for each model used in the session

            This ensures models used in a good conversation get credit, and
            models used in a bad conversation get penalized, proportionally
            to the propagation weight.

        Args:
            session_id: Session identifier (must match session_id used in track())
            signal_type: Feedback signal type (thumbs, rating, task_success, etc.)
            payload: Signal-specific data
            propagation_weight: How strongly to apply session feedback (0-1).
                0.0 = no effect, 1.0 = full effect. Default 0.5.

        Returns:
            Dict mapping query_id to success status

        Example:
            >>> # Track queries in a conversation
            >>> session_id = "conversation_abc"
            >>> for query in conversation:
            ...     decision = await router.route(query)
            ...     await collector.track(decision, session_id=session_id)
            ...
            >>> # User rates the overall conversation
            >>> results = await collector.record_session_feedback(
            ...     session_id=session_id,
            ...     signal_type="rating",
            ...     payload={"rating": 4},  # 4/5 stars
            ...     propagation_weight=0.5,  # 50% weight
            ... )
            >>> print(results)  # {"q1": True, "q2": True, "q3": True}
        """
        # Check idempotency
        idempotency_key = f"session:{session_id}:{signal_type}"
        if await self.store.was_processed(idempotency_key):
            logger.debug(f"Duplicate session feedback ignored: {idempotency_key}")
            return {}  # Already processed

        # Get adapter
        adapter = self._adapters.get(signal_type)
        if not adapter:
            logger.warning(f"No adapter for signal type: {signal_type}")
            return {}

        # Atomically get and delete all session queries
        queries = await self.store.get_and_delete_session(session_id)
        if not queries:
            logger.warning(f"No queries found for session: {session_id}")
            return {}

        # Create feedback event for adapter
        event = FeedbackEvent(
            query_id=f"session_{session_id}",
            signal_type=signal_type,
            payload=payload,
        )

        # Convert to reward
        mapping = adapter.to_reward(event)

        # Apply propagation weight to confidence
        propagated_confidence = mapping.confidence * propagation_weight

        # Update router for each query in the session
        results = {}
        for pending in queries:
            features = QueryFeatures(**pending.features)

            try:
                await self.router.update(
                    model_id=pending.model_id,
                    cost=pending.cost,
                    quality_score=mapping.reward,
                    latency=pending.latency,
                    features=features,
                    confidence=propagated_confidence,
                )
                results[pending.query_id] = True
            except Exception as e:
                logger.error(
                    f"Failed to update router for query {pending.query_id}: {e}"
                )
                results[pending.query_id] = False

        # Mark as processed for idempotency
        await self.store.mark_processed(idempotency_key, ttl_seconds=self.default_ttl)

        success_count = sum(1 for v in results.values() if v)
        logger.info(
            f"Session feedback recorded: session={session_id[:8]}..., "
            f"signal={signal_type}, reward={mapping.reward:.2f}, "
            f"propagation_weight={propagation_weight}, "
            f"queries_updated={success_count}/{len(queries)}"
        )
        return results

    async def get_session_queries(self, session_id: str) -> list[PendingQuery]:
        """Get all pending queries for a session (non-destructive).

        Use this to inspect what queries are tracked for a session
        before recording feedback.

        Args:
            session_id: Session identifier

        Returns:
            List of PendingQuery objects in the session
        """
        return await self.store.get_session_queries(session_id)
