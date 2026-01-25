"""Graceful shutdown handling for Conduit production deployments.

This module provides signal handling and shutdown coordination to ensure:
- No data loss: Bandit state is persisted before exit
- No broken connections: Database pools are closed properly
- No dropped requests: In-flight requests complete or timeout
- Auditability: Shutdown progress is logged at INFO level

The LifecycleManager integrates with FastAPI's lifespan events and can be
used standalone for CLI applications.

Usage with FastAPI:
    >>> from conduit.core.lifecycle import LifecycleManager
    >>>
    >>> manager = LifecycleManager(
    ...     router=router,
    ...     database=database,
    ...     shutdown_timeout=30.0,
    ... )
    >>>
    >>> @asynccontextmanager
    >>> async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ...     manager.install_signal_handlers()
    ...     yield
    ...     await manager.shutdown()

Standalone usage:
    >>> manager = LifecycleManager(router=router, database=database)
    >>> manager.install_signal_handlers()
    >>> # ... application runs ...
    >>> await manager.shutdown()
"""

import asyncio
import logging
import signal
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.core.database import Database
    from conduit.engines.router import Router

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Shutdown phases for tracking progress."""

    NOT_STARTED = "not_started"
    SIGNAL_RECEIVED = "signal_received"
    DRAINING_REQUESTS = "draining_requests"
    PERSISTING_STATE = "persisting_state"
    CLOSING_CONNECTIONS = "closing_connections"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ShutdownState:
    """Tracks shutdown progress and timing."""

    phase: ShutdownPhase = ShutdownPhase.NOT_STARTED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    signal_received: str | None = None
    in_flight_requests: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Calculate shutdown duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class LifecycleManager:
    """Manages application lifecycle with graceful shutdown support.

    Coordinates shutdown sequence:
    1. Receive signal (SIGTERM/SIGINT)
    2. Stop accepting new requests
    3. Wait for in-flight requests (with timeout)
    4. Persist bandit state to database
    5. Close database connection pool
    6. Log completion

    Thread Safety:
        The shutdown method uses asyncio.Lock to prevent concurrent
        shutdown attempts from multiple signal handlers.

    Error Handling:
        Shutdown continues even if individual steps fail. All errors
        are logged and collected in ShutdownState.errors.

    Attributes:
        router: Router instance for state persistence (optional)
        database: Database instance for connection management (optional)
        shutdown_timeout: Max seconds to wait for in-flight requests
        state: Current shutdown state
    """

    def __init__(
        self,
        router: "Router | None" = None,
        database: "Database | None" = None,
        shutdown_timeout: float = 30.0,
        on_shutdown_start: Callable[[], Coroutine[Any, Any, None]] | None = None,
        on_shutdown_complete: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        """Initialize lifecycle manager.

        Args:
            router: Router instance for state persistence. If None, state
                persistence step is skipped.
            database: Database instance for connection management. If None,
                connection closing step is skipped.
            shutdown_timeout: Maximum seconds to wait for in-flight requests
                to complete before forcing shutdown. Default: 30.0
            on_shutdown_start: Optional async callback invoked when shutdown begins.
                Use for custom cleanup before standard shutdown sequence.
            on_shutdown_complete: Optional async callback invoked after shutdown
                completes. Use for final notifications or logging.
        """
        self.router = router
        self.database = database
        self.shutdown_timeout = shutdown_timeout
        self.on_shutdown_start = on_shutdown_start
        self.on_shutdown_complete = on_shutdown_complete

        self.state = ShutdownState()
        self._shutdown_event = asyncio.Event()
        self._shutdown_lock = asyncio.Lock()
        self._request_counter = 0
        self._active_requests: set[str] = set()
        self._signal_handlers_installed = False

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self.state.phase not in (
            ShutdownPhase.NOT_STARTED,
            ShutdownPhase.COMPLETE,
        )

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown signal was received."""
        return self._shutdown_event.is_set()

    def install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        Handles SIGTERM (Kubernetes/Docker) and SIGINT (Ctrl+C).
        Safe to call multiple times; subsequent calls are no-ops.

        Note:
            Signal handlers can only be installed from the main thread.
            In async contexts, this should be called before starting
            the event loop or in a startup handler.
        """
        if self._signal_handlers_installed:
            logger.debug("Signal handlers already installed, skipping")
            return

        loop = asyncio.get_event_loop()

        def create_handler(sig: signal.Signals) -> Callable[[], None]:
            def handler() -> None:
                asyncio.create_task(self._handle_signal(sig))

            return handler

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, create_handler(sig))
            logger.debug(f"Installed handler for {sig.name}")

        self._signal_handlers_installed = True
        logger.info("Signal handlers installed for graceful shutdown")

    def remove_signal_handlers(self) -> None:
        """Remove installed signal handlers.

        Called automatically during shutdown to prevent recursive signals.
        """
        if not self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.remove_signal_handler(sig)
                logger.debug(f"Removed handler for {sig.name}")
            except (ValueError, RuntimeError):
                # Handler may not exist or loop may be closing
                pass

        self._signal_handlers_installed = False

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal.

        Args:
            sig: The signal that was received
        """
        signal_name = sig.name
        logger.info(f"Received {signal_name}, initiating graceful shutdown")

        self.state.signal_received = signal_name
        self._shutdown_event.set()

        # Trigger shutdown sequence
        await self.shutdown()

    def track_request_start(self, request_id: str) -> None:
        """Track start of a new request.

        Args:
            request_id: Unique identifier for the request

        Raises:
            RuntimeError: If shutdown is in progress
        """
        if self.shutdown_requested:
            raise RuntimeError("Cannot accept new requests during shutdown")

        self._active_requests.add(request_id)
        self._request_counter += 1
        self.state.in_flight_requests = len(self._active_requests)

    def track_request_end(self, request_id: str) -> None:
        """Track completion of a request.

        Args:
            request_id: Unique identifier for the request
        """
        self._active_requests.discard(request_id)
        self.state.in_flight_requests = len(self._active_requests)

    async def wait_for_requests(self) -> bool:
        """Wait for all in-flight requests to complete.

        Returns:
            True if all requests completed, False if timeout reached
        """
        if not self._active_requests:
            logger.info("No in-flight requests to drain")
            return True

        start_time = datetime.now(timezone.utc)
        initial_count = len(self._active_requests)
        logger.info(
            f"Waiting for {initial_count} in-flight requests "
            f"(timeout: {self.shutdown_timeout}s)"
        )

        try:
            while self._active_requests:
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                remaining = self.shutdown_timeout - elapsed

                if remaining <= 0:
                    logger.warning(
                        f"Timeout waiting for requests, "
                        f"{len(self._active_requests)} still in flight"
                    )
                    return False

                # Check every 100ms
                await asyncio.sleep(0.1)

            logger.info(
                f"All {initial_count} requests drained in "
                f"{(datetime.now(timezone.utc) - start_time).total_seconds():.1f}s"
            )
            return True

        except asyncio.CancelledError:
            logger.warning("Request drain cancelled")
            return False

    async def persist_state(self) -> bool:
        """Persist router state to database.

        Returns:
            True if persistence succeeded, False otherwise
        """
        if self.router is None:
            logger.debug("No router configured, skipping state persistence")
            return True

        try:
            # Router.close() handles state persistence internally
            logger.info("Persisting bandit state to database")
            await self.router.close()
            logger.info("Bandit state persisted successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to persist router state: {e}"
            logger.error(error_msg)
            self.state.errors.append(error_msg)
            return False

    async def close_database(self) -> bool:
        """Close database connection pool.

        Returns:
            True if close succeeded, False otherwise
        """
        if self.database is None:
            logger.debug("No database configured, skipping connection close")
            return True

        try:
            logger.info("Closing database connection pool")
            await self.database.disconnect()
            logger.info("Database connection pool closed")
            return True

        except Exception as e:
            error_msg = f"Failed to close database: {e}"
            logger.error(error_msg)
            self.state.errors.append(error_msg)
            return False

    async def shutdown(self) -> ShutdownState:
        """Execute graceful shutdown sequence.

        Shutdown sequence:
        1. Call on_shutdown_start callback (if provided)
        2. Remove signal handlers (prevent recursive signals)
        3. Drain in-flight requests (with timeout)
        4. Persist bandit state
        5. Close database connections
        6. Call on_shutdown_complete callback (if provided)

        This method is idempotent; calling multiple times returns
        the existing ShutdownState without re-executing shutdown.

        Returns:
            ShutdownState with shutdown results and timing
        """
        async with self._shutdown_lock:
            # Check if already completed
            if self.state.phase == ShutdownPhase.COMPLETE:
                logger.debug("Shutdown already completed")
                return self.state

            # Check if already in progress
            if self.is_shutting_down:
                logger.debug("Shutdown already in progress")
                return self.state

            # Start shutdown
            self.state.phase = ShutdownPhase.SIGNAL_RECEIVED
            self.state.started_at = datetime.now(timezone.utc)
            logger.info("Starting graceful shutdown sequence")

            # Call startup callback
            if self.on_shutdown_start:
                try:
                    await self.on_shutdown_start()
                except Exception as e:
                    error_msg = f"on_shutdown_start callback failed: {e}"
                    logger.error(error_msg)
                    self.state.errors.append(error_msg)

            # Remove signal handlers to prevent recursive shutdown
            self.remove_signal_handlers()

            # Phase 1: Drain requests
            self.state.phase = ShutdownPhase.DRAINING_REQUESTS
            logger.info(f"Phase 1/3: Draining {len(self._active_requests)} requests")
            await self.wait_for_requests()

            # Phase 2: Persist state
            self.state.phase = ShutdownPhase.PERSISTING_STATE
            logger.info("Phase 2/3: Persisting bandit state")
            await self.persist_state()

            # Phase 3: Close connections
            self.state.phase = ShutdownPhase.CLOSING_CONNECTIONS
            logger.info("Phase 3/3: Closing database connections")
            await self.close_database()

            # Complete
            self.state.completed_at = datetime.now(timezone.utc)
            if self.state.errors:
                self.state.phase = ShutdownPhase.FAILED
                logger.warning(
                    f"Shutdown completed with {len(self.state.errors)} errors "
                    f"in {self.state.duration_seconds:.1f}s"
                )
            else:
                self.state.phase = ShutdownPhase.COMPLETE
                logger.info(
                    f"Graceful shutdown completed in {self.state.duration_seconds:.1f}s"
                )

            # Call completion callback
            if self.on_shutdown_complete:
                try:
                    await self.on_shutdown_complete()
                except Exception as e:
                    error_msg = f"on_shutdown_complete callback failed: {e}"
                    logger.error(error_msg)
                    self.state.errors.append(error_msg)

            return self.state

    async def wait_for_shutdown(self) -> None:
        """Block until shutdown is requested.

        Useful for applications that need to wait for a signal
        before proceeding with shutdown logic.
        """
        await self._shutdown_event.wait()


def create_lifecycle_manager(
    router: "Router | None" = None,
    database: "Database | None" = None,
    shutdown_timeout: float = 30.0,
) -> LifecycleManager:
    """Factory function to create a configured LifecycleManager.

    Args:
        router: Router instance for state persistence
        database: Database instance for connection management
        shutdown_timeout: Max seconds to wait for in-flight requests

    Returns:
        Configured LifecycleManager instance

    Example:
        >>> manager = create_lifecycle_manager(
        ...     router=router,
        ...     database=database,
        ...     shutdown_timeout=30.0,
        ... )
        >>> manager.install_signal_handlers()
    """
    return LifecycleManager(
        router=router,
        database=database,
        shutdown_timeout=shutdown_timeout,
    )
