"""Unit tests for conduit.core.lifecycle module.

Tests for graceful shutdown handling and lifecycle management.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.lifecycle import (
    LifecycleManager,
    ShutdownPhase,
    ShutdownState,
    create_lifecycle_manager,
)


class TestShutdownState:
    """Tests for ShutdownState dataclass."""

    def test_initial_state(self):
        """Test ShutdownState has correct defaults."""
        state = ShutdownState()

        assert state.phase == ShutdownPhase.NOT_STARTED
        assert state.started_at is None
        assert state.completed_at is None
        assert state.signal_received is None
        assert state.in_flight_requests == 0
        assert state.errors == []

    def test_duration_seconds_not_started(self):
        """Test duration_seconds returns None when not started."""
        state = ShutdownState()
        assert state.duration_seconds is None

    def test_duration_seconds_in_progress(self):
        """Test duration_seconds calculates correctly during shutdown."""
        state = ShutdownState()
        state.started_at = datetime.now(timezone.utc)

        # Should return a non-negative duration
        duration = state.duration_seconds
        assert duration is not None
        assert duration >= 0

    def test_duration_seconds_completed(self):
        """Test duration_seconds uses completed_at when available."""
        state = ShutdownState()
        state.started_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        state.completed_at = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        assert state.duration_seconds == 5.0


class TestShutdownPhase:
    """Tests for ShutdownPhase enum."""

    def test_all_phases_exist(self):
        """Test all expected phases are defined."""
        phases = [
            ShutdownPhase.NOT_STARTED,
            ShutdownPhase.SIGNAL_RECEIVED,
            ShutdownPhase.DRAINING_REQUESTS,
            ShutdownPhase.PERSISTING_STATE,
            ShutdownPhase.CLOSING_CONNECTIONS,
            ShutdownPhase.COMPLETE,
            ShutdownPhase.FAILED,
        ]
        assert len(phases) == 7


class TestLifecycleManagerInit:
    """Tests for LifecycleManager initialization."""

    def test_init_with_no_dependencies(self):
        """Test LifecycleManager initializes without dependencies."""
        manager = LifecycleManager()

        assert manager.router is None
        assert manager.database is None
        assert manager.shutdown_timeout == 30.0
        assert manager.state.phase == ShutdownPhase.NOT_STARTED

    def test_init_with_dependencies(self):
        """Test LifecycleManager initializes with dependencies."""
        mock_router = MagicMock()
        mock_database = MagicMock()

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
            shutdown_timeout=60.0,
        )

        assert manager.router is mock_router
        assert manager.database is mock_database
        assert manager.shutdown_timeout == 60.0

    def test_init_with_callbacks(self):
        """Test LifecycleManager initializes with callbacks."""
        on_start = AsyncMock()
        on_complete = AsyncMock()

        manager = LifecycleManager(
            on_shutdown_start=on_start,
            on_shutdown_complete=on_complete,
        )

        assert manager.on_shutdown_start is on_start
        assert manager.on_shutdown_complete is on_complete


class TestLifecycleManagerProperties:
    """Tests for LifecycleManager properties."""

    def test_is_shutting_down_false_initially(self):
        """Test is_shutting_down is False initially."""
        manager = LifecycleManager()
        assert manager.is_shutting_down is False

    def test_is_shutting_down_true_during_shutdown(self):
        """Test is_shutting_down is True during shutdown phases."""
        manager = LifecycleManager()

        for phase in [
            ShutdownPhase.SIGNAL_RECEIVED,
            ShutdownPhase.DRAINING_REQUESTS,
            ShutdownPhase.PERSISTING_STATE,
            ShutdownPhase.CLOSING_CONNECTIONS,
        ]:
            manager.state.phase = phase
            assert manager.is_shutting_down is True

    def test_is_shutting_down_false_after_complete(self):
        """Test is_shutting_down is False after completion."""
        manager = LifecycleManager()
        manager.state.phase = ShutdownPhase.COMPLETE
        assert manager.is_shutting_down is False

    def test_shutdown_requested_false_initially(self):
        """Test shutdown_requested is False initially."""
        manager = LifecycleManager()
        assert manager.shutdown_requested is False

    def test_shutdown_requested_true_after_signal(self):
        """Test shutdown_requested is True after event set."""
        manager = LifecycleManager()
        manager._shutdown_event.set()
        assert manager.shutdown_requested is True


class TestRequestTracking:
    """Tests for request tracking methods."""

    def test_track_request_start(self):
        """Test track_request_start adds request to active set."""
        manager = LifecycleManager()

        manager.track_request_start("req-1")

        assert "req-1" in manager._active_requests
        assert manager.state.in_flight_requests == 1
        assert manager._request_counter == 1

    def test_track_request_start_multiple(self):
        """Test tracking multiple requests."""
        manager = LifecycleManager()

        manager.track_request_start("req-1")
        manager.track_request_start("req-2")
        manager.track_request_start("req-3")

        assert len(manager._active_requests) == 3
        assert manager.state.in_flight_requests == 3
        assert manager._request_counter == 3

    def test_track_request_start_during_shutdown_raises(self):
        """Test track_request_start raises during shutdown."""
        manager = LifecycleManager()
        manager._shutdown_event.set()

        with pytest.raises(RuntimeError, match="Cannot accept new requests"):
            manager.track_request_start("req-1")

    def test_track_request_end(self):
        """Test track_request_end removes request from active set."""
        manager = LifecycleManager()
        manager.track_request_start("req-1")

        manager.track_request_end("req-1")

        assert "req-1" not in manager._active_requests
        assert manager.state.in_flight_requests == 0

    def test_track_request_end_unknown_id(self):
        """Test track_request_end handles unknown request ID."""
        manager = LifecycleManager()

        # Should not raise
        manager.track_request_end("unknown-req")

        assert manager.state.in_flight_requests == 0


class TestWaitForRequests:
    """Tests for wait_for_requests method."""

    @pytest.mark.asyncio
    async def test_wait_for_requests_no_requests(self):
        """Test wait_for_requests returns immediately with no requests."""
        manager = LifecycleManager()

        result = await manager.wait_for_requests()

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_requests_completes(self):
        """Test wait_for_requests waits for request completion."""
        manager = LifecycleManager(shutdown_timeout=5.0)
        manager.track_request_start("req-1")

        async def complete_request():
            await asyncio.sleep(0.1)
            manager.track_request_end("req-1")

        # Start request completion in background
        asyncio.create_task(complete_request())

        result = await manager.wait_for_requests()

        assert result is True
        assert manager.state.in_flight_requests == 0

    @pytest.mark.asyncio
    async def test_wait_for_requests_timeout(self):
        """Test wait_for_requests returns False on timeout."""
        manager = LifecycleManager(shutdown_timeout=0.1)
        manager.track_request_start("req-1")

        # Don't complete the request
        result = await manager.wait_for_requests()

        assert result is False
        assert manager.state.in_flight_requests == 1


class TestPersistState:
    """Tests for persist_state method."""

    @pytest.mark.asyncio
    async def test_persist_state_no_router(self):
        """Test persist_state succeeds with no router."""
        manager = LifecycleManager()

        result = await manager.persist_state()

        assert result is True

    @pytest.mark.asyncio
    async def test_persist_state_calls_router_close(self):
        """Test persist_state calls router.close()."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        manager = LifecycleManager(router=mock_router)

        result = await manager.persist_state()

        assert result is True
        mock_router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_state_handles_error(self):
        """Test persist_state handles errors gracefully."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock(side_effect=Exception("Save failed"))

        manager = LifecycleManager(router=mock_router)

        result = await manager.persist_state()

        assert result is False
        assert len(manager.state.errors) == 1
        assert "Save failed" in manager.state.errors[0]


class TestCloseDatabase:
    """Tests for close_database method."""

    @pytest.mark.asyncio
    async def test_close_database_no_database(self):
        """Test close_database succeeds with no database."""
        manager = LifecycleManager()

        result = await manager.close_database()

        assert result is True

    @pytest.mark.asyncio
    async def test_close_database_calls_disconnect(self):
        """Test close_database calls database.disconnect()."""
        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock()

        manager = LifecycleManager(database=mock_database)

        result = await manager.close_database()

        assert result is True
        mock_database.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_database_handles_error(self):
        """Test close_database handles errors gracefully."""
        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock(side_effect=Exception("Close failed"))

        manager = LifecycleManager(database=mock_database)

        result = await manager.close_database()

        assert result is False
        assert len(manager.state.errors) == 1
        assert "Close failed" in manager.state.errors[0]


class TestShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_no_dependencies(self):
        """Test shutdown completes with no dependencies."""
        manager = LifecycleManager()

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert state.started_at is not None
        assert state.completed_at is not None
        assert state.errors == []

    @pytest.mark.asyncio
    async def test_shutdown_full_sequence(self):
        """Test shutdown executes full sequence."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock()

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
        )

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        mock_router.close.assert_called_once()
        mock_database.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_calls_callbacks(self):
        """Test shutdown calls start and complete callbacks."""
        on_start = AsyncMock()
        on_complete = AsyncMock()

        manager = LifecycleManager(
            on_shutdown_start=on_start,
            on_shutdown_complete=on_complete,
        )

        await manager.shutdown()

        on_start.assert_called_once()
        on_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Test shutdown is idempotent."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        manager = LifecycleManager(router=mock_router)

        # First shutdown
        state1 = await manager.shutdown()
        assert state1.phase == ShutdownPhase.COMPLETE

        # Second shutdown returns same state without re-executing
        state2 = await manager.shutdown()
        assert state2 is state1

        # Router.close should only be called once
        assert mock_router.close.call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_with_errors_marks_failed(self):
        """Test shutdown marks phase as FAILED when errors occur."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock(side_effect=Exception("State save failed"))

        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock(side_effect=Exception("Close failed"))

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
        )

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.FAILED
        assert len(state.errors) == 2

    @pytest.mark.asyncio
    async def test_shutdown_drains_requests(self):
        """Test shutdown waits for in-flight requests."""
        manager = LifecycleManager(shutdown_timeout=5.0)
        manager.track_request_start("req-1")

        async def complete_request():
            await asyncio.sleep(0.1)
            manager.track_request_end("req-1")

        asyncio.create_task(complete_request())

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert manager.state.in_flight_requests == 0

    @pytest.mark.asyncio
    async def test_shutdown_callback_error_does_not_prevent_completion(self):
        """Test callback errors don't prevent shutdown completion."""
        on_start = AsyncMock(side_effect=Exception("Callback failed"))

        manager = LifecycleManager(on_shutdown_start=on_start)

        state = await manager.shutdown()

        # Should still complete despite callback error
        assert state.phase in (ShutdownPhase.COMPLETE, ShutdownPhase.FAILED)
        assert any("Callback failed" in e for e in state.errors)


class TestSignalHandlers:
    """Tests for signal handler methods."""

    def test_install_signal_handlers_sets_flag(self):
        """Test install_signal_handlers sets installed flag."""
        manager = LifecycleManager()

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            manager.install_signal_handlers()

            assert manager._signal_handlers_installed is True
            assert mock_loop.add_signal_handler.call_count == 2

    def test_install_signal_handlers_idempotent(self):
        """Test install_signal_handlers is idempotent."""
        manager = LifecycleManager()
        manager._signal_handlers_installed = True

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            manager.install_signal_handlers()

            # Should not add handlers again
            mock_loop.add_signal_handler.assert_not_called()

    def test_remove_signal_handlers_clears_flag(self):
        """Test remove_signal_handlers clears installed flag."""
        manager = LifecycleManager()
        manager._signal_handlers_installed = True

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            manager.remove_signal_handlers()

            assert manager._signal_handlers_installed is False
            assert mock_loop.remove_signal_handler.call_count == 2

    def test_remove_signal_handlers_no_op_when_not_installed(self):
        """Test remove_signal_handlers is no-op when not installed."""
        manager = LifecycleManager()

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            manager.remove_signal_handlers()

            mock_loop.remove_signal_handler.assert_not_called()


class TestWaitForShutdown:
    """Tests for wait_for_shutdown method."""

    @pytest.mark.asyncio
    async def test_wait_for_shutdown_blocks_until_signal(self):
        """Test wait_for_shutdown blocks until shutdown event."""
        manager = LifecycleManager()

        async def trigger_shutdown():
            await asyncio.sleep(0.1)
            manager._shutdown_event.set()

        asyncio.create_task(trigger_shutdown())

        # Should return after event is set
        await asyncio.wait_for(manager.wait_for_shutdown(), timeout=1.0)

        assert manager.shutdown_requested is True


class TestCreateLifecycleManager:
    """Tests for create_lifecycle_manager factory function."""

    def test_create_lifecycle_manager_no_args(self):
        """Test factory creates manager with defaults."""
        manager = create_lifecycle_manager()

        assert isinstance(manager, LifecycleManager)
        assert manager.router is None
        assert manager.database is None
        assert manager.shutdown_timeout == 30.0

    def test_create_lifecycle_manager_with_args(self):
        """Test factory creates manager with provided args."""
        mock_router = MagicMock()
        mock_database = MagicMock()

        manager = create_lifecycle_manager(
            router=mock_router,
            database=mock_database,
            shutdown_timeout=60.0,
        )

        assert manager.router is mock_router
        assert manager.database is mock_database
        assert manager.shutdown_timeout == 60.0


class TestConcurrentShutdown:
    """Tests for concurrent shutdown handling."""

    @pytest.mark.asyncio
    async def test_concurrent_shutdown_calls(self):
        """Test concurrent shutdown calls are handled safely."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        manager = LifecycleManager(router=mock_router)

        # Start multiple shutdowns concurrently
        results = await asyncio.gather(
            manager.shutdown(),
            manager.shutdown(),
            manager.shutdown(),
        )

        # All should return the same state
        assert all(r.phase == ShutdownPhase.COMPLETE for r in results)

        # Router.close should only be called once
        assert mock_router.close.call_count == 1
