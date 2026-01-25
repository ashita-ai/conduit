"""Integration tests for graceful shutdown behavior.

Tests the full shutdown sequence including:
- Signal handling
- Request draining
- State persistence
- Database connection cleanup
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conduit.core.lifecycle import (
    LifecycleManager,
    ShutdownPhase,
)


class TestShutdownWithRequests:
    """Tests for shutdown with in-flight requests."""

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_multiple_requests(self):
        """Test shutdown waits for multiple in-flight requests."""
        manager = LifecycleManager(shutdown_timeout=5.0)

        # Start multiple requests
        for i in range(5):
            manager.track_request_start(f"req-{i}")

        async def complete_requests():
            """Complete requests one by one with delays."""
            for i in range(5):
                await asyncio.sleep(0.05)
                manager.track_request_end(f"req-{i}")

        asyncio.create_task(complete_requests())

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert manager.state.in_flight_requests == 0

    @pytest.mark.asyncio
    async def test_shutdown_timeout_with_stuck_requests(self):
        """Test shutdown times out when requests don't complete."""
        manager = LifecycleManager(shutdown_timeout=0.2)

        # Start a request that won't complete
        manager.track_request_start("stuck-req")

        state = await manager.shutdown()

        # Should complete with error logged but not fail
        assert state.phase == ShutdownPhase.COMPLETE
        assert manager.state.in_flight_requests == 1

    @pytest.mark.asyncio
    async def test_new_requests_rejected_during_shutdown(self):
        """Test new requests are rejected during shutdown."""
        manager = LifecycleManager()

        # Trigger shutdown signal
        manager._shutdown_event.set()

        # New request should be rejected
        with pytest.raises(RuntimeError, match="Cannot accept new requests"):
            manager.track_request_start("late-req")


class TestShutdownWithMockedDependencies:
    """Tests for shutdown with mocked router and database."""

    @pytest.mark.asyncio
    async def test_full_shutdown_sequence_order(self):
        """Test shutdown phases execute in correct order."""
        phases_executed: list[str] = []

        mock_router = MagicMock()

        async def mock_close():
            phases_executed.append("router_close")

        mock_router.close = mock_close

        mock_database = MagicMock()

        async def mock_disconnect():
            phases_executed.append("database_disconnect")

        mock_database.disconnect = mock_disconnect

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
        )

        # Add a request to test drain phase
        manager.track_request_start("req-1")

        async def complete_request():
            await asyncio.sleep(0.05)
            phases_executed.append("request_drained")
            manager.track_request_end("req-1")

        asyncio.create_task(complete_request())

        await manager.shutdown()

        # Verify order: drain -> persist -> close
        assert phases_executed.index("request_drained") < phases_executed.index(
            "router_close"
        )
        assert phases_executed.index("router_close") < phases_executed.index(
            "database_disconnect"
        )

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_router_error(self):
        """Test shutdown continues even if router.close() fails."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock(side_effect=Exception("Router close failed"))

        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock()

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
        )

        state = await manager.shutdown()

        # Database should still be closed despite router error
        mock_database.disconnect.assert_called_once()

        # State should reflect the error
        assert state.phase == ShutdownPhase.FAILED
        assert any("Router close failed" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_shutdown_continues_after_database_error(self):
        """Test shutdown completes even if database.disconnect() fails."""
        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        mock_database = MagicMock()
        mock_database.disconnect = AsyncMock(
            side_effect=Exception("Database close failed")
        )

        manager = LifecycleManager(
            router=mock_router,
            database=mock_database,
        )

        state = await manager.shutdown()

        # Router should still be closed
        mock_router.close.assert_called_once()

        # State should reflect the error
        assert state.phase == ShutdownPhase.FAILED
        assert any("Database close failed" in e for e in state.errors)


class TestShutdownCallbacks:
    """Tests for shutdown callbacks."""

    @pytest.mark.asyncio
    async def test_callbacks_execute_in_order(self):
        """Test callbacks execute at correct times."""
        execution_order: list[str] = []

        async def on_start():
            execution_order.append("start_callback")

        async def on_complete():
            execution_order.append("complete_callback")

        mock_router = MagicMock()

        async def mock_close():
            execution_order.append("router_close")

        mock_router.close = mock_close

        manager = LifecycleManager(
            router=mock_router,
            on_shutdown_start=on_start,
            on_shutdown_complete=on_complete,
        )

        await manager.shutdown()

        # Verify order
        assert execution_order == [
            "start_callback",
            "router_close",
            "complete_callback",
        ]

    @pytest.mark.asyncio
    async def test_start_callback_error_does_not_stop_shutdown(self):
        """Test shutdown continues if start callback fails."""

        async def failing_start():
            raise Exception("Start callback failed")

        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        manager = LifecycleManager(
            router=mock_router,
            on_shutdown_start=failing_start,
        )

        state = await manager.shutdown()

        # Router should still be closed
        mock_router.close.assert_called_once()

        # Error should be recorded
        assert any("Start callback failed" in e for e in state.errors)

    @pytest.mark.asyncio
    async def test_complete_callback_error_recorded(self):
        """Test complete callback errors are recorded."""

        async def failing_complete():
            raise Exception("Complete callback failed")

        manager = LifecycleManager(on_shutdown_complete=failing_complete)

        state = await manager.shutdown()

        # Error should be recorded
        assert any("Complete callback failed" in e for e in state.errors)


class TestConcurrentShutdownOperations:
    """Tests for concurrent shutdown operations."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_concurrent_shutdown_requests(self):
        """Test many concurrent shutdown requests are handled safely."""
        mock_router = MagicMock()
        close_count = 0

        async def counting_close():
            nonlocal close_count
            close_count += 1
            await asyncio.sleep(0.01)

        mock_router.close = counting_close

        manager = LifecycleManager(router=mock_router)

        # Start 10 concurrent shutdowns
        results = await asyncio.gather(*[manager.shutdown() for _ in range(10)])

        # All should complete
        assert all(r.phase == ShutdownPhase.COMPLETE for r in results)

        # Router should only be closed once
        assert close_count == 1

    @pytest.mark.asyncio
    async def test_shutdown_during_request_processing(self):
        """Test shutdown during active request processing."""
        manager = LifecycleManager(shutdown_timeout=2.0)
        processed_requests: list[str] = []

        async def process_request(request_id: str):
            """Simulate request processing."""
            manager.track_request_start(request_id)
            try:
                await asyncio.sleep(0.1)
                processed_requests.append(request_id)
            finally:
                manager.track_request_end(request_id)

        # Start multiple requests
        request_tasks = [
            asyncio.create_task(process_request(f"req-{i}")) for i in range(5)
        ]

        # Wait a bit then trigger shutdown
        await asyncio.sleep(0.05)
        shutdown_task = asyncio.create_task(manager.shutdown())

        # Wait for everything to complete
        await asyncio.gather(*request_tasks, shutdown_task)

        # All requests should have completed
        assert len(processed_requests) == 5
        assert manager.state.phase == ShutdownPhase.COMPLETE


class TestShutdownTiming:
    """Tests for shutdown timing and duration tracking."""

    @pytest.mark.asyncio
    async def test_shutdown_duration_tracked(self):
        """Test shutdown duration is tracked correctly."""
        manager = LifecycleManager()

        state = await manager.shutdown()

        assert state.started_at is not None
        assert state.completed_at is not None
        assert state.duration_seconds is not None
        assert state.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_shutdown_with_delay_tracks_duration(self):
        """Test shutdown duration includes processing time."""
        mock_router = MagicMock()

        async def slow_close():
            await asyncio.sleep(0.1)

        mock_router.close = slow_close

        manager = LifecycleManager(router=mock_router)

        state = await manager.shutdown()

        # Duration should be at least 0.1 seconds
        assert state.duration_seconds is not None
        assert state.duration_seconds >= 0.1


class TestSignalHandlerIntegration:
    """Tests for signal handler integration."""

    @pytest.mark.asyncio
    async def test_handle_signal_triggers_shutdown(self):
        """Test signal handler triggers full shutdown."""
        import signal

        mock_router = MagicMock()
        mock_router.close = AsyncMock()

        manager = LifecycleManager(router=mock_router)

        # Directly call the signal handler
        await manager._handle_signal(signal.SIGTERM)

        assert manager.state.signal_received == "SIGTERM"
        assert manager.state.phase == ShutdownPhase.COMPLETE
        mock_router.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_handler_removes_handlers_before_shutdown(self):
        """Test signal handlers are removed during shutdown to prevent recursion."""
        manager = LifecycleManager()

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            # Install handlers
            manager.install_signal_handlers()
            assert manager._signal_handlers_installed is True

            # Shutdown should remove handlers
            await manager.shutdown()
            assert manager._signal_handlers_installed is False


class TestShutdownWithRealAsyncOperations:
    """Tests with realistic async operations."""

    @pytest.mark.asyncio
    async def test_shutdown_with_database_like_operations(self):
        """Test shutdown with operations that simulate real database behavior."""
        operations_log: list[str] = []

        class MockDatabaseLike:
            """Mock that simulates real database pool behavior."""

            async def disconnect(self):
                operations_log.append("disconnect_start")
                # Simulate connection draining
                await asyncio.sleep(0.05)
                operations_log.append("disconnect_complete")

        class MockRouterLike:
            """Mock that simulates real router state save."""

            async def close(self):
                operations_log.append("close_start")
                # Simulate state serialization and DB write
                await asyncio.sleep(0.05)
                operations_log.append("close_complete")

        manager = LifecycleManager(
            router=MockRouterLike(),  # type: ignore[arg-type]
            database=MockDatabaseLike(),  # type: ignore[arg-type]
        )

        state = await manager.shutdown()

        assert state.phase == ShutdownPhase.COMPLETE
        assert operations_log == [
            "close_start",
            "close_complete",
            "disconnect_start",
            "disconnect_complete",
        ]
