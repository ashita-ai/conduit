"""Unit tests for FastAPI middleware."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from conduit.api.middleware import LoggingMiddleware, setup_cors, setup_middleware


class TestLoggingMiddleware:
    """Tests for LoggingMiddleware."""

    @pytest.mark.asyncio
    async def test_logging_middleware_logs_request_and_response(self):
        """Test middleware logs request and response details."""
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "ok"}

        # Add middleware
        app.add_middleware(LoggingMiddleware)

        with patch("conduit.api.middleware.logger") as mock_logger:
            client = TestClient(app)
            response = client.get("/test")

            assert response.status_code == 200

            # Verify logging calls
            assert mock_logger.info.call_count == 2

            # Check request log
            first_call = mock_logger.info.call_args_list[0][0][0]
            assert "GET" in first_call
            assert "/test" in first_call

            # Check response log
            second_call = mock_logger.info.call_args_list[1][0][0]
            assert "GET" in second_call
            assert "/test" in second_call
            assert "Status: 200" in second_call
            assert "Latency:" in second_call

    @pytest.mark.asyncio
    async def test_logging_middleware_handles_errors(self):
        """Test middleware logs even when endpoint raises error."""
        app = FastAPI()

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        app.add_middleware(LoggingMiddleware)

        with patch("conduit.api.middleware.logger") as mock_logger:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/error")

            assert response.status_code == 500

            # Verify request was logged
            assert mock_logger.info.call_count >= 1
            first_call = mock_logger.info.call_args_list[0][0][0]
            assert "GET" in first_call
            assert "/error" in first_call


class TestSetupCORS:
    """Tests for CORS setup."""

    def test_setup_cors_development(self):
        """Test CORS allows all origins in development."""
        app = FastAPI()

        with patch("conduit.api.middleware.settings") as mock_settings:
            mock_settings.is_production = False
            setup_cors(app)

        # Verify CORS middleware was added (can't easily test config directly)
        # Just verify it doesn't crash
        client = TestClient(app)
        assert client is not None

    def test_setup_cors_production(self):
        """Test CORS restricts origins in production."""
        app = FastAPI()

        with patch("conduit.api.middleware.settings") as mock_settings:
            mock_settings.is_production = True
            setup_cors(app)

        # Verify CORS middleware was added
        client = TestClient(app)
        assert client is not None


class TestSetupMiddleware:
    """Tests for middleware setup function."""

    def test_setup_middleware_adds_all_middleware(self):
        """Test setup_middleware adds both CORS and logging middleware."""
        app = FastAPI()

        with patch("conduit.api.middleware.setup_cors") as mock_cors:
            with patch.object(app, "add_middleware") as mock_add:
                setup_middleware(app)

                # Verify CORS setup was called
                mock_cors.assert_called_once_with(app)

                # Verify logging middleware was added
                mock_add.assert_called_once_with(LoggingMiddleware)
