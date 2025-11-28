"""Tests for observability modules (tracing and metrics).

Tests cover:
- Tracing decorator behavior with OTEL enabled/disabled
- Metrics recording functions with OTEL enabled/disabled
- Meter and tracer initialization
- Metric instrument creation
"""

from unittest.mock import MagicMock, patch

import pytest


class TestTracing:
    """Tests for observability/tracing.py module."""

    def test_get_tracer_returns_tracer(self):
        """Test get_tracer returns a tracer instance."""
        from conduit.observability.tracing import get_tracer

        tracer = get_tracer("test")
        assert tracer is not None

    def test_get_tracer_with_default_name(self):
        """Test get_tracer uses default name."""
        from conduit.observability.tracing import get_tracer

        tracer = get_tracer()
        assert tracer is not None

    def test_trace_operation_disabled_returns_original_function(self):
        """Test trace_operation returns original function when OTEL disabled."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_traces_enabled = False

            from conduit.observability.tracing import trace_operation

            @trace_operation("test_op")
            def my_func():
                return "result"

            # Should return the original function (not wrapped)
            result = my_func()
            assert result == "result"

    def test_trace_operation_disabled_traces_not_enabled(self):
        """Test trace_operation returns original when traces specifically disabled."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = False

            from conduit.observability.tracing import trace_operation

            @trace_operation("test_op")
            def my_func():
                return "sync_result"

            result = my_func()
            assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_trace_operation_async_function_disabled(self):
        """Test trace_operation with async function when disabled."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_traces_enabled = False

            from conduit.observability.tracing import trace_operation

            @trace_operation("async_op")
            async def my_async_func():
                return "async_result"

            result = await my_async_func()
            assert result == "async_result"

    def test_trace_operation_enabled_wraps_sync_function(self):
        """Test trace_operation wraps sync function when OTEL enabled."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True

            from conduit.observability.tracing import trace_operation

            @trace_operation("test_sync")
            def my_sync_func(x):
                return x * 2

            # Function should still work correctly
            result = my_sync_func(5)
            assert result == 10

    @pytest.mark.asyncio
    async def test_trace_operation_enabled_wraps_async_function(self):
        """Test trace_operation wraps async function when OTEL enabled."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True

            from conduit.observability.tracing import trace_operation

            @trace_operation("test_async", attributes={"custom": "attr"})
            async def my_async_func(x):
                return x + 1

            result = await my_async_func(10)
            assert result == 11

    def test_trace_operation_sync_with_exception(self):
        """Test trace_operation handles exceptions in sync functions."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True

            from conduit.observability.tracing import trace_operation

            @trace_operation("failing_sync")
            def failing_func():
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                failing_func()

    @pytest.mark.asyncio
    async def test_trace_operation_async_with_exception(self):
        """Test trace_operation handles exceptions in async functions."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True

            from conduit.observability.tracing import trace_operation

            @trace_operation("failing_async")
            async def failing_async_func():
                raise RuntimeError("Async error")

            with pytest.raises(RuntimeError, match="Async error"):
                await failing_async_func()

    def test_trace_operation_uses_function_name_as_default(self):
        """Test trace_operation uses function name when no operation_name provided."""
        with patch("conduit.observability.tracing.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True

            from conduit.observability.tracing import trace_operation

            @trace_operation()  # No operation_name
            def custom_function_name():
                return "test"

            result = custom_function_name()
            assert result == "test"


class TestMetrics:
    """Tests for observability/metrics.py module."""

    def test_get_meter_returns_meter(self):
        """Test get_meter returns a meter instance."""
        from conduit.observability.metrics import get_meter

        meter = get_meter("test")
        assert meter is not None

    def test_get_meter_cached(self):
        """Test get_meter returns cached meter on subsequent calls."""
        from conduit.observability import metrics

        # Reset the global meter
        metrics._meter = None

        meter1 = metrics.get_meter("test")
        meter2 = metrics.get_meter("test")

        # Should return the same cached instance
        assert meter1 is meter2

    def test_record_routing_decision_disabled(self):
        """Test record_routing_decision does nothing when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = False

            from conduit.core.models import QueryFeatures, RoutingDecision
            from conduit.observability.metrics import record_routing_decision

            features = QueryFeatures(
                embedding=[0.1] * 384,
                token_count=10,
                complexity_score=0.5,
                query_text="test",
            )
            decision = RoutingDecision(
                query_id="test-query-123",
                selected_model="gpt-4o-mini",
                confidence=0.95,
                features=features,
                reasoning="Test reasoning",
            )

            # Should not raise any exceptions
            record_routing_decision(decision, cost=0.001, latency_ms=100)

    def test_record_routing_decision_enabled(self):
        """Test record_routing_decision when OTEL enabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True

            from conduit.observability import metrics

            # Reset instruments
            metrics._routing_decisions_counter = None
            metrics._routing_cost_histogram = None
            metrics._routing_latency_histogram = None
            metrics._routing_confidence_histogram = None

            from conduit.core.models import QueryFeatures, RoutingDecision
            from conduit.observability.metrics import record_routing_decision

            features = QueryFeatures(
                embedding=[0.1] * 384,
                token_count=10,
                complexity_score=0.5,
                query_text="test",
            )
            decision = RoutingDecision(
                query_id="test-query-456",
                selected_model="gpt-4o-mini",
                confidence=0.95,
                features=features,
                reasoning="Test reasoning for enabled",
            )

            # Should not raise any exceptions
            record_routing_decision(decision, cost=0.001, latency_ms=100)

    def test_record_cache_hit_disabled(self):
        """Test record_cache_hit does nothing when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = False

            from conduit.observability.metrics import record_cache_hit

            # Should not raise any exceptions
            record_cache_hit()

    def test_record_cache_hit_enabled(self):
        """Test record_cache_hit when OTEL enabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True

            from conduit.observability import metrics

            metrics._cache_hits_counter = None

            from conduit.observability.metrics import record_cache_hit

            # Should not raise any exceptions
            record_cache_hit()

    def test_record_cache_miss_disabled(self):
        """Test record_cache_miss does nothing when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = False

            from conduit.observability.metrics import record_cache_miss

            # Should not raise any exceptions
            record_cache_miss()

    def test_record_cache_miss_enabled(self):
        """Test record_cache_miss when OTEL enabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True

            from conduit.observability import metrics

            metrics._cache_misses_counter = None

            from conduit.observability.metrics import record_cache_miss

            # Should not raise any exceptions
            record_cache_miss()

    def test_record_feedback_submission_disabled(self):
        """Test record_feedback_submission does nothing when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = False

            from conduit.observability.metrics import record_feedback_submission

            # Should not raise any exceptions
            record_feedback_submission("positive", explicit=True)

    def test_record_feedback_submission_enabled(self):
        """Test record_feedback_submission when OTEL enabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True

            from conduit.observability import metrics

            metrics._feedback_submissions_counter = None

            from conduit.observability.metrics import record_feedback_submission

            # Should not raise any exceptions
            record_feedback_submission("negative", explicit=False)

    def test_get_metrics_summary(self):
        """Test get_metrics_summary returns config info."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True
            mock_settings.otel_service_name = "test-service"
            mock_settings.otel_exporter_otlp_endpoint = "http://localhost:4317"

            from conduit.observability.metrics import get_metrics_summary

            summary = get_metrics_summary()

            assert summary["otel_enabled"] is True
            assert summary["metrics_enabled"] is True
            assert summary["service_name"] == "test-service"
            assert summary["exporter_endpoint"] == "http://localhost:4317"

    def test_ensure_instruments_creates_all_instruments(self):
        """Test _ensure_instruments creates all metric instruments."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = True

            from conduit.observability import metrics

            # Reset all instruments
            metrics._routing_decisions_counter = None
            metrics._routing_cost_histogram = None
            metrics._routing_latency_histogram = None
            metrics._routing_confidence_histogram = None
            metrics._cache_hits_counter = None
            metrics._cache_misses_counter = None
            metrics._feedback_submissions_counter = None

            metrics._ensure_instruments()

            # All instruments should be created
            assert metrics._routing_decisions_counter is not None
            assert metrics._routing_cost_histogram is not None
            assert metrics._routing_latency_histogram is not None
            assert metrics._routing_confidence_histogram is not None
            assert metrics._cache_hits_counter is not None
            assert metrics._cache_misses_counter is not None
            assert metrics._feedback_submissions_counter is not None

    def test_ensure_instruments_skips_when_disabled(self):
        """Test _ensure_instruments does nothing when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = False

            from conduit.observability import metrics

            # Reset all instruments
            metrics._routing_decisions_counter = None

            metrics._ensure_instruments()

            # Instruments should remain None
            assert metrics._routing_decisions_counter is None


class TestObservabilitySetup:
    """Tests for observability/setup.py module."""

    def test_setup_telemetry_disabled(self):
        """Test setup_telemetry does nothing when OTEL disabled."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = False

            from conduit.observability.setup import setup_telemetry

            # Should not raise any exceptions
            setup_telemetry()

    def test_setup_telemetry_already_instrumented(self):
        """Test setup_telemetry is idempotent."""
        from conduit.observability import setup as setup_module

        # Force _instrumented to True
        original_value = setup_module._instrumented
        setup_module._instrumented = True

        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True

            from conduit.observability.setup import setup_telemetry

            # Should return early without error
            setup_telemetry()

        # Restore
        setup_module._instrumented = original_value

    def test_setup_telemetry_enabled_with_tracing(self):
        """Test setup_telemetry initializes tracing when enabled."""
        from conduit.observability import setup as setup_module

        # Reset instrumented flag
        original_value = setup_module._instrumented
        setup_module._instrumented = False

        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True
            mock_settings.otel_metrics_enabled = False
            mock_settings.otel_service_name = "test-service"
            mock_settings.environment = "test"
            mock_settings.otel_exporter_otlp_endpoint = "http://localhost:4317"
            mock_settings.otel_exporter_otlp_headers = None

            with patch("conduit.observability.setup.OTLPSpanExporter") as mock_exporter:
                with patch("conduit.observability.setup.TracerProvider") as mock_provider:
                    with patch("conduit.observability.setup.BatchSpanProcessor"):
                        with patch("conduit.observability.setup.trace") as mock_trace:
                            with patch("conduit.observability.setup.RedisInstrumentor"):
                                from conduit.observability.setup import setup_telemetry

                                setup_telemetry()

                                # Verify trace provider was created and set
                                mock_provider.assert_called_once()
                                mock_trace.set_tracer_provider.assert_called_once()

        # Restore
        setup_module._instrumented = original_value

    def test_setup_telemetry_enabled_with_metrics(self):
        """Test setup_telemetry initializes metrics when enabled."""
        from conduit.observability import setup as setup_module

        # Reset instrumented flag
        original_value = setup_module._instrumented
        setup_module._instrumented = False

        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = False
            mock_settings.otel_metrics_enabled = True
            mock_settings.otel_service_name = "test-service"
            mock_settings.environment = "test"
            mock_settings.otel_exporter_otlp_endpoint = "http://localhost:4317"
            mock_settings.otel_exporter_otlp_headers = None

            with patch("conduit.observability.setup.OTLPMetricExporter") as mock_exporter:
                with patch("conduit.observability.setup.MeterProvider") as mock_provider:
                    with patch("conduit.observability.setup.PeriodicExportingMetricReader"):
                        with patch("conduit.observability.setup.metrics") as mock_metrics:
                            with patch("conduit.observability.setup.RedisInstrumentor"):
                                from conduit.observability.setup import setup_telemetry

                                setup_telemetry()

                                # Verify meter provider was created and set
                                mock_provider.assert_called_once()
                                mock_metrics.set_meter_provider.assert_called_once()

        # Restore
        setup_module._instrumented = original_value

    def test_setup_telemetry_with_headers(self):
        """Test setup_telemetry parses OTLP headers."""
        from conduit.observability import setup as setup_module

        # Reset instrumented flag
        original_value = setup_module._instrumented
        setup_module._instrumented = False

        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True
            mock_settings.otel_metrics_enabled = False
            mock_settings.otel_service_name = "test-service"
            mock_settings.environment = "test"
            mock_settings.otel_exporter_otlp_endpoint = "http://localhost:4317"
            mock_settings.otel_exporter_otlp_headers = "api-key=test123,x-custom=value"

            with patch("conduit.observability.setup.OTLPSpanExporter") as mock_exporter:
                with patch("conduit.observability.setup.TracerProvider"):
                    with patch("conduit.observability.setup.BatchSpanProcessor"):
                        with patch("conduit.observability.setup.trace"):
                            with patch("conduit.observability.setup.RedisInstrumentor"):
                                from conduit.observability.setup import setup_telemetry

                                setup_telemetry()

                                # Verify exporter was called with headers
                                call_kwargs = mock_exporter.call_args[1]
                                assert call_kwargs["headers"] == {
                                    "api-key": "test123",
                                    "x-custom": "value",
                                }

        # Restore
        setup_module._instrumented = original_value

    def test_setup_telemetry_with_fastapi_app(self):
        """Test setup_telemetry instruments FastAPI app."""
        from conduit.observability import setup as setup_module

        # Reset instrumented flag
        original_value = setup_module._instrumented
        setup_module._instrumented = False

        mock_app = MagicMock()

        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = False
            mock_settings.otel_metrics_enabled = False
            mock_settings.otel_service_name = "test-service"
            mock_settings.environment = "test"
            mock_settings.otel_exporter_otlp_endpoint = "http://localhost:4317"
            mock_settings.otel_exporter_otlp_headers = None

            with patch(
                "conduit.observability.setup.FastAPIInstrumentor"
            ) as mock_instrumentor:
                with patch("conduit.observability.setup.RedisInstrumentor"):
                    from conduit.observability.setup import setup_telemetry

                    setup_telemetry(app=mock_app)

                    # Verify FastAPI was instrumented
                    mock_instrumentor.instrument_app.assert_called_once_with(mock_app)

        # Restore
        setup_module._instrumented = original_value

    def test_shutdown_telemetry_disabled(self):
        """Test shutdown_telemetry does nothing when OTEL disabled."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = False

            from conduit.observability.setup import shutdown_telemetry

            # Should not raise any exceptions
            shutdown_telemetry()

    def test_shutdown_telemetry_enabled(self):
        """Test shutdown_telemetry calls provider shutdowns."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True
            mock_settings.otel_metrics_enabled = True

            with patch("conduit.observability.setup.trace") as mock_trace:
                with patch("conduit.observability.setup.metrics") as mock_metrics:
                    mock_trace_provider = MagicMock()
                    mock_trace_provider.shutdown = MagicMock()
                    mock_trace.get_tracer_provider.return_value = mock_trace_provider

                    mock_meter_provider = MagicMock()
                    mock_meter_provider.shutdown = MagicMock()
                    mock_metrics.get_meter_provider.return_value = mock_meter_provider

                    from conduit.observability.setup import shutdown_telemetry

                    shutdown_telemetry()

                    # Both providers should have shutdown called
                    mock_trace_provider.shutdown.assert_called_once()
                    mock_meter_provider.shutdown.assert_called_once()

    def test_shutdown_telemetry_traces_only(self):
        """Test shutdown_telemetry with only traces enabled."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True
            mock_settings.otel_metrics_enabled = False

            with patch("conduit.observability.setup.trace") as mock_trace:
                mock_trace_provider = MagicMock()
                mock_trace_provider.shutdown = MagicMock()
                mock_trace.get_tracer_provider.return_value = mock_trace_provider

                from conduit.observability.setup import shutdown_telemetry

                shutdown_telemetry()

                mock_trace_provider.shutdown.assert_called_once()

    def test_shutdown_telemetry_metrics_only(self):
        """Test shutdown_telemetry with only metrics enabled."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = False
            mock_settings.otel_metrics_enabled = True

            with patch("conduit.observability.setup.metrics") as mock_metrics:
                mock_meter_provider = MagicMock()
                mock_meter_provider.shutdown = MagicMock()
                mock_metrics.get_meter_provider.return_value = mock_meter_provider

                from conduit.observability.setup import shutdown_telemetry

                shutdown_telemetry()

                mock_meter_provider.shutdown.assert_called_once()

    def test_shutdown_telemetry_no_shutdown_method(self):
        """Test shutdown_telemetry handles providers without shutdown method."""
        with patch("conduit.observability.setup.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_traces_enabled = True
            mock_settings.otel_metrics_enabled = True

            with patch("conduit.observability.setup.trace") as mock_trace:
                with patch("conduit.observability.setup.metrics") as mock_metrics:
                    # Providers without shutdown attribute
                    mock_trace_provider = object()
                    mock_trace.get_tracer_provider.return_value = mock_trace_provider

                    mock_meter_provider = object()
                    mock_metrics.get_meter_provider.return_value = mock_meter_provider

                    from conduit.observability.setup import shutdown_telemetry

                    # Should not raise any exceptions
                    shutdown_telemetry()
