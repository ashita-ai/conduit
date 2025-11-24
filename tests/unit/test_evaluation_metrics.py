"""Unit tests for evaluation metrics OTEL export."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.database import Database
from conduit.observability.evaluation_exporter import EvaluationMetricsExporter
from conduit.observability.metrics import record_evaluation_metrics


class TestRecordEvaluationMetrics:
    """Tests for record_evaluation_metrics function."""

    def test_record_metrics_when_otel_disabled(self):
        """Test metrics recording is no-op when OTEL disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = False
            mock_settings.otel_metrics_enabled = True

            # Should not raise error, just skip
            record_evaluation_metrics(
                regret_oracle=0.05,
                regret_random=0.3,
                converged=True,
            )

    def test_record_metrics_when_metrics_disabled(self):
        """Test metrics recording is no-op when metrics disabled."""
        with patch("conduit.observability.metrics.settings") as mock_settings:
            mock_settings.otel_enabled = True
            mock_settings.otel_metrics_enabled = False

            # Should not raise error, just skip
            record_evaluation_metrics(
                regret_oracle=0.05,
                regret_random=0.3,
                converged=True,
            )

    @patch("conduit.observability.metrics._ensure_instruments")
    @patch("conduit.observability.metrics.settings")
    def test_record_all_metrics(self, mock_settings, mock_ensure):
        """Test recording all evaluation metrics."""
        mock_settings.otel_enabled = True
        mock_settings.otel_metrics_enabled = True

        # Mock instruments
        with patch(
            "conduit.observability.metrics._evaluation_regret_oracle_histogram"
        ) as mock_regret_oracle, patch(
            "conduit.observability.metrics._evaluation_regret_random_histogram"
        ) as mock_regret_random, patch(
            "conduit.observability.metrics._evaluation_convergence_histogram"
        ) as mock_convergence, patch(
            "conduit.observability.metrics._evaluation_quality_histogram"
        ) as mock_quality, patch(
            "conduit.observability.metrics._evaluation_cost_efficiency_histogram"
        ) as mock_cost_efficiency:

            record_evaluation_metrics(
                regret_oracle=0.05,
                regret_random=0.3,
                converged=True,
                quality_score=0.85,
                cost_efficiency=42.5,
                time_window="last_hour",
            )

            # Verify all metrics were recorded
            mock_regret_oracle.record.assert_called_once_with(
                0.05, {"time_window": "last_hour"}
            )
            mock_regret_random.record.assert_called_once_with(
                0.3, {"time_window": "last_hour"}
            )
            mock_convergence.record.assert_called_once_with(
                1.0, {"time_window": "last_hour"}
            )
            mock_quality.record.assert_called_once_with(
                0.85, {"time_window": "last_hour"}
            )
            mock_cost_efficiency.record.assert_called_once_with(
                42.5, {"time_window": "last_hour"}
            )

    @patch("conduit.observability.metrics._ensure_instruments")
    @patch("conduit.observability.metrics.settings")
    def test_record_partial_metrics(self, mock_settings, mock_ensure):
        """Test recording only some metrics (others None)."""
        mock_settings.otel_enabled = True
        mock_settings.otel_metrics_enabled = True

        with patch(
            "conduit.observability.metrics._evaluation_regret_oracle_histogram"
        ) as mock_regret_oracle, patch(
            "conduit.observability.metrics._evaluation_convergence_histogram"
        ) as mock_convergence:

            # Only record regret_oracle, skip others
            record_evaluation_metrics(
                regret_oracle=0.05,
                regret_random=None,
                converged=False,
                quality_score=None,
                cost_efficiency=None,
            )

            mock_regret_oracle.record.assert_called_once()
            mock_convergence.record.assert_called_once_with(
                0.0, {"time_window": "last_hour"}
            )

    @patch("conduit.observability.metrics._ensure_instruments")
    @patch("conduit.observability.metrics.settings")
    def test_convergence_boolean_conversion(self, mock_settings, mock_ensure):
        """Test convergence boolean is converted to 1.0/0.0."""
        mock_settings.otel_enabled = True
        mock_settings.otel_metrics_enabled = True

        with patch(
            "conduit.observability.metrics._evaluation_convergence_histogram"
        ) as mock_convergence:

            # Test converged=True -> 1.0
            record_evaluation_metrics(converged=True)
            mock_convergence.record.assert_called_with(1.0, {"time_window": "last_hour"})

            mock_convergence.reset_mock()

            # Test converged=False -> 0.0
            record_evaluation_metrics(converged=False)
            mock_convergence.record.assert_called_with(0.0, {"time_window": "last_hour"})


class TestEvaluationMetricsExporter:
    """Tests for EvaluationMetricsExporter class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test exporter initialization with defaults."""
        mock_db = MagicMock(spec=Database)

        exporter = EvaluationMetricsExporter(mock_db)

        assert exporter.db is mock_db
        assert exporter.interval_seconds == 60
        assert exporter.time_windows == ["last_hour", "last_day"]
        assert exporter.convergence_threshold == 0.05
        assert exporter.convergence_stability == 0.02
        assert exporter._running is False

    @pytest.mark.asyncio
    async def test_initialization_custom_params(self):
        """Test exporter initialization with custom parameters."""
        mock_db = MagicMock(spec=Database)

        exporter = EvaluationMetricsExporter(
            mock_db,
            interval_seconds=30,
            time_windows=["last_hour"],
            convergence_threshold=0.03,
        )

        assert exporter.interval_seconds == 30
        assert exporter.time_windows == ["last_hour"]
        assert exporter.convergence_threshold == 0.03

    @pytest.mark.asyncio
    async def test_check_convergence_true(self):
        """Test convergence detection when regret is low."""
        mock_db = MagicMock(spec=Database)
        exporter = EvaluationMetricsExporter(mock_db)

        # Regret below threshold -> converged
        assert exporter._check_convergence(0.03) is True

    @pytest.mark.asyncio
    async def test_check_convergence_false(self):
        """Test convergence detection when regret is high."""
        mock_db = MagicMock(spec=Database)
        exporter = EvaluationMetricsExporter(mock_db)

        # Regret above threshold -> not converged
        assert exporter._check_convergence(0.10) is False

    @pytest.mark.asyncio
    async def test_check_convergence_none(self):
        """Test convergence detection when regret is None."""
        mock_db = MagicMock(spec=Database)
        exporter = EvaluationMetricsExporter(mock_db)

        # None regret -> not converged
        assert exporter._check_convergence(None) is False

    @pytest.mark.asyncio
    async def test_export_once_success(self):
        """Test successful export of evaluation metrics."""
        mock_db = MagicMock(spec=Database)
        mock_db.fetch_latest_evaluation_metrics = AsyncMock(
            return_value={
                "regret_oracle": 0.05,
                "regret_random": 0.3,
                "quality_trend": 0.85,
                "cost_efficiency": 42.5,
                "convergence_rate": None,
            }
        )

        exporter = EvaluationMetricsExporter(
            mock_db, time_windows=["last_hour", "last_day"]
        )

        with patch("conduit.observability.evaluation_exporter.record_evaluation_metrics") as mock_record:
            await exporter.export_once()

            # Should have been called twice (once per time window)
            assert mock_record.call_count == 2

            # Check first call (last_hour)
            first_call = mock_record.call_args_list[0]
            assert first_call[1]["regret_oracle"] == 0.05
            assert first_call[1]["regret_random"] == 0.3
            assert first_call[1]["quality_score"] == 0.85
            assert first_call[1]["cost_efficiency"] == 42.5
            assert first_call[1]["converged"] is True  # regret < 0.05
            assert first_call[1]["time_window"] == "last_hour"

            # Check second call (last_day)
            second_call = mock_record.call_args_list[1]
            assert second_call[1]["time_window"] == "last_day"

    @pytest.mark.asyncio
    async def test_export_once_handles_errors(self):
        """Test export_once handles database errors gracefully."""
        mock_db = MagicMock(spec=Database)
        mock_db.fetch_latest_evaluation_metrics = AsyncMock(
            side_effect=Exception("Database error")
        )

        exporter = EvaluationMetricsExporter(mock_db)

        # Should not raise, just log warning
        await exporter.export_once()

    @pytest.mark.asyncio
    async def test_export_once_with_missing_metrics(self):
        """Test export when some metrics are None."""
        mock_db = MagicMock(spec=Database)
        mock_db.fetch_latest_evaluation_metrics = AsyncMock(
            return_value={
                "regret_oracle": 0.05,
                "regret_random": None,  # Missing
                "quality_trend": None,  # Missing
                "cost_efficiency": None,  # Missing
                "convergence_rate": None,
            }
        )

        exporter = EvaluationMetricsExporter(mock_db, time_windows=["last_hour"])

        with patch("conduit.observability.evaluation_exporter.record_evaluation_metrics") as mock_record:
            await exporter.export_once()

            # Should still export with partial data
            assert mock_record.call_count == 1
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["regret_oracle"] == 0.05
            assert call_kwargs["regret_random"] is None
            assert call_kwargs["quality_score"] is None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the exporter."""
        mock_db = MagicMock(spec=Database)
        mock_db.fetch_latest_evaluation_metrics = AsyncMock(return_value={})

        exporter = EvaluationMetricsExporter(mock_db, interval_seconds=0.1)

        # Start in background
        task = None
        try:
            import asyncio

            task = asyncio.create_task(exporter.start())

            # Wait a bit for it to start
            await asyncio.sleep(0.05)
            assert exporter._running is True

            # Stop it
            exporter.stop()
            await asyncio.sleep(0.15)  # Wait for loop to exit
            assert exporter._running is False

        finally:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using exporter as async context manager."""
        mock_db = MagicMock(spec=Database)
        mock_db.fetch_latest_evaluation_metrics = AsyncMock(return_value={})

        import asyncio

        exporter = EvaluationMetricsExporter(mock_db, interval_seconds=0.1)

        # Use as context manager
        async with exporter:
            await asyncio.sleep(0.05)
            assert exporter._running is True

        # Should be stopped after exit
        await asyncio.sleep(0.15)
        assert exporter._running is False
