"""Periodic evaluation metrics exporter for OpenTelemetry.

This module exports evaluation metrics from the evaluation_metrics table
to OTEL at regular intervals, similar to how LiteLLM exports callback data.

Features:
    - Async periodic export task
    - Configurable export interval
    - Graceful error handling (failures don't crash the app)
    - Automatic convergence detection
    - Multi-window support (last_hour, last_day, etc.)
"""

import asyncio
import logging
from typing import Optional

from conduit.core.database import Database
from conduit.observability.metrics import record_evaluation_metrics

logger = logging.getLogger(__name__)


class EvaluationMetricsExporter:
    """Periodic exporter for evaluation metrics to OTEL.

    Fetches latest evaluation metrics from database and exports them
    to OpenTelemetry at regular intervals.

    Example:
        >>> db = Database(database_url="postgresql://...")
        >>> await db.connect()
        >>> exporter = EvaluationMetricsExporter(db, interval_seconds=60)
        >>> # Start background task
        >>> asyncio.create_task(exporter.start())
    """

    def __init__(
        self,
        db: Database,
        interval_seconds: int = 60,
        time_windows: list[str] | None = None,
        convergence_threshold: float = 0.05,
        convergence_stability: float = 0.02,
    ):
        """Initialize evaluation metrics exporter.

        Args:
            db: Database instance for querying evaluation_metrics table
            interval_seconds: Export interval in seconds (default: 60)
            time_windows: Time windows to export (default: ["last_hour", "last_day"])
            convergence_threshold: Regret threshold for convergence (default: 0.05)
            convergence_stability: Std dev threshold for convergence (default: 0.02)
        """
        self.db = db
        self.interval_seconds = interval_seconds
        self.time_windows = time_windows or ["last_hour", "last_day"]
        self.convergence_threshold = convergence_threshold
        self.convergence_stability = convergence_stability
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(
            f"EvaluationMetricsExporter initialized: "
            f"interval={interval_seconds}s, windows={self.time_windows}"
        )

    async def export_once(self) -> None:
        """Export evaluation metrics once for all configured time windows.

        Fetches latest metrics from database and records them to OTEL.
        Errors are logged but not raised (fire-and-forget pattern).
        """
        try:
            for time_window in self.time_windows:
                # Fetch latest metrics for this time window
                metrics = await self.db.fetch_latest_evaluation_metrics(time_window)

                # Detect convergence based on regret
                converged = self._check_convergence(
                    regret_oracle=metrics.get("regret_oracle")
                )

                # Export to OTEL
                record_evaluation_metrics(
                    regret_oracle=metrics.get("regret_oracle"),
                    regret_random=metrics.get("regret_random"),
                    converged=converged,
                    quality_score=metrics.get("quality_trend"),
                    cost_efficiency=metrics.get("cost_efficiency"),
                    time_window=time_window,
                )

                logger.debug(
                    f"Exported evaluation metrics for {time_window}: "
                    f"regret_oracle={metrics.get('regret_oracle')}, "
                    f"converged={converged}"
                )

        except Exception as e:
            # Never crash the app due to metrics export failures
            logger.warning(f"Failed to export evaluation metrics: {e}")

    def _check_convergence(self, regret_oracle: float | None) -> bool:
        """Check if bandit has converged based on regret.

        Convergence criteria:
            - Regret vs Oracle < convergence_threshold (default: 5%)

        Args:
            regret_oracle: Latest regret vs oracle value

        Returns:
            True if converged, False otherwise

        Note:
            This is a simplified convergence check. Full convergence
            detection would also check stability (std dev over time).
        """
        if regret_oracle is None:
            return False

        # Simple convergence: regret is low
        return regret_oracle < self.convergence_threshold

    async def start(self) -> None:
        """Start periodic export task.

        Exports metrics at regular intervals until stopped.
        Safe to call multiple times (idempotent).

        Note:
            This is an infinite loop that should be run as a background task:
            `asyncio.create_task(exporter.start())`
        """
        if self._running:
            logger.debug("Exporter already running, skipping start")
            return

        self._running = True
        logger.info(
            f"Starting evaluation metrics export (interval={self.interval_seconds}s)"
        )

        while self._running:
            await self.export_once()
            await asyncio.sleep(self.interval_seconds)

    def stop(self) -> None:
        """Stop periodic export task.

        Gracefully stops the export loop at the next iteration.
        """
        if not self._running:
            logger.debug("Exporter not running, skipping stop")
            return

        self._running = False
        logger.info("Stopping evaluation metrics export")

    async def __aenter__(self) -> "EvaluationMetricsExporter":
        """Async context manager entry."""
        self._task = asyncio.create_task(self.start())
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        self.stop()
        if self._task:
            await self._task
