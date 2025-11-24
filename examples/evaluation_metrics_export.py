"""Example: Export evaluation metrics to OpenTelemetry.

This example shows how to set up periodic export of evaluation metrics
to an OTEL-compatible backend (e.g., Grafana Cloud, Honeycomb, Datadog).

Prerequisites:
    - DATABASE_URL configured with evaluation_metrics table
    - OTEL_ENABLED=true in environment
    - OTEL exporter endpoint configured

Usage:
    python examples/evaluation_metrics_export.py

The exporter will:
    1. Fetch latest evaluation metrics from database every 60 seconds
    2. Export regret, convergence, quality, and cost efficiency to OTEL
    3. Detect convergence status based on regret thresholds
    4. Continue running until Ctrl+C
"""

import asyncio
import logging
import os

from conduit.core.database import Database
from conduit.observability import setup_telemetry, shutdown_telemetry
from conduit.observability.evaluation_exporter import EvaluationMetricsExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run evaluation metrics exporter."""
    # Initialize OTEL (reads config from environment)
    setup_telemetry()

    # Connect to database
    db = Database()
    await db.connect()

    try:
        # Create exporter
        exporter = EvaluationMetricsExporter(
            db=db,
            interval_seconds=60,  # Export every 60 seconds
            time_windows=["last_hour", "last_day", "last_week"],  # Multiple windows
            convergence_threshold=0.05,  # 5% regret threshold
        )

        logger.info("Starting evaluation metrics export...")
        logger.info("Press Ctrl+C to stop")

        # Run exporter (infinite loop until interrupted)
        try:
            await exporter.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            exporter.stop()

    finally:
        # Cleanup
        await db.disconnect()
        shutdown_telemetry()
        logger.info("Export stopped")


if __name__ == "__main__":
    # Verify configuration
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL not configured")
        print("Set DATABASE_URL environment variable to your PostgreSQL connection string")
        exit(1)

    if not os.getenv("OTEL_ENABLED") == "true":
        print("WARNING: OTEL_ENABLED not set to 'true'")
        print("Metrics will not be exported. Set OTEL_ENABLED=true in environment")
        print("Also configure OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS")
        print()

    # Run
    asyncio.run(main())
