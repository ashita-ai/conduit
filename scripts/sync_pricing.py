#!/usr/bin/env python3
"""Sync LLM pricing data from LiteLLM to PostgreSQL database.

This script snapshots current pricing from LiteLLM's bundled model_cost database
and stores it in the model_prices table for historical tracking.

Usage:
    python scripts/sync_pricing.py [--dry-run] [--verbose] [--force]

Features:
- Uses LiteLLM's bundled pricing (no external API calls)
- Stores timestamped snapshots for historical analysis
- Supports dry-run mode for validation
- Includes cache pricing (read and creation costs)

Why snapshot pricing?
- Track pricing changes over time
- Understand routing decisions with historical context
- Audit cost calculations retroactively
- Compare model economics across time periods

Environment:
    DATABASE_URL - PostgreSQL connection string (required for actual sync)
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import litellm
from dotenv import load_dotenv

# Load .env file from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Skip models with unreasonably high pricing (likely data errors)
MAX_REASONABLE_COST_PER_MILLION = 1000.0  # $1000/1M tokens


def get_litellm_pricing() -> list[dict]:
    """Extract pricing data from LiteLLM's bundled model_cost.

    Returns:
        List of pricing records ready for database insertion
    """
    pricing_records = []
    snapshot_at = datetime.now(timezone.utc)
    skipped = 0

    for model_id, model_info in litellm.model_cost.items():
        # Skip non-chat models and the sample spec
        if model_id == "sample_spec":
            continue
        mode = model_info.get("mode", "")
        if mode and mode != "chat":
            continue

        # Skip models without input pricing
        if "input_cost_per_token" not in model_info:
            continue

        # Convert per-token to per-million
        input_cost = model_info.get("input_cost_per_token", 0.0) * 1_000_000
        output_cost = model_info.get("output_cost_per_token", 0.0) * 1_000_000

        # Skip unreasonable pricing (data errors in LiteLLM)
        if input_cost > MAX_REASONABLE_COST_PER_MILLION or output_cost > MAX_REASONABLE_COST_PER_MILLION:
            skipped += 1
            continue

        # Cache pricing (if available)
        cache_read_cost = model_info.get("cache_read_input_token_cost")

        pricing_records.append({
            "model_id": model_id,
            "input_cost_per_million": input_cost,
            "output_cost_per_million": output_cost,
            "cached_input_cost_per_million": (
                cache_read_cost * 1_000_000 if cache_read_cost else None
            ),
            "source": "litellm",
            "snapshot_at": snapshot_at,
        })

    logger.info(f"Extracted pricing for {len(pricing_records)} models from LiteLLM")
    if skipped:
        logger.info(f"Skipped {skipped} models with unreasonable pricing (>${MAX_REASONABLE_COST_PER_MILLION}/1M)")
    return pricing_records


async def get_latest_snapshot_date(database_url: str) -> str | None:
    """Get the date of the most recent pricing snapshot.

    Returns:
        ISO date string (YYYY-MM-DD) or None if no snapshots exist
    """
    try:
        # Disable statement cache for pgbouncer compatibility (Supabase, etc.)
        conn = await asyncpg.connect(database_url, statement_cache_size=0)
        try:
            row = await conn.fetchrow(
                "SELECT snapshot_at FROM model_prices ORDER BY snapshot_at DESC LIMIT 1"
            )
            if row and row["snapshot_at"]:
                return row["snapshot_at"].strftime("%Y-%m-%d")
            return None
        finally:
            await conn.close()
    except Exception as e:
        logger.debug(f"Could not check latest snapshot: {e}")
        return None


async def sync_pricing_to_database(
    pricing_records: list[dict],
    database_url: str,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Sync pricing data to PostgreSQL database.

    Args:
        pricing_records: List of pricing records from LiteLLM
        database_url: PostgreSQL connection string
        dry_run: If True, don't actually update database
        force: If True, insert even if today's snapshot exists

    Returns:
        Number of records inserted
    """
    # Dry run - just show what would be inserted
    if dry_run:
        logger.info(f"DRY RUN - Would insert {len(pricing_records)} pricing records")
        logger.info("Sample records:")
        for record in pricing_records[:5]:
            logger.info(
                f"  {record['model_id']}: "
                f"${record['input_cost_per_million']:.4f}/M in, "
                f"${record['output_cost_per_million']:.4f}/M out"
            )
        if len(pricing_records) > 5:
            logger.info(f"  ... and {len(pricing_records) - 5} more")
        return 0

    # Check if we already have a snapshot today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not force:
        latest_date = await get_latest_snapshot_date(database_url)
        if latest_date == today:
            logger.info(f"Pricing snapshot for {today} already exists. Use --force to add another.")
            return 0

    # Connect and insert using batch SQL for speed
    # Disable statement cache for pgbouncer compatibility (Supabase, etc.)
    logger.info(f"Inserting {len(pricing_records)} pricing records...")
    conn = await asyncpg.connect(database_url, statement_cache_size=0)
    try:
        # Build VALUES clauses for batch insert
        values = []
        for record in pricing_records:
            # Escape single quotes in model_id
            safe_id = record["model_id"].replace("'", "''")
            cache_val = record["cached_input_cost_per_million"]
            cache_str = str(cache_val) if cache_val is not None else "NULL"
            snapshot_str = record["snapshot_at"].isoformat()
            values.append(
                f"('{safe_id}', {record['input_cost_per_million']}, "
                f"{record['output_cost_per_million']}, {cache_str}, "
                f"'{record['source']}', '{snapshot_str}')"
            )

        # Insert in batches of 500 to avoid query size limits
        batch_size = 500
        inserted = 0
        for i in range(0, len(values), batch_size):
            batch = values[i : i + batch_size]
            sql = f"""INSERT INTO model_prices
                (model_id, input_cost_per_million, output_cost_per_million,
                 cached_input_cost_per_million, source, snapshot_at)
                VALUES {','.join(batch)}
                ON CONFLICT (model_id, snapshot_at) DO NOTHING"""
            await conn.execute(sql)
            inserted = min(i + batch_size, len(values))
            logger.info(f"  Inserted {inserted}/{len(values)} records...")

        logger.info(f"Sync complete: {len(values)} pricing records inserted")
        return len(values)
    finally:
        await conn.close()


def print_litellm_version_info():
    """Print LiteLLM version and pricing stats."""
    try:
        from litellm._version import version
    except (ImportError, AttributeError):
        version = "unknown"

    chat_models = sum(
        1 for m, info in litellm.model_cost.items()
        if m != "sample_spec"
        and info.get("mode", "") in ("", "chat")
        and "input_cost_per_token" in info
    )

    logger.info(f"LiteLLM version: {version}")
    logger.info(f"Chat models with pricing: {chat_models}")


async def main() -> int:
    """Main entry point for pricing sync script."""
    parser = argparse.ArgumentParser(
        description="Sync LLM pricing from LiteLLM to PostgreSQL database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without updating database",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force sync even if today's snapshot already exists",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print_litellm_version_info()

    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url and not args.dry_run:
        logger.error("DATABASE_URL environment variable not set")
        logger.info("Set DATABASE_URL to your PostgreSQL connection string")
        logger.info("Example: postgresql://user:pass@host:5432/dbname")
        return 1

    try:
        # Get pricing from LiteLLM
        pricing_records = get_litellm_pricing()

        # Sync to database
        inserted = await sync_pricing_to_database(
            pricing_records,
            database_url or "",
            dry_run=args.dry_run,
            force=args.force,
        )

        if args.dry_run:
            logger.info("Dry run complete - no changes made")
        elif inserted > 0:
            logger.info(f"Successfully stored {inserted} pricing snapshots")
        else:
            logger.info("No new snapshots stored (already up to date)")

        return 0

    except asyncpg.PostgresError as e:
        logger.error(f"Database error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
