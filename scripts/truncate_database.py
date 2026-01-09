#!/usr/bin/env python3
"""Truncate all database tables."""
import asyncio
import asyncpg
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


async def truncate_all_tables():
    """Truncate all tables in the database."""
    # Use direct connection for administrative tasks
    database_url = os.getenv("DIRECT_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("❌ No DATABASE_URL or DIRECT_URL found in environment")
        return

    try:
        conn = await asyncpg.connect(database_url)
        logger.info("✅ Connected to database")

        # Truncate tables in reverse dependency order
        tables = [
            "feedback",
            "implicit_feedback",
            "evaluation_metrics",
            "responses",
            "routing_decisions",
            "queries",
            "model_states",
            "bandit_state",
            "model_prices",
        ]

        for table in tables:
            try:
                await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.info(f"✅ Truncated {table}")
            except Exception as e:
                logger.warning(f"⚠️  Skipped {table}: {e}")

        await conn.close()
        logger.info("\n✅ Database truncation complete")

    except Exception as e:
        logger.error(f"❌ Failed to truncate database: {e}")


if __name__ == "__main__":
    asyncio.run(truncate_all_tables())
