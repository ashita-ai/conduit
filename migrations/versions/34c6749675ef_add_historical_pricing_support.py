"""add_historical_pricing_support

Revision ID: 34c6749675ef
Revises: 704a5c62805b
Create Date: 2025-11-29 00:14:47.858982

Adds support for historical pricing snapshots:
- Changes primary key from model_id to id (SERIAL)
- Adds UNIQUE constraint on (model_id, snapshot_at) for historical tracking
- This enables price history analysis and explains routing decisions over time
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '34c6749675ef'
down_revision: Union[str, Sequence[str], None] = '704a5c62805b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable historical pricing snapshots.

    Changes:
    1. Add id column as SERIAL (auto-increment integer)
    2. Drop existing PRIMARY KEY constraint on model_id
    3. Make id the new PRIMARY KEY
    4. Add UNIQUE constraint on (model_id, snapshot_at) for historical snapshots

    This allows multiple pricing snapshots per model over time, enabling:
    - Price history tracking
    - Analysis of cost trends
    - Understanding routing decisions in historical context
    """

    # Step 1: Add id column as SERIAL (auto-incrementing integer)
    # Use identity for PostgreSQL 10+ compatibility
    op.add_column(
        'model_prices',
        sa.Column('id', sa.Integer(), sa.Identity(always=False), nullable=False)
    )

    # Step 2: Drop existing PRIMARY KEY constraint on model_id
    op.drop_constraint('model_prices_pkey', 'model_prices', type_='primary')

    # Step 3: Set id as new PRIMARY KEY
    op.create_primary_key('model_prices_pkey', 'model_prices', ['id'])

    # Step 4: Add UNIQUE constraint on (model_id, snapshot_at)
    # This ensures we can have multiple snapshots per model, but only one per timestamp
    op.create_unique_constraint(
        'uq_model_prices_model_snapshot',
        'model_prices',
        ['model_id', 'snapshot_at']
    )

    # Step 5: Add index on model_id for fast lookups of latest pricing
    op.create_index(
        'idx_model_prices_model_id',
        'model_prices',
        ['model_id']
    )


def downgrade() -> None:
    """Revert to single-price-per-model schema.

    WARNING: This will DELETE all historical pricing data except the most recent
    snapshot for each model. Only run this if you're certain you want to lose
    pricing history.
    """

    # Step 1: Delete all but the most recent snapshot for each model
    # This is necessary because the old schema only supports one price per model
    op.execute("""
        DELETE FROM model_prices
        WHERE id NOT IN (
            SELECT DISTINCT ON (model_id) id
            FROM model_prices
            ORDER BY model_id, snapshot_at DESC
        )
    """)

    # Step 2: Drop indexes and constraints
    op.drop_index('idx_model_prices_model_id', table_name='model_prices')
    op.drop_constraint('uq_model_prices_model_snapshot', 'model_prices', type_='unique')

    # Step 3: Drop existing PRIMARY KEY on id
    op.drop_constraint('model_prices_pkey', 'model_prices', type_='primary')

    # Step 4: Restore PRIMARY KEY on model_id
    op.create_primary_key('model_prices_pkey', 'model_prices', ['model_id'])

    # Step 5: Drop id column
    op.drop_column('model_prices', 'id')
