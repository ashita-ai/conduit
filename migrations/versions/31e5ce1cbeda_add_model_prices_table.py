"""add_model_prices_table

Revision ID: 31e5ce1cbeda
Revises: 02c6d98e3c62
Create Date: 2025-11-18 16:31:50.123456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '31e5ce1cbeda'
down_revision: Union[str, Sequence[str], None] = '02c6d98e3c62'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add model_prices table with default pricing data."""

    # Create model_prices table
    op.create_table(
        'model_prices',
        sa.Column('model_id', sa.Text(), nullable=False),
        sa.Column('input_cost_per_million', sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column('output_cost_per_million', sa.Numeric(precision=12, scale=8), nullable=False),
        sa.Column('cached_input_cost_per_million', sa.Numeric(precision=12, scale=8), nullable=True),
        sa.Column('source', sa.Text(), nullable=True),
        sa.Column('snapshot_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('model_id'),
        sa.CheckConstraint('input_cost_per_million >= 0', name='model_prices_input_cost_check'),
        sa.CheckConstraint('output_cost_per_million >= 0', name='model_prices_output_cost_check'),
        sa.CheckConstraint('cached_input_cost_per_million IS NULL OR cached_input_cost_per_million >= 0', name='model_prices_cached_cost_check')
    )
    op.create_index('idx_model_prices_snapshot_at', 'model_prices', [sa.text('snapshot_at DESC')])

    # NOTE: Initial pricing data should be loaded using scripts/sync_pricing.py
    # Run: python scripts/sync_pricing.py
    # This fetches current pricing from llm-prices.com API (~90 models)
    #
    # For reference, here are the default models that should be available:
    # - gpt-4o-mini, gpt-4o, gpt-4o-2024-11-20
    # - claude-3.5-sonnet, claude-3-opus, claude-3-haiku
    # - Many more from Amazon, Google, Anthropic, xAI, etc.


def downgrade() -> None:
    """Remove model_prices table."""

    op.drop_index('idx_model_prices_snapshot_at', table_name='model_prices')
    op.drop_table('model_prices')
