"""add_model_latencies_table

Revision ID: dbc1e805b8da
Revises: d330b3ea662d
Create Date: 2025-11-24 13:36:05.914843

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dbc1e805b8da'
down_revision: Union[str, Sequence[str], None] = 'd330b3ea662d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create model_latencies table for historical latency tracking."""
    
    # Create model_latencies table
    op.create_table(
        'model_latencies',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('latency_seconds', sa.Float(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=True),
        sa.Column('region', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('latency_seconds >= 0.0', name='model_latencies_latency_check'),
        sa.CheckConstraint('token_count IS NULL OR token_count >= 0', name='model_latencies_tokens_check'),
        sa.CheckConstraint('complexity_score IS NULL OR (complexity_score >= 0.0 AND complexity_score <= 1.0)', name='model_latencies_complexity_check')
    )
    
    # Create indexes for efficient querying
    op.create_index('idx_latencies_model_time', 'model_latencies', ['model_id', sa.text('created_at DESC')])
    op.create_index('idx_latencies_created_at', 'model_latencies', [sa.text('created_at DESC')])
    op.create_index('idx_latencies_model_id', 'model_latencies', ['model_id'])
    
    # Create trigger to update updated_at timestamp
    op.execute("""
        CREATE OR REPLACE FUNCTION update_model_latencies_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER trigger_update_model_latencies_updated_at
        BEFORE UPDATE ON model_latencies
        FOR EACH ROW
        EXECUTE FUNCTION update_model_latencies_updated_at();
    """)


def downgrade() -> None:
    """Drop model_latencies table and related objects."""
    
    # Drop trigger and function
    op.execute("DROP TRIGGER IF EXISTS trigger_update_model_latencies_updated_at ON model_latencies")
    op.execute("DROP FUNCTION IF EXISTS update_model_latencies_updated_at()")
    
    # Drop table (indexes are dropped automatically)
    op.drop_table('model_latencies')
