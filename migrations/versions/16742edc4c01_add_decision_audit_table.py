"""add_decision_audit_table

Revision ID: 16742edc4c01
Revises: 34c6749675ef
Create Date: 2026-01-14 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '16742edc4c01'
down_revision: Union[str, Sequence[str], None] = '34c6749675ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create decision_audit table for routing decision audit trail.

    Stores detailed context for each routing decision:
    - Algorithm state at decision time (scores, phase, query count)
    - Feature vector used for contextual decisions
    - Constraints applied (cost budget, latency, etc.)
    - Selected model and reasoning

    Used for:
    - Debugging: Why did Conduit select model X for query Y?
    - Compliance: Regulatory audit of AI decision-making
    - Analysis: Post-mortem investigation of routing behavior
    """

    op.create_table(
        'decision_audit',
        # Primary key
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),

        # Foreign keys (nullable for flexibility - audit can outlive source data)
        sa.Column('decision_id', sa.Text(), nullable=False),
        sa.Column('query_id', sa.Text(), nullable=False),

        # Decision snapshot
        sa.Column('selected_model', sa.Text(), nullable=False),
        sa.Column('fallback_chain', postgresql.ARRAY(sa.Text()), server_default='{}', nullable=False),
        sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=False),

        # Algorithm context
        sa.Column('algorithm_phase', sa.Text(), nullable=False),
        sa.Column('query_count', sa.Integer(), nullable=False),

        # Scores at decision time (JSONB for flexibility)
        # Structure: {"model_id": {"mean": 0.7, "uncertainty": 0.15, "total": 0.85}, ...}
        sa.Column('arm_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False),

        # Feature vector (for contextual algorithms like LinUCB)
        # Stored as JSONB array for flexibility across embedding dimensions
        sa.Column('feature_vector', postgresql.JSONB(astext_type=sa.Text()), nullable=True),

        # Constraints applied
        sa.Column('constraints_applied', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),

        # Reasoning text
        sa.Column('reasoning', sa.Text(), nullable=True),

        # Timing
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),

        # Primary key constraint
        sa.PrimaryKeyConstraint('id'),

        # Check constraints
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='decision_audit_confidence_check'),
        sa.CheckConstraint('query_count >= 0', name='decision_audit_query_count_check'),
    )

    # Indexes for common query patterns
    op.create_index('idx_decision_audit_decision_id', 'decision_audit', ['decision_id'])
    op.create_index('idx_decision_audit_query_id', 'decision_audit', ['query_id'])
    op.create_index('idx_decision_audit_selected_model', 'decision_audit', ['selected_model'])
    op.create_index('idx_decision_audit_algorithm_phase', 'decision_audit', ['algorithm_phase'])
    op.create_index('idx_decision_audit_created_at', 'decision_audit', [sa.text('created_at DESC')])

    # Composite index for time-range queries by model
    op.create_index(
        'idx_decision_audit_model_time',
        'decision_audit',
        ['selected_model', sa.text('created_at DESC')]
    )


def downgrade() -> None:
    """Drop decision_audit table."""

    op.drop_index('idx_decision_audit_model_time', table_name='decision_audit')
    op.drop_index('idx_decision_audit_created_at', table_name='decision_audit')
    op.drop_index('idx_decision_audit_algorithm_phase', table_name='decision_audit')
    op.drop_index('idx_decision_audit_selected_model', table_name='decision_audit')
    op.drop_index('idx_decision_audit_query_id', table_name='decision_audit')
    op.drop_index('idx_decision_audit_decision_id', table_name='decision_audit')
    op.drop_table('decision_audit')
