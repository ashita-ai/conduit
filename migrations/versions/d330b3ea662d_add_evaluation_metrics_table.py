"""add_evaluation_metrics_table

Revision ID: d330b3ea662d
Revises: 31e5ce1cbeda
Create Date: 2025-11-23 03:18:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd330b3ea662d'
down_revision: Union[str, Sequence[str], None] = '31e5ce1cbeda'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add evaluation_metrics table for ongoing ML routing performance tracking.

    This table stores regret metrics (vs Oracle, vs Random, vs AlwaysBest baselines),
    convergence metrics, quality trends, and cost efficiency over time.

    Key metrics:
    - regret_vs_oracle: Distance from perfect routing (0.0 = Oracle performance)
    - regret_vs_random: Distance from random routing (negative = worse than random)
    - quality_trend: 7-day moving average of quality scores
    - cost_efficiency: Quality achieved per dollar spent
    """

    # evaluation_metrics table
    op.create_table(
        'evaluation_metrics',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('metric_name', sa.Text(), nullable=False),
        sa.Column('metric_value', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('baseline_comparison', sa.Text(), nullable=True),
        sa.Column('time_window', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("metric_name IN ('regret_vs_oracle', 'regret_vs_random', 'regret_vs_always_best', 'quality_trend', 'cost_efficiency', 'convergence_rate')", name='evaluation_metrics_name_check'),
        sa.CheckConstraint("time_window IN ('last_hour', 'last_day', 'last_week', 'last_month', 'all_time')", name='evaluation_metrics_window_check')
    )
    op.create_index('idx_evaluation_metrics_name', 'evaluation_metrics', ['metric_name'])
    op.create_index('idx_evaluation_metrics_window', 'evaluation_metrics', ['time_window'])
    op.create_index('idx_evaluation_metrics_created_at', 'evaluation_metrics', [sa.text('created_at DESC')])

    # Create view for recent evaluation summary
    op.execute("""
        CREATE OR REPLACE VIEW recent_evaluation_summary AS
        SELECT
            metric_name,
            time_window,
            AVG(metric_value) as avg_value,
            MIN(metric_value) as min_value,
            MAX(metric_value) as max_value,
            COUNT(*) as sample_count,
            MAX(created_at) as last_updated
        FROM evaluation_metrics
        WHERE created_at > NOW() - INTERVAL '7 days'
        GROUP BY metric_name, time_window
        ORDER BY metric_name, time_window
    """)


def downgrade() -> None:
    """Drop evaluation_metrics table and view."""

    # Drop view
    op.execute("DROP VIEW IF EXISTS recent_evaluation_summary")

    # Drop table
    op.drop_table('evaluation_metrics')
