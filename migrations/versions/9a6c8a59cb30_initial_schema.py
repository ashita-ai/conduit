"""initial schema

Revision ID: 9a6c8a59cb30
Revises:
Create Date: 2025-11-18 14:23:32.836358

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9a6c8a59cb30'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial Conduit schema."""

    # queries table
    op.create_table(
        'queries',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Text(), nullable=True),
        sa.Column('context', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('constraints', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("length(trim(text)) > 0", name='queries_text_check')
    )
    op.create_index('idx_queries_user_id', 'queries', ['user_id'])
    op.create_index('idx_queries_created_at', 'queries', [sa.text('created_at DESC')])

    # routing_decisions table
    op.create_table(
        'routing_decisions',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('query_id', sa.Text(), nullable=False),
        sa.Column('selected_model', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['query_id'], ['queries.id'], ondelete='CASCADE'),
        sa.CheckConstraint('confidence >= 0.0 AND confidence <= 1.0', name='routing_decisions_confidence_check')
    )
    op.create_index('idx_routing_decisions_query_id', 'routing_decisions', ['query_id'])
    op.create_index('idx_routing_decisions_model', 'routing_decisions', ['selected_model'])
    op.create_index('idx_routing_decisions_created_at', 'routing_decisions', [sa.text('created_at DESC')])
    op.create_index('idx_routing_decisions_confidence', 'routing_decisions', [sa.text('confidence DESC')])

    # responses table
    op.create_table(
        'responses',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('query_id', sa.Text(), nullable=False),
        sa.Column('model', sa.Text(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('cost', sa.Numeric(precision=10, scale=6), nullable=False),
        sa.Column('latency', sa.Numeric(precision=10, scale=3), nullable=False),
        sa.Column('tokens', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['query_id'], ['queries.id'], ondelete='CASCADE'),
        sa.CheckConstraint('cost >= 0.0', name='responses_cost_check'),
        sa.CheckConstraint('latency >= 0.0', name='responses_latency_check'),
        sa.CheckConstraint('tokens >= 0', name='responses_tokens_check')
    )
    op.create_index('idx_responses_query_id', 'responses', ['query_id'])
    op.create_index('idx_responses_model', 'responses', ['model'])
    op.create_index('idx_responses_created_at', 'responses', [sa.text('created_at DESC')])
    op.create_index('idx_responses_cost', 'responses', [sa.text('cost DESC')])
    op.create_index('idx_responses_latency', 'responses', [sa.text('latency DESC')])

    # feedback table
    op.create_table(
        'feedback',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('response_id', sa.Text(), nullable=False),
        sa.Column('quality_score', sa.Numeric(precision=3, scale=2), nullable=False),
        sa.Column('user_rating', sa.Integer(), nullable=True),
        sa.Column('met_expectations', sa.Boolean(), nullable=False),
        sa.Column('comments', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['response_id'], ['responses.id'], ondelete='CASCADE'),
        sa.CheckConstraint('quality_score >= 0.0 AND quality_score <= 1.0', name='feedback_quality_check'),
        sa.CheckConstraint('user_rating IS NULL OR (user_rating >= 1 AND user_rating <= 5)', name='feedback_rating_check')
    )
    op.create_index('idx_feedback_response_id', 'feedback', ['response_id'])
    op.create_index('idx_feedback_created_at', 'feedback', [sa.text('created_at DESC')])
    op.create_index('idx_feedback_quality_score', 'feedback', [sa.text('quality_score DESC')])
    op.create_index('idx_feedback_met_expectations', 'feedback', ['met_expectations'])

    # implicit_feedback table (Phase 2+)
    op.create_table(
        'implicit_feedback',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('response_id', sa.Text(), nullable=False),
        sa.Column('query_id', sa.Text(), nullable=False),
        sa.Column('retry_detected', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('retry_delay_seconds', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('task_abandoned', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('latency_accepted', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('error_occurred', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('error_type', sa.Text(), nullable=True),
        sa.Column('response_used', sa.Boolean(), server_default=sa.text('true'), nullable=False),
        sa.Column('followup_queries', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['response_id'], ['responses.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['query_id'], ['queries.id'], ondelete='CASCADE'),
        sa.CheckConstraint('retry_delay_seconds IS NULL OR retry_delay_seconds >= 0.0', name='implicit_feedback_delay_check'),
        sa.CheckConstraint('followup_queries >= 0', name='implicit_feedback_followup_check')
    )
    op.create_index('idx_implicit_feedback_response_id', 'implicit_feedback', ['response_id'])
    op.create_index('idx_implicit_feedback_query_id', 'implicit_feedback', ['query_id'])
    op.create_index('idx_implicit_feedback_created_at', 'implicit_feedback', [sa.text('created_at DESC')])
    op.create_index('idx_implicit_feedback_retry_detected', 'implicit_feedback', ['retry_detected'])
    op.create_index('idx_implicit_feedback_error_occurred', 'implicit_feedback', ['error_occurred'])

    # model_states table
    op.create_table(
        'model_states',
        sa.Column('model_id', sa.Text(), nullable=False),
        sa.Column('alpha', sa.Numeric(precision=20, scale=10), nullable=False),
        sa.Column('beta', sa.Numeric(precision=20, scale=10), nullable=False),
        sa.Column('total_requests', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('total_cost', sa.Numeric(precision=10, scale=6), server_default=sa.text('0.0'), nullable=False),
        sa.Column('avg_quality', sa.Numeric(precision=3, scale=2), server_default=sa.text('0.0'), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('model_id'),
        sa.CheckConstraint('alpha > 0', name='model_states_alpha_check'),
        sa.CheckConstraint('beta > 0', name='model_states_beta_check'),
        sa.CheckConstraint('total_requests >= 0', name='model_states_requests_check'),
        sa.CheckConstraint('total_cost >= 0.0', name='model_states_cost_check'),
        sa.CheckConstraint('avg_quality >= 0.0 AND avg_quality <= 1.0', name='model_states_quality_check')
    )
    op.create_index('idx_model_states_updated_at', 'model_states', [sa.text('updated_at DESC')])
    op.create_index('idx_model_states_avg_quality', 'model_states', [sa.text('avg_quality DESC')])

    # Create view for recent routing performance
    op.execute("""
        CREATE OR REPLACE VIEW recent_routing_performance AS
        SELECT
            rd.selected_model,
            COUNT(*) as total_routes,
            AVG(rd.confidence) as avg_confidence,
            AVG(r.cost) as avg_cost,
            AVG(r.latency) as avg_latency,
            AVG(CASE WHEN f.met_expectations THEN 1.0 ELSE 0.0 END) as success_rate
        FROM routing_decisions rd
        LEFT JOIN responses r ON r.query_id = rd.query_id
        LEFT JOIN feedback f ON f.response_id = r.id
        WHERE rd.created_at > NOW() - INTERVAL '7 days'
        GROUP BY rd.selected_model
        ORDER BY total_routes DESC
    """)


def downgrade() -> None:
    """Drop all Conduit schema objects."""

    # Drop view
    op.execute("DROP VIEW IF EXISTS recent_routing_performance")

    # Drop tables (reverse order due to foreign keys)
    op.drop_table('model_states')
    op.drop_table('implicit_feedback')
    op.drop_table('feedback')
    op.drop_table('responses')
    op.drop_table('routing_decisions')
    op.drop_table('queries')
