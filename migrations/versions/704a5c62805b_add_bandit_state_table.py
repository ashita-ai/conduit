"""add_bandit_state_table

Revision ID: 704a5c62805b
Revises: d330b3ea662d
Create Date: 2025-11-25 20:20:09.823404

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '704a5c62805b'
down_revision: Union[str, Sequence[str], None] = 'd330b3ea662d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create bandit_state table for persisting bandit algorithm state."""
    op.execute("""
        CREATE TABLE IF NOT EXISTS bandit_state (
            id SERIAL PRIMARY KEY,
            router_id VARCHAR(255) NOT NULL,
            bandit_id VARCHAR(255) NOT NULL,
            state_json JSONB NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(router_id, bandit_id)
        );

        CREATE INDEX IF NOT EXISTS idx_bandit_state_router_id
        ON bandit_state(router_id);

        CREATE INDEX IF NOT EXISTS idx_bandit_state_updated_at
        ON bandit_state(updated_at DESC);

        COMMENT ON TABLE bandit_state IS 'Persisted state for bandit algorithms (UCB1, LinUCB, Thompson Sampling, etc.)';
        COMMENT ON COLUMN bandit_state.router_id IS 'Unique identifier for the router instance';
        COMMENT ON COLUMN bandit_state.bandit_id IS 'Algorithm identifier (ucb1, linucb, hybrid_router, etc.)';
        COMMENT ON COLUMN bandit_state.state_json IS 'JSONB serialized BanditState or HybridRouterState';
        COMMENT ON COLUMN bandit_state.version IS 'Version counter for optimistic locking';
    """)


def downgrade() -> None:
    """Drop bandit_state table."""
    op.execute("""
        DROP INDEX IF EXISTS idx_bandit_state_updated_at;
        DROP INDEX IF EXISTS idx_bandit_state_router_id;
        DROP TABLE IF EXISTS bandit_state;
    """)
