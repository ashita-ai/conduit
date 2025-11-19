"""remove_metadata_column

Revision ID: 02c6d98e3c62
Revises: 9a6c8a59cb30
Create Date: 2025-11-18 16:30:47.123456

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '02c6d98e3c62'
down_revision: Union[str, Sequence[str], None] = '9a6c8a59cb30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Remove unused metadata column from routing_decisions table."""

    # Drop metadata column and its index
    op.drop_column('routing_decisions', 'metadata')


def downgrade() -> None:
    """Restore metadata column to routing_decisions table."""

    # Restore metadata column with default empty JSON
    op.add_column(
        'routing_decisions',
        sa.Column(
            'metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False
        )
    )
