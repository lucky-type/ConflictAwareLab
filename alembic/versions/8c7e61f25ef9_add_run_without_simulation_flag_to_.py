"""Add run_without_simulation flag to experiments

Revision ID: 8c7e61f25ef9
Revises: 5a1b0dd0f6b1
Create Date: 2026-02-09 03:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8c7e61f25ef9"
down_revision: Union[str, None] = "5a1b0dd0f6b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column(
            "run_without_simulation",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("experiments", "run_without_simulation")
