"""Add evaluation policy mode and heuristic algorithm fields to experiments

Revision ID: 5a1b0dd0f6b1
Revises: ee3a3eb4cdd6
Create Date: 2026-02-09 01:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5a1b0dd0f6b1"
down_revision: Union[str, None] = "ee3a3eb4cdd6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "experiments",
        sa.Column("evaluation_policy_mode", sa.String(length=20), nullable=False, server_default="model"),
    )
    op.add_column(
        "experiments",
        sa.Column("heuristic_algorithm", sa.String(length=50), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("experiments", "heuristic_algorithm")
    op.drop_column("experiments", "evaluation_policy_mode")
