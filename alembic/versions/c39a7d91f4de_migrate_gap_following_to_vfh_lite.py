"""Migrate legacy gap_following heuristic values to vfh_lite

Revision ID: c39a7d91f4de
Revises: 8c7e61f25ef9
Create Date: 2026-02-09 23:10:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c39a7d91f4de"
down_revision: Union[str, None] = "8c7e61f25ef9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "UPDATE experiments "
        "SET heuristic_algorithm='vfh_lite' "
        "WHERE heuristic_algorithm='gap_following'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE experiments "
        "SET heuristic_algorithm='gap_following' "
        "WHERE heuristic_algorithm='vfh_lite'"
    )

