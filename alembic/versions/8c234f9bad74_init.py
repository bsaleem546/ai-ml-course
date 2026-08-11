"""init

Revision ID: 8c234f9bad74
Revises: 20e7f26d9c67
Create Date: 2026-08-11 16:54:58.567556

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8c234f9bad74'
down_revision: Union[str, Sequence[str], None] = '20e7f26d9c67'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
