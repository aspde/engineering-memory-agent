"""Drop the historical decay_factor column from memories.

The column was a frozen snapshot of the removed Ebbinghaus decay factor
(default 1.0); nothing reads or writes it since the decay weighting was
taken off the ranking path (see ``backend/service/recall.py`` docstring).
The only code reference left was a docstring saying "keep it so rows don't
need migrating" — but the production database is empty, so that reason no
longer holds and the column is dead schema.

The column's historical presence is preserved in the narrative of
``docs/memory-system.md`` (召回统计 section) and ``tests/eval/reports/
decay_ab_report.md``; this migration only removes the live column.

revision: 0002_drop_decay_factor
revises: 0001_baseline
"""

from __future__ import annotations

from alembic import op

revision = "0002_drop_decay_factor"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE memories DROP COLUMN IF EXISTS decay_factor")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS "
        "decay_factor FLOAT DEFAULT 1.0"
    )
