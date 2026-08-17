"""Add an error column to patrol_logs.

A patrol that fails outside the structural-validation path (a provider
error after retries, a timeout, a mid-run cancellation, or a
non-serialisable findings payload) sets ``status = 'failed'`` but had
nowhere to record *why* — the reason lived only in the backend log. The
UI then showed a bare "失败" with null findings and no way to diagnose it.

This column captures ``run_patrol``'s ``error_msg`` so the failure reason
is queryable via ``GET /api/patrol/logs/{id}`` and shown in the detail
view. It stays NULL for completed and interrupted runs.

revision: 0003_add_patrol_error
revises: 0002_drop_decay_factor
"""

from __future__ import annotations

from alembic import op

revision = "0003_add_patrol_error"
down_revision = "0002_drop_decay_factor"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE patrol_logs ADD COLUMN IF NOT EXISTS error TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE patrol_logs DROP COLUMN IF EXISTS error")
