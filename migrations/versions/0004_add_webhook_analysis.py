"""Add an analysis column to webhook_logs.

Phase 3's event-driven response: after a delivery reaches its terminal
state, an analysis agent may judge the event against the memory store
(see ``backend/service/event_analysis.py``).  The verdict is a 1:1
downstream product of the delivery — same lifecycle, same trace id — so
it lives on the delivery row instead of a separate table.  NULL means no
analysis ran (feature disabled, source not opted in, or pre-migration
row); a JSON object records the verdict or why it was skipped/failed.

revision: 0004_add_webhook_analysis
revises: 0003_add_patrol_error
"""

from __future__ import annotations

from alembic import op

revision = "0004_add_webhook_analysis"
down_revision = "0003_add_patrol_error"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE webhook_logs ADD COLUMN IF NOT EXISTS analysis JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE webhook_logs DROP COLUMN IF EXISTS analysis")
