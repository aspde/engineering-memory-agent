"""Add the scenario_runs table.

Phase 4 thickening: scenario runs (currently only postmortem) were
ephemeral — the compose result went straight back to the caller and was
gone.  This table records every run (manual and event-triggered) with
its trigger, status, raw markdown, parsed JSON contract findings, and
error — the persistence anchor that the save-as-memory endpoint and the
event-trigger gate both read from.

revision: 0005_add_scenario_runs
revises: 0004_add_webhook_analysis
"""

from __future__ import annotations

from alembic import op

revision = "0005_add_scenario_runs"
down_revision = "0004_add_webhook_analysis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS scenario_runs (
            id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            scenario_key TEXT NOT NULL,
            trigger      TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'running',
            params       JSONB DEFAULT '{}',
            result_md    TEXT,
            findings     JSONB,
            contract_ok  BOOLEAN,
            error        TEXT,
            created_at   TIMESTAMPTZ DEFAULT now(),
            completed_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_scenario_runs_created "
        "ON scenario_runs (created_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS scenario_runs")
