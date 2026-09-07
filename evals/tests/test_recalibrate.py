"""Unit tests for evals/recalibrate.py — the model-swap orchestration CLI.

Driven with fabricated report JSONs through the `--reports` path (no LLM, no
subprocess).  Covers: dry-run derives-but-writes-nothing, --confirm writes
floors.json with the right channel + thresholds, and the channel-match refusal
aborts instead of writing a mismatched floors file.
"""

from __future__ import annotations

import json

import pytest

from evals.multi_run_gate import FLOORS_FILE
from evals.recalibrate import main as recalibrate_main


def _report(overall: dict, *, provider: str = "openai", model: str = "m1") -> dict:
    return {
        "results": [{"suite": "extraction", "judge": "deterministic", "overall": overall}],
        "run_provenance": {"provider": provider, "model": model, "judge": "p:m"},
    }


@pytest.fixture
def sample_reports(tmp_path):
    """Three report JSONs sharing one channel so derive works."""
    paths = []
    for i, v in enumerate((0.9, 0.9, 0.9)):
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(_report({"entity_f1": v})), encoding="utf-8")
        paths.append(str(p))
    return ",".join(paths)


@pytest.fixture
def move_floors(tmp_path):
    """Back up + restore evals/floors.json so tests never clobber the real
    committed floors (which are calibrated for the legacy channel)."""
    original = FLOORS_FILE.read_bytes()
    yield
    FLOORS_FILE.write_bytes(original)


def test_dry_run_writes_nothing(sample_reports, tmp_path):
    rc = recalibrate_main(["--model", "m1", "--provider", "openai",
                           "--reports", sample_reports])
    assert rc == 0


def test_confirm_writes_expected_floors(sample_reports, tmp_path, move_floors, monkeypatch):
    monkeypatch.setattr("evals.recalibrate.FLOORS_FILE", tmp_path / "floors.json")
    rc = recalibrate_main(["--model", "m1", "--provider", "openai",
                           "--reports", sample_reports, "--confirm"])
    assert rc == 0
    written = json.loads((tmp_path / "floors.json").read_text(encoding="utf-8"))
    assert written["calibrated_channel"]["model"] == "m1"
    assert written["calibrated_channel"]["provider"] == "openai"
    # entity_f1 0.90 - 0.05 → 0.85
    assert written["thresholds"]["entity_f1"] == 0.85


def test_channel_mismatch_aborts_without_write(sample_reports, tmp_path, move_floors, monkeypatch):
    monkeypatch.setattr("evals.recalibrate.FLOORS_FILE", tmp_path / "floors.json")
    # Reports are channel m1, but the script is told to calibrate FOR m2 —
    # that would reintroduce the mismatch the system prevents.
    rc = recalibrate_main(["--model", "m2", "--provider", "openai",
                           "--reports", sample_reports, "--confirm"])
    assert rc == 2
    assert not (tmp_path / "floors.json").exists()


def test_missing_reports_returns_2():
    rc = recalibrate_main(["--model", "m1", "--reports", ""])
    assert rc == 2
