"""Baseline provenance gate — every committed baseline states its channel.

The write baseline was committed without provenance; the repo then grew
three conflicting accounts of which LLM channel measured it (DeepSeek /
ox-alpha-free / none) and eval.yml gated provisionally against a baseline
from a different channel.  This test makes that accident unrepeatable: a
baseline that cannot answer "which model + judge measured you" fails
collection.

Two equivalent shapes are accepted (the assert is on the *information*,
not the field name — old baselines predate ``run_provenance``):

- ``run_provenance`` (current, auto-stamped by ``evals.core.run_provenance``)
  with non-empty ``model``;
- ``environment`` (legacy baselines) with non-empty ``llm_model``.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _iter_baseline_files():
    yield from sorted(_REPORTS.glob("*baseline*.json"))


def test_baselines_exist():
    """The gate has something to check — silent zero-baseline pass would
    defeat the whole point."""
    files = list(_iter_baseline_files())
    assert files, "no baseline files found under evals/reports/"
    # The three known baselines must be present (rename without updating
    # this test = provenance gate silently stops covering that file).
    names = {p.name for p in files}
    assert {
        "llm-eval-baseline.json",
        "llm-eval-semantic-baseline.json",
        "write-eval-baseline.json",
    } <= names


def test_every_baseline_states_its_model():
    for path in _iter_baseline_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        provenance = data.get("run_provenance") or {}
        environment = data.get("environment") or {}
        model = (
            provenance.get("model")
            or environment.get("llm_model")
        )
        judge = (
            provenance.get("judge")
            or environment.get("judge_model")
            or environment.get("judge_mode")
        )
        assert model, (
            f"{path.name}: baseline states no model — a baseline without "
            "provenance is not comparable and must not be committed "
            "(regenerate via run_llm_eval so run_provenance is stamped)"
        )
        assert judge, f"{path.name}: baseline states no judge channel"


def test_provenance_model_is_not_a_placeholder():
    """A provenance block with an empty/placeholder model is as useless as
    none — stamping machinery must not be bypassable with empty strings."""
    for path in _iter_baseline_files():
        data = json.loads(path.read_text(encoding="utf-8"))
        provenance = data.get("run_provenance") or {}
        environment = data.get("environment") or {}
        model = str(provenance.get("model") or environment.get("llm_model") or "").strip()
        assert model not in ("", "unknown", "n/a", "null"), (
            f"{path.name}: placeholder model value {model!r}"
        )
