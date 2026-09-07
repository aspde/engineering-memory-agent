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


from pathlib import Path as _Path


def test_floors_calibration_is_evidenced():
    """floors.json must be usable as a gate — and its `calibrated_from` must
    point at something that actually exists in the repo.

    The red run that motivated this (34027523857) had floors.json stamped
    with a channel (omen-alpha) while the thresholds were still DeepSeek-era
    numbers: the gate passed on channel-match, then failed on metrics the
    numbers were never calibrated for.  The channel match check alone can't
    catch a floors file whose `calibrated_from` names a run/note, so this
    asserts the calibration is *evidenced* — calibrated_from must mention
    either a committed report (baseline filename) or a CI run id, and the
    thresholds must be for the metric set the gate actually gates on.
    """
    import json

    floors_path = _Path(__file__).resolve().parents[1] / "floors.json"
    assert floors_path.exists(), "evals/floors.json must exist (the gate reads it)"
    data = json.loads(floors_path.read_text(encoding="utf-8"))
    channel = data.get("calibrated_channel") or {}
    assert channel.get("provider"), "floors.json: calibrated_channel.provider missing"
    assert channel.get("model"), "floors.json: calibrated_channel.model missing"
    from_run = str(channel.get("calibrated_from", "") or "")
    assert from_run, "floors.json: calibrated_from must record where the numbers came from"

    thresholds = data.get("thresholds") or {}
    # The gate gates on a known metric set — a floors file with none of the
    # gated metrics would be inert.
    gated = {"entity_f1", "relation_f1", "fact_coverage", "conflict_f1",
             "merge_fact_coverage", "worthy_f1", "tool_accuracy", "groundedness",
             "citation_rate", "expected_recall"}
    assert gated & set(thresholds), \
        "floors.json: no actual gate metric present — the file is inert"


def test_floors_thresholds_are_not_placeholder():
    """A floor of 0 or > 1 is a placeholder, not a calibration — it either
    gates nothing (0) or is impossible (1)."""
    import json

    floors_path = _Path(__file__).resolve().parents[1] / "floors.json"
    data = json.loads(floors_path.read_text(encoding="utf-8"))
    thresholds = data.get("thresholds") or {}
    for metric, floor in thresholds.items():
        assert 0 < float(floor) <= 1, (
            f"floors.json: {metric} floor {floor!r} is a placeholder "
            "(0 gates nothing, >1 impossible)"
        )
