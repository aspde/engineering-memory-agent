"""LLM judge vs human calibration — quantify judge agreement.

The eval's LLM-as-judge (:func:`tests.eval.llm_judge.judge_answer`) is an LLM
judging a model's answers; its verdicts have no oracle.  :mod:
``tests.eval.experiments.judge_calibration_samples`` supplies the missing oracle — a
hand-authored sample set with human verdicts.  This module runs the *real*
configured judge over those samples and quantifies how closely the judge
tracks the human:

- ``grounded_agreement`` — fraction of judged samples where the judge's
  ``grounded`` flag matches the human label.
- ``coverage_precision`` / ``coverage_recall`` / ``coverage_f1`` — mean of the
  per-sample P/R/F1 between the judge's ``covered_facts`` and the human's
  ``human_covered_facts`` sets.
- ``false_negative`` — human grounded but judge ungrounded (the ``ans-006``
  class of error: a faithful paraphrase penalized for not repeating the
  required-fact strings verbatim).
- ``false_positive`` — human ungrounded but judge grounded.

A judge call that fails or times out is recorded in ``judge_errors`` and
*excluded* from every denominator — the same semantic as ``llm_runner``'s
``judge_errors``, so a degraded judge channel reads as "not measured" rather
than "disagrees".

The judge runs on the configured ``LLM_JUDGE_*`` provider; a real run refuses
to start when that block is incomplete (the verdicts would otherwise come from
the primary model — self-judging, which defeats calibration).

Examples:
    # Validate the sample set (no LLM needed)
    python -m tests.eval.experiments.judge_calibration --validate-only

    # Cheap smoke run — first 3 samples only
    python -m tests.eval.experiments.judge_calibration --sample 3

    # Full calibration with Markdown + JSON reports
    python -m tests.eval.experiments.judge_calibration \
        --report-md tests/eval/reports/judge_calibration_report.md \
        --report-json tests/eval/reports/judge_calibration_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.shared.config import config
from tests.eval.experiments.judge_calibration_samples import (
    CALIBRATION_SAMPLES,
    CalibrationSample,
    validate_calibration_samples,
)
from tests.eval.llm_judge import judge_answer

# Judge callable signature, matching tests.eval.llm_judge.judge_answer.
JudgeCallable = Callable[[str, str, str, list[str]], Any]


# ── Metrics (pure, no I/O) ────────────────────────────────────────


def coverage_prf(
    predicted: Sequence[str], human: Sequence[str]
) -> tuple[float, float, float]:
    """P/R/F1 of the judge's ``covered_facts`` vs the human's.

    Unlike the extraction metrics (which treat a 0/0 denominator as 0.0), an
    empty *predicted* AND empty *human* set is *perfect agreement* here — both
    sides said nothing is covered — so it scores (1.0, 1.0, 1.0).  A one-sided
    empty set scores 0.0 on the corresponding metric:

        - predicted empty, human non-empty → precision 1.0 (no false
          positives), recall 0.0 (every fact missed) → F1 0.0.
        - predicted non-empty, human empty → precision 0.0 (nothing the judge
          credited was real), recall 1.0 → F1 0.0.
    """
    pred = set(predicted or [])
    hum = set(human or [])
    tp = len(pred & hum)
    fp = len(pred - hum)
    fn = len(hum - pred)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)


def compare_sample(sample: CalibrationSample, verdict: dict[str, Any]) -> dict[str, Any]:
    """One sample's human-vs-judge row (what the report renders)."""
    judge_grounded = bool(verdict.get("grounded", False))
    judge_covered = list(verdict.get("covered_facts") or [])
    precision, recall, f1 = coverage_prf(judge_covered, sample.human_covered_facts)
    return {
        "id": sample.id,
        "base_id": sample.base_id,
        "query": sample.query[:100],
        "human_grounded": sample.human_grounded,
        "judge_grounded": judge_grounded,
        "grounded_agree": sample.human_grounded == judge_grounded,
        "false_negative": sample.human_grounded and not judge_grounded,
        "false_positive": (not sample.human_grounded) and judge_grounded,
        "human_covered_facts": list(sample.human_covered_facts),
        "judge_covered_facts": judge_covered,
        "coverage_precision": precision,
        "coverage_recall": recall,
        "coverage_f1": f1,
        "ungrounded_claims": list(verdict.get("ungrounded_claims") or []),
        "answer": sample.answer,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Overall calibration metrics over *judged* rows (judge errors excluded).

    ``rows`` holds one entry per sample that produced a verdict; failed judge
    calls never land here, so the denominators are the number of judged
    samples (matching the ``judge_errors``-excluded semantic of the eval
    runner).  Empty rows → all zeros.
    """
    n = len(rows)
    if n == 0:
        return {
            "n_judged": 0,
            "grounded_agreement": 0.0,
            "coverage_precision": 0.0,
            "coverage_recall": 0.0,
            "coverage_f1": 0.0,
            "false_negative": 0.0,
            "false_positive": 0.0,
        }
    agreed = sum(1 for r in rows if r["grounded_agree"])
    return {
        "n_judged": n,
        "grounded_agreement": agreed / n,
        "coverage_precision": sum(r["coverage_precision"] for r in rows) / n,
        "coverage_recall": sum(r["coverage_recall"] for r in rows) / n,
        "coverage_f1": sum(r["coverage_f1"] for r in rows) / n,
        "false_negative": float(sum(1 for r in rows if r["false_negative"])),
        "false_positive": float(sum(1 for r in rows if r["false_positive"])),
    }


# ── Result ────────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    samples: list[CalibrationSample]
    per_sample: list[dict[str, Any]]
    judge_errors: list[dict[str, str]]
    judge_provider: str
    judge_model: str
    generated_at: str = ""

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_judged(self) -> int:
        return len(self.per_sample)

    @property
    def n_judge_errors(self) -> int:
        return len(self.judge_errors)

    @property
    def overall(self) -> dict[str, float]:
        overall = aggregate(self.per_sample)
        overall["n_samples"] = self.n_samples
        return overall


def _judge_identity() -> tuple[str, str]:
    """(provider, model) of the judge that grades the answers.

    Reads config without constructing the provider — the runner only needs
    the identity for reporting, and constructing would force an early
    provider (and its API-key validation) even for unit tests injecting a
    fake judge.
    """
    llm = config.llm
    if llm.judge_provider:
        return llm.judge_provider, llm.judge_model
    return llm.provider, llm.model


async def run_calibration(
    samples: Sequence[CalibrationSample] | None = None,
    judge: JudgeCallable | None = None,
) -> CalibrationResult:
    """Judge every sample and compare the verdict with the human label.

    ``judge`` defaults to :func:`tests.eval.llm_judge.judge_answer` (the real
    LLM judge, running on the configured ``LLM_JUDGE_*`` provider); unit tests
    inject a fake matching the same signature.  A raised exception from the
    judge is recorded in ``judge_errors`` and excluded from the metrics.
    """
    sample_list = list(samples) if samples is not None else list(CALIBRATION_SAMPLES)
    judge_fn = judge or judge_answer
    rows: list[dict[str, Any]] = []
    judge_errors: list[dict[str, str]] = []
    for s in sample_list:
        try:
            verdict = await judge_fn(s.query, s.context, s.answer, s.required_facts)
        except Exception as exc:
            judge_errors.append({"id": s.id, "error": str(exc)})
            continue
        rows.append(compare_sample(s, verdict))
    provider, model = _judge_identity()
    return CalibrationResult(
        samples=sample_list,
        per_sample=rows,
        judge_errors=judge_errors,
        judge_provider=provider,
        judge_model=model,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Report ────────────────────────────────────────────────────────


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def to_json(result: CalibrationResult) -> str:
    """Serialize the calibration result to a pretty JSON string."""
    payload = {
        "schema_version": 1,
        "note": (
            "LLM-as-judge vs human verdict calibration (P0-3). grounded_agreement "
            "= fraction of judged samples where the judge's grounded flag matches "
            "the human label; coverage_precision/recall/f1 = P/R/F1 of the judge's "
            "covered_facts vs the human's, averaged across judged samples; "
            "false_negative = human grounded but judge ungrounded (the ans-006 "
            "class of error); false_positive = human ungrounded but judge grounded. "
            "Judge failures are recorded in judge_errors and excluded from every "
            "denominator."
        ),
        "generated_at": result.generated_at,
        "judge_provider": result.judge_provider,
        "judge_model": result.judge_model,
        "n_samples": result.n_samples,
        "n_judged": result.n_judged,
        "n_judge_errors": result.n_judge_errors,
        "overall": result.overall,
        "per_sample": result.per_sample,
        "judge_errors": result.judge_errors,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_markdown(result: CalibrationResult) -> str:
    """Render a Markdown report: summary + per-sample + judge errors."""
    o = result.overall
    now = datetime.fromisoformat(result.generated_at).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    n_grounded = sum(1 for s in result.samples if s.human_grounded)
    n_synonym = sum(1 for s in result.samples if "同义改写" in s.notes)
    lines: list[str] = [
        "# LLM Judge Calibration Report",
        "",
        f"- Generated: {now}",
        f"- Judge provider: {result.judge_provider} ({result.judge_model})",
        f"- Samples: {result.n_samples} (judged {result.n_judged}, "
        f"judge errors {result.n_judge_errors})",
        f"- Human verdicts: {n_grounded} grounded / "
        f"{result.n_samples - n_grounded} ungrounded "
        f"({n_synonym} synonym-rewrite)",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---|",
        f"| grounded_agreement | {_fmt(o['grounded_agreement'])} |",
        f"| coverage_precision | {_fmt(o['coverage_precision'])} |",
        f"| coverage_recall | {_fmt(o['coverage_recall'])} |",
        f"| coverage_f1 | {_fmt(o['coverage_f1'])} |",
        f"| false_negative (human grounded, judge ungrounded) | "
        f"{o['false_negative']:.0f} |",
        f"| false_positive (human ungrounded, judge grounded) | "
        f"{o['false_positive']:.0f} |",
        "",
        "> `false_negative` is the `ans-006` class of error: a grounded answer "
        "judged ungrounded — typically a faithful paraphrase penalized for not "
        "repeating the required-fact strings verbatim, despite the judge "
        "prompt's \"允许同义改写\".",
        "",
        "## Per-sample",
        "",
        "| id | base | human | judge | agree | human covered | judge covered "
        "| cov P/R/F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result.per_sample:
        human = "grounded" if r["human_grounded"] else "ungrounded"
        judge = "grounded" if r["judge_grounded"] else "ungrounded"
        agree = "✓" if r["grounded_agree"] else "✗"
        prf = (
            f"{_fmt(r['coverage_precision'])}/"
            f"{_fmt(r['coverage_recall'])}/{_fmt(r['coverage_f1'])}"
        )
        lines.append(
            f"| {r['id']} | {r['base_id']} | {human} | {judge} | {agree} | "
            f"{', '.join(r['human_covered_facts']) or '-'} | "
            f"{', '.join(r['judge_covered_facts']) or '-'} | {prf} |"
        )
    lines.append("")

    lines += [
        "<details><summary>Per-sample answers & judge forensics</summary>",
        "",
    ]
    for r in result.per_sample:
        lines.append(f"**{r['id']}** (base {r['base_id']})")
        lines.append(f"- answer: `{r['answer']}`")
        if r["ungrounded_claims"]:
            lines.append(f"- judge ungrounded claims: {r['ungrounded_claims']}")
        if not r["grounded_agree"]:
            lines.append(
                f"- ✗ disagreement: human says "
                f"{'grounded' if r['human_grounded'] else 'ungrounded'}, "
                f"judge says {'grounded' if r['judge_grounded'] else 'ungrounded'}"
            )
        lines.append("")
    lines.append("</details>")

    if result.judge_errors:
        lines += [
            "## Judge errors",
            "",
            "| id | error |",
            "|---|---|",
        ]
        for e in result.judge_errors:
            lines.append(f"| {e['id']} | `{e['error']}` |")
        lines.append("")
        lines.append(
            "> These samples are excluded from every metric denominator — a "
            "judge outage reads as not-measured, not as disagreement."
        )

    return "\n".join(lines)


def write_json(result: CalibrationResult, path: str) -> str:
    """Write JSON report to ``path``. Returns the path."""
    p = Path(path)
    p.write_text(to_json(result), encoding="utf-8")
    return str(p)


def write_markdown(result: CalibrationResult, path: str) -> str:
    """Write Markdown report to ``path``. Returns the path."""
    p = Path(path)
    p.write_text(to_markdown(result), encoding="utf-8")
    return str(p)


def summarize(result: CalibrationResult) -> str:
    """One-line summary for stdout / CI logs."""
    o = result.overall
    return (
        f"[judge_calibration] agreement={_fmt(o['grounded_agreement'])} "
        f"coverage_f1={_fmt(o['coverage_f1'])} "
        f"fn={o['false_negative']:.0f} fp={o['false_positive']:.0f} "
        f"judged={result.n_judged}/{result.n_samples} "
        f"judge_errors={result.n_judge_errors}"
    )


# ── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.experiments.judge_calibration",
        description="Quantify how closely the LLM answer judge tracks human "
        "verdicts over a small hand-labeled sample set.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit to this many samples (cheap smoke runs).",
    )
    p.add_argument(
        "--report-md",
        default=None,
        help="Write Markdown report to this path.",
    )
    p.add_argument(
        "--report-json",
        default=None,
        help="Write JSON report to this path.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the sample set; skip judge calls.",
    )
    return p


def _guard_judge_provider(args: argparse.Namespace) -> None:
    """Fail fast when the calibration would run on the wrong judge.

    ``judge_answer`` falls back to the primary provider when no ``LLM_JUDGE_*``
    config block is set, so without this guard the calibration would measure
    the *primary model* — self-judging — which defeats the whole point of
    calibrating an independent judge.  ``--validate-only`` is exempt.
    Mirrors ``run_llm_eval._guard_judge_provider``.
    """
    if args.validate_only:
        return
    missing: list[str] = []
    if not config.llm.judge_provider:
        missing.append("LLM_JUDGE_PROVIDER")
    else:
        if not config.llm.judge_model:
            missing.append("LLM_JUDGE_MODEL")
        if not config.llm.judge_api_key:
            missing.append("LLM_JUDGE_API_KEY")
        if config.llm.judge_provider != "anthropic" and not config.llm.judge_base_url:
            missing.append("LLM_JUDGE_BASE_URL")
    if missing:
        raise SystemExit(
            "Judge calibration needs a complete dedicated judge provider "
            f"(missing: {', '.join(missing)}).  Without LLM_JUDGE_* the "
            "verdicts come from the primary model (self-judging) and the "
            "calibration measures nothing.  Run with --validate-only to "
            "skip judge calls."
        )


def _judge_channel_degraded(result: CalibrationResult, *, ratio: float = 0.5) -> bool:
    """True when judge failures cover at least *ratio* of the samples.

    A judge channel that fails on the majority of samples is a broken run: the
    agreement numbers that remain are what survived an outage, not a sample of
    judge behaviour.  Treat that as a run failure, not a quality signal.
    """
    if not result.n_judge_errors:
        return False
    return result.n_judge_errors >= ratio * result.n_samples


async def _run(args: argparse.Namespace) -> int:
    warnings = validate_calibration_samples()
    if warnings:
        print("⚠ calibration sample warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("✓ calibration samples validated", file=sys.stderr)

    if args.validate_only:
        return 0

    _guard_judge_provider(args)

    samples = list(CALIBRATION_SAMPLES)
    if args.sample and args.sample > 0:
        samples = samples[: args.sample]

    provider, model = _judge_identity()
    print(
        f"Running judge calibration: {len(samples)} samples, judge provider "
        f"{provider} ({model})",
        file=sys.stderr,
    )
    result = await run_calibration(samples=samples)
    print(summarize(result), file=sys.stderr)

    if args.report_md:
        print(
            f"✓ Markdown report → {write_markdown(result, args.report_md)}",
            file=sys.stderr,
        )
    if args.report_json:
        print(
            f"✓ JSON report → {write_json(result, args.report_json)}",
            file=sys.stderr,
        )

    if _judge_channel_degraded(result):
        print(
            f"✗ judge channel failed on {result.n_judge_errors}/"
            f"{result.n_samples} samples — agreement numbers not trustworthy, "
            "run failed",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
