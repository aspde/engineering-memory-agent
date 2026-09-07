"""One-command model recalibration — the model-swap runbook, mechanically.

The runbook (docs/engineering/llm-eval.md steps 2+3) is a paper checklist:
rerun the channel-sensitive suites, derive floors, confirm, update
`floors.json` + the price table, commit.  This script turns that checklist
into one command that drives the parts that are mechanical and PAUSES at the
one part that must stay human — confirming the suggested floors.

What stays human (by design, see runbook step 3): accepting the numbers.
A gate that can lower its own thresholds is not a gate, so this script
prints the derived floors and the channel they were measured on, then waits
for an explicit `--confirm` before writing anything.

Usage:
    # Run the channel-sensitive suites 3x, derive floors, confirm, write.
    python -m evals.recalibrate --model omen-alpha --provider openai \
        --suite tool_selection,extraction,answer --n-runs 3

    # Same but from existing reports (skip the eval runs).
    python -m evals.recalibrate --model omen-alpha \
        --reports run1.json,run2.json,run3.json --confirm

The `--model`/`--provider` pin the `calibrated_channel` block written to
floors.json; `--from-run` records the CI run (or local) that produced the
numbers, so a future reader knows *when* this calibration was earned.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from evals.multi_run_gate import (
    FLOORS_FILE,
    _run_eval_runs,
    check_channel_match,
    derive_floors,
    load_report,
    report_channel,
)

# The write-path floors and the two "always 1.000" capability metrics keep
# their values through a model swap UNLESS the new model measurably drops on
# them (runbook step 3).  This is the layered-recalibration premise.
_CHANNEL_SENSITIVE_DEFAULT = "tool_selection,extraction,answer"

# Price-table fallback note — recalibrate appends/applies the model's name so
# the cost dashboard isn't silently estimating, but the actual per-token
# prices need a human with the provider's pricing page.  The script surfaces
# this as a TODO, not because it can know the price.
_PRICE_TABLE_NOTE = (
    "Add `{model}` to backend/service/usage.py `_PRICE_RULES` with its "
    "provider pricing (input/output per 1M tokens) — without it the cost "
    "dashboard silently falls back to the default estimate."
)


def _derive_and_print(suggested: dict[str, dict[str, float]], channels: list[tuple[str, str]]) -> None:
    print("-- Suggested floors (ci95_lower - 0.05, rounded down) --")
    for suite in sorted(suggested):
        for metric in sorted(suggested[suite]):
            print(f"  {suite}/{metric}: {suggested[suite][metric]:.2f}")
    channel_str = ", ".join(f"{p}/{m}" for p, m in channels) or "unknown (no provenance)"
    print(f"\nMeasured on channel(s): {channel_str}")
    print("Confirm these numbers with `--confirm`, then the script updates")
    print("evals/floors.json (calibrated_channel + thresholds) and prints the")
    print("git + price-table steps.  Human judgement is part of the flow —")


def _write_floors(
    model: str,
    provider: str,
    from_run: str | None,
    suggested: dict[str, dict[str, float]],
    existing: dict[str, Any],
) -> None:
    existing["calibrated_channel"] = {
        "provider": provider,
        "model": model,
        "calibrated_from": from_run or "local run",
    }
    existing["thresholds"] = {metric: floor
                              for suite in suggested
                              for metric, floor in suggested[suite].items()}
    # Preserve notes from the prior file where keys still apply; drop stale ones
    # is a human call, so keep only keys present in the new thresholds plus the
    # free-form comment block.
    notes = existing.get("notes", {})
    existing.pop("notes", None)
    # Rewrite the file content: comment (stale warning) + channels + thresholds.
    content: dict[str, Any] = {"comment": existing.get("comment", "")}
    content["calibrated_channel"] = existing["calibrated_channel"]
    content["thresholds"] = existing["thresholds"]
    content["notes"] = notes
    with open(FLOORS_FILE, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evals.recalibrate",
        description=__doc__,
    )
    p.add_argument("--model", required=True, help="New gate-channel model name (e.g. omen-alpha).")
    p.add_argument("--provider", default="openai", help="Provider of the new channel.")
    p.add_argument("--from-run", default=None, help="CI run id (or note) that produced the numbers.")
    p.add_argument("--suite", default=_CHANNEL_SENSITIVE_DEFAULT,
                   help="Suites to rerun for calibration.  Default: the channel-sensitive set.")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--reports", default=None,
                   help="Comma-separated existing report JSONs (skip reruns). Mutually exclusive with --n-runs.")
    p.add_argument("--confirm", action="store_true",
                   help="Actually write floors.json.  Without it, the script only derives + prints.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.reports is not None:
        paths = [p.strip() for p in args.reports.split(",") if p.strip()]
        if not paths:
            print("--reports: no non-empty paths given", file=sys.stderr)
            return 2
        reports = [load_report(p) for p in paths]
    else:
        import argparse as _arg
        gate_args = _arg.Namespace(
            n_runs=args.n_runs, suite=args.suite, judge="deterministic",
        )
        runs = _run_eval_runs(gate_args, None)
        reports = [r for _, r in runs]

    suggested = derive_floors(reports)
    channels = sorted({c for c in (report_channel(r) for r in reports) if c})
    _derive_and_print(suggested, channels)

    if not args.confirm:
        print("\n(No changes written — re-run with --confirm to apply.)")
        return 0

    existing = {}
    if FLOORS_FILE.exists():
        existing = json.loads(FLOORS_FILE.read_text(encoding="utf-8"))

    # Verify the run channel matches the floors channel before writing —
    # recalibrating TO a channel while the reports are from a different one
    # would reintroduce the mismatch this whole system exists to prevent.
    mismatch = check_channel_match(
        {"calibrated_channel": {"provider": args.provider, "model": args.model}}, reports
    )
    if mismatch:
        print(f"X aborting: {mismatch.split(' but runs measured on ')[-1]}",
              file=sys.stderr)
        return 2

    _write_floors(args.model, args.provider, args.from_run, suggested, existing)
    print(f"\nOK  Wrote evals/floors.json (calibrated_channel={args.provider}/{args.model})")
    print(_PRICE_TABLE_NOTE.format(model=args.model))
    print("Next: 1) commit floors.json  2) set LLM_MODEL secret to the new model  3) trigger a run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
