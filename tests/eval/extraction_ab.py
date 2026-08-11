"""Extraction A/B — few-shot + function-calling vs the JSON-schema channel.

The three-stage extraction went through two optimizations:
  1. **few-shot examples** in the entity/relation prompts (v2 → v3);
  2. a dedicated **function-calling channel** — enum/required constraints hold
     at generation time — preferred on OpenAI-compatible providers, falling
     back to ``chat_structured`` (JSON schema + retry).

This script measures each channel on the 8-item extraction suite so a
prompt/model edit can be compared against the committed baseline
(``tests/eval/reports/llm-eval-baseline.json`` — zero-shot + json, 2026-08-09,
same provider):

  - **zero-shot + json**  = committed baseline (read from the JSON, not re-run)
  - **few-shot + json**   = current code with the tool channel disabled
  - **few-shot + func**   = current production code (tool channel preferred)

Judge is deterministic (substring/normalized matching) — no LLM judge, so
the numbers are reproducible and cheap (~24 LLM calls per arm).

Usage:
    python -m tests.eval.extraction_ab --report-md tests/eval/reports/extraction_ab_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from tests.eval.llm_ground_truth import load_extraction_items
from tests.eval.llm_runner import run_extraction

_BASELINE_PATH = Path(__file__).resolve().parent.parent.parent / "tests" / "eval" / "reports" / "llm-eval-baseline.json"

EXTRACTION_METRICS = (
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "entity_type_accuracy",
    "relation_precision",
    "relation_recall",
    "relation_f1",
    "summary_coverage",
)


def _load_baseline() -> dict[str, float]:
    """Read the committed zero-shot+json extraction numbers."""
    data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data["overall"]["extraction"].items()}


async def _run_arm(*, use_function_calling: bool) -> dict[str, float]:
    """Run the extraction suite with the tool channel on/off."""
    import backend.service.extraction as ext_mod

    items = load_extraction_items()
    if not use_function_calling:
        original = ext_mod._is_openai_compatible
        ext_mod._is_openai_compatible = lambda llm: False
    try:
        result = await run_extraction(items, judge="deterministic")
    finally:
        if not use_function_calling:
            ext_mod._is_openai_compatible = original
    return {k: float(result.overall.get(k, 0.0)) for k in EXTRACTION_METRICS}


def _write_report(baseline: dict, json_arm: dict, func_arm: dict, path: str) -> None:
    lines = [
        "# Extraction A/B — few-shot + function calling",
        "",
        "> 三阶段提取的两项优化：entity/relation prompt 加 few-shot examples（v2→v3），"
        "以及 OpenAI 兼容 provider 上的函数调用通道（enum 在生成期约束，失败降级到 "
        "`chat_structured`）。8 条标注 query，deterministic judge（纯子串/归一化匹配，无 LLM 裁判）。",
        "",
        "## 结果",
        "",
        "| 指标 | zero-shot + json（2026-08-09 基线） | few-shot + json | few-shot + 函数调用 |",
        "|------|------|------|------|",
    ]
    for k in EXTRACTION_METRICS:
        lines.append(
            f"| {k} | {baseline.get(k, float('nan')):.3f} | {json_arm.get(k, float('nan')):.3f} | "
            f"{func_arm.get(k, float('nan')):.3f} |"
        )
    lines += [
        "",
        "## 解读",
        "",
        "（跑完回填：few-shot 与函数调用各自提升了什么、relation_f1 是否从基线大幅改善。）",
        "",
        "## 边界",
        "",
        "- 基线来自 2026-08-09 committed 快照（同 provider deepseek-v4-flash），日期不同，模型可能漂移。",
        "- 8 条标注集较小，单指标 ±0.1 以内波动可能只是采样噪声；趋势比绝对值更有意义。",
        "- `few-shot + 函数调用`是当前生产代码；`few-shot + json`通过禁用工具通道获得。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.extraction_ab",
        description="A/B few-shot + function calling vs the JSON-schema channel on extraction.",
    )
    p.add_argument("--report-md", default=None, help="Write a Markdown report to this path.")
    return p


async def _main(report_md: str | None) -> int:
    baseline = _load_baseline()
    print("Running extraction suite: few-shot + json (tool channel disabled)…")
    json_arm = await _run_arm(use_function_calling=False)
    print("Running extraction suite: few-shot + function calling (current code)…")
    func_arm = await _run_arm(use_function_calling=True)

    print("\n=== Extraction A/B (8 items, deterministic judge) ===")
    print(f"  {'metric':<22} {'zero+json':>10} {'fs+json':>10} {'fs+func':>10}")
    for k in EXTRACTION_METRICS:
        print(
            f"  {k:<22} {baseline.get(k, float('nan')):>10.3f} "
            f"{json_arm.get(k, float('nan')):>10.3f} {func_arm.get(k, float('nan')):>10.3f}"
        )

    if report_md:
        _write_report(baseline, json_arm, func_arm, report_md)
        print(f"\n✓ Markdown report → {report_md}")
    return 0


def main() -> None:
    # Windows GBK consoles can't encode the ✓/→ chars this script prints —
    # reconfigure stdout so a console-encoding error can't kill the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args()
    rc = asyncio.run(_main(args.report_md))
    sys.exit(rc)


if __name__ == "__main__":
    main()
