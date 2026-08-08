"""Eval quality gates — assert overall metrics against minimum thresholds.

The weekly eval (``.github/workflows/eval.yml``) measures retrieval quality
but, until this module, never *failed* on regressions — a quality drop just
produced a lower number in the report.  :func:`check_thresholds` turns the
raw aggregate metrics into a pass/fail decision so CI can gate on them.

The function is pure and I/O-free: it takes the ``overall`` metric dict of an
``EvalResult`` (see ``tests.eval.runner``) and a ``{metric: minimum}`` map,
and returns which metrics fell short.  This keeps it trivially unit-testable
with fabricated data and independent of the (heavy) retrieval run.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThresholdResult:
    """Outcome of checking one result's metrics against its thresholds.

    ``passed`` is True only when *every* threshold is met.  ``failures``
    lists the shortfalls, each ``{"metric": ..., "min": ..., "actual": ...}``
    so the caller can print a precise "which metric, by how much" report.
    """

    passed: bool
    failures: list[dict[str, Any]] = field(default_factory=list)


def check_thresholds(
    overall: dict[str, float],
    thresholds: dict[str, float],
) -> ThresholdResult:
    """Check ``overall`` metrics against ``thresholds``.

    Args:
        overall: an ``EvalResult.overall`` mapping (metric key → mean score),
            e.g. ``{"recall@5": 0.96, "mrr": 0.91, "ndcg@5": 0.93, ...}``.
        thresholds: metric keys → minimum acceptable value, e.g.
            ``{"recall@5": 0.95, "mrr": 0.90}``.  Empty means "no gate" —
            always passes.

    Returns:
        ``ThresholdResult``.  A metric that is requested in ``thresholds`` but
        absent from ``overall`` is treated as 0.0 and therefore a failure —
        the CLI cannot gate on a metric it cannot read.  An empty
        ``thresholds`` always passes, preserving the no-gate default.
    """
    failures: list[dict[str, Any]] = []
    for metric, minimum in thresholds.items():
        actual = float(overall.get(metric, 0.0))
        if actual < minimum:
            failures.append(
                {
                    "metric": metric,
                    "min": minimum,
                    "actual": actual,
                }
            )
    return ThresholdResult(passed=not failures, failures=failures)


def print_threshold_failures(result: ThresholdResult) -> None:
    """Print each shortfall as a clear, one-line report (to stderr).

    Intended for CI logs: the output names the metric, the actual score, and
    the required minimum so a failing run says exactly why it failed.
    """
    if result.passed:
        return
    for f in result.failures:
        print(
            f"  ✗ {f['metric']}: {f['actual']:.3f} < required {f['min']:.3f}",
            file=sys.stderr,
        )
