"""Unit tests for tests/eval/judge_calibration.py — metrics + runner.

The real LLM judge is never called: ``run_calibration`` accepts an injected
``judge`` callable (matching ``llm_judge.judge_answer``'s signature), so all
aggregation logic is exercised with fake verdicts.  Covers:
    - coverage P/R/F1 (both-empty is perfect agreement; one-sided empties)
    - per-sample human-vs-judge classification (agree / false-negative /
      false-positive)
    - aggregate grounded_agreement / coverage means / fp / fn counts
    - judge failures recorded in judge_errors and excluded from the
      denominator
    - sample-set validation (covered ⊆ required_facts; [UNGROUNDED:] marker
      consistency with human_grounded)
"""

from __future__ import annotations

import pytest

import tests.eval.experiments.judge_calibration as cal_mod
from tests.eval.experiments.judge_calibration import (
    aggregate,
    compare_sample,
    coverage_prf,
    run_calibration,
)
from tests.eval.experiments.judge_calibration_samples import (
    CALIBRATION_SAMPLES,
    CalibrationSample,
    validate_calibration_samples,
)


# ── Fixtures ──────────────────────────────────────────────────────


def _sample(
    sid: str,
    *,
    grounded: bool = True,
    covered: list[str] | None = None,
    answer: str = "some answer",
) -> CalibrationSample:
    return CalibrationSample(
        id=sid,
        base_id="ans-001",
        query="选型是什么？",
        context="选用了 pgvector 而非 Elasticsearch。",
        required_facts=["pgvector", "Elasticsearch", "cosine"],
        answer=answer,
        human_grounded=grounded,
        human_covered_facts=list(covered or []),
        notes="",
    )


# ── coverage_prf ──────────────────────────────────────────────────


class TestCoveragePrf:
    def test_both_empty_is_perfect_agreement(self) -> None:
        # Unlike the extraction metrics, "both sides said nothing is covered"
        # is agreement, not a 0.0.
        assert coverage_prf([], []) == (1.0, 1.0, 1.0)

    def test_judge_missed_everything(self) -> None:
        # predicted empty, human non-empty: no false positives but 0 recall.
        assert coverage_prf([], ["pgvector", "cosine"]) == (1.0, 0.0, 0.0)

    def test_judge_credited_unreal_facts(self) -> None:
        # predicted non-empty, human empty: every credited fact is a false
        # positive, recall vacuous.
        assert coverage_prf(["pgvector"], []) == (0.0, 1.0, 0.0)

    def test_partial_overlap(self) -> None:
        # predicted [pgvector, cosine], human [pgvector, Elasticsearch]:
        # tp=1, fp=1, fn=1 → P=R=F1=0.5.
        p, r, f = coverage_prf(["pgvector", "cosine"], ["pgvector", "Elasticsearch"])
        assert (p, r, f) == (0.5, 0.5, 0.5)

    def test_perfect_overlap(self) -> None:
        assert coverage_prf(["pgvector"], ["pgvector"]) == (1.0, 1.0, 1.0)

    def test_synonym_not_credited_hurts_recall(self) -> None:
        # The judge fails to credit a paraphrased fact → recall drops while
        # precision stays 1.0.  This is the verbatim-only blind spot.
        p, r, f = coverage_prf(["pgvector"], ["pgvector", "cosine"])
        assert p == 1.0
        assert r == 0.5
        assert f == pytest.approx(2 / 3)


# ── compare_sample ────────────────────────────────────────────────


class TestCompareSample:
    def test_agreement_row(self) -> None:
        s = _sample("s1", grounded=True, covered=["pgvector"])
        row = compare_sample(
            s, {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []}
        )
        assert row["grounded_agree"] is True
        assert row["false_negative"] is False
        assert row["false_positive"] is False
        assert row["coverage_f1"] == 1.0

    def test_false_negative_row(self) -> None:
        # ans-006 class: human grounded, judge ungrounded.
        s = _sample("s2", grounded=True, covered=["pgvector"])
        row = compare_sample(
            s, {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]}
        )
        assert row["grounded_agree"] is False
        assert row["false_negative"] is True
        assert row["false_positive"] is False
        assert row["judge_grounded"] is False
        assert row["human_grounded"] is True

    def test_false_positive_row(self) -> None:
        # human ungrounded, judge grounded.
        s = _sample("s3", grounded=False, covered=[])
        row = compare_sample(
            s, {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []}
        )
        assert row["grounded_agree"] is False
        assert row["false_negative"] is False
        assert row["false_positive"] is True

    def test_carries_answer_and_claims_for_forensics(self) -> None:
        s = _sample("s4", grounded=True, covered=[], answer="the answer text")
        row = compare_sample(
            s, {"covered_facts": [], "grounded": True, "ungrounded_claims": ["fabricated"]}
        )
        assert row["answer"] == "the answer text"
        assert row["ungrounded_claims"] == ["fabricated"]


# ── aggregate ─────────────────────────────────────────────────────


class TestAggregate:
    def _rows(self) -> list[dict]:
        # Two agrees (s1, s3) + one false negative (s2) + one false positive
        # (s4).  Coverage: s1 → 1.0, s2 → 0.0 (missed all), s3 → 1.0 (both
        # empty), s4 → 0.0 (credited unreal facts).
        return [
            compare_sample(
                _sample("s1", grounded=True, covered=["pgvector"]),
                {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []},
            ),
            compare_sample(
                _sample("s2", grounded=True, covered=["pgvector"]),
                {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]},
            ),
            compare_sample(
                _sample("s3", grounded=False, covered=[]),
                {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]},
            ),
            compare_sample(
                _sample("s4", grounded=False, covered=[]),
                {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []},
            ),
        ]

    def test_counts_and_means(self) -> None:
        o = aggregate(self._rows())
        assert o["n_judged"] == 4
        assert o["grounded_agreement"] == pytest.approx(0.5)
        assert o["false_negative"] == 1.0
        assert o["false_positive"] == 1.0
        assert o["coverage_precision"] == pytest.approx(0.75)
        assert o["coverage_recall"] == pytest.approx(0.75)
        assert o["coverage_f1"] == pytest.approx(0.5)

    def test_empty_rows_are_zeros(self) -> None:
        o = aggregate([])
        assert o["n_judged"] == 0
        assert o["grounded_agreement"] == 0.0
        assert o["coverage_f1"] == 0.0
        assert o["false_negative"] == 0.0
        assert o["false_positive"] == 0.0

    def test_all_agree_is_1_0(self) -> None:
        rows = [
            compare_sample(
                _sample("s1", grounded=True, covered=["pgvector"]),
                {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []},
            ),
            compare_sample(
                _sample("s2", grounded=False, covered=[]),
                {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]},
            ),
        ]
        o = aggregate(rows)
        assert o["grounded_agreement"] == 1.0
        assert o["false_negative"] == 0.0
        assert o["false_positive"] == 0.0
        assert o["coverage_f1"] == 1.0


# ── run_calibration (injected fake judge) ─────────────────────────


class TestRunCalibration:
    @pytest.mark.asyncio
    async def test_uses_injected_judge_and_aggregates(self) -> None:
        # The fake judge reads the sample id from the start of the answer.
        # s1 agree-grounded, s2 agree-ungrounded, s3 disagreement (FN).
        async def _fake(query, context, answer, required_facts):
            sid = answer.split("|", 1)[0]
            if sid == "s1":
                return {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []}
            if sid == "s2":
                return {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]}
            return {"covered_facts": [], "grounded": False, "ungrounded_claims": ["x"]}

        samples = [
            _sample("s1", grounded=True, covered=["pgvector"], answer="s1|..."),
            _sample("s2", grounded=False, covered=[], answer="s2|..."),
            _sample("s3", grounded=True, covered=["pgvector"], answer="s3|..."),
        ]
        result = await run_calibration(samples=samples, judge=_fake)
        assert result.n_samples == 3
        assert result.n_judged == 3
        assert result.n_judge_errors == 0
        assert result.overall["grounded_agreement"] == pytest.approx(2 / 3)
        assert result.overall["false_negative"] == 1.0
        assert result.overall["false_positive"] == 0.0
        # s3 was the false negative (judge ungrounded the grounded answer).
        assert result.per_sample[2]["id"] == "s3"
        assert result.per_sample[2]["false_negative"] is True

    @pytest.mark.asyncio
    async def test_judge_failure_excluded_from_denominator(self) -> None:
        async def _fake(query, context, answer, required_facts):
            if answer.startswith("boom"):
                raise RuntimeError("judge timeout")
            return {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []}

        samples = [
            _sample("s1", grounded=True, covered=["pgvector"], answer="ok|..."),
            _sample("s2", grounded=True, covered=["pgvector"], answer="boom|..."),
            _sample("s3", grounded=True, covered=["pgvector"], answer="ok|..."),
        ]
        result = await run_calibration(samples=samples, judge=_fake)
        assert result.n_judge_errors == 1
        assert result.judge_errors == [{"id": "s2", "error": "judge timeout"}]
        # The failed sample is NOT in per_sample and does not move the
        # denominator: judged 2/3, both agree → agreement 1.0 (not 2/3).
        assert result.n_judged == 2
        assert result.overall["n_judged"] == 2.0
        assert result.overall["grounded_agreement"] == 1.0
        assert result.overall["false_negative"] == 0.0
        assert [r["id"] for r in result.per_sample] == ["s1", "s3"]

    @pytest.mark.asyncio
    async def test_all_judges_failed_reads_as_not_measured(self) -> None:
        async def _boom(query, context, answer, required_facts):
            raise RuntimeError("provider down")

        result = await run_calibration(
            samples=[_sample("s1", answer="a"), _sample("s2", answer="b")],
            judge=_boom,
        )
        assert result.n_judged == 0
        assert result.n_judge_errors == 2
        assert result.overall["grounded_agreement"] == 0.0

    @pytest.mark.asyncio
    async def test_default_judge_is_real_judge_answer(self, monkeypatch) -> None:
        """Without an injected judge, run_calibration calls judge_answer."""
        seen: list = []
        calls: dict[str, int] = {"n": 0}

        async def _fake_judge_answer(query, context, answer, required_facts):
            calls["n"] += 1
            seen.append((query, context, answer, required_facts))
            return {"covered_facts": [], "grounded": True, "ungrounded_claims": []}

        monkeypatch.setattr(cal_mod, "judge_answer", _fake_judge_answer)
        await run_calibration(samples=[_sample("s1")])
        assert calls["n"] == 1
        # The sample's content is forwarded to the real judge signature.
        q, ctx, ans, facts = seen[0]
        assert q == "选型是什么？"
        assert facts == ["pgvector", "Elasticsearch", "cosine"]


# ── sample-set validation ─────────────────────────────────────────


class TestValidateSamples:
    def test_builtin_set_is_valid(self) -> None:
        # The checked-in sample set must always pass its own validation.
        assert validate_calibration_samples(CALIBRATION_SAMPLES) == []

    def test_covered_must_be_subset_of_required(self) -> None:
        bad = _sample("bad", covered=["Qdrant"])
        with pytest.raises(ValueError, match="not in required_facts"):
            validate_calibration_samples([bad])

    def test_grounded_label_rejects_ungrounded_marker(self) -> None:
        bad = _sample("bad", grounded=True, answer="用了 Qdrant。[UNGROUNDED: x]")
        with pytest.raises(ValueError, match=r"\[UNGROUNDED:\] marker"):
            validate_calibration_samples([bad])

    def test_ungrounded_label_requires_marker(self) -> None:
        bad = _sample("bad", grounded=False, covered=[], answer="用了 Qdrant。")
        with pytest.raises(ValueError, match=r"no \[UNGROUNDED:\] marker"):
            validate_calibration_samples([bad])

    def test_duplicate_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate calibration sample id"):
            validate_calibration_samples([_sample("dup"), _sample("dup")])

    def test_empty_answer_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty answer"):
            validate_calibration_samples([_sample("bad", answer="")])


# ── CLI (validate-only path) ──────────────────────────────────────


class TestCliValidateOnly:
    @pytest.mark.asyncio
    async def test_validate_only_never_calls_the_judge(self, monkeypatch, capsys) -> None:
        """--validate-only validates and returns 0 without any judge call."""
        called: dict[str, bool] = {"ran": False}

        async def _fake_run(**kw):
            called["ran"] = True
            raise AssertionError("judge run must not happen under --validate-only")

        monkeypatch.setattr(cal_mod, "run_calibration", _fake_run)
        args = cal_mod._build_parser().parse_args(["--validate-only"])
        rc = await cal_mod._run(args)
        assert rc == 0
        assert called["ran"] is False
        assert "calibration samples validated" in capsys.readouterr().err

    def test_parser_accepts_report_and_sample_flags(self) -> None:
        args = cal_mod._build_parser().parse_args(
            ["--sample", "3", "--report-md", "a.md", "--report-json", "a.json"]
        )
        assert args.sample == 3
        assert args.report_md == "a.md"
        assert args.report_json == "a.json"
        assert args.validate_only is False
