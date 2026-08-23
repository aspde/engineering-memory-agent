"""One-off / archival research scripts for the EMA eval.

These are the experiments that *produced* the numbers now baked into the
production constants and reports — decay A/B, extraction A/B, similarity
threshold calibration, hard-negative discrimination, scale probing,
judge self-calibration, fingerprint review.  They are kept for reproducibility
and re-running on new data, but they are **not** part of the eval's main path
and do not run in CI:

- Retrieval eval gate: ``evals.run_eval`` (weekly, ``eval.yml``)
- LLM behavior eval: ``evals.run_llm_eval`` + the multi-run gate
  ``evals.multi_run_gate``
- Task-level e2e: ``evals.run_task_eval``

Move a script here when it was a one-off measurement, not a standing gate.
"""
