"""Analyze the ``llm_usage`` table by scenario: call count, tokens, latency, cost.

Answers the question "how much of the chat budget goes to rerank_llm (and
other scenarios)?"  Read-only — never writes to the DB.

Usage:
    python -m tests.perf.analyze_usage [--thread-prefix p95-]

Outputs two tables:
  1. All history, grouped by scenario (columns as in the schema).
  2. The same grouped by scenario, filtered to rows whose ``thread_id``
     starts with ``--thread-prefix`` (default ``p95-``: the turns written by
     ``measure_chat_p95.py``).
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv


async def _query(thread_prefix: str | None) -> list[tuple]:
    import asyncpg

    from backend.service.usage import estimate_cost

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        where = ""
        params: list = []
        if thread_prefix:
            where = "WHERE thread_id LIKE $1"
            params = [thread_prefix + "%"]
        rows = await conn.fetch(
            f"""
            SELECT scenario, provider, model,
                   count(*) AS n,
                   coalesce(sum(input_tokens),0) AS i,
                   coalesce(sum(output_tokens),0) AS o,
                   coalesce(sum(cache_read_tokens),0) AS cr,
                   coalesce(sum(cache_creation_tokens),0) AS cc,
                   coalesce(sum(total_tokens),0) AS t,
                   coalesce(sum(latency_ms),0) AS lat,
                   coalesce(avg(latency_ms),0)::int AS lat_avg
            FROM llm_usage
            {where}
            GROUP BY scenario, provider, model
            ORDER BY t DESC
            """,
            *params,
        )
    finally:
        await conn.close()

    out = []
    for r in rows:
        cost = estimate_cost(
            r["model"], r["i"], r["o"], r["cr"], r["cc"], provider=r["provider"]
        )
        out.append((r["scenario"], r["provider"], r["model"], r["n"], r["i"],
                    r["o"], r["cr"], r["cc"], r["t"], r["lat"], r["lat_avg"], cost))
    return out


def _print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("  (no rows)")
        return
    total_cost = sum(r[11] for r in rows)
    total_lat = sum(r[9] for r in rows)
    total_tok = sum(r[8] for r in rows)
    print(
        f"  {'scenario':<18} {'provider':<8} {'model':<26} {'n':>4} "
        f"{'tok(i/o/cr)':>22} {'tot_tok':>9} {'lat_ms':>9} {'lat_avg':>7} {'est$':>8}"
    )
    for sc, prov, model, n, i, o, cr, cc, t, lat, lat_avg, cost in rows:
        print(
            f"  {sc:<18} {prov:<8} {model:<26} {n:>4} "
            f"{f'{i}/{o}/{cr}':>22} {t:>9} {lat:>9} {lat_avg:>7} {cost:>8.5f}"
        )
    print(f"  {'─'*108}")
    print(
        f"  {'TOTAL':<18} {'':<8} {'':<26} {sum(r[3] for r in rows):>4} "
        f"{'':>22} {total_tok:>9} {total_lat:>9} {'':>7} {total_cost:>8.5f}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-prefix", default="p95-",
                        help="only count threads whose id starts with this (default p95-)")
    args = parser.parse_args()

    all_rows = await _query(None)
    _print_table("ALL history by scenario", all_rows)

    thread_rows = await _query(args.thread_prefix)
    _print_table(f"thread_id LIKE '{args.thread_prefix}%' by scenario", thread_rows)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
