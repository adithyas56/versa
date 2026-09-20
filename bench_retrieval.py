"""HNSW retrieval benchmark — a seeded 100k-row `interactions` table,
checked against TWO hard requirements, not one:

  1. p95 stage2_recall latency < 50ms
  2. stage2_recall returns >= 45 of 50 requested candidates under a
     realistic filter

A fast answer with too few candidates is a FAIL, not a pass with an
asterisk — a filtered HNSW scan can silently under-return (the graph
traversal has no guarantee of visiting enough candidates that also
satisfy an arbitrary WHERE predicate), and requirement 2 exists
specifically to catch that failure mode, which requirement 1 alone
would never surface.

Also runs EXPLAIN ANALYZE on a personal retrieval query and prints the
plan for a human to confirm partition pruning + an index scan, rather
than asserting on the plan text — the query plan is the evidence a
person reads, not something this script should silently pass/fail on
by string-matching a notoriously varied EXPLAIN format.

Usage:
    uv run python bench_retrieval.py --seed        # build the 100k rows (once)
    uv run python bench_retrieval.py --bench        # run the two requirements
    uv run python bench_retrieval.py --explain       # print EXPLAIN ANALYZE
    uv run python bench_retrieval.py --all           # all three, in order
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
from uuid import UUID, uuid4

import asyncpg
from dotenv import load_dotenv

from probe.db import create_pool
from probe.models import EntryState
from probe.retrieval import RetrievalContext, stage1_filter, stage2_recall
from probe.retrieval_config import RetrievalConfig

DIM = 768
N_CLUSTERS = 200  # embedding-distribution variety only -- no DB table anymore
N_LEARNERS = 800
N_ROWS = 100_000

# No topic_id predicate exists any more (topic-based clustering was
# removed -- see interactions.py's module docstring and migration
# 034's header). The realistic filter scenario requirement 2 checks is
# now entry_state, the one other selective predicate stage1_filter
# still exposes: one specific learner, entry_state = CONTINUING, with
# generously more than 50 matching rows seeded so a full 50 is
# actually achievable.
_HEAVY_ENTRY_STATE = "continuing"
_HEAVY_LEARNER_ROWS = 300


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL") or os.getenv(
        "PROBE_TEST_DATABASE_URL", "postgresql://probe:probe@localhost:5434/probe"
    )
    return url


def _random_unit_vector(rng: random.Random, dim: int = DIM) -> list[float]:
    v = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _clustered_vector(rng: random.Random, centroid: list[float], noise: float = 0.15) -> list[float]:
    v = [c + rng.gauss(0, noise) for c in centroid]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


async def seed(pool: asyncpg.Pool) -> None:
    rng = random.Random(42)
    print(f"Seeding {N_ROWS} interactions across {N_LEARNERS} learners...")

    async with pool.acquire() as conn:
        print("  clearing any prior benchmark data...")
        await conn.execute("TRUNCATE interactions, sessions, learners RESTART IDENTITY CASCADE")

        print("  creating learners...")
        learner_ids = [uuid4() for _ in range(N_LEARNERS)]
        await conn.copy_records_to_table(
            "learners", records=[(lid, f"bench-learner-{i}") for i, lid in enumerate(learner_ids)],
            columns=["id", "label"],
        )

        print("  creating one session per learner...")
        session_ids = [uuid4() for _ in learner_ids]
        await conn.copy_records_to_table(
            "sessions", records=list(zip(session_ids, learner_ids, strict=True)),
            columns=["id", "learner_id"],
        )

    # Cluster centroids exist only to give the seeded embeddings
    # realistic variety (some rows near each other, most not) -- there
    # is no DB table for this any more (topic clustering was removed;
    # see interactions.py's module docstring and migration 034's
    # header). Purely an in-script generation detail.
    cluster_centroids = [_random_unit_vector(rng) for _ in range(N_CLUSTERS)]

    # The "realistic filter" scenario: one learner, entry_state =
    # CONTINUING, heavily populated -- seeded FIRST so it's guaranteed
    # to exist regardless of the random distribution below.
    heavy_learner_id = learner_ids[0]
    heavy_session_id = session_ids[0]

    print(f"  building {N_ROWS} interaction rows in batches...")
    entry_states = ["cold_open", "continuing", "returning_after_gap", "stuck_repeat", "resolution"]
    help_levels = ["none", "hint", "worked_example", "direct_answer"]
    question_authors = ["learner", "system_option"]

    batch_size = 5000
    rows_built = 0
    async with pool.acquire() as conn:
        while rows_built < N_ROWS:
            batch = []
            for _ in range(min(batch_size, N_ROWS - rows_built)):
                cluster_idx = rng.randrange(N_CLUSTERS)
                embedding = _clustered_vector(rng, cluster_centroids[cluster_idx])
                if rows_built < _HEAVY_LEARNER_ROWS:
                    learner_id = heavy_learner_id
                    session_id = heavy_session_id
                    entry_state = _HEAVY_ENTRY_STATE
                else:
                    idx = rng.randrange(N_LEARNERS)
                    learner_id = learner_ids[idx]
                    session_id = session_ids[idx]
                    entry_state = rng.choice(entry_states)

                batch.append(
                    (
                        uuid4(), learner_id, session_id, rows_built,
                        f"synthetic question {rows_built}",
                        rng.choice(question_authors), None, rng.random() < 0.3,
                        f"synthetic response {rows_built}" if rng.random() < 0.8 else None,
                        entry_state, rng.randrange(0, 20), None, None,
                        "unknown", rng.choice(help_levels), None, embedding, None, None,
                    )
                )
                rows_built += 1

            await conn.copy_records_to_table(
                "interactions",
                records=batch,
                columns=[
                    "id", "learner_id", "session_id", "turn_number", "question_text",
                    "question_author", "originating_question", "did_branch",
                    "response_text", "entry_state", "recent_similar_count",
                    "prev_question_sim", "last_similar_turn_gap",
                    "prior_turn_outcome", "help_level", "elapsed_ms", "question_embedding",
                    "abstract_form", "abstract_embedding",
                ],
            )
            print(f"    {rows_built}/{N_ROWS}")

    print("Seeding complete.")
    print(f"  heavy_learner_id={heavy_learner_id}")
    print(f"  heavy entry_state={_HEAVY_ENTRY_STATE}")


async def bench(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        heavy_learner_id = await conn.fetchval(
            "SELECT learner_id FROM interactions WHERE entry_state = $1 "
            "GROUP BY learner_id ORDER BY count(*) DESC LIMIT 1",
            _HEAVY_ENTRY_STATE,
        )
        matching_count = await conn.fetchval(
            "SELECT count(*) FROM interactions WHERE learner_id = $1 AND entry_state = $2",
            heavy_learner_id, _HEAVY_ENTRY_STATE,
        )
        sample_vec = await conn.fetchval(
            "SELECT question_embedding FROM interactions "
            "WHERE learner_id = $1 AND entry_state = $2 LIMIT 1",
            heavy_learner_id, _HEAVY_ENTRY_STATE,
        )

    print(f"Realistic filter: learner={heavy_learner_id} entry_state={_HEAVY_ENTRY_STATE} "
          f"({matching_count} matching rows available)")
    query_vec = sample_vec.to_list()

    config = RetrievalConfig()
    ctx = RetrievalContext(entry_states=[EntryState.CONTINUING])
    where = stage1_filter(heavy_learner_id, ctx)

    # Requirement 1: p95 latency across N repeated queries.
    N_TRIALS = 50
    latencies = []
    recall_counts = []
    for _ in range(N_TRIALS):
        start = time.monotonic()
        hits = await stage2_recall(pool, query_vec, where, config)
        latencies.append((time.monotonic() - start) * 1000)
        question_hits = [h for h in hits if h.key == "question"]
        recall_counts.append(len(question_hits))

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    mean_recall = statistics.mean(recall_counts)
    min_recall = min(recall_counts)

    print(f"\n{'=' * 60}\nREQUIREMENT 1: p95 latency < 50ms\n{'=' * 60}")
    print(f"  p50={p50:.2f}ms  p95={p95:.2f}ms  max={max(latencies):.2f}ms")
    req1_pass = p95 < 50.0
    print(f"  RESULT: {'PASS' if req1_pass else 'FAIL'}")

    print(f"\n{'=' * 60}\nREQUIREMENT 2: stage2 returns >= 45 of 50 requested\n{'=' * 60}")
    print(f"  requested=50  mean_returned={mean_recall:.1f}  min_returned={min_recall}")
    req2_pass = min_recall >= 45
    print(f"  RESULT: {'PASS' if req2_pass else 'FAIL'}")

    print(f"\n{'=' * 60}\nOVERALL: {'PASS' if (req1_pass and req2_pass) else 'FAIL'}\n{'=' * 60}")


async def explain(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        heavy_learner_id = await conn.fetchval(
            "SELECT learner_id FROM interactions GROUP BY learner_id "
            "ORDER BY count(*) DESC LIMIT 1"
        )
        sample_vec = await conn.fetchval(
            "SELECT question_embedding FROM interactions WHERE learner_id = $1 LIMIT 1",
            heavy_learner_id,
        )
        print(f"EXPLAIN ANALYZE for a personal retrieval, learner_id={heavy_learner_id}")
        print("=" * 70)
        await conn.execute("SET hnsw.ef_search = 40")
        rows = await conn.fetch(
            f"""
            EXPLAIN (ANALYZE, BUFFERS, COSTS)
            SELECT id, learner_id, created_at,
                   1 - (question_embedding <=> $1) AS similarity
            FROM interactions_current i
            WHERE i.learner_id = $2
            ORDER BY question_embedding <=> $1
            LIMIT 50
            """,
            sample_vec, heavy_learner_id,
        )
        for r in rows:
            print(r["QUERY PLAN"])


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    pool = await create_pool(_database_url(), min_size=2, max_size=8)
    try:
        if args.seed or args.all:
            await seed(pool)
        if args.bench or args.all:
            await bench(pool)
        if args.explain or args.all:
            await explain(pool)
        if not (args.seed or args.bench or args.explain or args.all):
            parser.print_help()
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
