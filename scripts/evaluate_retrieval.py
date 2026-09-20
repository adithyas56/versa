"""Retrieval QUALITY evaluation (VERSA build context §24): speed alone is
not enough. Runs a small hand-labeled query set against the Moss engine
and reports Recall@k, top-result relevance, and — the critical safety
metric — the cross-learner leakage rate, which must be exactly 0.

Self-contained: uses a fixed synthetic corpus + labeled queries and the
in-process stub engine, so it runs offline and deterministically. A
non-zero leakage rate exits non-zero (a hard failure, §24/§37).

    uv run python scripts/evaluate_retrieval.py
"""

from __future__ import annotations

import asyncio
import sys

from probe.embeddings import EMBEDDING_DIM, StubEmbeddingClient
from probe.memory_index import MemoryDocument
from probe.memory_retrieval import retrieve_learner_memory
from probe.moss_client import MossConfig, MossService

LEARNER_A = "learner_a"
LEARNER_B = "learner_b"


def _vec(*xs: float) -> list[float]:
    return list(xs) + [0.0] * (EMBEDDING_DIM - len(xs))


# Corpus: each learner has topically-distinct memories with controlled
# vectors, plus a decoy so retrieval must actually discriminate.
CORPUS = [
    MemoryDocument("a_examples", "prefers concrete examples first",
                   {"learner_id": LEARNER_A, "status": "active", "topic": "style"}, _vec(1, 0, 0)),
    MemoryDocument("a_recursion", "studied recursion, struggled with base cases",
                   {"learner_id": LEARNER_A, "status": "active", "topic": "algorithms"}, _vec(0, 1, 0)),
    MemoryDocument("a_decoy", "unrelated note about cooking pasta",
                   {"learner_id": LEARNER_A, "status": "active", "topic": "misc"}, _vec(0, 0, 1)),
    MemoryDocument("b_theory", "prefers formal definitions first",
                   {"learner_id": LEARNER_B, "status": "active", "topic": "style"}, _vec(1, 0, 0)),
]

# (query text, learner, query vector, expected memory ids at top).
LABELED = [
    ("how should you explain things to me", LEARNER_A, _vec(1, 0, 0), {"a_examples"}),
    ("what did I struggle with in recursion", LEARNER_A, _vec(0, 1, 0), {"a_recursion"}),
    ("how should you explain things to me", LEARNER_B, _vec(1, 0, 0), {"b_theory"}),
]


async def _main() -> None:
    canned = {q: qv for q, _, qv, _ in LABELED}
    embed = StubEmbeddingClient(canned=canned)
    svc = await MossService.create(MossConfig(engine="stub", top_k=5), embed, seed_docs=CORPUS)

    hits_at_k = 0
    leaks = 0
    for query, learner, _qv, expected in LABELED:
        memories, _ = await retrieve_learner_memory(
            service=svc, learner_id=learner, user_text=query, max_inject=3
        )
        got_ids = {m.memory_id for m in memories}
        if expected & got_ids:
            hits_at_k += 1
        # Any returned memory belonging to another learner is a leak.
        for m in memories:
            if m.metadata["learner_id"] != learner:
                leaks += 1
                print(f"  LEAK: query as {learner} returned {m.memory_id} "
                      f"({m.metadata['learner_id']})")

    recall = hits_at_k / len(LABELED)
    leak_rate = leaks / max(1, sum(1 for _ in LABELED))
    print("\n=== RETRIEVAL QUALITY ===")
    print(f"labeled queries      : {len(LABELED)}")
    print(f"Recall@k             : {recall:.2f}")
    print(f"cross-learner leaks  : {leaks}")
    print(f"cross-learner rate   : {leak_rate:.3f}  (must be 0.000)")

    if leaks != 0:
        print("\nFAIL: cross-learner leakage detected.")
        sys.exit(1)
    print("\nPASS: zero cross-learner leakage.")


if __name__ == "__main__":
    asyncio.run(_main())
