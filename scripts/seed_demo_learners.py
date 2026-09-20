"""Seed the two intentionally-different demo learners (VERSA build
context §3.2/§5.3): Learner A (example-first) and Learner B (theory-
first). Each gets evidence-backed `learner_facts` in Postgres; run
`build_moss_index.py` afterwards to project them into the Moss index.

These are synthetic demo personas, not real students (§34.1). Idempotent
by label: re-running resumes the same two learners rather than creating
duplicates.

    uv run python scripts/seed_demo_learners.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from probe import embeddings as _embeddings
from probe.audit import TranscriptStore
from probe.db import create_pool
from probe.embeddings import StubEmbeddingClient, build_embedding_client
from probe.learner import LearnerStore
from probe.memory import LearnerFactStore
from probe.models import LearnerFact, LearnerFactType

# (label, [(fact_type, situation, resolution)]) — the evidence each demo
# persona carries (§3.2). Phrased in the learner's own terms, so the
# projected memory reads naturally (§11.4).
DEMO_LEARNERS = {
    "demo-learner-a": [
        (LearnerFactType.BRANCH_RESOLUTION,
         "how to explain a new concept like recursion",
         "start with a concrete real-world example before any formal definition"),
        (LearnerFactType.DIRECT_ANSWER,
         "learning recursion",
         "understood it via a worked example but struggled with the base case"),
        (LearnerFactType.DIRECT_ANSWER,
         "preferred explanation length",
         "prefers concise language and asks for worked examples"),
    ],
    "demo-learner-b": [
        (LearnerFactType.BRANCH_RESOLUTION,
         "how to explain a new concept",
         "give the formal definition and technical terminology first, examples after"),
        (LearnerFactType.DIRECT_ANSWER,
         "studying algorithmic complexity",
         "worked comfortably from the formal definitions of big-O"),
        (LearnerFactType.DIRECT_ANSWER,
         "when learning something new",
         "usually asks for the implementation only after the concept is clear"),
    ],
}


def _embedding_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    return build_embedding_client(api_key) if api_key else StubEmbeddingClient()


async def _main() -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check .env)")
    pool = await create_pool(url, min_size=1, max_size=4)
    learners = LearnerStore(pool)
    transcript = TranscriptStore(pool)
    facts = LearnerFactStore(pool)
    embed = _embedding_client()

    try:
        for label, fact_specs in DEMO_LEARNERS.items():
            learner = await learners.get_by_label(label)
            if learner is None:
                learner = await learners.create(label=label)
            session_id = await transcript.create_session(learner.id)
            print(f"{label}: learner {learner.id}")
            for i, (fact_type, situation, resolution) in enumerate(fact_specs):
                turn_id = await transcript.record_turn(session_id, i, situation)
                vector = await embed.embed(
                    f"{situation}\n{resolution}", task_type=_embeddings.TASK_DOCUMENT
                )
                await facts.add(
                    LearnerFact(
                        learner_id=learner.id,
                        session_id=session_id,
                        turn_index=i,
                        fact_type=fact_type,
                        situation=situation,
                        resolution=resolution,
                        embedding=vector,
                        source_turn_id=turn_id,
                    )
                )
                print(f"  + {fact_type.value}: {situation}")
        print("\nDone. Now run: uv run python scripts/build_moss_index.py")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
