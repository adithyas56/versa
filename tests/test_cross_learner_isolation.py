"""Mandatory cross-learner isolation test (VERSA build context §37 /
§24 critical safety metric: cross-learner leakage = 0). Learner A must
never retrieve Learner B's memory in normal operation, even when the two
learners' memories are semantically identical.
"""

import pytest

from probe.embeddings import EMBEDDING_DIM, StubEmbeddingClient
from probe.memory_index import MemoryDocument
from probe.memory_retrieval import SessionMemoryCache, retrieve_learner_memory
from probe.moss_client import MossConfig, MossService

LEARNER_A = "learner_a"
LEARNER_B = "learner_b"


def _vec(*, x: float = 0.0, y: float = 0.0) -> list[float]:
    return [x, y] + [0.0] * (EMBEDDING_DIM - 2)


def _doc(doc_id, learner_id, embedding):
    return MemoryDocument(
        id=doc_id,
        text="prefers concrete examples before abstract definitions",
        metadata={"learner_id": learner_id, "status": "active", "topic": "general"},
        embedding=embedding,
    )


async def _service(docs):
    embed = StubEmbeddingClient(canned={"how should you explain this to me": _vec(x=1.0)})
    cfg = MossConfig(engine="stub", top_k=10)
    return await MossService.create(cfg, embed, seed_docs=docs)


@pytest.mark.asyncio
async def test_learner_a_never_retrieves_learner_b_memory():
    # Identical, equally-relevant memories for both learners.
    docs = [
        _doc("mem_a1", LEARNER_A, _vec(x=1.0)),
        _doc("mem_a2", LEARNER_A, _vec(x=1.0)),
        _doc("mem_b1", LEARNER_B, _vec(x=1.0)),
        _doc("mem_b2", LEARNER_B, _vec(x=1.0)),
    ]
    svc = await _service(docs)
    memories, _ = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="how should you explain this to me"
    )
    assert memories, "learner A should get their own memories"
    for m in memories:
        assert m.metadata["learner_id"] == LEARNER_A
    # The mandatory assertion: NO learner_b memory returned.
    assert all(not m.memory_id.startswith("mem_b") for m in memories)


@pytest.mark.asyncio
async def test_session_cache_does_not_leak_across_learners():
    svc = await _service([])
    cache = SessionMemoryCache()
    cache.add(_doc("mem_b_fresh", LEARNER_B, _vec(x=1.0)))
    cache.add(_doc("mem_a_fresh", LEARNER_A, _vec(x=1.0)))
    memories, _ = await retrieve_learner_memory(
        service=svc,
        learner_id=LEARNER_A,
        user_text="how should you explain this to me",
        session_cache=cache,
    )
    ids = {m.memory_id for m in memories}
    assert "mem_a_fresh" in ids
    assert "mem_b_fresh" not in ids


@pytest.mark.asyncio
async def test_a_learner_with_no_memories_gets_empty_not_others():
    docs = [_doc("mem_b1", LEARNER_B, _vec(x=1.0))]
    svc = await _service(docs)
    memories, metrics = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="how should you explain this to me"
    )
    assert memories == []
    assert metrics.result_count == 0
