"""Unit tests for the Moss retrieval adapter + stub engine
(moss_client.py / memory_retrieval.py). Pure: StubMossEngine +
StubEmbeddingClient, deterministic, no Postgres, no network.
"""

import pytest

from probe.embeddings import EMBEDDING_DIM, StubEmbeddingClient
from probe.memory_index import MemoryDocument
from probe.memory_retrieval import (
    SessionMemoryCache,
    build_learner_context_block,
    retrieve_learner_memory,
)
from probe.moss_client import MossConfig, MossService, StubMossEngine

LEARNER_A = "learner_a"
LEARNER_B = "learner_b"


def _vec(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[float]:
    return [x, y, z] + [0.0] * (EMBEDDING_DIM - 3)


def _doc(doc_id, text, learner_id, embedding, *, status="active", topic="general", confidence="0.9"):
    return MemoryDocument(
        id=doc_id,
        text=text,
        metadata={
            "learner_id": learner_id,
            "status": status,
            "topic": topic,
            "confidence": confidence,
            "memory_type": "explanation_preference",
        },
        embedding=embedding,
    )


async def _service(docs, *, canned=None, top_k=5, min_score=0.0):
    embed = StubEmbeddingClient(canned=canned or {})
    cfg = MossConfig(engine="stub", top_k=top_k, min_score=min_score)
    return await MossService.create(cfg, embed, seed_docs=docs)


@pytest.mark.asyncio
async def test_retrieval_returns_ranked_results_with_measured_metrics():
    docs = [
        _doc("mem_1", "prefers concrete examples first", LEARNER_A, _vec(x=1.0)),
        _doc("mem_2", "something unrelated", LEARNER_A, _vec(z=1.0)),
    ]
    svc = await _service(docs, canned={"explain dynamic programming": _vec(x=1.0)})
    memories, metrics = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="explain dynamic programming"
    )
    assert memories[0].memory_id == "mem_1"  # the x-aligned doc ranks first
    assert metrics.engine == "stub-moss"
    assert metrics.used_local_index is True
    assert metrics.result_count == len(memories)
    assert metrics.latency_ms >= 0.0  # a real measured span, never fabricated
    assert "stub-moss" in metrics.sources


@pytest.mark.asyncio
async def test_min_score_threshold_drops_weak_matches():
    docs = [
        _doc("mem_strong", "aligned", LEARNER_A, _vec(x=1.0)),
        _doc("mem_weak", "orthogonal", LEARNER_A, _vec(z=1.0)),
    ]
    svc = await _service(
        docs, canned={"q": _vec(x=1.0)}, min_score=0.5
    )
    memories, _ = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q"
    )
    ids = {m.memory_id for m in memories}
    assert "mem_strong" in ids
    assert "mem_weak" not in ids  # cosine 0 < 0.5 threshold


@pytest.mark.asyncio
async def test_max_inject_caps_injected_count():
    docs = [_doc(f"mem_{i}", "aligned", LEARNER_A, _vec(x=1.0)) for i in range(6)]
    svc = await _service(docs, canned={"q": _vec(x=1.0)}, top_k=6)
    memories, _ = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q", max_inject=2
    )
    assert len(memories) == 2


@pytest.mark.asyncio
async def test_inactive_memories_are_never_returned():
    docs = [
        _doc("mem_active", "aligned", LEARNER_A, _vec(x=1.0)),
        _doc("mem_retired", "aligned", LEARNER_A, _vec(x=1.0), status="retired"),
    ]
    svc = await _service(docs, canned={"q": _vec(x=1.0)})
    memories, _ = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q"
    )
    ids = {m.memory_id for m in memories}
    assert ids == {"mem_active"}


@pytest.mark.asyncio
async def test_session_cache_gives_immediate_continuity():
    # A memory not yet in the index but present in the session cache is
    # still retrieved this turn (§17).
    svc = await _service([], canned={"q": _vec(x=1.0)})
    cache = SessionMemoryCache()
    cache.add(_doc("mem_fresh", "just created this turn", LEARNER_A, _vec(x=1.0)))
    memories, metrics = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q", session_cache=cache
    )
    assert any(m.memory_id == "mem_fresh" and m.source == "session" for m in memories)
    assert "session_cache" in metrics.sources


@pytest.mark.asyncio
async def test_engine_failure_degrades_to_fallback_marked():
    class _BoomEngine(StubMossEngine):
        async def query(self, *a, **k):
            raise RuntimeError("moss down")

    embed = StubEmbeddingClient()
    cfg = MossConfig(engine="stub")
    engine = _BoomEngine(cfg, embed)
    svc = MossService(engine, cfg)
    cache = SessionMemoryCache()
    cache.add(_doc("mem_fresh", "fallback item", LEARNER_A, _vec(x=1.0)))
    memories, metrics = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q", session_cache=cache
    )
    assert metrics.fell_back is True
    assert metrics.used_local_index is False
    # The session cache still serves context; never mislabeled as Moss.
    assert all(m.source == "session" for m in memories)


def test_context_block_is_empty_when_nothing_retrieved():
    assert build_learner_context_block([]) == ""


@pytest.mark.asyncio
async def test_context_block_frames_memories_without_leaking_ids():
    docs = [_doc("mem_1", "prefers concrete examples first", LEARNER_A, _vec(x=1.0))]
    svc = await _service(docs, canned={"q": _vec(x=1.0)})
    memories, _ = await retrieve_learner_memory(
        service=svc, learner_id=LEARNER_A, user_text="q"
    )
    block = build_learner_context_block(memories)
    assert "LEARNER MEMORY CONTEXT" in block
    assert "prefers concrete examples first" in block
    # §13: internal ids must not appear in the injected context.
    assert "mem_1" not in block
