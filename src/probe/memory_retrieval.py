"""The one authoritative learner-memory retrieval contract (§18/§26).

Every live tutoring turn that needs learner context goes through
`retrieve_learner_memory` — there is deliberately no second retrieval path in
the codebase for learner memory (§26: "one authoritative retrieval adapter").
It is where three of the context's hard requirements are enforced in exactly
one place:

- Learner isolation (§11.3/§34.5/§46 Failure 5): the `learner_id` filter is
  applied at the engine query AND re-checked on every returned record here.
  Cross-learner leakage is defended in depth, not trusted to the engine
  alone, and never driven by arbitrary UI input.

- Post-retrieval selection (§12.5): drop inactive/wrong-learner records,
  apply a configurable `min_score` threshold, cap the injected count, and
  keep ids+scores for telemetry — with no second LLM call in the hot path
  (§12/§46 Failure 6).

- Controlled fallback (§20.2): if the engine raises, degrade to the
  session-local cache and mark `fell_back` so the UI never labels a fallback
  as Moss-powered. Benchmarks pass no fallback and must fail loudly instead.

`SessionMemoryCache` is the small in-process short-term memory (§17): the few
most recent projections for the current session, so a fact created this turn
is usable next turn without waiting on an async index sync. It is NOT a
replacement for Moss — it holds only very recent local state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from probe.memory_index import MemoryDocument
from probe.moss_client import MossService
from probe.retrieval_metrics import RetrievalMetrics, RetrievedMemory


@dataclass
class SessionMemoryCache:
    """Bounded, per-session, in-process (§17). Newest first; a fixed cap
    keeps it "very recent local state" and nothing more."""

    max_items: int = 8
    _docs: list[MemoryDocument] = field(default_factory=list)

    def add(self, doc: MemoryDocument) -> None:
        self._docs = [d for d in self._docs if d.id != doc.id]
        self._docs.insert(0, doc)
        del self._docs[self.max_items :]

    def for_learner(self, learner_id: str) -> list[MemoryDocument]:
        # Learner isolation applies to the cache too — never merge another
        # learner's very-recent state into this learner's context.
        return [
            d
            for d in self._docs
            if d.metadata.get("learner_id") == learner_id
            and d.metadata.get("status", "active") == "active"
        ]


def _dedupe_keep_highest(memories: list[RetrievedMemory]) -> list[RetrievedMemory]:
    """A doc can surface from both the session cache and Moss; keep the
    higher-scored copy so it is neither double-injected nor dropped."""
    best: dict[str, RetrievedMemory] = {}
    for m in memories:
        cur = best.get(m.memory_id)
        if cur is None or m.score > cur.score:
            best[m.memory_id] = m
    return list(best.values())


async def retrieve_learner_memory(
    *,
    service: MossService,
    learner_id: str,
    user_text: str,
    session_context: str | None = None,
    session_cache: SessionMemoryCache | None = None,
    top_k: int | None = None,
    min_score: float | None = None,
    max_inject: int = 4,
) -> tuple[list[RetrievedMemory], RetrievalMetrics]:
    """Retrieve the learner memories relevant to `user_text`, filtered to
    `learner_id` and ranked, with retrieval-only latency measured.

    `session_context` optionally prepends deterministic context to the query
    (§12.2), e.g. a recent topic — kept short, never an LLM-built prompt.
    """
    cfg = service.config
    resolved_top_k = top_k if top_k is not None else cfg.top_k
    resolved_min_score = min_score if min_score is not None else cfg.min_score

    # §12.2: deterministic query construction only — no LLM call to decide
    # what to retrieve (§46 Failure 6).
    query_text = user_text
    if session_context:
        query_text = f"Current request: {user_text}\nRecent context: {session_context}"

    engine_filter = {"learner_id": learner_id, "status": "active"}
    sources: list[str] = []
    fell_back = False

    session_memories: list[RetrievedMemory] = []
    if session_cache is not None:
        cached = session_cache.for_learner(learner_id)
        if cached:
            sources.append("session_cache")
            session_memories = [
                RetrievedMemory(
                    memory_id=d.id,
                    text=d.text,
                    # Session-cache items have no engine score; a neutral
                    # high value keeps just-created memory available for
                    # immediate continuity without claiming a real similarity.
                    score=1.0,
                    metadata=dict(d.metadata),
                    source="session",
                )
                for d in cached
            ]

    engine_memories: list[RetrievedMemory] = []
    try:
        result = await service.query(
            query_text, top_k=resolved_top_k, filter=engine_filter
        )
        sources.append(service.engine.engine_name)
        engine_memories = [
            RetrievedMemory(
                memory_id=h.id,
                text=h.text,
                score=h.score,
                metadata=h.metadata,
                source="moss",
            )
            for h in result.hits
        ]
        latency_ms = result.time_taken_ms
        used_local = result.used_local_index
        index_name = result.index_name
    except Exception:
        # §20.2 controlled fallback: session-local context only, clearly
        # marked. Never silently reach for pgvector here.
        fell_back = True
        latency_ms = 0.0
        used_local = False
        index_name = cfg.index_name

    # §12.5 post-retrieval selection, applied uniformly to both layers.
    combined = _dedupe_keep_highest(session_memories + engine_memories)
    selected = [
        m
        for m in combined
        # Defense in depth: re-verify learner + active status regardless of
        # what the engine returned (§46 Failure 5).
        if m.metadata.get("learner_id") == learner_id
        and m.metadata.get("status", "active") == "active"
        # Threshold applies to engine hits; session items (score 1.0) always
        # pass, which is intended for immediate continuity.
        and (m.source == "session" or m.score >= resolved_min_score)
    ]
    selected.sort(key=lambda m: (m.source != "session", -m.score))
    selected = selected[:max_inject]

    metrics = RetrievalMetrics(
        latency_ms=latency_ms,
        top_k=resolved_top_k,
        result_count=len(selected),
        index_name=index_name,
        used_local_index=used_local,
        engine=service.engine.engine_name,
        sources=sources,
        fell_back=fell_back,
    )
    return selected, metrics


def build_learner_context_block(memories: list[RetrievedMemory]) -> str:
    """Format retrieved memories as the structured LEARNER MEMORY CONTEXT
    section injected into Gemini (§13/§31.2). Correctness-first, no invented
    traits, no internal ids/infra leaked. Returns "" when there is nothing
    to inject, so the caller can omit the section entirely rather than
    emit an empty header."""
    if not memories:
        return ""

    lines = [
        "LEARNER MEMORY CONTEXT",
        "",
        "The following are retrieved, evidence-backed memories about the current learner.",
        "Use them only when relevant to the current question.",
        "Do not invent new learner traits from this context.",
        "Do not mention internal memory IDs or retrieval infrastructure.",
        "",
    ]
    for i, m in enumerate(memories, start=1):
        confidence = m.metadata.get("confidence", "")
        topic = m.metadata.get("topic", "general")
        lines.append(f"[Memory {i}]")
        lines.append(f"Statement: {m.text}")
        if confidence:
            lines.append(f"Confidence: {confidence}")
        lines.append(f"Topic: {topic}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
