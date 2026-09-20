"""Latency/telemetry value types for the learner-memory retrieval path.

Kept in its own module (rather than inside `memory_retrieval.py`) so the
webserver, benchmark script, and tests can all import the shapes without
pulling in the Moss client lifecycle. Nothing here talks to Moss, Postgres,
or Gemini — these are plain data carriers plus one telemetry-event builder.

The single most important number this module names is
`RetrievalMetrics.latency_ms`: the time from issuing a retrieval to getting
results back. Per VERSA build context §22.1 and §46 Failure 3, this is the
Moss metric that must be *measured*, never hand-written — the UI, README,
and benchmark all read it from here. It is deliberately NOT the end-to-end
Gemini latency (§22.3); a caller that conflates the two is misusing this
type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RetrievedMemory:
    """One learner-memory record returned by the retrieval adapter.

    `source` records which layer actually produced it — "moss" (the
    long-term semantic index), "session" (the in-process short-term cache,
    VERSA build context §17), or "pgvector" (the benchmark/control path,
    §0.5). Telemetry surfaces this so a fallback response is never
    mislabeled as Moss-powered (§20.2, §46 Failure 1).

    `metadata` mirrors the Moss document metadata (all string values, since
    Moss metadata is `Dict[str, str]`) so learner_id/status/topic/confidence
    survive retrieval for post-filtering and display.
    """

    memory_id: str
    text: str
    score: float
    metadata: dict[str, str]
    source: str  # "moss" | "session" | "pgvector"


@dataclass
class RetrievalMetrics:
    """Everything the telemetry panel and benchmark report about one
    retrieval, separated from the results themselves so a caller can log
    metrics without holding onto the (potentially larger) memory text.

    `used_local_index` is the load-bearing honesty flag: True only when the
    query actually ran against a locally loaded index with no network round
    trip on the query itself (VERSA build context §7.3). `engine` names the
    retrieval engine ("moss" | "stub-moss" | "pgvector") so the UI can label
    the source truthfully and the benchmark can assert which path ran.
    """

    latency_ms: float
    top_k: int
    result_count: int
    index_name: str
    used_local_index: bool
    engine: str = "moss"
    # Which layers contributed to the final result set, in the order they
    # were consulted — e.g. ["session_cache", "moss"] (§17). Distinct from
    # `engine`, which names the long-term semantic engine specifically.
    sources: list[str] = field(default_factory=list)
    # True when the primary engine raised and the adapter degraded to a
    # controlled fallback (§20.2). Never set on a benchmark run, which must
    # fail loudly instead of switching engines (§20.3).
    fell_back: bool = False


def build_retrieval_event(
    *,
    learner_id: str,
    session_id: str | None,
    turn_id: str | None,
    query_text: str,
    memories: list[RetrievedMemory],
    metrics: RetrievalMetrics,
) -> dict:
    """One `learner_memory_retrieval` telemetry event (VERSA build context
    §30). Compact by design: result *ids* and *scores* are kept for
    diagnostics, but the raw memory text is not duplicated here (§30's "do
    not persist unnecessarily sensitive raw text if a compact representation
    is enough"). The query text is kept because it is the student's own
    words about a concept, not sensitive personal data, and is what makes an
    event readable at all.
    """
    return {
        "event_type": "learner_memory_retrieval",
        "timestamp": datetime.now(UTC).isoformat(),
        "learner_id": learner_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "engine": metrics.engine,
        "index_name": metrics.index_name,
        "query_text": query_text,
        "top_k": metrics.top_k,
        "result_count": metrics.result_count,
        "latency_ms": round(metrics.latency_ms, 3),
        "local_index": metrics.used_local_index,
        "sources": metrics.sources,
        "fell_back": metrics.fell_back,
        "result_ids": [m.memory_id for m in memories],
        "scores": [round(m.score, 4) for m in memories],
    }
