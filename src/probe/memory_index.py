"""Projection of durable learner state -> compact Moss memory documents.

VERSA build context §9.2/§10: Moss does not index the raw database. It holds
*compact searchable memory documents*, each a retrieval projection of a
durable Postgres row (a `LearnerFact` today; claims/capabilities can join
later through the same `MemoryDocument` shape). Postgres stays the source of
truth (§0.4); this module only builds the searchable projection and hands it
to whichever engine the retrieval adapter is using.

Two rules from the context shape everything here:

- §11.4 "indexing text": the projected `text` must be a natural-language
  statement a semantic query can match ("Learner prefers concrete examples
  before abstract definitions…"), never a bare label ("example preference").

- Moss metadata values are strings only (`Dict[str, str]` in the SDK), so
  every metadata field — confidence included — is stringified here. The
  `status` and `learner_id` keys are what the retrieval adapter filters on
  (§12.4), so they are always present.

`to_moss_document` is the only place that imports the `moss` SDK, and it does
so lazily: the projection itself (`project_fact`) is pure and testable with
no SDK, no network, and no credentials (§37 unit tests).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from probe.models import LearnerFact, LearnerFactType

if TYPE_CHECKING:  # pragma: no cover - typing only
    from moss import DocumentInfo

# Stable prefix so a projected document id is recognizable and never collides
# with a raw interaction/turn id in logs or telemetry.
_MEMORY_ID_PREFIX = "mem_"

# LearnerFactType -> one of the §10.1 memory_type vocabulary values. A branch
# resolution is the record of a resolved ambiguity; a direct answer is a
# recent-learning trace. Kept as an explicit map (not str(fact_type)) so the
# projected vocabulary stays the context's, not the DB enum's.
_MEMORY_TYPE_BY_FACT: dict[LearnerFactType, str] = {
    LearnerFactType.BRANCH_RESOLUTION: "ambiguity_resolution",
    LearnerFactType.DIRECT_ANSWER: "recent_learning",
}


@dataclass
class MemoryDocument:
    """Engine-agnostic projection: an id, the natural-language `text` that
    gets embedded/searched, and string-only `metadata`. `embedding` is
    optional and only populated when a caller already has the durable row's
    vector (the stub engine reuses it to avoid re-embedding; real Moss
    embeds `text` with its own model and ignores this).

    Deliberately not a Moss type — the same projection feeds the Moss engine,
    the stub engine, and the pgvector control path without any of them being
    a hard import for callers that only need the projection.
    """

    id: str
    text: str
    metadata: dict[str, str]
    embedding: list[float] | None = field(default=None, repr=False)


def _statement_from_fact(fact: LearnerFact) -> str:
    """Compose one natural-language sentence from a fact's situation +
    resolution. Both are already in the student's own terms (see
    `LearnerFact` docstring), so this reads as a memory a query can match on,
    satisfying §11.4 rather than emitting a terse label."""
    situation = fact.situation.strip().rstrip(".")
    resolution = fact.resolution.strip().rstrip(".")
    if fact.fact_type is LearnerFactType.BRANCH_RESOLUTION:
        return (
            f"When faced with {situation}, the learner resolved it as: "
            f"{resolution}."
        )
    return f"The learner worked on {situation}. Outcome: {resolution}."


def project_fact(fact: LearnerFact, *, topic: str = "general") -> MemoryDocument:
    """`LearnerFact` -> `MemoryDocument`. Pure: no SDK, no I/O.

    `confidence` is stringified per Moss's string-only metadata. A durable,
    written fact carries no single stored probability, so we record a fixed
    "1.0" for an explicit branch-click resolution (§10.2: an explicit user
    choice is strong evidence) and a more cautious "0.7" for a direct-answer
    trace, keeping the claim honest rather than inventing precision (§14.2).
    """
    confidence = "1.0" if fact.fact_type is LearnerFactType.BRANCH_RESOLUTION else "0.7"
    metadata: dict[str, str] = {
        "learner_id": str(fact.learner_id),
        "memory_type": _MEMORY_TYPE_BY_FACT.get(fact.fact_type, "recent_learning"),
        # §12.4/§20.2: retrieval filters on status == "active"; a projection
        # is active by construction (append-only facts are never retired),
        # but the key is always present so the filter is uniform.
        "status": "active",
        "topic": topic,
        "confidence": confidence,
        "source_interaction_id": str(fact.source_turn_id),
        "session_id": str(fact.session_id),
        "created_at": fact.created_at.isoformat(),
    }
    return MemoryDocument(
        id=f"{_MEMORY_ID_PREFIX}{fact.id}",
        text=_statement_from_fact(fact),
        metadata=metadata,
        embedding=list(fact.embedding) if fact.embedding else None,
    )


def project_facts(facts: list[LearnerFact]) -> list[MemoryDocument]:
    """Batch projection — used by the index-build and sync scripts."""
    return [project_fact(f) for f in facts]


async def load_memory_docs_from_facts(fact_store, *, limit: int | None = None) -> list[MemoryDocument]:
    """Project every durable `LearnerFact` into a `MemoryDocument` — the
    seed set for building/loading the Moss index at startup (§19) and for
    the `build_moss_index` / `sync_memory_to_moss` scripts. `fact_store` is
    a `memory.LearnerFactStore`; kept untyped here to avoid importing the
    store (and its asyncpg dependency) into this pure-projection module."""
    facts = await fact_store.list_all(limit=limit)
    return project_facts(facts)


def to_moss_document(doc: MemoryDocument) -> DocumentInfo:
    """`MemoryDocument` -> the Moss SDK's `DocumentInfo`. Lazily imports the
    SDK so nothing that only needs the projection pays for `moss` being
    importable/installed. `text` is what Moss embeds locally with its own
    model; the projected `embedding` is intentionally not forwarded so the
    real index stays in Moss's own vector space."""
    from moss import DocumentInfo

    return DocumentInfo(id=doc.id, text=doc.text, metadata=doc.metadata)


def to_moss_documents(docs: list[MemoryDocument]) -> list[DocumentInfo]:
    return [to_moss_document(d) for d in docs]
