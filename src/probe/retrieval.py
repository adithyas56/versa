"""Deterministic retrieval over personal history and population-level
patterns — three stages, no LLM call anywhere in this module, ~25ms
budget end to end. See `interaction_nodes.PredictSelection` for the
one caller that currently exists: it uses `retrieve()`'s output as
context for its own (LLM-based, swappable) scoring, but nothing in
here does that scoring itself.

STAGE 1 (`stage1_filter`) builds an inspectable SQL WHERE-clause
fragment over entry_state, help_level, resolved outcome (via
`interactions_current`, migration 034's joined view), recent_similar_count,
recency, and learner_id — plain parameterized SQL, no ORM, so the
predicate a query actually runs is always readable as text, not
reconstructed from an object graph. Personal-scope queries always
carry the learner_id predicate. There is deliberately no topic
predicate: topic-based clustering was removed (see interactions.py's
module docstring and models.Interaction's docstring for the failure
record); entry_state, recency, and learner scope are the filters that
remain, and the ~16ms this predicate used to cost was never a hard
constraint to begin with.

STAGE 2 (`stage2_recall`) runs the actual ANN queries against
`interactions_current`, combining stage1's WHERE fragment with an
`ORDER BY <embedding column> <=> query_vec LIMIT top_n` — one query
against `question_embedding` (always populated), one against
`current_abstract_embedding` (populated once the async abstraction job
has run; NULL rows are excluded by the query itself). Each candidate
is tagged by which column matched it. `hnsw.ef_search` is set per
`RetrievalConfig` before either query runs.

Each query ALSO computes the OTHER column's similarity to the same
query vector in the same SELECT (`RecallHit.counterpart_similarity`) --
both columns are already joined on `interactions_current`, so this
costs nothing extra and, critically, means a hit's counterpart value is
known even when that interaction never separately cleared the other
query's own top-`stage2_top_n` ANN cutoff. stage3_rerank's disagreement
penalty (see its own docstring) depends on always having both numbers,
not just whichever one two independent top-N cutoffs happened to both
surface.

Note the asymmetry this makes visible, not something the queries need
to account for: `current_abstract_embedding` can be NULL (not yet
abstracted), so a question-key hit's `counterpart_similarity` can be
NULL too -- that means "no abstract exists yet to compare against,"
never "compared and found dissimilar." An abstract-key hit's
`counterpart_similarity` (against `question_embedding`) is never NULL,
since that column is NOT NULL on every row.

A FILTERED HNSW scan can silently under-return: the graph traversal
has no guarantee of visiting enough candidates that also satisfy an
arbitrary WHERE predicate, so a query can come back fast with far
fewer than `top_n` rows even when `top_n`+ would satisfy the filter.
This is not a hypothetical — it is exactly the failure mode the
100k-row benchmark (see `bench_retrieval.py`) checks for as a hard
requirement, separate from latency: a fast answer with too few
candidates is not a passing result.

STAGE 3 (`stage3_rerank`) computes one deterministic weighted score
per candidate from `RetrievalWeights` — no LLM, no randomness, the
same candidates in the same filter state always rank the same way.

UNIFIED RETRIEVAL (`retrieve`) returns FIXED quotas — 4 personal + 1
population by default — each ranked independently within its own
scope and assembled afterward, never pooled together and truncated to
5. Population aggregates carry higher support (by construction: a
readable pattern already cleared 20+ distinct learners) and would
crowd out personal continuity in a single merged ranking, which is
exactly what makes a live session feel like it remembers a specific
learner rather than reciting a population average. Every candidate
keeps provenance: scope, source id, retrieval key, learner_id where
applicable, similarity, recency, outcome, support counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import exp
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from probe.domain_config import Domain
from probe.models import (
    EntryState,
    HelpLevel,
    RetrievalCandidate,
    TurnOutcomeLabel,
)
from probe.retrieval_config import RetrievalConfig

_QUESTION_KEY = "question"
_ABSTRACT_KEY = "abstract"


class RetrievalContext(BaseModel):
    """Every filter dimension stage1_filter builds a WHERE fragment
    over. All optional -- an unset field means "no constraint on this
    dimension," not "match nothing." `exclude_interaction_id` keeps a
    live turn from recalling itself while its own row already exists
    (e.g. once created for a click-resolution turn)."""

    entry_states: list[EntryState] | None = None
    help_levels: list[HelpLevel] | None = None
    resolved_outcome: TurnOutcomeLabel | None = None
    min_recent_similar_count: int | None = None
    max_age_days: float | None = None
    exclude_interaction_id: UUID | None = None
    # The domain switch's storage/retrieval exception (domain_config.py):
    # personal-scope retrieval must never surface a different domain's
    # interactions for the same learner_id. None means "no domain
    # filter" -- population scope has no domain column at all (see
    # `_population_recall`'s own docstring for that gap, left
    # unaddressed and reported rather than silently patched).
    domain: Domain | None = None


@dataclass
class _WhereFragment:
    """Plain, inspectable SQL -- `sql` is literal WHERE-clause text,
    `params` are its positional values starting at $1. Callers append
    their own params (the query vector, LIMIT) after these."""

    sql: str
    params: list


def stage1_filter(learner_id: UUID | None, ctx: RetrievalContext) -> _WhereFragment:
    """Builds the WHERE fragment personal-scope (learner_id set) and
    population-scope (learner_id None, filtering population_patterns
    instead -- see `_population_where`) retrieval both start from.

    Deliberately a pure function returning SQL text + params, not
    something that itself talks to the database -- the "inspectable"
    requirement means a caller (or a test) can read exactly what will
    run without executing it.
    """
    clauses: list[str] = []
    params: list = []

    def _next_param(value) -> str:
        params.append(value)
        return f"${len(params)}"

    if learner_id is not None:
        clauses.append(f"i.learner_id = {_next_param(learner_id)}")
    if ctx.entry_states:
        placeholder = _next_param([s.value for s in ctx.entry_states])
        clauses.append(f"i.entry_state = ANY({placeholder}::interaction_entry_state[])")
    if ctx.help_levels:
        placeholder = _next_param([h.value for h in ctx.help_levels])
        clauses.append(f"i.help_level = ANY({placeholder}::interaction_help_level[])")
    if ctx.resolved_outcome is not None:
        clauses.append(f"i.resolved_outcome = {_next_param(ctx.resolved_outcome.value)}")
    if ctx.min_recent_similar_count is not None:
        clauses.append(
            f"i.recent_similar_count >= {_next_param(ctx.min_recent_similar_count)}"
        )
    if ctx.max_age_days is not None:
        clauses.append(
            f"i.created_at >= now() - ({_next_param(ctx.max_age_days)} || ' days')::interval"
        )
    if ctx.exclude_interaction_id is not None:
        clauses.append(f"i.id != {_next_param(ctx.exclude_interaction_id)}")
    if ctx.domain is not None:
        clauses.append(f"i.domain = {_next_param(ctx.domain.value)}")

    sql = " AND ".join(clauses) if clauses else "TRUE"
    return _WhereFragment(sql=sql, params=params)


@dataclass
class RecallHit:
    interaction_id: UUID
    learner_id: UUID
    key: str  # _QUESTION_KEY | _ABSTRACT_KEY
    similarity: float
    created_at: datetime
    resolved_outcome: TurnOutcomeLabel | None
    text: str
    # The OTHER embedding column's similarity to the same query vector,
    # computed in the same row -- see stage2_recall's own docstring for
    # why this is always populated (when the underlying column isn't
    # NULL) rather than only present for candidates that separately
    # cleared both queries' top-N cutoffs. None only when there is no
    # abstract yet for this interaction to compare against.
    counterpart_similarity: float | None = None


async def stage2_recall(
    pool: asyncpg.Pool,
    query_vec: list[float],
    where: _WhereFragment,
    config: RetrievalConfig | None = None,
) -> list[RecallHit]:
    """Two ANN queries against `interactions_current` (migration 034's
    view), one per embedding column, each independently LIMIT'd to
    `config.stage2_top_n`. Every hit is tagged by which column matched
    it -- a candidate with both a question_embedding and a populated
    current_abstract_embedding can appear twice, once per key; stage3
    dedups by keeping the stronger of the two per interaction_id.
    """
    cfg = config or RetrievalConfig()
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(f"SET LOCAL hnsw.ef_search = {int(cfg.hnsw_ef_search)}")

            query_vec_param = f"${len(where.params) + 1}"
            limit_param = f"${len(where.params) + 2}"
            question_rows = await conn.fetch(
                f"""
                SELECT id, learner_id, created_at, resolved_outcome, question_text,
                       1 - (question_embedding <=> {query_vec_param}) AS similarity,
                       CASE WHEN current_abstract_embedding IS NULL THEN NULL
                            ELSE 1 - (current_abstract_embedding <=> {query_vec_param})
                       END AS counterpart_similarity
                FROM interactions_current i
                WHERE {where.sql}
                ORDER BY question_embedding <=> {query_vec_param}
                LIMIT {limit_param}
                """,
                *where.params,
                query_vec,
                cfg.stage2_top_n,
            )

            abstract_rows = await conn.fetch(
                f"""
                SELECT id, learner_id, created_at, resolved_outcome, current_abstract_form,
                       1 - (current_abstract_embedding <=> {query_vec_param}) AS similarity,
                       1 - (question_embedding <=> {query_vec_param}) AS counterpart_similarity
                FROM interactions_current i
                WHERE ({where.sql}) AND current_abstract_embedding IS NOT NULL
                ORDER BY current_abstract_embedding <=> {query_vec_param}
                LIMIT {limit_param}
                """,
                *where.params,
                query_vec,
                cfg.stage2_top_n,
            )

    hits: list[RecallHit] = []
    for row in question_rows:
        hits.append(
            RecallHit(
                interaction_id=row["id"],
                learner_id=row["learner_id"],
                key=_QUESTION_KEY,
                similarity=row["similarity"],
                created_at=row["created_at"],
                resolved_outcome=(
                    TurnOutcomeLabel(row["resolved_outcome"])
                    if row["resolved_outcome"]
                    else None
                ),
                text=row["question_text"],
                counterpart_similarity=row["counterpart_similarity"],
            )
        )
    for row in abstract_rows:
        hits.append(
            RecallHit(
                interaction_id=row["id"],
                learner_id=row["learner_id"],
                key=_ABSTRACT_KEY,
                similarity=row["similarity"],
                created_at=row["created_at"],
                resolved_outcome=(
                    TurnOutcomeLabel(row["resolved_outcome"])
                    if row["resolved_outcome"]
                    else None
                ),
                text=row["current_abstract_form"] or "",
                counterpart_similarity=row["counterpart_similarity"],
            )
        )
    return hits


def _recency_factor(created_at: datetime, half_life_days: float) -> float:
    age_days = (datetime.now(UTC) - created_at).total_seconds() / 86400.0
    return exp(-age_days / half_life_days) if half_life_days > 0 else 1.0


def stage3_rerank(
    hits: list[RecallHit],
    config: RetrievalConfig | None = None,
    n_supported_claims: dict[UUID, int] | None = None,
) -> list[RetrievalCandidate]:
    """Deterministic weighted score, no LLM, no randomness. Dedups
    multiple hits for the same interaction_id (one via question_embedding,
    one via current_abstract_embedding) by keeping the higher-similarity
    hit -- an interaction recalled by both is real signal, but it must
    not be double-counted as two separate candidates.

    DISAGREEMENT PENALTY: a candidate that matches strongly on the
    question key but weakly on the abstract key is lexical overlap, not
    a real match ("derivatives" pulling a financial-derivatives turn
    into a calculus query is the case this was written for -- see
    retrieval_config.RetrievalWeights.topic_abstract_disagreement_penalty
    for why that specific case turned out to have too small a gap to be
    fixed by this on the one session it's been checked against)  --
    downrank it via `w.topic_abstract_disagreement_penalty *
    max(0, question_sim - abstract_sim)`. Deliberately one-directional:
    a candidate that matches weakly on the question key but strongly on
    the abstract key (the same underlying MOVE on a different subject
    -- the actual cross-topic transfer this whole layer exists to
    surface) gets zero penalty, since question_sim - abstract_sim is
    negative there and the max(0, ...) floors it. Skipped entirely
    (penalty 0) when `counterpart_similarity` is None -- no abstract
    exists yet to compare against, which is not evidence of a weak
    match.

    `n_supported_claims` is keyed by interaction_id -- 0 (the config
    default) for an ordinary personal interaction; population-scope
    callers pass `population_patterns.support_count` through this same
    parameter, keyed by the pattern's own id, so one scoring function
    serves both scopes.
    """
    cfg = config or RetrievalConfig()
    w = cfg.weights
    claims = n_supported_claims or {}

    best_per_id: dict[UUID, RecallHit] = {}
    for hit in hits:
        current = best_per_id.get(hit.interaction_id)
        if current is None or hit.similarity > current.similarity:
            best_per_id[hit.interaction_id] = hit

    candidates: list[RetrievalCandidate] = []
    for interaction_id, hit in best_per_id.items():
        recency_days = (datetime.now(UTC) - hit.created_at).total_seconds() / 86400.0
        contradicted_bonus = (
            w.contradicted_outcome_bonus
            if hit.resolved_outcome is TurnOutcomeLabel.CONTRADICTED_INTENT
            else 0.0
        )
        support = min(
            claims.get(interaction_id, 0) / w.support_normalization_cap, 1.0
        )
        question_sim = hit.similarity if hit.key == _QUESTION_KEY else hit.counterpart_similarity
        abstract_sim = hit.similarity if hit.key == _ABSTRACT_KEY else hit.counterpart_similarity
        disagreement_penalty = (
            w.topic_abstract_disagreement_penalty * max(0.0, question_sim - abstract_sim)
            if question_sim is not None and abstract_sim is not None
            else 0.0
        )
        score = (
            w.semantic_similarity_weight * hit.similarity
            + w.recency_weight * _recency_factor(hit.created_at, w.recency_half_life_days)
            + contradicted_bonus
            + w.support_weight * support
            - disagreement_penalty
        )
        candidates.append(
            RetrievalCandidate(
                scope="personal",
                source_id=interaction_id,
                retrieval_key=hit.key,
                learner_id=hit.learner_id,
                similarity=hit.similarity,
                recency_days=recency_days,
                outcome=hit.resolved_outcome,
                support_count=claims.get(interaction_id),
                text=hit.text,
                score=score,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


async def _population_recall(
    pool: asyncpg.Pool,
    query_vec: list[float],
    config: RetrievalConfig,
) -> list[RetrievalCandidate]:
    """Population scope's own recall + rerank, kept separate from the
    personal-scope path above rather than sharing stage1_filter/
    stage2_recall's SQL -- a different table, a different readability
    gate (distinct_learner_count/max_per_learner_share), and no
    learner_id predicate at all (a pattern is not attributed to one
    learner).

    NOT domain-filtered: `population_patterns` (migration 034) has no
    `domain` column, and adding one would mean threading domain through
    the abstraction/aggregation pipeline (interaction_abstracts ->
    `probe aggregate-patterns`) as well as this table -- out of scope
    for what this feature's own spec asked for ("add a domain column
    to interactions"). This is a genuine, unaddressed gap: a population
    pattern aggregated from a mix of education- and general-domain
    abstracts could still surface here regardless of which domain is
    running. Reported as a known leak, not silently fixed."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL hnsw.ef_search = {int(config.hnsw_ef_search)}")
        rows = await conn.fetch(
            """
                SELECT id, abstract_form, support_count, distinct_learner_count,
                       max_per_learner_share, created_at,
                       1 - (embedding <=> $1) AS similarity
                FROM population_patterns
                WHERE distinct_learner_count >= $2 AND max_per_learner_share <= $3
                ORDER BY embedding <=> $1
                LIMIT $4
                """,
            query_vec,
            config.population_min_distinct_learners,
            config.population_max_learner_share,
            config.stage2_top_n,
        )

    w = config.weights
    candidates: list[RetrievalCandidate] = []
    for row in rows:
        recency_days = (datetime.now(UTC) - row["created_at"]).total_seconds() / 86400.0
        support = min(row["support_count"] / w.support_normalization_cap, 1.0)
        similarity = row["similarity"]
        score = (
            w.semantic_similarity_weight * similarity
            + w.recency_weight * _recency_factor(row["created_at"], w.recency_half_life_days)
            + w.support_weight * support
            # No contradicted_outcome_bonus at population scope --
            # population_patterns carries no per-row outcome; the
            # bonus is a personal-interaction signal only.
        )
        candidates.append(
            RetrievalCandidate(
                scope="population",
                source_id=row["id"],
                retrieval_key=_ABSTRACT_KEY,
                learner_id=None,
                similarity=similarity,
                recency_days=recency_days,
                outcome=None,
                support_count=row["support_count"],
                distinct_learner_count=row["distinct_learner_count"],
                text=row["abstract_form"],
                score=score,
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


@dataclass
class RetrievalResult:
    candidates: list[RetrievalCandidate]
    elapsed_ms: float = field(default=0.0)


async def retrieve(
    pool: asyncpg.Pool,
    learner_id: UUID,
    query_vec: list[float],
    ctx: RetrievalContext | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """The unified interface: fixed quotas (default 4 personal + 1
    population), each scope ranked independently, assembled after --
    never one merged pool truncated to 5. See module docstring for why.
    """
    cfg = config or RetrievalConfig()
    context = ctx or RetrievalContext()
    start = time.monotonic()

    where = stage1_filter(learner_id, context)
    personal_hits = await stage2_recall(pool, query_vec, where, cfg)
    personal_ranked = stage3_rerank(personal_hits, cfg)
    population_ranked = await _population_recall(pool, query_vec, cfg)

    assembled = (
        personal_ranked[: cfg.quotas.personal]
        + population_ranked[: cfg.quotas.population]
    )
    elapsed_ms = (time.monotonic() - start) * 1000
    return RetrievalResult(candidates=assembled, elapsed_ms=elapsed_ms)
