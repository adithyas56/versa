"""Population-pattern aggregation — the middle step of "raw
interaction -> abstract form -> aggregation -> multi-learner support ->
retrieval" (see models.PopulationPattern's own docstring). Derived from
abstract forms only, never raw transcripts.

Deliberately a batch job, not something that runs per-turn: grouping
every learner's abstracts against every other learner's is a global,
cross-learner operation, unlike everything else in this pipeline
(which is scoped to one learner and fires on one turn). Triggered on
demand via `probe aggregate-patterns` (cli.py), the same "explicit,
no automatic per-turn trigger" precedent as `probe consolidate-session`.

CLUSTERING: exact-text grouping on `abstract_form` would almost never
produce multi-learner support in practice -- two learners' abstracts
describing the same underlying pattern ("chose the worked example over
the stated rule" vs. "picked the example instead of the rule") are
different strings with a high-similarity embedding. This uses a greedy
threshold-attach algorithm (embed, compare against existing cluster
centroids by cosine similarity, attach above threshold or start a new
cluster, update the centroid as a running mean).

CAUTION, not a clean bill of health: a per-learner topic-clustering
mechanism used this exact update rule (running-mean centroid, single
fixed threshold) and was removed from interactions.py after a real run
showed it cascades -- averaging same-subject embeddings genericizes the
centroid rather than sharpening it, which makes the centroid MORE
attractive to unrelated content, which blends it further, compounding.
Nothing here has been tested against that failure mode at population
scale (the only live test to date used a handful of near-identical
synthetic vectors, which cannot reveal a cascade the way varied real
data did for topics). The readability gate below (>=20 distinct
learners, <=25% single-learner share) does not prevent a cascaded
mega-cluster from existing; it only requires that cluster have broad,
diverse support before retrieval will read from it -- a large enough
cascade could plausibly still clear that bar. What IS different from
the topic-resolution failure: this runs offline as a re-triggerable
batch job over `population_patterns`, a table nothing else's
correctness depends on and which itself is never marked immutable, so
a bad run costs a misleading suggested pattern, not a corrupted row
inside another store's permanent record. Revisit this algorithm with
the same scrutiny (a real replay against real multi-learner data)
before trusting its output at scale -- this comment is a flag, not a
verification. cosine_similarity/running_mean are still the right
primitives regardless of that open question; they moved to
vector_math.py once the class that used to own them
(topics.TopicStore) was deleted.

READABILITY GATE: only clusters clearing BOTH distinct_learner_count
>= 20 and max_per_learner_share <= 0.25 are written to
`population_patterns` at all -- an unreadable cluster is simply
dropped at the end of a run, not persisted in some other, unreadable
state. Re-running this job does not delete or update previous runs'
rows (population_patterns is append-only); a stale pattern is
naturally deprioritized by retrieval's own recency weighting, not by
being superseded in place -- see models.PopulationPattern.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel

from probe.interactions import InteractionAbstractStore
from probe.models import PopulationPattern
from probe.vector_math import cosine_similarity, running_mean


class PopulationAggregationConfig(BaseModel):
    """Starting values, expected to be retuned -- untested at real
    scale (see this module's own CAUTION note above)."""

    attach_threshold: float = 0.72
    min_distinct_learners: int = 20
    max_per_learner_share: float = 0.25
    # Safety ceiling passed through to InteractionAbstractStore.
    # list_all_latest -- see that method's own docstring.
    max_abstracts_considered: int = 50_000


class _Cluster:
    __slots__ = ("abstract_form", "centroid", "learner_counts", "n")

    def __init__(self, centroid: list[float], abstract_form: str, learner_id: UUID) -> None:
        self.centroid = centroid
        self.n = 1
        # The FIRST abstract to start a cluster becomes its label --
        # arbitrary but stable and human-readable, same spirit as
        # topics.py's "the label is just the question text that first
        # created the topic."
        self.abstract_form = abstract_form
        self.learner_counts: dict[UUID, int] = {learner_id: 1}

    def attach(self, embedding: list[float], learner_id: UUID) -> None:
        self.centroid = running_mean(self.centroid, embedding, self.n)
        self.n += 1
        self.learner_counts[learner_id] = self.learner_counts.get(learner_id, 0) + 1

    @property
    def distinct_learner_count(self) -> int:
        return len(self.learner_counts)

    @property
    def max_per_learner_share(self) -> float:
        return max(self.learner_counts.values()) / self.n

    def is_readable(self, config: PopulationAggregationConfig) -> bool:
        return (
            self.distinct_learner_count >= config.min_distinct_learners
            and self.max_per_learner_share <= config.max_per_learner_share
        )


def _cluster_abstracts(abstracts, config: PopulationAggregationConfig) -> list[_Cluster]:
    """Pure, deterministic given its input order -- no LLM, no
    randomness. Processes abstracts in the order given (callers pass
    them ordered by creation so the algorithm's behavior is stable and
    reproducible across runs on the same data)."""
    clusters: list[_Cluster] = []
    for abstract in abstracts:
        best: tuple[_Cluster, float] | None = None
        for cluster in clusters:
            similarity = cosine_similarity(cluster.centroid, abstract.abstract_embedding)
            if best is None or similarity > best[1]:
                best = (cluster, similarity)
        if best is not None and best[1] >= config.attach_threshold:
            best[0].attach(abstract.abstract_embedding, abstract.learner_id)
        else:
            clusters.append(
                _Cluster(abstract.abstract_embedding, abstract.abstract_form, abstract.learner_id)
            )
    return clusters


class PopulationPatternStore:
    """Append-only (see module docstring): each aggregation run inserts
    fresh rows; nothing here is ever updated or deleted."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, pattern: PopulationPattern) -> PopulationPattern:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO population_patterns (
                    id, abstract_form, embedding, support_count,
                    distinct_learner_count, max_per_learner_share,
                    representative_features, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                pattern.id,
                pattern.abstract_form,
                pattern.embedding,
                pattern.support_count,
                pattern.distinct_learner_count,
                pattern.max_per_learner_share,
                pattern.representative_features,
                pattern.created_at,
            )
        return pattern

    async def count_readable(self) -> int:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT count(*) FROM population_patterns "
                "WHERE distinct_learner_count >= $1 AND max_per_learner_share <= $2",
                20,
                0.25,
            )
        return int(count)


async def aggregate_population_patterns(
    abstract_store: InteractionAbstractStore,
    pattern_store: PopulationPatternStore,
    config: PopulationAggregationConfig | None = None,
) -> list[PopulationPattern]:
    """The aggregation step: cluster every learner's latest abstract,
    keep only clusters clearing both readability gates, append each as
    a new `population_patterns` row. Returns the patterns actually
    written (readable clusters only) -- an unreadable cluster is
    counted in the run but never persisted.
    """
    cfg = config or PopulationAggregationConfig()
    abstracts = await abstract_store.list_all_latest(cfg.max_abstracts_considered)
    clusters = _cluster_abstracts(abstracts, cfg)

    written: list[PopulationPattern] = []
    for cluster in clusters:
        if not cluster.is_readable(cfg):
            continue
        pattern = PopulationPattern(
            id=uuid4(),
            abstract_form=cluster.abstract_form,
            embedding=cluster.centroid,
            support_count=cluster.n,
            distinct_learner_count=cluster.distinct_learner_count,
            max_per_learner_share=cluster.max_per_learner_share,
            representative_features={},
        )
        written.append(await pattern_store.append(pattern))
    return written
