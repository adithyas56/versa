"""The interaction log — append-only, database-enforced (migration
034's trigger blocks UPDATE/DELETE on `interactions` directly; see
that migration's header for why every child table below carries its
own `learner_id` and a composite FK rather than referencing `id`
alone).

`InteractionRecorder` is the orchestrator loop.py actually calls: it
embeds the question (one embedding call, no LLM), runs ONE similarity
query against this learner's last `InteractionConfig.similarity_window`
questions (regardless of session boundary) to derive
`recent_similar_count`/`prev_question_sim`/`last_similar_turn_gap`,
computes `entry_state`/`prior_turn_outcome` from those plus prior rows,
and writes the `Interaction` row through `InteractionStore`. The five
`*Store` classes below are otherwise deliberately thin CRUD, matching
this codebase's existing store style (DisambiguationStore,
LearnerFactStore) — orchestration logic lives in one recorder/loop.py,
not scattered across store methods.

topics_removal: an earlier version of this module resolved a per-learner
topic (topics.py, a running-centroid cluster with a fixed attach
threshold) and derived entry_state/turns_on_topic from cluster
membership. Two live-tuning rounds plus a systematic replay against
real embeddings confirmed the mechanism itself is broken, not just its
threshold value: averaging same-subject embeddings into a centroid
genericizes it rather than sharpening it, which makes the centroid MORE
attractive to unrelated content, which blends it further -- a cascade
with no fixed threshold that escapes it (a live run collapsed 12 turns
across 3 unrelated subjects into ONE topic). Removed entirely rather
than patched: everything that mechanism was FOR is answerable as a
direct pairwise similarity comparison against a learner's recent
questions, with no cluster identity, no centroid, and nothing ever
averaged. See migration 034's header and retrieval_config.py's
`same_subject_threshold` for the rest of the removal record.

Not routed through `SessionLoop._call_node` / `node_calls` (CLAUDE.md
invariant 2): none of this does an LLM completion. The embedding call
and every store write here are persistence, the same category as
`TranscriptStore.record_turn`/`DisambiguationStore.create_turn`, which
are also called directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import asyncpg
from pydantic import BaseModel

from probe.domain_config import Domain
from probe.embeddings import TASK_QUERY, EmbeddingClient
from probe.models import (
    EntryState,
    HelpLevel,
    Interaction,
    InteractionAbstract,
    InteractionOption,
    InteractionPriorOutcome,
    Prediction,
    QuestionAuthor,
    ReferenceBinding,
    StatedPreference,
    TurnOutcome,
    TurnOutcomeLabel,
)
from probe.row_mapping import assert_row_consumed
from probe.vector_math import cosine_similarity

logger = logging.getLogger(__name__)


def _row_to_halfvec_list(mapped: dict, *columns: str) -> None:
    for col in columns:
        value = mapped.get(col)
        if value is not None:
            mapped[col] = value.to_list()


class InteractionConfig(BaseModel):
    """Tunable window size for entry_state's similarity computation.
    The similarity THRESHOLD itself lives in
    `retrieval_config.RetrievalConfig.same_subject_threshold`, not
    here -- see that field's own docstring for why (a retrieval-
    adjacent tunable of the same kind as the rerank weights, not a
    structural constant of the recorder)."""

    # recent_similar_count / prev_question_sim / last_similar_turn_gap
    # are all derived from ONE similarity query over this learner's
    # last N interactions (any session) -- same window for all three,
    # not three separate lookups.
    similarity_window: int = 20


# TurnOutcomeLabel's 5-value vocabulary collapsed onto
# InteractionPriorOutcome's smaller, best-effort-at-creation enum. Not
# given by the original spec verbatim; this is the one defensible
# reduction (a resolved OR moved-on turn both mean "nothing left
# unresolved from the learner's side").
_OUTCOME_TO_PRIOR: dict[TurnOutcomeLabel, InteractionPriorOutcome] = {
    TurnOutcomeLabel.MATCHED: InteractionPriorOutcome.RESOLVED,
    TurnOutcomeLabel.MOVED_ON: InteractionPriorOutcome.MOVED_ON,
    TurnOutcomeLabel.CONTRADICTED_INTENT: InteractionPriorOutcome.CONTRADICTED,
    # NOT collapsed into UNKNOWN -- see InteractionPriorOutcome's own
    # docstring for why these mean opposite things.
    TurnOutcomeLabel.DEFERRED: InteractionPriorOutcome.DEFERRED,
}

# classifier_version values for the two DETERMINISTIC (no LLM call)
# deferred cases -- distinguishable in the audit trail from a real
# ClassifyTurnOutcome run, which stamps its own model/prompt version.
STRUCTURAL_NO_RESPONSE_VERSION = "structural-no-response"
STRUCTURAL_SESSION_END_VERSION = "structural-session-end"


class InteractionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, interaction: Interaction) -> Interaction:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interactions (
                    id, learner_id, session_id, turn_number, question_text,
                    question_author, originating_question, did_branch,
                    response_text, entry_state, recent_similar_count,
                    prev_question_sim, last_similar_turn_gap,
                    prior_turn_outcome, help_level, elapsed_ms,
                    question_embedding, abstract_form, abstract_embedding,
                    domain, created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18, $19, $20, $21
                )
                """,
                interaction.id,
                interaction.learner_id,
                interaction.session_id,
                interaction.turn_number,
                interaction.question_text,
                interaction.question_author.value,
                interaction.originating_question,
                interaction.did_branch,
                interaction.response_text,
                interaction.entry_state.value,
                interaction.recent_similar_count,
                interaction.prev_question_sim,
                interaction.last_similar_turn_gap,
                interaction.prior_turn_outcome.value,
                interaction.help_level.value,
                interaction.elapsed_ms,
                interaction.question_embedding,
                interaction.abstract_form,
                interaction.abstract_embedding,
                interaction.domain.value,
                interaction.created_at,
            )
        return interaction

    async def get(self, learner_id: UUID, interaction_id: UUID) -> Interaction | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interactions WHERE learner_id = $1 AND id = $2",
                learner_id,
                interaction_id,
            )
        return None if row is None else self._row_to_interaction(row)

    async def get_at_turn(self, session_id: UUID, turn_number: int) -> Interaction | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interactions WHERE session_id = $1 AND turn_number = $2",
                session_id,
                turn_number,
            )
        return None if row is None else self._row_to_interaction(row)

    async def get_previous_in_session(
        self, session_id: UUID, turn_number: int
    ) -> Interaction | None:
        """The interaction row immediately before this turn_number in
        this session, if any -- the specific "prior turn" prior_turn_
        outcome is best-effort computed from, and the specific row
        RESOLUTION's did_branch check reads."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM interactions
                WHERE session_id = $1 AND turn_number < $2
                ORDER BY turn_number DESC LIMIT 1
                """,
                session_id,
                turn_number,
            )
        return None if row is None else self._row_to_interaction(row)

    async def has_any_in_session(self, session_id: UUID) -> bool:
        async with self._pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM interactions WHERE session_id = $1)",
                session_id,
            )
        return bool(exists)

    async def get_recent_for_learner(
        self, learner_id: UUID, limit: int, domain: Domain | None = None
    ) -> list[Interaction]:
        """This learner's last `limit` interactions, most-recent-first,
        ANY session -- the single window recent_similar_count/
        prev_question_sim/last_similar_turn_gap are all derived from
        (see InteractionConfig.similarity_window and this module's
        docstring). `created_at DESC, id DESC` rather than `created_at`
        alone: a tiebreak for interactions written in the same instant,
        same reasoning as node_calls.seq elsewhere in this codebase,
        without needing a dedicated seq column on a table this hot.

        `domain`, when given, restricts the window to that domain only
        -- the domain switch's storage/retrieval exception
        (domain_config.py): entry_state/recent_similar_count/
        prev_question_sim/last_similar_turn_gap must never be computed
        against a different domain's history for the same learner_id.
        """
        async with self._pool.acquire() as conn:
            if domain is None:
                rows = await conn.fetch(
                    """
                    SELECT * FROM interactions
                    WHERE learner_id = $1
                    ORDER BY created_at DESC, id DESC
                    LIMIT $2
                    """,
                    learner_id,
                    limit,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT * FROM interactions
                    WHERE learner_id = $1 AND domain = $2
                    ORDER BY created_at DESC, id DESC
                    LIMIT $3
                    """,
                    learner_id,
                    domain.value,
                    limit,
                )
        return [self._row_to_interaction(r) for r in rows]

    async def get_last_in_session(self, session_id: UUID) -> Interaction | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interactions WHERE session_id = $1 "
                "ORDER BY turn_number DESC LIMIT 1",
                session_id,
            )
        return None if row is None else self._row_to_interaction(row)

    async def get_last_response_bearing_from_other_session(
        self, learner_id: UUID, session_id: UUID
    ) -> Interaction | None:
        """This learner's most recent interaction, in a DIFFERENT
        session, that actually has a response -- the cross-session
        catch-up target for the DEFERRED-at-session-end rule: when a
        learner returns, their first question in the new session is
        the missing N+1 evidence for whatever their last resolved turn
        in the old session was."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM interactions
                WHERE learner_id = $1 AND session_id != $2 AND response_text IS NOT NULL
                ORDER BY created_at DESC LIMIT 1
                """,
                learner_id,
                session_id,
            )
        return None if row is None else self._row_to_interaction(row)

    def _row_to_interaction(self, row) -> Interaction:
        mapped = dict(row)
        _row_to_halfvec_list(mapped, "question_embedding", "abstract_embedding")
        assert_row_consumed(Interaction, mapped)
        return Interaction(**mapped)


class InteractionOptionStore:
    """Deliberately excluded from the immutability trigger — see
    migration 034's header comment: was_selected/selection_timestamp
    are populated on a LATER turn than creation."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create_many(self, options: list[InteractionOption]) -> list[InteractionOption]:
        if not options:
            return []
        async with self._pool.acquire() as conn, conn.transaction():
            for o in options:
                await conn.execute(
                    """
                    INSERT INTO interaction_options (
                        id, interaction_id, learner_id, option_id, branch_id,
                        option_text, shown_position, was_selected,
                        selection_timestamp, kind, axis, side, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    """,
                    o.id,
                    o.interaction_id,
                    o.learner_id,
                    o.option_id,
                    o.branch_id,
                    o.option_text,
                    o.shown_position,
                    o.was_selected,
                    o.selection_timestamp,
                    o.kind.value if o.kind else None,
                    o.axis.value if o.axis else None,
                    o.side,
                    o.created_at,
                )
        return options

    async def mark_selected(self, option_id: UUID) -> InteractionOption | None:
        """Flip was_selected/selection_timestamp for the interaction_options
        row matching this disambiguation option_id -- called from the
        click-resolution turn, a separate handle_turn call from the one
        that created this row."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE interaction_options
                SET was_selected = TRUE, selection_timestamp = $2
                WHERE option_id = $1
                RETURNING *
                """,
                option_id,
                datetime.now(timezone.utc),
            )
        return None if row is None else self._row_to_option(row)

    async def list_for_interaction(self, interaction_id: UUID) -> list[InteractionOption]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM interaction_options WHERE interaction_id = $1 "
                "ORDER BY shown_position",
                interaction_id,
            )
        return [self._row_to_option(r) for r in rows]

    def _row_to_option(self, row) -> InteractionOption:
        mapped = dict(row)
        assert_row_consumed(InteractionOption, mapped)
        return InteractionOption(**mapped)


class InteractionAbstractStore:
    """Append-only by convention (no update/delete method, no DELETE
    SQL) -- not under a hard DB trigger; only `interactions` itself was
    asked to be enforced at that level."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, abstract: InteractionAbstract) -> InteractionAbstract:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interaction_abstracts (
                    id, interaction_id, learner_id, abstract_form,
                    abstract_embedding, generator_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                abstract.id,
                abstract.interaction_id,
                abstract.learner_id,
                abstract.abstract_form,
                abstract.abstract_embedding,
                abstract.generator_version,
                abstract.created_at,
            )
        return abstract

    async def get_latest(self, interaction_id: UUID) -> InteractionAbstract | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interaction_abstracts WHERE interaction_id = $1 "
                "ORDER BY seq DESC LIMIT 1",
                interaction_id,
            )
        if row is None:
            return None
        mapped = dict(row)
        mapped.pop("seq", None)
        _row_to_halfvec_list(mapped, "abstract_embedding")
        assert_row_consumed(InteractionAbstract, mapped)
        return InteractionAbstract(**mapped)

    async def list_all_latest(self, limit: int = 50_000) -> list[InteractionAbstract]:
        """The latest version of every distinct interaction_id's
        abstract -- feeds population_patterns.aggregate_population_
        patterns. A single full pass, not paginated: fine at this
        pipeline's current scale, but a production version processing
        millions of abstracts would need a cursor here rather than one
        bulk fetch. `limit` is a hard ceiling so a runaway table can't
        make one aggregation run try to load everything into memory."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (interaction_id) *
                FROM interaction_abstracts
                ORDER BY interaction_id, seq DESC
                LIMIT $1
                """,
                limit,
            )
        results = []
        for row in rows:
            mapped = dict(row)
            mapped.pop("seq", None)
            _row_to_halfvec_list(mapped, "abstract_embedding")
            assert_row_consumed(InteractionAbstract, mapped)
            results.append(InteractionAbstract(**mapped))
        return results


class TurnOutcomeStore:
    """Append-only: re-running the classifier writes a new row under a
    new classifier_version; readers resolve to the latest version per
    interaction_id, never an update in place."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, outcome: TurnOutcome) -> TurnOutcome:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO turn_outcomes (
                    id, interaction_id, learner_id, next_question_text,
                    outcome, confidence, classifier_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                outcome.id,
                outcome.interaction_id,
                outcome.learner_id,
                outcome.next_question_text,
                outcome.outcome.value,
                outcome.confidence,
                outcome.classifier_version,
                outcome.created_at,
            )
        return outcome

    async def get_latest(self, interaction_id: UUID) -> TurnOutcome | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM turn_outcomes WHERE interaction_id = $1 "
                "ORDER BY seq DESC LIMIT 1",
                interaction_id,
            )
        return None if row is None else self._row_to_outcome(row)

    async def get_latest_many(
        self, interaction_ids: list[UUID]
    ) -> dict[UUID, TurnOutcome]:
        """Bulk latest-per-interaction_id lookup -- used by
        retrieval.stage3_rerank's contradicted-outcome bonus, which
        needs this for every candidate in one round-trip rather than
        one query per candidate."""
        if not interaction_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (interaction_id) *
                FROM turn_outcomes
                WHERE interaction_id = ANY($1::uuid[])
                ORDER BY interaction_id, seq DESC
                """,
                interaction_ids,
            )
        return {r["interaction_id"]: self._row_to_outcome(r) for r in rows}

    def _row_to_outcome(self, row) -> TurnOutcome:
        mapped = dict(row)
        mapped.pop("seq", None)
        assert_row_consumed(TurnOutcome, mapped)
        return TurnOutcome(**mapped)


class StatedPreferenceStore:
    """Append-only, same convention as TurnOutcomeStore -- see
    StatedPreference's own docstring for why this is a separate table
    from `interactions`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, preference: StatedPreference) -> StatedPreference:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO stated_preferences (
                    id, interaction_id, learner_id, has_preference,
                    stated_preference, label, classifier_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                preference.id,
                preference.interaction_id,
                preference.learner_id,
                preference.has_preference,
                preference.stated_preference,
                preference.label.value if preference.label else None,
                preference.classifier_version,
                preference.created_at,
            )
        return preference

    async def get_latest_for_learner(self, learner_id: UUID) -> StatedPreference | None:
        """The single most recent TRUE stated preference for this
        learner, across every session -- what Fix B's render step
        (loop.py/disambiguate.py's FinalAnswer) reads to build the
        structural-requirement line. None whenever this learner has
        never explicitly stated one."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM stated_preferences WHERE learner_id = $1 "
                "AND has_preference ORDER BY seq DESC LIMIT 1",
                learner_id,
            )
        return None if row is None else self._row_to_preference(row)

    def _row_to_preference(self, row) -> StatedPreference:
        mapped = dict(row)
        mapped.pop("seq", None)
        assert_row_consumed(StatedPreference, mapped)
        return StatedPreference(**mapped)


class ReferenceBindingStore:
    """Append-only (see ReferenceBinding's own docstring): no delete/
    update method, no UPDATE/DELETE SQL touching an existing row. This
    is the one store in this module whose write path reads before it
    writes -- `record_resolution`'s own read of the current latest row
    for a (learner_id, reference_text) pair decides the NEW row's
    `confirmation_count`, the same way `ThinkingStyleStore.confirm()`'s
    growing counter depends on the row's own prior value; the
    difference is that growth here always writes a fresh row, per this
    feature's explicit append-only requirement, never an UPDATE."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_latest(
        self, learner_id: UUID, reference_text: str
    ) -> ReferenceBinding | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reference_bindings WHERE learner_id = $1 "
                "AND reference_text = $2 ORDER BY seq DESC LIMIT 1",
                learner_id,
                reference_text,
            )
        return None if row is None else self._row_to_binding(row)

    async def list_latest_for_learner(self, learner_id: UUID) -> list[ReferenceBinding]:
        """The read side's one query -- latest row per distinct
        reference_text for this learner. `reference_bindings.py` does
        the exact-match filtering against a message's text afterward,
        in Python, deterministically -- no embedding, no LLM call."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (reference_text) *
                FROM reference_bindings
                WHERE learner_id = $1
                ORDER BY reference_text, seq DESC
                """,
                learner_id,
            )
        return [self._row_to_binding(r) for r in rows]

    async def record_resolution(
        self,
        learner_id: UUID,
        reference_text: str,
        resolved_to: str,
        evidence_interaction_id: UUID,
        classifier_version: str,
    ) -> ReferenceBinding:
        """The write side's one entry point (called only from loop.py's
        background classification task). Reads this pair's current
        latest row, if any: the SAME `resolved_to` is a genuine
        re-confirmation (`confirmation_count` = previous + 1); a
        DIFFERENT `resolved_to` means this phrase now means something
        else, and starts a fresh count at 1 -- the earlier row is never
        touched, it stays on record as what the phrase used to mean."""
        existing = await self.get_latest(learner_id, reference_text)
        confirmation_count = (
            existing.confirmation_count + 1
            if existing is not None and existing.resolved_to == resolved_to
            else 1
        )
        return await self.append(
            ReferenceBinding(
                learner_id=learner_id,
                reference_text=reference_text,
                resolved_to=resolved_to,
                evidence_interaction_id=evidence_interaction_id,
                confirmation_count=confirmation_count,
                classifier_version=classifier_version,
            )
        )

    async def append(self, binding: ReferenceBinding) -> ReferenceBinding:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO reference_bindings (
                    id, learner_id, reference_text, resolved_to,
                    evidence_interaction_id, confirmation_count,
                    last_confirmed_at, classifier_version, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                binding.id,
                binding.learner_id,
                binding.reference_text,
                binding.resolved_to,
                binding.evidence_interaction_id,
                binding.confirmation_count,
                binding.last_confirmed_at,
                binding.classifier_version,
                binding.created_at,
            )
        return binding

    def _row_to_binding(self, row) -> ReferenceBinding:
        mapped = dict(row)
        mapped.pop("seq", None)
        assert_row_consumed(ReferenceBinding, mapped)
        return ReferenceBinding(**mapped)


class PredictionStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, prediction: Prediction) -> Prediction:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO predictions (
                    id, interaction_id, learner_id, predicted_scores,
                    prediction_created_at, model_version,
                    retrieved_candidate_ids, retrieval_provenance
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                prediction.id,
                prediction.interaction_id,
                prediction.learner_id,
                prediction.predicted_scores,
                prediction.prediction_created_at,
                prediction.model_version,
                prediction.retrieved_candidate_ids,
                prediction.retrieval_provenance,
            )
        return prediction

    async def get_latest(self, interaction_id: UUID) -> Prediction | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM predictions WHERE interaction_id = $1 "
                "ORDER BY seq DESC LIMIT 1",
                interaction_id,
            )
        return None if row is None else self._row_to_prediction(row)

    def _row_to_prediction(self, row) -> Prediction:
        mapped = dict(row)
        mapped.pop("seq", None)
        assert_row_consumed(Prediction, mapped)
        return Prediction(**mapped)


def _similarity_stats(
    embedding: list[float],
    recent: list[Interaction],
    threshold: float,
) -> tuple[int, float | None, int | None]:
    """Pure function: derives all three of recent_similar_count/
    prev_question_sim/last_similar_turn_gap from ONE already-fetched
    window (`recent`, most-recent-first, any session) -- no query, no
    clustering, no averaging. `recent` empty means this is the
    learner's very first interaction ever: all three are their
    "nothing to compare against" values (0, None, None).
    """
    if not recent:
        return 0, None, None
    similarities = [cosine_similarity(embedding, past.question_embedding) for past in recent]
    prev_question_sim = similarities[0]
    recent_similar_count = sum(1 for s in similarities if s >= threshold)
    last_similar_turn_gap = next(
        (idx + 1 for idx, s in enumerate(similarities) if s >= threshold), None
    )
    return recent_similar_count, prev_question_sim, last_similar_turn_gap


class InteractionRecorder:
    """Orchestrates one interaction write: embed the question (one
    embedding call, no LLM), run one similarity comparison against this
    learner's recent questions, compute entry_state/prior_turn_outcome
    from that plus prior rows, then persist. Called directly from
    loop.py at each of its finish-points -- see loop.py's own comments
    for exactly where, and why IDs are threaded from the outer scope
    rather than through the FinalAnswer call chain.
    """

    def __init__(
        self,
        interaction_store: InteractionStore,
        turn_outcome_store: TurnOutcomeStore,
        embedding_client: EmbeddingClient,
        same_subject_threshold: float,
        config: InteractionConfig | None = None,
    ) -> None:
        self._interactions = interaction_store
        self._turn_outcomes = turn_outcome_store
        self._embeddings = embedding_client
        # Required, not defaulted here: this is retrieval_config.
        # RetrievalConfig.same_subject_threshold's value, threaded in by
        # the caller (cli.py) -- kept out of InteractionConfig so there
        # is exactly one place this number is defined, not two that can
        # drift apart. See that field's own docstring for the value and
        # its calibration record.
        self._same_subject_threshold = same_subject_threshold
        self._config = config or InteractionConfig()
        # Set as a side effect of the most recent record() call -- the
        # OTHER interaction (same-session previous, or a cross-session
        # catch-up target) that now has real N+1 evidence available for
        # ClassifyTurnOutcome, or None if nothing needs classifying this
        # turn. Same "last_call_count"-style side-channel convention
        # this codebase already uses (AssessAndBranch, FinalAnswer) --
        # loop.py reads this right after calling record() to decide
        # whether to fire the async classifier.
        self.last_classification_target: Interaction | None = None

    async def record(
        self,
        *,
        learner_id: UUID,
        session_id: UUID,
        turn_number: int,
        question_text: str,
        question_author: QuestionAuthor,
        originating_question: str | None,
        did_branch: bool,
        response_text: str | None,
        help_level: HelpLevel = HelpLevel.NONE,
        elapsed_ms: int | None = None,
        domain: Domain = Domain.EDUCATION,
    ) -> Interaction:
        self.last_classification_target = None
        embed_text = (
            originating_question
            if question_author is QuestionAuthor.SYSTEM_OPTION
            else question_text
        )
        assert embed_text is not None, (
            "originating_question is required when question_author is "
            "SYSTEM_OPTION -- a click turn must never embed the option "
            "generator's own phrasing"
        )

        embedding = await self._embeddings.embed(embed_text, task_type=TASK_QUERY)

        previous_in_session = await self._interactions.get_previous_in_session(
            session_id, turn_number
        )
        # Domain-filtered: the domain switch's storage/retrieval
        # exception (see get_recent_for_learner's own docstring) --
        # entry_state must never be computed against a different
        # domain's history for the same learner_id.
        recent = await self._interactions.get_recent_for_learner(
            learner_id, self._config.similarity_window, domain=domain
        )
        recent_similar_count, prev_question_sim, last_similar_turn_gap = _similarity_stats(
            embedding, recent, self._same_subject_threshold
        )
        entry_state = await self._compute_entry_state(
            previous_in_session=previous_in_session,
            prev_question_sim=prev_question_sim,
            last_similar_turn_gap=last_similar_turn_gap,
            immediately_prior=recent[0] if recent else None,
        )
        prior_turn_outcome = await self._compute_prior_turn_outcome(previous_in_session)

        interaction = Interaction(
            id=uuid4(),
            learner_id=learner_id,
            session_id=session_id,
            turn_number=turn_number,
            question_text=question_text,
            question_author=question_author,
            originating_question=originating_question,
            did_branch=did_branch,
            response_text=response_text,
            entry_state=entry_state,
            recent_similar_count=recent_similar_count,
            prev_question_sim=prev_question_sim,
            last_similar_turn_gap=last_similar_turn_gap,
            prior_turn_outcome=prior_turn_outcome,
            help_level=help_level,
            elapsed_ms=elapsed_ms,
            question_embedding=embedding,
            domain=domain,
        )
        created = await self._interactions.create(interaction)

        if response_text is None:
            # Known instantly, without waiting for a next turn: an
            # options-offered row has no N+1 evidence by construction.
            # No LLM call -- this is a structural fact, not a judgment.
            await self._turn_outcomes.append(
                TurnOutcome(
                    interaction_id=created.id,
                    learner_id=learner_id,
                    outcome=TurnOutcomeLabel.DEFERRED,
                    confidence=1.0,
                    classifier_version=STRUCTURAL_NO_RESPONSE_VERSION,
                )
            )
        elif previous_in_session is not None and previous_in_session.response_text is not None:
            # This turn's own question_text is the missing N+1 evidence
            # for the immediately preceding (same-session) resolved
            # interaction -- the caller (loop.py) fires the real
            # ClassifyTurnOutcome call for it, off the critical path.
            self.last_classification_target = previous_in_session
        elif previous_in_session is None:
            # Cold open: no prior interaction THIS session, but this
            # learner may have a trailing resolved interaction from an
            # earlier session that was never given N+1 evidence (session
            # ended, or was only ever stamped structural-session-end).
            # Their first question here IS that evidence now.
            catch_up = await self._interactions.get_last_response_bearing_from_other_session(
                learner_id, session_id
            )
            if catch_up is not None:
                self.last_classification_target = catch_up

        return created

    async def mark_trailing_interaction_deferred(self, session_id: UUID) -> None:
        """Session-end hook (called from wherever this session's
        consolidation/end-of-session step already lives, e.g.
        SessionLoop.consolidate_session): if this session's last
        interaction has a response but no N+1 ever arrived within it,
        write an explicit DEFERRED row rather than leaving the record
        silent. Superseded later (latest classifier_version wins) if
        the learner returns and a cross-session catch-up classification
        runs for real."""
        last = await self._interactions.get_last_in_session(session_id)
        if last is None:
            return
        if last.response_text is None:
            return  # already handled at its own creation time
        existing = await self._turn_outcomes.get_latest(last.id)
        if existing is not None:
            return  # already classified (or already deferred) -- don't overwrite
        await self._turn_outcomes.append(
            TurnOutcome(
                interaction_id=last.id,
                learner_id=last.learner_id,
                outcome=TurnOutcomeLabel.DEFERRED,
                confidence=1.0,
                classifier_version=STRUCTURAL_SESSION_END_VERSION,
            )
        )

    async def _compute_entry_state(
        self,
        *,
        previous_in_session: Interaction | None,
        prev_question_sim: float | None,
        last_similar_turn_gap: int | None,
        immediately_prior: Interaction | None,
    ) -> EntryState:
        """Direct comparisons over the stored similarity numbers, in
        this exact precedence order -- see interactions.py's module
        docstring / retrieval_config.same_subject_threshold for why
        each check is where it is:

            resolution           previous turn had did_branch = true
            cold_open            first turn of session
            continuing           prev_question_sim >= threshold
            stuck_repeat         continuing AND prior outcome ==
                                  contradicted_intent
            returning_after_gap  no recent match, but
                                  last_similar_turn_gap is not None
            topic_switch         no match anywhere in the window

        resolution is checked FIRST, before any similarity comparison:
        a click turn embeds Turn A's own originating_question, so it is
        trivially "similar" to the offer it was generated from by
        construction -- that similarity is not informative and must
        never be allowed to compete with the other checks.
        """
        if previous_in_session is not None and previous_in_session.did_branch:
            return EntryState.RESOLUTION
        if previous_in_session is None:
            return EntryState.COLD_OPEN

        if prev_question_sim is not None and prev_question_sim >= self._same_subject_threshold:
            if immediately_prior is not None:
                latest_outcome = await self._turn_outcomes.get_latest(immediately_prior.id)
                if (
                    latest_outcome is not None
                    and latest_outcome.outcome is TurnOutcomeLabel.CONTRADICTED_INTENT
                ):
                    return EntryState.STUCK_REPEAT
            return EntryState.CONTINUING

        if last_similar_turn_gap is not None:
            # Matched something in the window, just not the immediately
            # preceding question -- gap case not spelled out verbatim
            # in the original spec (recent_similar_count > 0 but
            # prev_question_sim below threshold); returning_after_gap is
            # the closest semantic fit and is what last_similar_turn_gap
            # exists to answer, so it resolves this case by construction
            # rather than needing a separate branch.
            return EntryState.RETURNING_AFTER_GAP

        # No match anywhere in the window -- the learner abandoned or
        # completed one thing and moved to another. Deliberately its
        # own state: folding this into CONTINUING would merge one of
        # the more informative signals here into the single most common
        # value in the table.
        return EntryState.TOPIC_SWITCH

    async def _compute_prior_turn_outcome(
        self, previous_in_session: Interaction | None
    ) -> InteractionPriorOutcome:
        if previous_in_session is None:
            return InteractionPriorOutcome.UNKNOWN
        outcome = await self._turn_outcomes.get_latest(previous_in_session.id)
        if outcome is None:
            return InteractionPriorOutcome.UNKNOWN
        return _OUTCOME_TO_PRIOR[outcome.outcome]
