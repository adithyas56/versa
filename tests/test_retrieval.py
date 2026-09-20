"""retrieval.py — deterministic, no LLM. Covers stage1's inspectable
SQL construction, stage2's tagging by embedding column, stage3's
dedup/scoring, and the unified retrieve()'s fixed 4+1 quota assembly
and population readability gate.
"""

from datetime import UTC
from uuid import uuid4

import pytest

from probe.models import EntryState, HelpLevel, QuestionAuthor, TurnOutcomeLabel
from probe.retrieval import (
    RecallHit,
    RetrievalContext,
    retrieve,
    stage1_filter,
    stage2_recall,
    stage3_rerank,
)
from probe.retrieval_config import RetrievalConfig, RetrievalWeights


def _vec(x: float) -> list[float]:
    return [x] + [0.0] * 767


# ─────────────────────────── stage1_filter ────────────────────────────


def test_stage1_always_carries_the_learner_predicate():
    learner_id = uuid4()
    where = stage1_filter(learner_id, RetrievalContext())
    assert where.sql == "i.learner_id = $1"
    assert where.params == [learner_id]


def test_stage1_with_no_filters_and_no_learner_is_unconditional():
    where = stage1_filter(None, RetrievalContext())
    assert where.sql == "TRUE"
    assert where.params == []


def test_stage1_adds_every_filter_dimension():
    learner_id = uuid4()
    ctx = RetrievalContext(
        entry_states=[EntryState.STUCK_REPEAT, EntryState.RETURNING_AFTER_GAP],
        help_levels=[HelpLevel.HINT],
        min_recent_similar_count=3,
        max_age_days=14,
    )
    where = stage1_filter(learner_id, ctx)
    assert "i.learner_id" in where.sql
    assert "i.entry_state = ANY(" in where.sql
    assert "i.help_level = ANY(" in where.sql
    assert "i.recent_similar_count >= " in where.sql
    assert "i.created_at >= now() - (" in where.sql
    # every placeholder actually has a matching param
    assert where.sql.count("$") >= len(where.params)


def test_stage1_exclude_interaction_id():
    interaction_id = uuid4()
    where = stage1_filter(None, RetrievalContext(exclude_interaction_id=interaction_id))
    assert "i.id !=" in where.sql
    assert interaction_id in where.params


def test_stage1_resolved_outcome_filter():
    where = stage1_filter(None, RetrievalContext(resolved_outcome=TurnOutcomeLabel.CONTRADICTED_INTENT))
    assert "i.resolved_outcome = " in where.sql
    assert "contradicted_intent" in where.params


# ─────────────────────────── stage3_rerank ────────────────────────────


def _hit(
    interaction_id=None, key="question", similarity=0.9, outcome=None, days_old=0, text="x",
    counterpart_similarity=None,
):
    from datetime import datetime, timedelta

    return RecallHit(
        interaction_id=interaction_id or uuid4(),
        learner_id=uuid4(),
        key=key,
        similarity=similarity,
        created_at=datetime.now(UTC) - timedelta(days=days_old),
        resolved_outcome=outcome,
        text=text,
        counterpart_similarity=counterpart_similarity,
    )


def test_rerank_dedups_by_interaction_id_keeping_the_stronger_hit():
    shared_id = uuid4()
    hits = [
        _hit(interaction_id=shared_id, key="question", similarity=0.6),
        _hit(interaction_id=shared_id, key="abstract", similarity=0.9),
    ]
    ranked = stage3_rerank(hits)
    assert len(ranked) == 1
    assert ranked[0].retrieval_key == "abstract"
    assert ranked[0].similarity == 0.9


def test_rerank_sorts_by_score_descending():
    hits = [_hit(similarity=0.3), _hit(similarity=0.9), _hit(similarity=0.6)]
    ranked = stage3_rerank(hits)
    scores = [c.score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_contradicted_outcome_gets_a_bonus():
    plain = _hit(similarity=0.5, outcome=None)
    contradicted = _hit(similarity=0.5, outcome=TurnOutcomeLabel.CONTRADICTED_INTENT)
    ranked = stage3_rerank(
        [plain, contradicted],
        RetrievalConfig(weights=RetrievalWeights(contradicted_outcome_bonus=0.5, recency_weight=0.0)),
    )
    by_id = {c.source_id: c for c in ranked}
    assert by_id[contradicted.interaction_id].score > by_id[plain.interaction_id].score


def test_rerank_more_recent_scores_higher_all_else_equal():
    old = _hit(similarity=0.5, days_old=100)
    recent = _hit(similarity=0.5, days_old=0)
    ranked = stage3_rerank([old, recent], RetrievalConfig(weights=RetrievalWeights(recency_weight=0.5)))
    by_id = {c.source_id: c for c in ranked}
    assert by_id[recent.interaction_id].score > by_id[old.interaction_id].score


def test_rerank_is_deterministic():
    """No randomness, no LLM: the SAME candidates score the SAME way on
    repeated calls. Recency itself is legitimately time-dependent (an
    interaction gets marginally older between any two calls), so this
    isolates determinism from that by weighting recency out --
    `_recency_factor` reads the wall clock by design and is not the
    thing being tested here."""
    config = RetrievalConfig(weights=RetrievalWeights(recency_weight=0.0))
    hits = [_hit(similarity=0.4), _hit(similarity=0.8), _hit(similarity=0.6)]
    r1 = [c.score for c in stage3_rerank(hits, config)]
    r2 = [c.score for c in stage3_rerank(hits, config)]
    assert r1 == r2


def test_rerank_empty_input():
    assert stage3_rerank([]) == []


def test_rerank_support_is_capped_at_one():
    """Two support counts, BOTH already past the normalization cap
    (default 50) -- a raw magnitude difference here must not translate
    into a score difference, or one heavily-aggregated candidate could
    dominate by sheer size alone."""
    hit = _hit(similarity=0.5)
    config = RetrievalConfig(
        weights=RetrievalWeights(support_weight=1.0, semantic_similarity_weight=0.0, recency_weight=0.0)
    )
    ranked_100 = stage3_rerank([hit], config, n_supported_claims={hit.interaction_id: 100})
    ranked_5000 = stage3_rerank([hit], config, n_supported_claims={hit.interaction_id: 5000})
    assert ranked_100[0].score == pytest.approx(1.0)
    assert ranked_5000[0].score == pytest.approx(ranked_100[0].score)


def test_disagreement_penalty_downranks_high_question_low_abstract_similarity():
    """The lexical-overlap false positive this penalty exists for:
    strong question-key match, weak abstract-key match -- e.g. "financial
    derivatives" pulled into a calculus query by the shared word alone."""
    lexical_overlap = _hit(key="question", similarity=0.8, counterpart_similarity=0.1)
    config = RetrievalConfig(
        weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.4)
    )
    penalized = stage3_rerank([lexical_overlap], config)[0]
    unpenalized = stage3_rerank(
        [lexical_overlap],
        RetrievalConfig(weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.0)),
    )[0]
    assert penalized.score < unpenalized.score
    assert penalized.score == pytest.approx(unpenalized.score - 0.4 * (0.8 - 0.1))


def test_disagreement_penalty_does_not_touch_the_reverse_case():
    """The cross-topic abstract-transfer case this must NOT penalize:
    weak question-key match, strong abstract-key match -- a different
    subject, same underlying move. This is the signal the abstract key
    exists to surface, not a defect."""
    abstract_transfer = _hit(key="abstract", similarity=0.8, counterpart_similarity=0.1)
    config = RetrievalConfig(
        weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.4)
    )
    penalized = stage3_rerank([abstract_transfer], config)[0]
    unpenalized = stage3_rerank(
        [abstract_transfer],
        RetrievalConfig(weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.0)),
    )[0]
    assert penalized.score == pytest.approx(unpenalized.score)


def test_disagreement_penalty_skipped_when_no_abstract_exists_yet():
    """counterpart_similarity=None means "nothing to compare against,"
    not "compared and found dissimilar" -- must not be penalized."""
    not_yet_abstracted = _hit(key="question", similarity=0.8, counterpart_similarity=None)
    config = RetrievalConfig(
        weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.4)
    )
    ranked = stage3_rerank([not_yet_abstracted], config)[0]
    assert ranked.score == pytest.approx(0.8)


def test_disagreement_penalty_is_zero_for_a_genuine_same_subject_continuation():
    """Scores reasonably on both axes -- small gap either way, penalty
    stays negligible."""
    same_subject = _hit(key="question", similarity=0.7, counterpart_similarity=0.65)
    config = RetrievalConfig(
        weights=RetrievalWeights(recency_weight=0.0, topic_abstract_disagreement_penalty=0.4)
    )
    ranked = stage3_rerank([same_subject], config)[0]
    assert ranked.score == pytest.approx(0.7 - 0.4 * 0.05)


# ───────────────────────── end-to-end (real DB) ────────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_stage2_tags_hits_by_which_embedding_column_matched(
    interaction_recorder, interaction_abstract_store, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="an answer",
    )
    from probe.models import InteractionAbstract

    await interaction_abstract_store.append(
        InteractionAbstract(
            interaction_id=interaction.id, learner_id=learner_id,
            abstract_form="asked a definitional question",
            abstract_embedding=interaction.question_embedding,  # same vector, deliberately
            generator_version="test-v1",
        )
    )
    where = stage1_filter(learner_id, RetrievalContext())
    hits = await stage2_recall(clean_pool, interaction.question_embedding, where, RetrievalConfig())
    keys = {h.key for h in hits}
    assert "question" in keys
    assert "abstract" in keys


@pytest.mark.asyncio(loop_scope="session")
async def test_stage2_excludes_interactions_with_no_abstract_from_abstract_key(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    where = stage1_filter(learner_id, RetrievalContext())
    hits = await stage2_recall(clean_pool, interaction.question_embedding, where, RetrievalConfig())
    assert all(h.key != "abstract" for h in hits), "no abstract exists yet -- must not appear as an abstract-key hit"
    assert any(h.key == "question" for h in hits)


@pytest.mark.asyncio(loop_scope="session")
async def test_retrieve_assembles_fixed_quotas_not_a_merged_top_five(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    q = "what is a derivative?"
    last = None
    for i in range(6):
        last = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=i,
            question_text=q, question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"answer {i}",
        )

    # A population pattern with the SAME vector and very high support --
    # if scopes were pooled and truncated to 5, this alone (highest
    # support) could crowd out all personal candidates.
    async with clean_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO population_patterns
                (id, abstract_form, embedding, support_count, distinct_learner_count, max_per_learner_share)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            uuid4(), "a readable population pattern", last.question_embedding, 1000, 100, 0.05,
        )

    result = await retrieve(clean_pool, learner_id, last.question_embedding, config=RetrievalConfig())
    scopes = [c.scope for c in result.candidates]
    assert scopes.count("personal") == 4, "personal quota must be filled independently of population support"
    assert scopes.count("population") == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_retrieve_excludes_unreadable_population_patterns(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    async with clean_pool.acquire() as conn:
        # too few distinct learners
        await conn.execute(
            "INSERT INTO population_patterns (id, abstract_form, embedding, support_count, distinct_learner_count, max_per_learner_share) VALUES ($1,$2,$3,$4,$5,$6)",
            uuid4(), "unreadable: too few learners", interaction.question_embedding, 100, 3, 0.1,
        )
        # one learner dominates
        await conn.execute(
            "INSERT INTO population_patterns (id, abstract_form, embedding, support_count, distinct_learner_count, max_per_learner_share) VALUES ($1,$2,$3,$4,$5,$6)",
            uuid4(), "unreadable: one learner dominates", interaction.question_embedding, 100, 25, 0.9,
        )
    result = await retrieve(clean_pool, learner_id, interaction.question_embedding, config=RetrievalConfig())
    texts = [c.text for c in result.candidates]
    assert "unreadable: too few learners" not in texts
    assert "unreadable: one learner dominates" not in texts


@pytest.mark.asyncio(loop_scope="session")
async def test_retrieve_personal_candidates_are_always_this_learner(
    interaction_recorder, transcript, learner_id, learner_store, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    # A DIFFERENT learner with a similar interaction -- must never leak
    # into this learner's personal-scope results.
    other_learner = await learner_store.create()
    other_session = await transcript.create_session(other_learner.id)
    await interaction_recorder.record(
        learner_id=other_learner.id, session_id=other_session, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a different answer",
    )
    result = await retrieve(clean_pool, learner_id, interaction.question_embedding, config=RetrievalConfig())
    personal = [c for c in result.candidates if c.scope == "personal"]
    assert all(c.learner_id == learner_id for c in personal)
