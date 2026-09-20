"""ClaimStore (create/get/search_similar/append_evidence/refresh/
contradict) and reconcile_candidate -- real DB, StubEmbeddingClient
(with `canned` used to force two different statements to read as "the
same claim" or leave them at the stub's default near-orthogonal
similarity for "not a match")."""

from uuid import uuid4

import pytest

from probe.claims import (
    ClaimRestatementConfig,
    maybe_restate_claims,
    merge_duplicate_claims,
    reconcile_candidate,
)
from probe.embeddings import EMBEDDING_DIM
from probe.llm import StubLLMClient
from probe.models import (
    ApproachAxis,
    Claim,
    ClaimCandidate,
    ClaimSource,
    ClaimStatus,
    ClaimWritePolicy,
    StatedPreferenceLabel,
)

_SAME_VECTOR = [0.5] * EMBEDDING_DIM


def _claim(learner_id, statement="wants concrete examples first", embedding=None, **overrides):
    defaults = dict(
        learner_id=learner_id,
        statement=statement,
        test="given a concrete_general axis choice, picks concrete",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        confidence=0.5,
        source=ClaimSource.INFERRED,
        write_policy=ClaimWritePolicy.SLOW_DRIFT,
        statement_embedding=embedding or list(_SAME_VECTOR),
    )
    defaults.update(overrides)
    return Claim(**defaults)


@pytest.mark.asyncio(loop_scope="session")
async def test_create_and_get_roundtrip(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id)
    await claim_store.create(claim)
    fetched = await claim_store.get(claim.id)
    assert fetched is not None
    assert fetched.statement == claim.statement
    assert fetched.value is StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT
    assert fetched.status is ClaimStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_context_scope_with_real_data_roundtrips_as_a_dict_not_a_string(
    claim_store, learner_id, clean_pool
):
    """The double-encoding fix: context_scope was unread in production,
    which is exactly why passing json.dumps(...) into a jsonb column
    that already has its own codec-level encoder went unnoticed --
    every existing row happened to be the empty-dict default, so the
    bug never surfaced as a wrong VALUE, only as a wrong TYPE nothing
    was checking. A non-empty dict makes the type distinction
    observable: a double-encoded value round-trips as a JSON string
    containing the original text, not as a dict."""
    claim = _claim(learner_id, context_scope={"topic": "quicksort", "difficulty": 3})
    await claim_store.create(claim)
    fetched = await claim_store.get(claim.id)
    assert fetched.context_scope == {"topic": "quicksort", "difficulty": 3}
    assert isinstance(fetched.context_scope, dict)


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_spans_every_learner(claim_store, learner_store, clean_pool):
    """score_predictions.py's cross-learner calibration check needs
    every claim regardless of learner -- unlike list_for_learner."""
    learner_a = (await learner_store.create()).id
    learner_b = (await learner_store.create()).id
    claim_a = _claim(learner_a, statement="claim for learner A")
    claim_b = _claim(learner_b, statement="claim for learner B")
    await claim_store.create(claim_a)
    await claim_store.create(claim_b)
    all_claims = await claim_store.list_all()
    ids = {c.id for c in all_claims}
    assert claim_a.id in ids
    assert claim_b.id in ids


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_finds_a_near_identical_embedding(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id, embedding=list(_SAME_VECTOR))
    await claim_store.create(claim)
    matches = await claim_store.search_similar(learner_id, list(_SAME_VECTOR))
    assert len(matches) == 1
    found, similarity = matches[0]
    assert found.id == claim.id
    assert similarity > 0.99


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_excludes_contradicted_claims(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id, embedding=list(_SAME_VECTOR))
    await claim_store.create(claim)
    await claim_store.contradict(claim.id)
    matches = await claim_store.search_similar(learner_id, list(_SAME_VECTOR))
    assert matches == []


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_excludes_retracted_claims(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id, embedding=list(_SAME_VECTOR))
    await claim_store.create(claim)
    await claim_store.retract(claim.id)
    matches = await claim_store.search_similar(learner_id, list(_SAME_VECTOR))
    assert matches == []


@pytest.mark.asyncio(loop_scope="session")
async def test_retract_is_terminal_even_after_refresh(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id)
    await claim_store.create(claim)
    await claim_store.retract(claim.id)
    refreshed = await claim_store.refresh(claim.id)
    assert refreshed.status is ClaimStatus.RETRACTED


@pytest.mark.asyncio(loop_scope="session")
async def test_contradict_is_terminal_even_after_refresh(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id)
    await claim_store.create(claim)
    await claim_store.contradict(claim.id)
    refreshed = await claim_store.refresh(claim.id)
    assert refreshed.status is ClaimStatus.CONTRADICTED


@pytest.mark.asyncio(loop_scope="session")
async def test_refresh_updates_confidence_from_evidence(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    claim = _claim(learner_id)
    await claim_store.create(claim)
    before = await claim_store.get(claim.id)

    await claim_store.append_evidence(
        ClaimEvidence(
            claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
            direction=EvidenceDirection.SUPPORTS, topic="calculus", session_id=session_id,
            test_fired=True, contradiction_was_possible=True,
        )
    )
    after = await claim_store.refresh(claim.id)
    assert after.confidence != before.confidence
    assert after.confidence > before.confidence


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_new_candidate_creates_a_claim(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidate = ClaimCandidate(
        statement="prefers brief answers", test="given a brevity_depth axis choice, picks brief",
        value=StatedPreferenceLabel.PREFERS_BREVITY, topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, interaction.id,
        candidate, contradiction_was_possible=True,
    )
    assert claim.status is ClaimStatus.CANDIDATE
    evidence = await claim_store.list_evidence(claim.id)
    assert len(evidence) == 1
    assert evidence[0].direction.value == "supports"


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_matching_value_appends_supporting_evidence(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    from probe.embeddings import StubEmbeddingClient
    from probe.models import QuestionAuthor

    statement_a = "wants concrete examples before the rule"
    statement_b = "prefers a worked example first"
    embed = StubEmbeddingClient(canned={statement_a: list(_SAME_VECTOR), statement_b: list(_SAME_VECTOR)})

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement=statement_a, test="test A", value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i0.id, first, contradiction_was_possible=True,
    )

    second = ClaimCandidate(
        statement=statement_b, test="test B", value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        topic="physics",
    )
    result = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i1.id, second, contradiction_was_possible=True,
    )

    assert result.id == claim.id  # matched, not a new claim
    evidence = await claim_store.list_evidence(claim.id)
    assert len(evidence) == 2
    assert {e.topic for e in evidence} == {"calculus", "physics"}
    assert all(e.direction.value == "supports" for e in evidence)


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_a_single_opposite_value_pick_lowers_confidence_but_does_not_close(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """The one-shot-contradiction design is gone: a single opposite-
    value pick, even eligible, moves the Beta posterior down but does
    NOT close the claim -- see `evaluate_contradiction`'s docstring for
    why (people are inconsistent; one off-persona pick doesn't disprove
    a standing trait). Same session as the supporting episode, same as
    before."""
    from probe.embeddings import StubEmbeddingClient
    from probe.models import QuestionAuthor

    statement_a = "wants concrete examples before the rule"
    statement_b = "wants the rule stated before any example"
    embed = StubEmbeddingClient(canned={statement_a: list(_SAME_VECTOR), statement_b: list(_SAME_VECTOR)})

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement=statement_a, test="test A", value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i0.id, first, contradiction_was_possible=True,
    )

    opposite = ClaimCandidate(
        statement=statement_b, test="test B", value=StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
        topic="physics",
    )
    result = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i1.id, opposite, contradiction_was_possible=True,
    )

    assert result.id == claim.id
    assert result.status is ClaimStatus.CANDIDATE  # NOT closed by one episode
    assert result.confidence < claim.confidence  # but the reversal moved the number down
    evidence = await claim_store.list_evidence(claim.id)
    directions = {e.topic: e.direction.value for e in evidence}
    assert directions["calculus"] == "supports"
    assert directions["physics"] == "contradicts"


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_sustained_cross_session_reversal_derives_contradicted(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """Three eligible contradicting episodes from three DISTINCT
    sessions -- against the claim's one supporting episode -- pushes
    confidence below the floor with contrary evidence from well over
    `contradiction_min_sessions` sessions, so the claim now legitimately
    earns `contradicted`."""
    from probe.embeddings import StubEmbeddingClient
    from probe.models import QuestionAuthor

    statement_a = "wants concrete examples before the rule"
    statement_b = "wants the rule stated before any example"
    embed = StubEmbeddingClient(canned={statement_a: list(_SAME_VECTOR), statement_b: list(_SAME_VECTOR)})

    session_0 = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_0, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    first = ClaimCandidate(
        statement=statement_a, test="test A", value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embed, learner_id, session_0, i0.id, first, contradiction_was_possible=True,
    )

    result = claim
    for n, topic in enumerate(["physics", "chemistry", "biology"], start=1):
        session_n = await transcript.create_session(learner_id)
        i_n = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_n, turn_number=0,
            question_text=f"q{n}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{n}",
        )
        opposite = ClaimCandidate(
            statement=statement_b, test="test B", value=StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
            topic=topic,
        )
        result = await reconcile_candidate(
            claim_store, embed, learner_id, session_n, i_n.id, opposite,
            contradiction_was_possible=True,
        )
        if result.status is ClaimStatus.CONTRADICTED:
            break

    assert result.id == claim.id
    assert result.status is ClaimStatus.CONTRADICTED


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_an_ineligible_contradiction_is_recorded_but_does_not_close_the_claim(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """The contradiction-gate fix: an opposite-value pick where
    contradiction_was_possible=False (an uncontestable, subject-kind
    episode) must still be written as evidence -- provenance, visible
    in the ledger -- but must NOT flip the claim to contradicted. An
    ineligible pick has zero power to raise confidence; it must also
    have zero power to terminally close a claim."""
    from probe.embeddings import StubEmbeddingClient
    from probe.models import QuestionAuthor

    statement_a = "wants concrete examples before the rule"
    statement_b = "wants the rule stated before any example"
    embed = StubEmbeddingClient(canned={statement_a: list(_SAME_VECTOR), statement_b: list(_SAME_VECTOR)})

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement=statement_a, test="test A", value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i0.id, first, contradiction_was_possible=True,
    )

    opposite = ClaimCandidate(
        statement=statement_b, test="test B", value=StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
        topic="physics",
    )
    result = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i1.id, opposite,
        contradiction_was_possible=False,  # the ineligible pick
    )

    assert result.id == claim.id
    assert result.status is ClaimStatus.CANDIDATE  # NOT contradicted
    evidence = await claim_store.list_evidence(claim.id)
    directions = {e.topic: e.direction.value for e in evidence}
    assert directions["calculus"] == "supports"
    assert directions["physics"] == "contradicts"  # still recorded, as provenance


@pytest.mark.asyncio(loop_scope="session")
async def test_reopen_moves_a_contradicted_claim_back_to_candidate(
    claim_store, learner_id, clean_pool
):
    claim = _claim(learner_id)
    await claim_store.create(claim)
    await claim_store.contradict(claim.id)
    reopened = await claim_store.reopen(claim.id)
    assert reopened.status is ClaimStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_reopen_is_a_noop_on_a_claim_that_is_not_contradicted(
    claim_store, learner_id, clean_pool
):
    claim = _claim(learner_id)
    await claim_store.create(claim)
    result = await claim_store.reopen(claim.id)
    assert result.status is ClaimStatus.CANDIDATE  # unchanged, no error


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_retracts_a_new_claim_founded_on_a_subject_kind_episode(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """The fix for the confabulation problem: a brand-new claim built
    entirely from a subject-kind (uncontestable topic) pick is never
    legitimate teaching-preference evidence -- it must be retracted
    immediately, not left live as a candidate competing for future
    evidence that belongs to a real claim."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidate = ClaimCandidate(
        statement="prefers theoretical rules over concrete examples", test="a confabulated test",
        value=StatedPreferenceLabel.RULE_BEFORE_EXAMPLE, topic="algorithms",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, interaction.id,
        candidate, contradiction_was_possible=False, episode_kind="subject",
    )
    assert claim.status is ClaimStatus.RETRACTED
    assert claim.confidence == 0.5  # the honest prior -- still recomputed before retracting


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_does_not_retract_a_stated_preference_claim(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """A stated preference also carries contradiction_was_possible=False
    (no live option set) but is a genuine self-report, not an
    uncontestable topic pick -- episode_kind=None must NOT trigger
    retraction."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidate = ClaimCandidate(
        statement="always wants brief answers", test="a real test",
        value=StatedPreferenceLabel.PREFERS_BREVITY, topic="general",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, interaction.id,
        candidate, contradiction_was_possible=False, episode_kind=None,
    )
    assert claim.status is ClaimStatus.CANDIDATE  # NOT retracted


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_does_not_retract_an_eligible_approach_kind_claim(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    from probe.models import ApproachAxis, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidate = ClaimCandidate(
        statement="prefers concrete examples first", test="a real test",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, topic="algorithms",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, interaction.id,
        candidate, contradiction_was_possible=True, axis=ApproachAxis.CONCRETE_GENERAL,
        episode_kind="approach",
    )
    assert claim.status is ClaimStatus.CANDIDATE  # NOT retracted


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_unrelated_label_below_similarity_creates_a_new_claim(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """Two genuinely different statements (the stub's default
    near-orthogonal hashing, no canned override) must never match --
    each becomes its own claim."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement="wants concrete examples before the rule", test="test A",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, topic="calculus",
    )
    claim_a = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i0.id, first,
        contradiction_was_possible=True,
    )
    second = ClaimCandidate(
        statement="wants every intermediate step shown, never skipped", test="test B",
        value=StatedPreferenceLabel.WANTS_STEPS_SHOWN, topic="physics",
    )
    claim_b = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i1.id, second,
        contradiction_was_possible=True,
    )
    assert claim_a.id != claim_b.id


# ─────────────────────────── axis-first reconciliation ──────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_matches_by_axis_even_when_statements_are_dissimilar(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """The fragmentation fix: two DIFFERENTLY-WORDED statements on the
    SAME axis reconcile to one claim, even though the stub's default
    (uncanned) hashing makes their embeddings near-orthogonal -- axis
    identity is an exact match, not a similarity judgment."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement="prefers everyday analogies over formal technical breakdowns",
        test="test A", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="biology",
    )
    claim_a = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i0.id, first,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    second = ClaimCandidate(
        statement="the person prefers intuitive explanations using familiar comparisons",
        test="test B", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="chemistry",
    )
    claim_b = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i1.id, second,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    assert claim_b.id == claim_a.id  # same claim despite dissimilar wording
    evidence = await claim_store.list_evidence(claim_a.id)
    assert len(evidence) == 2
    assert {e.topic for e in evidence} == {"biology", "chemistry"}
    assert all(e.direction.value == "supports" for e in evidence)


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_by_axis_single_opposite_value_does_not_close(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """Axis-first matching still decides direction by comparing values,
    but one opposite-value episode (same session) only lowers
    confidence -- same derived-contradiction rule as the similarity-
    based path (`evaluate_contradiction`)."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    first = ClaimCandidate(
        statement="prefers everyday analogies over formal technical breakdowns",
        test="test A", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="biology",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i0.id, first,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    opposite = ClaimCandidate(
        statement="wants the formal technical breakdown, not a loose comparison",
        test="test B", value=StatedPreferenceLabel.NO_ANALOGIES, topic="chemistry",
    )
    result = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i1.id, opposite,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    assert result.id == claim.id
    assert result.status is ClaimStatus.CANDIDATE
    assert result.confidence < claim.confidence


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_by_axis_does_not_reopen_a_contradicted_claim(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """A contradicted claim is terminal -- once sustained cross-session
    reversal has actually earned that status, a later same-axis pick
    starts a fresh claim rather than implicitly reopening it."""
    from probe.models import QuestionAuthor

    session_0 = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_0, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )

    first = ClaimCandidate(
        statement="prefers everyday analogies over formal technical breakdowns",
        test="test A", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="biology",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_0, i0.id, first,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    result = claim
    for n, topic in enumerate(["chemistry", "geology", "astronomy"], start=1):
        session_n = await transcript.create_session(learner_id)
        i_n = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_n, turn_number=0,
            question_text=f"q{n}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{n}",
        )
        opposite = ClaimCandidate(
            statement="wants the formal technical breakdown, not a loose comparison",
            test="test B", value=StatedPreferenceLabel.NO_ANALOGIES, topic=topic,
        )
        result = await reconcile_candidate(
            claim_store, embedding_client, learner_id, session_n, i_n.id, opposite,
            contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
        )
        if result.status is ClaimStatus.CONTRADICTED:
            break

    assert result.id == claim.id
    assert result.status is ClaimStatus.CONTRADICTED

    session_last = await transcript.create_session(learner_id)
    i_last = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_last, turn_number=0,
        question_text="q-last", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a-last",
    )
    third = ClaimCandidate(
        statement="prefers a simple everyday comparison to grasp a new idea",
        test="test C", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="physics",
    )
    fresh = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_last, i_last.id, third,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )
    assert fresh.id != claim.id
    assert fresh.status is ClaimStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_disambiguates_within_a_shared_axis_by_similarity(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """When more than one existing claim already shares an axis (a tie,
    or a pre-fix legacy fragment), similarity picks which one a new
    candidate continues -- axis identity narrows the field, it doesn't
    replace disambiguation entirely."""
    from probe.embeddings import StubEmbeddingClient
    from probe.models import QuestionAuthor

    vec_a = [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2)
    vec_b = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    vec_candidate = [0.9, 0.1] + [0.0] * (EMBEDDING_DIM - 2)  # closer to vec_a

    statement_a = "claim A statement"
    statement_b = "claim B statement"
    statement_c = "a new episode's statement"
    embed = StubEmbeddingClient(canned={
        statement_a: vec_a, statement_b: vec_b, statement_c: vec_candidate,
    })

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )
    i2 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=2,
        question_text="q2", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a2",
    )

    # Seed two DISTINCT claims that already share an axis -- simulating
    # a pre-fix fragment, since the fix itself would normally prevent
    # this from happening going forward.
    claim_a = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i0.id,
        ClaimCandidate(statement=statement_a, test="test A",
                        value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="biology"),
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )
    claim_b = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i1.id,
        ClaimCandidate(statement=statement_b, test="test B",
                        value=StatedPreferenceLabel.OTHER, topic="chemistry"),
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )
    assert claim_a.id != claim_b.id  # OTHER isn't related to WANTS_ANALOGIES -- two claims

    result = await reconcile_candidate(
        claim_store, embed, learner_id, session_id, i2.id,
        ClaimCandidate(statement=statement_c, test="test C",
                        value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="physics"),
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )
    assert result.id == claim_a.id  # nearer embedding wins the disambiguation


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_with_no_axis_falls_back_to_similarity(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    """axis=None (a stated-preference or contradicted_intent trigger
    with no live option set) uses the same pure-similarity matching as
    before axis existed -- unrelated statements never merge."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )
    first = ClaimCandidate(
        statement="prefers everyday analogies over formal technical breakdowns",
        test="test A", value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="biology",
    )
    claim_a = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i0.id, first,
        contradiction_was_possible=True, axis=None,
    )
    second = ClaimCandidate(
        statement="wants every intermediate step shown, never skipped",
        test="test B", value=StatedPreferenceLabel.WANTS_STEPS_SHOWN, topic="physics",
    )
    claim_b = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, i1.id, second,
        contradiction_was_possible=True, axis=None,
    )
    assert claim_a.id != claim_b.id


# ─────────────────────────── duplicate-claim merge ──────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_merge_duplicate_claims_collapses_same_axis_same_value_survivor_is_oldest(
    claim_store, learner_id, transcript, interaction_recorder, pool, clean_pool
):
    """The exact incident this closes: reopening previously-terminal
    claims can put two live, same-axis, same-value claims back on the
    board at once -- a merge pass collapses them, oldest survives, the
    loser's own evidence stays fully visible under its own id (never
    repointed) while a COPY lands on the survivor."""
    from datetime import UTC, datetime, timedelta

    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    session_old = await transcript.create_session(learner_id)
    i_old = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_old, turn_number=0,
        question_text="q_old", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_old",
    )
    session_new = await transcript.create_session(learner_id)
    i_new = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_new, turn_number=0,
        question_text="q_new", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_new",
    )

    now = datetime.now(UTC)
    claim_old = _claim(
        learner_id, statement="the older phrasing", value=StatedPreferenceLabel.WANTS_ANALOGIES,
        created_at=now - timedelta(days=5),
    )
    claim_new = _claim(
        learner_id, statement="the newer, differently-worded phrasing",
        value=StatedPreferenceLabel.WANTS_ANALOGIES, created_at=now,
    )
    await claim_store.create(claim_old)
    await claim_store.create(claim_new)
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_old.id, learner_id=learner_id, interaction_id=i_old.id,
        direction=EvidenceDirection.SUPPORTS, topic="biology", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_old, test_fired=True, contradiction_was_possible=True,
    ))
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_new.id, learner_id=learner_id, interaction_id=i_new.id,
        direction=EvidenceDirection.SUPPORTS, topic="chemistry", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_new, test_fired=True, contradiction_was_possible=True,
    ))

    survivors = await merge_duplicate_claims(pool, claim_store, learner_id)

    assert len(survivors) == 1
    assert survivors[0].id == claim_old.id  # oldest wins, not the fresher phrasing

    refreshed_old = await claim_store.get(claim_old.id)
    refreshed_new = await claim_store.get(claim_new.id)
    assert refreshed_old.status is ClaimStatus.CANDIDATE
    assert refreshed_new.status is ClaimStatus.SUPERSEDED
    assert refreshed_new.superseded_by == claim_old.id

    survivor_evidence = await claim_store.list_evidence(claim_old.id)
    assert len(survivor_evidence) == 2  # its own + a copy of the loser's
    assert {e.topic for e in survivor_evidence} == {"biology", "chemistry"}

    loser_evidence = await claim_store.list_evidence(claim_new.id)
    assert len(loser_evidence) == 1  # untouched -- never repointed, never deleted
    assert loser_evidence[0].topic == "chemistry"
    assert loser_evidence[0].claim_id == claim_new.id


@pytest.mark.asyncio(loop_scope="session")
async def test_merge_flips_direction_for_the_defined_opposite_value(
    claim_store, learner_id, transcript, interaction_recorder, pool, clean_pool
):
    """A loser whose asserted value is the survivor's defined opposite
    is still the same trait/axis -- but a row that SUPPORTED the loser
    actually CONTRADICTS the survivor, and must be copied over flipped,
    not verbatim."""
    from datetime import UTC, datetime, timedelta

    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    session_old = await transcript.create_session(learner_id)
    i_old = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_old, turn_number=0,
        question_text="q_old", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_old",
    )
    session_new = await transcript.create_session(learner_id)
    i_new = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_new, turn_number=0,
        question_text="q_new", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_new",
    )

    now = datetime.now(UTC)
    claim_old = _claim(
        learner_id, statement="wants analogies", value=StatedPreferenceLabel.WANTS_ANALOGIES,
        created_at=now - timedelta(days=5),
    )
    claim_new = _claim(
        learner_id, statement="wants no analogies", value=StatedPreferenceLabel.NO_ANALOGIES,
        created_at=now,
    )
    await claim_store.create(claim_old)
    await claim_store.create(claim_new)
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_old.id, learner_id=learner_id, interaction_id=i_old.id,
        direction=EvidenceDirection.SUPPORTS, topic="biology", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_old, test_fired=True, contradiction_was_possible=True,
    ))
    # This row SUPPORTS claim_new's own asserted value (no_analogies).
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_new.id, learner_id=learner_id, interaction_id=i_new.id,
        direction=EvidenceDirection.SUPPORTS, topic="chemistry", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_new, test_fired=True, contradiction_was_possible=True,
    ))

    survivors = await merge_duplicate_claims(pool, claim_store, learner_id)
    assert survivors[0].id == claim_old.id

    survivor_evidence = await claim_store.list_evidence(claim_old.id)
    copied = next(e for e in survivor_evidence if e.topic == "chemistry")
    assert copied.direction is EvidenceDirection.CONTRADICTS  # flipped, not verbatim


@pytest.mark.asyncio(loop_scope="session")
async def test_merge_leaves_axis_sharing_but_value_incompatible_claims_separate(
    claim_store, learner_id, transcript, interaction_recorder, pool, clean_pool
):
    """Sharing an axis is necessary but not sufficient -- a6c9cdb3-like
    claims (same axis, an unrelated value with no defined opposite
    relationship) are genuinely different traits and must not merge."""
    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )

    claim_a = _claim(learner_id, statement="wants analogies", value=StatedPreferenceLabel.WANTS_ANALOGIES)
    claim_b = _claim(
        learner_id, statement="wants concrete descriptions",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
    )
    await claim_store.create(claim_a)
    await claim_store.create(claim_b)
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_a.id, learner_id=learner_id, interaction_id=i0.id,
        direction=EvidenceDirection.SUPPORTS, topic="biology", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_id, test_fired=True, contradiction_was_possible=True,
    ))
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_b.id, learner_id=learner_id, interaction_id=i1.id,
        direction=EvidenceDirection.SUPPORTS, topic="chemistry", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_id, test_fired=True, contradiction_was_possible=True,
    ))

    survivors = await merge_duplicate_claims(pool, claim_store, learner_id)
    assert survivors == []
    assert (await claim_store.get(claim_a.id)).status is ClaimStatus.CANDIDATE
    assert (await claim_store.get(claim_b.id)).status is ClaimStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_search_similar_and_find_by_axis_exclude_superseded_claims(
    claim_store, learner_id, clean_pool
):
    claim = _claim(learner_id, embedding=list(_SAME_VECTOR))
    survivor = _claim(learner_id, statement="the survivor")
    await claim_store.create(claim)
    await claim_store.create(survivor)
    await claim_store.supersede(claim.id, survivor.id)

    matches = await claim_store.search_similar(learner_id, list(_SAME_VECTOR))
    assert all(found.id != claim.id for found, _ in matches)

    axis_claims = await claim_store.find_by_axis(learner_id, ApproachAxis.ANALOGY_FORMAL)
    assert all(c.id != claim.id for c in axis_claims)


@pytest.mark.asyncio(loop_scope="session")
async def test_refresh_is_a_noop_on_a_superseded_claim(claim_store, learner_id, clean_pool):
    claim = _claim(learner_id)
    survivor = _claim(learner_id, statement="the survivor")
    await claim_store.create(claim)
    await claim_store.create(survivor)
    await claim_store.supersede(claim.id, survivor.id)
    refreshed = await claim_store.refresh(claim.id)
    assert refreshed.status is ClaimStatus.SUPERSEDED


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_merges_reopened_fragments_before_matching(
    claim_store, learner_id, transcript, interaction_recorder, pool, clean_pool
):
    """The direct regression test for the live incident: two live,
    same-axis, same-value claims (as `reopen` can produce) must collapse
    into one BEFORE a new episode's evidence is attached -- not compete
    in an embedding-similarity lottery. Embeddings are rigged so the
    OLD (pre-merge) similarity tiebreak would have picked the NEWER
    claim; the merge-first fix must still land the new evidence on the
    OLDER survivor."""
    from datetime import UTC, datetime, timedelta

    from probe.embeddings import StubEmbeddingClient
    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    vec_old = [1.0, 0.0] + [0.0] * (EMBEDDING_DIM - 2)
    vec_new = [0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)
    vec_candidate = [0.05, 0.95] + [0.0] * (EMBEDDING_DIM - 2)  # closer to vec_new than vec_old

    statement_old, statement_new, statement_candidate = "old phrasing", "new phrasing", "candidate phrasing"
    embed = StubEmbeddingClient(canned={
        statement_old: vec_old, statement_new: vec_new, statement_candidate: vec_candidate,
    })

    session_old = await transcript.create_session(learner_id)
    i_old = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_old, turn_number=0,
        question_text="q_old", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_old",
    )
    session_new = await transcript.create_session(learner_id)
    i_new = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_new, turn_number=0,
        question_text="q_new", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_new",
    )
    session_candidate = await transcript.create_session(learner_id)
    i_candidate = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_candidate, turn_number=0,
        question_text="q_cand", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a_cand",
    )

    now = datetime.now(UTC)
    claim_old = _claim(
        learner_id, statement=statement_old, value=StatedPreferenceLabel.WANTS_ANALOGIES,
        embedding=vec_old, created_at=now - timedelta(days=5),
    )
    claim_new = _claim(
        learner_id, statement=statement_new, value=StatedPreferenceLabel.WANTS_ANALOGIES,
        embedding=vec_new, created_at=now,
    )
    await claim_store.create(claim_old)
    await claim_store.create(claim_new)
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_old.id, learner_id=learner_id, interaction_id=i_old.id,
        direction=EvidenceDirection.SUPPORTS, topic="biology", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_old, test_fired=True, contradiction_was_possible=True,
    ))
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim_new.id, learner_id=learner_id, interaction_id=i_new.id,
        direction=EvidenceDirection.SUPPORTS, topic="chemistry", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_new, test_fired=True, contradiction_was_possible=True,
    ))

    candidate = ClaimCandidate(
        statement=statement_candidate, test="test C",
        value=StatedPreferenceLabel.WANTS_ANALOGIES, topic="physics",
    )
    result = await reconcile_candidate(
        claim_store, embed, learner_id, session_candidate, i_candidate.id, candidate,
        contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    assert result.id == claim_old.id  # the older survivor, not the embedding-nearer claim_new
    refreshed_new = await claim_store.get(claim_new.id)
    assert refreshed_new.status is ClaimStatus.SUPERSEDED
    assert refreshed_new.superseded_by == claim_old.id
    survivor_evidence = await claim_store.list_evidence(claim_old.id)
    assert {e.topic for e in survivor_evidence} == {"biology", "chemistry", "physics"}


# ─────────────────────────── contaminated-evidence provenance ────────


@pytest.mark.asyncio(loop_scope="session")
async def test_mark_evidence_contaminated_sets_the_note_without_changing_anything_else(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """The one sanctioned mutation of an existing claim_evidence row --
    direction/topic/axis/eligibility all stay exactly as recorded; only
    the note is new."""
    from probe.models import ClaimEvidence, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    claim = _claim(learner_id)
    await claim_store.create(claim)
    evidence = await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
        direction=EvidenceDirection.CONTRADICTS, topic="algorithm execution and tracing",
        axis=ApproachAxis.ANALOGY_FORMAL, session_id=session_id,
        test_fired=True, contradiction_was_possible=True,
    ))
    assert evidence.provenance_note is None

    note = "keyword-collision: simulator intended analogy side, matcher selected formal option"
    updated = await claim_store.mark_evidence_contaminated(evidence.id, note)
    assert updated.provenance_note == note
    assert updated.direction is EvidenceDirection.CONTRADICTS
    assert updated.topic == "algorithm execution and tracing"
    assert updated.test_fired is True
    assert updated.contradiction_was_possible is True  # still counts in production confidence

    fetched = await claim_store.list_evidence(claim.id)
    assert fetched[0].provenance_note == note


# ─────────────────────────── claim statements (restatement) ────────


@pytest.mark.asyncio(loop_scope="session")
async def test_reconcile_writes_a_founding_statement_row_on_claim_creation(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidate = ClaimCandidate(
        statement="prefers brief answers", test="a real test",
        value=StatedPreferenceLabel.PREFERS_BREVITY, topic="calculus",
    )
    claim = await reconcile_candidate(
        claim_store, embedding_client, learner_id, session_id, interaction.id,
        candidate, contradiction_was_possible=True,
    )
    current = await claim_store.get_current_statement(claim.id)
    assert current is not None
    assert current.statement == claim.statement
    assert current.derived_from_evidence_count == 1

    history = await claim_store.list_statements(claim.id)
    assert len(history) == 1
    assert history[0].id == current.id


@pytest.mark.asyncio(loop_scope="session")
async def test_append_statement_is_append_only_and_current_resolves_to_latest(
    claim_store, learner_id, clean_pool
):
    from probe.models import ClaimStatementRecord

    claim = _claim(learner_id)
    await claim_store.create(claim)
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement="founding wording",
        derived_from_evidence_count=1, generator_version="extract-claim-v1",
    ))
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement="restated, more general wording",
        derived_from_evidence_count=6, generator_version="restate-claim-v1",
    ))

    current = await claim_store.get_current_statement(claim.id)
    assert current.statement == "restated, more general wording"

    history = await claim_store.list_statements(claim.id)
    assert [h.statement for h in history] == ["founding wording", "restated, more general wording"]


@pytest.mark.asyncio(loop_scope="session")
async def test_maybe_restate_claims_skips_below_the_evidence_threshold(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """Two new rows since the last statement, threshold is 4 -- no
    restatement, and definitely no LLM call (a canned response that
    would fail the test if invoked proves this)."""
    from probe.models import ClaimEvidence, ClaimStatementRecord, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    claim = _claim(learner_id, statement="founded on biology")
    await claim_store.create(claim)
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement=claim.statement,
        derived_from_evidence_count=1, generator_version="extract-claim-v1",
    ))
    for i, topic in enumerate(["biology", "computer science"]):
        interaction = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=i,
            question_text=f"q{i}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{i}",
        )
        await claim_store.append_evidence(ClaimEvidence(
            claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
            direction=EvidenceDirection.SUPPORTS, topic=topic, axis=ApproachAxis.ANALOGY_FORMAL,
            session_id=session_id, test_fired=True, contradiction_was_possible=True,
        ))

    llm = StubLLMClient(canned={"RESTATE:CLAIM": "SHOULD NOT BE CALLED"})
    restated = await maybe_restate_claims(claim_store, llm, learner_id)
    assert restated == []
    assert (await claim_store.get_current_statement(claim.id)).statement == "founded on biology"


@pytest.mark.asyncio(loop_scope="session")
async def test_maybe_restate_claims_skips_when_evidence_never_left_the_founding_domain(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """Enough new rows, but every one is still the founding topic --
    nothing to generalize away from yet, so the existing wording stays."""
    from probe.models import ClaimEvidence, ClaimStatementRecord, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    claim = _claim(learner_id, statement="founded on biology")
    await claim_store.create(claim)
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement=claim.statement,
        derived_from_evidence_count=1, generator_version="extract-claim-v1",
    ))
    for i in range(5):
        interaction = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=i,
            question_text=f"q{i}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{i}",
        )
        await claim_store.append_evidence(ClaimEvidence(
            claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
            direction=EvidenceDirection.SUPPORTS, topic="biology", axis=ApproachAxis.ANALOGY_FORMAL,
            session_id=session_id, test_fired=True, contradiction_was_possible=True,
        ))

    llm = StubLLMClient(canned={"RESTATE:CLAIM": "SHOULD NOT BE CALLED"})
    restated = await maybe_restate_claims(claim_store, llm, learner_id)
    assert restated == []


@pytest.mark.asyncio(loop_scope="session")
async def test_maybe_restate_claims_restates_once_both_conditions_hold(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """Enough new evidence, spanning topics outside the founding
    domain -- restates via the stubbed LLM call and appends a new
    claim_statements row without touching Claim.statement."""
    import json as _json

    from probe.models import ClaimEvidence, ClaimStatementRecord, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    claim = _claim(learner_id, statement="prefers analogies for biological explanations")
    await claim_store.create(claim)
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement=claim.statement,
        derived_from_evidence_count=1, generator_version="extract-claim-v1",
    ))
    topics = ["biology", "computer science", "algorithms", "concept explanation", "explanation style"]
    for i, topic in enumerate(topics):
        interaction = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=i,
            question_text=f"q{i}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{i}",
        )
        await claim_store.append_evidence(ClaimEvidence(
            claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
            direction=EvidenceDirection.SUPPORTS, topic=topic, axis=ApproachAxis.ANALOGY_FORMAL,
            session_id=session_id, test_fired=True, contradiction_was_possible=True,
        ))

    new_statement = "prefers everyday analogies over formal technical explanations in general"
    llm = StubLLMClient(canned={
        "RESTATE:CLAIM": _json.dumps({"statement": new_statement}),
    })
    node_calls = []

    async def _on_node_call(node_name, input_json, output_json):
        node_calls.append((node_name, input_json, output_json))

    restated = await maybe_restate_claims(
        claim_store, llm, learner_id, on_node_call=_on_node_call,
    )
    assert len(restated) == 1
    assert restated[0].id == claim.id

    current = await claim_store.get_current_statement(claim.id)
    assert current.statement == new_statement
    assert current.derived_from_evidence_count == 5
    assert current.generator_version == ClaimRestatementConfig().generator_version

    history = await claim_store.list_statements(claim.id)
    assert history[0].statement == "prefers analogies for biological explanations"  # founding, preserved
    assert history[-1].statement == new_statement

    fetched_claim = await claim_store.get(claim.id)
    assert fetched_claim.statement == "prefers analogies for biological explanations"  # never touched

    assert len(node_calls) == 1
    assert node_calls[0][0] == "ClaimRestater"


@pytest.mark.asyncio(loop_scope="session")
async def test_maybe_restate_claims_excludes_contaminated_evidence_from_the_llm_input(
    claim_store, learner_id, transcript, interaction_recorder, clean_pool
):
    """The real-data finding this closes: given a flagged row, the
    restater must not see it at all -- not merely be told to discount
    it. A canned response asserts on the actual prompt content (via
    StubLLMClient's own recorded `.prompts`) to prove the contaminated
    topic never reached the LLM, and the domain-spread gate is judged
    against clean evidence only."""
    from probe.models import ClaimEvidence, ClaimStatementRecord, EvidenceDirection, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    claim = _claim(learner_id, statement="founded on biology")
    await claim_store.create(claim)
    await claim_store.append_statement(ClaimStatementRecord(
        claim_id=claim.id, statement=claim.statement,
        derived_from_evidence_count=1, generator_version="extract-claim-v1",
    ))
    rows = [
        ("biology", EvidenceDirection.SUPPORTS, None),
        ("computer science", EvidenceDirection.SUPPORTS, None),
        ("algorithms", EvidenceDirection.SUPPORTS, None),
        # The ONLY row outside the founding domain is contaminated --
        # clean evidence alone never leaves "biology", so this must NOT
        # qualify for restatement at all.
        ("quicksort tracing", EvidenceDirection.CONTRADICTS, "keyword-collision: harness bug"),
    ]
    for i, (topic, direction, note) in enumerate(rows):
        interaction = await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=i,
            question_text=f"q{i}", question_author=QuestionAuthor.LEARNER,
            originating_question=None, did_branch=False, response_text=f"a{i}",
        )
        await claim_store.append_evidence(ClaimEvidence(
            claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
            direction=direction, topic=topic, axis=ApproachAxis.ANALOGY_FORMAL,
            session_id=session_id, test_fired=True, contradiction_was_possible=True,
            provenance_note=note,
        ))

    llm = StubLLMClient(canned={"RESTATE:CLAIM": "SHOULD NOT BE CALLED"})
    restated = await maybe_restate_claims(claim_store, llm, learner_id)
    assert restated == []  # clean evidence never left "biology" -- nothing to generalize away from

    # Now add one more genuinely-clean, out-of-domain row so the gate
    # opens, and confirm the contaminated topic never reaches the prompt.
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=len(rows),
        question_text="q-extra", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a-extra",
    )
    await claim_store.append_evidence(ClaimEvidence(
        claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
        direction=EvidenceDirection.SUPPORTS, topic="chemistry", axis=ApproachAxis.ANALOGY_FORMAL,
        session_id=session_id, test_fired=True, contradiction_was_possible=True,
    ))
    import json as _json
    new_statement = "prefers analogies across technical subjects generally"
    llm2 = StubLLMClient(canned={"RESTATE:CLAIM": _json.dumps({"statement": new_statement})})
    restated2 = await maybe_restate_claims(claim_store, llm2, learner_id)
    assert len(restated2) == 1
    assert "quicksort tracing" not in llm2.prompts[0]
    assert "chemistry" in llm2.prompts[0]
