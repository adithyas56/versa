"""InteractionStore + InteractionRecorder — the append-only write path
(migration 034). Covers entry_state computation for every branch,
recent_similar_count counting, prior_turn_outcome best-effort lookup,
and the deferred/cross-session-catch-up rules.
"""

import pytest

from probe.models import QuestionAuthor, TurnOutcome, TurnOutcomeLabel


@pytest.mark.asyncio(loop_scope="session")
async def test_cold_open_on_the_first_turn_of_a_session(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="an answer",
    )
    assert i0.entry_state.value == "cold_open"
    assert i0.recent_similar_count == 0
    assert i0.prior_turn_outcome.value == "unknown"


@pytest.mark.asyncio(loop_scope="session")
async def test_continuing_when_immediately_prior_question_is_similar(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer 1",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer 2",
    )
    assert i1.entry_state.value == "continuing"
    assert i1.recent_similar_count == 1  # count BEFORE this row
    assert i1.prev_question_sim == pytest.approx(1.0)


@pytest.mark.asyncio(loop_scope="session")
async def test_resolution_when_previous_turn_offered_branches(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="can you help with derivatives?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=True, response_text=None,
    )
    resolution = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="calculus derivatives", question_author=QuestionAuthor.SYSTEM_OPTION,
        originating_question="can you help with derivatives?", did_branch=False,
        response_text="calculus derivatives are...",
    )
    assert resolution.entry_state.value == "resolution"
    assert resolution.originating_question == "can you help with derivatives?"


@pytest.mark.asyncio(loop_scope="session")
async def test_an_offer_turn_gets_an_immediate_structural_deferred_outcome(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    offer = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="can you help with derivatives?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=True, response_text=None,
    )
    assert interaction_recorder.last_classification_target is None
    outcome = await turn_outcome_store.get_latest(offer.id)
    assert outcome is not None, "an offer turn must not be left silent"
    assert outcome.outcome is TurnOutcomeLabel.DEFERRED
    assert outcome.classifier_version == "structural-no-response"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resolved_turn_becomes_the_next_turns_classification_target(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    resolved = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="an answer",
    )
    assert interaction_recorder.last_classification_target is None

    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="ok what about integrals?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="integrals are...",
    )
    assert interaction_recorder.last_classification_target is not None
    assert interaction_recorder.last_classification_target.id == resolved.id


@pytest.mark.asyncio(loop_scope="session")
async def test_session_end_marks_a_trailing_resolved_turn_deferred(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    trailing = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="an answer",
    )
    assert await turn_outcome_store.get_latest(trailing.id) is None

    await interaction_recorder.mark_trailing_interaction_deferred(session_id)
    outcome = await turn_outcome_store.get_latest(trailing.id)
    assert outcome.outcome is TurnOutcomeLabel.DEFERRED
    assert outcome.classifier_version == "structural-session-end"


@pytest.mark.asyncio(loop_scope="session")
async def test_session_end_hook_never_overwrites_a_real_classification(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    trailing = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await turn_outcome_store.append(
        TurnOutcome(
            interaction_id=trailing.id, learner_id=learner_id,
            outcome=TurnOutcomeLabel.MATCHED, confidence=0.95, classifier_version="real-v1",
        )
    )
    await interaction_recorder.mark_trailing_interaction_deferred(session_id)
    outcome = await turn_outcome_store.get_latest(trailing.id)
    assert outcome.classifier_version == "real-v1", "must not clobber an existing classification"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_cold_open_in_a_new_session_catches_up_on_the_prior_sessions_trailing_turn(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session1 = await transcript.create_session(learner_id)
    trailing = await interaction_recorder.record(
        learner_id=learner_id, session_id=session1, turn_number=0,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )
    await interaction_recorder.mark_trailing_interaction_deferred(session1)

    session2 = await transcript.create_session(learner_id)
    returned = await interaction_recorder.record(
        learner_id=learner_id, session_id=session2, turn_number=0,
        question_text="back again, what about integrals?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a2",
    )
    assert returned.entry_state.value == "cold_open"
    assert interaction_recorder.last_classification_target is not None
    assert interaction_recorder.last_classification_target.id == trailing.id


@pytest.mark.asyncio(loop_scope="session")
async def test_returning_after_gap_when_a_match_exists_but_not_immediately_prior(
    interaction_recorder, transcript, learner_id, clean_pool
):
    """No time-based gap any more -- returning_after_gap is a WINDOW
    POSITION fact: something in the learner's recent-questions window
    matches, just not the immediately preceding question (that's what
    last_similar_turn_gap being non-None but prev_question_sim below
    threshold means; see interactions.py's _compute_entry_state)."""
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="an old answer",
    )
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="unrelated question", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="unrelated answer",
    )
    returning = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=2,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer again",
    )
    assert returning.entry_state.value == "returning_after_gap"
    assert returning.last_similar_turn_gap == 2
    assert returning.recent_similar_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_stuck_repeat_when_the_immediately_prior_match_was_contradicted(
    interaction_recorder, turn_outcome_store, transcript, learner_id, clean_pool
):
    """stuck_repeat is CONTINUING (immediately-prior question similar)
    PLUS that immediately-prior turn's outcome was contradicted -- not
    reachable through a returning_after_gap-style non-adjacent match,
    per the precedence order in interactions.py's _compute_entry_state."""
    session_id = await transcript.create_session(learner_id)
    first = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer 1",
    )
    await turn_outcome_store.append(
        TurnOutcome(
            interaction_id=first.id, learner_id=learner_id,
            outcome=TurnOutcomeLabel.CONTRADICTED_INTENT, confidence=0.9,
            classifier_version="test-v1",
        )
    )
    stuck = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer 2",
    )
    assert stuck.entry_state.value == "stuck_repeat"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_new_topic_mid_session_is_a_topic_switch_not_continuing(
    interaction_recorder, transcript, learner_id, clean_pool
):
    """The learner abandoned or completed one thing and moved to
    another -- must not be merged into CONTINUING, the single most
    common state in this table."""
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="what is a derivative?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="answer 1",
    )
    switched = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="totally unrelated subject about the French Revolution",
        question_author=QuestionAuthor.LEARNER, originating_question=None,
        did_branch=False, response_text="answer 2",
    )
    assert switched.entry_state.value == "topic_switch"


@pytest.mark.asyncio(loop_scope="session")
async def test_prior_turn_outcome_is_unknown_until_classified(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )
    assert i1.prior_turn_outcome.value == "unknown"


@pytest.mark.asyncio(loop_scope="session")
async def test_prior_turn_outcome_maps_the_classified_label_down(
    interaction_recorder, turn_outcome_store, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    i0 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q0", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a0",
    )
    await turn_outcome_store.append(
        TurnOutcome(
            interaction_id=i0.id, learner_id=learner_id,
            outcome=TurnOutcomeLabel.CONTRADICTED_INTENT, confidence=0.9,
            classifier_version="test-v1",
        )
    )
    i1 = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="q1", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a1",
    )
    assert i1.prior_turn_outcome.value == "contradicted"


@pytest.mark.asyncio(loop_scope="session")
async def test_prior_turn_outcome_deferred_is_not_collapsed_into_unknown(
    interaction_recorder, transcript, learner_id, clean_pool
):
    """UNKNOWN means "we don't know"; DEFERRED means "there is nothing
    to know yet, by construction" (the prior turn was an options-offered
    row). These must never be conflated -- see InteractionPriorOutcome's
    own docstring."""
    session_id = await transcript.create_session(learner_id)
    # An options-offered turn: its own outcome is written IMMEDIATELY
    # as structural DEFERRED (see interactions.py's record()).
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="can you help me with derivatives?", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=True, response_text=None,
    )
    following = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="calculus derivatives", question_author=QuestionAuthor.SYSTEM_OPTION,
        originating_question="can you help me with derivatives?", did_branch=False,
        response_text="an answer",
    )
    assert following.prior_turn_outcome.value == "deferred"
    assert following.prior_turn_outcome.value != "unknown"


@pytest.mark.asyncio(loop_scope="session")
async def test_embedding_a_system_option_turn_uses_originating_question(
    interaction_recorder, embedding_client, transcript, learner_id, clean_pool
):
    """Regression guard for the exact failure this feature's whole
    question_author/originating_question split exists to prevent: a
    click-resolution turn must never have its question_embedding
    computed from the option's own system-authored text."""
    session_id = await transcript.create_session(learner_id)
    original = "can you help me with derivatives?"
    option_text = "let's dive into calculus derivatives"

    embedding_client.canned[original] = [1.0] + [0.0] * 767
    embedding_client.canned[option_text] = [0.0, 1.0] + [0.0] * 766

    resolution = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text=option_text, question_author=QuestionAuthor.SYSTEM_OPTION,
        originating_question=original, did_branch=False, response_text="an answer",
    )
    assert resolution.question_embedding == [1.0] + [0.0] * 767
    assert original in embedding_client.texts
    assert option_text not in embedding_client.texts


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_originating_question_on_a_system_option_turn_is_a_bug_not_silent(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    with pytest.raises(AssertionError):
        await interaction_recorder.record(
            learner_id=learner_id, session_id=session_id, turn_number=0,
            question_text="option text", question_author=QuestionAuthor.SYSTEM_OPTION,
            originating_question=None, did_branch=False, response_text="a",
        )
