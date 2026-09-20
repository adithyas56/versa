"""The reference-resolution memory: ReferenceBindingStore's append-only
read/write behavior, reference_bindings.py's pure exact-match/confidence
formatting, and the full loop wiring -- a resolution gets classified off
the critical path, and a LATER turn using the same phrase gets it
injected into both AssessAndBranch (to suppress re-branching) and
FinalAnswer (as background), above learner_history_block.
"""

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from probe.disambiguate import DisambiguationStore
from probe.history_block import HistoryBlockConfig
from probe.interactions import InteractionRecorder
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import QuestionAuthor, ReferenceBinding
from probe.reference_bindings import (
    ReferenceBindingConfig,
    ReferenceBindingMatch,
    match_reference_bindings,
    render_reference_bindings_block,
)
from probe.retrieval_config import RetrievalConfig

_NOT_AMBIGUOUS = json.dumps({"needs_branches": False, "branches": []})
_NO_RESOLUTION = json.dumps(
    {"resolved": False, "reference_text": None, "resolved_to": None, "confidence": 0.0}
)


def _resolution(reference_text: str, resolved_to: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "resolved": True,
            "reference_text": reference_text,
            "resolved_to": resolved_to,
            "confidence": confidence,
        }
    )


# ─────────────────────────── pure match/render ──────────────────────


def _binding(
    reference_text: str = "the usual",
    resolved_to: str = "worked examples with real numbers before the general form",
    confirmation_count: int = 1,
    last_confirmed_at: datetime | None = None,
) -> ReferenceBinding:
    return ReferenceBinding(
        learner_id=uuid4(),
        reference_text=reference_text,
        resolved_to=resolved_to,
        evidence_interaction_id=uuid4(),
        confirmation_count=confirmation_count,
        last_confirmed_at=last_confirmed_at or datetime.now(UTC),
        classifier_version="test-v1",
    )


def test_match_is_exact_case_insensitive_substring():
    binding = _binding(reference_text="the usual")
    matches = match_reference_bindings("can you give me the USUAL approach?", [binding])
    assert len(matches) == 1
    assert matches[0].binding is binding


def test_match_finds_nothing_when_phrase_absent():
    binding = _binding(reference_text="the usual")
    assert match_reference_bindings("what's the chain rule?", [binding]) == []


def test_match_drops_binding_below_min_confidence():
    stale = _binding(
        confirmation_count=1,
        last_confirmed_at=datetime.now(UTC) - timedelta(days=365),
    )
    matches = match_reference_bindings(
        "the usual please", [stale], config=ReferenceBindingConfig(min_confidence=0.4)
    )
    assert matches == []


def test_match_keeps_a_single_recent_confirmation():
    fresh = _binding(confirmation_count=1, last_confirmed_at=datetime.now(UTC))
    matches = match_reference_bindings("the usual please", [fresh])
    assert len(matches) == 1
    assert matches[0].confidence > 0.4


def test_confidence_decays_with_age():
    fresh = _binding(last_confirmed_at=datetime.now(UTC))
    old = _binding(last_confirmed_at=datetime.now(UTC) - timedelta(days=60))
    fresh_conf = match_reference_bindings("the usual", [fresh])[0].confidence
    old_matches = match_reference_bindings(
        "the usual", [old], config=ReferenceBindingConfig(min_confidence=0.0)
    )
    assert old_matches[0].confidence < fresh_conf


def test_confidence_grows_with_confirmation_count():
    once = _binding(confirmation_count=1)
    thrice = _binding(confirmation_count=3)
    once_conf = match_reference_bindings("the usual", [once])[0].confidence
    thrice_conf = match_reference_bindings("the usual", [thrice])[0].confidence
    assert thrice_conf > once_conf


def test_render_empty_matches_returns_empty_string():
    assert render_reference_bindings_block([]) == ""


def test_render_includes_phrase_and_meaning_framed_as_prior_not_fact():
    binding = _binding(reference_text="the usual", resolved_to="worked examples first")
    block = render_reference_bindings_block([ReferenceBindingMatch(binding=binding, confidence=0.9)])
    assert "KNOWN REFERENCES FOR THIS LEARNER" in block
    assert '"the usual" has previously meant: worked examples first' in block
    assert "prior meanings, not current facts" in block
    assert "Never rewrite or restate the learner's own question" in block


# ─────────────────────────── ReferenceBindingStore ──────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_record_resolution_starts_at_confirmation_count_one(
    reference_binding_store, interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    binding = await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples first", evidence_interaction_id=interaction.id,
        classifier_version="test-v1",
    )
    assert binding.confirmation_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_record_resolution_reconfirms_the_same_meaning(
    reference_binding_store, interaction_recorder, transcript, learner_id, clean_pool
):
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
    await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples first", evidence_interaction_id=i0.id,
        classifier_version="test-v1",
    )
    second = await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples first", evidence_interaction_id=i1.id,
        classifier_version="test-v1",
    )
    assert second.confirmation_count == 2
    latest = await reference_binding_store.get_latest(learner_id, "the usual")
    assert latest.id == second.id


@pytest.mark.asyncio(loop_scope="session")
async def test_record_resolution_resets_count_on_a_changed_meaning(
    reference_binding_store, interaction_recorder, transcript, learner_id, clean_pool
):
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
    first = await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples first", evidence_interaction_id=i0.id,
        classifier_version="test-v1",
    )
    second = await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="a completely different approach", evidence_interaction_id=i1.id,
        classifier_version="test-v1",
    )
    assert second.confirmation_count == 1
    assert second.resolved_to == "a completely different approach"
    # The prior row is never touched -- it stays on record as what the
    # phrase used to mean (CLAUDE.md append-only family).
    async with clean_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM reference_bindings WHERE learner_id = $1 "
            "AND reference_text = 'the usual'",
            learner_id,
        )
    assert count == 2
    stored_first = await conn_fetchrow_by_id(clean_pool, first.id)
    assert stored_first["resolved_to"] == "worked examples first"


async def conn_fetchrow_by_id(pool, binding_id):
    async with pool.acquire() as conn:
        return await conn.fetchrow("SELECT * FROM reference_bindings WHERE id = $1", binding_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_list_latest_for_learner_returns_one_row_per_phrase(
    reference_binding_store, interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples first", evidence_interaction_id=interaction.id,
        classifier_version="test-v1",
    )
    await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="my project",
        resolved_to="the calculus homework due Friday", evidence_interaction_id=interaction.id,
        classifier_version="test-v1",
    )
    bindings = await reference_binding_store.list_latest_for_learner(learner_id)
    assert {b.reference_text for b in bindings} == {"the usual", "my project"}


# ─────────────────────────── loop wiring ──────────────────────


def _make_loop(
    transcript, node_calls, disambiguation_store, interaction_store,
    turn_outcome_store, interaction_option_store, interaction_abstract_store,
    prediction_store, stated_preference_store, reference_binding_store,
    embedding_client, pool, llm=None, diagnostics_store=None,
):
    recorder = InteractionRecorder(
        interaction_store, turn_outcome_store, embedding_client,
        same_subject_threshold=RetrievalConfig().same_subject_threshold,
    )
    return SessionLoop(
        transcript=transcript,
        node_calls=node_calls,
        llm=llm or StubLLMClient(),
        diagnostics_store=diagnostics_store,
        disambiguation_store=disambiguation_store,
        embedding_client=embedding_client,
        interaction_recorder=recorder,
        interaction_option_store=interaction_option_store,
        interaction_abstract_store=interaction_abstract_store,
        turn_outcome_store=turn_outcome_store,
        prediction_store=prediction_store,
        retrieval_pool=pool,
        stated_preference_store=stated_preference_store,
        reference_binding_store=reference_binding_store,
        history_block_config=HistoryBlockConfig(enabled=True),
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resolved_reference_is_classified_and_written(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    reference_binding_store, embedding_client, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:REFERENCE_RESOLUTION": _resolution(
                "the usual", "worked examples with real numbers first"
            ),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, reference_binding_store,
        embedding_client, clean_pool, llm=llm, diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "can you give me the usual approach?")
    await loop.wait_for_background_tasks()
    latest = await reference_binding_store.get_latest(learner_id, "the usual")
    assert latest is not None
    assert latest.resolved_to == "worked examples with real numbers first"


@pytest.mark.asyncio(loop_scope="session")
async def test_an_unresolved_turn_writes_nothing(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    reference_binding_store, embedding_client, diagnostics_store,
):
    """Deliberately sparse, unlike StatedPreference/TurnOutcome: a
    resolved=False classification writes NOTHING, not a row recording
    the negative."""
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:REFERENCE_RESOLUTION": _NO_RESOLUTION,
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, reference_binding_store,
        embedding_client, clean_pool, llm=llm, diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "what's the chain rule?")
    await loop.wait_for_background_tasks()
    async with clean_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM reference_bindings WHERE learner_id = $1", learner_id
        )
    assert count == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_click_resolution_turn_can_also_fire_the_classifier(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    reference_binding_store, embedding_client, diagnostics_store,
):
    """Unlike ClassifyStatedPreference (learner-turns-only), a branch
    selection is explicitly one of the three ways a reference gets
    settled -- see ReferenceBinding's own docstring."""
    import re

    two_branches = json.dumps(
        {
            "needs_branches": True,
            "branches": [
                {"statement": "wants the worked-example approach"},
                {"statement": "wants the formal-proof approach"},
            ],
        }
    )

    def _options_for_branches(prompt: str) -> str:
        ids = re.findall(r"id=([0-9a-f-]{36})", prompt)
        return json.dumps(
            {
                "kind": "subject",
                "axis": None,
                "options": [{"branch_id": bid, "text": f"option {i}"} for i, bid in enumerate(ids)],
            }
        )

    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": two_branches,
            "DISAMBIGUATE:OPTIONS": _options_for_branches,
            "FINAL:ANSWER": "resolved answer",
            "CLASSIFY:REFERENCE_RESOLUTION": _resolution(
                "the usual", "the worked-example approach"
            ),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, reference_binding_store,
        embedding_client, clean_pool, llm=llm, diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "can you give me the usual?")
    disamb = DisambiguationStore(clean_pool)
    latest_turn = await disamb.get_latest_turn(session_id)
    options = await disamb.list_options_for_turn(latest_turn.id)
    clicked = options[0]

    await loop.handle_turn(session_id, 1, clicked.text, clicked.id)
    await loop.wait_for_background_tasks()

    latest = await reference_binding_store.get_latest(learner_id, "the usual")
    assert latest is not None
    assert latest.resolved_to == "the worked-example approach"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_later_turn_gets_the_binding_injected_above_history_block(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    reference_binding_store, embedding_client, diagnostics_store,
):
    # Seed a binding directly (isolates this test from the classifier's
    # own behavior, already covered above) via a real prior interaction.
    seed_session = await transcript.create_session(learner_id)
    seed_recorder = InteractionRecorder(
        interaction_store, turn_outcome_store, embedding_client,
        same_subject_threshold=RetrievalConfig().same_subject_threshold,
    )
    seed_interaction = await seed_recorder.record(
        learner_id=learner_id, session_id=seed_session, turn_number=0,
        question_text="seed", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="seed answer",
    )
    await reference_binding_store.record_resolution(
        learner_id=learner_id, reference_text="the usual",
        resolved_to="worked examples with real numbers first",
        evidence_interaction_id=seed_interaction.id, classifier_version="test-v1",
    )

    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:REFERENCE_RESOLUTION": _NO_RESOLUTION,
        }
    )
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, reference_binding_store,
        embedding_client, clean_pool, llm=llm, diagnostics_store=diagnostics_store,
    )
    live_session = await transcript.create_session(learner_id)
    await loop.handle_turn(live_session, 0, "can you show me the usual for this one?")

    call = await node_calls.get_call_for_turn(live_session, 0, "FinalAnswer")
    ref_block = call.input_json["reference_bindings_block"]
    assert "KNOWN REFERENCES FOR THIS LEARNER" in ref_block
    assert "worked examples with real numbers first" in ref_block

    # node_calls captures the raw kwarg passed to AssessAndBranch.run()
    # -- the "don't branch on this" wrapper text lives inside
    # _assess_prompt's assembled prompt, not in the hint value itself.
    assess_call = await node_calls.get_call_for_turn(live_session, 0, "AssessAndBranch")
    assert "the usual" in assess_call.input_json["reference_binding_hint"]
    assert "KNOWN REFERENCES FOR THIS LEARNER" in assess_call.input_json["reference_binding_hint"]
