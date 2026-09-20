"""SessionLoop wiring for the interaction pipeline (interactions.py,
retrieval.py, interaction_nodes.py) on top of SessionMode.MINIMAL_BRANCH.
Raw store behavior is covered by test_interactions_store.py etc.; this
exercises the loop-level orchestration across all 5 finish-points.

The negative case matters as much as the positive ones: with no
interaction pipeline configured, every existing test in this suite
(112 of them, none touching this feature) must keep behaving exactly
as before -- that is what test_disambiguation_loop_wiring.py and
test_memory_loop_wiring.py already prove implicitly, by never passing
these new constructor arguments at all.
"""

import json
import re

import pytest

from probe.interactions import (
    InteractionRecorder,
)
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import TurnOutcomeLabel

_NOT_AMBIGUOUS = json.dumps({"needs_branches": False, "branches": []})
_TWO_BRANCHES = json.dumps(
    {
        "needs_branches": True,
        "branches": [
            {"statement": "wants the power rule explained"},
            {"statement": "wants a worked numeric example"},
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


def _make_loop(
    transcript, node_calls, disambiguation_store, interaction_store,
    turn_outcome_store, interaction_option_store, interaction_abstract_store,
    prediction_store, embedding_client, pool, llm=None, diagnostics_store=None,
):
    from probe.retrieval_config import RetrievalConfig

    recorder = InteractionRecorder(
        interaction_store,
        turn_outcome_store,
        embedding_client,
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
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_direct_answer_writes_a_learner_authored_interaction(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "a direct answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )

    message = await loop.handle_turn(session_id, 0, "what is a derivative?")
    assert message == "a direct answer"

    interaction = await interaction_store.get_at_turn(session_id, 0)
    assert interaction is not None
    assert interaction.question_author.value == "learner"
    assert interaction.did_branch is False
    assert interaction.response_text == "a direct answer"
    assert interaction.entry_state.value == "cold_open"

    await loop.wait_for_background_tasks()
    abstract = await interaction_abstract_store.get_latest(interaction.id)
    assert abstract is not None, "a response-bearing turn must get an abstraction"


@pytest.mark.asyncio(loop_scope="session")
async def test_no_interaction_pipeline_configured_writes_nothing(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store, diagnostics_store,
):
    """The pre-feature path: omitting every new constructor argument
    must behave exactly as before this feature existed."""
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "an answer"})
    loop = SessionLoop(
        transcript=transcript, node_calls=node_calls, llm=llm,
        diagnostics_store=diagnostics_store, disambiguation_store=disambiguation_store,
    )
    session_id = await transcript.create_session(learner_id)
    message = await loop.handle_turn(session_id, 0, "what is a derivative?")
    assert message == "an answer"

    from probe.interactions import InteractionStore

    interaction_store = InteractionStore(clean_pool)
    assert await interaction_store.get_at_turn(session_id, 0) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_an_options_offered_turn_writes_did_branch_true_and_no_response(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _TWO_BRANCHES, "DISAMBIGUATE:OPTIONS": _options_for_branches})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )

    message = await loop.handle_turn(session_id, 0, "can you help me with derivatives?")
    assert message == "Which of these did you mean?"

    interaction = await interaction_store.get_at_turn(session_id, 0)
    assert interaction.did_branch is True
    assert interaction.response_text is None

    # Immediate structural deferred -- no async wait needed, this is
    # written synchronously at record() time.
    outcome = await turn_outcome_store.get_latest(interaction.id)
    assert outcome.outcome is TurnOutcomeLabel.DEFERRED
    assert outcome.classifier_version == "structural-no-response"

    options = await interaction_option_store.list_for_interaction(interaction.id)
    assert len(options) == 2
    assert {o.shown_position for o in options} == {0, 1}
    assert all(not o.was_selected for o in options)


@pytest.mark.asyncio(loop_scope="session")
async def test_options_offered_turn_fires_a_prediction(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _TWO_BRANCHES, "DISAMBIGUATE:OPTIONS": _options_for_branches})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )
    await loop.handle_turn(session_id, 0, "can you help me with derivatives?")
    interaction = await interaction_store.get_at_turn(session_id, 0)

    await loop.wait_for_background_tasks()
    prediction = await prediction_store.get_latest(interaction.id)
    assert prediction is not None
    assert len(prediction.predicted_scores) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_a_click_resolution_records_system_option_author_and_originating_question(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _TWO_BRANCHES,
            "DISAMBIGUATE:OPTIONS": _options_for_branches,
            "FINAL:ANSWER": "resolved answer",
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )
    original = "can you help me with derivatives?"
    await loop.handle_turn(session_id, 0, original)

    latest_turn = await disambiguation_store.get_latest_turn(session_id)
    options = await disambiguation_store.list_options_for_turn(latest_turn.id)
    clicked = options[0]

    message = await loop.handle_turn(session_id, 1, clicked.text, clicked.id)
    assert message == "resolved answer"

    resolution = await interaction_store.get_at_turn(session_id, 1)
    assert resolution.question_author.value == "system_option"
    assert resolution.originating_question == original
    assert resolution.entry_state.value == "resolution"
    assert resolution.response_text == "resolved answer"

    offer = await interaction_store.get_at_turn(session_id, 0)
    updated_options = await interaction_option_store.list_for_interaction(offer.id)
    selected = [o for o in updated_options if o.was_selected]
    assert len(selected) == 1
    assert selected[0].option_id == clicked.id
    assert selected[0].selection_timestamp is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resolved_turn_gets_classified_once_the_next_question_arrives(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:TURN_OUTCOME": json.dumps(
                {"outcome": "matched", "confidence": 0.9, "abstains": False}
            ),
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )
    await loop.handle_turn(session_id, 0, "what is a derivative?")
    i0 = await interaction_store.get_at_turn(session_id, 0)
    assert await turn_outcome_store.get_latest(i0.id) is None

    await loop.handle_turn(session_id, 1, "ok what about integrals?")
    await loop.wait_for_background_tasks()

    outcome = await turn_outcome_store.get_latest(i0.id)
    assert outcome is not None
    assert outcome.outcome is TurnOutcomeLabel.MATCHED
    assert outcome.next_question_text == "ok what about integrals?"


@pytest.mark.asyncio(loop_scope="session")
async def test_empty_options_degrade_still_records_did_branch_true(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    """DisambiguationOptions produces nothing usable -> the turn
    degrades to a direct answer, but AssessAndBranch DID judge
    ambiguity genuine and emit branches -- did_branch must reflect
    that, not the eventual direct-answer outcome."""
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _TWO_BRANCHES,
            "DISAMBIGUATE:OPTIONS": "[]",
            "FINAL:ANSWER": "a direct fallback answer",
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )
    message = await loop.handle_turn(session_id, 0, "can you help me with derivatives?")
    assert message == "a direct fallback answer"

    interaction = await interaction_store.get_at_turn(session_id, 0)
    assert interaction.did_branch is True
    assert interaction.response_text == "a direct fallback answer"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failed_final_answer_records_no_response_not_the_failure_message(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client,
):
    class _FailingLLM(StubLLMClient):
        async def complete(self, prompt: str) -> str:
            if prompt.startswith("FINAL:ANSWER"):
                raise RuntimeError("simulated failure")
            return await super().complete(prompt)

    llm = _FailingLLM(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
    )
    message = await loop.handle_turn(session_id, 0, "what is a derivative?")
    assert "failed" in message.lower()

    interaction = await interaction_store.get_at_turn(session_id, 0)
    assert interaction is not None
    assert interaction.response_text is None, (
        "a FinalAnswer failure must be recorded as no-response, never as "
        "fabricated content"
    )
    await loop.wait_for_background_tasks()
    assert await interaction_abstract_store.get_latest(interaction.id) is None, (
        "no abstraction should be attempted over a non-existent response"
    )
