"""StatedPreferenceStore (Fix B's write-side storage) plus the full
loop wiring: a learner-authored turn gets classified off the critical
path, and a LATER turn's FinalAnswer call receives the most recent
stated preference rendered as a structural requirement -- never a
click/system_option turn, never folded into learner_history_block.
"""

import json

import pytest

from probe.disambiguate import DisambiguationStore
from probe.history_block import HistoryBlockConfig
from probe.interactions import InteractionRecorder
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import StatedPreference, StatedPreferenceLabel
from probe.retrieval_config import RetrievalConfig

_NOT_AMBIGUOUS = json.dumps({"needs_branches": False, "branches": []})
_HAS_PREFERENCE = json.dumps(
    {
        "has_preference": True,
        "stated_preference": "they want the concrete case before the abstract rule",
        "label": "concrete_before_abstract",
    }
)
_NO_PREFERENCE = json.dumps({"has_preference": False, "stated_preference": None, "label": None})


def _make_loop(
    transcript, node_calls, disambiguation_store, interaction_store,
    turn_outcome_store, interaction_option_store, interaction_abstract_store,
    prediction_store, stated_preference_store, embedding_client, pool, llm=None,
    diagnostics_store=None,
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
        # A minimal-length history block would need real retrieved
        # content unrelated to this test -- keep the assertions focused
        # on structural_requirement, not learner_history_block.
        history_block_config=HistoryBlockConfig(enabled=True),
    )


# ─────────────────────────── StatedPreferenceStore ──────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_append_and_get_latest_round_trip(
    stated_preference_store, interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    from probe.models import QuestionAuthor

    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await stated_preference_store.append(
        StatedPreference(
            interaction_id=interaction.id, learner_id=learner_id,
            has_preference=True, stated_preference="always show a number example first",
            label=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
            classifier_version="test-v1",
        )
    )
    latest = await stated_preference_store.get_latest_for_learner(learner_id)
    assert latest is not None
    assert latest.stated_preference == "always show a number example first"
    assert latest.label is StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_ignores_false_rows(
    stated_preference_store, interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    from probe.models import QuestionAuthor

    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await stated_preference_store.append(
        StatedPreference(
            interaction_id=interaction.id, learner_id=learner_id,
            has_preference=False, stated_preference=None,
            classifier_version="test-v1",
        )
    )
    assert await stated_preference_store.get_latest_for_learner(learner_id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_get_latest_returns_the_most_recent_true_row(
    stated_preference_store, interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    from probe.models import QuestionAuthor

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
    await stated_preference_store.append(
        StatedPreference(
            interaction_id=i0.id, learner_id=learner_id, has_preference=True,
            stated_preference="old preference", classifier_version="test-v1",
        )
    )
    await stated_preference_store.append(
        StatedPreference(
            interaction_id=i1.id, learner_id=learner_id, has_preference=True,
            stated_preference="new preference", classifier_version="test-v1",
        )
    )
    latest = await stated_preference_store.get_latest_for_learner(learner_id)
    assert latest.stated_preference == "new preference"


# ─────────────────────────── loop wiring ──────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stated_preference_is_classified_and_written(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    embedding_client, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:STATED_PREFERENCE": _HAS_PREFERENCE,
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(
        session_id, 0,
        "can you give me a number example first? I always want the concrete case first.",
    )
    await loop.wait_for_background_tasks()
    latest = await stated_preference_store.get_latest_for_learner(learner_id)
    assert latest is not None
    assert latest.stated_preference == "they want the concrete case before the abstract rule"
    assert latest.label is StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT


@pytest.mark.asyncio(loop_scope="session")
async def test_a_later_turn_gets_the_structural_requirement(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    embedding_client, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:STATED_PREFERENCE": _HAS_PREFERENCE,
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "I always want the concrete case before the abstract rule.")
    await loop.wait_for_background_tasks()

    call0 = await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer")
    assert call0.input_json["structural_requirement"] == "", (
        "the preference is classified from turn 0's own text but not written "
        "until AFTER turn 0's answer is generated -- it cannot apply to itself"
    )

    await loop.handle_turn(session_id, 1, "what's the chain rule?")
    call1 = await node_calls.get_call_for_turn(session_id, 1, "FinalAnswer")
    req = call1.input_json["structural_requirement"]
    assert "Structural requirement" in req
    # The label maps to a hand-written imperative -- the raw extracted
    # text is stored verbatim in stated_preferences but is NOT what
    # gets rendered into the prompt (see interaction_nodes.py's
    # render_structural_requirement for why).
    assert "Open with a concrete worked example" in req
    assert "Do not begin with the abstract rule" in req
    # Must never be folded into the same block as episodic history.
    assert req not in call1.input_json["learner_history_block"]


@pytest.mark.asyncio(loop_scope="session")
async def test_no_preference_leaves_the_requirement_empty(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    embedding_client, diagnostics_store,
):
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": _NOT_AMBIGUOUS,
            "FINAL:ANSWER": "an answer",
            "CLASSIFY:STATED_PREFERENCE": _NO_PREFERENCE,
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "what is a derivative?")
    await loop.wait_for_background_tasks()
    await loop.handle_turn(session_id, 1, "what's the chain rule?")
    call1 = await node_calls.get_call_for_turn(session_id, 1, "FinalAnswer")
    assert call1.input_json["structural_requirement"] == ""


@pytest.mark.asyncio(loop_scope="session")
async def test_a_click_resolution_turn_never_fires_the_classifier(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, stated_preference_store,
    embedding_client, diagnostics_store,
):
    """A click carries no free text from the student -- see
    StatedPreference's own docstring for why this must never be
    classified as if it were."""
    two_branches = json.dumps(
        {
            "needs_branches": True,
            "branches": [
                {"statement": "wants calculus derivatives"},
                {"statement": "wants financial derivatives"},
            ],
        }
    )
    import re

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
            "CLASSIFY:STATED_PREFERENCE": _HAS_PREFERENCE,
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, stated_preference_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    await loop.handle_turn(session_id, 0, "can you help me with derivatives?")
    disamb = DisambiguationStore(clean_pool)
    latest_turn = await disamb.get_latest_turn(session_id)
    options = await disamb.list_options_for_turn(latest_turn.id)
    clicked = options[0]

    await loop.handle_turn(session_id, 1, clicked.text, clicked.id)
    await loop.wait_for_background_tasks()

    # The offer turn (turn 0, question_author=learner) DID fire the
    # classifier; the click (turn 1, question_author=system_option)
    # must not have produced a SECOND row.
    async with clean_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM stated_preferences WHERE learner_id = $1", learner_id
        )
    assert count == 1
