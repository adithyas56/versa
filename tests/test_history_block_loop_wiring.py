"""Wiring for history_block.py into SessionLoop/FinalAnswer -- retrieval
already ran every turn before this (feeding PredictSelection); this is
what makes FinalAnswer actually read it. Raw formatting logic is
covered by test_history_block.py; this exercises the real pipeline
(real embeddings via StubEmbeddingClient, real retrieve() against a
real DB) end to end.
"""

import json

import pytest

from probe.disambiguate import DisambiguationStore
from probe.history_block import TEMPLATE_VERSION, HistoryBlockConfig
from probe.interactions import InteractionRecorder
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.retrieval_config import RetrievalConfig


def _make_loop(
    transcript, node_calls, disambiguation_store, interaction_store,
    turn_outcome_store, interaction_option_store, interaction_abstract_store,
    prediction_store, embedding_client, pool, llm=None, diagnostics_store=None,
    history_block_config=None,
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
        history_block_config=history_block_config,
    )


_NOT_AMBIGUOUS = json.dumps({"needs_branches": False, "branches": []})


@pytest.mark.asyncio(loop_scope="session")
async def test_a_later_turn_gets_a_nonempty_history_block_from_an_earlier_one(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client, diagnostics_store,
):
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "an answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")
    call0 = await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer")
    assert call0.input_json["learner_history_block"] == "", (
        "the very first turn has nothing to retrieve yet"
    )

    await loop.handle_turn(session_id, 1, "what is a derivative?")
    call1 = await node_calls.get_call_for_turn(session_id, 1, "FinalAnswer")
    block = call1.input_json["learner_history_block"]
    assert block != ""
    assert "what is a derivative?" in block
    assert "Never mention or reference it directly" in block

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.history_block_used is True
    assert diag.history_block_template_version == TEMPLATE_VERSION
    assert len(diag.history_block_source_ids) >= 1


@pytest.mark.asyncio(loop_scope="session")
async def test_history_block_disabled_by_config_stays_empty(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client, diagnostics_store,
):
    llm = StubLLMClient(canned={"ASSESS:BRANCH": _NOT_AMBIGUOUS, "FINAL:ANSWER": "an answer"})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
        history_block_config=HistoryBlockConfig(enabled=False),
    )

    await loop.handle_turn(session_id, 0, "what is a derivative?")
    await loop.handle_turn(session_id, 1, "what is a derivative?")
    call1 = await node_calls.get_call_for_turn(session_id, 1, "FinalAnswer")
    assert call1.input_json["learner_history_block"] == ""

    diag = await diagnostics_store.get_for_turn(session_id, 1)
    assert diag.history_block_used is False
    assert diag.history_block_template_version is None


@pytest.mark.asyncio(loop_scope="session")
async def test_an_options_offered_turn_gets_no_history_block(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client, diagnostics_store,
):
    """Offer turns produce no answer at all -- FinalAnswer never runs,
    so no history block is assembled or logged."""
    two_branches = json.dumps(
        {
            "needs_branches": True,
            "branches": [
                {"statement": "wants the power rule explained"},
                {"statement": "wants a worked numeric example"},
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

    llm = StubLLMClient(canned={"ASSESS:BRANCH": two_branches, "DISAMBIGUATE:OPTIONS": _options_for_branches})
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    message = await loop.handle_turn(session_id, 0, "can you help me with derivatives?")
    assert message == "Which of these did you mean?"
    assert await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer") is None

    diag = await diagnostics_store.get_for_turn(session_id, 0)
    assert diag.history_block_used is False


@pytest.mark.asyncio(loop_scope="session")
async def test_resolution_turn_embeds_the_original_question_not_the_option_copy(
    transcript, node_calls, clean_pool, learner_id, disambiguation_store,
    interaction_store, turn_outcome_store, interaction_option_store,
    interaction_abstract_store, prediction_store, embedding_client, diagnostics_store,
):
    """Same rule the storage side already follows (question_author/
    originating_question split) -- the history-block retrieval query on
    a click-resolution turn must embed the student's own original
    words, never the clicked option's system-authored copy."""
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
        }
    )
    session_id = await transcript.create_session(learner_id)
    loop = _make_loop(
        transcript, node_calls, disambiguation_store, interaction_store,
        turn_outcome_store, interaction_option_store, interaction_abstract_store,
        prediction_store, embedding_client, clean_pool, llm=llm,
        diagnostics_store=diagnostics_store,
    )
    original = "can you help me with derivatives?"
    await loop.handle_turn(session_id, 0, original)

    disamb = DisambiguationStore(clean_pool)
    latest_turn = await disamb.get_latest_turn(session_id)
    options = await disamb.list_options_for_turn(latest_turn.id)
    clicked = options[0]

    embedding_client.texts.clear()
    await loop.handle_turn(session_id, 1, clicked.text, clicked.id)
    assert original in embedding_client.texts
    assert clicked.text not in embedding_client.texts
