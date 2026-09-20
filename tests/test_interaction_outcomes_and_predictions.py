"""TurnOutcomeStore, InteractionAbstractStore, PredictionStore — all
append-only, "latest classifier_version/generator_version per
interaction_id wins" on read.

The seq-ordering tests are a direct regression guard: an earlier
version of get_latest() ordered by created_at alone and returned the
WRONG (stale) row when two appends landed in the same tight loop,
because created_at comes from Python's own clock at object-construction
time, not a DB-assigned monotonic value — exactly the failure this
codebase already hit once for node_calls (migration 026) and fixed the
same way (a BIGSERIAL seq column).
"""

import pytest

from probe.models import (
    InteractionAbstract,
    Prediction,
    QuestionAuthor,
    TurnOutcome,
    TurnOutcomeLabel,
)


async def _make_interaction(interaction_recorder, transcript, learner_id):
    session_id = await transcript.create_session(learner_id)
    return await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_outcome_latest_wins_even_under_identical_timestamps(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    interaction = await _make_interaction(interaction_recorder, transcript, learner_id)
    # Two appends back-to-back, close enough that created_at could tie.
    for outcome, version in (
        (TurnOutcomeLabel.CONTRADICTED_INTENT, "v1"),
        (TurnOutcomeLabel.MATCHED, "v2"),
    ):
        await turn_outcome_store.append(
            TurnOutcome(
                interaction_id=interaction.id, learner_id=learner_id,
                outcome=outcome, confidence=0.9, classifier_version=version,
            )
        )
    latest = await turn_outcome_store.get_latest(interaction.id)
    assert latest.classifier_version == "v2"
    assert latest.outcome is TurnOutcomeLabel.MATCHED


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_outcome_get_latest_many_is_correct_per_interaction(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    i1 = await _make_interaction(interaction_recorder, transcript, learner_id)
    i2 = await _make_interaction(interaction_recorder, transcript, learner_id)
    await turn_outcome_store.append(
        TurnOutcome(interaction_id=i1.id, learner_id=learner_id, outcome=TurnOutcomeLabel.MATCHED, confidence=0.9, classifier_version="v1")
    )
    await turn_outcome_store.append(
        TurnOutcome(interaction_id=i2.id, learner_id=learner_id, outcome=TurnOutcomeLabel.MOVED_ON, confidence=0.8, classifier_version="v1")
    )
    bulk = await turn_outcome_store.get_latest_many([i1.id, i2.id])
    assert bulk[i1.id].outcome is TurnOutcomeLabel.MATCHED
    assert bulk[i2.id].outcome is TurnOutcomeLabel.MOVED_ON


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_outcome_get_latest_many_empty_input(turn_outcome_store, clean_pool):
    assert await turn_outcome_store.get_latest_many([]) == {}


@pytest.mark.asyncio(loop_scope="session")
async def test_turn_outcome_get_latest_none_when_unclassified(
    interaction_recorder, transcript, learner_id, turn_outcome_store, clean_pool
):
    interaction = await _make_interaction(interaction_recorder, transcript, learner_id)
    assert await turn_outcome_store.get_latest(interaction.id) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_abstract_latest_wins_even_under_identical_timestamps(
    interaction_recorder, transcript, learner_id, interaction_abstract_store, clean_pool
):
    interaction = await _make_interaction(interaction_recorder, transcript, learner_id)
    for form, version in (("first form", "v1"), ("second form", "v2")):
        await interaction_abstract_store.append(
            InteractionAbstract(
                interaction_id=interaction.id, learner_id=learner_id,
                abstract_form=form, abstract_embedding=[0.1] * 768,
                generator_version=version,
            )
        )
    latest = await interaction_abstract_store.get_latest(interaction.id)
    assert latest.generator_version == "v2"
    assert latest.abstract_form == "second form"
    assert len(latest.abstract_embedding) == 768


@pytest.mark.asyncio(loop_scope="session")
async def test_prediction_latest_wins_and_round_trips_jsonb(
    interaction_recorder, transcript, learner_id, prediction_store, clean_pool
):
    interaction = await _make_interaction(interaction_recorder, transcript, learner_id)
    from uuid import uuid4

    opt_a, opt_b = uuid4(), uuid4()
    await prediction_store.append(
        Prediction(
            interaction_id=interaction.id, learner_id=learner_id,
            predicted_scores={str(opt_a): 0.3, str(opt_b): 0.7},
            model_version="v1",
            retrieved_candidate_ids=[str(uuid4())],
            retrieval_provenance=[{"scope": "personal", "similarity": 0.9}],
        )
    )
    await prediction_store.append(
        Prediction(
            interaction_id=interaction.id, learner_id=learner_id,
            predicted_scores={str(opt_a): 0.6, str(opt_b): 0.4},
            model_version="v2",
            retrieved_candidate_ids=[],
            retrieval_provenance=[],
        )
    )
    latest = await prediction_store.get_latest(interaction.id)
    assert latest.model_version == "v2"
    assert latest.predicted_scores == {str(opt_a): 0.6, str(opt_b): 0.4}
    assert isinstance(latest.predicted_scores, dict)


@pytest.mark.asyncio(loop_scope="session")
async def test_prediction_get_latest_none_before_any_prediction(
    interaction_recorder, transcript, learner_id, prediction_store, clean_pool
):
    interaction = await _make_interaction(interaction_recorder, transcript, learner_id)
    assert await prediction_store.get_latest(interaction.id) is None
