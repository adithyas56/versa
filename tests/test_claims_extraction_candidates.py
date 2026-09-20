"""select_extraction_candidates -- ranks SUBJECT-kind interactions by
prediction error, ranks APPROACH-kind interactions by axis coverage
first and prediction error only as a tie-break/fallback once an axis
already has eligible evidence, and unions in any turn classified
contradicted_intent or carrying a stated preference. Real DB: needs
real interactions/interaction_options/predictions/turn_outcomes/
stated_preferences/claim_evidence rows, not stubs."""

from uuid import uuid4

import pytest

from probe.claims import ExtractionConfig, select_extraction_candidates
from probe.disambiguate import DisambiguationStore
from probe.models import (
    AmbiguityKind,
    ApproachAxis,
    ClaimEvidence,
    DisambiguationBranch,
    EvidenceDirection,
    InteractionOption,
    Option,
    Prediction,
    QuestionAuthor,
    StatedPreference,
    TurnOutcome,
    TurnOutcomeLabel,
)


async def _offer_with_prediction(
    interaction_recorder, interaction_option_store, prediction_store,
    learner_id, session_id, turn_number, predicted_scores, selected_key, pool,
    kind=None, axis=None,
):
    """Creates an offer interaction (did_branch, no response) plus two
    interaction_options and a Prediction whose predicted_scores are
    keyed by the SAME option_ids just created. interaction_options.
    option_id/branch_id are hard FKs to disambiguation_options/
    disambiguation_branches (migration 034) -- both created first, same
    as test_interactions_append_only.py's own precedent."""
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=turn_number,
        question_text=f"offer {turn_number}", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=True, response_text=None,
    )
    disamb = DisambiguationStore(pool)
    d_turn = await disamb.create_turn(session_id, turn_number, needs_branches=True, turn_had_direct_answer=False)
    branches = await disamb.add_branches([
        DisambiguationBranch(disambiguation_turn_id=d_turn.id, session_id=session_id,
                              turn_index=turn_number, statement="reading a"),
        DisambiguationBranch(disambiguation_turn_id=d_turn.id, session_id=session_id,
                              turn_index=turn_number, statement="reading b"),
    ])
    disamb_options = await disamb.create_options([
        Option(branch_id=branches[0].id, generation_id=d_turn.id, session_id=session_id,
               turn_index=turn_number, text="option A"),
        Option(branch_id=branches[1].id, generation_id=d_turn.id, session_id=session_id,
               turn_index=turn_number, text="option B"),
    ])
    option_ids = [disamb_options[0].id, disamb_options[1].id]
    options = [
        InteractionOption(
            interaction_id=interaction.id, learner_id=learner_id, option_id=option_ids[0],
            branch_id=branches[0].id, option_text="option A", shown_position=0,
            was_selected=(selected_key == "a"), kind=kind, axis=axis,
        ),
        InteractionOption(
            interaction_id=interaction.id, learner_id=learner_id, option_id=option_ids[1],
            branch_id=branches[1].id, option_text="option B", shown_position=1,
            was_selected=(selected_key == "b"), kind=kind, axis=axis,
        ),
    ]
    await interaction_option_store.create_many(options)
    scores = {str(option_ids[0]): predicted_scores[0], str(option_ids[1]): predicted_scores[1]}
    await prediction_store.append(
        Prediction(interaction_id=interaction.id, learner_id=learner_id,
                   predicted_scores=scores, model_version="test-v1")
    )
    return interaction


@pytest.mark.asyncio(loop_scope="session")
async def test_ranks_subject_kind_by_prediction_error_and_takes_top_n(
    transcript, interaction_recorder, interaction_option_store, prediction_store,
    learner_id, clean_pool,
):
    """Offers with no declared kind fall into the subject-kind bucket
    (the common case for a plain topic-disambiguation offer) and are
    still ranked by surprise, same as before this feature existed."""
    session_id = await transcript.create_session(learner_id)
    # error = 1 - predicted_scores[selected]:
    surprising = await _offer_with_prediction(  # selected "b" (0.1) -> error 0.9
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 0, (0.9, 0.1), "b", clean_pool,
    )
    medium = await _offer_with_prediction(  # selected "a" (0.5) -> error 0.5
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 1, (0.5, 0.5), "a", clean_pool,
    )
    unsurprising = await _offer_with_prediction(  # selected "a" (0.8) -> error 0.2
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 2, (0.8, 0.2), "a", clean_pool,
    )

    candidates = await select_extraction_candidates(
        clean_pool, session_id, learner_id, ExtractionConfig(top_n_subject=2)
    )
    ids = {c.interaction_id for c in candidates}
    assert surprising.id in ids
    assert medium.id in ids
    assert unsurprising.id not in ids


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_floor_quota_is_not_crowded_out_by_subject_surprise(
    transcript, interaction_recorder, interaction_option_store, prediction_store,
    learner_id, clean_pool,
):
    """The diagnostic finding this fixes: subject-kind picks average
    much higher prediction error than approach-kind picks, so a single
    pooled top-N starves approach-kind episodes. Separate quotas mean
    an UNSURPRISING approach-kind pick (error 0.1) still gets its own
    slot even though three subject-kind picks are all more "surprising"."""
    session_id = await transcript.create_session(learner_id)
    approach_offer = await _offer_with_prediction(  # selected "a" (0.9) -> error 0.1
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 0, (0.9, 0.1), "a", clean_pool,
        kind=AmbiguityKind.APPROACH, axis=ApproachAxis.CONCRETE_GENERAL,
    )
    subject_1 = await _offer_with_prediction(  # error 0.9
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 1, (0.9, 0.1), "b", clean_pool,
        kind=AmbiguityKind.SUBJECT,
    )
    subject_2 = await _offer_with_prediction(  # error 0.8
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 2, (0.8, 0.2), "b", clean_pool,
        kind=AmbiguityKind.SUBJECT,
    )
    subject_3 = await _offer_with_prediction(  # error 0.7 -- excluded by top_n_subject=2
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 3, (0.7, 0.3), "b", clean_pool,
        kind=AmbiguityKind.SUBJECT,
    )

    candidates = await select_extraction_candidates(
        clean_pool, session_id, learner_id, ExtractionConfig(top_n_approach=1, top_n_subject=2)
    )
    ids = {c.interaction_id for c in candidates}
    assert approach_offer.id in ids  # its own quota -- not competing with subject surprise
    assert subject_1.id in ids
    assert subject_2.id in ids
    assert subject_3.id not in ids


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_uncovered_axes_beat_surprise_within_their_own_quota(
    transcript, interaction_recorder, interaction_option_store, prediction_store,
    learner_id, clean_pool,
):
    """Coverage before surprise: two approach-kind offers on two
    DIFFERENT axes, neither covered by any eligible evidence yet for
    this learner, both get selected even though both are deeply
    unsurprising -- an uncovered axis doesn't need to earn its slot by
    prediction error."""
    session_id = await transcript.create_session(learner_id)
    low_surprise_a = await _offer_with_prediction(  # error 0.1
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 0, (0.9, 0.1), "a", clean_pool,
        kind=AmbiguityKind.APPROACH, axis=ApproachAxis.CONCRETE_GENERAL,
    )
    even_lower_surprise_b = await _offer_with_prediction(  # error 0.05
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 1, (0.95, 0.05), "a", clean_pool,
        kind=AmbiguityKind.APPROACH, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    candidates = await select_extraction_candidates(
        clean_pool, session_id, learner_id, ExtractionConfig(top_n_approach=2)
    )
    ids = {c.interaction_id for c in candidates}
    assert low_surprise_a.id in ids
    assert even_lower_surprise_b.id in ids


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_covered_axis_loses_to_uncovered_axis_despite_higher_surprise(
    transcript, interaction_recorder, interaction_option_store, prediction_store,
    claim_store, learner_id, clean_pool,
):
    """Once an axis has real (eligible) evidence, surprise no longer
    exempts it from competing -- an uncovered axis with LOWER
    prediction error still wins the single available slot over a
    covered axis with HIGHER prediction error."""
    from probe.claims import ExtractionConfig as _Cfg  # local import avoids shadowing above
    from probe.models import Claim, ClaimSource, ClaimWritePolicy, StatedPreferenceLabel

    session_id = await transcript.create_session(learner_id)

    # Seed one ELIGIBLE evidence row on CONCRETE_GENERAL -- makes that
    # axis "covered" for this learner.
    seed_interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="seed", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    seed_claim = Claim(
        learner_id=learner_id, statement="seed claim", test="seed test",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, confidence=0.5,
        source=ClaimSource.INFERRED, write_policy=ClaimWritePolicy.SLOW_DRIFT,
        statement_embedding=[0.1] * 768,
    )
    await claim_store.create(seed_claim)
    await claim_store.append_evidence(
        ClaimEvidence(
            claim_id=seed_claim.id, learner_id=learner_id, interaction_id=seed_interaction.id,
            direction=EvidenceDirection.SUPPORTS, topic="seed-topic",
            axis=ApproachAxis.CONCRETE_GENERAL, session_id=session_id,
            test_fired=True, contradiction_was_possible=True,
        )
    )

    covered_high_surprise = await _offer_with_prediction(  # error 0.9, but axis already covered
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 1, (0.9, 0.1), "b", clean_pool,
        kind=AmbiguityKind.APPROACH, axis=ApproachAxis.CONCRETE_GENERAL,
    )
    uncovered_low_surprise = await _offer_with_prediction(  # error 0.1, axis still uncovered
        interaction_recorder, interaction_option_store, prediction_store,
        learner_id, session_id, 2, (0.9, 0.1), "a", clean_pool,
        kind=AmbiguityKind.APPROACH, axis=ApproachAxis.ANALOGY_FORMAL,
    )

    candidates = await select_extraction_candidates(
        clean_pool, session_id, learner_id, _Cfg(top_n_approach=1)
    )
    ids = {c.interaction_id for c in candidates}
    assert uncovered_low_surprise.id in ids
    assert covered_high_surprise.id not in ids


@pytest.mark.asyncio(loop_scope="session")
async def test_contradicted_intent_is_included_regardless_of_ranking(
    transcript, interaction_recorder, turn_outcome_store, learner_id, clean_pool,
):
    session_id = await transcript.create_session(learner_id)
    resolved = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await turn_outcome_store.append(
        TurnOutcome(interaction_id=resolved.id, learner_id=learner_id,
                    outcome=TurnOutcomeLabel.CONTRADICTED_INTENT, confidence=0.9,
                    classifier_version="test-v1")
    )
    candidates = await select_extraction_candidates(clean_pool, session_id, learner_id, ExtractionConfig())
    assert any(c.interaction_id == resolved.id for c in candidates)
    assert any("contradicted_intent" in c.reason for c in candidates)


@pytest.mark.asyncio(loop_scope="session")
async def test_stated_preference_is_included_regardless_of_ranking(
    transcript, interaction_recorder, stated_preference_store, learner_id, clean_pool,
):
    session_id = await transcript.create_session(learner_id)
    resolved = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    await stated_preference_store.append(
        StatedPreference(interaction_id=resolved.id, learner_id=learner_id, has_preference=True,
                          stated_preference="short answers please", classifier_version="test-v1")
    )
    candidates = await select_extraction_candidates(clean_pool, session_id, learner_id, ExtractionConfig())
    assert any(c.interaction_id == resolved.id for c in candidates)
    assert any("stated preference" in c.reason for c in candidates)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_quiet_session_with_nothing_surprising_produces_no_candidates(
    transcript, interaction_recorder, learner_id, clean_pool,
):
    """The anchor's whole point: most turns in most sessions produce
    NOTHING here. A session with no predictions, no contradicted_intent,
    no stated preference must return an empty list, not a guess."""
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    candidates = await select_extraction_candidates(clean_pool, session_id, learner_id, ExtractionConfig())
    assert candidates == []
