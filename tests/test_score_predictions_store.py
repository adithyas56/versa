"""score_predictions_for_all_learners -- the one I/O-boundary test:
confirms real claims/evidence built through reconcile_candidate (the
same path production uses) wire correctly into compute_prediction_trials
and bucket_trials. The arithmetic itself is covered by
test_score_predictions.py's pure-function tests against hand-derived
values; this only checks the plumbing."""
from uuid import uuid4

import pytest

from probe.claims import reconcile_candidate
from probe.models import ApproachAxis, ClaimCandidate, QuestionAuthor, StatedPreferenceLabel
from probe.score_predictions import score_predictions_for_all_learners


@pytest.mark.asyncio(loop_scope="session")
async def test_score_predictions_scores_a_real_claims_history(
    claim_store, learner_id, transcript, interaction_recorder, embedding_client, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interactions = []
    for i in range(3):
        interactions.append(
            await interaction_recorder.record(
                learner_id=learner_id, session_id=session_id, turn_number=i,
                question_text=f"q{i}", question_author=QuestionAuthor.LEARNER,
                originating_question=None, did_branch=False, response_text=f"a{i}",
            )
        )

    # Three supporting picks on the SAME axis, three DISTINCT sessions
    # (each reconcile_candidate call reuses `session_id` here for
    # simplicity, but distinct topics keep the cells apart the same
    # way distinct sessions would -- what matters for this plumbing
    # test is that >=2 eligible rows exist so a trial gets produced).
    statements = [
        "prefers everyday analogies over formal technical breakdowns",
        "the person prefers intuitive explanations using comparisons",
        "wants a simple analogy rather than the formal terminology",
    ]
    claim = None
    for i, statement in enumerate(statements):
        candidate = ClaimCandidate(
            statement=statement, test=f"test {i}",
            value=StatedPreferenceLabel.WANTS_ANALOGIES, topic=f"topic-{i}",
        )
        claim = await reconcile_candidate(
            claim_store, embedding_client, learner_id, session_id, interactions[i].id,
            candidate, contradiction_was_possible=True, axis=ApproachAxis.ANALOGY_FORMAL,
        )

    bins, overall_brier, total_trials_computed = await score_predictions_for_all_learners(claim_store)
    total_trials = sum(b.n_trials for b in bins)
    assert total_trials >= 2  # 3 eligible supporting rows on one claim -> 2 trials
    assert total_trials_computed == total_trials  # the extended [0.0, 1.0] range drops nothing
    assert overall_brier is not None
    assert overall_brier >= 0.0
    # All three picks supported the same claim, so every trial in this
    # (isolated test-DB) run must be a hit.
    assert all(b.n_hits == b.n_trials for b in bins if b.n_trials)
