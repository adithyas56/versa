"""compute_capability_prediction_trials -- the capability-side reader.
Same closed-form hand-derivation style as test_score_predictions.py;
confirms PredictionTrial/bucket_trials/brier_score are genuinely reused
unchanged (no capability-specific bucketing logic exists anywhere)."""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from probe.capability import CapabilityConfidenceConfig
from probe.models import CapabilityClaim, CapabilityEvidence, CapabilityLabel, ClaimWritePolicy, EvidenceDirection
from probe.score_predictions import bucket_trials, compute_capability_prediction_trials

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _claim(write_policy=ClaimWritePolicy.LOCKED) -> CapabilityClaim:
    return CapabilityClaim(
        learner_id=uuid4(),
        statement="The learner can trace worked steps to find an error.",
        test="Given a worked example with a seeded error, the learner will locate it.",
        skill=CapabilityLabel.TRACES_WORKED_STEPS,
        confidence=0.5,
        write_policy=write_policy,
    )


def _evidence(
    claim_id, direction=EvidenceDirection.SUPPORTS, age_days=0.0, session_id=None,
    skill=CapabilityLabel.TRACES_WORKED_STEPS, test_fired=True, contradiction_was_possible=True,
    provenance_note=None,
) -> CapabilityEvidence:
    return CapabilityEvidence(
        claim_id=claim_id, learner_id=uuid4(), interaction_id=uuid4(),
        direction=direction, skill=skill, session_id=session_id or uuid4(),
        test_fired=test_fired, contradiction_was_possible=contradiction_was_possible,
        created_at=NOW - timedelta(days=age_days), provenance_note=provenance_note,
    )


def test_a_single_eligible_row_produces_no_trials():
    claim = _claim()
    rows = [_evidence(claim.id, age_days=5.0)]
    assert compute_capability_prediction_trials(claim, rows) == []


def test_second_row_scored_against_the_first():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=10.0)
    row2 = _evidence(claim.id, age_days=0.0)
    trials = compute_capability_prediction_trials(claim, [row1, row2])
    assert len(trials) == 1
    # posterior mean: alpha=1+1=2, beta=1+0=1 -> 2/3
    assert abs(trials[0].predicted_confidence - (2 / 3)) < 1e-9
    assert trials[0].hit is True


def test_a_contradicting_row_is_scored_as_a_miss_not_excluded():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=10.0)
    row2 = _evidence(claim.id, age_days=0.0, direction=EvidenceDirection.CONTRADICTS)
    trials = compute_capability_prediction_trials(claim, [row1, row2])
    assert len(trials) == 1
    assert trials[0].hit is False


def test_ineligible_rows_are_excluded_from_trials_and_priors():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=20.0)
    ineligible = _evidence(claim.id, age_days=10.0, contradiction_was_possible=False)
    row3 = _evidence(claim.id, age_days=0.0)
    with_noise = compute_capability_prediction_trials(claim, [row1, ineligible, row3])
    without_noise = compute_capability_prediction_trials(claim, [row1, row3])
    assert len(with_noise) == 1
    assert with_noise[0].predicted_confidence == without_noise[0].predicted_confidence


def test_skill_filter_isolates_one_skill_from_another():
    claim = _claim()
    traces_row1 = _evidence(claim.id, age_days=20.0, skill=CapabilityLabel.TRACES_WORKED_STEPS)
    traces_row2 = _evidence(claim.id, age_days=10.0, skill=CapabilityLabel.TRACES_WORKED_STEPS)
    derives_row = _evidence(claim.id, age_days=5.0, skill=CapabilityLabel.DERIVES_FORWARD)

    traces_only = compute_capability_prediction_trials(
        claim, [traces_row1, traces_row2, derives_row], skill_filter=CapabilityLabel.TRACES_WORKED_STEPS
    )
    assert len(traces_only) == 1

    derives_only = compute_capability_prediction_trials(
        claim, [traces_row1, traces_row2, derives_row], skill_filter=CapabilityLabel.DERIVES_FORWARD
    )
    assert derives_only == []  # only 1 eligible row once traces' are filtered out


def test_exclude_contaminated_drops_a_flagged_row():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=20.0)
    contaminated = _evidence(claim.id, age_days=10.0, direction=EvidenceDirection.CONTRADICTS,
                              provenance_note="harness bug")
    row3 = _evidence(claim.id, age_days=0.0)
    cleaned = compute_capability_prediction_trials(claim, [row1, contaminated, row3], exclude_contaminated=True)
    assert len(cleaned) == 1
    assert cleaned[0].hit is True


def test_trials_feed_bucket_trials_unmodified():
    """Confirms genuine reuse -- PredictionTrial objects from the
    capability side bucket correctly through the exact same
    bucket_trials this module already uses for preference trials."""
    claim = _claim()
    row1 = _evidence(claim.id, age_days=10.0)
    row2 = _evidence(claim.id, age_days=0.0)
    trials = compute_capability_prediction_trials(claim, [row1, row2])
    bins = bucket_trials(trials)
    by_range = {(b.lower, b.upper): b for b in bins}
    # posterior mean 2/3 falls in [0.6, 0.7), not [0.7, 0.8) (that was
    # beat-chance's 0.75) -- see compute_capability_confidence's own
    # docstring for the switch.
    assert by_range[(0.6, 0.7)].n_trials == 1
    assert by_range[(0.6, 0.7)].n_hits == 1
