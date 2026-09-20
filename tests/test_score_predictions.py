"""score_predictions.py -- pure functions, no I/O, no randomness.
Predicted-confidence values reuse claims.compute_confidence directly
(already hand-verified in test_claims_confidence.py against the Beta
CDF closed form), so these tests check the TRIAL-EXTRACTION logic --
which rows become trials, what prior evidence each trial is scored
against, and the bucketing/scoring arithmetic on top -- not
compute_confidence's own math a second time.
"""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from probe.models import Claim, ClaimEvidence, ClaimSource, ClaimWritePolicy, EvidenceDirection, StatedPreferenceLabel
from probe.score_predictions import (
    CalibrationBin,
    PredictionTrial,
    brier_score,
    bucket_trials,
    compute_prediction_trials,
    format_reliability_diagram,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _claim(write_policy=ClaimWritePolicy.LOCKED) -> Claim:
    return Claim(
        learner_id=uuid4(),
        statement="wants concrete examples first",
        test="given a concrete_general axis choice, picks concrete",
        value=StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        confidence=0.5,
        source=ClaimSource.INFERRED,
        write_policy=write_policy,
        statement_embedding=[0.1] * 768,
    )


def _evidence(
    claim_id,
    direction: EvidenceDirection = EvidenceDirection.SUPPORTS,
    age_days: float = 0.0,
    session_id=None,
    topic: str = "topic-a",
    test_fired: bool = True,
    contradiction_was_possible: bool = True,
    provenance_note: str | None = None,
) -> ClaimEvidence:
    return ClaimEvidence(
        claim_id=claim_id,
        learner_id=uuid4(),
        interaction_id=uuid4(),
        direction=direction,
        topic=topic,
        session_id=session_id or uuid4(),
        test_fired=test_fired,
        contradiction_was_possible=contradiction_was_possible,
        created_at=NOW - timedelta(days=age_days),
        provenance_note=provenance_note,
    )


# ─────────────────────────── compute_prediction_trials ──────────────────


def test_a_single_eligible_row_produces_no_trials():
    claim = _claim()
    rows = [_evidence(claim.id, age_days=5.0)]
    assert compute_prediction_trials(claim, rows) == []


def test_second_row_scored_against_the_first():
    """LOCKED, both supports, different sessions -> prior=[row1] is
    one cell -> alpha=2, beta=1 -> confidence = 1 - 0.5**2 = 0.75."""
    claim = _claim()
    row1 = _evidence(claim.id, age_days=10.0)
    row2 = _evidence(claim.id, age_days=0.0)
    trials = compute_prediction_trials(claim, [row1, row2])
    assert len(trials) == 1
    assert abs(trials[0].predicted_confidence - 0.75) < 1e-9
    assert trials[0].hit is True
    assert trials[0].evidence_id == row2.id


def test_third_row_scored_against_the_first_two():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=20.0)
    row2 = _evidence(claim.id, age_days=10.0)
    row3 = _evidence(claim.id, age_days=0.0)
    trials = compute_prediction_trials(claim, [row1, row2, row3])
    assert len(trials) == 2
    # trial 1: prior=[row1] -> 1 - 0.5**2
    assert abs(trials[0].predicted_confidence - 0.75) < 1e-9
    # trial 2: prior=[row1, row2] -> 1 - 0.5**3
    assert abs(trials[1].predicted_confidence - 0.875) < 1e-9
    assert trials[0].hit is True and trials[1].hit is True


def test_a_contradicting_row_is_scored_as_a_miss_not_excluded():
    claim = _claim()
    row1 = _evidence(claim.id, age_days=10.0)
    row2 = _evidence(claim.id, age_days=0.0, direction=EvidenceDirection.CONTRADICTS)
    trials = compute_prediction_trials(claim, [row1, row2])
    assert len(trials) == 1
    assert trials[0].hit is False
    assert abs(trials[0].predicted_confidence - 0.75) < 1e-9  # predicted from the supporting prior alone


def test_ineligible_rows_are_excluded_from_trials_and_from_priors():
    """A row with contradiction_was_possible=False sits between two
    eligible rows -- it must not become a trial, and it must not count
    as prior evidence either (matching compute_confidence's own
    filter)."""
    claim = _claim()
    row1 = _evidence(claim.id, age_days=20.0)
    ineligible = _evidence(claim.id, age_days=10.0, contradiction_was_possible=False)
    row3 = _evidence(claim.id, age_days=0.0)

    with_noise = compute_prediction_trials(claim, [row1, ineligible, row3])
    without_noise = compute_prediction_trials(claim, [row1, row3])
    assert len(with_noise) == 1
    assert with_noise[0].predicted_confidence == without_noise[0].predicted_confidence
    assert with_noise[0].evidence_id == row3.id


def test_topic_prefix_isolates_one_instrument_method_from_another():
    """write_instrument_evidence tags topic=f"instrument:{primitive}" --
    topic_prefix is how score-predictions --split-by-source tells
    locate's own curve apart from predict's, both source=instrument."""
    claim = _claim()
    locate_row1 = _evidence(claim.id, age_days=20.0, topic="instrument:locate")
    locate_row2 = _evidence(claim.id, age_days=10.0, topic="instrument:locate")
    predict_row = _evidence(claim.id, age_days=5.0, topic="instrument:predict")

    locate_only = compute_prediction_trials(
        claim, [locate_row1, locate_row2, predict_row], topic_prefix="instrument:locate"
    )
    assert len(locate_only) == 1
    assert locate_only[0].evidence_id == locate_row2.id

    predict_only = compute_prediction_trials(
        claim, [locate_row1, locate_row2, predict_row], topic_prefix="instrument:predict"
    )
    assert predict_only == []  # only 1 eligible row once locate's are filtered out -- no trial


def test_same_session_axis_cell_collapsing_carries_through_to_trials():
    """Two rows in the SAME (session, topic-as-axis-fallback) cell
    collapse to one cell for the purposes of the NEXT trial's predicted
    confidence -- exactly like compute_confidence's own behavior."""
    claim = _claim()
    session_id = uuid4()
    row1 = _evidence(claim.id, age_days=20.0, session_id=session_id, topic="t1")
    row2 = _evidence(claim.id, age_days=10.0, session_id=session_id, topic="t1")  # same cell as row1
    row3 = _evidence(claim.id, age_days=0.0, session_id=uuid4(), topic="t2")
    trials = compute_prediction_trials(claim, [row1, row2, row3])
    # trial for row2: prior=[row1], same cell as row2 itself but row2 isn't in "prior" -> 1 cell -> 0.75
    assert abs(trials[0].predicted_confidence - 0.75) < 1e-9
    # trial for row3: prior=[row1, row2] collapses to ONE cell (same session+topic) -> still 1 - 0.5**2
    assert abs(trials[1].predicted_confidence - 0.75) < 1e-9


def test_exclude_contaminated_drops_a_flagged_row_from_trials_and_priors():
    """A row with provenance_note set is a human-diagnosed harness bug
    (see ClaimStore.mark_evidence_contaminated), not genuine learner
    behavior. With exclude_contaminated=True it must contribute to
    NEITHER a trial's own outcome nor the prior context used to predict
    a later trial -- the "as if it never happened" picture. Without the
    flag (production default), it counts exactly like any other row."""
    claim = _claim()
    row1 = _evidence(claim.id, age_days=20.0)
    contaminated = _evidence(
        claim.id, age_days=10.0, direction=EvidenceDirection.CONTRADICTS,
        provenance_note="keyword-collision: simulator intended analogy, matcher selected formal",
    )
    row3 = _evidence(claim.id, age_days=0.0)

    with_contamination = compute_prediction_trials(claim, [row1, contaminated, row3])
    assert len(with_contamination) == 2
    assert with_contamination[0].hit is False  # the contaminated row still scores as a miss by default

    cleaned = compute_prediction_trials(claim, [row1, contaminated, row3], exclude_contaminated=True)
    assert len(cleaned) == 1  # the contaminated row produces no trial of its own
    assert cleaned[0].hit is True
    assert cleaned[0].evidence_id == row3.id
    # predicted purely from row1 (one supporting cell), not diluted by the dropped contradiction
    assert abs(cleaned[0].predicted_confidence - 0.75) < 1e-9


# ─────────────────────────── bucket_trials ──────────────────


def _trial(predicted: float, hit: bool) -> PredictionTrial:
    return PredictionTrial(
        claim_id=uuid4(), evidence_id=uuid4(), occurred_at=NOW,
        predicted_confidence=predicted, hit=hit,
    )


def test_bucket_trials_groups_by_predicted_confidence():
    trials = [
        _trial(0.55, True), _trial(0.58, False),  # bucket [0.5, 0.6)
        _trial(0.72, True),                        # bucket [0.7, 0.8)
        _trial(0.95, True), _trial(0.91, True),    # bucket [0.9, 1.0]
    ]
    bins = bucket_trials(trials)
    by_range = {(b.lower, b.upper): b for b in bins}
    low_bin = by_range[(0.5, 0.6)]
    assert low_bin.n_trials == 2
    assert low_bin.n_hits == 1
    assert abs(low_bin.mean_predicted_confidence - 0.565) < 1e-9

    mid_bin = by_range[(0.7, 0.8)]
    assert mid_bin.n_trials == 1
    assert mid_bin.n_hits == 1

    top_bin = by_range[(0.9, 1.0)]
    assert top_bin.n_trials == 2
    assert top_bin.n_hits == 2

    empty_bin = by_range[(0.6, 0.7)]
    assert empty_bin.n_trials == 0
    assert empty_bin.mean_predicted_confidence is None
    assert empty_bin.observed_hit_rate is None


def test_bucket_trials_last_bin_is_inclusive_of_the_top_edge():
    trials = [_trial(1.0, True)]
    bins = bucket_trials(trials)
    top_bin = bins[-1]
    assert top_bin.lower == 0.9 and top_bin.upper == 1.0
    assert top_bin.n_trials == 1


def test_bucket_trials_with_no_trials_reports_all_empty_bins():
    bins = bucket_trials([])
    assert all(b.n_trials == 0 for b in bins)
    assert len(bins) == 10  # DEFAULT_BIN_EDGES spans [0.0, 1.0] in 0.1-wide buckets


# ─────────────────────────── brier_score ──────────────────


def test_brier_score_hand_computed():
    trials = [_trial(0.9, True), _trial(0.8, False)]
    # (1-0.9)**2 = 0.01; (0-0.8)**2 = 0.64; mean = 0.325
    assert abs(brier_score(trials) - 0.325) < 1e-9


def test_brier_score_perfect_calibration_is_zero():
    trials = [_trial(1.0, True), _trial(0.0, False)]
    assert brier_score(trials) == 0.0


def test_brier_score_of_no_trials_is_none():
    assert brier_score([]) is None


# ─────────────────────────── format_reliability_diagram ──────────────────


def test_format_reliability_diagram_includes_counts_and_brier():
    trials = [_trial(0.95, True), _trial(0.92, False)]
    bins = bucket_trials(trials)
    text = format_reliability_diagram(bins, brier_score(trials))
    assert "total trials: 2" in text
    assert "0.9, 1.0]" in text


def test_format_reliability_diagram_handles_no_trials():
    bins = bucket_trials([])
    text = format_reliability_diagram(bins, brier_score([]))
    assert "total trials: 0" in text


def test_low_confidence_trials_land_in_the_extended_range_not_dropped():
    """The bin-range fix: a claim predicting below chance (net
    contradicting evidence) must be counted, not silently absorbed by
    a bucket range that used to start at 0.5. Two totals matching by
    coincidence is exactly how the old gap went unnoticed."""
    trials = [_trial(0.125, True), _trial(0.35, False)]
    bins = bucket_trials(trials)
    assert sum(b.n_trials for b in bins) == 2
    by_range = {(b.lower, b.upper): b for b in bins}
    assert by_range[(0.1, 0.2)].n_trials == 1
    assert by_range[(0.3, 0.4)].n_trials == 1


def test_format_reliability_diagram_shows_total_trials_computed_alongside_bucketed_sum():
    trials = [_trial(0.95, True), _trial(0.92, False)]
    bins = bucket_trials(trials)
    text = format_reliability_diagram(bins, brier_score(trials), total_trials_computed=2)
    assert "total trials: 2 (of 2 scored)" in text
    assert "WARNING" not in text


def test_format_reliability_diagram_flags_a_mismatch_between_scored_and_bucketed():
    """If a caller narrows bin_edges (or a future edge-range mistake
    reintroduces a gap), the mismatch must be visible in the printed
    output, not just inferable by comparing two numbers by hand."""
    trials = [_trial(0.95, True), _trial(0.05, False)]
    bins = bucket_trials(trials, bin_edges=[0.5, 1.0])  # 0.05 falls outside this narrowed range
    text = format_reliability_diagram(bins, brier_score(trials), total_trials_computed=len(trials))
    assert "total trials: 1 (of 2 scored)" in text
    assert "WARNING" in text
