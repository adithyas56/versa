"""capability.compute_capability_confidence -- pure function, no I/O.
Reports the posterior mean alpha/(alpha+beta) (changed from beat-chance
2026-09-15 -- see the function's own docstring), so every expected
value here is a plain fraction, not a beta.cdf closed form."""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from probe.capability import CapabilityConfidenceConfig, compute_capability_confidence
from probe.models import CapabilityEvidence, CapabilityLabel, ClaimWritePolicy, EvidenceDirection

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _evidence(
    direction: EvidenceDirection,
    skill: CapabilityLabel = CapabilityLabel.TRACES_WORKED_STEPS,
    session_id=None,
    age_days: float = 0.0,
    test_fired: bool = True,
    contradiction_was_possible: bool = True,
) -> CapabilityEvidence:
    return CapabilityEvidence(
        claim_id=uuid4(), learner_id=uuid4(), interaction_id=uuid4(),
        direction=direction, skill=skill, session_id=session_id or uuid4(),
        test_fired=test_fired, contradiction_was_possible=contradiction_was_possible,
        created_at=NOW - timedelta(days=age_days),
    )


def test_no_evidence_is_exactly_chance():
    assert compute_capability_confidence([], ClaimWritePolicy.LOCKED, now=NOW) == 0.5


def test_one_fresh_supporting_cell_locked_no_decay():
    evidence = [_evidence(EvidenceDirection.SUPPORTS)]
    confidence = compute_capability_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    # alpha=1+1=2, beta=1+0=1 -> 2/3
    assert abs(confidence - (2 / 3)) < 1e-9


def test_one_fresh_contradicting_cell_locked_no_decay():
    evidence = [_evidence(EvidenceDirection.CONTRADICTS)]
    confidence = compute_capability_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    # alpha=1+0=1, beta=1+1=2 -> 1/3
    assert abs(confidence - (1 / 3)) < 1e-9


def test_two_distinct_sessions_same_skill_are_two_cells():
    evidence = [
        _evidence(EvidenceDirection.SUPPORTS, session_id=uuid4()),
        _evidence(EvidenceDirection.SUPPORTS, session_id=uuid4()),
    ]
    confidence = compute_capability_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    # alpha=1+2=3, beta=1+0=1 -> 3/4
    assert abs(confidence - (3 / 4)) < 1e-9


def test_duplicate_rows_in_the_same_session_collapse_to_one_cell():
    session_id = uuid4()
    one = [_evidence(EvidenceDirection.SUPPORTS, session_id=session_id)]
    three = [_evidence(EvidenceDirection.SUPPORTS, session_id=session_id) for _ in range(3)]
    assert compute_capability_confidence(one, ClaimWritePolicy.LOCKED, now=NOW) == \
        compute_capability_confidence(three, ClaimWritePolicy.LOCKED, now=NOW)


def test_ineligible_row_contributes_nothing():
    evidence = [_evidence(EvidenceDirection.SUPPORTS, contradiction_was_possible=False)]
    assert compute_capability_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW) == 0.5


def test_decay_reduces_an_old_cells_contribution():
    cfg = CapabilityConfidenceConfig()
    evidence = [_evidence(EvidenceDirection.SUPPORTS, age_days=10.0)]
    confidence = compute_capability_confidence(evidence, ClaimWritePolicy.SLOW_DRIFT, cfg, now=NOW)
    weight = cfg.slow_drift_decay ** 10.0
    # alpha=1+weight, beta=1 -> (1+weight) / (2+weight)
    assert abs(confidence - ((1 + weight) / (2 + weight))) < 1e-9
