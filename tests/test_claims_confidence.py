"""claims.compute_confidence / evaluate_promotion / clamp_confidence_for_decisions
-- pure functions, no I/O, no DB. Hand-computed expected values use the
CLOSED FORM of the Beta CDF at the two special cases these tests are
built around, independent of scipy: for a Beta(a, 1) distribution,
CDF(x) = x**a; for a Beta(1, b) distribution, CDF(x) = 1 - (1-x)**b.
Every test here keeps either alpha or beta pinned at 1 (via alpha0=1,
beta0=1 and evidence entirely in one direction) specifically so the
expected value can be derived with plain `**`, not by re-deriving
through scipy's own beta.cdf a second time -- that would just be
asserting the function equals itself.
"""
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from probe.claims import (
    ClaimConfidenceConfig,
    clamp_confidence_for_decisions,
    compute_confidence,
    evaluate_contradiction,
    evaluate_promotion,
)
from probe.models import ApproachAxis, ClaimEvidence, ClaimWritePolicy, EvidenceDirection

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _evidence(
    direction: EvidenceDirection,
    topic: str = "topic-a",
    session_id=None,
    age_days: float = 0.0,
    axis: ApproachAxis | None = None,
    test_fired: bool = True,
    contradiction_was_possible: bool = True,
) -> ClaimEvidence:
    return ClaimEvidence(
        claim_id=uuid4(),
        learner_id=uuid4(),
        interaction_id=uuid4(),
        direction=direction,
        topic=topic,
        axis=axis,
        session_id=session_id or uuid4(),
        test_fired=test_fired,
        contradiction_was_possible=contradiction_was_possible,
        created_at=NOW - timedelta(days=age_days),
    )


def test_no_evidence_is_exactly_chance():
    # alpha0=beta0=1 (uniform prior) with zero evidence -> Beta(1,1) is
    # uniform on [0,1], whose CDF at 0.5 is exactly 0.5.
    confidence = compute_confidence([], ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence == 0.5


def test_one_fresh_supporting_cell_locked_no_decay():
    # alpha = 1 + 1.0**0 = 2, beta = 1 -> Beta(2,1): CDF(0.5) = 0.5**2.
    evidence = [_evidence(EvidenceDirection.SUPPORTS, age_days=0.0)]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    expected = 1 - 0.5**2
    assert abs(confidence - expected) < 1e-9


def test_two_fresh_supporting_cells_different_sessions_locked():
    # Two DISTINCT (session, topic) cells -> alpha = 1 + 2 = 3, beta = 1.
    evidence = [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=uuid4()),
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-b", session_id=uuid4()),
    ]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    expected = 1 - 0.5**3
    assert abs(confidence - expected) < 1e-9


def test_duplicate_raw_rows_in_the_same_session_topic_collapse_to_one_cell():
    """The spec's own instruction: count distinct (session, topic)
    cells, not raw evidence rows. Three raw rows in the SAME
    (session, topic) must produce the IDENTICAL confidence as one."""
    session_id = uuid4()
    one_row = [_evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=session_id)]
    three_rows = [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=session_id)
        for _ in range(3)
    ]
    confidence_one = compute_confidence(one_row, ClaimWritePolicy.LOCKED, now=NOW)
    confidence_three = compute_confidence(three_rows, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence_one == confidence_three


def test_row_with_no_real_choice_is_ignored_even_though_it_is_real_evidence():
    """The exact failure mode a live e2e run caught: a subject-kind pick
    (contradiction_was_possible=False) is uncontestable and must
    contribute nothing to s/f, even though the row is a genuine,
    correctly-recorded evidence row. With the only row filtered out,
    confidence must fall back to the no-evidence prior (0.5), not to
    whatever a single supporting cell would have produced."""
    evidence = [_evidence(EvidenceDirection.SUPPORTS, contradiction_was_possible=False)]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence == 0.5


def test_row_where_the_test_never_fired_is_also_ignored():
    evidence = [_evidence(EvidenceDirection.SUPPORTS, test_fired=False)]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence == 0.5


def test_ineligible_rows_do_not_dilute_eligible_ones():
    """A mix of one real (eligible) cell and several uncontestable
    (ineligible) rows must score IDENTICALLY to the real cell alone --
    the ineligible rows are provenance, not evidence."""
    session_id = uuid4()
    real_alone = [_evidence(EvidenceDirection.SUPPORTS, session_id=session_id, axis=ApproachAxis.BREVITY_DEPTH)]
    real_plus_noise = real_alone + [
        _evidence(EvidenceDirection.SUPPORTS, contradiction_was_possible=False, session_id=uuid4())
        for _ in range(5)
    ]
    confidence_alone = compute_confidence(real_alone, ClaimWritePolicy.LOCKED, now=NOW)
    confidence_with_noise = compute_confidence(real_plus_noise, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence_alone == confidence_with_noise


def test_same_session_same_axis_different_topics_collapse_to_one_cell():
    """Two eligible rows in the SAME session, testing the SAME axis, but
    labeled with DIFFERENT topics, are one observation of that axis --
    a chatty session isn't two independent samples just because the
    extractor assigned two topic strings."""
    session_id = uuid4()
    one_row = [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=session_id,
                  axis=ApproachAxis.RIGOR_INTUITION)
    ]
    two_rows_two_topics = one_row + [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-b", session_id=session_id,
                  axis=ApproachAxis.RIGOR_INTUITION)
    ]
    confidence_one = compute_confidence(one_row, ClaimWritePolicy.LOCKED, now=NOW)
    confidence_two = compute_confidence(two_rows_two_topics, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence_one == confidence_two


def test_same_session_different_axis_are_distinct_cells():
    session_id = uuid4()
    evidence = [
        _evidence(EvidenceDirection.SUPPORTS, session_id=session_id, axis=ApproachAxis.RIGOR_INTUITION),
        _evidence(EvidenceDirection.SUPPORTS, session_id=session_id, axis=ApproachAxis.BREVITY_DEPTH),
    ]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    expected = 1 - 0.5**3  # two distinct cells -> alpha = 1 + 2
    assert abs(confidence - expected) < 1e-9


def test_rows_with_no_axis_fall_back_to_session_topic_collapsing():
    """Evidence from a trigger with no live option set (a stated
    preference, an unresolved contradiction) carries axis=None --
    those rows must still collapse by (session, topic), same as before
    this field existed, rather than each counting as its own cell."""
    session_id = uuid4()
    one_row = [_evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=session_id, axis=None)]
    three_rows_same_topic = one_row + [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=session_id, axis=None)
        for _ in range(2)
    ]
    confidence_one = compute_confidence(one_row, ClaimWritePolicy.LOCKED, now=NOW)
    confidence_three = compute_confidence(three_rows_same_topic, ClaimWritePolicy.LOCKED, now=NOW)
    assert confidence_one == confidence_three


def test_one_fresh_contradicting_cell_locked_no_decay():
    # alpha = 1, beta = 1 + 1.0**0 = 2 -> Beta(1,2): CDF(0.5) = 1-(1-0.5)**2.
    # confidence = 1 - CDF = (1-0.5)**2 = 0.5**2.
    evidence = [_evidence(EvidenceDirection.CONTRADICTS, age_days=0.0)]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    expected = 0.5**2
    assert abs(confidence - expected) < 1e-9


def test_decay_reduces_an_old_cells_contribution():
    cfg = ClaimConfidenceConfig()
    # alpha = 1 + 0.97**10 (slow_drift decay over 10 days), beta = 1.
    evidence = [_evidence(EvidenceDirection.SUPPORTS, age_days=10.0)]
    confidence = compute_confidence(evidence, ClaimWritePolicy.SLOW_DRIFT, config=cfg, now=NOW)
    weight = cfg.slow_drift_decay**10.0
    expected = 1 - 0.5 ** (1 + weight)
    assert abs(confidence - expected) < 1e-9


def test_fast_decay_kills_old_evidence_almost_entirely():
    """fast_decay (lambda=0.6) at 10 days old contributes almost
    nothing -- confidence should sit close to the no-evidence baseline
    of 0.5, well below the locked/no-decay case's 0.75."""
    fresh_locked = compute_confidence(
        [_evidence(EvidenceDirection.SUPPORTS, age_days=0.0)], ClaimWritePolicy.LOCKED, now=NOW
    )
    old_fast_decay = compute_confidence(
        [_evidence(EvidenceDirection.SUPPORTS, age_days=10.0)], ClaimWritePolicy.FAST_DECAY, now=NOW
    )
    assert old_fast_decay < fresh_locked
    assert abs(old_fast_decay - 0.5) < 0.05


def test_locked_never_decays_regardless_of_age():
    fresh = compute_confidence(
        [_evidence(EvidenceDirection.SUPPORTS, age_days=0.0)], ClaimWritePolicy.LOCKED, now=NOW
    )
    ancient = compute_confidence(
        [_evidence(EvidenceDirection.SUPPORTS, age_days=3650.0)], ClaimWritePolicy.LOCKED, now=NOW
    )
    assert fresh == ancient


def test_population_prior_shifts_the_baseline():
    """A population prior with the same shape as one supporting cell
    (alpha0=2, beta0=1) with ZERO evidence must equal one fresh
    supporting cell under the uninformative default (alpha0=1, beta0=1)."""
    cfg = ClaimConfidenceConfig()
    with_prior = compute_confidence(
        [], ClaimWritePolicy.LOCKED, config=cfg, now=NOW, population_prior=(2.0, 1.0)
    )
    with_one_cell_default_prior = compute_confidence(
        [_evidence(EvidenceDirection.SUPPORTS, age_days=0.0)], ClaimWritePolicy.LOCKED, now=NOW
    )
    assert abs(with_prior - with_one_cell_default_prior) < 1e-9


# ─────────────────────────── clamp ──────────────────────────


def test_clamp_caps_high_confidence_at_point_seven():
    assert clamp_confidence_for_decisions(0.99) == 0.7


def test_clamp_floors_low_confidence_at_point_three():
    assert clamp_confidence_for_decisions(0.01) == 0.3


def test_clamp_passes_through_the_middle_unchanged():
    assert clamp_confidence_for_decisions(0.5) == 0.5


# ─────────────────────────── promotion ──────────────────────────


def test_promotion_never_fires_under_the_default_clamp():
    """Deliberate, not a bug: decision_clamp_max=0.7 < promotion_
    confidence_threshold=0.8, so no amount of evidence can promote a
    claim right now -- see evaluate_promotion's own docstring. This
    test exists to catch an accidental "fix" of that gap rather than a
    deliberate, reviewed change to the clamp."""
    lots_of_evidence = [
        _evidence(EvidenceDirection.SUPPORTS, topic=f"topic-{i}", session_id=uuid4())
        for i in range(20)
    ]
    should_promote, raw_confidence = evaluate_promotion(
        lots_of_evidence, ClaimWritePolicy.LOCKED, now=NOW
    )
    assert should_promote is False
    assert raw_confidence > 0.8  # the RAW math is confident; the clamp still blocks the decision


def test_promotion_requires_the_topic_gate_even_if_the_clamp_is_lifted():
    """Isolates the topic-count gate from the confidence clamp by
    passing a config whose clamp doesn't bind -- confirms confidence
    alone, within one topic, is not enough."""
    cfg = ClaimConfidenceConfig(decision_clamp_max=1.0)
    one_topic_evidence = [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=uuid4())
        for _ in range(10)
    ]
    should_promote, _ = evaluate_promotion(one_topic_evidence, ClaimWritePolicy.LOCKED, cfg, now=NOW)
    assert should_promote is False

    two_topic_evidence = [
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-a", session_id=uuid4()),
        _evidence(EvidenceDirection.SUPPORTS, topic="topic-b", session_id=uuid4()),
    ]
    should_promote, _ = evaluate_promotion(two_topic_evidence, ClaimWritePolicy.LOCKED, cfg, now=NOW)
    assert should_promote is True


# ─────────────────────────── derived contradiction ──────────────────


def test_a_single_contradicting_cell_never_derives_contradiction():
    """One eligible contradicting cell already drops confidence below
    the default floor (0.25 < 0.3) -- but it supplies only ONE session,
    below contradiction_min_sessions=2, so the claim must not close on
    it alone. This is the exact behavior change from the old one-shot
    design (see evaluate_contradiction's own docstring)."""
    evidence = [_evidence(EvidenceDirection.CONTRADICTS, age_days=0.0)]
    assert compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW) < 0.3
    assert evaluate_contradiction(evidence, ClaimWritePolicy.LOCKED, now=NOW) is False


def test_two_contradicting_cells_from_distinct_sessions_derive_contradiction():
    # No supporting evidence at all: alpha=1, beta=1+2=3 -> Beta(1,3):
    # CDF(0.5) = 1-0.5**3 = 0.875 -> confidence = 0.125, well under 0.3.
    evidence = [
        _evidence(EvidenceDirection.CONTRADICTS, topic="a", session_id=uuid4()),
        _evidence(EvidenceDirection.CONTRADICTS, topic="b", session_id=uuid4()),
    ]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    assert abs(confidence - 0.125) < 1e-9
    assert evaluate_contradiction(evidence, ClaimWritePolicy.LOCKED, now=NOW) is True


def test_two_contradicting_cells_in_the_same_session_still_count_as_one_session():
    """Same raw confidence as the distinct-sessions case above (two
    cells, different topics), but both cells share one session_id --
    distinct SESSIONS is what's gated, not distinct cells, so this must
    NOT derive contradiction even though confidence is just as low."""
    session_id = uuid4()
    evidence = [
        _evidence(EvidenceDirection.CONTRADICTS, topic="a", session_id=session_id),
        _evidence(EvidenceDirection.CONTRADICTS, topic="b", session_id=session_id),
    ]
    confidence = compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW)
    assert abs(confidence - 0.125) < 1e-9  # confidence alone would suggest contradiction
    assert evaluate_contradiction(evidence, ClaimWritePolicy.LOCKED, now=NOW) is False


def test_ineligible_contradicting_rows_never_derive_contradiction():
    """Many ineligible (uncontestable) contradicting rows across many
    distinct sessions still contribute nothing -- same eligibility
    filter compute_confidence itself applies."""
    evidence = [
        _evidence(EvidenceDirection.CONTRADICTS, session_id=uuid4(), contradiction_was_possible=False)
        for _ in range(5)
    ]
    assert compute_confidence(evidence, ClaimWritePolicy.LOCKED, now=NOW) == 0.5
    assert evaluate_contradiction(evidence, ClaimWritePolicy.LOCKED, now=NOW) is False


def test_contradiction_requires_confidence_to_actually_cross_the_floor_too():
    """Two contradicting sessions against one supporting cell satisfies
    the session-count gate but confidence (0.3125) doesn't clear the
    default floor (0.3) -- both conditions are required, session count
    alone isn't sufficient. A third contradicting session pushes
    confidence to 0.1875, crossing the floor."""
    session_support = uuid4()
    two_sessions = [
        _evidence(EvidenceDirection.SUPPORTS, topic="s", session_id=session_support),
        _evidence(EvidenceDirection.CONTRADICTS, topic="a", session_id=uuid4()),
        _evidence(EvidenceDirection.CONTRADICTS, topic="b", session_id=uuid4()),
    ]
    confidence_two = compute_confidence(two_sessions, ClaimWritePolicy.LOCKED, now=NOW)
    assert abs(confidence_two - 0.3125) < 1e-9
    assert evaluate_contradiction(two_sessions, ClaimWritePolicy.LOCKED, now=NOW) is False

    three_sessions = two_sessions + [
        _evidence(EvidenceDirection.CONTRADICTS, topic="c", session_id=uuid4())
    ]
    confidence_three = compute_confidence(three_sessions, ClaimWritePolicy.LOCKED, now=NOW)
    assert abs(confidence_three - 0.1875) < 1e-9
    assert evaluate_contradiction(three_sessions, ClaimWritePolicy.LOCKED, now=NOW) is True
