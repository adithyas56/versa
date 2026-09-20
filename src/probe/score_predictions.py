"""The calibration check `compute_confidence`'s own docstring points
at: `clamp_confidence_for_decisions` holds every claim's decision-
facing confidence to [0.3, 0.7] "until a reliability diagram exists to
show the raw number is actually calibrated." Nothing in this codebase
has ever built that diagram — `wants_steps_shown` reaching 0.875 with
3 distinct topics in a real run (2026-09-14 synthetic verification) is
arithmetic, not a measured probability, until this module says
otherwise. This is that check.

THE CORE IDEA: a claim's `claim_evidence` history already records,
for every eligible episode, whether it confirmed the claim's asserted
side (`direction=supports`) or falsified it (`direction=contradicts`).
That is exactly a scored prediction — no new "run this claim's test
against a future episode" machinery is needed, because reconciliation
already did that scoring the moment the evidence row was written. What
was missing is only the retrospective view: at each point where a new
piece of eligible evidence arrived, what confidence would
`compute_confidence` have reported using ONLY the evidence strictly
BEFORE it? That number is the "prediction"; whether the new row
supports or contradicts is the "outcome." Bucketing (prediction,
outcome) pairs by predicted confidence and checking whether the
observed hit rate in each bucket tracks the bucket's own confidence
IS a reliability diagram.

WHY THIS DOESN'T DOUBLE AS A LOOK-AHEAD CHEAT: `compute_prediction_trials`
computes each trial's `predicted_confidence` from evidence rows dated
strictly before the trial row (`now=row.created_at` passed into the
existing, unmodified `compute_confidence`), so it reuses the exact
decay/cell-collapsing math production already runs — this is not a
new, parallel confidence formula that might disagree with the one
that actually gates promotion.

WHY A CLAIM'S FIRST ELIGIBLE ROW ISN'T SCORED: with zero prior
evidence, "predicted confidence" is just the uninformative population
prior (0.5 by default) — scoring that trial would test whether the
PRIOR is calibrated, not whether this claim's confidence tracks
anything. A claim needs at least one prior eligible row before its
next trial means something.

WHY A CONTRADICTING ROW ISN'T TREATED SPECIALLY: `contradicted` is a
DERIVED status (`evaluate_contradiction`) requiring confidence to have
genuinely fallen AND contrary evidence from multiple distinct sessions
-- a single contradicting row almost never ends a claim by itself
(earlier, `reconcile_candidate` called `store.contradict()` the moment
ANY eligible CONTRADICTS direction was decided, making a contradicting
row always the claim's last; a live run found this orphaned every
later same-axis episode into a fresh singleton claim the moment one
noisy pick landed). So a claim's evidence history now usually keeps
growing past a miss, which only means MORE trials accrue per claim,
not fewer -- this module still counts TRIALS, not claims, and every
trial (hit or miss) is recorded in the bucket its OWN pre-trial
confidence belongs to, using only information available before it
happened. A model whose 0.9-confidence claims are actually wrong 30%
of the time will show that as excess misses landing in the 0.9 bucket
across many claims' rows; a well-calibrated model will show few misses
there because few SHOULD occur. In the rare case a claim IS eventually
derived-contradicted, its trial history up to that point is scored
exactly the same way -- nothing here depends on where or whether a
claim's evidence history ends.

READ-ONLY, NO PERSISTENCE (yet): this queries `claims`/`claim_evidence`
and computes in memory; it writes nothing new to the database and
makes no LLM call, so none of CLAUDE.md's append-only invariants apply
to it. If calibration tracking over time becomes worth persisting,
that is a deliberate follow-up, not something to add speculatively
here.

A SECOND READER, NOT NEW MACHINERY: `compute_capability_prediction_trials`/
`score_capability_predictions_for_all_learners` below apply the
identical idea to `capability_claims`/`capability_evidence`
(capability.py) — a capability evidence row's `direction` is already a
scored prediction the same way a preference one is, so the only new
code is reading a second table and calling
`capability.compute_capability_confidence` instead of
`claims.compute_confidence`. `PredictionTrial`/`CalibrationBin`/
`bucket_trials`/`brier_score`/`format_reliability_diagram` are reused
completely unchanged — none of them know or care which claim
vocabulary produced a trial. This exists because locate/predict
evidence used to accumulate silently with no way to check whether
their confidence meant anything, the same position the preference side
was in before this module existed (see the module-opening paragraph
above) — except capability claims have no promotion/clamp gate at all
yet (deliberately, see capability.py's own docstring), so uncalibrated
capability confidence is currently worse-covered than the preference
side ever was, not better. Checked directly before building this: as
of this module's own writing nothing outside `instruments.py`/`loop.py`
/`webserver.py`'s WRITE paths reads a `CapabilityClaim` at all — no
retrieval, no prompt-building, no rendering — so today a wrong number
here is inert, not yet acted on. That is exactly why now is the right
time to add this reader, before a router or a rendering path gives
that number a job to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from probe.capability import CapabilityClaimStore, CapabilityConfidenceConfig, compute_capability_confidence
from probe.claims import ClaimConfidenceConfig, ClaimStore, compute_confidence
from probe.models import (
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityLabel,
    Claim,
    ClaimEvidence,
    EvidenceDirection,
    EvidenceSource,
)

DEFAULT_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@dataclass
class PredictionTrial:
    claim_id: UUID
    evidence_id: UUID
    occurred_at: datetime
    predicted_confidence: float
    hit: bool


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    n_trials: int
    n_hits: int
    mean_predicted_confidence: float | None

    @property
    def observed_hit_rate(self) -> float | None:
        return self.n_hits / self.n_trials if self.n_trials else None


def compute_prediction_trials(
    claim: Claim,
    evidence_rows: list[ClaimEvidence],
    config: ClaimConfidenceConfig | None = None,
    exclude_contaminated: bool = False,
    source_filter: EvidenceSource | None = None,
    topic_prefix: str | None = None,
) -> list[PredictionTrial]:
    """`evidence_rows` must already be in the order they were recorded
    (`ClaimStore.list_evidence` guarantees this via `ORDER BY seq`).
    Filters to eligible rows (`test_fired AND contradiction_was_possible`
    — the same filter `compute_confidence` itself applies, see this
    module's own docstring for why an uncontestable pick isn't a
    trial), then scores every eligible row after the first against the
    confidence `compute_confidence` would have reported from the
    eligible rows strictly before it. A claim with fewer than 2
    eligible rows produces no trials at all.

    `exclude_contaminated`, when true, drops any row with a
    `provenance_note` set (a human-diagnosed harness bug, not genuine
    learner behavior -- see `ClaimStore.mark_evidence_contaminated`)
    BEFORE the eligibility filter, so it contributes to neither a
    trial's own outcome nor the prior-evidence context used to predict
    a later trial. Production confidence (`compute_confidence` as
    actually called by `ClaimStore.refresh`) never does this -- a
    contaminated row is honest about what happened in that session and
    still counts there. This flag exists only so the two pictures
    (with and without flagged rows) can be compared side by side.

    `source_filter`, when given, restricts to rows from that
    `EvidenceSource` alone (as if the claim's whole evidence history
    were only ever that source) -- the gate a new evidence-producing
    path (instruments.py) has to clear before it's trusted alongside
    the existing one: does THIS source's own calibration curve look
    like the established one, checked in isolation rather than
    diluted together.

    `topic_prefix`, when given, restricts further by `topic.startswith(...)`
    -- `write_instrument_evidence` writes `topic=f"instrument:{primitive}"`,
    so this is how "--split-by-source" tells locate's own curve apart
    from predict's, both of which share `source=instrument`. Different
    axis than `source_filter` (mechanism vs. specific method), so both
    can be combined."""
    cfg = config or ClaimConfidenceConfig()
    if exclude_contaminated:
        evidence_rows = [r for r in evidence_rows if r.provenance_note is None]
    if source_filter is not None:
        evidence_rows = [r for r in evidence_rows if r.source == source_filter]
    if topic_prefix is not None:
        evidence_rows = [r for r in evidence_rows if r.topic.startswith(topic_prefix)]
    eligible = [r for r in evidence_rows if r.test_fired and r.contradiction_was_possible]
    trials: list[PredictionTrial] = []
    for i in range(1, len(eligible)):
        prior_rows = eligible[:i]
        row = eligible[i]
        predicted_confidence = compute_confidence(
            prior_rows, claim.write_policy, cfg, now=row.created_at
        )
        trials.append(
            PredictionTrial(
                claim_id=claim.id,
                evidence_id=row.id,
                occurred_at=row.created_at,
                predicted_confidence=predicted_confidence,
                hit=row.direction is EvidenceDirection.SUPPORTS,
            )
        )
    return trials


def bucket_trials(
    trials: list[PredictionTrial], bin_edges: list[float] | None = None
) -> list[CalibrationBin]:
    """Fixed-edge bucketing (not quantile-based): with the small trial
    counts this codebase will have for a long time, quantile bins would
    constantly reshape themselves as new data arrives, making two runs
    incomparable. Fixed edges stay meaningful even when most bins are
    empty -- an empty bin is itself the honest answer ("no data here
    yet"), not something to hide by merging it away."""
    edges = bin_edges or DEFAULT_BIN_EDGES
    bins: list[CalibrationBin] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = [
            t for t in trials
            if lower <= t.predicted_confidence < upper
            or (upper == edges[-1] and t.predicted_confidence == upper)
        ]
        n = len(in_bin)
        n_hits = sum(1 for t in in_bin if t.hit)
        mean_predicted = sum(t.predicted_confidence for t in in_bin) / n if n else None
        bins.append(CalibrationBin(lower, upper, n, n_hits, mean_predicted))
    return bins


def brier_score(trials: list[PredictionTrial]) -> float | None:
    """Mean squared error between predicted confidence and the binary
    outcome (1.0 for a hit, 0.0 for a miss) -- 0 is perfect, 0.25 is
    what an uninformative always-0.5 forecaster scores. A single
    number to watch trend over time; the bucket table is where the
    actual miscalibration (if any) would show up."""
    if not trials:
        return None
    return sum((t.predicted_confidence - (1.0 if t.hit else 0.0)) ** 2 for t in trials) / len(trials)


def format_reliability_diagram(
    bins: list[CalibrationBin], overall_brier: float | None, total_trials_computed: int | None = None
) -> str:
    """`total_trials_computed`, when given, is the raw count of trials
    BEFORE bucketing (`len(all_trials)` at the call site) -- printed
    alongside the bucketed sum so a gap between the two is visible
    rather than silently absorbed. `DEFAULT_BIN_EDGES` now spans the
    full [0.0, 1.0] a confidence value can take, so the two numbers
    should always match; this argument exists so a future edge-range
    mistake (or a non-default `bin_edges` a caller narrows) shows up
    immediately instead of requiring someone to notice a coincidence,
    which is exactly how the previous gap (edges starting at 0.5,
    silently dropping every trial predicted below chance) went
    unnoticed until two totals happened to come out equal by luck."""
    lines = [f"{'confidence bucket':>18}  {'n':>5}  {'hits':>5}  {'observed':>9}  {'predicted':>9}"]
    for i, b in enumerate(bins):
        closing = "]" if i == len(bins) - 1 else ")"
        bucket_label = f"[{b.lower:.1f}, {b.upper:.1f}{closing}"
        observed = f"{b.observed_hit_rate:.3f}" if b.observed_hit_rate is not None else "--"
        predicted = f"{b.mean_predicted_confidence:.3f}" if b.mean_predicted_confidence is not None else "--"
        lines.append(
            f"{bucket_label:>18}  {b.n_trials:>5}  {b.n_hits:>5}  {observed:>9}  {predicted:>9}"
        )
    bucketed_total = sum(b.n_trials for b in bins)
    brier_line = f"{overall_brier:.4f}" if overall_brier is not None else "n/a (no trials yet)"
    if total_trials_computed is None:
        lines.append(f"\ntotal trials: {bucketed_total}   brier score: {brier_line}")
    else:
        mismatch = (
            f"  [WARNING: {total_trials_computed - bucketed_total} scored trial(s) fell "
            f"outside the bucket range]"
            if total_trials_computed != bucketed_total else ""
        )
        lines.append(
            f"\ntotal trials: {bucketed_total} (of {total_trials_computed} scored)"
            f"   brier score: {brier_line}{mismatch}"
        )
    return "\n".join(lines)


async def score_predictions_for_all_learners(
    store: ClaimStore,
    config: ClaimConfidenceConfig | None = None,
    bin_edges: list[float] | None = None,
    exclude_contaminated: bool = False,
    source_filter: EvidenceSource | None = None,
    topic_prefix: str | None = None,
) -> tuple[list[CalibrationBin], float | None, int]:
    """The I/O boundary: pulls every claim across every learner (this
    is a calibration question about the confidence FORMULA, not about
    any one learner's model, so it deliberately pools across all of
    them — same "batch job, cross-learner" precedent as
    population_patterns.aggregate_population_patterns), scores every
    claim's evidence history, and buckets the result. `exclude_
    contaminated`/`source_filter`/`topic_prefix` pass straight through
    to `compute_prediction_trials` — see its own docstring for all
    three.

    Returns `(bins, overall_brier, total_trials_computed)` -- the third
    value is `len(all_trials)` BEFORE bucketing, for
    `format_reliability_diagram`'s own mismatch check (see its
    docstring for why that visibility matters)."""
    claims = await store.list_all()
    all_trials: list[PredictionTrial] = []
    for claim in claims:
        evidence = await store.list_evidence(claim.id)
        all_trials.extend(
            compute_prediction_trials(
                claim, evidence, config, exclude_contaminated, source_filter, topic_prefix
            )
        )
    bins = bucket_trials(all_trials, bin_edges)
    return bins, brier_score(all_trials), len(all_trials)


# ─────────────────────────── capability-side reader ─────────────────────


def compute_capability_prediction_trials(
    claim: CapabilityClaim,
    evidence_rows: list[CapabilityEvidence],
    config: CapabilityConfidenceConfig | None = None,
    exclude_contaminated: bool = False,
    skill_filter: CapabilityLabel | None = None,
) -> list[PredictionTrial]:
    """The capability-side counterpart to `compute_prediction_trials` —
    identical trial-extraction logic (score every eligible row after
    the first against `compute_capability_confidence` from the
    eligible rows strictly before it), reusing `PredictionTrial`
    unchanged: it doesn't know or care which claim vocabulary produced
    a trial. `skill_filter`, when given, restricts to one
    `CapabilityLabel` alone — the capability-side equivalent of
    `compute_prediction_trials`'s `topic_prefix` (locate's own curve
    told apart from predict's), but exact-match on `skill` rather than
    a topic-string prefix, since `CapabilityEvidence` carries `skill`
    directly rather than deriving it from a formatted topic string."""
    cfg = config or CapabilityConfidenceConfig()
    if exclude_contaminated:
        evidence_rows = [r for r in evidence_rows if r.provenance_note is None]
    if skill_filter is not None:
        evidence_rows = [r for r in evidence_rows if r.skill == skill_filter]
    eligible = [r for r in evidence_rows if r.test_fired and r.contradiction_was_possible]
    trials: list[PredictionTrial] = []
    for i in range(1, len(eligible)):
        prior_rows = eligible[:i]
        row = eligible[i]
        predicted_confidence = compute_capability_confidence(
            prior_rows, claim.write_policy, cfg, now=row.created_at
        )
        trials.append(
            PredictionTrial(
                claim_id=claim.id,
                evidence_id=row.id,
                occurred_at=row.created_at,
                predicted_confidence=predicted_confidence,
                hit=row.direction is EvidenceDirection.SUPPORTS,
            )
        )
    return trials


async def score_capability_predictions_for_all_learners(
    store: CapabilityClaimStore,
    config: CapabilityConfidenceConfig | None = None,
    bin_edges: list[float] | None = None,
    exclude_contaminated: bool = False,
    skill_filter: CapabilityLabel | None = None,
) -> tuple[list[CalibrationBin], float | None, int]:
    """The capability-side counterpart to
    `score_predictions_for_all_learners` — same cross-learner batch
    read, same bucketing, over `capability_claims`/`capability_evidence`
    instead. Returns the identical 3-tuple shape."""
    claims = await store.list_all()
    all_trials: list[PredictionTrial] = []
    for claim in claims:
        evidence = await store.list_evidence(claim.id)
        all_trials.extend(
            compute_capability_prediction_trials(claim, evidence, config, exclude_contaminated, skill_filter)
        )
    bins = bucket_trials(all_trials, bin_edges)
    return bins, brier_score(all_trials), len(all_trials)
