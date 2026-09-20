"""The claim layer (migration 042) — a durable, cross-session model of
a learner's standing traits, built ONLY from episodes where something
was actually surprising, never by reading every session. See this
review's own framing: a fluent model produces a plausible-sounding
claim for any input, and extracting everywhere fills the store with
confabulation that reads identically to a real finding. Everything in
this module exists to keep that from happening.

FOUR stages, deliberately kept separate:

`select_extraction_candidates` (per session, at session-end) ranks that
session's interactions by PREDICTION ERROR — how much probability the
already-fired, frozen `Prediction` gave to the option NOT taken — and
returns only the top few, plus any turn classified `contradicted_intent`
or carrying a stated preference. Most turns in most sessions produce
nothing here, by design.

`ClaimExtractor` (one LLM call per selected episode) answers a single,
narrow question — "what would have to be true about this person for
this episode to have been unsurprising?" — never "what does this
reveal," which invites insight-shaped prose instead of a testable
claim. Every candidate it proposes must carry a `test`: a prediction
over a FUTURE option set, not a description of the past one. A
candidate without one is rejected before it ever reaches storage.
Competing candidates from one underdetermined episode are kept side by
side, not collapsed to "the most plausible."

`reconcile_candidate` matches a new candidate against this learner's
existing claims AXIS FIRST, statement second. When the source episode
has a live axis, any non-contradicted claim already carrying evidence
on that same axis for this learner IS the same claim — axis identity
is an exact match, not a similarity judgment, because axis is a
closed, structural label persisted at option-generation time (see
`disambiguate.py`), not text two different extractions might phrase
differently. A live e2e run found the alternative (matching purely by
statement-embedding similarity) fragmenting one trait into near-
duplicate claims whenever the extractor worded two episodes of the
same axis differently enough to fall under the similarity threshold —
and each fragment then carries only its own share of the evidence, so
a trait confirmed four times across four topics could sit as two
claims with two topics each, neither reaching the promotion gate's
>=2-distinct-topics requirement. Similarity is still used, but only to
disambiguate WHICH claim a candidate continues when more than one
already shares that axis (a tie, or a pre-fix legacy fragment), and as
the sole matching strategy when the episode has no axis at all (a
stated-preference or contradicted_intent trigger). Either way, a match
appends an evidence row (supports or contradicts, decided by comparing
the candidate's `value` against the matched claim's); no match creates
a new `claims` row. A claim's `statement`/`test`/`value` are never
edited after creation — see `Claim`'s own docstring for why status/
confidence changing in place is still "append-only" in this
codebase's sense.

`compute_confidence`/`evaluate_promotion` are pure functions, unit-
tested against hand-computed values (test_claims_confidence.py) — no
I/O, no randomness. `clamp_confidence_for_decisions` caps the value any
DECISION (promotion, rendering) is allowed to see at [0.3, 0.7] until a
reliability diagram exists to show the raw number is actually
calibrated — see that function's own docstring for the direct,
intended consequence: no claim can promote under the current
configuration. That is correct, not a bug to route around.

`render_claim_constraint` reuses `interaction_nodes.
render_structural_requirement` verbatim — the same label -> hand-
written-imperative, no-LLM-on-the-read-path lookup that worked 5/5 for
stated preferences, per this feature's own spec. Only `status=promoted`
claims render; candidates never do.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import asyncpg
from pydantic import BaseModel, Field

from probe.embeddings import TASK_SIMILARITY, EmbeddingClient
from probe.interaction_nodes import render_structural_requirement
from probe.llm import LLMClient
from probe.models import (
    AmbiguityKind,
    ApproachAxis,
    Claim,
    ClaimCandidate,
    ClaimEvidence,
    ClaimExtractionResult,
    ClaimSource,
    ClaimStatementRecord,
    ClaimStatus,
    ClaimWritePolicy,
    EvidenceDirection,
    StatedPreferenceLabel,
    TurnOutcomeLabel,
)
from probe.row_mapping import assert_row_consumed
from probe.vector_math import cosine_similarity

logger = logging.getLogger(__name__)

CLAIM_EXTRACTOR_VERSION = "extract-claim-v1"

# Labels this codebase's closed vocabulary treats as direct opposites —
# used by reconciliation to decide supports vs. contradicts. A pair not
# listed here (e.g. WANTS_STEPS_SHOWN vs PREFERS_BREVITY, or anything
# involving OTHER) has no defined opposite: reconciliation treats that
# as "not actually the same claim" rather than guessing a direction.
_OPPOSITE_LABELS: dict[StatedPreferenceLabel, StatedPreferenceLabel] = {
    StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT: StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
    StatedPreferenceLabel.RULE_BEFORE_EXAMPLE: StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
    StatedPreferenceLabel.WANTS_ANALOGIES: StatedPreferenceLabel.NO_ANALOGIES,
    StatedPreferenceLabel.NO_ANALOGIES: StatedPreferenceLabel.WANTS_ANALOGIES,
}


# ─────────────────────────── config ──────────────────────────────────


class ExtractionConfig(BaseModel):
    # Separate floor quotas per kind, not one pooled top-N -- a real
    # diagnostic (2026-09-15) found subject-kind picks average 0.60
    # prediction error against 0.42 for approach-kind, so a single
    # surprise-ranked top-N systematically starves approach-kind
    # episodes: subject splits get confidently-but-often-wrongly
    # predicted because a topic prior carries them ("derivatives means
    # calculus"), which is surprise about the WORLD, not about the
    # learner. A fresh learner has no behavioral history yet, so
    # PREDICT:SELECTION has nothing to be confidently wrong about on
    # approach axes -- near-uniform predictions there aren't a sign
    # nothing interesting happened, they're a sign there's no prior to
    # violate. See `select_extraction_candidates`'s own docstring for
    # how approach-kind ranking actually works (coverage before
    # surprise), which is the more important half of this fix.
    top_n_approach: int = 3
    top_n_subject: int = 2
    # Cosine similarity above which a new candidate is treated as "the
    # same claim" as an existing one, rather than a new claim.
    similarity_match_threshold: float = 0.83


class ClaimConfidenceConfig(BaseModel):
    """See `compute_confidence`'s own docstring for the formula this
    parameterizes. `alpha0`/`beta0` are the default (uniform, maximally
    uninformative) Beta prior -- "weak population prior where data
    exists, defaults otherwise" per the spec; no population-level
    source is wired up in this codebase yet (nothing currently
    aggregates a base rate for how often learners pick, say, the
    concrete side of an axis), so every caller gets the uninformative
    default unless it explicitly passes `population_prior` to
    `compute_confidence`."""

    alpha0: float = 1.0
    beta0: float = 1.0
    locked_decay: float = 1.0
    slow_drift_decay: float = 0.97
    fast_decay_decay: float = 0.6
    # Below this, a decision (promotion) may not use the raw confidence
    # at face value -- see `clamp_confidence_for_decisions`.
    decision_clamp_min: float = 0.3
    decision_clamp_max: float = 0.7
    promotion_confidence_threshold: float = 0.8
    promotion_min_distinct_topics: int = 2
    # `contradicted` is DERIVED from these two, not a one-shot flip on
    # the first eligible contradicting row -- see `evaluate_contradiction`'s
    # own docstring for the incident this replaces. A single contrary
    # pick is exactly the kind of noise the Beta posterior (and
    # write_policy=slow_drift's decay) already exists to absorb; it
    # should move the number, not end the claim. Reusing
    # `decision_clamp_min` as the floor is deliberate: a claim already
    # too weak for any decision to act on, AND backed by contrary
    # evidence from multiple distinct sessions (not one noisy episode),
    # has earned a terminal status -- it isn't merely being made inert.
    contradiction_confidence_floor: float = 0.3  # == decision_clamp_min's default, deliberately
    contradiction_min_sessions: int = 2


class ClaimRestatementConfig(BaseModel):
    """Threshold gate for `maybe_restate_claims` -- see that function's
    own docstring. Deliberately conservative on both counts: restating
    on every session would invite wording drift for no gain (a claim
    whose evidence hasn't moved doesn't need new wording), and
    restating before evidence has actually generalized past the
    founding episode's own domain would have nothing true to say that
    the existing wording doesn't already say."""

    min_new_evidence_rows: int = 4
    generator_version: str = "restate-claim-v1"


# ─────────────────────────── confidence math (pure) ───────────────────


def _decay_rate(write_policy: ClaimWritePolicy, config: ClaimConfidenceConfig) -> float:
    return {
        ClaimWritePolicy.LOCKED: config.locked_decay,
        ClaimWritePolicy.SLOW_DRIFT: config.slow_drift_decay,
        ClaimWritePolicy.FAST_DECAY: config.fast_decay_decay,
    }[write_policy]


@dataclass
class _EvidenceCell:
    direction: EvidenceDirection
    age_days: float
    session_id: UUID


def _collapse_to_cells(evidence_rows: list[ClaimEvidence], now: datetime) -> list[_EvidenceCell]:
    """Filters to rows where the claim's test genuinely could have
    failed (`test_fired AND contradiction_was_possible`), THEN groups
    the survivors into cells and collapses each group to ONE, aged by
    its most recent row.

    A real run found the cost of skipping the filter: a claim built
    entirely from `kind=subject` episodes (a learner picking a TOPIC,
    not weighing an approach) reached 0.875 confidence, because every
    row was counted as a confirmation even though none of them could
    have disconfirmed anything -- there was no live "other side" of an
    approach axis being weighed. A subject pick is not evidence about
    HOW this person wants to be taught; it never was, and no amount of
    it should move this number. Rows failing the filter are still
    stored (kept as provenance, visible on the review surface) -- they
    just contribute nothing to `s`/`f`.

    Grouped by (session_id, axis) when `axis` is set -- "count distinct
    (session, topic) cells, not raw evidence rows" per the original
    spec, refined after the same run: two confirmations of the SAME
    axis in the SAME session are one observation regardless of what
    topic label each carried, since that is a single repeated moment,
    not two independent samples. Falls back to (session_id, topic) for
    evidence with no axis at all (a stated-preference or
    contradicted_intent trigger, where there is no axis to have
    repeated)."""
    eligible = [row for row in evidence_rows if row.test_fired and row.contradiction_was_possible]
    latest_at: dict[tuple, datetime] = {}
    direction_of: dict[tuple, EvidenceDirection] = {}
    for row in eligible:
        key = (row.session_id, row.axis) if row.axis is not None else (row.session_id, row.topic, "no_axis")
        if key not in latest_at or row.created_at > latest_at[key]:
            latest_at[key] = row.created_at
            direction_of[key] = row.direction
    cells = []
    for key, created_at in latest_at.items():
        age_days = max(0.0, (now - created_at).total_seconds() / 86400)
        cells.append(_EvidenceCell(direction=direction_of[key], age_days=age_days, session_id=key[0]))
    return cells


def compute_confidence(
    evidence_rows: list[ClaimEvidence],
    write_policy: ClaimWritePolicy,
    config: ClaimConfidenceConfig | None = None,
    now: datetime | None = None,
    population_prior: tuple[float, float] | None = None,
    k: int = 2,
) -> float:
    """1 - BetaCDF(1/k; alpha0+s, beta0+f) -- the posterior probability
    this learner's true rate of choosing this claim's asserted side
    beats chance (1/k; k=2 for the single-axis approach splits that are
    this codebase's only source of this kind of evidence so far).
    `s`/`f` are decay-weighted counts of DISTINCT (session, axis-or-
    topic) cells (`_collapse_to_cells`), not raw evidence-row counts --
    and only among rows where `test_fired AND contradiction_was_possible`
    (see that function's own docstring for the real failure this
    filter closes: a claim built from `kind=subject` picks, which never
    could have been disconfirmed, must not accrue confidence at all).

    Pure function: no I/O. `now` defaults to the real clock for
    production callers; tests pass a fixed value so hand-computed
    results are exact and reproducible.
    """
    from scipy.stats import beta as beta_dist

    cfg = config or ClaimConfidenceConfig()
    now = now or datetime.now(UTC)
    alpha0, beta0 = population_prior or (cfg.alpha0, cfg.beta0)
    decay = _decay_rate(write_policy, cfg)

    cells = _collapse_to_cells(evidence_rows, now)
    s = sum(decay ** c.age_days for c in cells if c.direction is EvidenceDirection.SUPPORTS)
    f = sum(decay ** c.age_days for c in cells if c.direction is EvidenceDirection.CONTRADICTS)

    alpha = alpha0 + s
    beta_param = beta0 + f
    chance = 1.0 / k
    return float(1.0 - beta_dist.cdf(chance, alpha, beta_param))


def clamp_confidence_for_decisions(
    confidence: float, config: ClaimConfidenceConfig | None = None
) -> float:
    """Clamps to [0.3, 0.7] before ANY decision (promotion, rendering)
    reads this value -- per the spec's own explicit instruction. Not
    yet validated against real outcomes (that needs a reliability
    diagram, which does not exist), so no decision is allowed to act on
    the raw number's full range. The review surface shows the RAW
    (unclamped) value from `compute_confidence` directly -- this clamp
    governs behavior, not visibility.

    Direct, intended consequence: with the default
    `promotion_confidence_threshold` of 0.8 and this clamp's max of
    0.7, no claim can ever satisfy the promotion condition right now.
    That is correct, not a bug -- see `evaluate_promotion`'s own
    docstring.
    """
    cfg = config or ClaimConfidenceConfig()
    return max(cfg.decision_clamp_min, min(cfg.decision_clamp_max, confidence))


def _distinct_supporting_topics(evidence_rows: list[ClaimEvidence]) -> int:
    return len({r.topic for r in evidence_rows if r.direction is EvidenceDirection.SUPPORTS})


def evaluate_promotion(
    evidence_rows: list[ClaimEvidence],
    write_policy: ClaimWritePolicy,
    config: ClaimConfidenceConfig | None = None,
    now: datetime | None = None,
) -> tuple[bool, float]:
    """Returns (should_promote, raw_confidence). `raw_confidence` is
    the UNCLAMPED `compute_confidence` output, for storage/review.
    `should_promote` is decided against the CLAMPED value (see
    `clamp_confidence_for_decisions`) AND requires at least
    `promotion_min_distinct_topics` distinct topics among SUPPORTING
    evidence -- "the cross-topic gate is the important half:
    confirmations all within one subject are a domain fact wearing a
    preference costume," per the spec. Under the current clamp
    configuration, `should_promote` is always False regardless of how
    much evidence accrues -- deliberate, pending calibration."""
    cfg = config or ClaimConfidenceConfig()
    raw = compute_confidence(evidence_rows, write_policy, cfg, now)
    decision_confidence = clamp_confidence_for_decisions(raw, cfg)
    should_promote = (
        decision_confidence > cfg.promotion_confidence_threshold
        and _distinct_supporting_topics(evidence_rows) >= cfg.promotion_min_distinct_topics
    )
    return should_promote, raw


def evaluate_contradiction(
    evidence_rows: list[ClaimEvidence],
    write_policy: ClaimWritePolicy,
    config: ClaimConfidenceConfig | None = None,
    now: datetime | None = None,
) -> bool:
    """Whether a live claim's accumulated evidence has earned the
    terminal `contradicted` status -- DERIVED from the same confidence
    math every other reading of this claim uses, not a one-shot flip
    triggered by a single opposite-value pick.

    A real run found the cost of the one-shot version: a claim closed
    terminally on its very first contradicting episode, in the same
    session that had just supported it, permanently orphaning every
    later same-axis episode into a fresh singleton claim (`find_by_axis`
    won't match a contradicted claim -- see its own docstring). People
    are inconsistent and change slowly; one off-persona pick doesn't
    disprove a standing trait any more than one on-persona pick proves
    one. `compute_confidence`'s Beta posterior already down-weights a
    claim that keeps getting contradicted -- contradiction is supposed
    to move the number, not end the claim.

    So this requires BOTH: raw confidence has fallen below
    `contradiction_confidence_floor` (already too weak for any decision
    to act on -- see `clamp_confidence_for_decisions`), AND that fall is
    backed by eligible contradicting evidence from at least
    `contradiction_min_sessions` DISTINCT sessions (`_collapse_to_cells`
    already collapses same-session repeats to one cell; this counts
    distinct session_ids among the CONTRADICTS cells specifically). One
    bad episode can lower confidence but, by construction, can never
    supply more than one session -- a genuine, repeated reversal is what
    earns the terminal status.
    """
    cfg = config or ClaimConfidenceConfig()
    now = now or datetime.now(UTC)
    cells = _collapse_to_cells(evidence_rows, now)
    raw = compute_confidence(evidence_rows, write_policy, cfg, now)
    contradicting_sessions = {c.session_id for c in cells if c.direction is EvidenceDirection.CONTRADICTS}
    return (
        raw < cfg.contradiction_confidence_floor
        and len(contradicting_sessions) >= cfg.contradiction_min_sessions
    )


# ─────────────────────────── rendering (pure, no LLM) ──────────────────


def render_claim_constraint(claim: Claim) -> str:
    """Reuses `interaction_nodes.render_structural_requirement`
    verbatim -- "the same form that worked 5/5 for stated preferences,"
    per this feature's own spec, rather than a second hand-written
    imperative set to keep in sync with the first. Only ever called for
    a `status=promoted` claim; a candidate is not rendered (see this
    module's own docstring) -- callers are responsible for that check,
    this function does not re-verify it."""
    return render_structural_requirement(claim.value, claim.statement)


# ─────────────────────────── extraction (one LLM call per episode) ────


def _format_options_block(options: list[dict]) -> str:
    lines = []
    for o in options:
        tag = ""
        if o.get("kind") == "approach" and o.get("axis"):
            tag = f" [axis: {o['axis']}]"
        mark = " <-- CHOSEN" if o.get("was_selected") else ""
        lines.append(f'- "{o["option_text"]}"{tag}{mark}')
    return "\n".join(lines)


def _extraction_prompt(
    question_text: str,
    options: list[dict],
    response_text: str | None,
    trigger_reason: str,
    next_question_text: str | None = None,
    episode_kind: str | None = None,
) -> str:
    options_block = _format_options_block(options) if options else "(no option set was offered)"
    response_block = f'\nWhat they were given in response: "{response_text}"\n' if response_text else ""
    # A live diagnostic found the extractor still assigning a teaching-
    # style value to subject-kind (topic-disambiguation) picks despite
    # those being structurally uncontestable -- harmless to confidence/
    # status now that reconcile_candidate gates on eligibility, but
    # still filling the store with rows that mean nothing. This block
    # tells the extractor what kind of choice it's actually looking
    # at, since that information already exists on the option set
    # (kind/axis, migration 041) and shouldn't have to be re-derived
    # from prose.
    kind_block = ""
    if episode_kind == "subject":
        kind_block = (
            "\nThis was a choice between different SUBJECT-MATTER "
            "readings of an ambiguous question (which topic or sense "
            "they meant), not between different ways of teaching the "
            "SAME thing. A subject pick usually reveals what they "
            "wanted to talk about, not how they want to be taught, and "
            "very rarely licenses a claim about a standing teaching-"
            "style preference. Propose a candidate here only if the "
            "choice genuinely reveals something beyond topic intent; "
            "the default, honest answer for a subject-kind pick is an "
            "empty list.\n"
        )
    # A contradicted_intent trigger says the response missed -- what
    # they wanted INSTEAD is in what they asked next, not in the failed
    # episode alone. Without this block, this prompt asks "what does
    # this misread reveal" with the answer withheld -- a real run
    # confirmed that produces zero candidates every time, honestly, but
    # for the wrong reason.
    correction_block = (
        f'\nAfter that response, their very next message was: "{next_question_text}" -- '
        "read together with the response above, this shows what they actually "
        "wanted instead of what they got.\n"
        if next_question_text else ""
    )
    return (
        "EXTRACT:CLAIM\n"
        f'The person\'s message: "{question_text}"\n'
        f"\nWhy this episode was selected for review: {trigger_reason}\n"
        f"\nOptions actually on offer, in full (what they chose against is half "
        f"the information):\n{options_block}\n"
        f"{response_block}"
        f"{correction_block}"
        f"{kind_block}"
        "\nThe question is NOT \"what does this episode reveal about the "
        "person.\" It is: what would have to be TRUE about this person for "
        "this specific choice, given everything else that was genuinely on "
        "offer, to have been UNSURPRISING? Answer only with a standing trait "
        "specific enough to make a real prediction -- not a one-off, "
        "context-specific reaction to this particular question.\n\n"
        "Every candidate you propose must carry a TEST: a prediction about "
        "what this same person would choose in a FUTURE, similarly-shaped "
        "situation -- name the kind of situation and which side they would "
        "pick. A candidate without a genuinely testable prediction must not "
        "be proposed at all.\n\n"
        "If more than one distinct explanation would equally well account "
        "for this choice, propose them as SEPARATE, competing candidates -- "
        "do not pick the one that sounds most plausible and discard the "
        "rest. Reporting that the evidence underdetermines the answer is a "
        "better outcome than guessing.\n\n"
        "If nothing here licenses a real, testable claim about a standing "
        "trait, return an empty list -- that is the expected, normal answer "
        "for most episodes.\n\n"
        "Each candidate's \"value\" must be exactly one of: "
        "concrete_before_abstract, rule_before_example, wants_steps_shown, "
        "prefers_brevity, wants_analogies, no_analogies, other.\n\n"
        'Respond with JSON: [{"statement": "...", "test": "...", "value": '
        '"<one of the labels above>", "topic": "<the domain-neutral subject '
        'area this episode was about>"}, ...]'
    )


def _parse_extraction_response(raw: str) -> ClaimExtractionResult:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ClaimExtractionResult(candidates=[])
    if not isinstance(data, list):
        return ClaimExtractionResult(candidates=[])
    candidates: list[ClaimCandidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        statement, test, topic = item.get("statement"), item.get("test"), item.get("topic")
        if not all(isinstance(x, str) and x.strip() for x in (statement, test, topic)):
            continue  # no test (or no statement/topic) -- rejected, not stored
        try:
            value = StatedPreferenceLabel(item.get("value"))
        except ValueError:
            continue  # an unparseable label is treated the same as "no test": rejected
        candidates.append(
            ClaimCandidate(statement=statement.strip(), test=test.strip(), value=value, topic=topic.strip())
        )
    return ClaimExtractionResult(candidates=candidates)


class ClaimExtractor:
    """One fast-tier call per episode selected by
    `select_extraction_candidates` -- never per turn. See this module's
    own docstring for the exact question this asks and why it's phrased
    that way, and for why an untestable or absent candidate is
    discarded here rather than stored as a weaker claim."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self.last_call_count: int = 0

    async def run(
        self,
        question_text: str,
        options: list[dict],
        response_text: str | None,
        trigger_reason: str,
        next_question_text: str | None = None,
        episode_kind: str | None = None,
    ) -> ClaimExtractionResult:
        self.last_call_count = 0
        raw = await self._llm.complete(
            _extraction_prompt(
                question_text, options, response_text, trigger_reason,
                next_question_text, episode_kind,
            )
        )
        self.last_call_count = 1
        return _parse_extraction_response(raw)


# ─────────────────────────── candidate selection (per session) ────────


@dataclass
class ExtractionCandidate:
    interaction_id: UUID
    reason: str  # human-readable, logged into node_calls via the caller


async def select_extraction_candidates(
    pool: asyncpg.Pool, session_id: UUID, learner_id: UUID, config: ExtractionConfig | None = None
) -> list[ExtractionCandidate]:
    """Ranks this session's resolved, predicted interactions by KIND
    FIRST, with a different ranker per kind, then unions in any
    interaction this session classified `contradicted_intent` or that
    carries a genuine stated preference. Deliberately NOT "every turn":
    most turns contribute nothing here, by design (see this module's
    own docstring).

    SUBJECT-kind: ranked by PREDICTION ERROR (1 minus however much
    probability the frozen `Prediction` gave the option actually
    taken) -- surprise is a fine signal here, a topic pick that
    violated what the predictor expected is genuinely informative.
    Top `config.top_n_subject`.

    APPROACH-kind: NOT surprise-ranked by default. An axis with no
    ELIGIBLE evidence yet for this learner (no claim_evidence row
    where `test_fired AND contradiction_was_possible`) is ranked
    ahead of everything else, regardless of its prediction error --
    "coverage before surprise." A fresh learner has no behavioral
    history, so PREDICT:SELECTION has nothing to be confidently wrong
    about on an approach axis yet; a near-uniform prediction there
    isn't evidence the pick was unremarkable, it's evidence there was
    no prior to violate. Once an axis already has real evidence,
    surprise becomes meaningful for it and takes over -- an approach
    pick on an ALREADY-COVERED axis competes for the remaining slots
    by prediction error, same as subject-kind does. Top
    `config.top_n_approach`, uncovered axes filling first."""
    cfg = config or ExtractionConfig()

    async with pool.acquire() as conn:
        surprise_rows = await conn.fetch(
            """
            SELECT p.interaction_id, p.predicted_scores, io.option_id AS selected_option_id,
                   io.kind, io.axis
            FROM predictions p
            JOIN interaction_options io
                ON io.interaction_id = p.interaction_id AND io.was_selected
            JOIN interactions i ON i.id = p.interaction_id
            WHERE i.session_id = $1
            """,
            session_id,
        )
        covered_axis_rows = await conn.fetch(
            """
            SELECT DISTINCT axis FROM claim_evidence
            WHERE learner_id = $1 AND test_fired AND contradiction_was_possible
                AND axis IS NOT NULL
            """,
            learner_id,
        )
        contradicted_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (t.interaction_id) t.interaction_id
            FROM turn_outcomes t
            JOIN interactions i ON i.id = t.interaction_id
            WHERE i.session_id = $1
            ORDER BY t.interaction_id, t.seq DESC
            """,
            session_id,
        )
        stated_pref_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (sp.interaction_id) sp.interaction_id, sp.has_preference
            FROM stated_preferences sp
            JOIN interactions i ON i.id = sp.interaction_id
            WHERE i.session_id = $1
            ORDER BY sp.interaction_id, sp.seq DESC
            """,
            session_id,
        )
        # contradicted_intent needs the actual label, fetched separately
        # since the DISTINCT ON above only grabbed the interaction_id.
        outcome_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (t.interaction_id) t.interaction_id, t.outcome
            FROM turn_outcomes t
            JOIN interactions i ON i.id = t.interaction_id
            WHERE i.session_id = $1
            ORDER BY t.interaction_id, t.seq DESC
            """,
            session_id,
        )

    covered_axes = {r["axis"] for r in covered_axis_rows}

    approach_uncovered = []
    approach_covered = []
    subject = []
    for row in surprise_rows:
        scores = row["predicted_scores"] or {}
        selected_prob = float(scores.get(str(row["selected_option_id"]), 0.0))
        prediction_error = 1.0 - selected_prob
        entry = (prediction_error, row["interaction_id"], row["axis"])
        if row["kind"] == AmbiguityKind.APPROACH.value:
            (approach_covered if row["axis"] in covered_axes else approach_uncovered).append(entry)
        else:
            subject.append(entry)

    # Uncovered axes fill first regardless of prediction error --
    # "coverage before surprise" (see this function's own docstring).
    # Sorting by prediction_error within each bucket is only a
    # deterministic tie-break, not a claim that surprise matters here.
    approach_uncovered.sort(key=lambda e: e[0], reverse=True)
    approach_covered.sort(key=lambda e: e[0], reverse=True)
    subject.sort(key=lambda e: e[0], reverse=True)

    candidates: dict[UUID, str] = {}
    approach_uncovered_selected = approach_uncovered[: cfg.top_n_approach]
    remaining = cfg.top_n_approach - len(approach_uncovered_selected)
    approach_covered_selected = approach_covered[:remaining] if remaining > 0 else []

    for prediction_error, interaction_id, axis in approach_uncovered_selected:
        candidates[interaction_id] = (
            f"approach-kind pick on axis '{axis}', which has no eligible evidence "
            f"yet for this learner -- coverage takes priority over surprise until "
            f"an axis has real evidence"
        )
    for prediction_error, interaction_id, axis in approach_covered_selected:
        candidates[interaction_id] = (
            f"approach-kind pick on axis '{axis}' (already has eligible evidence), "
            f"ranked in this session's top {cfg.top_n_approach} remaining approach "
            f"slot(s) by prediction error ({prediction_error:.2f} probability on "
            f"the option not taken)"
        )
    for prediction_error, interaction_id, axis in subject[: cfg.top_n_subject]:
        candidates[interaction_id] = (
            f"subject-kind pick, ranked in this session's top {cfg.top_n_subject} "
            f"by prediction error ({prediction_error:.2f} probability on the "
            f"option not taken)"
        )
    for row in outcome_rows:
        if row["outcome"] == TurnOutcomeLabel.CONTRADICTED_INTENT.value:
            candidates.setdefault(row["interaction_id"], "classified contradicted_intent")
    for row in stated_pref_rows:
        if row["has_preference"]:
            candidates.setdefault(row["interaction_id"], "carries an explicitly stated preference")

    return [ExtractionCandidate(interaction_id=iid, reason=reason) for iid, reason in candidates.items()]


async def _load_episode(pool: asyncpg.Pool, interaction_id: UUID) -> dict | None:
    """Everything ClaimExtractor's prompt needs about one episode --
    the question actually asked (never the option-copy on a
    click-resolution turn, same rule as everywhere else in this
    codebase), the response given, the FULL option set with
    kind/axis/selection (migration 041's persisted columns -- no
    reverse-engineering from text), and -- a real run's own finding --
    the outcome classification AND `next_question_text` for this
    interaction. A `contradicted_intent` classification says the
    response missed; what the learner actually wanted instead is in
    what they asked NEXT, which is exactly what `next_question_text`
    holds. Without it, the extractor was being asked "what does this
    misread reveal" with the answer withheld -- an empty candidate list
    was the honest response to that, not evidence the anchor doesn't
    work."""
    async with pool.acquire() as conn:
        interaction = await conn.fetchrow(
            "SELECT id, session_id, learner_id, question_text, originating_question, "
            "question_author, response_text FROM interactions WHERE id = $1",
            interaction_id,
        )
        if interaction is None:
            return None
        options = await conn.fetch(
            "SELECT option_text, was_selected, kind, axis FROM interaction_options "
            "WHERE interaction_id = $1 ORDER BY shown_position",
            interaction_id,
        )
        outcome_row = await conn.fetchrow(
            "SELECT outcome, next_question_text FROM turn_outcomes "
            "WHERE interaction_id = $1 ORDER BY seq DESC LIMIT 1",
            interaction_id,
        )
        # The options actually on offer for a "chose against" episode
        # live on the OFFER turn, not the click-resolution turn that
        # follows it -- if this interaction has none of its own (it IS
        # the click), look at the turn it resolved.
        if not options and interaction["question_author"] == "system_option":
            offer = await conn.fetchrow(
                "SELECT id FROM interactions WHERE session_id = $1 "
                "AND question_text = $2 AND did_branch ORDER BY created_at DESC LIMIT 1",
                interaction["session_id"], interaction["originating_question"],
            )
            if offer is not None:
                options = await conn.fetch(
                    "SELECT option_text, was_selected, kind, axis FROM interaction_options "
                    "WHERE interaction_id = $1 ORDER BY shown_position",
                    offer["id"],
                )
    question_text = (
        interaction["originating_question"]
        if interaction["question_author"] == "system_option"
        else interaction["question_text"]
    )
    return {
        "learner_id": interaction["learner_id"],
        "session_id": interaction["session_id"],
        "question_text": question_text,
        "response_text": interaction["response_text"],
        "options": [dict(o) for o in options],
        "outcome": outcome_row["outcome"] if outcome_row else None,
        "next_question_text": outcome_row["next_question_text"] if outcome_row else None,
    }


# ─────────────────────────── store + reconciliation ────────────────────


class ClaimStore:
    """Append-only in this codebase's established sense (CLAUDE.md
    invariants 1, 4, 6-11): no delete method, no DELETE SQL.
    `statement`/`test`/`value` are set once at `create` and never
    touched again; `confidence`/`status`/`updated_at` change only via
    the UPDATE this class's own `refresh` method issues, the same
    status-transition-via-UPDATE convention `ThinkingStyleStore`/
    `DisambiguationStore` already use."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, claim: Claim) -> Claim:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO claims (
                    id, learner_id, statement, test, value, confidence, source,
                    write_policy, context_scope, status, statement_embedding,
                    created_at, updated_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                claim.id, claim.learner_id, claim.statement, claim.test,
                claim.value.value, claim.confidence, claim.source.value,
                claim.write_policy.value, claim.context_scope,
                claim.status.value, claim.statement_embedding,
                claim.created_at, claim.updated_at,
            )
        return claim

    async def get(self, claim_id: UUID) -> Claim | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM claims WHERE id = $1", claim_id)
        return None if row is None else self._row_to_claim(row)

    async def list_for_learner(self, learner_id: UUID) -> list[Claim]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM claims WHERE learner_id = $1 ORDER BY created_at", learner_id
            )
        return [self._row_to_claim(r) for r in rows]

    async def list_promoted_for_learner(self, learner_id: UUID) -> list[Claim]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM claims WHERE learner_id = $1 AND status = 'promoted' "
                "ORDER BY created_at",
                learner_id,
            )
        return [self._row_to_claim(r) for r in rows]

    async def list_all(self) -> list[Claim]:
        """Every claim, across every learner -- for cross-learner batch
        analysis (score_predictions.py's calibration check) where the
        question is "does confidence track correctness in general,"
        not "what does this one learner's model look like." Same
        precedent as population_patterns.py reading across learners for
        its own batch job."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM claims ORDER BY created_at")
        return [self._row_to_claim(r) for r in rows]

    async def search_similar(
        self, learner_id: UUID, embedding: list[float], limit: int = 5
    ) -> list[tuple[Claim, float]]:
        """Nearest live claims for this learner by cosine similarity on
        `statement_embedding` -- excludes all three terminal statuses:
        `contradicted` (a settled, closed fact about this trait having
        been wrong), `retracted` (never legitimately established in the
        first place), and `superseded` (a duplicate of another live
        claim, merged into it) -- none must compete for evidence that
        belongs to a real (or the surviving) claim. None is ever
        deleted."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *, 1 - (statement_embedding <=> $2) AS similarity
                FROM claims
                WHERE learner_id = $1 AND status NOT IN ('contradicted', 'retracted', 'superseded')
                ORDER BY statement_embedding <=> $2
                LIMIT $3
                """,
                learner_id, embedding, limit,
            )
        results = []
        for row in rows:
            mapped = dict(row)
            similarity = mapped.pop("similarity")
            results.append((self._row_to_claim(mapped), similarity))
        return results

    async def find_by_axis(self, learner_id: UUID, axis: ApproachAxis) -> list[Claim]:
        """Live claims for this learner carrying at least one evidence
        row on `axis` -- the exact-match half of reconciliation (see
        `reconcile_candidate`'s docstring): axis identity, not
        embedding distance, decides whether a new candidate on this
        axis continues an existing claim. Same terminal-status
        exclusion as `search_similar` (contradicted, retracted, AND
        superseded) -- a closed, never-legitimate, or merged-away claim
        is never implicitly reopened by a same-axis pick; a fresh one
        starts instead (or, for superseded specifically, the survivor
        it was merged into is what `reconcile_candidate` should find
        here on its own merits)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT c.*
                FROM claims c
                JOIN claim_evidence ce ON ce.claim_id = c.id
                WHERE c.learner_id = $1 AND ce.axis = $2
                    AND c.status NOT IN ('contradicted', 'retracted', 'superseded')
                ORDER BY c.created_at
                """,
                learner_id, axis.value,
            )
        return [self._row_to_claim(r) for r in rows]

    async def append_evidence(self, evidence: ClaimEvidence) -> ClaimEvidence:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO claim_evidence (
                    id, claim_id, learner_id, interaction_id, direction, topic, axis,
                    session_id, test_fired, contradiction_was_possible, created_at,
                    provenance_note, source
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                """,
                evidence.id, evidence.claim_id, evidence.learner_id, evidence.interaction_id,
                evidence.direction.value, evidence.topic,
                evidence.axis.value if evidence.axis else None, evidence.session_id,
                evidence.test_fired, evidence.contradiction_was_possible, evidence.created_at,
                evidence.provenance_note, evidence.source.value,
            )
        return evidence

    async def mark_evidence_contaminated(self, evidence_id: UUID, note: str) -> ClaimEvidence:
        """The one sanctioned exception to `ClaimEvidence` having no
        mutable field: a human, after diagnosing a harness bug, annotates
        which row came from it. This is deliberately NOT automatic --
        "if you could detect contamination automatically you'd have
        prevented it." Does not touch `direction`/`topic`/`axis`/
        `test_fired`/`contradiction_was_possible` -- the row still means
        exactly what it always meant and still counts in production
        confidence; this only adds a fact that wasn't knowable at write
        time (which harness bug produced it), the same way `Claim.
        superseded_by` adds a fact a status transition alone can't
        carry."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE claim_evidence SET provenance_note = $2 WHERE id = $1 RETURNING *",
                evidence_id, note,
            )
        if row is None:
            raise KeyError(f"claim_evidence {evidence_id} not found")
        return self._row_to_evidence(row)

    async def list_evidence(self, claim_id: UUID) -> list[ClaimEvidence]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM claim_evidence WHERE claim_id = $1 ORDER BY seq", claim_id
            )
        return [self._row_to_evidence(r) for r in rows]

    async def refresh(
        self, claim_id: UUID, config: ClaimConfidenceConfig | None = None
    ) -> Claim:
        """Recomputes confidence from this claim's full evidence
        history and decides its status -- UPDATEs `confidence`/`status`/
        `updated_at` in one statement. Never reopens an already-terminal
        claim (`contradicted`/`retracted` stay fully visible with their
        evidence intact, they just never render again); never demotes
        `promoted` back to `candidate`.

        `contradicted` is now DERIVED here (`evaluate_contradiction`),
        checked BEFORE promotion -- a claim earns it only after
        confidence has genuinely fallen AND that fall is backed by
        contrary evidence from multiple distinct sessions, not on the
        first opposite-value pick. See `evaluate_contradiction`'s own
        docstring for the incident this replaces (a claim used to close
        terminally on one contradicting episode, orphaning every later
        same-axis episode into a fresh singleton claim for the rest of
        the learner's history). This check runs regardless of current
        status (even `promoted`) -- sustained reversal should be able to
        end a claim the same way a single one never should."""
        claim = await self.get(claim_id)
        if claim is None:
            raise KeyError(f"claim {claim_id} not found")
        if claim.status in (ClaimStatus.CONTRADICTED, ClaimStatus.RETRACTED, ClaimStatus.SUPERSEDED):
            return claim  # terminal; evidence can still be appended, status will not move again

        evidence_rows = await self.list_evidence(claim_id)
        should_promote, raw_confidence = evaluate_promotion(evidence_rows, claim.write_policy, config)
        should_contradict = evaluate_contradiction(evidence_rows, claim.write_policy, config)
        if should_contradict:
            new_status = ClaimStatus.CONTRADICTED
        elif claim.status is ClaimStatus.CANDIDATE and should_promote:
            new_status = ClaimStatus.PROMOTED
        else:
            new_status = claim.status

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE claims SET confidence = $2, status = $3, updated_at = $4
                WHERE id = $1
                RETURNING *
                """,
                claim_id, raw_confidence, new_status.value, datetime.now(UTC),
            )
        return self._row_to_claim(row)

    async def contradict(self, claim_id: UUID) -> Claim:
        """The one other status transition: reconciliation calls this
        when a new episode directly contradicts an existing claim AND
        that episode was eligible (`contradiction_was_possible` --
        see `reconcile_candidate`'s own docstring for why an
        ineligible pick must not be allowed to call this). Terminal --
        a contradicted claim is never promoted again, but stays fully
        on record (CLAUDE.md invariants 1/4/6-11's own resurrection-
        over-deletion principle)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE claims SET status = 'contradicted', updated_at = $2 "
                "WHERE id = $1 RETURNING *",
                claim_id, datetime.now(UTC),
            )
        if row is None:
            raise KeyError(f"claim {claim_id} not found")
        return self._row_to_claim(row)

    async def retract(self, claim_id: UUID) -> Claim:
        """Terminal, like `contradict`, but for a different reason: this
        claim's FOUNDING evidence row was itself ineligible -- it was
        never legitimately established, as opposed to having been
        established and later disproven. `reconcile_candidate` calls
        this immediately after creating a new claim from a subject-kind
        (uncontestable) episode, so it never sits live long enough to
        compete in `search_similar`/`find_by_axis` for evidence that
        belongs to a real claim. Never deletes the row -- same
        resurrection-over-deletion principle as every other terminal
        status here; the record that this was proposed and rejected
        stays fully visible."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE claims SET status = 'retracted', updated_at = $2 "
                "WHERE id = $1 RETURNING *",
                claim_id, datetime.now(UTC),
            )
        if row is None:
            raise KeyError(f"claim {claim_id} not found")
        return self._row_to_claim(row)

    async def supersede(self, claim_id: UUID, survivor_id: UUID) -> Claim:
        """Terminal, like `contradict`/`retract`, but for a third
        reason: this claim turned out to be a duplicate of another LIVE
        claim -- same learner, same axis, same value -- not disproven,
        not illegitimate, just redundant. `merge_duplicate_claims` calls
        this on every loser AFTER copying its evidence onto `survivor_id`
        (see that function's own docstring for why the evidence is
        COPIED, not repointed: `ClaimEvidence` has "no mutable field at
        all, not even a status" -- mutating an existing row's `claim_id`
        would break that guarantee). Never deletes the row -- the
        loser's own evidence stays exactly as originally recorded, fully
        visible; `superseded_by` is the only new fact."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE claims SET status = 'superseded', superseded_by = $2, updated_at = $3 "
                "WHERE id = $1 RETURNING *",
                claim_id, survivor_id, datetime.now(UTC),
            )
        if row is None:
            raise KeyError(f"claim {claim_id} not found")
        return self._row_to_claim(row)

    async def reopen(self, claim_id: UUID) -> Claim:
        """The correction for a claim that was closed before the
        eligibility gate existed on `contradict` -- moves status back
        from `contradicted` to `candidate` via UPDATE (still append-
        only in this codebase's sense: no evidence row is touched,
        this only corrects a status field the same way every other
        transition here does). No-ops (returns the claim unchanged) if
        it isn't currently `contradicted` -- this is a one-time repair
        path for claims wrongly closed by evidence the confidence math
        itself would have refused to count, not a general un-contradict
        button. Callers should follow with `refresh` so confidence
        reflects the claim's real (eligible-only) evidence history
        once it's live again."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE claims SET status = 'candidate', updated_at = $2 "
                "WHERE id = $1 AND status = 'contradicted' RETURNING *",
                claim_id, datetime.now(UTC),
            )
        if row is None:
            existing = await self.get(claim_id)
            if existing is None:
                raise KeyError(f"claim {claim_id} not found")
            return existing  # not contradicted -- nothing to reopen, unchanged
        return self._row_to_claim(row)

    async def append_statement(self, record: ClaimStatementRecord) -> ClaimStatementRecord:
        """Append-only, same pattern as `append_evidence` -- never edits
        or replaces a prior row. `Claim.statement` (set once at
        creation) stays the historical founding text forever; THIS
        table is what `get_current_statement`/`list_statements` read
        for the current best rendering. See `ClaimStatementRecord`'s
        own docstring."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO claim_statements (
                    id, claim_id, statement, derived_from_evidence_count,
                    generator_version, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6)
                """,
                record.id, record.claim_id, record.statement,
                record.derived_from_evidence_count, record.generator_version, record.created_at,
            )
        return record

    async def get_current_statement(self, claim_id: UUID) -> ClaimStatementRecord | None:
        """The latest rendering -- what every reader should show as
        "the" statement. `None` only for a claim created before this
        table existed and never backfilled; callers fall back to
        `Claim.statement` in that case."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM claim_statements WHERE claim_id = $1 ORDER BY seq DESC LIMIT 1",
                claim_id,
            )
        if row is None:
            return None
        mapped = dict(row)
        mapped.pop("seq", None)
        return ClaimStatementRecord(**mapped)

    async def list_statements(self, claim_id: UUID) -> list[ClaimStatementRecord]:
        """Full rendering history, oldest first -- so the wording's
        drift as evidence accumulated is visible on request, the first
        row always being the founding text from claim creation."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM claim_statements WHERE claim_id = $1 ORDER BY seq", claim_id
            )
        results = []
        for r in rows:
            mapped = dict(r)
            mapped.pop("seq", None)
            results.append(ClaimStatementRecord(**mapped))
        return results

    def _row_to_claim(self, row) -> Claim:
        mapped = dict(row)
        embedding = mapped.get("statement_embedding")
        if embedding is not None and not isinstance(embedding, list):
            mapped["statement_embedding"] = embedding.to_list()
        if isinstance(mapped.get("context_scope"), str):
            mapped["context_scope"] = json.loads(mapped["context_scope"])
        assert_row_consumed(Claim, mapped)
        return Claim(**mapped)

    def _row_to_evidence(self, row) -> ClaimEvidence:
        mapped = dict(row)
        mapped.pop("seq", None)
        assert_row_consumed(ClaimEvidence, mapped)
        return ClaimEvidence(**mapped)


def _direction_against(existing_value: StatedPreferenceLabel, candidate_value: StatedPreferenceLabel) -> EvidenceDirection | None:
    """Same value supports; the defined opposite (`_OPPOSITE_LABELS`)
    contradicts; anything else is not a relationship this function can
    confidently classify, so it returns None rather than guess."""
    if existing_value == candidate_value:
        return EvidenceDirection.SUPPORTS
    if _OPPOSITE_LABELS.get(existing_value) == candidate_value:
        return EvidenceDirection.CONTRADICTS
    return None


async def _merge_claims_group(
    store: ClaimStore,
    learner_id: UUID,
    group: list[Claim],
    confidence_config: ClaimConfidenceConfig | None = None,
) -> Claim:
    """Collapses `group` (>= 2 LIVE claims, already confirmed
    value-compatible by the caller -- same axis, same value or the
    defined opposite) into one. The OLDEST survives (longest evidence
    history, earliest founding -- the cleanest rule available, not a
    similarity judgment: see `reconcile_candidate`'s own docstring for
    why embedding distance is exactly the lottery this closes). Every
    other claim's evidence is COPIED onto the survivor -- never
    repointed; `ClaimEvidence` has no mutable field, not even a status
    (see `ClaimStore.supersede`), so an existing row's `claim_id` is
    never rewritten. `created_at`/`session_id`/`interaction_id`/`topic`/
    `axis`/`test_fired`/`contradiction_was_possible` all come across
    unchanged, so `compute_confidence`'s age-based decay stays
    historically accurate. `direction` is flipped when (and only when)
    the loser's asserted value is the survivor's defined OPPOSITE --
    otherwise a row that genuinely contradicted the loser would be
    miscounted as supporting the survivor. Each loser is then marked
    `superseded` (pointing at the survivor); the survivor is `refresh`ed
    once, after all groups' evidence has landed."""
    survivor, *losers = sorted(group, key=lambda c: c.created_at)
    for loser in losers:
        flip = loser.value != survivor.value  # already confirmed compatible: equal, or the defined opposite
        for ev in await store.list_evidence(loser.id):
            direction = ev.direction
            if flip:
                direction = (
                    EvidenceDirection.CONTRADICTS if direction is EvidenceDirection.SUPPORTS
                    else EvidenceDirection.SUPPORTS
                )
            await store.append_evidence(
                ClaimEvidence(
                    claim_id=survivor.id, learner_id=learner_id, interaction_id=ev.interaction_id,
                    direction=direction, topic=ev.topic, axis=ev.axis, session_id=ev.session_id,
                    test_fired=ev.test_fired, contradiction_was_possible=ev.contradiction_was_possible,
                    created_at=ev.created_at, provenance_note=ev.provenance_note, source=ev.source,
                )
            )
        await store.supersede(loser.id, survivor.id)
    return await store.refresh(survivor.id, confidence_config)


async def merge_duplicate_claims(
    pool: asyncpg.Pool,
    store: ClaimStore,
    learner_id: UUID,
    confidence_config: ClaimConfidenceConfig | None = None,
) -> list[Claim]:
    """Repair pass: finds every group of LIVE claims for `learner_id`
    that shares an axis AND is value-compatible (see `_direction_against`)
    -- the exact-match half of reconciliation, applied BETWEEN existing
    claims rather than between a new candidate and the existing set --
    and merges each group via `_merge_claims_group`. Closes the gap
    `reconcile_candidate` alone can't: it only ever asks "does this NEW
    candidate match something existing," never "are two things that
    already exist actually the same thing" -- the gap `ClaimStore.reopen`
    (or any future repair) can reintroduce by putting a previously-
    terminal claim back among the live, axis-sharing set. Safe to run
    repeatedly: a learner with no duplicates returns an empty list, and
    a claim already `superseded` is excluded from every later pass (its
    own status, not membership in a group, decides that).

    Returns the refreshed survivor of every group that was actually
    merged (empty if there was nothing to merge)."""
    live_claims = [
        c for c in await store.list_for_learner(learner_id)
        if c.status in (ClaimStatus.CANDIDATE, ClaimStatus.PROMOTED)
    ]
    if len(live_claims) < 2:
        return []
    live_by_id = {c.id: c for c in live_claims}

    async with pool.acquire() as conn:
        axis_rows = await conn.fetch(
            "SELECT DISTINCT claim_id, axis FROM claim_evidence "
            "WHERE learner_id = $1 AND axis IS NOT NULL",
            learner_id,
        )
    axes_by_claim: dict[UUID, set[str]] = {}
    for r in axis_rows:
        if r["claim_id"] in live_by_id:  # ignore axis rows belonging to already-terminal claims
            axes_by_claim.setdefault(r["claim_id"], set()).add(r["axis"])

    already_merged: set[UUID] = set()
    survivors: list[Claim] = []
    all_axes = {a for axes in axes_by_claim.values() for a in axes}
    for axis in sorted(all_axes):
        sharing_ids = [
            cid for cid, axes in axes_by_claim.items()
            if axis in axes and cid not in already_merged
        ]
        if len(sharing_ids) < 2:
            continue
        anchor = live_by_id[sharing_ids[0]]
        group = [
            live_by_id[cid] for cid in sharing_ids
            if cid == anchor.id or _direction_against(anchor.value, live_by_id[cid].value) is not None
        ]
        if len(group) < 2:
            continue
        survivor = await _merge_claims_group(store, learner_id, group, confidence_config)
        already_merged.update(c.id for c in group)
        survivors.append(survivor)
    return survivors


# ─────────────────────────── statement restatement (session-end) ──────


def _dominant_axis(evidence_rows: list[ClaimEvidence]) -> ApproachAxis | None:
    """The most-represented non-null axis across a claim's evidence --
    used only to give the restatement prompt a label for what's being
    described. A claim's evidence is usually all one axis; the rare
    mixed case (e.g. a pre-fix legacy claim) just gets its plurality,
    not a crash or a guess at "the" axis when several are equally
    present."""
    axes = [e.axis for e in evidence_rows if e.axis is not None]
    if not axes:
        return None
    return Counter(axes).most_common(1)[0][0]


def _restatement_prompt(
    axis: ApproachAxis | None,
    value: StatedPreferenceLabel,
    evidence_rows: list[ClaimEvidence],
    current_statement: str,
) -> str:
    evidence_lines = "\n".join(
        f"- {e.direction.value} | topic: {e.topic}" for e in evidence_rows
    )
    axis_line = f"Axis: {axis.value}\n" if axis is not None else ""
    return (
        "RESTATE:CLAIM\n"
        "This is a standing claim about a learner's teaching-style "
        "preference.\n"
        f"{axis_line}"
        f"Asserted value: {value.value}\n\n"
        f"Evidence accumulated so far, across every topic this claim "
        f"has been tested against:\n{evidence_lines}\n\n"
        f'Current statement (may be stale -- it was written from '
        f'whichever single episode first founded this claim, before '
        f'the evidence below had accumulated): "{current_statement}"\n\n'
        "Write ONE sentence, true of ALL of the evidence above, not "
        "anchored to any single topic or domain -- describe the "
        "standing preference in general terms. Do not name a specific "
        "subject area (a domain word like biology, algorithms, a "
        "particular topic) unless every row above is from that exact "
        "domain.\n\n"
        'Respond with JSON: {"statement": "..."}'
    )


def _parse_restatement_response(raw: str) -> str | None:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    statement = data.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return None
    return statement.strip()


async def restate_claim(
    llm: LLMClient,
    claim: Claim,
    evidence_rows: list[ClaimEvidence],
    axis: ApproachAxis | None,
) -> str | None:
    """One fast-tier call: asks for a statement true of the claim's
    WHOLE evidence history (topics + directions only -- `ClaimEvidence`
    never copies original episode text, see its own docstring), not
    anchored to whichever single episode founded the claim. Returns
    `None` on any unparseable response -- the caller then simply
    doesn't append a new `claim_statements` row, leaving the existing
    (possibly stale) wording in place rather than risk appending a
    garbled one."""
    raw = await llm.complete(_restatement_prompt(axis, claim.value, evidence_rows, claim.statement))
    return _parse_restatement_response(raw)


async def maybe_restate_claims(
    store: ClaimStore,
    llm: LLMClient,
    learner_id: UUID,
    config: ClaimRestatementConfig | None = None,
    on_node_call: Callable[[str, dict, object], Awaitable[None]] | None = None,
) -> list[Claim]:
    """Session-end check (see `SessionLoop.consolidate_session`, called
    AFTER `merge_duplicate_claims` so a just-consolidated survivor's
    full, merged evidence history is what gets evaluated).

    `Claim.statement` is set once at creation and never edited (see
    that model's own docstring) -- exactly the founding episode's
    framing, which can read as misleadingly narrow once evidence
    accumulates across other domains (the incident that motivated this:
    a claim founded on a biology episode, its statement still saying
    "...biological..." after evidence from computer science and
    algorithms piled up). `statement` is a RENDERING of (axis, value)
    against accumulated evidence, not the claim's identity -- axis and
    value stay immutable, evidence attaches by them, only the prose
    describing them is regenerable. So this never touches `Claim.
    statement`; it appends a row to `claim_statements` (append-only,
    same pattern as `turn_outcomes`) and readers resolve to the latest.

    Restates a claim only when BOTH:
      - evidence has grown by at least `config.min_new_evidence_rows`
        since the claim's last statement was written (a claim whose
        evidence hasn't moved doesn't need new wording -- regenerating
        it every session would just invite drift for no gain), and
      - that evidence now spans a topic outside the founding episode's
        own domain (otherwise the existing wording is still accurate;
        there is nothing to generalize away from yet).

    A human-flagged (`provenance_note`) row is excluded from what the
    LLM actually sees, and from the domain-spread check -- a live check
    on real data found the cost of not doing this: given all 5 rows
    including one known-contaminated contradiction, the model didn't
    ignore the outlier, it explained it, inventing a specific,
    plausible-sounding "but favors formal explanations when tracing
    exact procedural steps" nuance that was true of a harness bug, not
    the learner. Production confidence/status still count a flagged
    row in full (see `ClaimEvidence.provenance_note`'s own docstring
    for why) -- but a narrative asked to characterize "what this person
    is like" is a different kind of consumer than a bounded statistic,
    and it will confabulate a specific reason for an outlier rather
    than discount it the way the Beta posterior implicitly does.
    `derived_from_evidence_count` still records the TOTAL (including
    flagged) count, so the growth threshold above isn't gameable by
    contamination and stays comparable to a claim's real evidence
    volume on the review surface.

    Returns every claim actually restated this pass (empty if none
    qualified)."""
    cfg = config or ClaimRestatementConfig()
    restated: list[Claim] = []
    live_claims = [
        c for c in await store.list_for_learner(learner_id)
        if c.status in (ClaimStatus.CANDIDATE, ClaimStatus.PROMOTED)
    ]
    for claim in live_claims:
        current = await store.get_current_statement(claim.id)
        if current is None:
            continue  # defensive -- every claim gets a founding row at creation
        evidence_rows = await store.list_evidence(claim.id)
        if len(evidence_rows) - current.derived_from_evidence_count < cfg.min_new_evidence_rows:
            continue
        clean_rows = [e for e in evidence_rows if e.provenance_note is None]
        if not clean_rows:
            continue  # nothing genuine left to describe
        founding_topic = min(clean_rows, key=lambda e: e.created_at).topic
        if all(e.topic == founding_topic for e in clean_rows):
            continue  # nothing outside the founding domain to generalize away from yet

        axis = _dominant_axis(clean_rows)
        new_statement = await restate_claim(llm, claim, clean_rows, axis)
        if on_node_call is not None:
            await on_node_call(
                "ClaimRestater",
                {
                    "claim_id": str(claim.id), "axis": axis.value if axis else None,
                    "value": claim.value.value, "evidence_count": len(evidence_rows),
                    "clean_evidence_count": len(clean_rows),
                    "current_statement": current.statement,
                },
                {"new_statement": new_statement},
            )
        if new_statement is None:
            continue
        await store.append_statement(
            ClaimStatementRecord(
                claim_id=claim.id, statement=new_statement,
                derived_from_evidence_count=len(evidence_rows),
                generator_version=cfg.generator_version,
            )
        )
        restated.append(claim)
    return restated


async def reconcile_candidate(
    store: ClaimStore,
    embedding_client: EmbeddingClient,
    learner_id: UUID,
    session_id: UUID,
    interaction_id: UUID,
    candidate: ClaimCandidate,
    contradiction_was_possible: bool,
    axis: ApproachAxis | None = None,
    episode_kind: str | None = None,
    source: ClaimSource = ClaimSource.INFERRED,
    write_policy: ClaimWritePolicy = ClaimWritePolicy.SLOW_DRIFT,
    extraction_config: ExtractionConfig | None = None,
    confidence_config: ClaimConfidenceConfig | None = None,
) -> Claim:
    """Matches `candidate` against this learner's existing claims AXIS
    FIRST, statement second -- see this module's own docstring for why:
    axis identity is a closed, structural label persisted at option-
    generation time, not text two extractions might phrase differently,
    so two claims sharing an axis for one learner are treated as the
    SAME claim outright rather than judged by embedding distance.
    Semantic similarity is used only (a) to pick among several existing
    claims that already share the axis (a tie, or a pre-fix fragment),
    and (b) as the sole strategy when the episode carries no axis at
    all (a stated-preference or contradicted_intent trigger).

    Either path decides evidence direction the same way: same `value`
    as the matched claim supports, the defined opposite
    (`_OPPOSITE_LABELS`) contradicts, anything else is NOT treated as a
    match (this function can't confidently classify a direction for
    it) and reconciliation falls through to the next candidate match or
    to creating a new claim. No match creates a brand new `claims` row,
    seeded with its own first supporting evidence.

    A CONTRADICTS direction never closes the claim directly here --
    every path below writes the evidence row, then unconditionally
    calls `store.refresh`, which recomputes confidence from the whole
    (eligible-filtered) history and decides status itself
    (`evaluate_contradiction`; see that function's docstring). An
    ineligible (uncontestable, subject-kind) contradicting row is still
    written either way (provenance, visible in the ledger) but is
    invisible to both the confidence math and the contradiction check,
    same as any other ineligible row -- it can neither raise confidence
    nor contribute a session toward closing the claim. A live run found
    the cost of the OLD one-shot design twice over: once, a claim closed
    on 0 of 2 eligible rows (killed by evidence the confidence math
    itself would have refused to count); again, a claim closed on its
    very FIRST eligible contradicting row, in the same session that had
    just supported it, permanently orphaning every later same-axis
    episode into a fresh singleton claim (`find_by_axis` won't match a
    contradicted claim). Both are the same underlying mistake: treating
    one episode as sufficient to end a claim about a person, when only
    sustained, cross-session reversal should.

    `axis` is also carried onto the evidence row so `compute_confidence`'s
    cell-collapsing can group by (session, axis) rather than
    (session, topic).

    `episode_kind` ("approach", "subject", or None for no live option
    set) gates a SEPARATE terminal transition: a brand-new claim
    founded entirely on a `subject`-kind episode is retracted
    immediately (see `ClaimStore.retract`) rather than left live as a
    `candidate` -- a topic pick was never legitimate teaching-
    preference evidence, so a claim built from one was never
    legitimately established. `None` (stated-preference/contradicted_
    intent triggers) is deliberately exempt even though it also
    implies `contradiction_was_possible=False` -- a self-report is
    real evidence, not an uncontestable pick."""
    cfg = extraction_config or ExtractionConfig()
    embedding = await embedding_client.embed(candidate.statement, task_type=TASK_SIMILARITY)

    matched_claim: Claim | None = None
    direction: EvidenceDirection | None = None

    if axis is not None:
        axis_claims = await store.find_by_axis(learner_id, axis)
        # If more than one already-live claim on this axis is value-
        # compatible with THIS candidate, they're duplicates of each
        # other, not competitors for a similarity tiebreak -- merge them
        # into one before matching, so collisions can't reaccumulate the
        # way they did when `reopen` put a previously-terminal claim
        # back among a live, axis-sharing set (see `merge_duplicate_claims`'s
        # own docstring for the incident this closes).
        compatible_with_candidate = [
            c for c in axis_claims if _direction_against(c.value, candidate.value) is not None
        ]
        if len(compatible_with_candidate) > 1:
            survivor = await _merge_claims_group(
                store, learner_id, compatible_with_candidate, confidence_config
            )
            merged_ids = {c.id for c in compatible_with_candidate}
            axis_claims = [survivor] + [c for c in axis_claims if c.id not in merged_ids]

        if len(axis_claims) == 1:
            candidates_in_axis_order = axis_claims
        else:
            candidates_in_axis_order = sorted(
                axis_claims,
                key=lambda c: cosine_similarity(embedding, c.statement_embedding),
                reverse=True,
            )
        for existing in candidates_in_axis_order:
            found = _direction_against(existing.value, candidate.value)
            if found is not None:
                matched_claim, direction = existing, found
                break
            # Same axis but an unrelated value relationship -- axis
            # identity alone doesn't make this a match; keep looking
            # at the next axis-sharing claim (relevant only when more
            # than one already shares the axis).

    if matched_claim is None:
        # No axis, or no axis-sharing claim related to this candidate
        # -- fall back to semantic similarity across the learner's
        # whole claim set, same strategy used before axis existed.
        matches = await store.search_similar(learner_id, embedding, limit=3)
        for existing, similarity in matches:
            if similarity < cfg.similarity_match_threshold:
                continue
            found = _direction_against(existing.value, candidate.value)
            if found is not None:
                matched_claim, direction = existing, found
                break
            # Similar prose, unrelated/ambiguous label relationship --
            # not treated as a match; keep looking at the next nearest
            # claim.

    if matched_claim is None:
        claim = Claim(
            learner_id=learner_id,
            statement=candidate.statement,
            test=candidate.test,
            value=candidate.value,
            confidence=0.5,  # extraction-time placeholder; refresh() below sets the real value
            source=source,
            write_policy=write_policy,
            status=ClaimStatus.CANDIDATE,
            statement_embedding=embedding,
        )
        await store.create(claim)
        # The founding row of this claim's statement-rendering history
        # (see ClaimStatementRecord's own docstring) -- same text as
        # claim.statement, written once here so maybe_restate_claims
        # always has a baseline evidence count to diff future growth
        # against.
        await store.append_statement(
            ClaimStatementRecord(
                claim_id=claim.id, statement=claim.statement,
                derived_from_evidence_count=1, generator_version=CLAIM_EXTRACTOR_VERSION,
            )
        )
        await store.append_evidence(
            ClaimEvidence(
                claim_id=claim.id, learner_id=learner_id, interaction_id=interaction_id,
                direction=EvidenceDirection.SUPPORTS, topic=candidate.topic, axis=axis,
                session_id=session_id,
                test_fired=True, contradiction_was_possible=contradiction_was_possible,
            )
        )
        refreshed = await store.refresh(claim.id, confidence_config)
        # A brand-new claim founded ENTIRELY on a subject-kind
        # (uncontestable) episode was never legitimately established --
        # a topic pick carries no teaching-preference information (see
        # _extraction_prompt's own subject-kind caveat). Retract it
        # immediately so it never sits live long enough to compete in
        # search_similar/find_by_axis for evidence that belongs to a
        # real claim -- exactly the failure a live run found: such a
        # claim absorbed a genuine eligible contradiction and closed
        # terminally, spending real evidence to kill a premise that
        # should never have existed. Deliberately NOT applied when
        # episode_kind is None (a stated-preference or contradicted_
        # intent trigger with no live option set) -- that evidence is a
        # genuine self-report, not an uncontestable topic pick, even
        # though it also carries contradiction_was_possible=False.
        if episode_kind == AmbiguityKind.SUBJECT.value:
            return await store.retract(refreshed.id)
        return refreshed

    await store.append_evidence(
        ClaimEvidence(
            claim_id=matched_claim.id, learner_id=learner_id, interaction_id=interaction_id,
            direction=direction, topic=candidate.topic, axis=axis, session_id=session_id,
            test_fired=True, contradiction_was_possible=contradiction_was_possible,
        )
    )
    # No branching on `direction` here -- refresh() recomputes
    # confidence from the whole eligible-filtered history and decides
    # status itself (evaluate_contradiction requires sustained,
    # cross-session reversal; see this function's own docstring and
    # refresh()'s). A single contradicting row, eligible or not, moves
    # the number and nothing more.
    return await store.refresh(matched_claim.id, confidence_config)


# ─────────────────────────── session-end orchestration ─────────────────


async def extract_claims_for_session(
    pool: asyncpg.Pool,
    store: ClaimStore,
    extractor: ClaimExtractor,
    embedding_client: EmbeddingClient,
    session_id: UUID,
    learner_id: UUID,
    extraction_config: ExtractionConfig | None = None,
    confidence_config: ClaimConfidenceConfig | None = None,
    on_node_call: Callable[[str, dict, object], Awaitable[None]] | None = None,
) -> list[Claim]:
    """The session-end entry point (see loop.SessionLoop.consolidate_session):
    selects candidates, runs `ClaimExtractor` on each, and reconciles
    every resulting candidate. `on_node_call`, when given, is AWAITED
    once per `ClaimExtractor.run()` with (node_name, input_json,
    output_json) so the caller can persist it to `node_calls`
    (CLAUDE.md invariant 2 -- every node invocation must be recorded)
    without this function needing to know about `SessionLoop` or
    `NodeCallStore` at all."""
    cfg = extraction_config or ExtractionConfig()
    candidates = await select_extraction_candidates(pool, session_id, learner_id, cfg)
    touched: list[Claim] = []
    for candidate in candidates:
        episode = await _load_episode(pool, candidate.interaction_id)
        if episode is None:
            continue
        # Computed BEFORE the extractor call (not just for evidence
        # bookkeeping afterward) so the prompt itself can be told what
        # kind of choice this was -- see _extraction_prompt's own
        # docstring for why a subject-kind pick needs that caveat.
        if not episode["options"]:
            episode_kind = None
        elif any(o.get("kind") == AmbiguityKind.APPROACH.value for o in episode["options"]):
            episode_kind = AmbiguityKind.APPROACH.value
        else:
            episode_kind = AmbiguityKind.SUBJECT.value
        result = await extractor.run(
            question_text=episode["question_text"],
            options=episode["options"],
            response_text=episode["response_text"],
            trigger_reason=candidate.reason,
            next_question_text=episode.get("next_question_text"),
            episode_kind=episode_kind,
        )
        if on_node_call is not None:
            await on_node_call(
                "ClaimExtractor",
                {
                    "question_text": episode["question_text"], "options": episode["options"],
                    "response_text": episode["response_text"], "trigger_reason": candidate.reason,
                    "next_question_text": episode.get("next_question_text"),
                    "episode_kind": episode_kind,
                },
                result.model_dump(mode="json"),
            )
        has_real_choice = episode_kind == AmbiguityKind.APPROACH.value
        # The live axis this episode actually tested, straight off the
        # option-set row -- not re-derived from text (models.py /
        # disambiguate.py's own design principle: decisions are recorded
        # as data, never reverse-engineered later). None when the episode
        # had no approach-kind option set (a subject pick, a stated
        # preference, an unresolved contradiction).
        episode_axis_raw = next(
            (o.get("axis") for o in episode["options"]
             if o.get("kind") == AmbiguityKind.APPROACH.value and o.get("axis")),
            None,
        )
        episode_axis = ApproachAxis(episode_axis_raw) if episode_axis_raw else None
        for proposed in result.candidates:
            claim = await reconcile_candidate(
                store, embedding_client, learner_id, session_id, candidate.interaction_id,
                proposed, contradiction_was_possible=has_real_choice, axis=episode_axis,
                episode_kind=episode_kind,
                extraction_config=cfg, confidence_config=confidence_config,
            )
            touched.append(claim)
    return touched
