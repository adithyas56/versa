"""The capability/mastery claim type — a SEPARATE home for evidence
about what a learner CAN DO, kept apart from `claims.py`'s preference
claims by construction, not by a filter.

WHY THIS EXISTS: `method_capabilities.py` originally declared `locate`/
`predict` (both PERFORMANCE-measuring) with `claim_types=
(StatedPreferenceLabel...,)`, because `StatedPreferenceLabel` was the
only claim vocabulary this codebase had. Checking what real predict
evidence had actually landed on confirmed the category error that
implied: every claim it wrote to was a genuine `wants_steps_shown`/
`rule_before_example` PREFERENCE claim, its confidence built entirely
from capability observations ("correctly traced the swap forward").
That's mechanical, not subtle — "can trace a swap correctly" is not a
preference, and the two can diverge in either direction (someone can
prefer worked examples and still be perfectly able to derive forward
without them).

WHAT'S DIFFERENT FROM `claims.py`, DELIBERATELY:

- `CapabilityLabel` (models.py), never `StatedPreferenceLabel`. A
  disjoint vocabulary, not a subset — nothing here is "borrowed" from
  the preference side.
- No axis. `ApproachAxis` exists because a preference axis is
  bidirectional (concrete <-> general, analogy <-> formal) and evidence
  needs to say WHICH pole. A skill has no opposite pole — you either
  demonstrate it or you don't — so there is nothing for an axis
  concept to do here, and `CapabilityClaimStore.find_by_skill` matches
  by exact skill, never axis-sharing-then-similarity the way
  `ClaimStore.find_by_axis` does.
- No `statement_embedding` on `CapabilityClaim` — with no similarity-
  based fallback matching path (skill is always known, never None the
  way axis can be), there is nothing for an embedding to disambiguate.
- No merge pass, no derived-contradiction floor, no restatement, no
  promotion gate yet. `claims.py` earned all of that machinery through
  real incidents on the preference side over a long session; building
  it here speculatively, before a single real incident demands it,
  would be exactly the premature generality this codebase avoids
  elsewhere. `refresh` recomputes confidence only. Status stays
  `candidate` until a later, evidence-driven build adds a promotion
  rule — `CapabilityStatus.PROMOTED` exists in the schema so that
  addition never needs a migration, but nothing sets it yet.

WHAT'S DELIBERATELY THE SAME, AND WHY DUPLICATED RATHER THAN SHARED:
the Beta-posterior CELL-BUILDING (evidence rows -> a decay-weighted
alpha/beta) started out identical to `compute_confidence` in
claims.py, and still is — it only needs evidence rows with a
direction, an age, and a decay-weighted cell key, and neither module
cares what a claim's value MEANS. That math is reusable in spirit but
this module reimplements it against `CapabilityEvidence`'s own shape
(`skill` where `ClaimEvidence` has `axis`+`topic`) rather than
generalizing `claims.py`'s version to accept either shape. Two
reasons: this session was explicitly told to leave `claims.py`'s
confidence math unchanged, and duplicating the cell-building is
cheaper and more legible than a shared abstraction serving two claim
kinds that are supposed to stay structurally separate on every other
axis. What's now DIFFERENT, deliberately: the final alpha/beta -> scalar
step. `claims.compute_confidence` still reports beat-chance (`1 -
BetaCDF`); `compute_capability_confidence` below reports the posterior
mean instead (see that function's own docstring for the recompute
that motivated the change and why capability was the side it was safe
to change first — nothing reads a capability claim yet, so there was
no live behavior to disturb). The two are expected to diverge now;
that is not drift to fix, it is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import asyncpg
from pydantic import BaseModel

from probe.models import (
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityLabel,
    CapabilityStatus,
    ClaimWritePolicy,
    EvidenceDirection,
    EvidenceSource,
)


class CapabilityConfidenceConfig(BaseModel):
    """Deliberately smaller than `ClaimConfidenceConfig`: no
    decision-clamp, no promotion threshold, no derived-contradiction
    floor — none of that is implemented yet on the capability side
    (see this module's own docstring for why). Just the Beta prior and
    the three write-policy decay rates, which `compute_capability_confidence`
    needs regardless of what gets built on top of it later."""

    alpha0: float = 1.0
    beta0: float = 1.0
    locked_decay: float = 1.0
    slow_drift_decay: float = 0.97
    fast_decay_decay: float = 0.6


def _decay_rate(write_policy: ClaimWritePolicy, config: CapabilityConfidenceConfig) -> float:
    return {
        ClaimWritePolicy.LOCKED: config.locked_decay,
        ClaimWritePolicy.SLOW_DRIFT: config.slow_drift_decay,
        ClaimWritePolicy.FAST_DECAY: config.fast_decay_decay,
    }[write_policy]


@dataclass
class _EvidenceCell:
    direction: EvidenceDirection
    age_days: float


def _collapse_to_cells(evidence_rows: list[CapabilityEvidence], now: datetime) -> list[_EvidenceCell]:
    """Same collapsing rule as `claims._collapse_to_cells`: filters to
    eligible rows (`test_fired AND contradiction_was_possible`), groups
    by (session, skill), collapses each group to its most recent row.
    Simpler than the preference-side version — `skill` is never None,
    so there's no axis-or-topic fallback branch to carry."""
    eligible = [r for r in evidence_rows if r.test_fired and r.contradiction_was_possible]
    latest_at: dict[tuple, datetime] = {}
    direction_of: dict[tuple, EvidenceDirection] = {}
    for row in eligible:
        key = (row.session_id, row.skill)
        if key not in latest_at or row.created_at > latest_at[key]:
            latest_at[key] = row.created_at
            direction_of[key] = row.direction
    cells = []
    for key, created_at in latest_at.items():
        age_days = max(0.0, (now - created_at).total_seconds() / 86400)
        cells.append(_EvidenceCell(direction=direction_of[key], age_days=age_days))
    return cells


def compute_capability_confidence(
    evidence_rows: list[CapabilityEvidence],
    write_policy: ClaimWritePolicy,
    config: CapabilityConfidenceConfig | None = None,
    now: datetime | None = None,
) -> float:
    """Posterior mean alpha0+s / (alpha0+s + beta0+f) — the current
    best estimate of this skill's true supports rate, which is what a
    reported "confidence 0.87" is actually claiming.

    REVISED 2026-09-15, CHANGED FROM BEAT-CHANCE: this function
    originally returned `1 - BetaCDF(1/k; alpha, beta)` — "probability
    this skill beats a coin flip" — matching `claims.compute_confidence`'s
    own formula (see this module's own docstring for why duplicated
    rather than shared; that duplication is exactly what made it safe
    to change one side without the other). Recomputing the existing
    randomized synthetic volume (score_predictions.py's capability
    reader) against both quantities confirmed the two answer different
    questions: LOCATE and PREDICT both showed severe apparent
    overconfidence under beat-chance (a [0.8,0.9)-predicted bucket
    observing a ~44-54% hit rate), because beat-chance saturates toward
    1.0 quickly and was crowding trials of very different true skill
    into one bucket. Switching to the posterior mean redistributed
    those same trials across buckets that tracked their own observed
    rate much more closely, and Brier improved on both (LOCATE 0.2216
    -> 0.2077, PREDICT 0.2262 -> 0.2132). Beat-chance is not wrong, it
    is answering "has this claim earned trust," which is the right
    question for a PROMOTION decision, not for what a calibration
    curve or a decision-facing confidence clamp should report.
    Capability has no promotion gate yet (see this module's own
    docstring) — nothing here needs beat-chance today, so nothing was
    kept speculatively; a future promotion gate should compute it
    itself rather than this function switching back."""
    cfg = config or CapabilityConfidenceConfig()
    now = now or datetime.now(UTC)
    decay = _decay_rate(write_policy, cfg)
    cells = _collapse_to_cells(evidence_rows, now)
    s = sum(decay ** c.age_days for c in cells if c.direction is EvidenceDirection.SUPPORTS)
    f = sum(decay ** c.age_days for c in cells if c.direction is EvidenceDirection.CONTRADICTS)
    alpha = cfg.alpha0 + s
    beta_param = cfg.beta0 + f
    return alpha / (alpha + beta_param)


class CapabilityClaimStore:
    """Append-only, same convention as `ClaimStore`: no delete method,
    no DELETE SQL. `statement`/`test`/`skill` are set once at `create`;
    `confidence`/`status`/`updated_at` change only via `refresh`."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, claim: CapabilityClaim) -> CapabilityClaim:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO capability_claims (
                    id, learner_id, statement, test, skill, confidence,
                    write_policy, status, created_at, updated_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                claim.id, claim.learner_id, claim.statement, claim.test,
                claim.skill.value, claim.confidence, claim.write_policy.value,
                claim.status.value, claim.created_at, claim.updated_at,
            )
        return claim

    async def get(self, claim_id: UUID) -> CapabilityClaim | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM capability_claims WHERE id = $1", claim_id)
        return None if row is None else self._row_to_claim(row)

    async def list_for_learner(self, learner_id: UUID) -> list[CapabilityClaim]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM capability_claims WHERE learner_id = $1 ORDER BY created_at",
                learner_id,
            )
        return [self._row_to_claim(r) for r in rows]

    async def list_all(self) -> list[CapabilityClaim]:
        """Every capability claim, across every learner -- the same
        "batch job, cross-learner" precedent as `ClaimStore.list_all`:
        a calibration question about the confidence FORMULA, not about
        any one learner's model. `score_predictions.py`'s capability-
        side reader uses this."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM capability_claims ORDER BY created_at")
        return [self._row_to_claim(r) for r in rows]

    async def find_by_skill(self, learner_id: UUID, skill: CapabilityLabel) -> CapabilityClaim | None:
        """The capability-side equivalent of `find_by_axis` — but
        exact-match on `skill`, not axis-sharing-then-similarity:
        capability isn't bidirectional (see this module's own
        docstring), so at most one live claim per (learner, skill)
        should ever exist. Returns the single match, or None."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM capability_claims WHERE learner_id = $1 AND skill = $2 "
                "ORDER BY created_at LIMIT 1",
                learner_id, skill.value,
            )
        return None if row is None else self._row_to_claim(row)

    async def append_evidence(self, evidence: CapabilityEvidence) -> CapabilityEvidence:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO capability_evidence (
                    id, claim_id, learner_id, interaction_id, direction, skill,
                    session_id, test_fired, contradiction_was_possible, created_at,
                    provenance_note, source
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                evidence.id, evidence.claim_id, evidence.learner_id, evidence.interaction_id,
                evidence.direction.value, evidence.skill.value, evidence.session_id,
                evidence.test_fired, evidence.contradiction_was_possible, evidence.created_at,
                evidence.provenance_note, evidence.source.value,
            )
        return evidence

    async def list_evidence(self, claim_id: UUID) -> list[CapabilityEvidence]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM capability_evidence WHERE claim_id = $1 ORDER BY created_at",
                claim_id,
            )
        return [self._row_to_evidence(r) for r in rows]

    async def refresh(
        self, claim_id: UUID, config: CapabilityConfidenceConfig | None = None
    ) -> CapabilityClaim:
        """Recomputes confidence from this claim's full evidence
        history. No status transition yet (see this module's own
        docstring) — `status` is left exactly as it was."""
        claim = await self.get(claim_id)
        if claim is None:
            raise KeyError(f"capability claim {claim_id} not found")
        evidence_rows = await self.list_evidence(claim_id)
        raw_confidence = compute_capability_confidence(evidence_rows, claim.write_policy, config)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE capability_claims SET confidence = $2, updated_at = $3 "
                "WHERE id = $1 RETURNING *",
                claim_id, raw_confidence, datetime.now(UTC),
            )
        return self._row_to_claim(row)

    def _row_to_claim(self, row) -> CapabilityClaim:
        mapped = dict(row)
        return CapabilityClaim(
            id=mapped["id"], learner_id=mapped["learner_id"], statement=mapped["statement"],
            test=mapped["test"], skill=CapabilityLabel(mapped["skill"]),
            confidence=mapped["confidence"], write_policy=ClaimWritePolicy(mapped["write_policy"]),
            status=CapabilityStatus(mapped["status"]), created_at=mapped["created_at"],
            updated_at=mapped["updated_at"],
        )

    def _row_to_evidence(self, row) -> CapabilityEvidence:
        mapped = dict(row)
        return CapabilityEvidence(
            id=mapped["id"], claim_id=mapped["claim_id"], learner_id=mapped["learner_id"],
            interaction_id=mapped["interaction_id"], direction=EvidenceDirection(mapped["direction"]),
            skill=CapabilityLabel(mapped["skill"]), session_id=mapped["session_id"],
            test_fired=mapped["test_fired"], contradiction_was_possible=mapped["contradiction_was_possible"],
            created_at=mapped["created_at"], provenance_note=mapped["provenance_note"],
            source=EvidenceSource(mapped["source"]),
        )
