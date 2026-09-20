"""The instrument layer's foundation (migrations 050-052) — a second
way to produce `claim_evidence`, alongside the existing disambiguation-
click path. A click is cheap but shallow: reconciliation can only read
which of the tutor's own pre-written options a learner picked. An
INSTRUMENT is a purpose-built interaction (a worked example with a
seeded error, a slider, an ordering task) whose event stream is
interpreted by a hand-written, deterministic CONTRACT — never an LLM
reading events and guessing a direction, which would just reintroduce
the confabulation risk claims.py's own module docstring exists to keep
out, one layer downstream.

THREE APPEND-ONLY TABLES, each with exactly the same no-DELETE, no-
retroactive-edit discipline every other store in this codebase uses:

- `interaction_contracts` — the recipe. Immutable once written; see
  `InteractionContract`'s own docstring in models.py for the predicate
  shape and the mandatory `uninformative_when`.
- `instruments` — one row per PRESENTATION of a contract to a learner.
  `completed_at`/`abandoned` resolve via UPDATE once known, nothing
  else about a row changes after `create`.
- `instrument_events` — the raw capture. No mutable field at all, not
  even a status: `interpret_instrument` reads this table, nothing
  (least of all an LLM) ever writes an interpretation back onto it.

SIX PRIMITIVES are named (`InstrumentPrimitive` in models.py) and the
event schema (`InstrumentEventType`, also models.py) is proven against
all six from the start — reshaping the schema per instrument later is
how event schemas rot. Only `locate` (error-spotting: a worked example
with one seeded error, the learner clicks where they think it's wrong)
is actually built here; the other five are typed stubs below
(`ChooseEventPayload` etc.) documenting each primitive's distinguishing
payload fields, not yet wired to anything. Every primitive shares three
properties by construction, not by convention any single primitive has
to re-implement: `elapsed_ms` on every event gives timing for free;
`InstrumentEventType.MOVE`/`REVISE` exist on the ONE shared enum so any
primitive with a revision trail can record one without a schema
change; `SUBMIT` vs `ABANDON` is what distinguishes completion from
abandonment for all six, not something each primitive decides on its
own.

WHY GENERATION AND MODE-SELECTION AREN'T HERE YET: this build is
explicitly the plumbing, isolated from generation quality. The one
contract below is hand-authored, not model-generated, specifically so
that if the resulting evidence looks wrong, the fault is locatable to
one layer (the plumbing) rather than smeared across "was the contract
badly written, badly interpreted, or is the plumbing itself broken."
No instrument-selection logic exists in `SessionLoop` either (no mode
selector) — presenting an instrument is a standalone flow
(`present_instrument` below), not a turn `handle_turn` can trigger.

THE GATE: instrument-derived evidence carries `source=instrument` on
its `ClaimEvidence` row (vs. the existing default, `source=click`) so
`score_predictions.py --split-by-source` can check the two
populations' reliability diagrams separately. If instrument evidence
is systematically overconfident relative to click evidence, the
contract is being written or interpreted loosely — that gets found and
fixed before a second instrument exists, not after several are built
on the same unverified assumption.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TypedDict
from uuid import UUID, uuid4

import asyncpg

from probe.audit import TranscriptStore
from probe.capability import CapabilityClaimStore, CapabilityConfidenceConfig
from probe.claims import ClaimConfidenceConfig, ClaimStore
from probe.embeddings import EMBEDDING_DIM
from probe.interactions import InteractionStore
from probe.models import (
    ApproachAxis,
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityLabel,
    Claim,
    ClaimEvidence,
    ClaimSource,
    ClaimWritePolicy,
    EntryState,
    EvidenceDirection,
    EvidenceSource,
    Instrument,
    InstrumentEvent,
    InstrumentEventType,
    InstrumentPrimitive,
    Interaction,
    InteractionContract,
    MeasurementKind,
    QuestionAuthor,
    StatedPreferenceLabel,
)

logger = logging.getLogger(__name__)

INSTRUMENT_CONTRACT_VERSION = "hand-authored-v1"


# ─────────────────────────── typed event-payload shapes ────────────────
#
# One per primitive, documenting what `instrument_events.payload` holds
# for that primitive's `click`/`value_change`/`submit` rows. Only
# `LocateEventPayload` is used at runtime in this build; the other five
# exist so the next instrument's author starts from a named shape
# instead of inventing one ad hoc.


class ChooseEventPayload(TypedDict, total=False):
    """choose: pick one of N discrete options (the generalized form of
    the existing disambiguation click)."""

    option_id: str
    option_index: int


class OrderEventPayload(TypedDict, total=False):
    """order: arrange items into a sequence -- one row per swap/move,
    giving a revision trail for free."""

    item_id: str
    from_index: int
    to_index: int


class LocateEventPayload(TypedDict, total=False):
    """locate: click a point/element on a rendered artifact. The one
    primitive implemented in this build -- see `LOCATE_DEMO_CONTRACT`."""

    clicked_element: str
    x: float
    y: float


class AdjustEventPayload(TypedDict, total=False):
    """adjust: move a continuous or discrete control (slider, stepper)
    -- each `value_change` is one point on the revision trail."""

    control_id: str
    value: float


class PredictEventPayload(TypedDict, total=False):
    """predict: commit to a forecast before an outcome is revealed --
    see `PREDICT_DEMO_SPEC` for the one built contract's actual shape."""

    chosen_option: str


class ConstructEventPayload(TypedDict, total=False):
    """construct: assemble a free-form artifact from parts -- the
    richest primitive; its payload will need a sub-schema of its own
    once built. This stub only reserves the name."""

    artifact: dict


# ─────────────────────────── stores (append-only) ───────────────────────


class InteractionContractStore:
    """No delete/update methods -- a contract is fully immutable once
    written (migration 050's own header)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, contract: InteractionContract) -> InteractionContract:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO interaction_contracts (
                    id, target_claim_id, measures, target_axis, target_value,
                    target_skill, primitive, supports_when, contradicts_when,
                    uninformative_when, generator_version, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                """,
                contract.id, contract.target_claim_id, contract.measures.value,
                contract.target_axis.value if contract.target_axis else None,
                contract.target_value.value if contract.target_value else None,
                contract.target_skill.value if contract.target_skill else None,
                contract.primitive.value,
                contract.supports_when, contract.contradicts_when,
                contract.uninformative_when, contract.generator_version,
                contract.created_at,
            )
        return contract

    async def get(self, contract_id: UUID) -> InteractionContract | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM interaction_contracts WHERE id = $1", contract_id
            )
        return None if row is None else _row_to_contract(row)


class InstrumentStore:
    """`completed_at`/`abandoned` resolve via UPDATE once known --
    same status-transition-via-UPDATE convention as `claims.status`;
    every other field is fixed at `create` (migration 051's own
    header)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def create(self, instrument: Instrument) -> Instrument:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO instruments (
                    id, interaction_id, learner_id, session_id, contract_id,
                    primitive, spec, presented_at, completed_at, abandoned
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
                """,
                instrument.id, instrument.interaction_id, instrument.learner_id,
                instrument.session_id, instrument.contract_id, instrument.primitive.value,
                instrument.spec, instrument.presented_at,
                instrument.completed_at, instrument.abandoned,
            )
        return instrument

    async def get(self, instrument_id: UUID) -> Instrument | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM instruments WHERE id = $1", instrument_id)
        return None if row is None else _row_to_instrument(row)

    async def mark_completed(self, instrument_id: UUID, completed_at: datetime | None = None) -> Instrument:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE instruments SET completed_at = $2 WHERE id = $1 RETURNING *",
                instrument_id, completed_at or datetime.now(UTC),
            )
        if row is None:
            raise KeyError(f"instrument {instrument_id} not found")
        return _row_to_instrument(row)

    async def mark_abandoned(self, instrument_id: UUID) -> Instrument:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE instruments SET abandoned = TRUE WHERE id = $1 RETURNING *",
                instrument_id,
            )
        if row is None:
            raise KeyError(f"instrument {instrument_id} not found")
        return _row_to_instrument(row)


class InstrumentEventStore:
    """Pure append -- no update/delete method exists, and none should
    ever be added (migration 052's own header: "nothing ever writes an
    interpretation back onto a row here")."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def append(self, event: InstrumentEvent) -> InstrumentEvent:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO instrument_events (
                    id, instrument_id, seq, event_type, payload, elapsed_ms, created_at
                ) VALUES ($1,$2,$3,$4,$5,$6,$7)
                """,
                event.id, event.instrument_id, event.seq, event.event_type.value,
                event.payload, event.elapsed_ms, event.created_at,
            )
        return event

    async def list_for_instrument(self, instrument_id: UUID) -> list[InstrumentEvent]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM instrument_events WHERE instrument_id = $1 ORDER BY seq",
                instrument_id,
            )
        return [_row_to_event(r) for r in rows]


def _row_to_contract(row) -> InteractionContract:
    # JSONB columns decode straight to dict/list -- probe.db registers
    # a jsonb codec on every connection, so no manual json.loads here.
    mapped = dict(row)
    return InteractionContract(
        id=mapped["id"], target_claim_id=mapped["target_claim_id"],
        measures=MeasurementKind(mapped["measures"]),
        target_axis=ApproachAxis(mapped["target_axis"]) if mapped["target_axis"] else None,
        target_value=StatedPreferenceLabel(mapped["target_value"]) if mapped["target_value"] else None,
        target_skill=CapabilityLabel(mapped["target_skill"]) if mapped["target_skill"] else None,
        primitive=InstrumentPrimitive(mapped["primitive"]),
        supports_when=mapped["supports_when"], contradicts_when=mapped["contradicts_when"],
        uninformative_when=mapped["uninformative_when"],
        generator_version=mapped["generator_version"], created_at=mapped["created_at"],
    )


def _row_to_instrument(row) -> Instrument:
    mapped = dict(row)
    return Instrument(
        id=mapped["id"], interaction_id=mapped["interaction_id"],
        learner_id=mapped["learner_id"], session_id=mapped["session_id"],
        contract_id=mapped["contract_id"], primitive=InstrumentPrimitive(mapped["primitive"]),
        spec=mapped["spec"], presented_at=mapped["presented_at"],
        completed_at=mapped["completed_at"], abandoned=mapped["abandoned"],
    )


def _row_to_event(row) -> InstrumentEvent:
    mapped = dict(row)
    return InstrumentEvent(
        id=mapped["id"], instrument_id=mapped["instrument_id"], seq=mapped["seq"],
        event_type=InstrumentEventType(mapped["event_type"]), payload=mapped["payload"],
        elapsed_ms=mapped["elapsed_ms"], created_at=mapped["created_at"],
    )


# ─────────────────────────── presentation (creates the FK targets) ─────


async def _create_instrument_row(
    interaction_store: InteractionStore,
    instrument_store: InstrumentStore,
    contract: InteractionContract,
    spec: dict,
    learner_id: UUID,
    session_id: UUID,
    turn_number: int,
    entry_state: EntryState,
) -> Instrument:
    """Shared by both presentation entry points below. Creates the
    minimal real `interactions` row instrument-derived evidence will
    need (`claim_evidence.interaction_id` is foreign-keyed to
    `interactions`, unchanged by this build). `question_embedding` is a
    zero vector: nothing about retrieval or similarity depends on this
    placeholder row being embedded meaningfully, and this build is told
    to leave retrieval unchanged."""
    interaction = Interaction(
        learner_id=learner_id, session_id=session_id, turn_number=turn_number,
        question_text=f"[instrument:{contract.primitive.value}] presented",
        question_author=QuestionAuthor.LEARNER, did_branch=False,
        entry_state=entry_state, question_embedding=[0.0] * EMBEDDING_DIM,
    )
    await interaction_store.create(interaction)
    instrument = Instrument(
        interaction_id=interaction.id, learner_id=learner_id, session_id=session_id,
        contract_id=contract.id, primitive=contract.primitive, spec=spec,
    )
    await instrument_store.create(instrument)
    return instrument


async def present_instrument(
    transcript: TranscriptStore,
    interaction_store: InteractionStore,
    instrument_store: InstrumentStore,
    contract: InteractionContract,
    spec: dict,
    learner_id: UUID,
) -> Instrument:
    """The STANDALONE entry point (webserver.py's /instrument page):
    an instrument presentation is its own fresh session, not riding on
    an existing chat session's turn numbering. For an instrument
    presented mid-session instead, see `present_instrument_in_session`
    — `SessionLoop`'s manual instrument-turn (loop.py) uses that one."""
    session_id = await transcript.create_session(learner_id)
    return await _create_instrument_row(
        interaction_store, instrument_store, contract, spec, learner_id,
        session_id, turn_number=0, entry_state=EntryState.COLD_OPEN,
    )


async def present_instrument_in_session(
    interaction_store: InteractionStore,
    instrument_store: InstrumentStore,
    contract: InteractionContract,
    spec: dict,
    learner_id: UUID,
    session_id: UUID,
    turn_number: int,
) -> Instrument:
    """An instrument presented AS A TURN within an ONGOING session —
    "an instrument presented mid-session, at the point where the
    system is uncertain, tells you about a learner learning" (vs. the
    standalone page, which only tells you about someone doing an
    exercise). Reuses the caller's own `session_id`/`turn_number`
    rather than minting a fresh session. `entry_state=CONTINUING`,
    never `COLD_OPEN` — by construction this session already has prior
    turns."""
    return await _create_instrument_row(
        interaction_store, instrument_store, contract, spec, learner_id,
        session_id, turn_number, entry_state=EntryState.CONTINUING,
    )


async def ensure_demo_target_claim(
    claim_store: ClaimStore,
    learner_id: UUID,
    axis: ApproachAxis,
    value: StatedPreferenceLabel,
    statement: str,
    test: str,
) -> Claim:
    """DEMO-ONLY scaffolding, not part of the instrument layer proper:
    a hand-authored contract needs a real `target_claim_id` (this build
    doesn't implement instrument-driven claim CREATION — see
    `write_instrument_evidence`'s own docstring for why), so this finds
    or seeds ONE manually-authored claim on the given axis/value for a
    learner exercising a demo. A human author is standing in for the
    not-yet-built generation path here, the same way each contract
    itself is hand-authored rather than model-generated. Never called
    by `write_instrument_evidence` itself — only by demo presentation
    flows (webserver.py), one per contract's own axis/value."""
    existing = await claim_store.find_by_axis(learner_id, axis)
    for c in existing:
        if c.value is value:
            return c
    claim = Claim(
        learner_id=learner_id, statement=statement, test=test, value=value,
        confidence=0.5, source=ClaimSource.INFERRED, write_policy=ClaimWritePolicy.SLOW_DRIFT,
        statement_embedding=[0.0] * EMBEDDING_DIM,
    )
    await claim_store.create(claim)
    return claim


async def ensure_demo_target_capability_claim(
    capability_store: CapabilityClaimStore,
    learner_id: UUID,
    skill: CapabilityLabel,
    statement: str,
    test: str,
) -> CapabilityClaim:
    """The capability-side counterpart to `ensure_demo_target_claim` —
    same DEMO-ONLY scaffolding role, but `find_by_skill` (exact match)
    instead of `find_by_axis`+value-compatibility, since capability
    isn't bidirectional (see capability.py's own module docstring):
    there is no "opposite skill" a second claim could legitimately
    represent, so at most one live claim per (learner, skill) is ever
    expected."""
    existing = await capability_store.find_by_skill(learner_id, skill)
    if existing is not None:
        return existing
    claim = CapabilityClaim(
        learner_id=learner_id, statement=statement, test=test, skill=skill,
        confidence=0.5, write_policy=ClaimWritePolicy.SLOW_DRIFT,
    )
    await capability_store.create(claim)
    return claim


# ─────────────────────────── interpretation (pure, deterministic) ──────


class InstrumentOutcome(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNINFORMATIVE = "uninformative"


def _event_matches(event: InstrumentEvent, predicate: dict) -> bool:
    """A predicate is `{"event": <event_type>, "field"?: <payload key>,
    "equals"|"not_equals"?: <value>}`. `field` absent means the
    predicate constrains only the event TYPE (e.g. `{"event":
    "abandon"}`) -- deliberately the shape for exit conditions the
    contract can't read further into."""
    if event.event_type.value != predicate.get("event"):
        return False
    field = predicate.get("field")
    if field is None:
        return True
    value = event.payload.get(field)
    if "equals" in predicate:
        return value == predicate["equals"]
    if "not_equals" in predicate:
        return value != predicate["not_equals"]
    return False


def interpret_instrument(
    contract: InteractionContract, events: list[InstrumentEvent]
) -> InstrumentOutcome:
    """Pure function, no I/O, no LLM. Checks `uninformative_when`
    FIRST — an explicit exit condition (abandonment, a response the
    contract names as unreadable) always wins even if some other event
    in the same stream happens to also match a supports/contradicts
    predicate, since the contract itself declared that condition
    uninterpretable. Then `supports_when`, then `contradicts_when`.

    An event stream that matches NONE of the three predicate lists —
    the timeout case: no submit, no abandon, nothing the contract
    anticipated, just a trail that stops — falls through to
    UNINFORMATIVE by default. This is deliberate, not a missing case:
    the alternative is guessing a direction for data the contract was
    never written to read, which is exactly the confabulation risk
    `uninformative_when`'s own mandatory-non-empty rule exists to
    close off. A well-formed contract still SHOULD name its known exit
    conditions explicitly (this default is the last resort, not a
    substitute for that)."""
    for predicate in contract.uninformative_when:
        if any(_event_matches(e, predicate) for e in events):
            return InstrumentOutcome.UNINFORMATIVE
    for predicate in contract.supports_when:
        if any(_event_matches(e, predicate) for e in events):
            return InstrumentOutcome.SUPPORTS
    for predicate in contract.contradicts_when:
        if any(_event_matches(e, predicate) for e in events):
            return InstrumentOutcome.CONTRADICTS
    return InstrumentOutcome.UNINFORMATIVE


# ─────────────────────────── evidence writing ───────────────────────────


async def write_instrument_evidence(
    claim_store: ClaimStore,
    capability_store: CapabilityClaimStore,
    contract: InteractionContract,
    instrument: Instrument,
    outcome: InstrumentOutcome,
    confidence_config: ClaimConfidenceConfig | None = None,
    capability_confidence_config: CapabilityConfidenceConfig | None = None,
) -> ClaimEvidence | CapabilityEvidence | None:
    """A `supports`/`contradicts` outcome writes one evidence row
    through the EXACT existing path for whichever claim kind
    `contract.measures` names — same confidence math, same everything
    each store already had; the only new fact on the row itself is
    `source=instrument`. `uninformative` writes nothing.

    ROUTING IS THE FIX: a PREFERENCE contract writes through
    `ClaimStore` (`claim_evidence`, unchanged); a PERFORMANCE contract
    writes through `CapabilityClaimStore` (`capability_evidence`)
    instead — see `capability.py`'s own module docstring for the
    incident this closes (performance observations were silently
    accumulating into a preference claim's confidence). Which branch
    runs is decided by `contract.measures`, never by inspecting
    `target_claim_id` — `InteractionContract`'s own shape validator
    (models.py) already guarantees `target_axis`/`target_value` are
    None whenever `measures=PERFORMANCE` and `target_skill` is None
    whenever `measures=PREFERENCE`, so there is no live path here that
    could route a capability observation into a preference claim by
    mistake.

    `contradiction_was_possible=True` whenever an outcome is produced
    at all: a well-formed contract's `supports_when`/`contradicts_when`
    are both reachable by construction (that's what makes it a real
    test, not a foregone conclusion), so if this outcome fired, the
    other direction genuinely could have instead.

    `contract.target_claim_id is None` is out of scope for this build
    ("no instrument generation yet" — there is no statement/test text
    to found a new claim with, since the contract schema carries
    neither). Logged and discarded rather than guessed at."""
    if outcome is InstrumentOutcome.UNINFORMATIVE:
        return None
    if contract.target_claim_id is None:
        logger.warning(
            "Instrument %s's contract %s has no target_claim_id -- claim creation "
            "from an instrument is out of scope for this build; evidence discarded.",
            instrument.id, contract.id,
        )
        return None
    direction = (
        EvidenceDirection.SUPPORTS if outcome is InstrumentOutcome.SUPPORTS
        else EvidenceDirection.CONTRADICTS
    )

    if contract.measures is MeasurementKind.PERFORMANCE:
        capability_evidence = CapabilityEvidence(
            claim_id=contract.target_claim_id, learner_id=instrument.learner_id,
            interaction_id=instrument.interaction_id, direction=direction,
            skill=contract.target_skill, session_id=instrument.session_id,
            test_fired=True, contradiction_was_possible=True, source=EvidenceSource.INSTRUMENT,
        )
        await capability_store.append_evidence(capability_evidence)
        await capability_store.refresh(contract.target_claim_id, capability_confidence_config)
        return capability_evidence

    evidence = ClaimEvidence(
        claim_id=contract.target_claim_id, learner_id=instrument.learner_id,
        interaction_id=instrument.interaction_id, direction=direction,
        topic=f"instrument:{contract.primitive.value}", axis=contract.target_axis,
        session_id=instrument.session_id, test_fired=True,
        contradiction_was_possible=True, source=EvidenceSource.INSTRUMENT,
    )
    await claim_store.append_evidence(evidence)
    await claim_store.refresh(contract.target_claim_id, confidence_config)
    return evidence


# ─────────────────────────── locate: hand-authored demo contract ───────
#
# A worked "solve for x" example with one seeded arithmetic error.
# Legitimate pedagogy on its own terms (spot the mistake), trivially
# writable as a contract (clicked the seeded element, clicked a
# specific diagnostic wrong element, or neither), and it exercises the
# full path (present -> capture -> interpret -> write evidence) with no
# drag mechanics.
#
# Axis/value: successfully locating an error requires tracing each
# worked step rather than skimming to the result -- SUPPORTS
# WORKED_STEPS_RESULT/WANTS_STEPS_SHOWN. Clicking the FINAL line
# specifically (as if an error "must" be in the answer, without having
# verified any intermediate step) is the one wrong click that's
# actually diagnostic of the opposite pattern -- CONTRADICTS. Any other
# click (a correct-but-unexamined step, blank space) or no terminal
# event at all is uninformative: getting it wrong for an unknown reason
# says nothing about a standing teaching-style preference, and guessing
# one is exactly the confabulation this whole layer exists to avoid.

LOCATE_DEMO_SPEC: dict = {
    "prompt": "One of the steps below contains an error. Click the step where it first appears.",
    "steps": [
        {"id": "step_1", "text": "2x + 6 = 14"},
        {"id": "step_2", "text": "2x = 14 - 6"},
        {"id": "step_3", "text": "2x = 6"},
        {"id": "step_4", "text": "x = 3"},
    ],
}
LOCATE_SEEDED_ERROR_ELEMENT = "step_3"  # 14 - 6 = 8, not 6
LOCATE_DIAGNOSTIC_WRONG_ELEMENT = "step_4"  # the final line -- skipped verifying steps


def build_locate_demo_contract(target_claim_id: UUID | None) -> InteractionContract:
    """The one hand-authored contract this build ships. `generator_
    version="hand-authored-v1"` marks it explicitly as not model-
    generated -- see this module's own docstring for why that
    isolation matters. `measures=PERFORMANCE`/`target_skill` (never
    `target_axis`/`target_value`) — locate measures whether the
    learner can trace worked steps, not a framing preference; see
    capability.py's own module docstring for the incident that fixed
    this shape."""
    return InteractionContract(
        id=uuid4(),
        target_claim_id=target_claim_id,
        measures=MeasurementKind.PERFORMANCE,
        target_skill=CapabilityLabel.TRACES_WORKED_STEPS,
        primitive=InstrumentPrimitive.LOCATE,
        supports_when=[
            {"event": "click", "field": "clicked_element", "equals": LOCATE_SEEDED_ERROR_ELEMENT},
        ],
        contradicts_when=[
            {"event": "click", "field": "clicked_element", "equals": LOCATE_DIAGNOSTIC_WRONG_ELEMENT},
        ],
        uninformative_when=[
            {"event": "abandon"},
        ],
        generator_version=INSTRUMENT_CONTRACT_VERSION,
    )


# ─────────────────────────── predict: hand-authored demo contract ──────
#
# A small piece of machinery (a two-variable swap) is shown step by
# step; the learner predicts the resulting value of `y` BEFORE it's
# revealed, by choosing among four candidate answers. Closest of the
# six primitives to a pure capability probe (method_capabilities.py's
# own reasoning for why this was the second one built): correctly
# deriving the outcome requires mentally executing the steps forward,
# not judging a finished worked example the way locate does.
#
# Axis/value: correctly predicting the outcome from the RULE (the
# sequence of operations) before seeing it worked out supports RULE_
# BEFORE_EXAMPLE -- engaging with the abstract mechanism ahead of the
# concrete resolved instance. The one wrong answer that's actually
# diagnostic is `y`'s ORIGINAL value (9) -- picking it means reading the
# variable's starting value off the page without tracking the swap at
# all, the behavioral opposite of deriving forward from the rule, so it
# CONTRADICTS this same target claim directly (write_instrument_evidence
# writes CONTRADICTS onto contract.target_claim_id itself -- there is no
# separate "opposite claim" lookup here, unlike reconcile_candidate's
# axis-first matching between DIFFERENT claims). Any other wrong pick,
# or no submission at all, is uninformative for the same reason
# locate's is: guessing a direction for an unread wrong answer is
# exactly the confabulation this layer exists to avoid.
#
# `interpret_instrument` needed ZERO changes to support this: the
# predicate shape ({"event", "field", "equals"}) is already generic
# over payload field names, so `chosen_option` here and
# `clicked_element` in locate are handled identically. That this
# required no special-casing is itself the check this build asked for.

PREDICT_DEMO_SPEC: dict = {
    "prompt": "Trace this step by step, then predict what happens before the answer is revealed.",
    "machinery": ["x = 5", "y = 9", "temp = x", "x = y", "y = temp"],
    "question": "What is the value of y after this runs?",
    "options": [
        {"id": "opt_5", "text": "5"},
        {"id": "opt_9", "text": "9"},
        {"id": "opt_14", "text": "14"},
        {"id": "opt_undefined", "text": "undefined -- temp was never given a starting value"},
    ],
}
PREDICT_CORRECT_OPTION = "opt_5"  # the swap's actual result: y ends up holding x's original value
PREDICT_DIAGNOSTIC_WRONG_OPTION = "opt_9"  # y's UNCHANGED starting value -- didn't trace the swap


def build_predict_demo_contract(target_claim_id: UUID | None) -> InteractionContract:
    """The second hand-authored contract this build ships -- same
    discipline as `build_locate_demo_contract`: not model-generated,
    isolating the primitive's plumbing from generation quality.
    `measures=PERFORMANCE`/`target_skill` — predict measures whether
    the learner can derive an outcome forward, not a framing
    preference (this contract originally set `target_axis`/
    `target_value` here; that was the exact incident capability.py's
    own module docstring describes)."""
    return InteractionContract(
        id=uuid4(),
        target_claim_id=target_claim_id,
        measures=MeasurementKind.PERFORMANCE,
        target_skill=CapabilityLabel.DERIVES_FORWARD,
        primitive=InstrumentPrimitive.PREDICT,
        supports_when=[
            {"event": "click", "field": "chosen_option", "equals": PREDICT_CORRECT_OPTION},
        ],
        contradicts_when=[
            {"event": "click", "field": "chosen_option", "equals": PREDICT_DIAGNOSTIC_WRONG_OPTION},
        ],
        uninformative_when=[
            {"event": "abandon"},
        ],
        generator_version=INSTRUMENT_CONTRACT_VERSION,
    )


# ─────────────────────────── demo-contract registry ─────────────────────
#
# Declared data, same spirit as method_capabilities.py: one entry per
# IMPLEMENTED primitive, everything a presentation flow needs without
# knowing which primitive it is. Both the standalone webserver route
# and SessionLoop's manual instrument-turn (loop.py) call
# `present_demo_instrument` against this registry rather than each
# having its own per-primitive if/else.


@dataclass(frozen=True)
class DemoContractDef:
    """`measures` decides which pair of the remaining fields is
    populated — mirrors `InteractionContract`'s own polymorphic shape
    (models.py) exactly, for the identical reason: a PREFERENCE entry
    sets `target_axis`+`target_value`; a PERFORMANCE entry sets
    `target_skill` instead. `_ensure_target_claim` below is what reads
    `measures` to call the right "ensure a target exists" helper and
    return a `target_claim_id` generically, so `present_demo_instrument`/
    `present_demo_instrument_in_session` never need their own
    per-measures branching."""

    build_contract: Callable[[UUID | None], InteractionContract]
    spec: dict
    measures: MeasurementKind
    claim_statement: str
    claim_test: str
    target_axis: ApproachAxis | None = None
    target_value: StatedPreferenceLabel | None = None
    target_skill: CapabilityLabel | None = None


DEMO_CONTRACTS: dict[InstrumentPrimitive, DemoContractDef] = {
    InstrumentPrimitive.LOCATE: DemoContractDef(
        build_contract=build_locate_demo_contract,
        spec=LOCATE_DEMO_SPEC,
        measures=MeasurementKind.PERFORMANCE,
        target_skill=CapabilityLabel.TRACES_WORKED_STEPS,
        claim_statement="The learner can trace worked steps closely enough to catch an error in them.",
        claim_test="Given a worked example with one seeded error, the learner will locate it by "
                    "tracing each step rather than skimming to the result.",
    ),
    InstrumentPrimitive.PREDICT: DemoContractDef(
        build_contract=build_predict_demo_contract,
        spec=PREDICT_DEMO_SPEC,
        measures=MeasurementKind.PERFORMANCE,
        target_skill=CapabilityLabel.DERIVES_FORWARD,
        claim_statement="The learner can derive a mechanism's outcome forward before it's revealed.",
        claim_test="Given a small piece of machinery traced step by step, the learner will predict "
                    "the correct outcome by deriving it forward rather than guessing.",
    ),
}


async def _ensure_target_claim(
    claim_store: ClaimStore, capability_store: CapabilityClaimStore,
    definition: DemoContractDef, learner_id: UUID,
) -> UUID:
    """Reads `definition.measures` to call the right "ensure a demo
    target exists" helper and returns just the id — the one place that
    branches on measures so `present_demo_instrument`/
    `present_demo_instrument_in_session` don't each need their own."""
    if definition.measures is MeasurementKind.PERFORMANCE:
        capability_claim = await ensure_demo_target_capability_claim(
            capability_store, learner_id, definition.target_skill,
            definition.claim_statement, definition.claim_test,
        )
        return capability_claim.id
    claim = await ensure_demo_target_claim(
        claim_store, learner_id, definition.target_axis, definition.target_value,
        definition.claim_statement, definition.claim_test,
    )
    return claim.id


async def present_demo_instrument(
    transcript: TranscriptStore,
    interaction_store: InteractionStore,
    instrument_store: InstrumentStore,
    contract_store: InteractionContractStore,
    claim_store: ClaimStore,
    capability_store: CapabilityClaimStore,
    primitive: InstrumentPrimitive,
    learner_id: UUID,
) -> Instrument:
    """The standalone webserver route's entry point (its own fresh
    session — see `present_instrument`'s own docstring). For an
    instrument presented mid-session instead, see
    `present_demo_instrument_in_session`, which `SessionLoop`'s manual
    instrument-turn (loop.py) calls. Looks up `DEMO_CONTRACTS[primitive]`,
    ensures a target claim exists in whichever store `measures` names
    (`_ensure_target_claim` — demo scaffolding, not instrument-driven
    claim creation), persists a fresh contract row, and presents it.
    Raises `KeyError` for a primitive with no entry (choose/order/
    adjust/construct) rather than silently doing nothing — an
    unimplemented primitive should fail loudly here, not pretend to
    work."""
    definition = DEMO_CONTRACTS[primitive]
    target_claim_id = await _ensure_target_claim(claim_store, capability_store, definition, learner_id)
    contract = definition.build_contract(target_claim_id)
    await contract_store.create(contract)
    return await present_instrument(
        transcript, interaction_store, instrument_store, contract, definition.spec, learner_id,
    )


async def present_demo_instrument_in_session(
    interaction_store: InteractionStore,
    instrument_store: InstrumentStore,
    contract_store: InteractionContractStore,
    claim_store: ClaimStore,
    capability_store: CapabilityClaimStore,
    primitive: InstrumentPrimitive,
    learner_id: UUID,
    session_id: UUID,
    turn_number: int,
) -> Instrument:
    """`SessionLoop`'s manual instrument-turn (loop.py) calls this —
    same registry lookup as `present_demo_instrument`, but presents
    into an ONGOING session/turn sequence via
    `present_instrument_in_session` instead of minting a fresh one.
    Raises `KeyError` for an unimplemented primitive, same as the
    standalone entry point."""
    definition = DEMO_CONTRACTS[primitive]
    target_claim_id = await _ensure_target_claim(claim_store, capability_store, definition, learner_id)
    contract = definition.build_contract(target_claim_id)
    await contract_store.create(contract)
    return await present_instrument_in_session(
        interaction_store, instrument_store, contract, definition.spec,
        learner_id, session_id, turn_number,
    )
