"""InteractionContractStore/InstrumentStore/InstrumentEventStore,
present_instrument, and write_instrument_evidence -- real DB, real
claim_store (the "through the existing path" claim)."""
from uuid import uuid4

import pytest
import pytest_asyncio

from probe.capability import CapabilityClaimStore
from probe.claims import ClaimStore
from probe.instruments import (
    InstrumentEventStore,
    InstrumentOutcome,
    InstrumentStore,
    InteractionContractStore,
    build_locate_demo_contract,
    interpret_instrument,
    present_instrument,
    write_instrument_evidence,
)
from probe.models import (
    CapabilityClaim,
    CapabilityLabel,
    Claim,
    ClaimSource,
    ClaimStatus,
    ClaimWritePolicy,
    EvidenceSource,
    InstrumentEvent,
    InstrumentEventType,
    StatedPreferenceLabel,
)


@pytest_asyncio.fixture(loop_scope="session")
async def contract_store(clean_pool):
    return InteractionContractStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def instrument_store(clean_pool):
    return InstrumentStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def instrument_event_store(clean_pool):
    return InstrumentEventStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def capability_store(clean_pool):
    return CapabilityClaimStore(clean_pool)


def _claim(learner_id) -> Claim:
    return Claim(
        learner_id=learner_id,
        statement="wants every intermediate step shown, never skipped",
        test="given a worked_steps_result axis choice, picks steps shown",
        value=StatedPreferenceLabel.WANTS_STEPS_SHOWN,
        confidence=0.5,
        source=ClaimSource.INFERRED,
        write_policy=ClaimWritePolicy.SLOW_DRIFT,
        statement_embedding=[0.1] * 768,
    )


def _capability_claim(learner_id, skill=CapabilityLabel.TRACES_WORKED_STEPS) -> CapabilityClaim:
    return CapabilityClaim(
        learner_id=learner_id,
        statement="The learner can trace worked steps to find an error.",
        test="Given a worked example with a seeded error, the learner will locate it.",
        skill=skill,
        confidence=0.5,
        write_policy=ClaimWritePolicy.SLOW_DRIFT,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_contract_create_and_get_roundtrip(contract_store, clean_pool):
    contract = build_locate_demo_contract(target_claim_id=None)
    await contract_store.create(contract)
    fetched = await contract_store.get(contract.id)
    assert fetched is not None
    assert fetched.supports_when == contract.supports_when
    assert fetched.contradicts_when == contract.contradicts_when
    assert fetched.uninformative_when == contract.uninformative_when
    assert fetched.target_axis == contract.target_axis
    assert fetched.target_value == contract.target_value
    assert fetched.generator_version == "hand-authored-v1"


@pytest.mark.asyncio(loop_scope="session")
async def test_present_instrument_creates_a_real_session_and_interaction(
    capability_store, contract_store, instrument_store, transcript, interaction_store, learner_id, clean_pool
):
    from probe.instruments import LOCATE_DEMO_SPEC

    claim = _capability_claim(learner_id)
    await capability_store.create(claim)
    contract = build_locate_demo_contract(target_claim_id=claim.id)
    await contract_store.create(contract)

    instrument = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )
    assert instrument.completed_at is None
    assert instrument.abandoned is False
    assert instrument.spec == LOCATE_DEMO_SPEC

    fetched = await instrument_store.get(instrument.id)
    assert fetched.id == instrument.id
    assert fetched.contract_id == contract.id


@pytest.mark.asyncio(loop_scope="session")
async def test_instrument_mark_completed_and_abandoned(instrument_store, contract_store, learner_id, transcript, interaction_store, capability_store, clean_pool):
    from probe.instruments import LOCATE_DEMO_SPEC

    claim = _capability_claim(learner_id)
    await capability_store.create(claim)
    contract = build_locate_demo_contract(target_claim_id=claim.id)
    await contract_store.create(contract)
    instrument = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )

    completed = await instrument_store.mark_completed(instrument.id)
    assert completed.completed_at is not None

    instrument2 = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )
    abandoned = await instrument_store.mark_abandoned(instrument2.id)
    assert abandoned.abandoned is True


@pytest_asyncio.fixture(loop_scope="session")
async def real_instrument(capability_store, contract_store, instrument_store, transcript, interaction_store, learner_id, clean_pool):
    from probe.instruments import LOCATE_DEMO_SPEC

    claim = _capability_claim(learner_id)
    await capability_store.create(claim)
    contract = build_locate_demo_contract(target_claim_id=claim.id)
    await contract_store.create(contract)
    return await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_events_append_and_list_ordered_by_seq(instrument_event_store, real_instrument, clean_pool):
    instrument_id = real_instrument.id
    await instrument_event_store.append(
        InstrumentEvent(instrument_id=instrument_id, seq=1, event_type=InstrumentEventType.CLICK,
                         payload={"clicked_element": "step_3"}, elapsed_ms=1200)
    )
    await instrument_event_store.append(
        InstrumentEvent(instrument_id=instrument_id, seq=0, event_type=InstrumentEventType.START,
                         payload={}, elapsed_ms=0)
    )
    events = await instrument_event_store.list_for_instrument(instrument_id)
    assert [e.seq for e in events] == [0, 1]
    assert events[1].payload == {"clicked_element": "step_3"}


@pytest.mark.asyncio(loop_scope="session")
async def test_duplicate_seq_is_rejected(instrument_event_store, real_instrument, clean_pool):
    instrument_id = real_instrument.id
    await instrument_event_store.append(
        InstrumentEvent(instrument_id=instrument_id, seq=0, event_type=InstrumentEventType.START,
                         payload={}, elapsed_ms=0)
    )
    with pytest.raises(Exception):
        await instrument_event_store.append(
            InstrumentEvent(instrument_id=instrument_id, seq=0, event_type=InstrumentEventType.ABANDON,
                             payload={}, elapsed_ms=5000)
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_write_instrument_evidence_supports_writes_through_the_existing_path(
    claim_store, capability_store, contract_store, instrument_store, transcript, interaction_store,
    learner_id, clean_pool,
):
    """locate is PERFORMANCE-measuring -- its evidence writes through
    CapabilityClaimStore, not ClaimStore (see capability.py's own
    module docstring for the incident this fixes). `claim_store` is
    still passed to `write_instrument_evidence` (it takes both stores
    unconditionally, routing by contract.measures) even though this
    particular contract never touches it."""
    from probe.instruments import LOCATE_DEMO_SPEC, LOCATE_SEEDED_ERROR_ELEMENT

    capability_claim = _capability_claim(learner_id)
    await capability_store.create(capability_claim)
    contract = build_locate_demo_contract(target_claim_id=capability_claim.id)
    await contract_store.create(contract)
    instrument = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )

    events = [
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.START,
                         payload={}, elapsed_ms=0),
        InstrumentEvent(instrument_id=instrument.id, seq=1, event_type=InstrumentEventType.CLICK,
                         payload={"clicked_element": LOCATE_SEEDED_ERROR_ELEMENT}, elapsed_ms=1400),
    ]
    outcome = interpret_instrument(contract, events)
    assert outcome is InstrumentOutcome.SUPPORTS

    evidence = await write_instrument_evidence(claim_store, capability_store, contract, instrument, outcome)
    assert evidence is not None
    assert evidence.source is EvidenceSource.INSTRUMENT
    assert evidence.contradiction_was_possible is True
    assert evidence.test_fired is True

    refreshed = await capability_store.get(capability_claim.id)
    assert refreshed.confidence > capability_claim.confidence  # the existing confidence math moved it

    stored_evidence = await capability_store.list_evidence(capability_claim.id)
    assert len(stored_evidence) == 1
    assert stored_evidence[0].source is EvidenceSource.INSTRUMENT

    # And nothing at all landed on the ordinary preference claim store --
    # the whole point of the fix.
    assert (await claim_store.list_for_learner(learner_id)) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_write_instrument_evidence_uninformative_writes_nothing(
    claim_store, capability_store, contract_store, instrument_store, transcript, interaction_store,
    learner_id, clean_pool,
):
    from probe.instruments import LOCATE_DEMO_SPEC

    capability_claim = _capability_claim(learner_id)
    await capability_store.create(capability_claim)
    contract = build_locate_demo_contract(target_claim_id=capability_claim.id)
    await contract_store.create(contract)
    instrument = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )

    events = [
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.ABANDON,
                         payload={}, elapsed_ms=3000),
    ]
    outcome = interpret_instrument(contract, events)
    assert outcome is InstrumentOutcome.UNINFORMATIVE

    evidence = await write_instrument_evidence(claim_store, capability_store, contract, instrument, outcome)
    assert evidence is None
    stored_evidence = await capability_store.list_evidence(capability_claim.id)
    assert stored_evidence == []


@pytest.mark.asyncio(loop_scope="session")
async def test_write_instrument_evidence_without_target_claim_is_discarded(
    claim_store, capability_store, contract_store, instrument_store, transcript, interaction_store,
    learner_id, clean_pool,
):
    """Out of scope for this build ("no instrument generation yet") --
    logged and discarded rather than guessing a new claim's statement/
    test text the contract doesn't carry."""
    from probe.instruments import LOCATE_DEMO_SPEC, LOCATE_SEEDED_ERROR_ELEMENT

    contract = build_locate_demo_contract(target_claim_id=None)
    await contract_store.create(contract)
    instrument = await present_instrument(
        transcript, interaction_store, instrument_store, contract, LOCATE_DEMO_SPEC, learner_id,
    )
    events = [
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.CLICK,
                         payload={"clicked_element": LOCATE_SEEDED_ERROR_ELEMENT}, elapsed_ms=900),
    ]
    outcome = interpret_instrument(contract, events)
    assert outcome is InstrumentOutcome.SUPPORTS
    evidence = await write_instrument_evidence(claim_store, capability_store, contract, instrument, outcome)
    assert evidence is None
