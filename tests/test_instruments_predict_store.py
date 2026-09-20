"""present_demo_instrument / DEMO_CONTRACTS registry, and predict's
write_instrument_evidence through the existing capability-evidence
path -- real DB. Mirrors test_instruments_store.py's locate coverage;
kept separate so a reviewer can see the second primitive added no new
mechanism, just a second registry entry.

Both locate and predict are PERFORMANCE-measuring (method_capabilities.py),
so both route through CapabilityClaimStore, not ClaimStore -- see
capability.py's own module docstring for the incident that made this
the correct routing."""
import pytest

from probe.capability import CapabilityClaimStore
from probe.claims import ClaimStore
from probe.instruments import (
    DEMO_CONTRACTS,
    PREDICT_CORRECT_OPTION,
    InstrumentEventStore,
    InstrumentOutcome,
    InstrumentPrimitive,
    InstrumentStore,
    InteractionContractStore,
    interpret_instrument,
    present_demo_instrument,
    write_instrument_evidence,
)
from probe.models import CapabilityLabel, EvidenceSource, InstrumentEvent, InstrumentEventType


@pytest.mark.asyncio(loop_scope="session")
async def test_present_demo_instrument_via_registry_for_both_primitives(
    claim_store, transcript, interaction_store, learner_id, clean_pool
):
    contract_store = InteractionContractStore(clean_pool)
    instrument_store = InstrumentStore(clean_pool)
    capability_store = CapabilityClaimStore(clean_pool)

    locate_instrument = await present_demo_instrument(
        transcript, interaction_store, instrument_store, contract_store, claim_store,
        capability_store, InstrumentPrimitive.LOCATE, learner_id,
    )
    predict_instrument = await present_demo_instrument(
        transcript, interaction_store, instrument_store, contract_store, claim_store,
        capability_store, InstrumentPrimitive.PREDICT, learner_id,
    )
    assert locate_instrument.primitive is InstrumentPrimitive.LOCATE
    assert predict_instrument.primitive is InstrumentPrimitive.PREDICT
    assert locate_instrument.spec == DEMO_CONTRACTS[InstrumentPrimitive.LOCATE].spec
    assert predict_instrument.spec == DEMO_CONTRACTS[InstrumentPrimitive.PREDICT].spec

    # Each primitive gets its OWN target capability claim (different
    # skill) -- not sharing locate's claim, and neither lands in
    # ClaimStore at all.
    locate_contract = await contract_store.get(locate_instrument.contract_id)
    predict_contract = await contract_store.get(predict_instrument.contract_id)
    assert locate_contract.target_claim_id != predict_contract.target_claim_id
    predict_claim = await capability_store.get(predict_contract.target_claim_id)
    assert predict_claim.skill is CapabilityLabel.DERIVES_FORWARD
    assert (await claim_store.list_for_learner(learner_id)) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_present_demo_instrument_raises_for_an_unimplemented_primitive(
    claim_store, transcript, interaction_store, learner_id, clean_pool
):
    contract_store = InteractionContractStore(clean_pool)
    instrument_store = InstrumentStore(clean_pool)
    capability_store = CapabilityClaimStore(clean_pool)
    with pytest.raises(KeyError):
        await present_demo_instrument(
            transcript, interaction_store, instrument_store, contract_store, claim_store,
            capability_store, InstrumentPrimitive.CHOOSE, learner_id,
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_predict_write_instrument_evidence_through_the_existing_path(
    claim_store, transcript, interaction_store, learner_id, clean_pool
):
    contract_store = InteractionContractStore(clean_pool)
    instrument_store = InstrumentStore(clean_pool)
    capability_store = CapabilityClaimStore(clean_pool)
    instrument = await present_demo_instrument(
        transcript, interaction_store, instrument_store, contract_store, claim_store,
        capability_store, InstrumentPrimitive.PREDICT, learner_id,
    )
    contract = await contract_store.get(instrument.contract_id)
    claim_before = await capability_store.get(contract.target_claim_id)

    events = [
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.START,
                         payload={}, elapsed_ms=0),
        InstrumentEvent(instrument_id=instrument.id, seq=1, event_type=InstrumentEventType.CLICK,
                         payload={"chosen_option": PREDICT_CORRECT_OPTION}, elapsed_ms=2100),
    ]
    outcome = interpret_instrument(contract, events)
    assert outcome is InstrumentOutcome.SUPPORTS

    evidence = await write_instrument_evidence(claim_store, capability_store, contract, instrument, outcome)
    assert evidence.source is EvidenceSource.INSTRUMENT
    assert evidence.contradiction_was_possible is True

    claim_after = await capability_store.get(contract.target_claim_id)
    assert claim_after.confidence > claim_before.confidence
