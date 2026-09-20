"""score_capability_predictions_for_all_learners -- the one I/O-
boundary test, mirroring test_score_predictions_store.py: confirms
real capability evidence built through write_instrument_evidence (the
same path production uses) wires correctly into
compute_capability_prediction_trials and bucket_trials."""
import pytest

from probe.capability import CapabilityClaimStore
from probe.claims import ClaimStore
from probe.instruments import (
    LOCATE_SEEDED_ERROR_ELEMENT,
    InstrumentEventType,
    InstrumentPrimitive,
    InteractionContractStore,
    InstrumentStore,
    interpret_instrument,
    present_demo_instrument,
    write_instrument_evidence,
)
from probe.models import InstrumentEvent
from probe.score_predictions import score_capability_predictions_for_all_learners


@pytest.mark.asyncio(loop_scope="session")
async def test_score_capability_predictions_scores_a_real_capability_history(
    claim_store, transcript, interaction_store, learner_id, clean_pool
):
    contract_store = InteractionContractStore(clean_pool)
    instrument_store = InstrumentStore(clean_pool)
    capability_store = CapabilityClaimStore(clean_pool)

    target_claim_id = None
    for round_ in range(3):
        instrument = await present_demo_instrument(
            transcript, interaction_store, instrument_store, contract_store, claim_store,
            capability_store, InstrumentPrimitive.LOCATE, learner_id,
        )
        contract = await contract_store.get(instrument.contract_id)
        target_claim_id = contract.target_claim_id
        events = [
            InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.START,
                             payload={}, elapsed_ms=0),
            InstrumentEvent(instrument_id=instrument.id, seq=1, event_type=InstrumentEventType.CLICK,
                             payload={"clicked_element": LOCATE_SEEDED_ERROR_ELEMENT}, elapsed_ms=800),
        ]
        outcome = interpret_instrument(contract, events)
        await write_instrument_evidence(claim_store, capability_store, contract, instrument, outcome)

    bins, overall_brier, total_trials_computed = await score_capability_predictions_for_all_learners(
        capability_store
    )
    total_trials = sum(b.n_trials for b in bins)
    assert total_trials >= 2  # 3 supporting rounds on one (find_by_skill-reused) claim -> 2 trials
    assert total_trials_computed == total_trials
    assert overall_brier is not None
    assert all(b.n_hits == b.n_trials for b in bins if b.n_trials)  # every round supported -> every trial a hit

    evidence = await capability_store.list_evidence(target_claim_id)
    assert len(evidence) == 3  # all three rounds reused the same claim via find_by_skill
