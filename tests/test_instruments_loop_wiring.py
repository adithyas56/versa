"""SessionLoop.present_instrument_turn / record_instrument_event /
finalize_instrument_turn -- the step-4 wiring: an instrument presented
mid-session, not on the standalone page. No mode selector: nothing
here is called by handle_turn, only directly, matching the manual-
trigger scope this build asked for."""
import pytest

from probe.audit import NodeCallStore, TranscriptStore
from probe.claims import ClaimStore
from probe.instruments import (
    PREDICT_CORRECT_OPTION,
    InstrumentEventStore,
    InstrumentOutcome,
    InstrumentPrimitive,
)
from probe.interactions import InteractionRecorder, InteractionStore, TurnOutcomeStore
from probe.llm import StubLLMClient
from probe.loop import SessionLoop
from probe.models import EvidenceSource, InstrumentEvent, InstrumentEventType


def _build_loop(pool, embedding_client):
    interaction_store = InteractionStore(pool)
    turn_outcome_store = TurnOutcomeStore(pool)
    recorder = InteractionRecorder(
        interaction_store, turn_outcome_store, embedding_client, same_subject_threshold=0.85,
    )
    return SessionLoop(
        transcript=TranscriptStore(pool),
        node_calls=NodeCallStore(pool),
        llm=StubLLMClient(),
        interaction_recorder=recorder,
        turn_outcome_store=turn_outcome_store,
        retrieval_pool=pool,
        claim_store=ClaimStore(pool),
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_present_instrument_turn_lands_at_the_next_real_turn_number(
    clean_pool, embedding_client, transcript, learner_id
):
    """Turn numbering comes from `transcript.list_turns` -- the same
    lookup `consolidate_session` already uses for `claim_turn_index` --
    not from the separate `interactions` table, so this simulates two
    prior turns the way `handle_turn` itself would (via `record_turn`)."""
    loop = _build_loop(clean_pool, embedding_client)
    session_id = await transcript.create_session(learner_id)
    for i in range(2):
        await transcript.record_turn(session_id, i, f"q{i}")

    instrument = await loop.present_instrument_turn(session_id, InstrumentPrimitive.PREDICT)
    assert instrument.session_id == session_id  # same session, not a fresh one

    from probe.instruments import InstrumentStore

    fetched = await InstrumentStore(clean_pool).get(instrument.id)
    async with clean_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT turn_number FROM interactions WHERE id = $1", fetched.interaction_id
        )
    assert row["turn_number"] == 2  # next real slot, not 0


@pytest.mark.asyncio(loop_scope="session")
async def test_full_instrument_turn_flow_writes_evidence_through_the_normal_path(
    clean_pool, embedding_client, transcript, learner_id
):
    loop = _build_loop(clean_pool, embedding_client)
    session_id = await transcript.create_session(learner_id)

    instrument = await loop.present_instrument_turn(session_id, InstrumentPrimitive.PREDICT)

    await loop.record_instrument_event(
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.START,
                         payload={}, elapsed_ms=0)
    )
    await loop.record_instrument_event(
        InstrumentEvent(instrument_id=instrument.id, seq=1, event_type=InstrumentEventType.CLICK,
                         payload={"chosen_option": PREDICT_CORRECT_OPTION}, elapsed_ms=1900)
    )

    outcome = await loop.finalize_instrument_turn(instrument.id)
    assert outcome is InstrumentOutcome.SUPPORTS

    events = await InstrumentEventStore(clean_pool).list_for_instrument(instrument.id)
    assert len(events) == 2  # both events landed through record_instrument_event

    from probe.capability import CapabilityClaimStore
    from probe.instruments import InteractionContractStore, InstrumentStore

    fetched_instrument = await InstrumentStore(clean_pool).get(instrument.id)
    assert fetched_instrument.completed_at is not None
    assert fetched_instrument.abandoned is False

    # predict is PERFORMANCE-measuring -- its evidence lands in
    # CapabilityClaimStore, not ClaimStore (capability.py's own module
    # docstring: the incident this routing fixes).
    contract = await InteractionContractStore(clean_pool).get(instrument.contract_id)
    evidence = await CapabilityClaimStore(clean_pool).list_evidence(contract.target_claim_id)
    assert len(evidence) == 1
    assert evidence[0].source is EvidenceSource.INSTRUMENT
    assert (await ClaimStore(clean_pool).list_for_learner(learner_id)) == []


@pytest.mark.asyncio(loop_scope="session")
async def test_abandoned_instrument_turn_writes_no_evidence(
    clean_pool, embedding_client, transcript, learner_id
):
    loop = _build_loop(clean_pool, embedding_client)
    session_id = await transcript.create_session(learner_id)
    instrument = await loop.present_instrument_turn(session_id, InstrumentPrimitive.LOCATE)

    await loop.record_instrument_event(
        InstrumentEvent(instrument_id=instrument.id, seq=0, event_type=InstrumentEventType.ABANDON,
                         payload={}, elapsed_ms=2500)
    )
    outcome = await loop.finalize_instrument_turn(instrument.id)
    assert outcome is InstrumentOutcome.UNINFORMATIVE

    from probe.instruments import InstrumentStore

    fetched_instrument = await InstrumentStore(clean_pool).get(instrument.id)
    assert fetched_instrument.abandoned is True
    assert fetched_instrument.completed_at is None
