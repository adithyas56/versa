"""CapabilityClaimStore -- real DB. Mirrors test_claims_store.py's
style for the equivalent preference-side operations."""
from uuid import uuid4

import pytest
import pytest_asyncio

from probe.capability import CapabilityClaimStore
from probe.models import (
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityLabel,
    CapabilityStatus,
    ClaimWritePolicy,
    EvidenceDirection,
    EvidenceSource,
    QuestionAuthor,
)


@pytest_asyncio.fixture(loop_scope="session")
async def capability_store(clean_pool):
    return CapabilityClaimStore(clean_pool)


def _claim(learner_id, skill=CapabilityLabel.TRACES_WORKED_STEPS, **overrides) -> CapabilityClaim:
    defaults = dict(
        learner_id=learner_id,
        statement="The learner can trace worked steps to find an error.",
        test="Given a worked example with a seeded error, the learner will locate it.",
        skill=skill,
        confidence=0.5,
        write_policy=ClaimWritePolicy.SLOW_DRIFT,
    )
    defaults.update(overrides)
    return CapabilityClaim(**defaults)


@pytest.mark.asyncio(loop_scope="session")
async def test_create_and_get_roundtrip(capability_store, learner_id, clean_pool):
    claim = _claim(learner_id)
    await capability_store.create(claim)
    fetched = await capability_store.get(claim.id)
    assert fetched is not None
    assert fetched.skill is CapabilityLabel.TRACES_WORKED_STEPS
    assert fetched.status is CapabilityStatus.CANDIDATE


@pytest.mark.asyncio(loop_scope="session")
async def test_find_by_skill_matches_exactly_not_by_similarity(capability_store, learner_id, clean_pool):
    matching = _claim(learner_id, skill=CapabilityLabel.TRACES_WORKED_STEPS)
    other_skill = _claim(learner_id, skill=CapabilityLabel.DERIVES_FORWARD)
    await capability_store.create(matching)
    await capability_store.create(other_skill)

    found = await capability_store.find_by_skill(learner_id, CapabilityLabel.TRACES_WORKED_STEPS)
    assert found is not None
    assert found.id == matching.id

    none_found = await capability_store.find_by_skill(uuid4(), CapabilityLabel.TRACES_WORKED_STEPS)
    assert none_found is None


@pytest.mark.asyncio(loop_scope="session")
async def test_refresh_updates_confidence_from_evidence(
    capability_store, learner_id, transcript, interaction_recorder, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    claim = _claim(learner_id)
    await capability_store.create(claim)
    await capability_store.append_evidence(CapabilityEvidence(
        claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
        direction=EvidenceDirection.SUPPORTS, skill=CapabilityLabel.TRACES_WORKED_STEPS,
        session_id=session_id, test_fired=True, contradiction_was_possible=True,
        source=EvidenceSource.INSTRUMENT,
    ))
    refreshed = await capability_store.refresh(claim.id)
    assert refreshed.confidence > claim.confidence
    assert refreshed.status is CapabilityStatus.CANDIDATE  # no promotion logic yet -- deliberate


@pytest.mark.asyncio(loop_scope="session")
async def test_ineligible_evidence_does_not_move_confidence(
    capability_store, learner_id, transcript, interaction_recorder, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    claim = _claim(learner_id)
    await capability_store.create(claim)
    await capability_store.append_evidence(CapabilityEvidence(
        claim_id=claim.id, learner_id=learner_id, interaction_id=interaction.id,
        direction=EvidenceDirection.SUPPORTS, skill=CapabilityLabel.TRACES_WORKED_STEPS,
        session_id=session_id, test_fired=False, contradiction_was_possible=True,
    ))
    refreshed = await capability_store.refresh(claim.id)
    assert refreshed.confidence == 0.5
