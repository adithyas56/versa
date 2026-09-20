"""demo_fixture.seed_demo_fixture -- real DB, no LLM/embedding call.
Confirms the two portraits are genuinely opposite and re-seeding is
idempotent (no duplicate claims on a second call)."""
import pytest

from probe.claims import ClaimStore
from probe.demo_fixture import (
    ABSTRACT_PORTRAIT_LABEL,
    CONCRETE_PORTRAIT_LABEL,
    seed_demo_fixture,
)
from probe.models import ClaimStatus, StatedPreferenceLabel


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_demo_fixture_creates_two_opposite_portraits(clean_pool):
    learners = await seed_demo_fixture(clean_pool)
    assert learners["concrete"].label == CONCRETE_PORTRAIT_LABEL
    assert learners["abstract"].label == ABSTRACT_PORTRAIT_LABEL

    claim_store = ClaimStore(clean_pool)
    concrete_claims = await claim_store.list_for_learner(learners["concrete"].id)
    abstract_claims = await claim_store.list_for_learner(learners["abstract"].id)

    concrete_values = {c.value for c in concrete_claims}
    abstract_values = {c.value for c in abstract_claims}
    assert concrete_values == {
        StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, StatedPreferenceLabel.WANTS_ANALOGIES,
    }
    assert abstract_values == {
        StatedPreferenceLabel.RULE_BEFORE_EXAMPLE, StatedPreferenceLabel.NO_ANALOGIES,
    }
    # Genuinely opposite, not just different -- every concrete claim's
    # value has a defined opposite among the abstract claims.
    assert all(c.status is ClaimStatus.PROMOTED for c in concrete_claims + abstract_claims)
    assert all(c.confidence == 0.85 for c in concrete_claims + abstract_claims)


@pytest.mark.asyncio(loop_scope="session")
async def test_seed_demo_fixture_is_idempotent(clean_pool):
    await seed_demo_fixture(clean_pool)
    learners_again = await seed_demo_fixture(clean_pool)

    claim_store = ClaimStore(clean_pool)
    concrete_claims = await claim_store.list_for_learner(learners_again["concrete"].id)
    assert len(concrete_claims) == 2  # not 4 -- the second call added nothing new
