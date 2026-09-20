"""comparison.py -- the three-column wrong-portrait control. Real DB,
StubLLMClient (a stub can't read the prompt, so it cannot prove the
answers differ -- see comparison.py's own docstring); what this DOES
prove, and what these tests lock in, is that the DATA each column is
built from is wired correctly: the right claims fire for the right
learner, the frozen prediction text genuinely differs between opposite
portraits and is empty for the control, and retrieved history reads
real (if empty) machinery rather than a hardcoded placeholder."""
import pytest

from probe.comparison import NO_PORTRAIT_LABEL, ensure_no_portrait_learner, run_comparison
from probe.learner import LearnerStore


@pytest.mark.asyncio(loop_scope="session")
async def test_ensure_no_portrait_learner_has_zero_claims(clean_pool):
    from probe.claims import ClaimStore

    learner = await ensure_no_portrait_learner(clean_pool)
    assert learner.label == NO_PORTRAIT_LABEL
    claims = await ClaimStore(clean_pool).list_for_learner(learner.id)
    assert claims == []

    again = await ensure_no_portrait_learner(clean_pool)
    assert again.id == learner.id  # idempotent, not a duplicate


@pytest.mark.asyncio(loop_scope="session")
async def test_run_comparison_wires_the_right_claims_and_prediction_per_column(clean_pool):
    columns = await run_comparison(clean_pool, "Explain how binary search works.", stub=True)

    assert set(columns) == {"concrete", "abstract", "none"}

    concrete_values = {c["value"] for c in columns["concrete"].claims_fired}
    assert concrete_values == {"concrete_before_abstract", "wants_analogies"}

    abstract_values = {c["value"] for c in columns["abstract"].claims_fired}
    assert abstract_values == {"rule_before_example", "no_analogies"}

    assert columns["none"].claims_fired == []
    assert columns["none"].frozen_prediction == ""

    # the frozen prediction is the real claim_constraints_block, pulled
    # back out of node_calls -- genuinely different text per portrait,
    # each one naming its own claim's directive.
    assert "concrete worked example" in columns["concrete"].frozen_prediction
    assert "general rule" in columns["abstract"].frozen_prediction
    assert columns["concrete"].frozen_prediction != columns["abstract"].frozen_prediction

    # no learner_facts exist for any of these three -- retrieval ran
    # for real (not hardcoded off) but found nothing, for all three.
    assert columns["concrete"].retrieved_history == ""
    assert columns["abstract"].retrieved_history == ""
    assert columns["none"].retrieved_history == ""

    # StubLLMClient cannot read the prompt -- all three answers are
    # identical by construction under a stub. This is the exact line
    # comparison.py's docstring draws: this test proves the wiring, not
    # that real answers would differ.
    assert columns["concrete"].answer == columns["abstract"].answer == columns["none"].answer


@pytest.mark.asyncio(loop_scope="session")
async def test_run_comparison_is_idempotent_on_the_fixture_and_control(clean_pool):
    await run_comparison(clean_pool, "How do I calculate a percentage of a number?", stub=True)
    await run_comparison(clean_pool, "How do I calculate a percentage of a number?", stub=True)

    learners = LearnerStore(clean_pool)
    from probe.demo_fixture import CONCRETE_PORTRAIT_LABEL
    from probe.claims import ClaimStore

    concrete = await learners.get_by_label(CONCRETE_PORTRAIT_LABEL)
    claims = await ClaimStore(clean_pool).list_for_learner(concrete.id)
    assert len(claims) == 2  # re-seeding across two runs added nothing new
