"""ClaimExtractor -- pure node test, StubLLMClient only. Covers: the
prompt carries the full option set (not just the selection), a
well-formed candidate parses, a candidate missing its test is rejected,
an unparseable label is rejected, and multiple competing candidates
from one episode both survive (never collapsed to one)."""

import json

import pytest

from probe.claims import ClaimExtractor
from probe.llm import StubLLMClient
from probe.models import StatedPreferenceLabel


@pytest.mark.asyncio
async def test_prompt_carries_the_full_option_set_not_just_the_selection():
    llm = StubLLMClient()
    node = ClaimExtractor(llm)
    await node.run(
        question_text="can you explain the power rule?",
        options=[
            {"option_text": "worked example first", "was_selected": True, "kind": "approach", "axis": "concrete_general"},
            {"option_text": "general formula first", "was_selected": False, "kind": "approach", "axis": "concrete_general"},
        ],
        response_text="here's a worked example...",
        trigger_reason="ranked in top 2 by prediction error",
    )
    prompt = llm.prompts[-1]
    assert "worked example first" in prompt
    assert "general formula first" in prompt  # the one NOT taken -- half the information
    assert "<-- CHOSEN" in prompt
    assert "concrete_general" in prompt


@pytest.mark.asyncio
async def test_subject_kind_episode_gets_the_decline_by_default_caveat():
    """A live diagnostic found the extractor still assigning a
    teaching-style value to subject-kind (topic) picks -- harmless to
    confidence/status now, but still filling the store with meaningless
    rows. episode_kind='subject' should add the caveat telling the
    extractor a subject pick reveals topic intent, not teaching style."""
    llm = StubLLMClient()
    node = ClaimExtractor(llm)
    await node.run("q", [], "a", "reason", episode_kind="subject")
    prompt = llm.prompts[-1]
    assert "SUBJECT-MATTER" in prompt
    assert "reveals what they wanted to talk about" in prompt


@pytest.mark.asyncio
async def test_approach_kind_episode_has_no_subject_caveat():
    llm = StubLLMClient()
    node = ClaimExtractor(llm)
    await node.run("q", [], "a", "reason", episode_kind="approach")
    prompt = llm.prompts[-1]
    assert "SUBJECT-MATTER" not in prompt


@pytest.mark.asyncio
async def test_well_formed_candidate_parses():
    canned = json.dumps([
        {"statement": "wants concrete examples first", "test": "given a concrete_general "
         "axis choice, will pick the concrete option", "value": "concrete_before_abstract",
         "topic": "calculus"},
    ])
    llm = StubLLMClient(canned={"EXTRACT:CLAIM": canned})
    node = ClaimExtractor(llm)
    result = await node.run("q", [], "a", "reason")
    assert len(result.candidates) == 1
    assert result.candidates[0].value is StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT


@pytest.mark.asyncio
async def test_candidate_missing_a_test_is_rejected():
    canned = json.dumps([
        {"statement": "wants concrete examples first", "test": "", "value": "concrete_before_abstract",
         "topic": "calculus"},
    ])
    llm = StubLLMClient(canned={"EXTRACT:CLAIM": canned})
    node = ClaimExtractor(llm)
    result = await node.run("q", [], "a", "reason")
    assert result.candidates == []


@pytest.mark.asyncio
async def test_candidate_with_unparseable_label_is_rejected():
    canned = json.dumps([
        {"statement": "wants concrete examples first", "test": "a real test",
         "value": "not_a_real_label", "topic": "calculus"},
    ])
    llm = StubLLMClient(canned={"EXTRACT:CLAIM": canned})
    node = ClaimExtractor(llm)
    result = await node.run("q", [], "a", "reason")
    assert result.candidates == []


@pytest.mark.asyncio
async def test_empty_list_is_the_normal_no_claim_result():
    llm = StubLLMClient(canned={"EXTRACT:CLAIM": "[]"})
    node = ClaimExtractor(llm)
    result = await node.run("q", [], "a", "reason")
    assert result.candidates == []


@pytest.mark.asyncio
async def test_competing_candidates_from_one_episode_both_survive():
    canned = json.dumps([
        {"statement": "wants concrete examples first", "test": "test A",
         "value": "concrete_before_abstract", "topic": "calculus"},
        {"statement": "prefers brief answers", "test": "test B",
         "value": "prefers_brevity", "topic": "calculus"},
    ])
    llm = StubLLMClient(canned={"EXTRACT:CLAIM": canned})
    node = ClaimExtractor(llm)
    result = await node.run("q", [], "a", "reason")
    assert len(result.candidates) == 2
    values = {c.value for c in result.candidates}
    assert values == {StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, StatedPreferenceLabel.PREFERS_BREVITY}
