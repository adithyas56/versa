"""ClassifyTurnOutcome, GenerateAbstractForm, LLMSelectionPredictor
(interaction_nodes.py) — all fast-tier, all optional, all fired off
the critical path by loop.py. Tested here against StubLLMClient, no
real API calls.
"""

import json
from uuid import uuid4

import pytest

from probe.interaction_nodes import (
    ClassifierConfig,
    ClassifyStatedPreference,
    ClassifyTurnOutcome,
    GenerateAbstractForm,
    LLMSelectionPredictor,
    render_structural_requirement,
)
from probe.llm import StubLLMClient
from probe.models import (
    InteractionOption,
    RetrievalCandidate,
    StatedPreferenceLabel,
    TurnOutcomeLabel,
)

# ─────────────────────────── ClassifyTurnOutcome ───────────────────────


@pytest.mark.asyncio
async def test_default_stub_response_is_an_abstention():
    node = ClassifyTurnOutcome(StubLLMClient())
    result = await node.run("q", "r", "next q")
    assert result.abstains is True
    assert node.last_call_count == 1


@pytest.mark.asyncio
async def test_a_confident_canned_classification_is_not_an_abstention():
    llm = StubLLMClient(
        canned={
            "CLASSIFY:TURN_OUTCOME": json.dumps(
                {"outcome": "matched", "confidence": 0.95, "abstains": False}
            )
        }
    )
    node = ClassifyTurnOutcome(llm)
    result = await node.run("q", "r", "next q")
    assert result.outcome is TurnOutcomeLabel.MATCHED
    assert result.abstains is False


@pytest.mark.asyncio
async def test_low_confidence_forces_abstention_even_if_model_says_otherwise():
    """Belt and suspenders: a model can under-report its own
    uncertainty by leaving abstains=false; the wrapper must not trust
    that alone."""
    llm = StubLLMClient(
        canned={
            "CLASSIFY:TURN_OUTCOME": json.dumps(
                {"outcome": "matched", "confidence": 0.2, "abstains": False}
            )
        }
    )
    node = ClassifyTurnOutcome(llm, ClassifierConfig(min_confidence=0.6))
    result = await node.run("q", "r", "next q")
    assert result.abstains is True


@pytest.mark.asyncio
async def test_unparseable_response_degrades_to_abstention_not_a_crash():
    llm = StubLLMClient(canned={"CLASSIFY:TURN_OUTCOME": "not json at all"})
    node = ClassifyTurnOutcome(llm)
    result = await node.run("q", "r", "next q")
    assert result.abstains is True
    assert result.outcome is TurnOutcomeLabel.DEFERRED


@pytest.mark.asyncio
async def test_an_invalid_outcome_value_degrades_to_abstention():
    llm = StubLLMClient(
        canned={"CLASSIFY:TURN_OUTCOME": json.dumps({"outcome": "not_a_real_outcome", "confidence": 0.9})}
    )
    node = ClassifyTurnOutcome(llm)
    result = await node.run("q", "r", "next q")
    assert result.abstains is True


@pytest.mark.asyncio
async def test_selected_option_is_threaded_into_the_prompt_when_present():
    llm = StubLLMClient()
    node = ClassifyTurnOutcome(llm)
    await node.run("q", "r", "next q", prior_selected_option="the chosen option")
    assert "the chosen option" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_no_selected_option_omits_that_block():
    llm = StubLLMClient()
    node = ClassifyTurnOutcome(llm)
    await node.run("q", "r", "next q")
    assert "selecting this option" not in llm.prompts[-1]


# ─────────────────────────── ClassifyStatedPreference ───────────────────


@pytest.mark.asyncio
async def test_default_stub_response_has_no_preference():
    node = ClassifyStatedPreference(StubLLMClient())
    result = await node.run("what is a derivative?")
    assert result.has_preference is False
    assert result.stated_preference is None
    assert node.last_call_count == 1


@pytest.mark.asyncio
async def test_a_canned_explicit_preference_is_captured_verbatim_with_its_label():
    llm = StubLLMClient(
        canned={
            "CLASSIFY:STATED_PREFERENCE": json.dumps(
                {
                    "has_preference": True,
                    "stated_preference": "they want the concrete case before the abstract rule",
                    "label": "concrete_before_abstract",
                }
            )
        }
    )
    node = ClassifyStatedPreference(llm)
    result = await node.run(
        "can you give me a plain number example first, before the formal "
        "definition? I always want the concrete case before the abstract rule."
    )
    assert result.has_preference is True
    assert result.stated_preference == "they want the concrete case before the abstract rule"
    assert result.label is StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT


@pytest.mark.asyncio
async def test_a_missing_or_invalid_label_falls_back_to_other():
    """The raw text is still real signal even when the model's own
    categorization attempt failed or was omitted -- must not be
    discarded, must not raise."""
    llm = StubLLMClient(
        canned={
            "CLASSIFY:STATED_PREFERENCE": json.dumps(
                {"has_preference": True, "stated_preference": "keep it short", "label": "brief"}
            )
        }
    )
    node = ClassifyStatedPreference(llm)
    result = await node.run("keep your answers short please")
    assert result.has_preference is True
    assert result.stated_preference == "keep it short"
    assert result.label is StatedPreferenceLabel.OTHER


@pytest.mark.asyncio
async def test_has_preference_true_with_no_text_is_treated_as_no_preference():
    """Belt and suspenders: a malformed true/null pair must not produce
    a structural requirement built from nothing."""
    llm = StubLLMClient(
        canned={
            "CLASSIFY:STATED_PREFERENCE": json.dumps(
                {"has_preference": True, "stated_preference": None}
            )
        }
    )
    node = ClassifyStatedPreference(llm)
    result = await node.run("what is a derivative?")
    assert result.has_preference is False
    assert result.stated_preference is None


@pytest.mark.asyncio
async def test_unparseable_response_is_treated_as_no_preference():
    llm = StubLLMClient(canned={"CLASSIFY:STATED_PREFERENCE": "not json"})
    node = ClassifyStatedPreference(llm)
    result = await node.run("what is a derivative?")
    assert result.has_preference is False


def test_render_structural_requirement_maps_label_to_a_hand_written_imperative():
    """The raw text must NOT appear verbatim for a labeled preference --
    that's the whole point of the label -> imperative lookup instead of
    wrapping the raw extraction (a live re-run found the raw-text
    wrapper reads as a description a model can agree with while still
    generating its default answer shape)."""
    raw_text = "they want the concrete case before the abstract rule"
    text = render_structural_requirement(StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT, raw_text)
    assert "Structural requirement" in text
    assert raw_text not in text
    # The three required properties: imperative mood + explicit avoid
    # clause + priority assertion.
    assert "Open with a concrete worked example" in text
    assert "Do not begin with the abstract rule" in text
    assert "takes priority over your default" in text


@pytest.mark.parametrize(
    "label",
    [
        StatedPreferenceLabel.CONCRETE_BEFORE_ABSTRACT,
        StatedPreferenceLabel.RULE_BEFORE_EXAMPLE,
        StatedPreferenceLabel.WANTS_STEPS_SHOWN,
        StatedPreferenceLabel.PREFERS_BREVITY,
        StatedPreferenceLabel.WANTS_ANALOGIES,
        StatedPreferenceLabel.NO_ANALOGIES,
    ],
)
def test_every_specific_label_has_an_imperative_with_a_negative_clause(label):
    """Every hand-written imperative must name what to avoid explicitly
    -- the negative clause is not optional, per the feature's own
    review: it was present in the phrasing that worked and absent from
    the one that didn't."""
    text = render_structural_requirement(label, "irrelevant raw text")
    assert "irrelevant raw text" not in text
    assert "Do not" in text
    assert "takes priority" in text


def test_other_label_falls_back_to_generic_wrapper_around_raw_text():
    text = render_structural_requirement(StatedPreferenceLabel.OTHER, "wants everything in rhyme")
    assert "Structure your answer so that: wants everything in rhyme" in text
    assert "takes priority over your default explanation format" in text


def test_none_label_also_falls_back_to_generic_wrapper():
    """A defensive path: `latest_pref.label` should never actually be
    None when `has_preference` was true, but the renderer must not
    crash if it somehow is."""
    text = render_structural_requirement(None, "wants everything in rhyme")
    assert "Structure your answer so that: wants everything in rhyme" in text


# ─────────────────────────── GenerateAbstractForm ──────────────────────


@pytest.mark.asyncio
async def test_default_stub_abstract():
    node = GenerateAbstractForm(StubLLMClient())
    result = await node.run("what is a derivative?", "an answer")
    assert result.abstract_form == "stub abstract form"
    assert node.last_call_count == 1


@pytest.mark.asyncio
async def test_canned_abstract_form():
    llm = StubLLMClient(canned={"ABSTRACT:FORM": json.dumps({"abstract_form": "asked for a definition"})})
    node = GenerateAbstractForm(llm)
    result = await node.run("q", "r")
    assert result.abstract_form == "asked for a definition"


@pytest.mark.asyncio
async def test_unparseable_abstract_falls_back_to_a_generic_form():
    llm = StubLLMClient(canned={"ABSTRACT:FORM": "garbage"})
    node = GenerateAbstractForm(llm)
    result = await node.run("q", "r")
    assert result.abstract_form  # non-empty, some fallback string
    assert "garbage" not in result.abstract_form


# ─────────────────────────── LLMSelectionPredictor ─────────────────────


def _option(option_id=None, text="an option") -> InteractionOption:
    return InteractionOption(
        interaction_id=uuid4(), learner_id=uuid4(), option_id=option_id or uuid4(),
        branch_id=uuid4(), option_text=text, shown_position=0,
    )


@pytest.mark.asyncio
async def test_no_options_returns_empty_without_a_call():
    llm = StubLLMClient()
    predictor = LLMSelectionPredictor(llm)
    scores = await predictor.predict([], [])
    assert scores == {}
    assert predictor.last_call_count == 0


@pytest.mark.asyncio
async def test_default_stub_falls_back_to_uniform_distribution():
    options = [_option(), _option()]
    predictor = LLMSelectionPredictor(StubLLMClient())
    scores = await predictor.predict(options, [])
    assert len(scores) == 2
    assert all(v == pytest.approx(0.5) for v in scores.values())


@pytest.mark.asyncio
async def test_canned_scores_are_parsed_and_keyed_by_option_id():
    a, b = uuid4(), uuid4()
    options = [_option(option_id=a), _option(option_id=b)]
    llm = StubLLMClient(canned={"PREDICT:SELECTION": json.dumps({str(a): 0.8, str(b): 0.2})})
    predictor = LLMSelectionPredictor(llm)
    scores = await predictor.predict(options, [])
    assert scores[a] == 0.8
    assert scores[b] == 0.2


@pytest.mark.asyncio
async def test_an_id_not_among_the_offered_options_is_ignored():
    a = uuid4()
    options = [_option(option_id=a)]
    stray = uuid4()
    llm = StubLLMClient(canned={"PREDICT:SELECTION": json.dumps({str(a): 0.9, str(stray): 0.5})})
    predictor = LLMSelectionPredictor(llm)
    scores = await predictor.predict(options, [])
    assert set(scores.keys()) == {a}


@pytest.mark.asyncio
async def test_retrieval_context_is_threaded_into_the_prompt():
    options = [_option()]
    llm = StubLLMClient()
    predictor = LLMSelectionPredictor(llm)
    candidate = RetrievalCandidate(
        scope="personal", source_id=uuid4(), retrieval_key="question",
        similarity=0.85, text="a past similar situation",
        outcome=TurnOutcomeLabel.CONTRADICTED_INTENT,
    )
    await predictor.predict(options, [candidate])
    assert "a past similar situation" in llm.prompts[-1]
    assert "contradicted_intent" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_no_retrieval_context_omits_that_block():
    options = [_option()]
    llm = StubLLMClient()
    predictor = LLMSelectionPredictor(llm)
    await predictor.predict(options, [])
    assert "Relevant history" not in llm.prompts[-1]
