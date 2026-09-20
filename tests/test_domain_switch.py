"""The domain switch (domain_config.py): PROBE_DOMAIN resolution, the
per-node prompt substitutions for the six domain-aware nodes, and the
one storage/retrieval touch this feature explicitly carves out --
`interactions.domain` plus the retrieval/get_recent_for_learner filter.

GenerateAbstractForm is deliberately NOT tested here for a domain
parameter -- it doesn't take one (see domain_config.py's own module
docstring); its behavior is audited by hand in this feature's
verification report, not by a unit test asserting on prompt wording it
was never given a lever to change.
"""

import json

import pytest

from probe.disambiguate import AssessAndBranch, DisambiguationOptions, FinalAnswer
from probe.domain_config import Domain, DomainConfig, load_domain_config
from probe.interaction_nodes import (
    ClassifyReferenceResolution,
    ClassifyStatedPreference,
    ClassifyTurnOutcome,
)
from probe.llm import StubLLMClient
from probe.models import DisambiguationBranch, QuestionAuthor
from probe.retrieval import RetrievalContext, stage1_filter


# ─────────────────────────── load_domain_config ──────────────────────


def test_defaults_to_education_when_unset():
    assert load_domain_config({}).domain is Domain.EDUCATION


def test_reads_general_from_env_dict():
    assert load_domain_config({"PROBE_DOMAIN": "general"}).domain is Domain.GENERAL


def test_explicit_education_from_env_dict():
    assert load_domain_config({"PROBE_DOMAIN": "education"}).domain is Domain.EDUCATION


def test_unrecognized_value_raises_rather_than_silently_falling_back():
    with pytest.raises(ValueError, match="PROBE_DOMAIN"):
        load_domain_config({"PROBE_DOMAIN": "medical"})


def test_education_and_general_differ_on_every_actor_facing_field():
    edu = DomainConfig.education()
    gen = DomainConfig.general()
    assert edu.actor_noun != gen.actor_noun
    assert edu.assistant_noun != gen.assistant_noun
    assert edu.outcome_judgment_phrase != gen.outcome_judgment_phrase
    assert "tutor" not in gen.final_answer_role_line
    assert "student" not in gen.final_answer_role_line
    assert "learner" not in gen.final_answer_closing_line


# ─────────────────────────── AssessAndBranch ──────────────────────


@pytest.mark.asyncio
async def test_assess_and_branch_general_mode_drops_student_and_tutor():
    llm = StubLLMClient()
    node = AssessAndBranch(llm, domain_config=DomainConfig.general())
    await node.run("can you help me plan this?")
    prompt = llm.prompts[-1]
    assert "student" not in prompt
    assert "tutor" not in prompt
    assert "across interpretations or intents" in prompt
    assert "concrete request, a named task" in prompt


@pytest.mark.asyncio
async def test_assess_and_branch_education_mode_is_the_default():
    llm = StubLLMClient()
    node = AssessAndBranch(llm)  # no domain_config -- must default to education
    await node.run("what's the chain rule?")
    prompt = llm.prompts[-1]
    assert "student" in prompt
    assert "across subjects or approaches" in prompt


# ─────────────────────────── DisambiguationOptions ──────────────────────


@pytest.mark.asyncio
async def test_disambiguation_options_general_mode_phrases_as_interpretations():
    llm = StubLLMClient(
        canned={"DISAMBIGUATE:OPTIONS": json.dumps({"kind": "subject", "axis": None, "options": []})}
    )
    node = DisambiguationOptions(llm, domain_config=DomainConfig.general())
    branches = [
        DisambiguationBranch(disambiguation_turn_id=__import__("uuid").uuid4(),
                              session_id=__import__("uuid").uuid4(), turn_index=0,
                              statement="wants a budget plan"),
        DisambiguationBranch(disambiguation_turn_id=__import__("uuid").uuid4(),
                              session_id=__import__("uuid").uuid4(), turn_index=0,
                              statement="wants a timeline"),
    ]
    await node.run(branches)
    prompt = llm.prompts[-1]
    assert "tutor" not in prompt
    assert "interpretation of what the person is asking for" in prompt


# ─────────────────────────── FinalAnswer ──────────────────────


@pytest.mark.asyncio
async def test_final_answer_general_mode_uses_assistant_register():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "an answer"})
    node = FinalAnswer(llm, domain_config=DomainConfig.general())
    await node.run("help me decide between two job offers")
    prompt = llm.prompts[-1]
    assert "You are an assistant having a conversation with a person." in prompt
    assert "tutor" not in prompt
    assert "student" not in prompt
    assert "learner" not in prompt


@pytest.mark.asyncio
async def test_final_answer_education_mode_unchanged_register():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "an answer"})
    node = FinalAnswer(llm)
    await node.run("what's the chain rule?")
    prompt = llm.prompts[-1]
    assert "You are a tutor having a conversation with a student." in prompt


@pytest.mark.asyncio
async def test_final_answer_context_block_still_present_regardless_of_domain():
    """The "confirmed they meant" phrase (branch resolution context) is
    not itself domain language -- must survive in both modes."""
    llm = StubLLMClient(canned={"FINAL:ANSWER": "an answer"})
    node = FinalAnswer(llm, domain_config=DomainConfig.general())
    await node.run("ok", branch_context="wants a budget plan")
    assert "confirmed they meant" in llm.prompts[-1]
    assert "person's earlier message was ambiguous" in llm.prompts[-1]


# ─────────────────────────── ClassifyTurnOutcome ──────────────────────


@pytest.mark.asyncio
async def test_classify_turn_outcome_general_mode_asks_about_usefulness():
    llm = StubLLMClient()
    node = ClassifyTurnOutcome(llm, domain_config=DomainConfig.general())
    await node.run("prior", "response", "next")
    prompt = llm.prompts[-1]
    assert "whether the next message suggests the response was useful" in prompt
    assert "tutor" not in prompt
    assert "student" not in prompt


@pytest.mark.asyncio
async def test_classify_turn_outcome_education_mode_asks_whether_answer_landed():
    llm = StubLLMClient()
    node = ClassifyTurnOutcome(llm)
    await node.run("prior", "response", "next")
    assert "whether the next question suggests the answer landed" in llm.prompts[-1]


@pytest.mark.asyncio
async def test_classify_turn_outcome_option_block_survives_general_mode():
    llm = StubLLMClient()
    node = ClassifyTurnOutcome(llm, domain_config=DomainConfig.general())
    await node.run("prior", "response", "next", prior_selected_option="option text")
    assert "selecting this option" in llm.prompts[-1]


# ─────────────────────────── ClassifyStatedPreference ──────────────────────


@pytest.mark.asyncio
async def test_stated_preference_general_mode_drops_teaching_language():
    llm = StubLLMClient()
    node = ClassifyStatedPreference(llm, domain_config=DomainConfig.general())
    await node.run("always give me the short version")
    prompt = llm.prompts[-1]
    assert "taught" not in prompt
    assert "teaching style" not in prompt
    assert "helped in general" in prompt
    assert "response style" in prompt
    assert "tutor" not in prompt
    assert "assistant" in prompt


@pytest.mark.asyncio
async def test_stated_preference_education_mode_unchanged():
    llm = StubLLMClient()
    node = ClassifyStatedPreference(llm)
    await node.run("always show me a number example first")
    prompt = llm.prompts[-1]
    assert "taught in general" in prompt
    assert "teaching style" in prompt


# ─────────────────────────── ClassifyReferenceResolution ──────────────────────


@pytest.mark.asyncio
async def test_reference_resolution_general_mode_uses_person_and_assistant():
    llm = StubLLMClient()
    node = ClassifyReferenceResolution(llm, domain_config=DomainConfig.general())
    await node.run("recent history", "the usual again", "some response")
    prompt = llm.prompts[-1]
    assert "student" not in prompt
    assert "tutor" not in prompt
    assert "person" in prompt
    assert "assistant" in prompt


# ─────────────────────────── retrieval: stage1_filter domain clause ──────────────────────


def test_stage1_filter_adds_no_domain_clause_by_default():
    from probe.retrieval import RetrievalContext
    from uuid import uuid4

    where = stage1_filter(uuid4(), RetrievalContext())
    assert "domain" not in where.sql


def test_stage1_filter_adds_domain_clause_when_given():
    from uuid import uuid4

    where = stage1_filter(uuid4(), RetrievalContext(domain=Domain.GENERAL))
    assert "i.domain" in where.sql
    assert "general" in where.params


# ─────────────────────────── interactions: domain storage ──────────────────────


@pytest.mark.asyncio(loop_scope="session")
async def test_interaction_domain_defaults_to_education(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    assert interaction.domain is Domain.EDUCATION
    async with clean_pool.acquire() as conn:
        stored = await conn.fetchval(
            "SELECT domain FROM interactions WHERE id = $1", interaction.id
        )
    assert stored == "education"


@pytest.mark.asyncio(loop_scope="session")
async def test_get_recent_for_learner_domain_filter_excludes_other_domain(
    interaction_recorder, transcript, learner_id, clean_pool
):
    session_id = await transcript.create_session(learner_id)
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="education question", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
        domain=Domain.EDUCATION,
    )
    await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=1,
        question_text="general question", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
        domain=Domain.GENERAL,
    )
    from probe.interactions import InteractionStore

    store = InteractionStore(clean_pool)
    education_only = await store.get_recent_for_learner(learner_id, 10, domain=Domain.EDUCATION)
    general_only = await store.get_recent_for_learner(learner_id, 10, domain=Domain.GENERAL)
    assert [i.question_text for i in education_only] == ["education question"]
    assert [i.question_text for i in general_only] == ["general question"]
    unfiltered = await store.get_recent_for_learner(learner_id, 10)
    assert len(unfiltered) == 2
