"""Pure node-level tests for disambiguate.py, no DB — StubLLMClient
only. Store-level persistence is covered by test_disambiguation_store.py;
full-turn wiring (click resolution, typed-past threading) is covered by
test_disambiguation_loop_wiring.py.
"""

import json

import pytest

from probe.disambiguate import AssessAndBranch, DisambiguationOptions, FinalAnswer
from probe.llm import StubLLMClient
from probe.models import AmbiguityKind, ApproachAxis, DisambiguationBranch


@pytest.mark.asyncio(loop_scope="session")
async def test_unambiguous_message_needs_no_branches_and_costs_one_call():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps({"needs_branches": False, "branches": []}),
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("what is the derivative of x^2?")
    assert result.needs_branches is False
    assert result.branch_statements == []
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_ambiguous_message_produces_two_to_four_distinct_branches():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps(
                {
                    "needs_branches": True,
                    "branches": [
                        {"statement": "wants the power rule explained"},
                        {"statement": "wants a worked numeric example"},
                        {"statement": "wants to know why the rule works"},
                    ],
                }
            ),
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("can you help with derivatives?")
    assert result.needs_branches is True
    assert len(result.branch_statements) == 3
    assert len(set(result.branch_statements)) == 3
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_duplicate_reading_is_rejected_and_regenerated():
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json.dumps(
                {
                    "needs_branches": True,
                    "branches": [
                        {"statement": "wants the derivative of x squared"},
                        {"statement": "wants the derivative of x squared"},
                    ],
                }
            )
        return json.dumps(
            {
                "needs_branches": True,
                "branches": [
                    {"statement": "wants the derivative of x squared"},
                    {"statement": "wants to check their own attempt at it"},
                ],
            }
        )

    llm = StubLLMClient(canned={"ASSESS:BRANCH": _respond})
    node = AssessAndBranch(llm)
    result = await node.run("about x^2")
    assert result.needs_branches is True
    assert len(result.branch_statements) == 2
    assert node.last_call_count == 2  # rejected once, retried once


@pytest.mark.asyncio(loop_scope="session")
async def test_exhausted_retries_degrade_to_not_ambiguous_rather_than_crash():
    llm = StubLLMClient(canned={"ASSESS:BRANCH": "not json at all"})
    node = AssessAndBranch(llm)
    result = await node.run("anything")
    assert result.needs_branches is False
    assert result.branch_statements == []
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_branch_count_outside_range_is_rejected():
    llm = StubLLMClient(
        canned={
            "ASSESS:BRANCH": json.dumps(
                {"needs_branches": True, "branches": [{"statement": "only one reading"}]}
            )
        }
    )
    node = AssessAndBranch(llm)
    result = await node.run("anything")
    # A single retry with the same malformed shape exhausts attempts and
    # degrades to not-ambiguous, same as any other unparseable response.
    assert result.needs_branches is False
    assert node.last_call_count == 2


def _branch(statement: str) -> DisambiguationBranch:
    import uuid

    return DisambiguationBranch(
        disambiguation_turn_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        turn_index=0,
        statement=statement,
    )


def _subject_response(pairs: list[tuple], axis: str | None = None) -> str:
    """pairs: list of (branch, text). Builds the new
    {"kind": "subject", "axis": null, "options": [...]}" wire shape."""
    return json.dumps(
        {
            "kind": "subject",
            "axis": None,
            "options": [{"branch_id": str(b.id), "text": t} for b, t in pairs],
        }
    )


def _approach_response(pairs: list[tuple], axis: str) -> str:
    sides = ["first", "second"]
    return json.dumps(
        {
            "kind": "approach",
            "axis": axis,
            "options": [
                {"branch_id": str(b.id), "text": t, "side": sides[i % 2]}
                for i, (b, t) in enumerate(pairs)
            ],
        }
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_one_option_per_branch():
    branches = [_branch("reading a"), _branch("reading b")]

    def _respond(_prompt: str) -> str:
        return _subject_response(
            [(branches[0], "Is it about reading a?"), (branches[1], "Is it about reading b?")]
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert result.kind is AmbiguityKind.SUBJECT
    assert result.axis is None
    assert len(result.proposals) == 2
    assert {p.branch_id for p in result.proposals} == {branches[0].id, branches[1].id}


@pytest.mark.asyncio(loop_scope="session")
async def test_duplicate_branch_mapping_is_rejected_and_retried():
    branches = [_branch("reading a"), _branch("reading b")]
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Both options map to the same branch -- invalid.
            return _subject_response([(branches[0], "one"), (branches[0], "two")])
        return _subject_response([(branches[0], "one"), (branches[1], "two")])

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert len(result.proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_exhausted_option_retries_return_empty_not_a_crash():
    branches = [_branch("reading a"), _branch("reading b")]
    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": "garbage"})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert result.proposals == []
    assert result.kind is None
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_no_branches_means_no_call_at_all():
    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": "should never be called"})
    node = DisambiguationOptions(llm)
    result = await node.run([])
    assert result.proposals == []


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_produces_exactly_two_options_with_axis():
    branches = [_branch("formal treatment"), _branch("intuitive treatment")]

    def _respond(_prompt: str) -> str:
        return _approach_response(
            [
                (branches[0], "Would you like the formal limit definition?"),
                (branches[1], "Would you like the intuitive slope picture?"),
            ],
            axis="rigor_intuition",
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches, message="explain derivatives", recent_history="")
    assert result.kind is AmbiguityKind.APPROACH
    assert result.axis is ApproachAxis.RIGOR_INTUITION
    assert len(result.proposals) == 2
    # side is decided and persisted at generation time -- a downstream
    # reader never re-derives which option is which pole from text.
    sides = {p.side for p in result.proposals}
    assert sides == {"first", "second"}


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_missing_side_is_rejected_and_retried():
    """Every approach-kind option must declare its side -- a response
    missing it is as invalid as a missing axis, not silently accepted
    with side=None (which would just reintroduce the text-inference
    problem this field exists to close)."""
    branches = [_branch("a"), _branch("b")]
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json.dumps({
                "kind": "approach", "axis": "concrete_general",
                "options": [
                    {"branch_id": str(branches[0].id), "text": "concrete one"},
                    {"branch_id": str(branches[1].id), "text": "general one"},
                ],
            })
        return _approach_response(
            [(branches[0], "concrete one"), (branches[1], "general one")],
            axis="concrete_general",
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert len(result.proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_both_options_same_side_is_rejected_and_retried():
    """Two options both claiming "first" (or both "second") isn't a
    real split along the axis -- reject and retry rather than accept a
    set with no genuine opposite pole."""
    branches = [_branch("a"), _branch("b")]
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return json.dumps({
                "kind": "approach", "axis": "analogy_formal",
                "options": [
                    {"branch_id": str(branches[0].id), "text": "analogy one", "side": "first"},
                    {"branch_id": str(branches[1].id), "text": "analogy two", "side": "first"},
                ],
            })
        return _approach_response(
            [(branches[0], "analogy version"), (branches[1], "formal version")],
            axis="analogy_formal",
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert len(result.proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_subject_kind_option_with_a_side_is_rejected():
    """A subject-kind option has no axis, hence no side -- one showing
    up anyway is as invalid as a subject-kind response also carrying
    an axis."""
    branches = [_branch("topic a"), _branch("topic b")]

    def _respond(_prompt: str) -> str:
        return json.dumps({
            "kind": "subject", "axis": None,
            "options": [
                {"branch_id": str(branches[0].id), "text": "topic a?", "side": "first"},
                {"branch_id": str(branches[1].id), "text": "topic b?"},
            ],
        })

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert result.proposals == []  # exhausted retries -- degrades to no options
    assert result.kind is None


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_with_three_options_is_rejected_and_retried():
    """A single axis admits exactly two sides -- a confounded three-way
    approach response must be rejected, not silently truncated."""
    branches = [_branch("a"), _branch("b"), _branch("c")]
    attempts = {"n": 0}

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _approach_response(
                [(branches[0], "one"), (branches[1], "two"), (branches[2], "three")],
                axis="scope_narrow_broad",
            )
        return _approach_response(
            [(branches[0], "narrow version of this"), (branches[1], "broad version of this")],
            axis="scope_narrow_broad",
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert len(result.proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_approach_kind_with_mismatched_lengths_is_rejected_and_retried():
    branches = [_branch("a"), _branch("b")]
    attempts = {"n": 0}
    short = "Short one?"
    long = "A " + "much " * 20 + "longer one that blows the length-match tolerance?"

    def _respond(_prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _approach_response([(branches[0], short), (branches[1], long)], axis="brevity_depth")
        return _approach_response(
            [(branches[0], "Short version A?"), (branches[1], "Short version B?")],
            axis="brevity_depth",
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    assert len(result.proposals) == 2
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_subject_kind_rejects_a_response_that_also_carries_an_axis():
    branches = [_branch("a"), _branch("b")]

    def _respond(_prompt: str) -> str:
        return json.dumps(
            {
                "kind": "subject",
                "axis": "concrete_general",  # invalid: subject-kind must not carry an axis
                "options": [{"branch_id": str(branches[0].id), "text": "one"},
                            {"branch_id": str(branches[1].id), "text": "two"}],
            }
        )

    llm = StubLLMClient(canned={"DISAMBIGUATE:OPTIONS": _respond})
    node = DisambiguationOptions(llm)
    result = await node.run(branches)
    # Exhausts retries against the same invalid (memoryless stub) response.
    assert result.proposals == []
    assert node.last_call_count == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_uses_branch_context_when_given():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "the answer"})
    node = FinalAnswer(llm)
    result = await node.run("help me", branch_context="wants the power rule explained")
    assert result == "the answer"
    assert "wants the power rule explained" in llm.prompts[-1]
    assert node.last_call_count == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_direct_call_has_no_branch_scaffolding():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "direct answer"})
    node = FinalAnswer(llm)
    result = await node.run("what's the derivative of x^2?")
    assert result == "direct answer"
    assert "confirmed they meant" not in llm.prompts[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_receives_recent_history_to_resolve_references():
    """Regression test for the live-confirmed failure: a bare reference
    ("sketch that") answered with zero conversation context produced a
    completely unrelated topic. FinalAnswer must actually see
    recent_history when given it, not just accept the parameter."""
    llm = StubLLMClient(canned={"FINAL:ANSWER": "answer grounded in history"})
    node = FinalAnswer(llm)
    history = "turn 8 student: what about (something)^n?\nturn 8 tutor: bring n down..."
    result = await node.run(
        "can you sketch that?", branch_context=None, recent_history=history
    )
    assert result == "answer grounded in history"
    assert history in llm.prompts[-1]


@pytest.mark.asyncio(loop_scope="session")
async def test_final_answer_with_no_recent_history_omits_the_history_block():
    llm = StubLLMClient(canned={"FINAL:ANSWER": "answer"})
    node = FinalAnswer(llm)
    await node.run("what's the derivative of x^2?", recent_history="")
    assert "Recent conversation" not in llm.prompts[-1]
