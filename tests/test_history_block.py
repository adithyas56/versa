"""history_block.py's pure formatting half — no DB, no LLM. Assembly
(`assemble_history_block`'s I/O) is covered by
test_history_block_loop_wiring.py against the real interaction
pipeline instead."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from probe.history_block import (
    HistoryBlockConfig,
    HistoryItem,
    _condense,
    _when_label,
    render_history_block,
)


def test_render_empty_input_returns_empty_string():
    assert render_history_block([]) == ""


def test_condense_takes_first_n_sentences():
    text = "First sentence. Second sentence. Third sentence."
    assert _condense(text, max_sentences=2, max_chars=1000) == "First sentence. Second sentence."


def test_condense_truncates_long_single_sentence_with_ellipsis():
    text = "word " * 100 + "."
    excerpt = _condense(text, max_sentences=2, max_chars=50)
    assert len(excerpt) <= 51  # 50 + the ellipsis character
    assert excerpt.endswith("…")


def test_condense_none_and_empty_return_none():
    assert _condense(None, 2, 200) is None
    assert _condense("   ", 2, 200) is None


def test_when_label_same_session_is_this_session():
    sid = uuid4()
    assert _when_label(sid, datetime.now(UTC), sid) == "Earlier in this session"


def test_when_label_different_session_same_day_is_earlier_today():
    other = uuid4()
    current = uuid4()
    assert _when_label(other, datetime.now(UTC), current) == "Earlier today"


def test_when_label_different_day_is_an_earlier_session():
    other = uuid4()
    current = uuid4()
    old = datetime.now(UTC) - timedelta(days=5)
    assert _when_label(other, old, current) == "In an earlier session"


def test_personal_paragraph_includes_every_present_field():
    item = HistoryItem(
        is_population=False,
        source_id=uuid4(),
        when_label="In an earlier session",
        asked="what is a derivative?",
        given_excerpt="A derivative measures rate of change.",
        next_question="what about integrals?",
        outcome_phrase="which showed that answer had actually landed",
    )
    block = render_history_block([item])
    assert "In an earlier session, they asked 'what is a derivative?'" in block
    assert "A derivative measures rate of change." in block
    assert "what about integrals?" in block
    assert "which showed that answer had actually landed" in block


def test_personal_paragraph_drops_null_clauses_without_filler():
    item = HistoryItem(
        is_population=False,
        source_id=uuid4(),
        when_label="Earlier today",
        asked="what is a derivative?",
        given_excerpt="A derivative measures rate of change.",
        # chose, next_question, outcome_phrase, abstract_form all None
    )
    block = render_history_block([item])
    assert "what is a derivative?" in block
    for absent_marker in ("None", "null", "N/A", "not available", "no response"):
        assert absent_marker not in block


def test_bare_asked_only_item_is_dropped_as_filler():
    """An offer turn (options shown, no response yet) or a turn
    nothing has classified/abstracted yet renders with only `when` +
    `asked` -- that is filler, not history, and must not consume a
    slot in the block at all (see _is_useful_personal_item)."""
    bare = HistoryItem(
        is_population=False, source_id=uuid4(), when_label="Earlier today", asked="q",
    )
    assert render_history_block([bare]) == ""


def test_resolution_turn_renders_asked_as_original_and_chose_as_option_text():
    """On a resolution turn, `asked` must be the student's own original
    ambiguous message (originating_question), and `chose` the clicked
    option's own copy -- never conflated, matching the codebase's
    existing question_author/originating_question split."""
    item = HistoryItem(
        is_population=False,
        source_id=uuid4(),
        when_label="Earlier today",
        asked="can you help me with derivatives?",
        chose="Let's tackle calculus derivatives",
    )
    block = render_history_block([item])
    assert "can you help me with derivatives?" in block
    assert "Let's tackle calculus derivatives" in block
    assert "went with" in block


def test_population_paragraph_uses_third_person_plural_never_this_learner():
    item = HistoryItem(
        is_population=True,
        source_id=uuid4(),
        abstract_form="chose the worked example over the stated rule",
        distinct_learner_count=25,
    )
    block = render_history_block([item])
    assert "Across other learners" in block
    assert "25 learners" in block
    assert "chose the worked example over the stated rule" in block
    # Must never be attributed to "this" learner specifically.
    assert "they asked" not in block
    assert "In an earlier session" not in block


def test_population_item_rendered_before_personal_items():
    population = HistoryItem(
        is_population=True, source_id=uuid4(), abstract_form="a pattern", distinct_learner_count=20,
    )
    personal = HistoryItem(
        is_population=False, source_id=uuid4(), when_label="Earlier today", asked="q",
        given_excerpt="an excerpt.",
    )
    block = render_history_block([population, personal])
    assert block.index("Across other learners") < block.index("Earlier today")


def test_block_carries_the_do_not_narrate_instruction():
    item = HistoryItem(
        is_population=False, source_id=uuid4(), when_label="Earlier today", asked="q",
        given_excerpt="an excerpt.",
    )
    block = render_history_block([item])
    assert "Never mention or reference it directly" in block


def test_block_truncates_to_config_cap():
    items = [
        HistoryItem(
            is_population=False,
            source_id=uuid4(),
            when_label="Earlier today",
            asked="a fairly long question " * 20,
            given_excerpt="a fairly long response. " * 20,
        )
        for _ in range(5)
    ]
    block = render_history_block(items, HistoryBlockConfig(max_block_chars=300))
    assert len(block) <= 320  # small slack for the closing newline
