"""interactions.py must be append-only (interactions itself) or at
least never-deleting (everything else here) — same AST-based scan as
test_memory_append_only.py, plus a real DB-level enforcement test for
the one table that has an actual Postgres trigger (migration 034),
since that invariant is explicitly meant to hold even if the
application code tried to violate it.
"""

import ast
import re
from pathlib import Path

import asyncpg
import pytest

import probe.interactions as interactions_module


def _assert_module_has_no_delete(module) -> None:
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)

    docstring_nodes: set[int] = set()
    for parent in ast.walk(tree):
        if isinstance(
            parent, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(parent, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    delete_kw = re.compile(r"\bDELETE\b", re.IGNORECASE)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            assert not delete_kw.search(node.value), (
                f"{module.__name__}: string literal at line {node.lineno} "
                "contains DELETE"
            )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lname = node.name.lower()
            assert not (
                lname.startswith("delete") or lname.startswith("remove")
            ), f"{module.__name__}: function {node.name!r} at line {node.lineno} looks like a removal method"


def test_interactions_module_has_no_delete():
    _assert_module_has_no_delete(interactions_module)


def test_migration_034_has_no_delete_sql():
    migrations_dir = Path(interactions_module.__file__).resolve().parent / "migrations"
    source = (migrations_dir / "034_interactions.sql").read_text()
    code_only = "\n".join(line.split("--", 1)[0] for line in source.splitlines())
    code_only = re.sub(r"\bON\s+DELETE\s+\w+", "", code_only, flags=re.IGNORECASE)
    # The trigger function's own error message MENTIONS the DELETE
    # operation by name (TG_OP) and the trigger definition names the
    # event ("BEFORE UPDATE OR DELETE ON interactions") -- both are the
    # invariant being enforced, not a violation of it. Strip those two
    # known, deliberate occurrences before asserting none remain.
    code_only = code_only.replace("BEFORE UPDATE OR DELETE ON interactions", "")
    code_only = re.sub(r"%\s*is not allowed[^']*", "", code_only)
    assert not re.search(r"\bDELETE\s+FROM\b", code_only, re.IGNORECASE), (
        "migration 034 contains a DELETE FROM statement"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_interactions_update_is_blocked_at_the_database(
    interaction_recorder, transcript, learner_id, clean_pool
):
    """Not just an AST scan -- migration 034's trigger must actually
    reject an UPDATE against a real row, at the database level,
    regardless of what application code attempts."""
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
        async with clean_pool.acquire() as conn:
            await conn.execute(
                "UPDATE interactions SET response_text = 'tampered' WHERE id = $1",
                interaction.id,
            )


@pytest.mark.asyncio(loop_scope="session")
async def test_interactions_delete_is_blocked_at_the_database(
    interaction_recorder, transcript, learner_id, clean_pool
):
    from probe.models import QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=False, response_text="a",
    )
    with pytest.raises(asyncpg.exceptions.RaiseError, match="immutable"):
        async with clean_pool.acquire() as conn:
            await conn.execute("DELETE FROM interactions WHERE id = $1", interaction.id)


@pytest.mark.asyncio(loop_scope="session")
async def test_interaction_options_is_deliberately_not_under_the_trigger(
    interaction_recorder, interaction_option_store, transcript, learner_id, clean_pool,
):
    """The one deliberate exception: was_selected/selection_timestamp
    are populated on a LATER turn than creation (a decision record, not
    an interaction record) -- this must keep working, not regress into
    accidentally being covered by a future, broader trigger."""
    from probe.disambiguate import DisambiguationStore
    from probe.models import DisambiguationBranch, Option, QuestionAuthor

    session_id = await transcript.create_session(learner_id)
    interaction = await interaction_recorder.record(
        learner_id=learner_id, session_id=session_id, turn_number=0,
        question_text="q", question_author=QuestionAuthor.LEARNER,
        originating_question=None, did_branch=True, response_text=None,
    )
    disamb = DisambiguationStore(clean_pool)
    d_turn = await disamb.create_turn(session_id, 0, needs_branches=True, turn_had_direct_answer=False)
    branches = await disamb.add_branches([
        DisambiguationBranch(disambiguation_turn_id=d_turn.id, session_id=session_id, turn_index=0, statement="s"),
    ])
    options = [Option(branch_id=branches[0].id, generation_id=d_turn.id, session_id=session_id, turn_index=0, text="t")]
    await disamb.create_options(options)

    from probe.models import InteractionOption

    io = InteractionOption(
        interaction_id=interaction.id, learner_id=learner_id, option_id=options[0].id,
        branch_id=branches[0].id, option_text="t", shown_position=0,
    )
    await interaction_option_store.create_many([io])
    marked = await interaction_option_store.mark_selected(options[0].id)
    assert marked.was_selected is True
