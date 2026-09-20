"""The wrong-portrait comparison view `demo_fixture.py` deliberately
left unbuilt: same question, run once per fixture learner plus a THIRD
learner with zero claims (the control), so a difference between the two
portraits' answers can be told apart from a difference that would have
shown up anyway with no personalization at all.

WHY A THIRD COLUMN, NOT JUST TWO: two answers differing proves nothing
by itself — an LLM's sampling alone can make two calls on an identical
prompt read differently. The no-portrait baseline is what makes the
comparison a control rather than an anecdote: if the concrete and
abstract answers differ from EACH OTHER but neither differs from the
baseline, the portraits aren't doing anything and any apparent
difference between them is noise. A real effect looks like both
portraits pulling away from the baseline in the direction their own
claims point.

WHERE "THE CLAIMS THAT FIRED" AND "THE FROZEN PREDICTION" COME FROM:
this reuses loop.py's existing mechanism rather than building a new
one. `SessionLoop._handle_normal_turn` already renders every promoted
claim into `claim_constraints_block` (via `claims.render_claim_
constraint`) and passes it as a named kwarg into the `FinalAnswer`
node call — which CLAUDE.md invariant 2 already requires land in
`node_calls.input_json` verbatim, before the LLM call that consumes it
runs. So "the frozen prediction" is not new machinery: it is that same
`claim_constraints_block`, read back out of `node_calls` after the
turn instead of watched live — frozen in exactly the sense the demo
fixture's own claims are frozen (LOCKED, PROMOTED, not decaying), and
"with its scores" is each contributing claim's own confidence, read
straight off `ClaimStore.list_promoted_for_learner`, no recomputation.

A CONCRETE FINDING FROM BUILDING THIS: neither `cli.py` nor
`webserver.py` construct a `SessionLoop` with `claim_store` set for
the chat path — only the instrument routes (present/finalize) build a
`ClaimStore` at all, for a different purpose (writing capability
evidence). That means today, in the actual running system, a promoted
claim NEVER reaches a real chat turn's prompt — the mechanism loop.py
already has for it is wired to nothing. This module is the first
caller that wires `claim_store` into a `SessionLoop` for the chat path
at all; without that wiring, both fixture portraits would produce
identically un-personalized prompts and this comparison would test
nothing.

RETRIEVED HISTORY IS EXPECTED TO BE EMPTY: fixture learners have
claims but no `learner_facts` (demo_fixture.py never writes any, and
never runs a real session), so `memory_context` will read as empty for
all three columns on a first run. That is itself part of the control:
it rules out retrieval as the source of any difference the answers
show, leaving claims as the only remaining candidate explanation --
exactly the trace the comparison exists to make.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import asyncpg

from probe.audit import NodeCallStore, TranscriptStore
from probe.claims import ClaimStore
from probe.demo_fixture import DEMO_QUESTION_SET, seed_demo_fixture
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.embeddings import EmbeddingClient, StubEmbeddingClient
from probe.learner import LearnerStore
from probe.llm import LLMClient, StubLLMClient
from probe.loop import SessionLoop
from probe.memory import LearnerFactStore

NO_PORTRAIT_LABEL = "demo-portrait-none"


@dataclass
class ComparisonColumn:
    learner_label: str
    session_id: UUID
    claims_fired: list[dict]
    frozen_prediction: str
    retrieved_history: str
    answer: str


async def ensure_no_portrait_learner(pool: asyncpg.Pool):
    """The control: idempotent like the two fixture portraits, but
    deliberately never seeded with any claim. Never call `ClaimStore.
    create` for this learner -- the whole point is zero claims."""
    learners = LearnerStore(pool)
    existing = await learners.get_by_label(NO_PORTRAIT_LABEL)
    return existing if existing is not None else await learners.create(label=NO_PORTRAIT_LABEL)


def _build_comparison_loop(pool: asyncpg.Pool, llm: LLMClient, embedding_client: EmbeddingClient) -> SessionLoop:
    """The one place `claim_store` is wired into a `SessionLoop` for
    the CHAT path (see this module's own docstring) -- deliberately
    minimal beyond that: only the stores this view's four fields
    actually need (transcript/node_calls for the audit trail,
    claim_store for claim_constraints_block, learner_fact_store+
    embedding_client so retrieved_history reflects a real, if empty,
    search rather than being hardwired off)."""
    return SessionLoop(
        transcript=TranscriptStore(pool),
        node_calls=NodeCallStore(pool),
        llm=llm,
        diagnostics_store=TurnDiagnosticsStore(pool),
        disambiguation_store=DisambiguationStore(pool),
        learner_fact_store=LearnerFactStore(pool),
        embedding_client=embedding_client,
        claim_store=ClaimStore(pool),
    )


async def _run_one_column(
    pool: asyncpg.Pool, loop: SessionLoop, claim_store: ClaimStore, learner_label: str, learner_id: UUID, question: str,
) -> ComparisonColumn:
    transcript = TranscriptStore(pool)
    node_calls = NodeCallStore(pool)
    session_id = await transcript.create_session(learner_id)

    answer = await loop.handle_turn(session_id, 0, question)

    promoted = await claim_store.list_promoted_for_learner(learner_id)
    claims_fired = [
        {"value": c.value.value, "confidence": c.confidence}
        for c in promoted
    ]

    call = await node_calls.get_call_for_turn(session_id, 0, "FinalAnswer")
    frozen_prediction = ""
    retrieved_history = ""
    if call is not None:
        frozen_prediction = str(call.input_json.get("claim_constraints_block") or "")
        retrieved_history = str(call.input_json.get("memory_context") or "")

    return ComparisonColumn(
        learner_label=learner_label, session_id=session_id, claims_fired=claims_fired,
        frozen_prediction=frozen_prediction, retrieved_history=retrieved_history, answer=answer,
    )


async def run_comparison(
    pool: asyncpg.Pool, question: str, stub: bool = False, llm: LLMClient | None = None,
) -> dict[str, ComparisonColumn]:
    """Seeds the two fixture portraits plus the no-portrait control
    (all idempotent), then runs the SAME question through one fresh
    turn-0 session per learner, returning the three columns keyed
    "concrete"/"abstract"/"none". `stub` (or a passed-in `llm`) lets a
    test exercise the full wiring with no API key and no cost -- the
    caveat that implies (a stub can't read the prompt, so all three
    `answer` fields will be textually identical under it) is exactly
    the line this module's docstring draws between what a stub proves
    (the DATA wiring is correct) and what only a real LLM call can show
    (whether the answers actually differ)."""
    portraits = await seed_demo_fixture(pool)
    no_portrait = await ensure_no_portrait_learner(pool)

    resolved_llm = llm if llm is not None else (StubLLMClient() if stub else None)
    if resolved_llm is None:
        raise ValueError("run_comparison needs either stub=True or an explicit llm=")
    embedding_client = StubEmbeddingClient()
    loop = _build_comparison_loop(pool, resolved_llm, embedding_client)
    claim_store = ClaimStore(pool)

    columns: dict[str, ComparisonColumn] = {}
    for key, learner in (
        ("concrete", portraits["concrete"]),
        ("abstract", portraits["abstract"]),
        ("none", no_portrait),
    ):
        columns[key] = await _run_one_column(pool, loop, claim_store, key, learner.id, question)
    return columns


def format_comparison(question: str, columns: dict[str, ComparisonColumn]) -> str:
    lines = [f"QUESTION: {question}", "=" * 72]
    for key in ("concrete", "abstract", "none"):
        col = columns[key]
        lines.append(f"\n--- {key.upper()} ({col.learner_label}, session {col.session_id}) ---")
        if col.claims_fired:
            lines.append("claims fired:")
            for c in col.claims_fired:
                lines.append(f"  {c['value']} (confidence={c['confidence']:.2f})")
        else:
            lines.append("claims fired: (none)")
        lines.append(f"retrieved history: {col.retrieved_history or '(empty)'}")
        lines.append(f"frozen prediction: {col.frozen_prediction or '(empty)'}")
        lines.append(f"answer: {col.answer}")
    concrete_vs_none = columns["concrete"].answer != columns["none"].answer
    abstract_vs_none = columns["abstract"].answer != columns["none"].answer
    concrete_vs_abstract = columns["concrete"].answer != columns["abstract"].answer
    lines.append("\n--- control check ---")
    lines.append(f"concrete answer differs from no-portrait baseline: {concrete_vs_none}")
    lines.append(f"abstract answer differs from no-portrait baseline: {abstract_vs_none}")
    lines.append(f"concrete answer differs from abstract answer: {concrete_vs_abstract}")
    if concrete_vs_abstract and not (concrete_vs_none or abstract_vs_none):
        lines.append(
            "WARNING: portraits differ from each other but neither differs from the "
            "baseline -- this is the exact failure mode the third column exists to catch."
        )
    return "\n".join(lines)


async def run_comparison_for_fixed_questions(
    pool: asyncpg.Pool, stub: bool = False, llm: LLMClient | None = None,
) -> str:
    """Runs every question in `demo_fixture.DEMO_QUESTION_SET`, one
    three-column comparison each, concatenated into one report."""
    reports = []
    for question in DEMO_QUESTION_SET:
        columns = await run_comparison(pool, question, stub=stub, llm=llm)
        reports.append(format_comparison(question, columns))
    return "\n\n".join(reports)
