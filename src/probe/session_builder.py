"""The single place a fully-wired `SessionLoop` is assembled — called by
BOTH `cli.py` (`probe chat` / `probe consolidate-session`) and
`webserver.py` (`probe serve`).

WHY THIS MODULE EXISTS: it didn't, until 2026-09-15. Before this,
`cli.py` and `webserver.py` each built their own `SessionLoop`
independently. `cli.py`'s `_build_interaction_pipeline` wired
`claim_store`, `stated_preference_store`, `reference_binding_store`,
and the whole interaction/retrieval pipeline (`interaction_recorder`,
`retrieval_pool`, `history_block`'s dependencies) into the CLI path —
and its own docstring said, in writing, that this was "deliberately
NOT into `probe serve` ... until the hand-read described in this
feature's own review has actually happened there too." That hand-read
never happened. Every session run through `probe serve` — the actual
web UI, the one a browser hits — ran with NONE of it: no claims, no
stated preferences, no reference bindings, no history block, no
interaction pipeline at all. Confirmed by running a real session
through the real HTTP routes and reading `node_calls`/`turn_diagnostics`
back (see `tests/test_webserver_interaction_pipeline_wiring.py`): every
one of those blocks reached `FinalAnswer`'s prompt once wired in.

This is the fifth instance of the same failure class in this
codebase's history (see git log / CLAUDE.md's own incident-driven
invariants for the others — the hypothesis store running empty,
`load_dotenv` never being called, the Gemini schema silently
overriding the prompt): something built, verified in isolation, and
never wired into the path that actually runs. The fix here is
structural, not a one-time sync: `cli.py` and `webserver.py` both call
`build_session_loop` below. Neither constructs a `SessionLoop` any
other way. A future optional layer added to this function reaches both
entry points by construction — there is no second place to remember to
update, because there is no second place at all.
"""

from __future__ import annotations

from typing import Callable

import asyncpg

from probe.audit import NodeCallStore, TranscriptStore
from probe.claims import ClaimStore
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.domain_config import DomainConfig
from probe.embeddings import EmbeddingClient
from probe.interactions import (
    InteractionAbstractStore,
    InteractionOptionStore,
    InteractionRecorder,
    InteractionStore,
    PredictionStore,
    ReferenceBindingStore,
    StatedPreferenceStore,
    TurnOutcomeStore,
)
from probe.llm import ModelTierClients
from probe.loop import SessionLoop
from probe.memory import LearnerFactStore, ThinkingStyleStore
from probe.moss_client import MossService
from probe.retrieval_config import RetrievalConfig


def build_interaction_pipeline_stores(pool: asyncpg.Pool, embedding_client: EmbeddingClient) -> dict:
    """The interaction/retrieval pipeline (interactions.py, retrieval.py,
    history_block.py, claims.py, reference_bindings.py) — every
    `SessionLoop` kwarg that is additive on top of minimal_branch and
    gated on a store being present. `FinalAnswer`'s prompt reads
    `learner_history_block` (retrieval), `structural_requirement`
    (stated_preferences, when this learner has explicitly stated one),
    `reference_bindings_block` (an exact-match lookup of known recurring-
    phrase meanings), and `claim_constraints_block` (promoted claims) —
    see loop.py's own comments at each read site for exactly how each is
    built. `AssessAndBranch` also reads reference_bindings, to suppress
    branching on something already known. Predictions
    (interaction_nodes.LLMSelectionPredictor) still feed nothing back
    into what the learner sees."""
    interaction_store = InteractionStore(pool)
    turn_outcome_store = TurnOutcomeStore(pool)
    recorder = InteractionRecorder(
        interaction_store,
        turn_outcome_store,
        embedding_client,
        same_subject_threshold=RetrievalConfig().same_subject_threshold,
    )
    return {
        "interaction_recorder": recorder,
        "interaction_option_store": InteractionOptionStore(pool),
        "interaction_abstract_store": InteractionAbstractStore(pool),
        "turn_outcome_store": turn_outcome_store,
        "prediction_store": PredictionStore(pool),
        "retrieval_pool": pool,
        "stated_preference_store": StatedPreferenceStore(pool),
        "reference_binding_store": ReferenceBindingStore(pool),
        "claim_store": ClaimStore(pool),
    }


def build_session_loop(
    pool: asyncpg.Pool,
    tiers: ModelTierClients,
    embedding_client: EmbeddingClient,
    domain_config: DomainConfig | None = None,
    on_node_start: Callable[[str], None] | None = None,
    moss_service: MossService | None = None,
) -> SessionLoop:
    """The one and only place a `SessionLoop` gets constructed in this
    codebase's two entry points (`cli.py`, `webserver.py`) — every
    optional store wired in, unconditionally. `on_node_start` is the
    one param that genuinely differs by caller (webserver.py forwards
    node-progress to an SSE queue; the CLI has nothing to forward to),
    so it stays a parameter rather than being hidden away too."""
    return SessionLoop(
        transcript=TranscriptStore(pool),
        node_calls=NodeCallStore(pool),
        llm=tiers.fast,
        model_tier_clients=tiers,
        diagnostics_store=TurnDiagnosticsStore(pool),
        on_node_start=on_node_start,
        disambiguation_store=DisambiguationStore(pool),
        learner_fact_store=LearnerFactStore(pool),
        thinking_style_store=ThinkingStyleStore(pool),
        embedding_client=embedding_client,
        domain_config=domain_config,
        moss_service=moss_service,
        **build_interaction_pipeline_stores(pool, embedding_client),
    )
