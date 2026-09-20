import os
from pathlib import Path

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from probe.audit import NodeCallStore, TranscriptStore
from probe.db import create_pool
from probe.diagnostics import TurnDiagnosticsStore
from probe.disambiguate import DisambiguationStore
from probe.embeddings import StubEmbeddingClient
from probe.learner import LearnerStore
from probe.memory import LearnerFactStore, ThinkingStyleStore

# Root cause of two real data-loss incidents in this project's history:
# `.env` has defined a genuinely separate PROBE_TEST_DATABASE_URL
# (pointing at `probe_test`, not the dev database `probe`) the whole
# time, but nothing here ever called load_dotenv() -- os.getenv() was
# silently reading an unset environment variable and falling through
# to the SAME hardcoded default DATABASE_URL also falls back to,
# meaning the "isolated" test database and the dev database were the
# same physical database whenever neither var was exported into the
# shell by hand. Loading .env here is the actual fix; the fallback
# below now only matters if .env itself is missing.
load_dotenv()
DATABASE_URL = os.getenv(
    "PROBE_TEST_DATABASE_URL",
    "postgresql://probe:probe@localhost:5434/probe",
)

MIGRATIONS_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "probe" / "migrations"
)
MIGRATIONS = sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pool():
    pool = await create_pool(DATABASE_URL, min_size=1, max_size=4)
    async with pool.acquire() as conn:
        # Fresh schema for the test session. DROP is DDL cleanup here in
        # tests only — the stores themselves must never remove rows
        # (CLAUDE.md invariants). The list is deliberately still
        # comprehensive (includes the tables migration 032 retires) so
        # a run against a pre-032 schema is cleaned too; every DROP is
        # IF EXISTS.
        await conn.execute("DROP TABLE IF EXISTS predictions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS instrument_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS instruments CASCADE")
        await conn.execute("DROP TABLE IF EXISTS interaction_contracts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS capability_evidence CASCADE")
        await conn.execute("DROP TABLE IF EXISTS capability_claims CASCADE")
        await conn.execute("DROP TABLE IF EXISTS claim_statements CASCADE")
        await conn.execute("DROP TABLE IF EXISTS claim_evidence CASCADE")
        await conn.execute("DROP TABLE IF EXISTS claims CASCADE")
        await conn.execute("DROP TABLE IF EXISTS reference_bindings CASCADE")
        await conn.execute("DROP TABLE IF EXISTS stated_preferences CASCADE")
        await conn.execute("DROP TABLE IF EXISTS turn_outcomes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS interaction_abstracts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS interaction_options CASCADE")
        await conn.execute("DROP TABLE IF EXISTS interactions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS population_patterns CASCADE")
        await conn.execute("DROP TABLE IF EXISTS topics CASCADE")  # pre-removal schema cleanup
        await conn.execute("DROP TYPE IF EXISTS interaction_turn_outcome")
        await conn.execute("DROP TYPE IF EXISTS interaction_help_level")
        await conn.execute("DROP TYPE IF EXISTS interaction_prior_outcome")
        await conn.execute("DROP TYPE IF EXISTS interaction_entry_state")
        await conn.execute("DROP TYPE IF EXISTS interaction_question_author")
        await conn.execute("DROP TYPE IF EXISTS stated_preference_label")
        await conn.execute("DROP TABLE IF EXISTS evidence_records CASCADE")
        await conn.execute("DROP TABLE IF EXISTS node_calls CASCADE")
        await conn.execute("DROP TABLE IF EXISTS turn_diagnostics CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypothesis_tier_changes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learner_facts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS thinking_style_candidates CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_options CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_branches CASCADE")
        await conn.execute("DROP TABLE IF EXISTS disambiguation_turns CASCADE")
        await conn.execute("DROP TABLE IF EXISTS options CASCADE")
        await conn.execute("DROP TABLE IF EXISTS branches CASCADE")
        await conn.execute("DROP TABLE IF EXISTS branch_generations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS world_model_revision_evidence CASCADE")
        await conn.execute("DROP TABLE IF EXISTS world_model_revisions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypothesis_concepts CASCADE")
        await conn.execute("DROP TABLE IF EXISTS evidence_refs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS turns CASCADE")
        await conn.execute("DROP TABLE IF EXISTS sessions CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learners CASCADE")
        await conn.execute("DROP TABLE IF EXISTS hypotheses CASCADE")
        await conn.execute("DROP TABLE IF EXISTS learner_overlay CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_prerequisites CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_nodes CASCADE")
        await conn.execute("DROP TABLE IF EXISTS concept_graphs CASCADE")
        await conn.execute("DROP TYPE IF EXISTS evidence_source_type")
        await conn.execute("DROP TYPE IF EXISTS learner_fact_type")
        await conn.execute("DROP TYPE IF EXISTS thinking_style_status")
        await conn.execute("DROP TYPE IF EXISTS option_status")
        await conn.execute("DROP TYPE IF EXISTS branch_status")
        await conn.execute("DROP TYPE IF EXISTS revision_status")
        await conn.execute("DROP TYPE IF EXISTS overlay_state")
        await conn.execute("DROP TYPE IF EXISTS evidence_polarity")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_tier")
        await conn.execute("DROP TYPE IF EXISTS hypothesis_layer")
        for migration in MIGRATIONS:
            await conn.execute(migration.read_text())
        # On a genuinely first-ever bootstrap (extension didn't exist
        # yet when this exact connection was created — see db.py's
        # _init_connection), the vector codec silently failed to
        # register at connection-init time. The migrations just replayed
        # are guaranteed to have created the extension, so register it
        # now, before this connection goes back to the pool.
        from pgvector.asyncpg import register_vector

        await register_vector(conn)
    try:
        yield pool
    finally:
        await pool.close()


@pytest_asyncio.fixture(loop_scope="session")
async def clean_pool(pool):
    async with pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE evidence_records, node_calls, turn_diagnostics, turns, "
            "sessions, learners, disambiguation_options, disambiguation_branches, "
            "disambiguation_turns, learner_facts, thinking_style_candidates, "
            "predictions, instrument_events, instruments, interaction_contracts, "
            "capability_evidence, capability_claims, "
            "claim_statements, claim_evidence, claims, stated_preferences, reference_bindings, "
            "turn_outcomes, "
            "interaction_abstracts, "
            "interaction_options, interactions, population_patterns "
            "RESTART IDENTITY CASCADE"
        )
    return pool


@pytest_asyncio.fixture(loop_scope="session")
async def transcript(clean_pool):
    return TranscriptStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def evidence_store(clean_pool):
    from probe.evidence import EvidenceStore

    return EvidenceStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def node_calls(clean_pool):
    return NodeCallStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def diagnostics_store(clean_pool):
    return TurnDiagnosticsStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def disambiguation_store(clean_pool):
    return DisambiguationStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def learner_fact_store(clean_pool):
    return LearnerFactStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def thinking_style_store(clean_pool):
    return ThinkingStyleStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def interaction_store(clean_pool):
    from probe.interactions import InteractionStore

    return InteractionStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def interaction_option_store(clean_pool):
    from probe.interactions import InteractionOptionStore

    return InteractionOptionStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def interaction_abstract_store(clean_pool):
    from probe.interactions import InteractionAbstractStore

    return InteractionAbstractStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def turn_outcome_store(clean_pool):
    from probe.interactions import TurnOutcomeStore

    return TurnOutcomeStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def prediction_store(clean_pool):
    from probe.interactions import PredictionStore

    return PredictionStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def stated_preference_store(clean_pool):
    from probe.interactions import StatedPreferenceStore

    return StatedPreferenceStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def reference_binding_store(clean_pool):
    from probe.interactions import ReferenceBindingStore

    return ReferenceBindingStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def claim_store(clean_pool):
    from probe.claims import ClaimStore

    return ClaimStore(clean_pool)


@pytest_asyncio.fixture
def interaction_recorder(interaction_store, turn_outcome_store, embedding_client):
    from probe.interactions import InteractionRecorder
    from probe.retrieval_config import RetrievalConfig

    return InteractionRecorder(
        interaction_store,
        turn_outcome_store,
        embedding_client,
        same_subject_threshold=RetrievalConfig().same_subject_threshold,
    )


@pytest.fixture
def embedding_client():
    """A fresh StubEmbeddingClient per test — holds no shared state
    worth reusing across tests (just a `canned` dict and a `texts`
    log)."""
    return StubEmbeddingClient()


@pytest_asyncio.fixture(loop_scope="session")
async def learner_store(clean_pool):
    return LearnerStore(clean_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def learner_id(learner_store):
    """A fresh learner per test, for tests that just need *a* valid
    learner_id to satisfy sessions.learner_id's FK and don't care about
    learner identity itself."""
    learner = await learner_store.create()
    return learner.id
