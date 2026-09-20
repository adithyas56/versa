from __future__ import annotations

import argparse
import asyncio
import os
import sys
from uuid import UUID

from dotenv import load_dotenv

from probe.db import create_pool
from probe.domain_config import load_domain_config
from probe.embeddings import (
    EmbeddingClient,
    StubEmbeddingClient,
    build_embedding_client,
)
from probe.learner import LearnerStore
from probe.llm import ModelTierClients, StubLLMClient, build_tier_clients
from probe.loop import SessionLoop
from probe.interactions import InteractionAbstractStore
from probe.models import Learner
from probe.session_builder import build_session_loop
from probe.population_patterns import (
    PopulationAggregationConfig,
    PopulationPatternStore,
    aggregate_population_patterns,
)
from probe import migrate as _migrate


def _database_url() -> str:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set (check .env)", file=sys.stderr)
        sys.exit(2)
    return url


def _require_gemini_api_key() -> str:
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print(
            "error: GEMINI_API_KEY not set (check .env) — pass --stub to "
            "run against StubLLMClient instead of the real Gemini API",
            file=sys.stderr,
        )
        sys.exit(2)
    return key


def _build_tier_clients(use_stub: bool) -> ModelTierClients:
    if use_stub:
        stub = StubLLMClient()
        return ModelTierClients(fast=stub, capable=stub, best=stub)
    return build_tier_clients(_require_gemini_api_key())


def _build_embedding_client(use_stub: bool) -> EmbeddingClient:
    if use_stub:
        return StubEmbeddingClient()
    return build_embedding_client(_require_gemini_api_key())


async def _resolve_learner(store: LearnerStore, spec: str) -> Learner:
    """--learner accepts either an existing learner's UUID or a label.

    A UUID must already exist (there's no "create by guessing an id").
    A label resumes the matching learner if one exists, else creates a
    new one — this is the session's identity, resolved once here, not
    per-turn state.
    """
    try:
        learner_id = UUID(spec)
    except ValueError:
        learner_id = None
    if learner_id is not None:
        learner = await store.get(learner_id)
        if learner is None:
            print(f"error: no learner with id {spec}", file=sys.stderr)
            sys.exit(2)
        return learner

    learner = await store.get_by_label(spec)
    if learner is not None:
        return learner
    return await store.create(label=spec)


def _build_loop(
    pool, tiers: ModelTierClients, embedding_client: EmbeddingClient, domain_config
) -> SessionLoop:
    """Thin wrapper over `session_builder.build_session_loop` -- the one
    shared assembly point `probe chat`/`probe consolidate-session` and
    `probe serve` (webserver.py) both call, so the two entry points
    cannot silently diverge in which optional stores they wire in the
    way they did before 2026-09-15 (see session_builder.py's own
    docstring for that incident)."""
    return build_session_loop(pool, tiers, embedding_client, domain_config=domain_config)


async def _chat(learner_spec: str, use_stub: bool) -> None:
    tiers = _build_tier_clients(use_stub)
    embedding_client = _build_embedding_client(use_stub)
    # The ONE place PROBE_DOMAIN is read for the CLI path (see
    # domain_config.load_domain_config's own docstring) -- resolved
    # once here, passed down as a plain DomainConfig object; nothing
    # past this point reads the environment variable again.
    domain_config = load_domain_config()
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        learner = await _resolve_learner(LearnerStore(pool), learner_spec)
        label_suffix = f" (label={learner.label!r})" if learner.label else ""
        print(f"probe: learner {learner.id}{label_suffix}")
        print("probe: minimal_branch mode — no concept graph")
        print(f"probe: domain = {domain_config.domain.value}")
        print(
            "probe: interaction/retrieval pipeline ON (dry run -- writes "
            "interactions/topics/outcomes/abstracts/predictions; nothing "
            "it produces reaches this session's own responses yet)"
        )
        loop = _build_loop(pool, tiers, embedding_client, domain_config)
        await loop.run_interactive(learner.id)
    finally:
        await pool.close()


async def _consolidate_session(session_id_str: str, use_stub: bool) -> None:
    try:
        session_id = UUID(session_id_str)
    except ValueError:
        print(f"error: {session_id_str!r} is not a valid session id", file=sys.stderr)
        sys.exit(2)

    tiers = _build_tier_clients(use_stub)
    embedding_client = _build_embedding_client(use_stub)
    domain_config = load_domain_config()
    pool = await create_pool(_database_url(), min_size=1, max_size=4)
    try:
        loop = _build_loop(pool, tiers, embedding_client, domain_config)
        # Deliberate, unambiguous trigger — no turn-count gate (unlike
        # run_interactive's own auto-consolidate on exit): this command
        # exists specifically to consolidate a session on demand,
        # regardless of how many turns it has.
        result = await loop.consolidate_session(session_id)
        if result is None:
            print(
                "probe: nothing to consolidate — this session wrote no "
                "learner_facts (a BASELINE session, or it never resolved "
                "anything)"
            )
            return
        print(
            f"probe: thinking-style candidate {result.id}\n"
            f"  path_summary: {result.path_summary}\n"
            f"  confirmation_count={result.confirmation_count} "
            f"status={result.status.value}\n"
            f"  session_ids: {[str(s) for s in result.session_ids]}"
        )
    finally:
        await pool.close()


async def _aggregate_patterns() -> None:
    """`probe aggregate-patterns` — the on-demand batch step of
    population_patterns.py's pipeline (raw interaction -> abstract form
    -> aggregation -> multi-learner support -> retrieval). Deliberately
    explicit and on-demand, same "no automatic per-turn trigger"
    precedent as `probe consolidate-session`: clustering every
    learner's abstracts against every other learner's is a global
    operation, not something to redo on every turn.
    """
    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        abstract_store = InteractionAbstractStore(pool)
        pattern_store = PopulationPatternStore(pool)
        written = await aggregate_population_patterns(
            abstract_store, pattern_store, PopulationAggregationConfig()
        )
        if not written:
            print(
                "probe: no readable population patterns this run -- every "
                "cluster fell short of >=20 distinct learners or exceeded "
                "the 25% single-learner share cap"
            )
            return
        print(f"probe: wrote {len(written)} readable population pattern(s):")
        for pattern in written:
            print(
                f"  {pattern.id}: {pattern.abstract_form!r} "
                f"(support={pattern.support_count}, "
                f"distinct_learners={pattern.distinct_learner_count}, "
                f"max_share={pattern.max_per_learner_share:.2f})"
            )
    finally:
        await pool.close()


async def _run_migrations(status_only: bool, do_baseline: bool) -> None:
    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            if status_only:
                applied, pending = await _migrate.status(conn)
                print(f"migrations: {len(applied)} applied, {len(pending)} pending")
                for name in applied:
                    print(f"  [x] {name}")
                for name in pending:
                    print(f"  [ ] {name}")
                return
            if do_baseline:
                stamped = await _migrate.baseline(conn)
                if stamped:
                    print(
                        f"probe migrate: recorded {len(stamped)} migration(s) as "
                        f"already-applied without running them "
                        f"({stamped[0]} .. {stamped[-1]})"
                    )
                else:
                    print("probe migrate: nothing to baseline - ledger already complete")
                return
            applied = await _migrate.apply_all(
                conn, on_apply=lambda name: print(f"  applied {name}")
            )
            if applied:
                print(f"probe migrate: applied {len(applied)} migration(s)")
            else:
                print("probe migrate: database already up to date")
    finally:
        await pool.close()


async def _review_claims(learner_spec: str) -> None:
    """`probe review-claims` -- the claim layer's review surface (per
    its own spec: "list every claim with its statement, test, status,
    confidence, evidence count, topic spread, and links to the
    interactions behind it"). Read-only; makes no LLM call and writes
    nothing.

    Shows the CURRENT (latest `claim_statements` row) statement, not
    `Claim.statement` directly -- see `claims.maybe_restate_claims`'s
    own docstring for why those can differ. When they differ, both are
    printed (current, then the original founding wording) so the
    drift is visible without a separate query. Any evidence row
    carrying a `provenance_note` (a human-diagnosed harness bug) shows
    it inline, so a reader doesn't have to remember which row was
    flagged."""
    from probe.claims import ClaimStore

    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        learner = await _resolve_learner(LearnerStore(pool), learner_spec)
        store = ClaimStore(pool)
        claims = await store.list_for_learner(learner.id)
        if not claims:
            print(f"probe: no claims for learner {learner.id}")
            return
        for claim in claims:
            evidence = await store.list_evidence(claim.id)
            topics = sorted({e.topic for e in evidence})
            current_statement = await store.get_current_statement(claim.id)
            print(f"\n[{claim.status.value}] {claim.id}")
            if current_statement is not None and current_statement.statement != claim.statement:
                print(f"  statement (current): {current_statement.statement}")
                print(f"  statement (founding): {claim.statement}")
            else:
                print(f"  statement: {claim.statement}")
            print(f"  test: {claim.test}")
            print(f"  value: {claim.value.value}   confidence: {claim.confidence:.3f}")
            print(f"  source: {claim.source.value}   write_policy: {claim.write_policy.value}")
            print(f"  evidence: {len(evidence)} row(s) across {len(topics)} topic(s): {topics}")
            for e in evidence:
                note = f"  [FLAGGED: {e.provenance_note}]" if e.provenance_note else ""
                print(
                    f"    - {e.direction.value} | topic={e.topic!r} | "
                    f"interaction={e.interaction_id} | session={e.session_id} | "
                    f"test_fired={e.test_fired} contradiction_possible="
                    f"{e.contradiction_was_possible}{note}"
                )
    finally:
        await pool.close()


async def _score_predictions(exclude_contaminated: bool, split_by_source: bool) -> None:
    """`probe score-predictions` -- the reliability-diagram check
    score_predictions.py exists for: scores every claim's evidence
    history against itself (see that module's own docstring for why
    that's not circular) and prints observed-vs-predicted hit rate per
    confidence bucket, plus an overall Brier score. Read-only, no LLM
    call, pools across every learner (a calibration question about the
    confidence formula, not about any one learner).

    `--exclude-contaminated` prints BOTH pictures -- the full one
    (identical to production confidence, contaminated rows included)
    and one with every human-flagged (`provenance_note`) row dropped
    entirely -- so a real miscalibration can be told apart from a known
    harness bug's effect on the curve, without ever hiding the
    contaminated rows from production itself.

    `--split-by-source` prints three curves separately: click (from the
    preference side, `ClaimStore`), then locate and predict individually
    from the CAPABILITY side (`CapabilityClaimStore`, filtered by
    `skill` — both are PERFORMANCE-measuring, so neither one's evidence
    lands in `ClaimStore` at all; see capability.py's own module
    docstring for the incident that made this the correct split). The
    check instruments.py's own module docstring names as the gate a
    new evidence-producing METHOD has to clear before another one gets
    built: if a method's evidence is systematically overconfident
    relative to click evidence, its contract is being written or
    interpreted loosely."""
    from probe.capability import CapabilityClaimStore
    from probe.claims import ClaimStore
    from probe.models import CapabilityLabel, EvidenceSource
    from probe.score_predictions import (
        format_reliability_diagram,
        score_capability_predictions_for_all_learners,
        score_predictions_for_all_learners,
    )

    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        store = ClaimStore(pool)
        bins, overall_brier, total_trials = await score_predictions_for_all_learners(store)
        print("ALL EVIDENCE (production picture):")
        print(format_reliability_diagram(bins, overall_brier, total_trials))
        if exclude_contaminated:
            clean_bins, clean_brier, clean_total = await score_predictions_for_all_learners(
                store, exclude_contaminated=True
            )
            print("\nEXCLUDING FLAGGED (provenance_note) ROWS:")
            print(format_reliability_diagram(clean_bins, clean_brier, clean_total))
        if split_by_source:
            capability_store = CapabilityClaimStore(pool)
            click_bins, click_brier, click_total = await score_predictions_for_all_learners(
                store, source_filter=EvidenceSource.CLICK
            )
            print("\nCLICK EVIDENCE ONLY:")
            print(format_reliability_diagram(click_bins, click_brier, click_total))
            for label, skill in (
                ("LOCATE", CapabilityLabel.TRACES_WORKED_STEPS),
                ("PREDICT", CapabilityLabel.DERIVES_FORWARD),
            ):
                s_bins, s_brier, s_total = await score_capability_predictions_for_all_learners(
                    capability_store, skill_filter=skill
                )
                print(f"\n{label} EVIDENCE ONLY (capability):")
                print(format_reliability_diagram(s_bins, s_brier, s_total))
    finally:
        await pool.close()


async def _seed_demo_fixture() -> None:
    """`probe seed-demo-fixture` -- (re)applies demo_fixture.py's two
    hand-authored, opposite-portrait learners. Idempotent; no LLM call,
    no embedding call. See that module's own docstring for what it
    does and does not build."""
    from probe.demo_fixture import DEMO_QUESTION_SET, seed_demo_fixture

    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        learners = await seed_demo_fixture(pool)
        print("probe: demo fixture seeded (idempotent -- safe to re-run):")
        for role, learner in learners.items():
            print(f"  {role}: {learner.label} ({learner.id})")
        print(f"\nfixed question set ({len(DEMO_QUESTION_SET)} questions):")
        for q in DEMO_QUESTION_SET:
            print(f"  - {q}")
    finally:
        await pool.close()


async def _compare_portraits(question: str | None, use_stub: bool) -> None:
    """`probe compare-portraits` -- comparison.py's three-column wrong-
    portrait control: same question, the two fixture portraits plus a
    zero-claims control, side by side. Seeds/reuses the fixture and the
    control learner (idempotent). Without --stub this makes a real
    Gemini call per column (3 calls per question) -- the only way to
    see whether the answers actually differ, not just whether the
    claim-derived prompt does (see comparison.py's own docstring on
    what a stub run can and cannot prove)."""
    from probe.comparison import format_comparison, run_comparison
    from probe.demo_fixture import DEMO_QUESTION_SET

    tiers = _build_tier_clients(use_stub)
    pool = await create_pool(_database_url(), min_size=1, max_size=2)
    try:
        questions = [question] if question else DEMO_QUESTION_SET
        for q in questions:
            columns = await run_comparison(pool, q, llm=tiers.best)
            print(format_comparison(q, columns))
            print()
    finally:
        await pool.close()


def _serve(host: str, port: int) -> None:
    """`probe serve` — the calm single-page UI (probe/static/) backed by
    the Starlette API in webserver.py, over the same SessionLoop the CLI
    drives. Imported lazily so `probe chat` never pulls in Starlette.
    This is the only web UI; the old Streamlit `probe web` was removed."""
    from probe.webserver import serve

    serve(host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe")
    subparsers = parser.add_subparsers(dest="command")
    chat_parser = subparsers.add_parser(
        "chat", help="start an interactive minimal_branch session loop"
    )
    chat_parser.add_argument(
        "--learner",
        required=True,
        help="learner label (resumes if it exists, creates if not) "
        "or an existing learner's UUID",
    )
    chat_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient/StubEmbeddingClient instead of the "
        "real Gemini API (no GEMINI_API_KEY needed, no cost)",
    )
    consolidate_parser = subparsers.add_parser(
        "consolidate-session",
        help="background step 6-8 of the memory layer (memory.py) for "
        "one session on demand: label its facts' order-structure and "
        "compare against this learner's thinking_style_candidates",
    )
    consolidate_parser.add_argument("session_id", help="session id (UUID) to consolidate")
    consolidate_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient/StubEmbeddingClient instead of the "
        "real Gemini API (no GEMINI_API_KEY needed, no cost)",
    )
    subparsers.add_parser(
        "aggregate-patterns",
        help="cluster every learner's latest interaction_abstracts row "
        "and write readable (>=20 distinct learners, <=25% single-"
        "learner share) clusters to population_patterns -- the "
        "on-demand aggregation step retrieval reads from",
    )
    subparsers.add_parser(
        "seed-demo-fixture",
        help="(re)apply demo_fixture.py's two hand-authored, opposite-"
        "portrait learners plus a fixed question set -- idempotent, "
        "no LLM/embedding call, for a reproducible comparison demo",
    )
    compare_parser = subparsers.add_parser(
        "compare-portraits",
        help="three-column wrong-portrait comparison (comparison.py): "
        "same question run for the concrete portrait, the abstract "
        "portrait, and a zero-claims control, showing which claims "
        "fired, the frozen claim_constraints_block each one produced, "
        "and the resulting answer -- seeds the demo fixture if needed",
    )
    compare_parser.add_argument(
        "--question", default=None,
        help="one question to run (default: every question in "
        "demo_fixture.DEMO_QUESTION_SET)",
    )
    compare_parser.add_argument(
        "--stub",
        action="store_true",
        help="use StubLLMClient instead of the real Gemini API -- proves "
        "the claims/prediction wiring is correct, but under a stub all "
        "three answers are identical by construction (a stub cannot "
        "read the prompt), so this cannot show whether the answers "
        "actually differ",
    )
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="apply pending SQL migrations to DATABASE_URL, in order, "
        "once each (idempotent; tracked in a schema_migrations table)",
    )
    migrate_parser.add_argument(
        "--status",
        action="store_true",
        help="show applied/pending migrations and exit without changing anything",
    )
    migrate_parser.add_argument(
        "--baseline",
        action="store_true",
        help="record every migration as already-applied WITHOUT running "
        "it - for a database that already has the full schema but no "
        "schema_migrations ledger (e.g. a hand-migrated dev DB)",
    )
    review_claims_parser = subparsers.add_parser(
        "review-claims",
        help="read-only listing of one learner's claims (claims.py): "
        "statement, test, status, confidence, evidence count, topic "
        "spread, and the interactions each evidence row points at",
    )
    review_claims_parser.add_argument(
        "--learner", required=True,
        help="learner label or an existing learner's UUID",
    )
    score_predictions_parser = subparsers.add_parser(
        "score-predictions",
        help="read-only reliability-diagram check (score_predictions.py): "
        "scores every claim's evidence history against itself and "
        "prints observed-vs-predicted hit rate per confidence bucket "
        "plus an overall Brier score -- the check clamp_confidence_for_"
        "decisions' [0.3, 0.7] clamp is waiting on before it can move",
    )
    score_predictions_parser.add_argument(
        "--exclude-contaminated", action="store_true",
        help="also print the reliability diagram with every human-flagged "
        "(provenance_note) evidence row excluded, alongside the normal "
        "(production) picture",
    )
    score_predictions_parser.add_argument(
        "--split-by-source", action="store_true",
        help="also print the click-derived and instrument-derived reliability "
        "diagrams separately (instruments.py) -- the gate a new instrument "
        "has to clear before another one gets built",
    )
    serve_parser = subparsers.add_parser(
        "serve",
        help="launch the calm single-page web UI (Starlette API + probe/static/)",
    )
    # Defaults are deployment-first: bind 0.0.0.0 and take the port from
    # $PORT (Cloud Run / any PaaS injects it) so the container works with
    # no extra flags. Locally it's still reachable at localhost:8000;
    # pass --host 127.0.0.1 to restrict it.
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8000"))
    )
    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(_chat(args.learner, args.stub))
    elif args.command == "consolidate-session":
        asyncio.run(_consolidate_session(args.session_id, args.stub))
    elif args.command == "aggregate-patterns":
        asyncio.run(_aggregate_patterns())
    elif args.command == "seed-demo-fixture":
        asyncio.run(_seed_demo_fixture())
    elif args.command == "compare-portraits":
        asyncio.run(_compare_portraits(args.question, args.stub))
    elif args.command == "review-claims":
        asyncio.run(_review_claims(args.learner))
    elif args.command == "score-predictions":
        asyncio.run(_score_predictions(args.exclude_contaminated, args.split_by_source))
    elif args.command == "migrate":
        asyncio.run(_run_migrations(args.status, args.baseline))
    elif args.command == "serve":
        _serve(args.host, args.port)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
