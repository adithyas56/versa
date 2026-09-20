# Versa — Architecture

Versa is the product; `probe` is the Python package. This document covers
the system architecture with an emphasis on the Moss retrieval layer added
for the hackathon. For the reasoning-mode history (why the concept graph and
planner were retired), see [README.md](README.md).

## The three systems, and the one rule that separates them

```text
PostgreSQL   = durable system of record   (interactions, evidence, claims, learner_facts, audit)
Moss         = hot-path semantic retrieval (compact learner-memory documents, loaded locally)
Gemini       = language reasoning          (response generation; owns no durable learner truth)
Versa/probe  = orchestration + UX          (disambiguation, evidence rules, personalization)
```

Postgres is the source of truth. A Moss document is a *retrieval
projection* of a durable row, never the row itself. Gemini output is a
generated response, never promoted to a durable fact except through Versa's
existing evidence rules.

## The live tutoring turn

```text
Student message
      │
      ▼
EmbedAndSearchFacts + ConfirmFactMatch ──(known ambiguity)──► skip branching
      │ (not resolved by memory)
      ▼
AssessAndBranch ──(ambiguous)──► GenerateOptions ──► student clicks / types
      │ (unambiguous, or after a click)
      ▼
_run_final_answer
      │   builds, each failure-isolated:
      │     • structural_requirement (stated preference)
      │     • claim_constraints_block (promoted claims)
      │     • reference_bindings_block
      │     • learner_history_block  (pgvector over interactions)
      │     • moss_memory_block      ◄── Moss learner-memory retrieval (this layer)
      ▼
FinalAnswer (Gemini best tier) ──► response shown to student
      │
      ▼
WriteLearnerFact ──► learner_facts (Postgres)
      │
      ├─► SessionMemoryCache.add(doc)        (synchronous: immediate continuity, §17)
      └─► MossService.sync_documents([doc])  (background: off the response path, §16.1)
```

The Moss read happens inside `SessionLoop._run_final_answer`, alongside the
other pre-rendered context blocks, and its output is injected into
`FinalAnswer`'s prompt as `moss_memory_block`. This is the single point
where Moss is on the live path — there is no second learner-memory retrieval
implementation.

## The Moss retrieval layer (modules)

| Module | Responsibility |
|---|---|
| `memory_index.py` | Pure projection: `LearnerFact` → `MemoryDocument` (natural-language text + string metadata) and → the Moss SDK's `DocumentInfo`. |
| `moss_client.py` | Client lifecycle + config (`MossConfig`), two engines behind one `MossEngine` interface (`RealMossEngine` wrapping the `moss` SDK; `StubMossEngine`, in-process), and the long-lived `MossService`. |
| `memory_retrieval.py` | The one authoritative `retrieve_learner_memory` adapter — centralized learner filtering, session cache, post-retrieval selection, controlled fallback — plus `build_learner_context_block` for Gemini and `SessionMemoryCache`. |
| `retrieval_metrics.py` | `RetrievedMemory`, `RetrievalMetrics`, and the telemetry-event builder. Latency here is retrieval-only, never end-to-end. |

### Engine selection (stub-first)

`build_moss_engine` picks the engine from config + credential availability:

- `VERSA_RETRIEVAL_ENGINE=stub` → always the in-process `StubMossEngine`.
- real `MOSS_PROJECT_ID`/`MOSS_PROJECT_KEY` present → `RealMossEngine`.
- otherwise → the stub, **unless** `MOSS_REQUIRED=true`, which raises so a
  production config fails loudly instead of silently degrading.

The stub is a *Moss-shaped* engine (local semantic search over the project's
embeddings), not the pgvector control path — choosing it never routes live
retrieval through pgvector. It exists for the same reason `StubLLMClient`
does: offline, deterministic, free development and CI.

### Learner isolation (defense in depth)

Cross-learner leakage must be exactly zero. The `learner_id` filter is
applied at the engine query (`{"learner_id": ..., "status": "active"}`,
translated to Moss's `$and`/`$eq` filter DSL) **and** re-verified on every
returned record inside `retrieve_learner_memory`. The session cache is
filtered by learner too. The filter is never driven by arbitrary UI input.

## Client lifecycle

The `MossService` is built once, at application startup
(`webserver._build_moss_service` in the Starlette lifespan), seeded from the
durable `learner_facts` already in Postgres, and the index is loaded locally
before any query. It is shared by every session's `SessionLoop`. There is no
per-message client construction.

## Deployment

```text
Internet → Cloud Run (Starlette + Moss runtime, local loaded index) → Cloud SQL (Postgres)
                                    │
                                  Gemini
```

The Moss index is loaded into the process before serving retrieval requests.
Cold-start index load is measured and reported separately from warm query
latency (never reported as query latency). If Cloud Run scales to multiple
instances, each holds its own locally-loaded snapshot; the async per-turn
sync plus the startup reseed from Postgres keep them current.

## Benchmark branch

```text
same synthetic query workload
   ├── Moss local retrieval      (scripts/benchmark_retrieval.py --engine moss)
   └── pgvector baseline         (scripts/benchmark_retrieval.py --engine pgvector)
```

The benchmark is a separate path from the live product. It never falls back
between engines: a requested engine that fails makes the run fail. See
[BENCHMARKS.md](BENCHMARKS.md).
