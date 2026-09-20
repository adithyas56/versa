# Change Report — Moss Learner-Memory Retrieval Integration

**Date:** 2026-09-18
**Scope:** Add Moss as the low-latency semantic retrieval layer for learner
memory on Versa's live tutoring path (stub-first), plus written deliverables.
**Build spec:** `VERSA_MOSS_HACKATHON_BUILD_CONTEXT`.

---

## 1. Executive summary

Moss is now a real retrieval dependency on the live answer path: every
answer-producing turn retrieves this learner's evidence-backed memories from
a locally-loaded Moss index and injects them into the Gemini final-answer
prompt. PostgreSQL remains the durable source of truth; Moss is the hot-path
retrieval runtime.

The integration is **stub-first**: with no Moss credentials it runs against
an in-process stub engine (offline, deterministic, free), so the whole path
is testable now and flips to real Moss by adding `MOSS_PROJECT_ID` /
`MOSS_PROJECT_KEY`. No benchmark numbers are fabricated.

**Verification status:** all 28 new stub-based tests pass; scripts run and
produce measured output; projection validated against the real `moss` SDK.
The DB-backed integration tests were **not run** in this environment because
Docker/Postgres would not start — see §5.

---

## 2. New files

### Core retrieval layer (`src/probe/`)

| File | Lines | Purpose | Stage |
|---|---:|---|---|
| `retrieval_metrics.py` | 108 | `RetrievedMemory`, `RetrievalMetrics`, telemetry-event builder. Retrieval-only latency. | Complete, unit-tested |
| `memory_index.py` | 142 | Pure `LearnerFact → MemoryDocument` projection + lazy `→ moss.DocumentInfo`. | Complete, unit-tested |
| `moss_client.py` | 422 | `MossConfig`, `MossEngine` interface, `RealMossEngine` (real SDK), `StubMossEngine` (in-process, numpy-vectorized), `MossService` lifecycle. | Complete, unit-tested |
| `memory_retrieval.py` | 206 | The one authoritative `retrieve_learner_memory` adapter + `SessionMemoryCache` + `build_learner_context_block`. | Complete, unit-tested |

### Scripts (`scripts/`)

| File | Lines | Purpose | Stage |
|---|---:|---|---|
| `seed_demo_learners.py` | 105 | Two demo personas (example-first / theory-first). | Complete; needs DB to run |
| `build_moss_index.py` | 74 | Project facts → build/load Moss index. | Complete; needs DB to run |
| `sync_memory_to_moss.py` | 52 | Batch upsert facts into Moss. | Complete; needs DB to run |
| `benchmark_retrieval.py` | 242 | Moss vs pgvector, measured P50/P95/P99, warm-vs-build separated. | Complete; Moss path run (10k docs), pgvector path needs DB |
| `evaluate_retrieval.py` | 88 | Recall@k + zero cross-learner-leakage gate. | Complete; run, passes (Recall 1.00, 0 leaks) |

### Tests (`tests/`)

| File | Lines | Purpose | Stage |
|---|---:|---|---|
| `test_memory_projection.py` | 84 | Projection metadata/text/memory-type mapping. | Passing |
| `test_moss_retrieval.py` | 155 | Adapter: ranking, metrics, threshold, cap, session cache, fallback, context block. | Passing |
| `test_cross_learner_isolation.py` | 82 | **Mandatory** zero cross-learner leakage. | Passing |
| `test_learner_filtering.py` | 60 | Filter DSL translation + config-from-env. | Passing |
| `test_benchmark_sanity.py` | 50 | Percentile math + ordered measured stats. | Passing |

### Documentation (repo root)

| File | Purpose | Stage |
|---|---|---|
| `PRD.md` | Product requirements. | Complete |
| `ARCHITECTURE.md` | System + Moss-layer architecture, diagrams, deployment. | Complete |
| `BENCHMARKS.md` | Methodology + measured reference run + honesty notes. | Complete (stub reference numbers; real-Moss numbers pending credentials) |
| `DEMO_SCRIPT.md` | 90–120s demo flow. | Complete |
| `CHANGES_MOSS_INTEGRATION.md` | This report. | Complete |

---

## 3. Modified files

| File | Change | Stage |
|---|---|---|
| `src/probe/loop.py` (+129) | `SessionLoop` gains `moss_service` + `moss_max_inject`; `_run_final_answer` retrieves from Moss and injects `moss_memory_block`; `_sync_new_fact_to_memory_index` (session cache sync + background Moss upsert); `last_moss_metrics`/`last_moss_memories` for telemetry. | Complete; import-verified. DB integration test pending. |
| `src/probe/disambiguate.py` (+15) | `FinalAnswer.run` gains `moss_memory_block` param, placed in the prompt after `learner_history_block`. | Complete; verified block lands in prompt and empty→omitted. |
| `src/probe/session_builder.py` (+3) | `build_session_loop` accepts and forwards `moss_service`. | Complete |
| `src/probe/webserver.py` (+89) | Build `MossService` once at startup (lifespan, seeded from Postgres); thread into every loop; emit `retrieval` telemetry in the turn's `done` SSE event; added module logger. | Complete; import-verified. Live run pending DB. |
| `src/probe/memory.py` (+14) | `LearnerFactStore.list_all()` — read-only seed source (append-only invariant untouched). | Complete |
| `src/probe/static/app.js` (+43) | `mossPanelHtml` + wire retrieval telemetry into the answer render. | Complete |
| `src/probe/static/index.html` (+14) | CSS for the MOSS MEMORY RETRIEVAL panel. | Complete |
| `.env.example` (+30) | All `MOSS_*` + `VERSA_*` variables documented. | Complete |
| `pyproject.toml` (+1) | `moss>=1.12.0` dependency. | Complete |
| `README.md` (+39) | Moss in the architecture diagram, layers, setup steps, project layout, built-with. | Complete |
| `uv.lock` (+26) | Lock update for `moss` + `inferedge-moss-core`. | Complete |

---

## 4. Live-path data flow (current state)

```
Student message
  → disambiguation flow (unchanged: assess / options / click)
  → _run_final_answer
       builds moss_memory_block via retrieve_learner_memory
       (learner-filtered Moss query → build_learner_context_block)
  → FinalAnswer prompt (Gemini)  ← Moss memory injected here
  → answer shown + MOSS MEMORY RETRIEVAL telemetry panel
  → WriteLearnerFact → learner_facts (Postgres, source of truth)
       → SessionMemoryCache.add (sync, immediate continuity)
       → MossService.sync_documents (background, off response path)
```

Learner isolation is enforced centrally in `retrieve_learner_memory`
(engine-level filter + per-record re-check + cache filter).

---

## 5. Verification status

**Verified (no Postgres needed):**
- 28 new stub tests pass (`test_memory_projection`, `test_moss_retrieval`,
  `test_cross_learner_isolation`, `test_learner_filtering`,
  `test_benchmark_sanity`).
- `benchmark_retrieval.py --engine moss` ran on 10,000 synthetic docs and
  reported measured percentiles.
- `evaluate_retrieval.py`: Recall@k = 1.00, cross-learner leaks = 0.
- All modules import cleanly under `uv`; `FinalAnswer` prompt placement and
  the telemetry builder verified directly; projection validated against the
  real `moss` SDK types.
- ruff clean on new files except one intentional broad `except` (controlled
  fallback), matching the codebase's existing style.

**Not verified here (blocked):**
- DB-backed integration tests (`test_disambiguation_loop_wiring`,
  `test_webserver_interaction_pipeline_wiring`, etc.) — **Docker Desktop did
  not start in this environment**, so Postgres was unavailable. The loop /
  webserver edits are additive with safe defaults (`moss_service=None` skips
  the new paths; new `FinalAnswer` param defaults to `""`), so existing tests
  are expected to pass, but this is reasoned, not proven.
  - To verify: `docker compose up -d` then `uv run pytest -q`.

---

## 6. Follow-ups / notes

- **Stub ≠ real Moss numbers.** The default runtime uses the offline stub
  engine; its latency is dominated by stub embedding, not search. Set real
  `MOSS_*` credentials and re-run for demo/BENCHMARKS figures.
- **Out of code scope (per agreement):** demo video recording and Cloud Run
  deployment — covered by `DEMO_SCRIPT.md` and `ARCHITECTURE.md`.
- **Housekeeping (pre-existing, not changed here):** compiled `__pycache__/*.pyc`
  files appear tracked in git and show as modified. Consider adding
  `__pycache__/` to `.gitignore` and `git rm --cached` them — unrelated to
  this feature.
