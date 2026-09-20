# Versa — Retrieval Benchmarks

Every number in this document is **measured** by `scripts/benchmark_retrieval.py`.
None are hand-written or copied from Moss's own published results. Re-run the
script to reproduce them; results depend on hardware, dataset size, and
whether the real Moss engine or the in-process stub engine is active.

## What is measured

- **Warm query latency** — the retrieval-only time from issuing a query to
  receiving results. For the real Moss engine this is the SDK's
  `SearchResult.time_taken_ms`; for the stub engine it is the engine's own
  `perf_counter` span. This is **not** end-to-end Gemini latency.
- **Index build/load** — reported separately and explicitly excluded from
  query latency (cold-start work must never be reported as query latency).
- **Percentiles** — P50 / P95 / P99, plus mean / min / max, over a warmed
  query set (the first N queries are discarded as warmup).

## How to run

```bash
# Moss engine (stub when no MOSS credentials are set; real Moss when they are)
uv run python scripts/benchmark_retrieval.py --engine moss --docs 10000 --queries 200

# pgvector baseline (requires DATABASE_URL — the retrieval path Versa used before Moss)
uv run python scripts/benchmark_retrieval.py --engine pgvector --docs 10000 --queries 200
```

Methodology reported by the script: documents indexed, warmup count,
measured-query count, top_k, index build/load time, and the full percentile
spread.

## Reference run (in-process stub engine)

Recorded on the development machine with the **stub** engine (no Moss
credentials), 10,000 synthetic documents, top_k=5, 20 warmup + 200 measured
queries:

```text
=== RETRIEVAL BENCHMARK (stub-moss) ===
documents indexed : 10,000
queries measured  : 200 (after 20 warmup)
top_k             : 5
index build/load  : ~3,386 ms   (NOT counted in query latency)
--- warm query latency (ms) ---
P50  : ~57
P95  : ~224
P99  : ~238
mean : ~87
```

### Reading these numbers honestly

The stub engine is a **local correctness/offline stand-in**, not a
representation of real Moss performance. Its per-query cost is dominated by
the deterministic stub *embedding* of the query text (a Python RNG over 768
dimensions), not by the vectorized cosine search. It exists so the entire
retrieval path — projection, filtering, injection, telemetry — runs and is
testable with no credentials, no network, and no cost.

**Real Moss numbers require real Moss credentials.** With
`MOSS_PROJECT_ID`/`MOSS_PROJECT_KEY` set, the same script exercises the real
`MossClient`, whose locally-loaded native index reports its own
`time_taken_ms` per query — the number that belongs in the demo and
submission. Set the credentials and re-run before recording final figures.

## Before / after (the product claim)

The comparison Versa makes is a **product-path** comparison, stated
narrowly:

> Moss reduces the *retrieval portion* of the learner-personalization path.

It is **not** a claim that Moss makes the Gemini model itself faster, and
**not** an algorithmic apples-to-apples micro-benchmark. The pgvector engine
times a real cosine-ANN query against a `vector(768)` column (the durable
retrieval path Versa used before Moss); the Moss engine times a query
against a locally-loaded index. The script documents exactly what each
engine's timing includes.

## Retrieval quality

Speed alone is insufficient. `scripts/evaluate_retrieval.py` reports
Recall@k on a small hand-labeled set and — the critical safety metric — the
cross-learner leakage rate, which must be exactly 0. A non-zero leakage rate
exits non-zero.

```text
=== RETRIEVAL QUALITY ===
labeled queries      : 3
Recall@k             : 1.00
cross-learner leaks  : 0
cross-learner rate   : 0.000  (must be 0.000)
PASS: zero cross-learner leakage.
```

Cross-learner isolation is additionally asserted as a unit test
(`tests/test_cross_learner_isolation.py`) so it is checked on every CI run,
not only when the evaluation script is invoked by hand.
