"""Reproducible retrieval benchmark: Moss vs the pgvector baseline
(VERSA build context §23). Every number this prints is MEASURED here —
nothing is hard-coded (§0.6/§46 Failure 3). Warm query latency is
reported separately from index build/load (§22.4/§46 Failure 8), and a
requested engine that fails makes the run FAIL rather than silently
switching engines (§20.3).

Usage (from repo root):

    uv run python scripts/benchmark_retrieval.py --engine moss --docs 10000 --queries 200
    uv run python scripts/benchmark_retrieval.py --engine pgvector --docs 10000 --queries 200

The `moss` engine uses the in-process StubMossEngine when no Moss
credentials are configured (stub-first, §0.7) and the real Moss cloud
index when they are — either way the query runs against a locally loaded
index and the reported latency is `SearchResult.time_taken_ms` / the
engine's own measured span. The `pgvector` engine benchmarks the durable
`learner_facts` cosine-ANN path that Versa used before Moss (§25 before/
after), and requires DATABASE_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass

from probe.embeddings import EMBEDDING_DIM, StubEmbeddingClient
from probe.memory_index import MemoryDocument
from probe.moss_client import MossConfig, MossService


def _synthetic_docs(n: int) -> list[MemoryDocument]:
    """Deterministic synthetic learner-memory documents (§23.1). Not real
    student data (§34.1). The engine embeds each doc's text at add time."""
    docs: list[MemoryDocument] = []
    templates = [
        "prefers concrete examples before abstract definitions",
        "recently studied recursion and struggled with base cases",
        "likes concise, technical terminology",
        "asks for worked examples before the formal statement",
        "resolved the ambiguity around 'tree' as a data structure",
        "prefers formal definitions before examples",
        "recently studied algorithmic complexity",
    ]
    # Embeddings are computed lazily by the engine; here we only need the
    # text + metadata. The stub engine embeds text at add time.
    for i in range(n):
        learner = f"learner_{i % 50}"
        text = f"{templates[i % len(templates)]} (memory {i})"
        docs.append(
            MemoryDocument(
                id=f"mem_{i}",
                text=text,
                metadata={
                    "learner_id": learner,
                    "status": "active",
                    "topic": "general",
                },
                embedding=None,  # let the engine embed the text
            )
        )
    return docs


def _query_set(n: int) -> list[str]:
    base = [
        "Explain dynamic programming like you explained recursion.",
        "How should you explain this concept to me?",
        "What do I usually prefer when learning algorithms?",
        "What confusion have I had around tree traversal?",
        "Explain graph traversal in my usual style.",
    ]
    return [base[i % len(base)] for i in range(n)]


@dataclass
class Stats:
    engine: str
    doc_count: int
    query_count: int
    top_k: int
    warmup: int
    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float
    build_ms: float

    def render(self) -> str:
        return (
            f"\n=== RETRIEVAL BENCHMARK ({self.engine}) ===\n"
            f"documents indexed : {self.doc_count:,}\n"
            f"queries measured  : {self.query_count} (after {self.warmup} warmup)\n"
            f"top_k             : {self.top_k}\n"
            f"index build/load  : {self.build_ms:.1f} ms  (NOT counted in query latency)\n"
            f"--- warm query latency (ms) ---\n"
            f"P50  : {self.p50:.3f}\n"
            f"P95  : {self.p95:.3f}\n"
            f"P99  : {self.p99:.3f}\n"
            f"mean : {self.mean:.3f}\n"
            f"min  : {self.min:.3f}\n"
            f"max  : {self.max:.3f}\n"
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _stats(engine: str, latencies: list[float], *, doc_count: int, top_k: int,
           warmup: int, build_ms: float) -> Stats:
    return Stats(
        engine=engine,
        doc_count=doc_count,
        query_count=len(latencies),
        top_k=top_k,
        warmup=warmup,
        p50=_percentile(latencies, 0.50),
        p95=_percentile(latencies, 0.95),
        p99=_percentile(latencies, 0.99),
        mean=statistics.fmean(latencies),
        min=min(latencies),
        max=max(latencies),
        build_ms=build_ms,
    )


async def _bench_moss(docs, queries, top_k, warmup) -> Stats:
    embed = StubEmbeddingClient()
    config = MossConfig.from_env()
    config.top_k = top_k
    t0 = time.perf_counter()
    service = await MossService.create(config, embed, seed_docs=docs)
    build_ms = (time.perf_counter() - t0) * 1000  # index build/load, kept separate

    all_q = queries[:warmup] + queries
    latencies: list[float] = []
    for i, q in enumerate(all_q):
        result = await service.query(q, top_k=top_k, filter={"status": "active"})
        if i >= warmup:  # discard warmup (§22.4)
            latencies.append(result.time_taken_ms)
    return _stats(
        service.engine.engine_name, latencies,
        doc_count=len(docs), top_k=top_k, warmup=warmup, build_ms=build_ms,
    )


async def _bench_pgvector(docs, queries, top_k, warmup) -> Stats:
    """Baseline: the durable `learner_facts` cosine-ANN path. Requires
    DATABASE_URL. Fails loudly if the DB is unavailable — never falls back
    to Moss (§20.3)."""
    import os
    import uuid

    import asyncpg
    from dotenv import load_dotenv

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("pgvector benchmark requires DATABASE_URL")

    embed = StubEmbeddingClient()
    pool = await asyncpg.create_pool(url, min_size=1, max_size=4)
    table = f"bench_facts_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"CREATE TABLE {table} (id text, learner_id text, "
            f"embedding vector({EMBEDDING_DIM}))"
        )
        from pgvector.asyncpg import register_vector

        await register_vector(conn)
        for d in docs:
            vec = await embed.embed(d.text)
            await conn.execute(
                f"INSERT INTO {table} (id, learner_id, embedding) VALUES ($1,$2,$3)",
                d.id, d.metadata["learner_id"], vec,
            )
    build_ms = (time.perf_counter() - t0) * 1000

    all_q = queries[:warmup] + queries
    latencies: list[float] = []
    try:
        async with pool.acquire() as conn:
            from pgvector.asyncpg import register_vector

            await register_vector(conn)
            for i, q in enumerate(all_q):
                qvec = await embed.embed(q)
                start = time.perf_counter()
                await conn.fetch(
                    f"SELECT id FROM {table} ORDER BY embedding <=> $1 LIMIT $2",
                    qvec, top_k,
                )
                elapsed = (time.perf_counter() - start) * 1000
                if i >= warmup:
                    latencies.append(elapsed)
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {table}")
        await pool.close()
    return _stats(
        "pgvector", latencies,
        doc_count=len(docs), top_k=top_k, warmup=warmup, build_ms=build_ms,
    )


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Versa retrieval benchmark")
    parser.add_argument("--engine", choices=["moss", "pgvector"], required=True)
    parser.add_argument("--docs", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    docs = _synthetic_docs(args.docs)
    queries = _query_set(args.queries)

    if args.engine == "moss":
        stats = await _bench_moss(docs, queries, args.top_k, args.warmup)
    else:
        stats = await _bench_pgvector(docs, queries, args.top_k, args.warmup)
    print(stats.render())


if __name__ == "__main__":
    asyncio.run(_main())
