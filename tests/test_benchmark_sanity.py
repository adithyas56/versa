"""Sanity checks for the retrieval benchmark (VERSA build context §26).
A benchmark that reports nonsense percentiles is worse than none — these
pin the percentile math and the end-to-end Moss run producing ordered,
non-fabricated stats.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("benchmark_retrieval")


def test_percentile_is_monotonic_and_bounded():
    values = [float(i) for i in range(100)]
    p50 = bench._percentile(values, 0.50)
    p95 = bench._percentile(values, 0.95)
    p99 = bench._percentile(values, 0.99)
    assert 0.0 <= p50 <= p95 <= p99 <= 99.0


def test_percentile_empty_is_zero():
    assert bench._percentile([], 0.5) == 0.0


@pytest.mark.asyncio
async def test_moss_benchmark_produces_ordered_measured_stats():
    docs = bench._synthetic_docs(300)
    queries = bench._query_set(30)
    stats = await bench._bench_moss(docs, queries, top_k=5, warmup=5)
    assert stats.engine == "stub-moss"
    assert stats.doc_count == 300
    assert stats.query_count == 30
    # Real measured latencies: ordered and non-negative, never fabricated.
    assert stats.min >= 0.0
    assert stats.p50 <= stats.p95 <= stats.p99
    assert stats.build_ms >= 0.0  # build time is tracked separately
