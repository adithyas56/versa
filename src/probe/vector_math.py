"""Pure vector math shared by two otherwise-unrelated callers:
`population_patterns.py`'s cross-learner clustering (a batch job,
unaffected by the change that removed per-learner topic clustering)
and `interactions.py`'s per-turn similarity computation (which
replaced topic clustering after two rounds of tuning plus a systematic
replay confirmed a fixed-threshold running centroid has no working
constant for this text length -- see topics_removal in interactions.py
and TopicConfig's git history for the full failure record).

No state, no I/O -- these two functions used to live in topics.py
alongside the class that owned the failed mechanism; extracted here so
neither remaining caller has to import from, or depend on, a module
named for a concept that no longer exists.
"""

from __future__ import annotations


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def running_mean(old_centroid: list[float], new_vec: list[float], n: int) -> list[float]:
    """New centroid after attaching one more vector to a running mean
    of `n` prior vectors. Still exactly correct for population_patterns'
    batch clustering (cross-learner, offline, not the per-turn online
    context where this exact computation caused the cascading-merge
    failure)."""
    return [
        old + (new - old) / (n + 1)
        for old, new in zip(old_centroid, new_vec, strict=True)
    ]
