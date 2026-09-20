"""Moss client lifecycle, configuration, and the stub engine.

VERSA build context §18/§19: one long-lived, application-level Moss engine —
never a client per user message — that loads the learner-memory index once
and answers in-memory queries with no per-query network round trip (§7.3).
`memory_retrieval.py` is the single caller; nothing else in the codebase
should touch a raw `moss.MossClient`.

Two engines implement one interface (`MossEngine`):

- `RealMossEngine` wraps the installed `moss` SDK (`MossClient`). Used when
  `MOSS_PROJECT_ID`/`MOSS_PROJECT_KEY` are configured.

- `StubMossEngine` runs the same contract fully in-process using the
  project's existing `EmbeddingClient` + cosine similarity — the Moss
  analogue of `StubLLMClient` (§20.1/§36 Mode B). It needs no credentials
  and no network, so the whole live path, its tests, and the two-learner
  demo run offline. It is a *Moss-shaped* engine, not the pgvector control
  path (§0.5/§46 Failure 2): choosing the stub never silently routes live
  retrieval through pgvector.

The engine boundary speaks a small, neutral filter dict
(`{"learner_id": ..., "status": "active"}`); `RealMossEngine` translates it
to Moss's `$and`/`$eq` filter DSL, and the stub applies it directly. That
keeps the learner-isolation filter (§11.3/§34.5) in one shape across engines.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from probe.embeddings import TASK_DOCUMENT, TASK_QUERY, EmbeddingClient
from probe.memory_index import MemoryDocument, to_moss_documents
from probe.vector_math import cosine_similarity

DEFAULT_INDEX_NAME = "versa-learner-memory"


@dataclass
class MossHit:
    """One engine-level result, before the retrieval adapter's learner
    filtering / thresholding / capping (§12.5). Deliberately flatter than
    `RetrievedMemory` — no `source` yet, since the engine does not know
    whether it is the live Moss path or the benchmark control."""

    id: str
    text: str
    score: float
    metadata: dict[str, str]


@dataclass
class MossQueryResult:
    hits: list[MossHit]
    # The engine's own measured retrieval time. RealMossEngine reads the
    # SDK's `SearchResult.time_taken_ms` (§46 Failure 3: use the real metric,
    # never a hand-written number); the stub measures its own perf_counter
    # span. Either way this is retrieval-only latency (§22.1), not end-to-end.
    time_taken_ms: float
    index_name: str
    used_local_index: bool


class MossEngine(Protocol):
    engine_name: str
    index_name: str
    is_local: bool

    async def ensure_index(self, docs: list[MemoryDocument]) -> None: ...
    async def load(self) -> None: ...
    async def add_documents(self, docs: list[MemoryDocument]) -> None: ...
    async def query(
        self,
        query_text: str,
        *,
        top_k: int,
        filter: dict[str, str] | None = None,
    ) -> MossQueryResult: ...


@dataclass
class MossConfig:
    """Runtime configuration, read from the §27 environment variables.

    Defaults are stub-first (§0.7): no credentials required, `required` off,
    so a fresh checkout runs the live path against the stub engine. Set
    `MOSS_REQUIRED=true` in the deployed/production configuration to make a
    missing real engine fail loudly at startup (§20.1) instead of degrading.
    """

    project_id: str | None = None
    project_key: str | None = None
    index_name: str = DEFAULT_INDEX_NAME
    top_k: int = 5
    min_score: float = 0.0
    auto_refresh: bool = False
    refresh_seconds: int = 300
    required: bool = False
    # "moss" prefers the real SDK when credentials exist, else the stub.
    # "stub" forces the in-process engine even if credentials are present
    # (useful for offline tests/demos). "pgvector" is not a value here — the
    # control path lives in the benchmark, never in the live engine (§0.5).
    engine: str = "moss"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MossConfig:
        e = env if env is not None else os.environ

        def _bool(key: str, default: bool) -> bool:
            raw = e.get(key)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            project_id=e.get("MOSS_PROJECT_ID") or None,
            project_key=e.get("MOSS_PROJECT_KEY") or None,
            index_name=e.get("MOSS_INDEX_NAME", DEFAULT_INDEX_NAME),
            top_k=int(e.get("MOSS_TOP_K", "5")),
            min_score=float(e.get("MOSS_MIN_SCORE", "0.0")),
            auto_refresh=_bool("MOSS_AUTO_REFRESH", False),
            refresh_seconds=int(e.get("MOSS_REFRESH_SECONDS", "300")),
            required=_bool("MOSS_REQUIRED", False),
            engine=e.get("VERSA_RETRIEVAL_ENGINE", "moss").strip().lower(),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.project_id and self.project_key)


def _to_moss_filter(filter: dict[str, str] | None) -> dict | None:
    """Neutral `{field: value}` dict -> Moss's `$and`/`$eq` filter DSL (see
    the SDK's `query` docstring). Single-condition filters still go through
    `$and` for one uniform shape."""
    if not filter:
        return None
    conditions = [
        {"field": k, "condition": {"$eq": v}} for k, v in filter.items()
    ]
    return {"$and": conditions}


def _matches_filter(metadata: dict[str, str], filter: dict[str, str] | None) -> bool:
    if not filter:
        return True
    return all(metadata.get(k) == v for k, v in filter.items())


class RealMossEngine:
    """Thin wrapper over `moss.MossClient` (§28 SDK). All SDK methods are
    async and the loaded index is queried in-memory (§7.3), so `query` never
    makes a network round trip after `load()`."""

    engine_name = "moss"
    is_local = True

    def __init__(self, config: MossConfig) -> None:
        from moss import MossClient  # lazy: only imported on the real path

        self._config = config
        self.index_name = config.index_name
        self._client = MossClient(config.project_id, config.project_key)
        self._loaded = False

    async def ensure_index(self, docs: list[MemoryDocument]) -> None:
        """Create the index if it does not exist yet; otherwise leave it
        (documents are kept current via `add_documents`, upsert). Index
        *creation* is a build step, not part of any user-facing turn
        (§16.1)."""
        existing = {i.name for i in await self._client.list_indexes()}
        if self.index_name not in existing:
            await self._client.create_index(
                self.index_name, to_moss_documents(docs), wait=True
            )

    async def load(self) -> None:
        await self._client.load_index(
            self.index_name,
            auto_refresh=self._config.auto_refresh,
            polling_interval_in_seconds=self._config.refresh_seconds,
        )
        self._loaded = True

    async def add_documents(self, docs: list[MemoryDocument]) -> None:
        from moss import MutationOptions

        if not docs:
            return
        await self._client.add_docs(
            self.index_name,
            to_moss_documents(docs),
            MutationOptions(upsert=True),
        )

    async def query(
        self,
        query_text: str,
        *,
        top_k: int,
        filter: dict[str, str] | None = None,
    ) -> MossQueryResult:
        from moss import QueryOptions

        options = QueryOptions(top_k=top_k, filter=_to_moss_filter(filter))
        result = await self._client.query(self.index_name, query_text, options)
        hits = [
            MossHit(
                id=d.id,
                text=d.text,
                score=float(d.score),
                metadata=dict(d.metadata or {}),
            )
            for d in result.docs
        ]
        # SearchResult.time_taken_ms is the SDK's own measured retrieval time.
        latency = float(result.time_taken_ms) if result.time_taken_ms is not None else 0.0
        return MossQueryResult(
            hits=hits,
            time_taken_ms=latency,
            index_name=self.index_name,
            used_local_index=True,
        )


class StubMossEngine:
    """In-process Moss stand-in: stores projected documents, embeds their
    text once with the project `EmbeddingClient`, and answers queries by
    cosine similarity + the same neutral metadata filter. Deterministic
    under `StubEmbeddingClient` (§37), free, offline.

    Marked `used_local_index=True` and reports its own measured span, but
    names itself `stub-moss` in telemetry so it is never displayed as if the
    real Moss cloud index had run (§20.2/§46 Failure 3 honesty)."""

    engine_name = "stub-moss"
    is_local = True

    def __init__(self, config: MossConfig, embedding_client: EmbeddingClient) -> None:
        self._config = config
        self.index_name = config.index_name
        self._embed = embedding_client
        self._docs: dict[str, MemoryDocument] = {}
        self._vectors: dict[str, list[float]] = {}
        # A stacked, L2-normalized matrix of the current doc vectors, in
        # `_order`, rebuilt lazily on the first query after any add. Lets a
        # query be one vectorized dot product instead of a Python loop over
        # every doc -- honest local brute-force search, still "stub-moss",
        # just not artificially slow for the offline demo. Falls back to a
        # pure-Python loop when numpy is unavailable.
        self._matrix = None
        self._order: list[str] = []
        self._dirty = True

    async def ensure_index(self, docs: list[MemoryDocument]) -> None:
        await self.add_documents(docs)

    async def load(self) -> None:  # nothing to load — already in-process
        return None

    async def add_documents(self, docs: list[MemoryDocument]) -> None:
        for doc in docs:
            self._docs[doc.id] = doc
            # Reuse the durable row's vector when present (avoids a re-embed);
            # otherwise embed the projected text as a DOCUMENT, mirroring
            # WriteLearnerFact's asymmetric query/document embedding roles.
            if doc.embedding is not None:
                self._vectors[doc.id] = list(doc.embedding)
            else:
                self._vectors[doc.id] = await self._embed.embed(
                    doc.text, task_type=TASK_DOCUMENT
                )
        self._dirty = True

    async def query(
        self,
        query_text: str,
        *,
        top_k: int,
        filter: dict[str, str] | None = None,
    ) -> MossQueryResult:
        start = time.perf_counter()
        query_vec = await self._embed.embed(query_text, task_type=TASK_QUERY)
        scored = self._search(query_vec, filter)
        scored.sort(key=lambda h: h.score, reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return MossQueryResult(
            hits=scored[:top_k],
            time_taken_ms=elapsed_ms,
            index_name=self.index_name,
            used_local_index=True,
        )

    def _rebuild_matrix(self):
        try:
            import numpy as np
        except ImportError:
            self._matrix = None
            return
        self._order = list(self._docs.keys())
        if not self._order:
            self._matrix = np.zeros((0, 0))
            self._dirty = False
            return
        mat = np.array([self._vectors[i] for i in self._order], dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        self._dirty = False

    def _search(self, query_vec: list[float], filter: dict[str, str] | None) -> list[MossHit]:
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            if self._dirty:
                self._rebuild_matrix()
            if self._matrix is None or len(self._order) == 0:
                return []
            q = np.array(query_vec, dtype=float)
            qn = np.linalg.norm(q)
            if qn != 0:
                q = q / qn
            sims = self._matrix @ q
            hits: list[MossHit] = []
            for idx, doc_id in enumerate(self._order):
                doc = self._docs[doc_id]
                if not _matches_filter(doc.metadata, filter):
                    continue
                hits.append(
                    MossHit(id=doc.id, text=doc.text, score=float(sims[idx]),
                            metadata=dict(doc.metadata))
                )
            return hits
        # Pure-Python fallback.
        hits = []
        for doc_id, doc in self._docs.items():
            if not _matches_filter(doc.metadata, filter):
                continue
            hits.append(
                MossHit(id=doc.id, text=doc.text,
                        score=cosine_similarity(query_vec, self._vectors[doc_id]),
                        metadata=dict(doc.metadata))
            )
        return hits

    @property
    def doc_count(self) -> int:
        return len(self._docs)


def build_moss_engine(
    config: MossConfig,
    embedding_client: EmbeddingClient,
) -> MossEngine:
    """Pick the engine per config + credential availability (§20.1).

    - `engine == "stub"` -> always the stub.
    - real credentials present (and the SDK importable) -> `RealMossEngine`.
    - otherwise -> the stub, UNLESS `required` is set, which raises so a
      production/deploy config that expects real Moss fails loudly rather
      than silently degrading.
    """
    if config.engine == "stub":
        return StubMossEngine(config, embedding_client)

    if config.has_credentials:
        try:
            return RealMossEngine(config)
        except ImportError as exc:  # SDK not installed
            if config.required:
                raise RuntimeError(
                    "MOSS_REQUIRED is set but the `moss` SDK is not installed"
                ) from exc

    if config.required:
        raise RuntimeError(
            "MOSS_REQUIRED is set but MOSS_PROJECT_ID/MOSS_PROJECT_KEY are missing"
        )
    return StubMossEngine(config, embedding_client)


class MossService:
    """The long-lived application service (§19). Built once at startup, holds
    the loaded engine, and is the object `memory_retrieval` queries. Also the
    write side of the async memory sync (§16 `sync_documents`)."""

    def __init__(self, engine: MossEngine, config: MossConfig) -> None:
        self.engine = engine
        self.config = config

    @classmethod
    async def create(
        cls,
        config: MossConfig,
        embedding_client: EmbeddingClient,
        *,
        seed_docs: list[MemoryDocument] | None = None,
    ) -> MossService:
        engine = build_moss_engine(config, embedding_client)
        if seed_docs:
            await engine.ensure_index(seed_docs)
        await engine.load()
        return cls(engine, config)

    async def query(
        self,
        query_text: str,
        *,
        top_k: int,
        filter: dict[str, str] | None = None,
    ) -> MossQueryResult:
        return await self.engine.query(query_text, top_k=top_k, filter=filter)

    async def sync_documents(self, docs: list[MemoryDocument]) -> None:
        """Append/update memory projections in the index. Called off the
        user-facing response path (§16.1) after Postgres has been written."""
        await self.engine.add_documents(docs)
