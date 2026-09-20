"""Sync all durable learner-memory facts into the Moss index (upsert)
— the batch counterpart to the per-turn async sync the live loop does
(VERSA build context §16). Safe to re-run: `add_docs` upserts by id.

    uv run python scripts/sync_memory_to_moss.py

With real MOSS credentials this upserts into the loaded cloud index; with
the stub engine it rebuilds the in-process index (ephemeral to the run).
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from probe.db import create_pool
from probe.memory import LearnerFactStore
from probe.memory_index import load_memory_docs_from_facts
from probe.moss_client import MossConfig, MossService


def _embedding_client():
    from probe.embeddings import StubEmbeddingClient, build_embedding_client

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    return build_embedding_client(api_key) if api_key else StubEmbeddingClient()


async def _main() -> None:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check .env)")
    pool = await create_pool(url, min_size=1, max_size=4)
    config = MossConfig.from_env()
    try:
        docs = await load_memory_docs_from_facts(LearnerFactStore(pool))
        # Load the existing index first, then upsert (a no-op ensure when
        # it already exists), so we never blow away a live cloud index.
        service = await MossService.create(config, _embedding_client())
        await service.sync_documents(docs)
        print(f"synced {len(docs)} memory documents into '{config.index_name}' "
              f"(engine: {service.engine.engine_name})")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
