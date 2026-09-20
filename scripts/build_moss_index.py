"""Build (or rebuild) the Moss learner-memory index from the durable
`learner_facts` in Postgres (VERSA build context §11/§19). Index creation
is a build step, never part of a user-facing turn (§16.1).

    uv run python scripts/build_moss_index.py
    uv run python scripts/build_moss_index.py --synthetic 10000   # add scale

With real MOSS credentials this creates/loads the cloud-built index; with
no credentials it validates the full projection -> index -> load pipeline
against the in-process stub engine (the index is then ephemeral to the
run, which is expected for the stub — the webserver reseeds from Postgres
at startup).
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv

from probe.db import create_pool
from probe.memory import LearnerFactStore
from probe.memory_index import MemoryDocument, load_memory_docs_from_facts
from probe.moss_client import MossConfig, MossService


def _embedding_client():
    from probe.embeddings import StubEmbeddingClient, build_embedding_client

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    return build_embedding_client(api_key) if api_key else StubEmbeddingClient()


def _synthetic(n: int) -> list[MemoryDocument]:
    return [
        MemoryDocument(
            id=f"synthetic_{i}",
            text=f"synthetic learner memory number {i} about a topic",
            metadata={"learner_id": f"synthetic_{i % 100}", "status": "active", "topic": "general"},
        )
        for i in range(n)
    ]


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build the Moss learner-memory index")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="add N synthetic memory docs (for scale testing)")
    args = parser.parse_args()

    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set (check .env)")
    pool = await create_pool(url, min_size=1, max_size=4)
    config = MossConfig.from_env()
    try:
        docs = await load_memory_docs_from_facts(LearnerFactStore(pool))
        if args.synthetic:
            docs = docs + _synthetic(args.synthetic)
        print(f"projecting {len(docs)} memory documents into index "
              f"'{config.index_name}' via engine '{config.engine}'...")
        service = await MossService.create(config, _embedding_client(), seed_docs=docs)
        print(f"index ready (engine: {service.engine.engine_name}, "
              f"local: {service.engine.is_local})")
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(_main())
