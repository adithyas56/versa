"""Smallest safe real-Moss connectivity check (VERSA remaining-build §11).

Reads MOSS_PROJECT_ID / MOSS_PROJECT_KEY from .env (never prints them),
constructs the real Moss client, and runs a single READ operation
(`list_indexes`) — no index creation, no writes, negligible cost. Confirms
credentials are accepted before any indexing work.

    uv run python scripts/moss_connectivity_check.py
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from probe.moss_client import MossConfig


async def _main() -> int:
    load_dotenv()
    config = MossConfig.from_env()
    if not config.has_credentials:
        print("Moss credentials: MISSING (set MOSS_PROJECT_ID / MOSS_PROJECT_KEY)")
        return 1
    try:
        from moss import MossClient
    except ImportError:
        print("Moss SDK not installed")
        return 1

    client = MossClient(config.project_id, config.project_key)
    try:
        indexes = await client.list_indexes()
    except Exception as exc:  # noqa: BLE001 - surface a readable failure, no secrets
        print(f"Moss connectivity FAILED: {type(exc).__name__}: {exc}")
        return 1

    names = [i.name for i in indexes]
    print("Moss credentials accepted")
    print(f"existing indexes: {len(names)}")
    print(f"target index '{config.index_name}' exists: {config.index_name in names}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
