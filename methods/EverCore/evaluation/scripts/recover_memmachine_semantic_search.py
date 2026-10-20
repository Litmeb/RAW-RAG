"""Append semantic memories to an existing MemMachine search artifact.

This is a recovery tool for runs created before the adapter kept episodic and
semantic results in separate top-k budgets.  It preserves the already saved
episodic results and performs semantic-only lookups against the same MemMachine
projects, without running the evaluation Search stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import aiohttp


def _semantic_result(feature: dict[str, Any]) -> dict[str, Any] | None:
    value = feature.get("value")
    if not value:
        return None
    name = feature.get("feature_name") or feature.get("tag") or "fact"
    category = feature.get("category") or "semantic"
    return {
        "content": f"{category} / {name}: {value}",
        "score": 0.0,
        "user_id": str(feature.get("set_id") or ""),
        "metadata": {"memory_type": "semantic", "raw": feature},
    }


def _format_context(results: list[dict[str, Any]]) -> str:
    episodic = []
    semantic = []
    for result in results:
        content = str(result.get("content") or "").strip()
        if not content:
            continue
        if (result.get("metadata") or {}).get("memory_type") == "semantic":
            semantic.append(f"- {content}")
        else:
            episodic.append(content)

    sections = []
    if episodic:
        sections.append("<EPISODIC MEMORY>\n" + "\n".join(episodic) + "\n</EPISODIC MEMORY>")
    if semantic:
        sections.append("<SEMANTIC MEMORY>\n" + "\n".join(semantic) + "\n</SEMANTIC MEMORY>")
    return "\n\n".join(sections)


async def _semantic_lookup(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    org_id: str,
    project_id: str,
    query: str,
    top_k: int,
    semaphore: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    payload = {
        "org_id": org_id,
        "project_id": project_id,
        "query": query,
        "top_k": top_k,
        "expand_context": 0,
        "agent_mode": False,
        "types": ["semantic"],
    }
    async with semaphore:
        async with session.post(f"{base_url.rstrip('/')}/memories/search", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
    features = ((data.get("content") or {}).get("semantic_memory") or [])
    return [item for feature in features if (item := _semantic_result(feature))]


async def recover(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    search_path = run_dir / "search_results.json"
    rows = json.loads(search_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty list in {search_path}")

    backup_path = run_dir / "search_results.pre_semantic_recovery.json"
    if not backup_path.exists():
        shutil.copy2(search_path, backup_path)

    connector = aiohttp.TCPConnector(limit=args.workers)
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    semaphore = asyncio.Semaphore(args.workers)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            _semantic_lookup(
                session,
                base_url=args.base_url,
                org_id=str(row["retrieval_metadata"]["org_id"]),
                project_id=str(row["retrieval_metadata"]["project_id"]),
                query=str(row["query"]),
                top_k=args.semantic_top_k,
                semaphore=semaphore,
            )
            for row in rows
        ]
        semantic_groups = await asyncio.gather(*tasks)

    for row, semantic_results in zip(rows, semantic_groups, strict=True):
        episodic_results = [
            result
            for result in row.get("results", [])
            if (result.get("metadata") or {}).get("memory_type") != "semantic"
        ]
        merged_results = episodic_results + semantic_results
        metadata = row.setdefault("retrieval_metadata", {})
        metadata["requested_top_k"] = int(metadata.get("top_k", len(episodic_results)))
        metadata["top_k"] = len(merged_results)
        metadata["semantic_recovery"] = {
            "source": "MemMachine semantic-only API lookup",
            "episodic_result_count": len(episodic_results),
            "semantic_result_count": len(semantic_results),
            "combined_result_count": len(merged_results),
        }
        metadata["formatted_context"] = _format_context(merged_results)
        row["results"] = merged_results

    search_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    checkpoint_path = run_dir / "checkpoint_default.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.exists() else {}
    checkpoint["run_name"] = checkpoint.get("run_name", "default")
    checkpoint["completed_stages"] = ["add", "search"]
    checkpoint["search_results"] = rows
    checkpoint.pop("answer_results", None)
    checkpoint.pop("eval_results", None)
    checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    recovered = sum(len(group) for group in semantic_groups)
    print(f"Recovered {recovered} semantic results for {len(rows)} questions.")
    print(f"Updated: {search_path}")
    print(f"Backup:  {backup_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8080/api/v2")
    parser.add_argument("--semantic-top-k", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(recover(parse_args()))
