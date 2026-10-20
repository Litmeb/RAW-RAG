"""Build offline MemoryAgentBench RAG runs that use embedding ranking only.

The source ``normal-recover-rerank`` runs retain every embedding candidate in
``retrieval_metadata.all_rerank_results``.  This utility reuses those persisted
candidates, restoring their original ``embedding_rank`` order and selecting the
same final K used by the source run.  It therefore isolates the reranker without
re-running Add, embeddings, or Search.

Destination runs contain only ``search_results.json``, a hard-linked
``rag_index.json``, and an audit manifest.  Run Answer/Evaluate afterwards with
the destination run name.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_SUFFIX = "normal-recover-rerank"
DEFAULT_RUN_NAME = "embedding-only-topk15-20260815"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_chunk_lookup(index_data: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for conversation_id, conv_index in index_data.get("conversations", {}).items():
        by_key: dict[str, dict[str, Any]] = {}
        for chunk in conv_index.get("chunks", []):
            if chunk.get("chunk_id"):
                by_key[str(chunk["chunk_id"])] = chunk
            if chunk.get("chunk_index") is not None:
                by_key[f"chunk_index:{chunk['chunk_index']}"] = chunk
        lookup[conversation_id] = by_key
    return lookup


def expand_chunk_to_turns(chunk: dict[str, Any], turn_context: dict[str, Any] | None) -> str:
    """Match RAG's optional turn-expanded context formatting."""
    if not turn_context or chunk.get("start_token") is None or chunk.get("end_token") is None:
        return str(chunk.get("content", ""))
    overlaps = []
    for index, span in enumerate(turn_context.get("turn_spans", [])):
        start, end = int(span[0]), int(span[1])
        if end <= int(chunk["start_token"]):
            continue
        if start >= int(chunk["end_token"]):
            break
        overlaps.append(index)
    if not overlaps:
        return str(chunk.get("content", ""))
    first, last = overlaps[0], overlaps[-1]
    parts = []
    if chunk.get("chunk_index", 0) == 0 or first == 0:
        parts.extend([str(turn_context.get("header", "")), ""])
    parts.extend(turn_context.get("turn_lines", [])[first : last + 1])
    return "\n".join(parts)


def format_context(results: list[dict[str, Any]], turn_context: dict[str, Any] | None) -> str:
    if not results:
        return "(No relevant memories found)"
    formatted = []
    for rank, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        display_content = expand_chunk_to_turns(
            {
                "content": result.get("content", ""),
                "chunk_index": metadata.get("chunk_index"),
                "start_token": metadata.get("start_token"),
                "end_token": metadata.get("end_token"),
            },
            turn_context,
        )
        formatted.append(
            f"Memory {rank} (chunk={metadata.get('chunk_index')}, "
            f"similarity={float(metadata.get('similarity_score', 0.0)):.6f}, "
            f"rerank={metadata.get('rerank_score')}):\n{display_content}"
        )
    return "\n\n".join(formatted)


def embedding_candidates(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return persisted candidates in the order a rerank-disabled adapter uses."""
    candidates = metadata.get("all_rerank_results")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("missing retrieval_metadata.all_rerank_results")
    if any(item.get("embedding_rank") is None for item in candidates):
        raise ValueError("embedding candidate is missing embedding_rank")
    ordered = sorted(candidates, key=lambda item: int(item["embedding_rank"]))
    ranks = [int(item["embedding_rank"]) for item in ordered]
    if ranks != list(range(1, len(ordered) + 1)):
        raise ValueError(f"embedding ranks must be contiguous from 1, got {ranks[:5]}...")
    return ordered


def rebuild_record_embedding_only(
    record: dict[str, Any],
    conv_index: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Rebuild one saved result as the equivalent rerank-disabled search output."""
    metadata = dict(record.get("retrieval_metadata") or {})
    candidates = embedding_candidates(metadata)
    selected_k = int(top_k if top_k is not None else metadata.get("rerank_top_k", 0))
    if selected_k <= 0:
        raise ValueError("top_k must be positive or source rerank_top_k must be positive")

    results: list[dict[str, Any]] = []
    all_embedding_results: list[dict[str, Any]] = []
    for embedding_rank, item in enumerate(candidates, 1):
        chunk = chunk_lookup.get(str(item.get("chunk_id")))
        if chunk is None and item.get("chunk_index") is not None:
            chunk = chunk_lookup.get(f"chunk_index:{item['chunk_index']}")
        if chunk is None:
            raise KeyError(
                "chunk missing from rag_index for "
                f"conversation={record.get('conversation_id')!r}, item={item!r}"
            )
        base = {
            "chunk_id": chunk.get("chunk_id"),
            "chunk_index": chunk.get("chunk_index"),
            "start_turn": chunk.get("start_turn"),
            "end_turn": chunk.get("end_turn"),
            "start_token": chunk.get("start_token"),
            "end_token": chunk.get("end_token"),
            "embedding_rank": embedding_rank,
            "similarity_score": item.get("similarity_score"),
        }
        all_embedding_results.append(base)
        if embedding_rank <= selected_k:
            results.append(
                {
                    "content": chunk.get("content", ""),
                    "score": float(item.get("similarity_score", 0.0)),
                    "metadata": {**base, "rerank_rank": None, "rerank_score": None},
                }
            )

    metadata.pop("all_rerank_results", None)
    metadata.update(
        {
            "formatted_context": format_context(results, conv_index.get("turn_context")),
            "top_k": len(results),
            # Keep this field for schema compatibility: it remains the final
            # selection width, not evidence that a reranker was invoked.
            "rerank_top_k": selected_k,
            "reranker_enabled": False,
            "retrieval_mode": "embedding_only",
            "all_embedding_results": all_embedding_results,
            "embedding_only_selection": {
                "mode": "fixed_top_k",
                "selected_count": len(results),
                "top_k": selected_k,
            },
        }
    )
    rebuilt = dict(record)
    rebuilt["results"] = results
    rebuilt["retrieval_metadata"] = metadata
    return rebuilt


def source_run_dirs(results_root: Path, source_suffix: str) -> list[Path]:
    suffix = f"-{source_suffix}"
    return sorted(
        path
        for path in results_root.glob("memoryagentbench-rag-*")
        if path.is_dir() and path.name.endswith(suffix) and (path / "search_results.json").exists()
    )


def build_run(source_dir: Path, destination_dir: Path, top_k: int | None, dry_run: bool) -> dict[str, Any]:
    records = load_json(source_dir / "search_results.json")
    index = load_json(source_dir / "rag_index.json")
    lookup_by_conversation = build_chunk_lookup(index)
    rebuilt_records = []
    for record in records:
        conversation_id = record.get("conversation_id")
        conv_index = index.get("conversations", {}).get(conversation_id)
        if conv_index is None:
            raise KeyError(f"conversation {conversation_id!r} missing from rag_index.json")
        rebuilt_records.append(
            rebuild_record_embedding_only(
                record, conv_index, lookup_by_conversation.get(conversation_id, {}), top_k
            )
        )

    selected_counts = [len(record["results"]) for record in rebuilt_records]
    manifest = {
        "source_run": source_dir.name,
        "retrieval_mode": "embedding_only",
        "reranker_enabled": False,
        "selection_top_k_override": top_k,
        "questions": len(rebuilt_records),
        "selected_count_min": min(selected_counts, default=0),
        "selected_count_max": max(selected_counts, default=0),
        "rag_index": "hardlink to source rag_index.json",
    }
    if not dry_run:
        if destination_dir.exists():
            raise FileExistsError(f"destination already exists: {destination_dir}")
        destination_dir.mkdir(parents=True)
        save_json(destination_dir / "search_results.json", rebuilt_records)
        os.link(source_dir / "rag_index.json", destination_dir / "rag_index.json")
        save_json(destination_dir / "embedding_only_rewrite_meta.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--source-suffix", default=DEFAULT_SOURCE_SUFFIX)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--top-k", type=int, help="Override each source run's final top-k")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")

    root = args.results_root.resolve()
    sources = source_run_dirs(root, args.source_suffix)
    if not sources:
        raise FileNotFoundError(f"no source runs ending in -{args.source_suffix} under {root}")
    manifests = []
    for source in sources:
        destination = source.with_name(
            f"{source.name.removesuffix('-' + args.source_suffix)}-{args.run_name}"
        )
        manifest = build_run(source, destination, args.top_k, args.dry_run)
        manifests.append({"destination_run": destination.name, **manifest})
        print(f"{'Would create' if args.dry_run else 'Created'} {destination.name}")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
