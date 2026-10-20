"""Rewrite saved RAG search results using offline dynamic top-k selection.

This script reuses persisted ``all_rerank_results`` scores plus ``rag_index.json``
to rebuild ``search_results.json`` without re-running embedding or rerank calls.
It is meant for temporary experiment workflows where Search artifacts already
exist and only Answer/Evaluate need to be re-run with a different selection rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_chunk_lookup(index_data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for conv_id, conv_index in index_data.get("conversations", {}).items():
        by_key: Dict[str, Dict[str, Any]] = {}
        for chunk in conv_index.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            chunk_index = chunk.get("chunk_index")
            if chunk_id:
                by_key[str(chunk_id)] = chunk
            if chunk_index is not None:
                by_key[f"chunk_index:{chunk_index}"] = chunk
        lookup[conv_id] = by_key
    return lookup


def turns_overlapping_range(
    start_token: int,
    end_token: int,
    turn_spans: List[Dict[str, Any]],
) -> List[int]:
    overlapping: List[int] = []
    for idx, span in enumerate(turn_spans):
        if isinstance(span, dict):
            span_start = int(span.get("start_token", 0))
            span_end = int(span.get("end_token", 0))
        else:
            span_start = int(span[0])
            span_end = int(span[1])
        if span_end <= start_token:
            continue
        if span_start >= end_token:
            break
        overlapping.append(idx)
    return overlapping


def expand_chunk_to_turns(chunk: Dict[str, Any], turn_context: Optional[Dict[str, Any]]) -> str:
    if not turn_context:
        return chunk.get("content", "")

    header = turn_context.get("header", "")
    turn_lines = turn_context.get("turn_lines", [])
    turn_spans = turn_context.get("turn_spans", [])
    start = chunk.get("start_token")
    end = chunk.get("end_token")
    if start is None or end is None:
        return chunk.get("content", "")

    overlapping = turns_overlapping_range(int(start), int(end), turn_spans)
    if not overlapping:
        return chunk.get("content", "")

    first_turn = overlapping[0]
    last_turn = overlapping[-1]
    parts: List[str] = []
    if chunk.get("chunk_index", 0) == 0 or first_turn == 0:
        parts.extend([header, ""])
    parts.extend(turn_lines[first_turn : last_turn + 1])
    return "\n".join(parts)


def format_context(
    results: List[Dict[str, Any]],
    turn_context: Optional[Dict[str, Any]],
) -> str:
    if not results:
        return "(No relevant memories found)"

    parts: List[str] = []
    for idx, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        chunk = {
            "content": result.get("content", ""),
            "chunk_index": metadata.get("chunk_index"),
            "start_token": metadata.get("start_token"),
            "end_token": metadata.get("end_token"),
        }
        display_content = expand_chunk_to_turns(chunk, turn_context)
        parts.append(
            f"Memory {idx} "
            f"(chunk={metadata.get('chunk_index')}, "
            f"similarity={float(metadata.get('similarity_score', 0.0)):.6f}, "
            f"rerank={metadata.get('rerank_score')}):\n"
            f"{display_content}"
        )
    return "\n\n".join(parts)


def get_rank_score(item: Dict[str, Any]) -> float:
    score = item.get("rerank_score")
    if score is None:
        score = item.get("similarity_score", 0.0)
    return float(score)


def select_topk(
    all_rerank_results: List[Dict[str, Any]],
    fixed_top_k: Optional[int],
    score_threshold: Optional[float],
    fallback_top_k: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if fixed_top_k is not None:
        selected = all_rerank_results[:fixed_top_k]
        return selected, {
            "enabled": False,
            "mode": "fixed_top_k",
            "score_threshold": None,
            "fallback_top_k": fixed_top_k,
            "selected_count": len(selected),
        }

    if score_threshold is None or fallback_top_k is None:
        raise ValueError("Dynamic selection requires score_threshold and fallback_top_k")

    threshold_selected = [
        item
        for item in all_rerank_results
        if get_rank_score(item) >= score_threshold
    ]
    if len(threshold_selected) >= fallback_top_k:
        return threshold_selected, {
            "enabled": True,
            "mode": "score_threshold",
            "score_threshold": score_threshold,
            "fallback_top_k": fallback_top_k,
            "selected_count": len(threshold_selected),
        }

    selected = all_rerank_results[:fallback_top_k]
    return selected, {
        "enabled": True,
        "mode": "fallback_top_k",
        "score_threshold": score_threshold,
        "fallback_top_k": fallback_top_k,
        "selected_count": len(selected),
        "threshold_selected_count": len(threshold_selected),
    }


def build_results_for_record(
    record: Dict[str, Any],
    conv_index: Dict[str, Any],
    chunk_lookup: Dict[str, Dict[str, Any]],
    fixed_top_k: Optional[int],
    score_threshold: Optional[float],
    fallback_top_k: Optional[int],
) -> Dict[str, Any]:
    metadata = record.get("retrieval_metadata", {})
    all_rerank_results = metadata.get("all_rerank_results")
    if not all_rerank_results:
        raise ValueError(
            f"Missing retrieval_metadata.all_rerank_results for query={record.get('query')!r} "
            f"in conversation={record.get('conversation_id')!r}"
        )

    selected_items, selection_metadata = select_topk(
        all_rerank_results=all_rerank_results,
        fixed_top_k=fixed_top_k,
        score_threshold=score_threshold,
        fallback_top_k=fallback_top_k,
    )

    rebuilt_results: List[Dict[str, Any]] = []
    for new_rank, item in enumerate(selected_items, 1):
        chunk = None
        chunk_id = item.get("chunk_id")
        if chunk_id is not None:
            chunk = chunk_lookup.get(str(chunk_id))
        if chunk is None and item.get("chunk_index") is not None:
            chunk = chunk_lookup.get(f"chunk_index:{item.get('chunk_index')}")
        if chunk is None:
            raise KeyError(
                f"Unable to locate chunk for conversation={record.get('conversation_id')!r}, "
                f"chunk_id={chunk_id!r}, chunk_index={item.get('chunk_index')!r}"
            )

        rebuilt_results.append(
            {
                "content": chunk.get("content", ""),
                "score": get_rank_score(item),
                "metadata": {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "start_turn": chunk.get("start_turn"),
                    "end_turn": chunk.get("end_turn"),
                    "start_token": chunk.get("start_token"),
                    "end_token": chunk.get("end_token"),
                    "embedding_rank": item.get("embedding_rank"),
                    "rerank_rank": new_rank,
                    "similarity_score": item.get("similarity_score"),
                    "rerank_score": item.get("rerank_score"),
                },
            }
        )

    rebuilt_metadata = dict(metadata)
    rebuilt_metadata["formatted_context"] = format_context(
        rebuilt_results,
        conv_index.get("turn_context"),
    )
    rebuilt_metadata["top_k"] = len(rebuilt_results)
    rebuilt_metadata["rerank_top_k"] = len(rebuilt_results)
    rebuilt_metadata["dynamic_top_k"] = selection_metadata

    rebuilt_record = dict(record)
    rebuilt_record["results"] = rebuilt_results
    rebuilt_record["retrieval_metadata"] = rebuilt_metadata
    return rebuilt_record


def backup_if_exists(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup_path = path.with_name(path.name + ".bak")
    if backup_path.exists():
        backup_path.unlink()
    path.replace(backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Run output directory containing search_results.json and rag_index.json.",
    )
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--fixed-top-k",
        type=int,
        help="Select exactly the first K reranked chunks, matching normal rerank_top_k behavior.",
    )
    selection_group.add_argument(
        "--score-threshold",
        type=float,
        help="Keep all reranked chunks whose score is >= this threshold.",
    )
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        help="If too few chunks pass the threshold, fall back to top-k reranked chunks.",
    )
    parser.add_argument(
        "--purge-downstream",
        action="store_true",
        help="Back up answer/eval/report outputs so downstream stages are guaranteed to rerun cleanly.",
    )
    args = parser.parse_args()

    if args.fixed_top_k is not None and args.fixed_top_k <= 0:
        raise ValueError("--fixed-top-k must be positive")
    if args.score_threshold is not None and args.fallback_top_k is None:
        raise ValueError("--fallback-top-k is required with --score-threshold")
    if args.fallback_top_k is not None and args.fallback_top_k <= 0:
        raise ValueError("--fallback-top-k must be positive")

    output_dir = Path(args.output_dir).resolve()
    search_results_path = output_dir / "search_results.json"
    rag_index_path = output_dir / "rag_index.json"
    checkpoint_path = output_dir / "checkpoint_default.json"

    search_results = load_json(search_results_path)
    rag_index = load_json(rag_index_path)
    chunk_lookup_by_conv = build_chunk_lookup(rag_index)

    rewritten_records: List[Dict[str, Any]] = []
    mode_counts: Dict[str, int] = {"score_threshold": 0, "fallback_top_k": 0}
    selected_counts: List[int] = []

    for record in search_results:
        conv_id = record.get("conversation_id")
        conv_index = rag_index.get("conversations", {}).get(conv_id)
        if conv_index is None:
            raise KeyError(f"Conversation {conv_id!r} not found in rag_index.json")

        rewritten = build_results_for_record(
            record=record,
            conv_index=conv_index,
            chunk_lookup=chunk_lookup_by_conv.get(conv_id, {}),
            fixed_top_k=args.fixed_top_k,
            score_threshold=args.score_threshold,
            fallback_top_k=args.fallback_top_k,
        )
        dynamic_meta = rewritten["retrieval_metadata"].get("dynamic_top_k", {})
        mode = str(dynamic_meta.get("mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        selected_counts.append(int(dynamic_meta.get("selected_count", len(rewritten["results"]))))
        rewritten_records.append(rewritten)

    save_json(search_results_path, rewritten_records)

    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)
        checkpoint["search_results"] = rewritten_records
        completed = [stage for stage in checkpoint.get("completed_stages", []) if stage == "search"]
        checkpoint["completed_stages"] = completed or ["search"]
        checkpoint.pop("answer_results", None)
        checkpoint.pop("eval_results", None)
        save_json(checkpoint_path, checkpoint)

    purged_paths: List[str] = []
    if args.purge_downstream:
        for filename in ("answer_results.json", "eval_results.json", "report.txt"):
            backup = backup_if_exists(output_dir / filename)
            if backup is not None:
                purged_paths.append(backup.name)

    avg_selected = sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
    print(f"Rewrote {len(rewritten_records)} search records in: {search_results_path}")
    print(
        "Top-k rewrite summary: "
        f"fixed_top_k={args.fixed_top_k}, "
        f"threshold={args.score_threshold}, fallback_top_k={args.fallback_top_k}, "
        f"avg_selected={avg_selected:.2f}, "
        f"score_threshold_mode={mode_counts.get('score_threshold', 0)}, "
        f"fallback_mode={mode_counts.get('fallback_top_k', 0)}, "
        f"fixed_mode={mode_counts.get('fixed_top_k', 0)}"
    )
    if purged_paths:
        print("Backed up downstream outputs: " + ", ".join(purged_paths))


if __name__ == "__main__":
    main()
