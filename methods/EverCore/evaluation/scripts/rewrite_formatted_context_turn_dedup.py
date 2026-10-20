"""Rewrite formatted_context with turn-level dedup (display-only).

This script updates existing run artifacts in-place without re-running search/answer:
- search_results.json: retrieval_metadata.formatted_context
- answer_results.json: formatted_context

It preserves retrieval scores and answer text, and only rewrites display context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def backup_file(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        backup.unlink()
    path.replace(backup)
    return backup


def turns_overlapping_range(start: int, end: int, turn_spans: List[Any]) -> List[int]:
    return [
        idx
        for idx, span in enumerate(turn_spans)
        if int(span[0]) < end and int(span[1]) > start
    ]


def get_overlapping_turn_range(
    chunk: Dict[str, Any],
    turn_context: Optional[Dict[str, Any]],
) -> Optional[Tuple[int, int]]:
    if not turn_context:
        return None
    turn_spans = turn_context.get("turn_spans", [])
    if not turn_spans:
        return None
    start = chunk.get("start_token")
    end = chunk.get("end_token")
    if start is None or end is None:
        return None
    overlapping = turns_overlapping_range(int(start), int(end), turn_spans)
    if not overlapping:
        return None
    return overlapping[0], overlapping[-1]


def render_turn_range(
    first_turn: int,
    last_turn: int,
    turn_context: Dict[str, Any],
    include_header: bool,
    skip_turns: Optional[Set[int]] = None,
) -> str:
    turn_lines = turn_context.get("turn_lines", [])
    header = turn_context.get("header", "")
    if not turn_lines:
        return ""

    parts: List[str] = []
    if include_header:
        parts.extend([header, ""])

    turn_ids = range(first_turn, last_turn + 1)
    if skip_turns:
        turn_ids = [turn_idx for turn_idx in turn_ids if turn_idx not in skip_turns]
    parts.extend(turn_lines[turn_idx] for turn_idx in turn_ids)
    return "\n".join(parts)


def format_context_with_turn_dedup(
    results: List[Dict[str, Any]],
    turn_context: Optional[Dict[str, Any]],
) -> str:
    if not results:
        return "(No relevant memories found)"

    seen_turns: Set[int] = set()
    parts: List[str] = []
    memory_idx = 1

    for result in results:
        metadata = result.get("metadata", {})
        chunk = {
            "content": result.get("content", ""),
            "chunk_index": metadata.get("chunk_index"),
            "start_token": metadata.get("start_token"),
            "end_token": metadata.get("end_token"),
        }

        display_content = result.get("content", "")
        if turn_context:
            turn_range = get_overlapping_turn_range(chunk, turn_context)
            if turn_range is not None:
                first_turn, last_turn = turn_range
                display_content = render_turn_range(
                    first_turn=first_turn,
                    last_turn=last_turn,
                    turn_context=turn_context,
                    include_header=(chunk.get("chunk_index", 0) == 0 or first_turn == 0),
                    skip_turns=seen_turns,
                )
                seen_turns.update(range(first_turn, last_turn + 1))
                if not display_content.strip():
                    continue

        sim = float(metadata.get("similarity_score", 0.0))
        parts.append(
            f"Memory {memory_idx} "
            f"(chunk={metadata.get('chunk_index')}, "
            f"similarity={sim:.6f}, "
            f"rerank={metadata.get('rerank_score')}):\n"
            f"{display_content}"
        )
        memory_idx += 1

    return "\n\n".join(parts) if parts else "(No relevant memories found)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Run output dir containing search_results.json, answer_results.json, rag_index.json",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    search_path = output_dir / "search_results.json"
    answer_path = output_dir / "answer_results.json"
    index_path = output_dir / "rag_index.json"

    search_results = load_json(search_path)
    answer_results = load_json(answer_path)
    rag_index = load_json(index_path)

    conv_map = rag_index.get("conversations", {})
    rewritten_search = 0
    formatted_by_key: Dict[Tuple[str, str], str] = {}

    for record in search_results:
        conv_id = record.get("conversation_id")
        query = record.get("query", "")
        conv_index = conv_map.get(conv_id, {})
        turn_context = conv_index.get("turn_context")
        results = record.get("results", [])
        new_formatted = format_context_with_turn_dedup(results, turn_context)

        meta = dict(record.get("retrieval_metadata", {}))
        meta["formatted_context"] = new_formatted
        record["retrieval_metadata"] = meta
        formatted_by_key[(str(conv_id), str(query))] = new_formatted
        rewritten_search += 1

    rewritten_answer = 0
    missing = 0
    for record in answer_results:
        key = (str(record.get("conversation_id")), str(record.get("question", "")))
        new_formatted = formatted_by_key.get(key)
        if new_formatted is None:
            missing += 1
            continue
        record["formatted_context"] = new_formatted
        rewritten_answer += 1

    b1 = backup_file(search_path)
    b2 = backup_file(answer_path)
    save_json(search_path, search_results)
    save_json(answer_path, answer_results)

    print(f"Rewrote search formatted_context: {rewritten_search}")
    print(f"Rewrote answer formatted_context: {rewritten_answer}")
    print(f"Answer records without match: {missing}")
    if b1:
        print(f"Backup: {b1.name}")
    if b2:
        print(f"Backup: {b2.name}")


if __name__ == "__main__":
    main()

