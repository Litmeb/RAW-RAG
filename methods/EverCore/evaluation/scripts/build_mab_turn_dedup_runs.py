"""Build display-only exact-text-dedup RAG runs from saved MemoryAgentBench searches.

The source ``normal-recover-rerank`` runs have already completed Add and
Search.  This utility copies their search artifacts into new, immutable run
directories and changes only ``retrieval_metadata.formatted_context``.  It
removes sentence/paragraph units that are exactly duplicated (after whitespace
normalization) by a higher-ranked selected chunk.  Thus embedding retrieval,
reranking, selected chunks, and the RAG index are identical to the standard
baseline; only the reader's duplicate overlap is ablated.

Destination runs deliberately omit answer/evaluation outputs.  Run Answer and
Evaluate with the destination run name after this script completes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_SUFFIX = "normal-recover-rerank"
DEFAULT_RUN_NAME = "turn-dedup-topk15-20260815"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalized_unit(unit: str) -> str:
    return re.sub(r"\s+", " ", unit).strip().casefold()


def text_units(content: str) -> list[str]:
    """Split conservatively without reordering or reconstructing source text."""
    return [unit for unit in re.split(r"(?<=[.!?])\s+|\n{2,}", content) if unit.strip()]


def render_deduplicated_context(
    results: list[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Format selected chunks while removing exact prior text units."""
    if not results:
        return "(No relevant memories found)", {
            "displayed_memories": 0,
            "unique_text_units": 0,
            "suppressed_text_units": 0,
            "fully_suppressed_memories": 0,
        }

    seen_units: set[str] = set()
    parts: list[str] = []
    suppressed_text_units = 0
    fully_suppressed_memories = 0

    for original_rank, result in enumerate(results, 1):
        kept_units: list[str] = []
        for unit in text_units(str(result.get("content", ""))):
            key = normalized_unit(unit)
            if key and key in seen_units:
                suppressed_text_units += 1
                continue
            if key:
                seen_units.add(key)
            kept_units.append(unit)
        if not kept_units:
            fully_suppressed_memories += 1
            continue
        display_content = "\n\n".join(kept_units)
        parts.append(_format_memory(original_rank, result, display_content))

    return "\n\n".join(parts) if parts else "(No relevant memories found)", {
        "displayed_memories": len(parts),
        "unique_text_units": len(seen_units),
        "suppressed_text_units": suppressed_text_units,
        "fully_suppressed_memories": fully_suppressed_memories,
    }


def _format_memory(rank: int, result: dict[str, Any], content: str) -> str:
    metadata = result.get("metadata") or {}
    similarity = float(metadata.get("similarity_score", 0.0))
    return (
        f"Memory {rank} (chunk={metadata.get('chunk_index')}, "
        f"similarity={similarity:.6f}, rerank={metadata.get('rerank_score')}):\n{content}"
    )


def build_run(source_dir: Path, destination_dir: Path, dry_run: bool) -> dict[str, Any]:
    records = load_json(source_dir / "search_results.json")
    rewritten: list[dict[str, Any]] = []
    totals = {"displayed_memories": 0, "unique_text_units": 0, "suppressed_text_units": 0, "fully_suppressed_memories": 0}

    for record in records:
        context, stats = render_deduplicated_context(list(record.get("results") or []))
        metadata = dict(record.get("retrieval_metadata") or {})
        metadata["formatted_context"] = context
        metadata["display_deduplication"] = {"enabled": True, "unit": "complete_turn", **stats}
        rewritten.append({**record, "retrieval_metadata": metadata})
        for key, value in stats.items():
            totals[key] += value

    manifest = {
        "source_run": source_dir.name,
        "ablation": "exact_text_dedup_display_only",
        "deduplication": "A sentence/paragraph is shown once when its normalized text exactly matches an earlier selected chunk; retrieval results are unchanged.",
        "questions": len(rewritten),
        **totals,
        "rag_index": "hardlink to source rag_index.json",
        "downstream_outputs": "intentionally omitted; run answer/evaluate on this destination",
    }
    if not dry_run:
        if destination_dir.exists():
            raise FileExistsError(f"refusing to overwrite destination: {destination_dir}")
        destination_dir.mkdir(parents=True)
        save_json(destination_dir / "search_results.json", rewritten)
        os.link(source_dir / "rag_index.json", destination_dir / "rag_index.json")
        save_json(destination_dir / "turn_dedup_rewrite_meta.json", manifest)
    return manifest


def source_run_dirs(results_root: Path, source_suffix: str) -> list[Path]:
    suffix = f"-{source_suffix}"
    return sorted(
        path
        for path in results_root.glob("memoryagentbench-rag-*")
        if path.is_dir() and path.name.endswith(suffix) and (path / "search_results.json").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--source-suffix", default=DEFAULT_SOURCE_SUFFIX)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    sources = source_run_dirs(results_root, args.source_suffix)
    if not sources:
        raise FileNotFoundError(f"no source runs ending in -{args.source_suffix} under {results_root}")

    manifests = []
    for source in sources:
        destination = source.with_name(
            f"{source.name.removesuffix('-' + args.source_suffix)}-{args.run_name}"
        )
        manifest = build_run(source, destination, args.dry_run)
        manifests.append({"destination_run": destination.name, **manifest})
        print(f"{'Would create' if args.dry_run else 'Created'} {destination.name}")
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
