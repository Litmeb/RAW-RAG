"""Prepare offline 5k RAG controls without retrieval, reader, or judge calls.

Keep in-budget formatted evidence verbatim. For an over-budget question, first
replace ALL expanded entries with their saved raw chunks in the existing order.
Only if that still exceeds the budget, retain the longest complete-entry prefix;
an oversized first entry alone is token-prefix clipped. Never skip entries,
deduplicate, reorder, or fill unused space. The budget excludes the outer prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import tiktoken


DEFAULT_RUN_NAME = "5k-unexpand-first-20260920"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
RUNS = {
    "locomo": {
        "source": "locomo-rag-fixed-settings",
        "questions": 1540,
        "answer_model": "gpt-4o-mini",
    },
    "longmemeval": {
        "source": "longmemeval-rag-256-32-128-15-turn-expand-time-filter-gpt4.1mini",
        "questions": 500,
        "answer_model": "gpt-4.1-mini",
    },
    "meme": {
        "source": "meme-rag-normal-15",
        "questions": 1188,
        "answer_model": "gpt-4o-mini",
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def token_count(text: str, encoding: Any) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def raw_entry(rank: int, result: dict[str, Any]) -> str:
    """Match RAGAdapter's non-expanded display, including score precision."""
    metadata = result["metadata"]
    content = result["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("Selected RAG chunk has no raw content")
    return (
        f"Memory {rank} (chunk={metadata.get('chunk_index')}, "
        f"similarity={metadata['similarity_score']:.6f}, "
        f"rerank={metadata.get('rerank_score')}):\n{content}"
    )


def pack_prefix(entries: list[str], encoding: Any, budget: int) -> tuple[str, int, bool]:
    kept: list[str] = []
    for entry in entries:
        candidate = "\n\n".join([*kept, entry])
        if token_count(candidate, encoding) <= budget:
            kept.append(entry)
            continue
        if kept:
            break
        tokens = encoding.encode(entry, disallowed_special=())[:budget]
        # A token boundary can bisect UTF-8; omit the incomplete character.
        clipped = encoding.decode(tokens, errors="ignore")
        while token_count(clipped, encoding) > budget:
            tokens.pop()
            clipped = encoding.decode(tokens, errors="ignore")
        return clipped, 1, True
    return "\n\n".join(kept), len(kept), False


def rewrite_record(
    record: dict[str, Any], encoding: Any, budget: int = 5000
) -> tuple[dict[str, Any], dict[str, Any]]:
    if budget < 1:
        raise ValueError("Token budget must be positive")
    metadata = dict(record["retrieval_metadata"])
    original = metadata["formatted_context"]
    if not isinstance(original, str) or not original:
        raise ValueError("Expected a nonempty saved formatted_context")
    results = record["results"]
    before = token_count(original, encoding)
    context = original
    kept_count = len(results)
    unexpanded_tokens = None
    first_clipped = False
    if before > budget:
        if not results:
            raise ValueError("Cannot recover raw chunks for over-budget evidence")
        entries = [raw_entry(rank, result) for rank, result in enumerate(results, 1)]
        context = "\n\n".join(entries)
        unexpanded_tokens = token_count(context, encoding)
        if unexpanded_tokens > budget:
            context, kept_count, first_clipped = pack_prefix(entries, encoding, budget)

    after = token_count(context, encoding)
    if not context or after > budget:
        raise ValueError("Intervention produced empty or over-budget evidence")
    stats = {
        "tokenizer": "o200k_base",
        "budget": budget,
        "tokens_before": before,
        "tokens_after_unexpansion": unexpanded_tokens,
        "tokens_after": after,
        "turn_expansion_disabled": before > budget,
        "prefix_clipped": unexpanded_tokens is not None and unexpanded_tokens > budget,
        "first_entry_clipped": first_clipped,
        "entries_before": len(results),
        "entries_after": kept_count,
        "context_changed": context != original,
    }
    metadata["formatted_context"] = context
    metadata["context_budget_intervention"] = stats
    # Preserve the selected chunks and candidate ranking as retrieval provenance.
    # The answer fast path uses formatted_context, not results, for reader input.
    return {**record, "retrieval_metadata": metadata}, stats


def build_run(
    root: Path, dataset: str, run_name: str, budget: int, encoding: Any,
    dry_run: bool = False, source_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = source_spec if source_spec is not None else RUNS[dataset]
    source = root / spec["source"]
    destination = root / f"{dataset}-rag-{run_name}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    search_path = source / "search_results.json"
    source_hash = file_hash(search_path)
    records = load_json(search_path)
    if len(records) != spec["questions"]:
        raise ValueError(f"{source}: expected {spec['questions']} questions, got {len(records)}")
    if len({record["question_id"] for record in records}) != len(records):
        raise ValueError(f"{source}: duplicate question IDs")

    # Verify that these searches are the actual main-run reader inputs.
    answers = load_json(source / "answer_results.json")
    answer_contexts = {row["question_id"]: row["formatted_context"] for row in answers}
    if len(answers) != len(records) or set(answer_contexts) != {
        row["question_id"] for row in records
    }:
        raise ValueError(f"{source}: source answer/search question sets differ")
    for row in records:
        if answer_contexts[row["question_id"]] != row["retrieval_metadata"]["formatted_context"]:
            raise ValueError(f"{source}: source answer/search context differs")
    del answers, answer_contexts

    rewritten, stats = [], []
    for record in records:
        updated, audit = rewrite_record(record, encoding, budget)
        rewritten.append(updated)
        stats.append(audit)
    source_eval = load_json(source / "eval_results.json")
    manifest = {
        "dataset": dataset,
        "system": "rag",
        "run_name": run_name,
        "source_run": str(source.resolve()),
        "output_dir": destination.name,
        "source_search_sha256": source_hash,
        "answer_model": spec["answer_model"],
        "judge_model": source_eval["metadata"]["model"],
        "judge_num_runs": source_eval["metadata"]["num_runs"],
        "questions": len(records),
        "tokenizer": "o200k_base",
        "budget": budget,
        "policy": "keep_if_fits_else_unexpand_all_then_complete_entry_prefix_min1",
        "order": "unchanged saved display order (chronological for MEME)",
        "results": "unchanged retrieval provenance; reader uses formatted_context",
        "changed_questions": sum(s["context_changed"] for s in stats),
        "unexpanded_questions": sum(s["turn_expansion_disabled"] for s in stats),
        "prefix_clipped_questions": sum(s["prefix_clipped"] for s in stats),
        "first_entry_clipped_questions": sum(s["first_entry_clipped"] for s in stats),
        "max_tokens_before": max(s["tokens_before"] for s in stats),
        "max_tokens_after": max(s["tokens_after"] for s in stats),
        "mean_tokens_before": sum(s["tokens_before"] for s in stats) / len(stats),
        "mean_tokens_after": sum(s["tokens_after"] for s in stats) / len(stats),
        "downstream_outputs": "omitted; rerun answer/evaluate for all questions",
    }
    if file_hash(search_path) != source_hash:
        raise RuntimeError(f"Source changed during preparation: {search_path}")
    if not dry_run:
        destination.mkdir()
        save_json(destination / "search_results.json", rewritten)
        manifest["output_search_sha256"] = file_hash(destination / "search_results.json")
        save_json(destination / "context_cap_summary.json", manifest)
    return manifest


def load_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    path = path.resolve()
    specs = load_json(path)
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Source manifest must map dataset names to run specifications")
    resolved = {}
    for dataset, spec in specs.items():
        if dataset not in RUNS or not isinstance(spec, dict):
            raise ValueError(f"Unsupported dataset specification: {dataset}")
        if not isinstance(spec.get("source"), str) or not spec["source"].strip():
            raise ValueError(f"{dataset}: source must be a nonempty path")
        if type(spec.get("questions")) is not int or spec["questions"] < 1:
            raise ValueError(f"{dataset}: questions must be a positive expected count")
        if not isinstance(spec.get("answer_model"), str) or not spec["answer_model"].strip():
            raise ValueError(f"{dataset}: answer_model is required")
        source = Path(spec["source"])
        if not source.is_absolute():
            source = path.parent / source
        resolved[dataset] = {**spec, "source": str(source.resolve())}
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-manifest", type=Path)
    source_group.add_argument("--historical-sources", action="store_true",
                              help="Use unpublished historical run names; requires those artifacts.")
    parser.add_argument("--dataset", choices=tuple(RUNS), action="append")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_name):
        parser.error("--run-name must contain only letters, digits, hyphens, or underscores")
    root = args.results_root.resolve()
    specs = load_source_manifest(args.source_manifest) if args.source_manifest else RUNS
    selected = args.dataset or list(specs)
    if any(dataset not in specs for dataset in selected):
        parser.error("Each selected dataset must have a source manifest entry")
    summary_path = root / f"rag_context_cap_{args.run_name}.json"
    for path in [summary_path, *(root / f"{dataset}-rag-{args.run_name}" for dataset in selected)]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    for spec in (specs[dataset] for dataset in selected):
        for filename in ("search_results.json", "answer_results.json", "eval_results.json"):
            path = root / spec["source"] / filename
            if not path.is_file():
                raise FileNotFoundError(path)
    encoding = tiktoken.get_encoding("o200k_base")
    summaries = []
    root.mkdir(parents=True, exist_ok=True)
    for dataset in selected:
        summary = build_run(root, dataset, args.run_name, args.budget, encoding,
                            args.dry_run, source_spec=specs[dataset])
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    if not args.dry_run:
        save_json(summary_path, summaries)
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
