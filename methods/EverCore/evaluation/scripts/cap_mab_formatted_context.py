"""Copy the selected MemoryAgentBench runs and cap ranked context by token budget.

Only persisted search output is changed. The longest prefix of complete ranked
entries that fits the budget is kept. If the first entry alone is too long, its
text is truncated to the budget so every non-empty source context retains some
top-ranked information.
Downstream artifacts are copied only when every question in a source is already
within budget; otherwise answer/evaluate must be rerun for that destination.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import tiktoken


DEFAULT_BUDGET = 5000
RUN_SUFFIX = "token-budget-5000-min1-best-20260713"

# These directories reproduce the per-source scores in methods/results.md.
RUN_VARIANTS = {
    "rag": {
        "default": "normal-recover-rerank",
    },
    "emem": {"default": "normal"},
    "mem0": {"default": "normal-simplified-prompt"},
    "evermemos": {"default": "normal-new-converter-label-atomic-context"},
    "memmachine": {"default": "normal"},
    "memos": {"default": "normal-simplified-prompt"},
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def result_segments(context: str, results: list[dict[str, Any]]) -> list[str]:
    """Split context into ranked entries using exact persisted result contents."""
    if not context:
        return []
    if not results:
        return [context]

    spans: list[tuple[int, int]] = []
    cursor = 0
    for result in results:
        content = str(result.get("content") or "")
        if not content:
            raise ValueError("ranked result has empty content")
        start = context.find(content, cursor)
        if start < 0:
            raise ValueError("ranked result content is not present in formatted_context")
        end = start + len(content)
        spans.append((start, end))
        cursor = end

    segments: list[str] = []
    entry_start = 0
    for index, (_, content_end) in enumerate(spans):
        entry_end = spans[index + 1][0] if index + 1 < len(spans) else len(context)
        # Everything between two contents (separator plus the next header) belongs
        # to the next ranked entry. Find its last paragraph boundary when possible.
        if index + 1 < len(spans):
            between = context[content_end : spans[index + 1][0]]
            boundary = between.rfind("\n\n")
            if boundary >= 0:
                entry_end = content_end + boundary + 2
            else:
                entry_end = content_end
        segments.append(context[entry_start:entry_end])
        entry_start = entry_end
    return segments


def cap_record(record: dict[str, Any], encoding: Any, budget: int) -> tuple[dict[str, Any], dict[str, int]]:
    copied = dict(record)
    metadata = dict(record.get("retrieval_metadata") or {})
    context = str(metadata.get("formatted_context") or "")
    results = list(record.get("results") or [])
    original_token_ids = encoding.encode(context)
    original_tokens = len(original_token_ids)
    segments = result_segments(context, results)

    kept: list[str] = []
    capped_token_ids: list[int] = []
    first_item_truncated = False
    for segment in segments:
        candidate = "".join([*kept, segment])
        candidate_tokens = encoding.encode(candidate)
        if len(candidate_tokens) > budget:
            if not kept and segment:
                kept.append(encoding.decode(candidate_tokens[:budget]))
                capped_token_ids = candidate_tokens[:budget]
                first_item_truncated = True
            break
        kept.append(segment)
        capped_token_ids = candidate_tokens
    capped_context = "".join(kept)
    capped_tokens = len(capped_token_ids)
    kept_count = min(len(kept), len(results))

    metadata["formatted_context"] = capped_context
    metadata["context_token_budget"] = budget
    metadata["context_tokenizer"] = "o200k_base"
    metadata["context_tokens_before_cap"] = original_tokens
    metadata["context_tokens_after_cap"] = capped_tokens
    metadata["context_ranked_items_before_cap"] = len(results)
    metadata["context_ranked_items_after_cap"] = kept_count
    metadata["context_first_item_truncated"] = first_item_truncated
    metadata["context_was_capped"] = capped_context != context
    copied["results"] = results[:kept_count]
    copied["retrieval_metadata"] = metadata
    return copied, {
        "before": original_tokens,
        "after": capped_tokens,
        "items_before": len(results),
        "items_after": kept_count,
        "empty_after": int(bool(context) and not capped_context),
        "first_item_truncated": int(first_item_truncated),
        "capped": int(capped_context != context),
    }


def parse_run_name(name: str, system: str) -> tuple[str, str]:
    prefix = f"memoryagentbench-{system}-"
    rest = name.removeprefix(prefix)
    split, source_and_variant = rest.split("-", 1)
    variants = RUN_VARIANTS[system]
    possible = sorted(set(variants.values()), key=len, reverse=True)
    for variant in possible:
        suffix = f"-{variant}"
        if source_and_variant.endswith(suffix):
            return split, source_and_variant[: -len(suffix)]
    raise ValueError(f"cannot parse selected run directory: {name}")


def selected_sources(results_root: Path) -> list[tuple[str, str, str, Path]]:
    selected: list[tuple[str, str, str, Path]] = []
    for system, variants in RUN_VARIANTS.items():
        for source_dir in results_root.glob(f"memoryagentbench-{system}-*"):
            if not (source_dir / "search_results.json").exists():
                continue
            variant = variants.get("default")
            for source, special_variant in variants.items():
                if source != "default" and f"-{source}-{special_variant}" in source_dir.name:
                    variant = special_variant
                    break
            if variant and source_dir.name.endswith(f"-{variant}"):
                split, source = parse_run_name(source_dir.name, system)
                expected = variants.get(source, variants["default"])
                if variant == expected:
                    selected.append((system, split, source, source_dir))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--run-suffix", default=RUN_SUFFIX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.results_root.resolve()
    encoding = tiktoken.get_encoding("o200k_base")
    runs = selected_sources(root)
    if len(runs) != 84:
        raise RuntimeError(f"expected 84 selected source runs (6 x 14), found {len(runs)}")

    summary: list[dict[str, Any]] = []
    for system, split, source, src in runs:
        records = load_json(src / "search_results.json")
        capped_records = []
        stats = []
        for record in records:
            capped, item_stats = cap_record(record, encoding, args.budget)
            capped_records.append(capped)
            stats.append(item_stats)

        dst = root / f"memoryagentbench-{system}-{split}-{source}-{args.run_suffix}"
        capped_questions = sum(item["capped"] for item in stats)
        row = {
            "system": system,
            "split": split,
            "source": source,
            "source_dir": src.name,
            "output_dir": dst.name,
            "questions": len(records),
            "capped_questions": capped_questions,
            "empty_context_questions": sum(item["empty_after"] for item in stats),
            "first_item_truncated_questions": sum(
                item["first_item_truncated"] for item in stats
            ),
            "max_tokens_before": max(item["before"] for item in stats),
            "max_tokens_after": max(item["after"] for item in stats),
            "needs_answer_evaluate": bool(capped_questions),
        }
        summary.append(row)
        if args.dry_run:
            continue
        if dst.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {dst}")
        dst.mkdir(parents=True)
        save_json(dst / "search_results.json", capped_records)
        # Unchanged sources can retain their existing downstream result. A capped
        # source intentionally has no downstream files, making rerun scope explicit.
        if not capped_questions:
            for filename in ("answer_results.json", "eval_results.json", "report.txt"):
                path = src / filename
                if path.exists():
                    shutil.copy2(path, dst / filename)
        save_json(dst / "context_cap_summary.json", row)

    summary_path = root / f"memoryagentbench_context_cap_{args.budget}_{args.run_suffix}.json"
    if not args.dry_run:
        save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"selected_runs={len(summary)} capped_sources={sum(r['needs_answer_evaluate'] for r in summary)}")
    if not args.dry_run:
        print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
