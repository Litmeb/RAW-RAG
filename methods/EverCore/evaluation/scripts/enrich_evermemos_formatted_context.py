"""Create a reused EverMemos run with atomic facts in formatted_context.

The script does not rerun add or search. It reads an existing EverMemos result
directory, looks up retrieved memcells, rewrites search_results.json with a
formatted_context that includes episode text plus atomic facts, and writes a
new result directory suitable for rerunning only answer/evaluate.

Example:
    python evaluation/scripts/enrich_evermemos_formatted_context.py \
      --src-run-dir evaluation/results/memoryagentbench-evermemos-...-old \
      --dst-run-dir evaluation/results/memoryagentbench-evermemos-...-atomic-context

Then run:
    python -m evaluation.cli --dataset memoryagentbench --system evermemos \
      --mab-split Conflict_Resolution --mab-source factconsolidation_sh_262k \
      --output-dir evaluation/results/memoryagentbench-evermemos-...-atomic-context \
      --stages answer evaluate
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def iter_memcell_files(run_dir: Path) -> Iterable[Path]:
    memcells_dir = run_dir / "memcells"
    if not memcells_dir.exists():
        return []
    return sorted(memcells_dir.glob("memcell_list_*.json"))


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def atomic_fact_strings(memcell: Optional[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not memcell:
        return "", []
    atomic = memcell.get("atomic_fact") or {}
    if not isinstance(atomic, dict):
        return "", []

    time_text = normalize_text(atomic.get("time"))
    facts = atomic.get("atomic_fact") or []
    if not isinstance(facts, list):
        facts = [facts]

    rendered: List[str] = []
    for fact in facts:
        if isinstance(fact, dict):
            text = normalize_text(fact.get("fact") or fact.get("content"))
        else:
            text = normalize_text(fact)
        if text:
            rendered.append(text)
    return time_text, rendered


def original_data_text(memcell: Optional[Dict[str, Any]], max_chars: int) -> str:
    if not memcell:
        return ""
    pieces: List[str] = []
    for item in memcell.get("original_data") or []:
        message = (item or {}).get("message") or {}
        content = normalize_text(message.get("content"))
        if content:
            pieces.append(content)
    text = "\n\n".join(pieces)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def build_memcell_index(
    run_dir: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    by_subject_episode: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_subject: Dict[str, List[Dict[str, Any]]] = {}

    for path in iter_memcell_files(run_dir):
        data = load_json(path)
        if not isinstance(data, list):
            continue
        for memcell in data:
            if not isinstance(memcell, dict):
                continue
            subject = normalize_text(memcell.get("subject"))
            episode = normalize_text(memcell.get("episode"))
            if subject:
                by_subject.setdefault(subject, []).append(memcell)
            if subject and episode:
                by_subject_episode[(subject, episode)] = memcell

    return by_subject_episode, by_subject


def find_memcell(
    result: Dict[str, Any],
    by_subject_episode: Dict[Tuple[str, str], Dict[str, Any]],
    by_subject: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    metadata = result.get("metadata") or {}
    subject = normalize_text(metadata.get("subject"))
    content = normalize_text(result.get("content"))
    if subject and content:
        exact = by_subject_episode.get((subject, content))
        if exact:
            return exact

    candidates = by_subject.get(subject, [])
    if not candidates:
        return None

    summary = normalize_text(metadata.get("summary"))
    if summary:
        for memcell in candidates:
            if normalize_text(memcell.get("summary")).startswith(summary):
                return memcell

    return candidates[0]


def format_doc(
    index: int,
    result: Dict[str, Any],
    memcell: Optional[Dict[str, Any]],
    max_atomic_facts: int,
    include_original_data: bool,
    max_original_chars: int,
) -> Tuple[str, bool, int]:
    metadata = result.get("metadata") or {}
    subject = normalize_text(metadata.get("subject")) or "N/A"
    episode = normalize_text(result.get("content"))
    score = result.get("score")

    lines = [f"Memory {index}:", f"Title: {subject}"]
    if score is not None:
        try:
            lines.append(f"Score: {float(score):.6f}")
        except (TypeError, ValueError):
            lines.append(f"Score: {score}")

    time_text, facts = atomic_fact_strings(memcell)
    if time_text:
        lines.append(f"Atomic fact time: {time_text}")

    if episode:
        lines.extend(["Episode summary:", episode])

    included_facts = facts
    if max_atomic_facts > 0:
        included_facts = facts[:max_atomic_facts]

    if included_facts:
        lines.append("Atomic facts:")
        lines.extend(f"- {fact}" for fact in included_facts)
        if max_atomic_facts > 0 and len(facts) > max_atomic_facts:
            lines.append(f"- ... {len(facts) - max_atomic_facts} more atomic facts omitted")

    if include_original_data:
        raw_text = original_data_text(memcell, max_original_chars)
        if raw_text:
            lines.extend(["Original data:", raw_text])

    return "\n".join(lines), bool(included_facts), len(facts)


def rewrite_search_results(
    search_results: List[Dict[str, Any]],
    by_subject_episode: Dict[Tuple[str, str], Dict[str, Any]],
    by_subject: Dict[str, List[Dict[str, Any]]],
    max_atomic_facts: int,
    include_original_data: bool,
    max_original_chars: int,
) -> Dict[str, int]:
    stats = {
        "questions": 0,
        "docs": 0,
        "docs_matched_to_memcell": 0,
        "docs_with_atomic_facts": 0,
        "atomic_facts_available": 0,
    }

    for record in search_results:
        stats["questions"] += 1
        rendered_docs: List[str] = []

        for doc_index, result in enumerate(record.get("results") or [], start=1):
            stats["docs"] += 1
            memcell = find_memcell(result, by_subject_episode, by_subject)
            if memcell:
                stats["docs_matched_to_memcell"] += 1
            rendered, has_atomic, fact_count = format_doc(
                index=doc_index,
                result=result,
                memcell=memcell,
                max_atomic_facts=max_atomic_facts,
                include_original_data=include_original_data,
                max_original_chars=max_original_chars,
            )
            if has_atomic:
                stats["docs_with_atomic_facts"] += 1
            stats["atomic_facts_available"] += fact_count
            rendered_docs.append(rendered)

        context = "\n\n---\n\n".join(rendered_docs)
        metadata = dict(record.get("retrieval_metadata") or {})
        metadata["formatted_context"] = context
        metadata["formatted_context_source"] = "episode_plus_atomic_facts"
        metadata["formatted_context_include_original_data"] = include_original_data
        metadata["formatted_context_max_atomic_facts_per_doc"] = max_atomic_facts
        record["retrieval_metadata"] = metadata

    return stats


def rewrite_answer_contexts(
    answer_results: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> int:
    by_qid = {
        record.get("question_id"): (record.get("retrieval_metadata") or {}).get(
            "formatted_context", ""
        )
        for record in search_results
    }
    rewritten = 0
    for answer in answer_results:
        context = by_qid.get(answer.get("question_id"))
        if context:
            answer["formatted_context"] = context
            rewritten += 1
    return rewritten


def copy_optional(src_dir: Path, dst_dir: Path, filename: str) -> bool:
    src = src_dir / filename
    if not src.exists():
        return False
    shutil.copy2(src, dst_dir / filename)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-run-dir", required=True, help="Existing run directory")
    parser.add_argument("--dst-run-dir", required=True, help="New run directory to create")
    parser.add_argument(
        "--max-atomic-facts-per-doc",
        type=int,
        default=0,
        help="0 means include all atomic facts",
    )
    parser.add_argument(
        "--include-original-data",
        action="store_true",
        help="Also append raw original messages from each matched memcell",
    )
    parser.add_argument(
        "--max-original-chars-per-doc",
        type=int,
        default=20000,
        help="Only used with --include-original-data; 0 means unlimited",
    )
    parser.add_argument(
        "--copy-memcells",
        action="store_true",
        help="Copy memcells/ into the new directory for inspection/reproducibility",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_run_dir).resolve()
    dst_dir = Path(args.dst_run_dir).resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source run dir not found: {src_dir}")
    if dst_dir.exists():
        raise FileExistsError(f"Destination already exists: {dst_dir}")

    search_path = src_dir / "search_results.json"
    if not search_path.exists():
        raise FileNotFoundError(f"Missing search_results.json: {search_path}")

    by_subject_episode, by_subject = build_memcell_index(src_dir)
    if not by_subject:
        raise FileNotFoundError(f"No memcells found under: {src_dir / 'memcells'}")

    dst_dir.mkdir(parents=True)
    search_results = load_json(search_path)
    if not isinstance(search_results, list):
        raise ValueError("search_results.json must contain a list")

    stats = rewrite_search_results(
        search_results=search_results,
        by_subject_episode=by_subject_episode,
        by_subject=by_subject,
        max_atomic_facts=args.max_atomic_facts_per_doc,
        include_original_data=args.include_original_data,
        max_original_chars=args.max_original_chars_per_doc,
    )
    save_json(dst_dir / "search_results.json", search_results)

    copied_answer = False
    answer_path = src_dir / "answer_results.json"
    if answer_path.exists():
        answer_results = load_json(answer_path)
        if isinstance(answer_results, list):
            stats["answer_contexts_rewritten"] = rewrite_answer_contexts(
                answer_results, search_results
            )
            save_json(dst_dir / "answer_results.json", answer_results)
            copied_answer = True

    for filename in ["pipeline.log"]:
        copy_optional(src_dir, dst_dir, filename)

    if args.copy_memcells and (src_dir / "memcells").exists():
        shutil.copytree(src_dir / "memcells", dst_dir / "memcells")

    manifest = {
        "source_run_dir": str(src_dir),
        "destination_run_dir": str(dst_dir),
        "include_original_data": args.include_original_data,
        "max_atomic_facts_per_doc": args.max_atomic_facts_per_doc,
        "max_original_chars_per_doc": args.max_original_chars_per_doc,
        "copied_answer_results_with_rewritten_context": copied_answer,
        "stats": stats,
        "next_step": (
            "Run evaluation.cli with --stages answer evaluate and --output-dir "
            "pointing to destination_run_dir. No add/search stage is required."
        ),
    }
    save_json(dst_dir / "enriched_context_manifest.json", manifest)

    print(f"Created enriched run directory: {dst_dir}")
    print(f"Questions rewritten: {stats['questions']}")
    print(
        "Matched docs: "
        f"{stats['docs_matched_to_memcell']}/{stats['docs']} "
        f"({stats['docs_with_atomic_facts']} docs with atomic facts)"
    )
    print(f"Atomic facts available: {stats['atomic_facts_available']}")
    if copied_answer:
        print(f"Rewrote existing answer contexts: {stats['answer_contexts_rewritten']}")
    print("No checkpoint was copied, so answer/evaluate will rerun from search_results.json.")


if __name__ == "__main__":
    main()
