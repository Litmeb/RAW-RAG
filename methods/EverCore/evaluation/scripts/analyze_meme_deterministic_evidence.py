"""Estimate MEME evidence sufficiency without an LLM judge.

This diagnostic uses the official MEME question fields and deterministic
normalized-string checks over each system's retrieved context.  It is a
conservative task-complete proxy: all required values (and task-specific
state/change terms) must be visible in the context.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
DATA = EVAL_DIR / "data/meme/meme_filler32k.json"
RESULTS = EVAL_DIR / "results"
OUT = EVAL_DIR / "analysis/meme_deterministic_evidence"

RUNS = {
    "rag": ["meme-rag-normal"],
    "emem": ["meme-emem-normal"],
    "mem0_local": ["meme-mem0_local-normal"],
    "evermemos": ["meme-evermemos-normal-3"],
    "memmachine": ["meme-memmachine-normal"],
    "memos_local": ["meme-memos_local-fine-deepseek"],
    "memu_local": ["meme-memu_local-normal-reanswer-20260731"],
}


def norm(value: Any) -> str:
    text = "" if value is None else str(value).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for child in value.values():
            out.extend(strings(child))
        return out
    if isinstance(value, list):
        out = []
        for child in value:
            out.extend(strings(child))
        return out
    text = str(value).strip()
    return [text] if text else []


def qid_parts(qid: str) -> tuple[str, str, int]:
    match = re.match(r"meme_(.+)_(before|after)_qa(\d+)$", qid)
    if not match:
        raise ValueError(f"Unexpected MEME question id: {qid}")
    return match.group(1), match.group(2), int(match.group(3))


def required_terms(question: dict[str, Any], episode: dict[str, Any], phase: str) -> tuple[list[str], str]:
    task = str(question.get("task_type") or "")
    values = strings(question.get("entity_values"))
    terms = list(values)
    root = str(episode.get("root") or "")
    change = episode.get("root_change") or {}
    if task == "Cas":
        terms.append(change.get("before" if phase == "before" else "after", ""))
    elif task == "Abs":
        if phase == "after":
            terms.extend([change.get("after", ""), *values])
    elif task == "Del":
        # The official deletion session contains the old value and one of
        # several possible explicit removal instructions.  The OR condition
        # is applied in ``main`` below.
        pass
    terms = [norm(t) for t in terms if norm(t)]
    # A task's values may repeat in prose; requirements are unique normalized terms.
    unique = list(dict.fromkeys(terms))
    return unique, task


def load_scores(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("detailed_results", {}) if isinstance(data, dict) else {}
    rows = []
    if isinstance(items, dict):
        for value in items.values():
            rows.extend(value if isinstance(value, list) else [value])
    elif isinstance(items, list):
        rows = items
    scores = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("question_id"):
            continue
        score = row.get("filtered_u_pass") if row.get("phase") == "after" else row.get("metric_value", row.get("score", row.get("u_pass", 0)))
        try:
            scores[str(row["question_id"])] = float(score)
        except (TypeError, ValueError):
            scores[str(row["question_id"])] = 0.0
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    episodes = json.loads(args.data.read_text(encoding="utf-8"))
    episodes_by_id = {str(episode.get("episode_id")): episode for episode in episodes}
    records: list[dict[str, Any]] = []
    for system, run_names in RUNS.items():
        for run_name in run_names:
            run = args.results_root / run_name
            answers_path, eval_path = run / "answer_results.json", run / "eval_results.json"
            if not answers_path.exists() or not eval_path.exists():
                continue
            answers = json.loads(answers_path.read_text(encoding="utf-8"))
            scores = load_scores(eval_path)
            for item in answers:
                qid = str(item.get("question_id"))
                if qid not in scores:
                    continue
                episode_id, phase, qidx = qid_parts(qid)
                episode = episodes_by_id[episode_id]
                question = episode[f"{phase}_questions"]["questions"][qidx]
                terms, task = required_terms(question, episode, phase)
                context = norm(item.get("formatted_context"))
                hits = [term for term in terms if term in context]
                if task == "Del":
                    deletion_cues = ["remove that from my memory", "please remove", "deleted", "removed"]
                    cue_hits = [cue for cue in deletion_cues if norm(cue) in context]
                    sufficient = bool(terms) and len(hits) == len(terms) and bool(cue_hits)
                    required_count = len(terms) + 1
                    matched_count = len(hits) + int(bool(cue_hits))
                    missing = [term for term in terms if term not in context]
                    if not cue_hits:
                        missing.append("<deletion cue>")
                else:
                    sufficient = bool(terms) and len(hits) == len(terms)
                    required_count = len(terms)
                    matched_count = len(hits)
                    missing = [term for term in terms if term not in context]
                records.append({
                    "system": system,
                    "question_id": qid,
                    "conversation_id": item.get("conversation_id"),
                    "phase": phase,
                    "task_type": task,
                    "score": scores[qid],
                    "correct": int(scores[qid] >= 0.5),
                    "required_term_count": required_count,
                    "matched_term_count": matched_count,
                    "evidence_sufficient": int(sufficient),
                    "missing_terms": " | ".join(missing),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with (args.output_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(records)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["system"], row["task_type"])].append(row)
    summary = []
    for (system, task), rows in sorted(grouped.items()):
        sufficient = [r for r in rows if r["evidence_sufficient"]]
        summary.append({
            "system": system, "task_type": task, "n": len(rows),
            "evidence_sufficient_rate": mean(r["evidence_sufficient"] for r in rows),
            "sufficient_n": len(sufficient),
            "reader_utilization": (mean(r["correct"] for r in sufficient) if sufficient else ""),
            "sufficient_correct_n": sum(r["correct"] for r in sufficient),
        })
    with (args.output_dir / "summary_by_system_task.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(summary[0]) if summary else []
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    overall = []
    for system in sorted({r["system"] for r in records}):
        rows = [r for r in records if r["system"] == system]
        sufficient = [r for r in rows if r["evidence_sufficient"]]
        overall.append({"system": system, "n": len(rows), "evidence_sufficient_rate": mean(r["evidence_sufficient"] for r in rows), "sufficient_n": len(sufficient), "reader_utilization": mean(r["correct"] for r in sufficient) if sufficient else "", "sufficient_correct_n": sum(r["correct"] for r in sufficient)})
    with (args.output_dir / "summary_by_system.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(overall[0]) if overall else []
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(overall)
    print(f"Wrote {len(records)} MEME rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
