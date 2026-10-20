"""Audit MEME retrieval contexts for deterministic answer-value visibility.

This audit intentionally ignores task execution paths.  A question is covered
when every normalized value in its official ``entity_values`` field occurs in
the system's complete formatted retrieval context.  It does not require
chronology, dependency rules, triggers, or deletion events.

For Del-after and Abs-after questions, ``entity_values`` describes the deleted
or invalidated prior value rather than the complete status answer.  Those rows
are retained for a uniform seven-system surface-result audit and explicitly
flagged as status-answer limitations; their coverage must not be interpreted as
proof that the context supports "removed" or "uncertain".
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.scripts.analyze_meme_deterministic_evidence import (
    DATA,
    RESULTS,
    RUNS,
    load_scores,
    norm,
    qid_parts,
    strings,
)


EVAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EVAL_DIR / "analysis/meme_answer_only_evidence_20260818"
DEFAULT_SEMANTIC = (
    EVAL_DIR
    / "analysis/meme_seven_systems_full_flash_20260817"
    / "llm_answerability_spotcheck.csv"
)
DEFAULT_TASK_SPECIFIC = (
    EVAL_DIR
    / "analysis/meme_task_specific_evidence_v2_20260818"
    / "per_question.csv"
)


@dataclass(frozen=True)
class AnswerOnlyVerification:
    sufficient: bool
    required_values: tuple[str, ...]
    matched_values: tuple[str, ...]
    missing_values: tuple[str, ...]


def answer_values(question: dict[str, Any]) -> tuple[str, ...]:
    """Return unique normalized atomic answer values in official order."""
    return tuple(dict.fromkeys(value for item in strings(question.get("entity_values")) if (value := norm(item))))


def verify_answer_only(context_raw: str, question: dict[str, Any]) -> AnswerOnlyVerification:
    context = norm(context_raw)
    required = answer_values(question)
    matched = tuple(value for value in required if value in context)
    missing = tuple(value for value in required if value not in context)
    return AnswerOnlyVerification(
        sufficient=bool(required) and not missing,
        required_values=required,
        matched_values=matched,
        missing_values=missing,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for group, rows in sorted(grouped.items()):
        sufficient = [row for row in rows if row["answer_only_sufficient"]]
        output.append(
            {
                **dict(zip(keys, group)),
                "n": len(rows),
                "accuracy": mean(row["correct"] for row in rows),
                "answer_only_coverage": mean(row["answer_only_sufficient"] for row in rows),
                "sufficient_n": len(sufficient),
                "reader_utilization": mean(row["correct"] for row in sufficient) if sufficient else "",
                "sufficient_correct_n": sum(row["correct"] for row in sufficient),
                "status_answer_limitation_n": sum(row["status_answer_limitation"] for row in rows),
            }
        )
    return output


def load_boolean_csv(path: Path, value_field: str) -> dict[tuple[str, str], bool]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = str(row.get(value_field, "")).strip().lower()
            if raw in {"true", "false", "1", "0"}:
                result[(row["system"], row["question_id"])] = raw in {"true", "1"}
    return result


def compare_existing(
    records: list[dict[str, Any]],
    semantic: dict[tuple[str, str], bool],
    task_specific: dict[tuple[str, str], bool],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for group, rows in sorted(grouped.items()):
        semantic_rows = [
            (bool(row["answer_only_sufficient"]), semantic[(row["system"], row["question_id"])])
            for row in rows
            if (row["system"], row["question_id"]) in semantic
        ]
        task_rows = [
            (bool(row["answer_only_sufficient"]), task_specific[(row["system"], row["question_id"])])
            for row in rows
            if (row["system"], row["question_id"]) in task_specific
        ]
        output.append(
            {
                **dict(zip(keys, group)),
                "n": len(rows),
                "answer_only_rate": mean(row["answer_only_sufficient"] for row in rows),
                "semantic_valid_n": len(semantic_rows),
                "semantic_rate": mean(actual for _, actual in semantic_rows) if semantic_rows else "",
                "answer_semantic_agreement": mean(predicted == actual for predicted, actual in semantic_rows) if semantic_rows else "",
                "answer_only_semantic_only": sum(not predicted and actual for predicted, actual in semantic_rows),
                "answer_only_only_vs_semantic": sum(predicted and not actual for predicted, actual in semantic_rows),
                "task_specific_valid_n": len(task_rows),
                "task_specific_rate": mean(actual for _, actual in task_rows) if task_rows else "",
                "answer_task_specific_agreement": mean(predicted == actual for predicted, actual in task_rows) if task_rows else "",
                "answer_only_task_only": sum(not predicted and actual for predicted, actual in task_rows),
                "answer_only_only_vs_task": sum(predicted and not actual for predicted, actual in task_rows),
            }
        )
    return output


def pct(value: Any) -> str:
    return "n/a" if value == "" else f"{float(value) * 100:.2f}%"


def render_report(
    system_summary: list[dict[str, Any]],
    task_phase_summary: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> str:
    comparisons = {row["system"]: row for row in comparison}
    lines = [
        "# MEME seven-system answer-only deterministic audit",
        "",
        "A question is covered iff every normalized official `entity_values` value appears "
        "in the complete formatted retrieval context. The rule is identical for every task "
        "and ignores chronology, dependency rules, triggers, deletion events, and causal propagation.",
        "",
        "## System results",
        "",
        "| System | Accuracy | Answer-only coverage | Conditional utilization | Task-specific coverage | Semantic coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in system_summary:
        other = comparisons[row["system"]]
        lines.append(
            f"| {row['system']} | {pct(row['accuracy'])} | {pct(row['answer_only_coverage'])} | "
            f"{pct(row['reader_utilization'])} | {pct(other['task_specific_rate'])} | "
            f"{pct(other['semantic_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Conditional utilization is `P(correct | all answer values visible)`.",
            "",
            "## Coverage by task and phase (pooled across systems)",
            "",
            "| Task | Phase | n | Coverage | Utilization |",
            "|---|---|---:|---:|---:|",
        ]
    )
    pooled: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_phase_summary:
        pooled[(row["task_type"], row["phase"])].append(row)
    for (task, phase), rows in sorted(pooled.items()):
        n = sum(int(row["n"]) for row in rows)
        sufficient_n = sum(int(row["sufficient_n"]) for row in rows)
        correct_n = sum(int(row["sufficient_correct_n"]) for row in rows)
        lines.append(
            f"| {task} | {phase} | {n} | {sufficient_n / n * 100:.2f}% | "
            f"{correct_n / sufficient_n * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is surface answer-value visibility, not semantic entailment. Exact normalized "
            "substring matching can miss paraphrases and accept negated, hypothetical, stale, or contradictory mentions.",
            "- For `Tr`, all values are required but their order is deliberately ignored.",
            "- For `Cas`, only the replacement answer value is required; rule and trigger evidence are ignored.",
            "- For `Del-after`, the official `entity_values` field is the deleted old value. Coverage therefore means "
            "that value is still visible, not that the answer `explicitly removed` is supported.",
            "- For `Abs-after`, coverage means the prior value is visible, not that invalidation or uncertainty is supported.",
            "- The deterministic audit scans complete contexts, while the existing semantic audit used a gold-localized "
            "excerpt capped at 9,000 characters; their raw difference still includes an input-scope confound.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--semantic-csv", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument("--task-specific-csv", type=Path, default=DEFAULT_TASK_SPECIFIC)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {args.output_dir}")

    episodes = json.loads(args.data.read_text(encoding="utf-8"))
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    records: list[dict[str, Any]] = []
    configured_runs = [
        (system, run_name)
        for system, run_names in RUNS.items()
        for run_name in run_names
    ]
    for system, run_name in configured_runs:
        run = args.results_root / run_name
        answers = json.loads((run / "answer_results.json").read_text(encoding="utf-8"))
        scores = load_scores(run / "eval_results.json")
        for item in answers:
            question_id = str(item.get("question_id"))
            if question_id not in scores:
                continue
            episode_id, phase, question_index = qid_parts(question_id)
            question = episodes_by_id[episode_id][f"{phase}_questions"]["questions"][question_index]
            result = verify_answer_only(str(item.get("formatted_context") or ""), question)
            task_type = str(question.get("task_type") or "")
            records.append(
                {
                    "system": system,
                    "run_name": run_name,
                    "question_id": question_id,
                    "conversation_id": item.get("conversation_id"),
                    "phase": phase,
                    "task_type": task_type,
                    "score": scores[question_id],
                    "correct": int(scores[question_id] >= 0.5),
                    "answer_only_sufficient": int(result.sufficient),
                    "required_value_count": len(result.required_values),
                    "matched_value_count": len(result.matched_values),
                    "required_values": " | ".join(result.required_values),
                    "missing_values": " | ".join(result.missing_values),
                    "status_answer_limitation": int(phase == "after" and task_type in {"Del", "Abs"}),
                }
            )

    if not records:
        raise ValueError("No aligned answer/evaluation records found")
    expected = len(configured_runs) * 1188
    if len(records) != expected:
        raise ValueError(f"Expected {expected} rows for seven complete runs, found {len(records)}")

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_question.csv", records)
    system_summary = aggregate(records, ("system",))
    task_phase_summary = aggregate(records, ("system", "task_type", "phase"))
    write_csv(args.output_dir / "summary_by_system.csv", system_summary)
    write_csv(args.output_dir / "summary_by_system_task_phase.csv", task_phase_summary)

    semantic = load_boolean_csv(args.semantic_csv, "llm_answerable")
    task_specific = load_boolean_csv(args.task_specific_csv, "evidence_sufficient")
    comparison_system = compare_existing(records, semantic, task_specific, ("system",))
    comparison_task_phase = compare_existing(
        records, semantic, task_specific, ("task_type", "phase")
    )
    write_csv(args.output_dir / "comparison_by_system.csv", comparison_system)
    write_csv(args.output_dir / "comparison_by_task_phase.csv", comparison_task_phase)
    (args.output_dir / "report.md").write_text(
        render_report(system_summary, task_phase_summary, comparison_system), encoding="utf-8"
    )
    manifest = {
        "version": "meme-answer-only-evidence-v1",
        "estimand": "all normalized official entity_values visible in complete formatted context",
        "records": len(records),
        "runs": [{"system": system, "run_name": run} for system, run in configured_runs],
        "semantic_comparison": str(args.semantic_csv),
        "task_specific_comparison": str(args.task_specific_csv),
        "coverage_rule": "all answer values present; no task-specific path requirements",
        "status_answer_limitations": ["Del-after", "Abs-after"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} answer-only MEME rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
