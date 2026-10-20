"""Analyze the sampled MEME native/oracle/raw continuity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken


EVALUATION_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_TAG = "meme-continuity-gold-fields-task-readout-conv0-5-v5-20260717"
DEFAULT_OUTPUT = EVALUATION_DIR / "analysis/meme_continuity_task_readout_conv0_5_v5"
DEFAULT_SELECTION_ONLY_REPORT = (
    EVALUATION_DIR / "analysis/meme_continuity_gold_fields_conv0_5_v3.json"
)
SYSTEMS = {
    "mem0": "meme-mem0-normal-merged",
    "memos": "meme-memos-time-filter",
    "emem": "meme-emem-normal",
    "evermemos": "meme-evermemos-normal-3",
}
TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")
FILTERED_TASKS = {"Del", "Cas", "Abs"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def detailed_rows(run_dir: Path) -> list[dict[str, Any]]:
    detailed = load_json(run_dir / "eval_results.json")["detailed_results"]
    if isinstance(detailed, dict):
        return [row for rows in detailed.values() for row in rows]
    return list(detailed)


def first_judge_pass(row: dict[str, Any]) -> bool:
    runs = row.get("judge_runs") or []
    if runs:
        return bool(runs[0].get("u_pass"))
    return bool(row.get("u_pass"))


def scored_rows(run_dir: Path, question_ids: set[str]) -> list[dict[str, Any]]:
    rows = [row for row in detailed_rows(run_dir) if str(row["question_id"]) in question_ids]
    before = {
        (str(row["conversation_id"]), str(row.get("entity_key") or "")): first_judge_pass(row)
        for row in rows
        if row.get("phase") == "before"
    }
    result = []
    for row in rows:
        value = first_judge_pass(row)
        task = str(row.get("task_type") or "")
        if row.get("phase") == "after" and task in FILTERED_TASKS:
            value = value and before.get(
                (str(row["conversation_id"]), str(row.get("entity_key") or "")),
                False,
            )
        result.append({**row, "first_run_official_pass": int(value)})
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = [row for row in rows if row.get("phase") == "before"]
    after = [row for row in rows if row.get("phase") == "after"]
    per_task = {}
    for task in TASKS:
        task_rows = [row for row in after if row.get("task_type") == task]
        correct = sum(row["first_run_official_pass"] for row in task_rows)
        per_task[task] = {
            "correct": correct,
            "total": len(task_rows),
            "accuracy": correct / len(task_rows) if task_rows else 0.0,
        }
    overall = float(np.mean([per_task[task]["accuracy"] for task in TASKS]))
    before_correct = sum(first_judge_pass(row) for row in before)
    return {
        "questions": len(rows),
        "before_correct": before_correct,
        "before_total": len(before),
        "before_accuracy": before_correct / len(before) if before else 0.0,
        "after_questions": len(after),
        "task_accuracy": per_task,
        "overall_macro_accuracy": overall,
        "after_pass_by_id": {
            str(row["question_id"]): int(row["first_run_official_pass"])
            for row in after
        },
    }


def context_summary(run_dir: Path, question_ids: set[str]) -> dict[str, float]:
    answers = load_json(run_dir / "answer_results.json")
    encoding = tiktoken.get_encoding("o200k_base")
    values = [
        len(
            encoding.encode(
                str(row.get("formatted_context") or ""), disallowed_special=()
            )
        )
        for row in answers
        if str(row["question_id"]) in question_ids
    ]
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def episode_task_counts(
    rows: list[dict[str, Any]], conversations: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    conversation_index = {value: index for index, value in enumerate(conversations)}
    task_index = {value: index for index, value in enumerate(TASKS)}
    correct = np.zeros((len(conversations), len(TASKS)), dtype=np.float64)
    total = np.zeros_like(correct)
    for row in rows:
        if row.get("phase") != "after":
            continue
        conv = str(row["conversation_id"])
        task = str(row.get("task_type") or "")
        correct[conversation_index[conv], task_index[task]] += row[
            "first_run_official_pass"
        ]
        total[conversation_index[conv], task_index[task]] += 1
    return correct, total


def paired_episode_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    conversations = sorted(
        {str(row["conversation_id"]) for row in left if row.get("phase") == "after"},
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    left_correct, left_total = episode_task_counts(left, conversations)
    right_correct, right_total = episode_task_counts(right, conversations)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(conversations), size=(count, len(conversations)))
        left_accuracy = (
            left_correct[indices].sum(axis=1) / left_total[indices].sum(axis=1)
        ).mean(axis=1)
        right_accuracy = (
            right_correct[indices].sum(axis=1) / right_total[indices].sum(axis=1)
        ).mean(axis=1)
        differences[start : start + count] = left_accuracy - right_accuracy
    low, high = np.quantile(differences, [0.025, 0.975])
    left_full = (left_correct.sum(axis=0) / left_total.sum(axis=0)).mean()
    right_full = (right_correct.sum(axis=0) / right_total.sum(axis=0)).mean()
    return {
        "difference": float(left_full - right_full),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resampling_unit": "conversation/episode",
        "episodes": len(conversations),
    }


def coverage_summary(run_dir: Path) -> dict[str, Any]:
    rows = load_json(run_dir / "meme_continuity_oracle_audit.json")
    per_task = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task_type"] == task]
        per_task[task] = {
            "questions": len(task_rows),
            "any_support_value_observed": sum(
                row["any_support_value_observed"] for row in task_rows
            ),
            "all_support_values_observed": sum(
                row["all_support_values_observed"] for row in task_rows
            ),
        }
    return {
        "questions": len(rows),
        "any_support_value_observed": sum(row["any_support_value_observed"] for row in rows),
        "all_support_values_observed": sum(row["all_support_values_observed"] for row in rows),
        "mean_candidate_pool_size": sum(row["candidate_pool_size"] for row in rows)
        / len(rows),
        "mean_selected_memories": sum(row["selected_count"] for row in rows) / len(rows),
        "questions_with_empty_context": sum(row["selected_count"] == 0 for row in rows),
        "task_coverage": per_task,
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selection-only-report", type=Path, default=DEFAULT_SELECTION_ONLY_REPORT
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    raw_dir = RESULTS_DIR / f"meme-raw-{args.tag}"
    raw_question_ids = {
        str(row["question_id"]) for row in load_json(raw_dir / "search_results.json")
    }
    raw_searches = load_json(raw_dir / "search_results.json")
    conversation_ids = sorted(
        {str(row["conversation_id"]) for row in raw_searches},
        key=lambda value: int(value.rsplit("_", 1)[-1]),
    )
    raw_rows = scored_rows(raw_dir, raw_question_ids)
    raw_summary = summarize(raw_rows)
    raw_summary["run_dir"] = str(raw_dir.resolve())
    raw_summary["context_tokens"] = context_summary(raw_dir, raw_question_ids)
    selection_only = None
    if args.selection_only_report.exists():
        candidate = load_json(args.selection_only_report.resolve())
        if int(candidate.get("sample", {}).get("questions", -1)) == len(raw_question_ids):
            selection_only = candidate
    if selection_only:
        raw_summary["selection_only_overall_macro_accuracy"] = selection_only[
            "raw_gold_evidence"
        ]["overall_macro_accuracy"]
        raw_summary["task_readout_gain"] = (
            raw_summary["overall_macro_accuracy"]
            - raw_summary["selection_only_overall_macro_accuracy"]
        )

    report: dict[str, Any] = {
        "experiment": "MEME gold-field continuity pilot",
        "sample": {
            "conversation_ids": conversation_ids,
            "episodes": len(conversation_ids),
            "questions": len(raw_question_ids),
            "after_questions": raw_summary["after_questions"],
        },
        "scoring": {
            "judge_run": "first judge run for every condition",
            "overall": "macro mean of ER, Agg, Tr, Del, Cas, Abs",
            "filtering": "Del/Cas/Abs after pass requires the paired before pass",
            "bootstrap": (
                f"{args.bootstrap_samples:,} paired resamples of "
                f"{len(conversation_ids)} episodes"
            ),
            "task_aware_readout": (
                "The v5 context states task/update semantics but contains neither "
                "entity_values nor gold_answer."
            ),
        },
        "raw_gold_evidence": raw_summary,
        "systems": {},
        "limitations": [
            "The memory oracle candidate pool is the union of objects observed across native searches, not a complete backend store.",
            "The task-aware policy supplies transition semantics but exposes neither the gold answer nor gold entity values.",
        ],
    }

    for system, native_name in SYSTEMS.items():
        native_dir = RESULTS_DIR / native_name
        oracle_dir = RESULTS_DIR / f"meme-{system}-{args.tag}"
        native_rows = scored_rows(native_dir, raw_question_ids)
        oracle_rows = scored_rows(oracle_dir, raw_question_ids)
        native = summarize(native_rows)
        oracle = summarize(oracle_rows)
        native["run_dir"] = str(native_dir.resolve())
        oracle["run_dir"] = str(oracle_dir.resolve())
        native["context_tokens"] = context_summary(native_dir, raw_question_ids)
        oracle["context_tokens"] = context_summary(oracle_dir, raw_question_ids)
        oracle["observed_pool_coverage"] = coverage_summary(oracle_dir)
        if selection_only:
            oracle["selection_only_overall_macro_accuracy"] = selection_only[
                "systems"
            ][system]["memory_oracle"]["overall_macro_accuracy"]
            oracle["task_readout_gain"] = (
                oracle["overall_macro_accuracy"]
                - oracle["selection_only_overall_macro_accuracy"]
            )

        native_acc = native["overall_macro_accuracy"]
        oracle_acc = oracle["overall_macro_accuracy"]
        raw_acc = raw_summary["overall_macro_accuracy"]
        original_gap = raw_acc - native_acc
        repair = oracle_acc - native_acc
        residual = raw_acc - oracle_acc
        recovery = repair / original_gap if original_gap > 0 else None
        report["systems"][system] = {
            "native": native,
            "memory_oracle": oracle,
            "effects": {
                "raw_minus_native": original_gap,
                "memory_oracle_minus_native": repair,
                "raw_minus_memory_oracle": residual,
                "recovery_fraction": recovery,
                "episode_bootstrap_memory_oracle_minus_native": paired_episode_bootstrap(
                    oracle_rows,
                    native_rows,
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                ),
                "episode_bootstrap_raw_minus_memory_oracle": paired_episode_bootstrap(
                    raw_rows,
                    oracle_rows,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 1,
                ),
            },
        }

    output_prefix = args.output_prefix.resolve()
    dump_json(output_prefix.with_suffix(".json"), report)

    if selection_only:
        header = "| System | Native | Selection-only oracle | Task-aware oracle | Task-aware raw | Readout gain | Raw−oracle | Any/all support coverage |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| System | Native | Task-aware oracle | Task-aware raw | Oracle−native | Raw−oracle | Any/all support coverage |\n|---|---:|---:|---:|---:|---:|---:|"
    table_rows = []
    for system, values in report["systems"].items():
        coverage = values["memory_oracle"]["observed_pool_coverage"]
        if selection_only:
            table_rows.append(
                f"| {system} | {pct(values['native']['overall_macro_accuracy'])} | "
                f"{pct(values['memory_oracle']['selection_only_overall_macro_accuracy'])} | "
                f"{pct(values['memory_oracle']['overall_macro_accuracy'])} | "
                f"{pct(raw_summary['overall_macro_accuracy'])} | "
                f"{pp(values['memory_oracle']['task_readout_gain'])} | "
                f"{pp(values['effects']['raw_minus_memory_oracle'])} | "
                f"{coverage['any_support_value_observed']}/{coverage['questions']} / "
                f"{coverage['all_support_values_observed']}/{coverage['questions']} |"
            )
        else:
            table_rows.append(
                f"| {system} | {pct(values['native']['overall_macro_accuracy'])} | "
                f"{pct(values['memory_oracle']['overall_macro_accuracy'])} | "
                f"{pct(raw_summary['overall_macro_accuracy'])} | "
                f"{pp(values['effects']['memory_oracle_minus_native'])} | "
                f"{pp(values['effects']['raw_minus_memory_oracle'])} | "
                f"{coverage['any_support_value_observed']}/{coverage['questions']} / "
                f"{coverage['all_support_values_observed']}/{coverage['questions']} |"
            )

    task_header = "| Condition | " + " | ".join(TASKS) + " | Overall |\n|---|" + "---:|" * 7
    task_rows = []
    conditions = [("Raw gold evidence", raw_summary)]
    for system, values in report["systems"].items():
        conditions.extend(
            [(f"{system} native", values["native"]), (f"{system} oracle", values["memory_oracle"])]
        )
    for name, summary in conditions:
        cells = [pct(summary["task_accuracy"][task]["accuracy"]) for task in TASKS]
        task_rows.append(
            f"| {name} | " + " | ".join(cells) + f" | {pct(summary['overall_macro_accuracy'])} |"
        )

    transition_header = "| System | Cas native→oracle | Abs native→oracle | Del native→oracle | Agg native→oracle | Tr native→oracle |\n|---|---:|---:|---:|---:|---:|"
    transition_rows = []
    effects_header = "| System | Oracle−native (95% CI) | Raw−oracle (95% CI) |\n|---|---:|---:|"
    effects_rows = []
    coverage_header = "| System | ER | Agg | Tr | Del | Cas | Abs |\n|---|---:|---:|---:|---:|---:|---:|"
    coverage_rows = []
    for system, values in report["systems"].items():
        native_tasks = values["native"]["task_accuracy"]
        oracle_tasks = values["memory_oracle"]["task_accuracy"]
        transition_rows.append(
            f"| {system} | "
            + " | ".join(
                f"{pct(native_tasks[task]['accuracy'])}→{pct(oracle_tasks[task]['accuracy'])}"
                for task in ("Cas", "Abs", "Del", "Agg", "Tr")
            )
            + " |"
        )
        oracle_native_ci = values["effects"][
            "episode_bootstrap_memory_oracle_minus_native"
        ]
        raw_oracle_ci = values["effects"][
            "episode_bootstrap_raw_minus_memory_oracle"
        ]
        effects_rows.append(
            f"| {system} | {pp(oracle_native_ci['difference'])} "
            f"[{pp(oracle_native_ci['ci95_low'])}, {pp(oracle_native_ci['ci95_high'])}] | "
            f"{pp(raw_oracle_ci['difference'])} "
            f"[{pp(raw_oracle_ci['ci95_low'])}, {pp(raw_oracle_ci['ci95_high'])}] |"
        )
        task_coverage = values["memory_oracle"]["observed_pool_coverage"][
            "task_coverage"
        ]
        coverage_rows.append(
            f"| {system} | "
            + " | ".join(
                f"{task_coverage[task]['any_support_value_observed']}/"
                f"{task_coverage[task]['questions']} / "
                f"{task_coverage[task]['all_support_values_observed']}/"
                f"{task_coverage[task]['questions']}"
                for task in TASKS
            )
            + " |"
        )

    raw_delta_text = (
        " The task-aware policy changes its Overall from "
        f"{pct(raw_summary['selection_only_overall_macro_accuracy'])} to "
        f"{pct(raw_summary['overall_macro_accuracy'])}."
        if selection_only
        else ""
    )

    markdown = f"""# MEME Gold-field Continuity Evaluation

Scope: {len(conversation_ids)} `meme_filler32k` episodes, {len(raw_question_ids)} questions ({raw_summary['after_questions']} after-state questions). All conditions are scored from the first GPT-4o judge run; Del/Cas/Abs use official trivial-pass filtering. The v5 readout policy states task and state-update semantics but exposes neither `entity_values` nor `gold_answer`; it is placed after the evidence, and before-phase cascades are explicitly treated as hypothetical.

## Main comparison

{header}
{chr(10).join(table_rows)}

The memory oracle uses `target_entities`, `entity_values`, `gold_answer`, cascade-source state, and a phase-safe union of deduplicated memory objects observed across all native searches for that conversation. “Any/all support coverage” reports whether at least one / every required old, new, history, or cascade-source value exists literally in that pool.

## Task breakdown

{task_header}
{chr(10).join(task_rows)}

## State-task shifts

{transition_header}
{chr(10).join(transition_rows)}

## Paired episode bootstrap

{effects_header}
{chr(10).join(effects_rows)}

## Observed-pool support coverage by task

Each cell is `any support / task questions / all support / task questions`.

{coverage_header}
{chr(10).join(coverage_rows)}

## Interpretation

- Raw evidence is grounded in official MEME `evidence_session_indices` and narrowed per question using the gold fields; mean context is {raw_summary['context_tokens']['mean']:.1f} tokens.{raw_delta_text}
- Exact supporting-value coverage is incomplete in every observed memory pool, so failures cannot be assigned solely to final retrieval or the reader.
- Cas and Abs remain hard even with raw gold evidence. The supplied sessions contain conditional dependencies and the later root change, but do not directly state the derived answer; the reader still has to execute the state transition. This is a readout/reasoning ceiling of the current prompt, not evidence that raw continuity was lost.
- Native Abs accuracy is partly an ignorance advantage: sparse or missing memories often make the model answer "uncertain," which MEME rewards. Gold-field selection restores the previous value but cannot guarantee that the memory representation preserved the upstream invalidation relation, so Abs can decrease even when evidence recall improves.
- The observed-pool oracle is intentionally weaker than a complete-store oracle. Its main value is that its candidate boundary and evidence coverage are auditable and identical across systems.

Machine-readable results: `{output_prefix.with_suffix('.json')}`.
"""
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
