"""Measure MEME structural-relation retention conditional on value retention.

The pooled primary estimand covers Tr/Cas/Abs after-state questions only:

    P(required structural support | task-specific target values are complete)

Tr requires chronological order support. Cas and Abs require both a dependency
rule and the upstream after-state trigger. Del-after is reported separately
because retaining the deleted value is not a desirable prerequisite.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.scripts.analyze_meme_task_specific_significance import (
    omnibus_max_range,
    pairwise_rows,
)


EVALUATION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    EVALUATION_DIR
    / "analysis"
    / "meme_task_specific_evidence_v2_20260818"
    / "per_question.csv"
)
DEFAULT_OUTPUT = EVALUATION_DIR / "analysis" / "meme_relation_retention_20260818"
STRUCTURAL_TASKS = ("Tr", "Cas", "Abs")


def add_relation_complete(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["relation_complete"] = False
    tr = result["task_type"].eq("Tr")
    cas_abs = result["task_type"].isin(("Cas", "Abs"))
    result.loc[tr, "relation_complete"] = result.loc[tr, "order_support"].astype(bool)
    result.loc[cas_abs, "relation_complete"] = (
        result.loc[cas_abs, "dependency_rule_support"].astype(bool)
        & result.loc[cas_abs, "trigger_support"].astype(bool)
    )
    return result


def load_episode_counts(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(path)
    required = {
        "system",
        "question_id",
        "conversation_id",
        "phase",
        "task_type",
        "value_complete",
        "order_support",
        "dependency_rule_support",
        "trigger_support",
        "deletion_event_support",
        "non_disclosure",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    structural = frame[
        frame["task_type"].isin(STRUCTURAL_TASKS) & frame["phase"].eq("after")
    ].copy()
    structural = add_relation_complete(structural)
    structural["eligible"] = structural["value_complete"].astype(bool)
    structural["retained"] = structural["eligible"] & structural["relation_complete"]

    systems = sorted(structural["system"].unique().tolist())
    episodes = sorted(structural["conversation_id"].unique().tolist())
    eligible = np.zeros((len(episodes), len(systems)), dtype=np.int64)
    retained = np.zeros_like(eligible)
    for episode_index, episode in enumerate(episodes):
        episode_rows = structural[structural["conversation_id"].eq(episode)]
        expected_questions: list[str] | None = None
        for system_index, system in enumerate(systems):
            rows = episode_rows[episode_rows["system"].eq(system)].sort_values("question_id")
            questions = rows["question_id"].tolist()
            if rows["question_id"].duplicated().any():
                raise ValueError(f"Duplicate questions for {system}/{episode}")
            if expected_questions is None:
                expected_questions = questions
            elif questions != expected_questions:
                raise ValueError(f"Unpaired structural questions for {system}/{episode}")
            eligible[episode_index, system_index] = int(rows["eligible"].sum())
            retained[episode_index, system_index] = int(rows["retained"].sum())
    if np.any(retained > eligible):
        raise ValueError("Retained relation count exceeds eligible value-complete count")
    return systems, episodes, eligible, retained, frame


def bootstrap_retention(
    eligible: np.ndarray,
    retained: np.ndarray,
    samples: int,
    seed: int,
    batch_size: int = 1000,
) -> np.ndarray:
    episode_count, system_count = eligible.shape
    draws = np.empty((samples, system_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(0, episode_count, size=(stop - start, episode_count))
        denominator = eligible[indices].sum(axis=1, dtype=np.float64)
        numerator = retained[indices].sum(axis=1, dtype=np.float64)
        draws[start:stop] = numerator / denominator
    return draws


def task_rows(structural: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (system, task), group in structural.groupby(["system", "task_type"], sort=True):
        eligible = group[group["eligible"]]
        retained = int(eligible["relation_complete"].sum())
        denominator = len(eligible)
        row: dict[str, object] = {
            "system": system,
            "task_type": task,
            "total_questions": len(group),
            "value_complete_n": denominator,
            "relation_retained_n": retained,
            "relation_retention": retained / denominator if denominator else float("nan"),
            "relation_loss": 1.0 - retained / denominator if denominator else float("nan"),
            "order_support_rate_given_values": "",
            "dependency_rule_rate_given_values": "",
            "trigger_rate_given_values": "",
        }
        if task == "Tr":
            row["required_structure"] = "chronological_order"
            row["order_support_rate_given_values"] = row["relation_retention"]
        else:
            row["required_structure"] = "dependency_and_trigger"
            row["dependency_rule_rate_given_values"] = float(
                eligible["dependency_rule_support"].mean()
            )
            row["trigger_rate_given_values"] = float(eligible["trigger_support"].mean())
        rows.append(row)
    return rows


def deletion_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    deletion = frame[frame["task_type"].eq("Del") & frame["phase"].eq("after")]
    rows = []
    for system, group in deletion.groupby("system", sort=True):
        rows.append(
            {
                "system": system,
                "n": len(group),
                "deletion_event_support_n": int(group["deletion_event_support"].sum()),
                "deletion_event_support_rate": float(group["deletion_event_support"].mean()),
                "old_value_non_disclosure_n": int(group["non_disclosure"].sum()),
                "old_value_non_disclosure_rate": float(group["non_disclosure"].mean()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    systems, episodes, eligible, retained, full_frame = load_episode_counts(args.input)
    estimates = retained.sum(axis=0) / eligible.sum(axis=0)
    draws = bootstrap_retention(eligible, retained, args.bootstrap_samples, args.seed)
    omnibus = omnibus_max_range(estimates, draws)
    pairs = pairwise_rows(systems, estimates, draws)

    structural = full_frame[
        full_frame["task_type"].isin(STRUCTURAL_TASKS) & full_frame["phase"].eq("after")
    ].copy()
    structural = add_relation_complete(structural)
    structural["eligible"] = structural["value_complete"].astype(bool)
    per_task = task_rows(structural)
    deletion = deletion_rows(full_frame)

    system_rows = []
    for index, system in enumerate(systems):
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        system_rows.append(
            {
                "system": system,
                "structural_questions_n": int(
                    (structural["system"].eq(system)).sum()
                ),
                "value_complete_n": int(eligible[:, index].sum()),
                "relation_retained_n": int(retained[:, index].sum()),
                "relation_retention": estimates[index],
                "relation_retention_ci95_low": low,
                "relation_retention_ci95_high": high,
                "relation_loss": 1.0 - estimates[index],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "system_estimates.csv", system_rows)
    write_csv(args.output_dir / "task_breakdown.csv", per_task)
    write_csv(args.output_dir / "deletion_support.csv", deletion)
    write_csv(args.output_dir / "pairwise_comparisons.csv", pairs)
    (args.output_dir / "omnibus_test.json").write_text(
        json.dumps(
            {
                "test": "episode-clustered centered max-range bootstrap",
                "episodes": len(episodes),
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                **omnibus,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    significant = sum(bool(row["significant_holm_0_05"]) for row in pairs)
    lines = [
        "# MEME relation retention conditional on value completeness",
        "",
        "Primary estimand over Tr/Cas/Abs after-state questions:",
        "`P(required structural support | task-specific target values complete)`.",
        "Tr requires chronological order; Cas/Abs require both dependency-rule and upstream-trigger support.",
        "Del-after is reported separately because old-value retention is not a desirable prerequisite.",
        "",
        "## Overall",
        "",
        f"Observed range: **{omnibus['observed_range'] * 100:.2f} pp**; centered max-range bootstrap p = **{omnibus['p_value']:.6g}**. ",
        f"Holm-significant pairs: **{significant}/21**.",
        "",
        "| System | Values complete | Relation retained | Retention (95% CI) | Relation loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(system_rows, key=lambda item: float(item["relation_retention"]), reverse=True):
        lines.append(
            f"| {row['system']} | {row['value_complete_n']}/{row['structural_questions_n']} | "
            f"{row['relation_retained_n']}/{row['value_complete_n']} | "
            f"{pct(row['relation_retention'])} [{pct(row['relation_retention_ci95_low'])}, "
            f"{pct(row['relation_retention_ci95_high'])}] | {pct(row['relation_loss'])} |"
        )
    lines += [
        "",
        "## By task",
        "",
        "| System | Task | Values complete | Order | Dependency | Trigger | All required | Loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(per_task, key=lambda item: (str(item["system"]), str(item["task_type"]))):
        lines.append(
            f"| {row['system']} | {row['task_type']} | {row['value_complete_n']}/{row['total_questions']} | "
            f"{pct(row['order_support_rate_given_values']) if row['order_support_rate_given_values'] != '' else '—'} | "
            f"{pct(row['dependency_rule_rate_given_values']) if row['dependency_rule_rate_given_values'] != '' else '—'} | "
            f"{pct(row['trigger_rate_given_values']) if row['trigger_rate_given_values'] != '' else '—'} | "
            f"{row['relation_retained_n']}/{row['value_complete_n']} ({pct(row['relation_retention'])}) | "
            f"{pct(row['relation_loss'])} |"
        )
    lines += [
        "",
        "## Del-after (separate)",
        "",
        "| System | Targeted deletion event | Old-value non-disclosure |",
        "| --- | ---: | ---: |",
    ]
    for row in sorted(deletion, key=lambda item: float(item["deletion_event_support_rate"]), reverse=True):
        lines.append(
            f"| {row['system']} | {row['deletion_event_support_n']}/{row['n']} "
            f"({pct(row['deletion_event_support_rate'])}) | "
            f"{row['old_value_non_disclosure_n']}/{row['n']} "
            f"({pct(row['old_value_non_disclosure_rate'])}) |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "This is a retrieved-context diagnostic, so it combines representation compression, retrieval, and context assembly.",
        "System-specific value-complete denominators also mean the pairwise comparison is conditional, not a pure causal compression effect.",
        "Exact-value anchors and deterministic relation cue lexicons can produce verifier false negatives.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
