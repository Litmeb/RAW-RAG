"""Episode-clustered significance tests for MEME task-specific evidence metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    EVALUATION_DIR
    / "analysis"
    / "meme_task_specific_evidence_v2_20260818"
    / "per_question.csv"
)
DEFAULT_OUTPUT = (
    EVALUATION_DIR / "analysis" / "meme_task_specific_significance_20260818"
)


def bootstrap_sign_pvalue(values: np.ndarray) -> float:
    n = len(values)
    lower = (np.count_nonzero(values <= 0.0) + 1) / (n + 1)
    upper = (np.count_nonzero(values >= 0.0) + 1) / (n + 1)
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_episode_counts(
    path: Path,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {
        "system",
        "question_id",
        "conversation_id",
        "correct",
        "evidence_sufficient",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    systems = sorted(frame["system"].unique().tolist())
    episodes = sorted(frame["conversation_id"].unique().tolist())
    question_counts = np.zeros(len(episodes), dtype=np.int64)
    sufficient_counts = np.zeros((len(episodes), len(systems)), dtype=np.int64)
    sufficient_correct_counts = np.zeros_like(sufficient_counts)
    for episode_index, episode in enumerate(episodes):
        episode_rows = frame[frame["conversation_id"] == episode]
        expected_questions = None
        for system_index, system in enumerate(systems):
            rows = episode_rows[episode_rows["system"] == system].sort_values(
                "question_id"
            )
            if rows["question_id"].duplicated().any():
                raise ValueError(f"Duplicate questions for {system}/{episode}")
            questions = rows["question_id"].tolist()
            if expected_questions is None:
                expected_questions = questions
                question_counts[episode_index] = len(questions)
            elif questions != expected_questions:
                raise ValueError(f"Unpaired questions for {system}/{episode}")
            sufficient = rows["evidence_sufficient"].astype(bool).to_numpy()
            correct = rows["correct"].astype(bool).to_numpy()
            sufficient_counts[episode_index, system_index] = sufficient.sum()
            sufficient_correct_counts[episode_index, system_index] = np.count_nonzero(
                sufficient & correct
            )
    return (
        systems,
        episodes,
        question_counts,
        sufficient_counts,
        sufficient_correct_counts,
    )


def bootstrap_metrics(
    question_counts: np.ndarray,
    sufficient_counts: np.ndarray,
    sufficient_correct_counts: np.ndarray,
    samples: int,
    seed: int,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    episode_count = len(question_counts)
    system_count = sufficient_counts.shape[1]
    coverage = np.empty((samples, system_count), dtype=np.float64)
    utilization = np.empty_like(coverage)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        indices = rng.integers(
            0, episode_count, size=(stop - start, episode_count)
        )
        total_questions = question_counts[indices].sum(axis=1, dtype=np.float64)
        sufficient = sufficient_counts[indices].sum(axis=1, dtype=np.float64)
        sufficient_correct = sufficient_correct_counts[indices].sum(
            axis=1, dtype=np.float64
        )
        coverage[start:stop] = sufficient / total_questions[:, None]
        utilization[start:stop] = sufficient_correct / sufficient
    return coverage, utilization


def omnibus_max_range(estimates: np.ndarray, draws: np.ndarray) -> dict[str, float]:
    observed_range = float(estimates.max() - estimates.min())
    centered = draws - estimates
    null_ranges = centered.max(axis=1) - centered.min(axis=1)
    pvalue = (np.count_nonzero(null_ranges >= observed_range) + 1) / (
        len(draws) + 1
    )
    return {"observed_range": observed_range, "p_value": pvalue}


def pairwise_rows(
    systems: list[str], estimates: np.ndarray, draws: np.ndarray
) -> list[dict[str, object]]:
    rows = []
    for left in range(len(systems)):
        for right in range(left + 1, len(systems)):
            differences = draws[:, left] - draws[:, right]
            low, high = np.quantile(differences, [0.025, 0.975])
            rows.append(
                {
                    "system_a": systems[left],
                    "system_b": systems[right],
                    "difference_a_minus_b": estimates[left] - estimates[right],
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_bootstrap_two_sided": bootstrap_sign_pvalue(differences),
                }
            )
    adjusted = holm_adjust(
        [float(row["p_bootstrap_two_sided"]) for row in rows]
    )
    for row, adjusted_p in zip(rows, adjusted, strict=True):
        row["p_holm"] = adjusted_p
        row["significant_holm_0_05"] = adjusted_p < 0.05
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_table(
    title: str,
    rows: list[dict[str, object]],
    metric_key: str,
) -> list[str]:
    output = [
        f"## {title}",
        "",
        "| System | Numerator / denominator | Estimate | 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: float(item[metric_key]), reverse=True):
        output.append(
            f"| {row['system']} | {row[metric_key + '_numerator']} / "
            f"{row[metric_key + '_denominator']} | {float(row[metric_key]) * 100:.2f}% | "
            f"[{float(row[metric_key + '_ci95_low']) * 100:.2f}, "
            f"{float(row[metric_key + '_ci95_high']) * 100:.2f}]% |"
        )
    return output


def pair_table(title: str, rows: list[dict[str, object]]) -> list[str]:
    significant = [row for row in rows if row["significant_holm_0_05"]]
    output = [
        f"## {title} ({len(significant)} / {len(rows)} Holm-significant)",
        "",
        "| A | B | A-B | 95% CI | raw p | Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(significant, key=lambda item: float(item["p_holm"])):
        output.append(
            f"| {row['system_a']} | {row['system_b']} | "
            f"{float(row['difference_a_minus_b']) * 100:+.2f} pp | "
            f"[{float(row['ci95_low']) * 100:+.2f}, {float(row['ci95_high']) * 100:+.2f}] | "
            f"{float(row['p_bootstrap_two_sided']):.6g} | "
            f"{float(row['p_holm']):.6g} |"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--metric-description",
        default="the deterministic task-specific v2 native-evidence audit",
        help="Description written into the report for the input metric.",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    systems, episodes, questions, sufficient, sufficient_correct = (
        load_episode_counts(args.input)
    )
    coverage_estimates = sufficient.sum(axis=0) / questions.sum()
    utilization_estimates = sufficient_correct.sum(axis=0) / sufficient.sum(axis=0)
    coverage_draws, utilization_draws = bootstrap_metrics(
        questions,
        sufficient,
        sufficient_correct,
        args.bootstrap_samples,
        args.seed,
    )
    coverage_omnibus = omnibus_max_range(coverage_estimates, coverage_draws)
    utilization_omnibus = omnibus_max_range(utilization_estimates, utilization_draws)
    coverage_pairs = pairwise_rows(systems, coverage_estimates, coverage_draws)
    utilization_pairs = pairwise_rows(
        systems, utilization_estimates, utilization_draws
    )

    system_rows = []
    for index, system in enumerate(systems):
        coverage_low, coverage_high = np.quantile(
            coverage_draws[:, index], [0.025, 0.975]
        )
        utilization_low, utilization_high = np.quantile(
            utilization_draws[:, index], [0.025, 0.975]
        )
        system_rows.append(
            {
                "system": system,
                "coverage_numerator": int(sufficient[:, index].sum()),
                "coverage_denominator": int(questions.sum()),
                "coverage": coverage_estimates[index],
                "coverage_ci95_low": coverage_low,
                "coverage_ci95_high": coverage_high,
                "utilization_numerator": int(sufficient_correct[:, index].sum()),
                "utilization_denominator": int(sufficient[:, index].sum()),
                "utilization": utilization_estimates[index],
                "utilization_ci95_low": utilization_low,
                "utilization_ci95_high": utilization_high,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "system_estimates.csv", system_rows)
    write_csv(args.output_dir / "coverage_pairwise.csv", coverage_pairs)
    write_csv(args.output_dir / "utilization_pairwise.csv", utilization_pairs)
    omnibus = {
        "method": "paired episode-clustered centered max-range bootstrap",
        "episodes": len(episodes),
        "questions": int(questions.sum()),
        "systems": len(systems),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "coverage": coverage_omnibus,
        "utilization": utilization_omnibus,
    }
    (args.output_dir / "omnibus_tests.json").write_text(
        json.dumps(omnibus, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# MEME task-specific coverage and utilization significance",
        "",
        "Primary unit: 100 paired MEME episodes. Each bootstrap replicate resamples whole episodes with shared "
        "indices across all seven systems, retaining all before/after questions in the selected episodes. Metrics "
        f"use {args.metric_description}.",
        "",
        "## Omnibus tests",
        "",
        f"- Coverage range: **{coverage_omnibus['observed_range'] * 100:.2f} pp**; "
        f"centered max-range bootstrap p = **{coverage_omnibus['p_value']:.6g}**.",
        f"- Utilization range: **{utilization_omnibus['observed_range'] * 100:.2f} pp**; "
        f"centered max-range bootstrap p = **{utilization_omnibus['p_value']:.6g}**.",
        "",
    ]
    lines += metric_table("System coverage", system_rows, "coverage")
    lines += [""] + pair_table("Coverage pairwise tests", coverage_pairs)
    lines += [""] + metric_table("System utilization", system_rows, "utilization")
    lines += [""] + pair_table("Utilization pairwise tests", utilization_pairs)
    lines += [
        "",
        "Percentile intervals and two-sided bootstrap sign p-values use 50,000 replicates; pairwise p-values are "
        "Holm-corrected across 21 comparisons separately for each metric. The utilization denominator is "
        "system-specific, so utilization comparisons do not isolate a pure reader effect from differences in which "
        "questions each system makes evidence-sufficient. Deterministic verifier misspecification is outside the "
        "bootstrap uncertainty.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
