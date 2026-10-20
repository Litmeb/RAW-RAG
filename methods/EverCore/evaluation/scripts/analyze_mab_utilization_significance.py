"""Cross-system significance tests for MemoryAgentBench reader utilization.

The input is the full semantic-sufficiency audit.  Questions are resampled with
the same indices for every system within each benchmark source, preserving both
cross-system pairing and the original source mixture.  Reader utilization is
recomputed as ``sum(correct & sufficient) / sum(sufficient)`` in every bootstrap
replicate.
"""

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
    / "mab_seven_systems_full_flash_20260816"
    / "llm_answerability_spotcheck.csv"
)
DEFAULT_OUTPUT = (
    EVALUATION_DIR / "analysis" / "mab_utilization_significance_20260818"
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


def load_matrices(path: Path) -> tuple[list[str], list[tuple[str, np.ndarray, np.ndarray]]]:
    frame = pd.read_csv(path)
    required = {
        "system",
        "split",
        "source",
        "question_id",
        "correctish",
        "llm_answerable",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    systems = sorted(frame["system"].unique().tolist())
    blocks: list[tuple[str, np.ndarray, np.ndarray]] = []
    for (split, source), group in frame.groupby(["split", "source"], sort=True):
        if group.duplicated(["system", "question_id"]).any():
            raise ValueError(f"Duplicate system/question rows in {split}/{source}")
        expected_questions = None
        sufficient_columns = []
        correct_columns = []
        for system in systems:
            rows = group[group["system"] == system].sort_values("question_id")
            questions = rows["question_id"].tolist()
            if expected_questions is None:
                expected_questions = questions
            elif questions != expected_questions:
                raise ValueError(f"Unpaired questions for {system} in {split}/{source}")
            sufficient = rows["llm_answerable"].astype(bool).to_numpy(dtype=np.int8)
            correct = rows["correctish"].astype(bool).to_numpy(dtype=np.int8)
            sufficient_columns.append(sufficient)
            correct_columns.append(sufficient * correct)
        blocks.append(
            (
                f"{split}/{source}",
                np.column_stack(sufficient_columns),
                np.column_stack(correct_columns),
            )
        )
    return systems, blocks


def run_bootstrap(
    blocks: list[tuple[str, np.ndarray, np.ndarray]],
    samples: int,
    seed: int,
    batch_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sufficient = sum((block[1].sum(axis=0) for block in blocks), start=np.zeros(blocks[0][1].shape[1]))
    correct = sum((block[2].sum(axis=0) for block in blocks), start=np.zeros(blocks[0][1].shape[1]))
    estimates = correct / sufficient
    draws = np.empty((samples, len(estimates)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        batch_n = stop - start
        batch_sufficient = np.zeros((batch_n, len(estimates)), dtype=np.float64)
        batch_correct = np.zeros_like(batch_sufficient)
        for _, sufficient_matrix, correct_matrix in blocks:
            n = len(sufficient_matrix)
            indices = rng.integers(0, n, size=(batch_n, n))
            batch_sufficient += sufficient_matrix[indices].sum(axis=1)
            batch_correct += correct_matrix[indices].sum(axis=1)
        draws[start:stop] = batch_correct / batch_sufficient
    return sufficient, correct, draws


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")

    systems, blocks = load_matrices(args.input)
    sufficient, correct, draws = run_bootstrap(
        blocks, args.bootstrap_samples, args.seed
    )
    estimates = correct / sufficient
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_rows = []
    for index, system in enumerate(systems):
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        system_rows.append(
            {
                "system": system,
                "sufficient_n": int(sufficient[index]),
                "sufficient_and_correct_n": int(correct[index]),
                "reader_utilization": estimates[index],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    write_csv(args.output_dir / "system_estimates.csv", system_rows)

    pair_rows = []
    for left in range(len(systems)):
        for right in range(left + 1, len(systems)):
            differences = draws[:, left] - draws[:, right]
            low, high = np.quantile(differences, [0.025, 0.975])
            pair_rows.append(
                {
                    "system_a": systems[left],
                    "system_b": systems[right],
                    "difference_a_minus_b": estimates[left] - estimates[right],
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_bootstrap_two_sided": bootstrap_sign_pvalue(differences),
                }
            )
    adjusted = holm_adjust([float(row["p_bootstrap_two_sided"]) for row in pair_rows])
    for row, p_adjusted in zip(pair_rows, adjusted, strict=True):
        row["p_holm"] = p_adjusted
        row["significant_holm_0_05"] = p_adjusted < 0.05
    write_csv(args.output_dir / "pairwise_comparisons.csv", pair_rows)

    observed_range = float(estimates.max() - estimates.min())
    centered_draws = draws - estimates
    null_ranges = centered_draws.max(axis=1) - centered_draws.min(axis=1)
    omnibus_p = (np.count_nonzero(null_ranges >= observed_range) + 1) / (
        args.bootstrap_samples + 1
    )
    omnibus = {
        "test": "paired source-stratified bootstrap max-range test",
        "observed_range": observed_range,
        "p_value": omnibus_p,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    (args.output_dir / "omnibus_test.json").write_text(
        json.dumps(omnibus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    significant = [row for row in pair_rows if row["significant_holm_0_05"]]
    lines = [
        "# MemoryAgentBench cross-system reader-utilization significance",
        "",
        f"Input: `{args.input.as_posix()}`",
        "",
        "Questions were resampled with shared indices across all seven systems within each of the 14 sources. "
        "Each replicate recomputed P(correct | semantically sufficient context). Pairwise two-sided bootstrap "
        "sign p-values were corrected across all 21 comparisons with Holm's method.",
        "",
        "## Omnibus test",
        "",
        f"Observed max-minus-min utilization range: **{observed_range * 100:.2f} pp**. "
        f"Centered max-range bootstrap p = **{omnibus_p:.6g}**.",
        "",
        "## System estimates",
        "",
        "| System | Sufficient | Correct | Utilization | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(system_rows, key=lambda item: float(item["reader_utilization"]), reverse=True):
        lines.append(
            f"| {row['system']} | {row['sufficient_n']} | {row['sufficient_and_correct_n']} | "
            f"{float(row['reader_utilization']) * 100:.2f}% | "
            f"[{float(row['ci95_low']) * 100:.2f}, {float(row['ci95_high']) * 100:.2f}]% |"
        )
    lines += [
        "",
        "## Holm-significant pairwise differences",
        "",
        "| A | B | A-B | 95% CI | raw p | Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(significant, key=lambda item: float(item["p_holm"])):
        lines.append(
            f"| {row['system_a']} | {row['system_b']} | "
            f"{float(row['difference_a_minus_b']) * 100:+.2f} pp | "
            f"[{float(row['ci95_low']) * 100:+.2f}, {float(row['ci95_high']) * 100:+.2f}] | "
            f"{float(row['p_bootstrap_two_sided']):.6g} | {float(row['p_holm']):.6g} |"
        )
    if not significant:
        lines.append("| None | | | | | |")
    lines += [
        "",
        "## Interpretation limit",
        "",
        "The sufficient-context denominator is system-specific. Therefore this tests differences between the "
        "observed conditional utilization estimands; it does not prove a pure reader effect independent of which "
        "questions each system made answerable. The bootstrap covers question-sampling uncertainty, not semantic-judge "
        "or answer-generation rerun variance.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
