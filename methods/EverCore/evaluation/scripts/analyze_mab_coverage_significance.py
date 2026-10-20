"""Paired cross-system tests for MemoryAgentBench semantic evidence coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

EVALUATION_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    EVALUATION_DIR
    / "analysis"
    / "mab_seven_systems_full_flash_20260816"
    / "llm_answerability_spotcheck.csv"
)
DEFAULT_OUTPUT = EVALUATION_DIR / "analysis" / "mab_coverage_significance_20260818"


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def load_coverage_blocks(path: Path) -> tuple[list[str], list[tuple[str, np.ndarray]]]:
    frame = pd.read_csv(path)
    required = {"system", "split", "source", "question_id", "llm_answerable"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    systems = sorted(frame["system"].unique().tolist())
    blocks: list[tuple[str, np.ndarray]] = []
    for (split, source), group in frame.groupby(["split", "source"], sort=True):
        if group.duplicated(["system", "question_id"]).any():
            raise ValueError(f"Duplicate system/question rows in {split}/{source}")
        expected_questions = None
        columns = []
        for system in systems:
            rows = group[group["system"] == system].sort_values("question_id")
            questions = rows["question_id"].tolist()
            if expected_questions is None:
                expected_questions = questions
            elif questions != expected_questions:
                raise ValueError(f"Unpaired questions for {system} in {split}/{source}")
            columns.append(rows["llm_answerable"].astype(bool).to_numpy(dtype=np.int8))
        blocks.append((f"{split}/{source}", np.column_stack(columns)))
    return systems, blocks


def cochran_q(matrix: np.ndarray) -> tuple[float, int, float]:
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("matrix must contain at least two paired systems")
    k = matrix.shape[1]
    column_totals = matrix.sum(axis=0, dtype=np.float64)
    row_totals = matrix.sum(axis=1, dtype=np.float64)
    total = float(column_totals.sum())
    denominator = k * total - float(np.square(row_totals).sum())
    if denominator == 0:
        raise ValueError("Cochran's Q is undefined without within-question variation")
    statistic = (k - 1) * (
        k * float(np.square(column_totals).sum()) - total**2
    ) / denominator
    degrees_of_freedom = k - 1
    return statistic, degrees_of_freedom, float(chi2.sf(statistic, degrees_of_freedom))


def bootstrap_coverage(
    blocks: list[tuple[str, np.ndarray]],
    samples: int,
    seed: int,
    batch_size: int = 500,
) -> np.ndarray:
    total_questions = sum(len(matrix) for _, matrix in blocks)
    system_count = blocks[0][1].shape[1]
    draws = np.empty((samples, system_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        batch_n = stop - start
        totals = np.zeros((batch_n, system_count), dtype=np.float64)
        for _, matrix in blocks:
            n = len(matrix)
            indices = rng.integers(0, n, size=(batch_n, n))
            totals += matrix[indices].sum(axis=1)
        draws[start:stop] = totals / total_questions
    return draws


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

    systems, blocks = load_coverage_blocks(args.input)
    matrix = np.vstack([block[1] for block in blocks])
    estimates = matrix.mean(axis=0)
    draws = bootstrap_coverage(blocks, args.bootstrap_samples, args.seed)
    q_statistic, q_df, q_pvalue = cochran_q(matrix)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    system_rows = []
    for index, system in enumerate(systems):
        low, high = np.quantile(draws[:, index], [0.025, 0.975])
        system_rows.append(
            {
                "system": system,
                "n": len(matrix),
                "semantically_sufficient_n": int(matrix[:, index].sum()),
                "semantic_evidence_coverage": estimates[index],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    write_csv(args.output_dir / "system_estimates.csv", system_rows)

    pair_rows = []
    for left in range(len(systems)):
        for right in range(left + 1, len(systems)):
            left_only = int(np.count_nonzero((matrix[:, left] == 1) & (matrix[:, right] == 0)))
            right_only = int(np.count_nonzero((matrix[:, left] == 0) & (matrix[:, right] == 1)))
            discordant = left_only + right_only
            pvalue = (
                binomtest(left_only, discordant, 0.5, alternative="two-sided").pvalue
                if discordant
                else 1.0
            )
            difference_draws = draws[:, left] - draws[:, right]
            low, high = np.quantile(difference_draws, [0.025, 0.975])
            pair_rows.append(
                {
                    "system_a": systems[left],
                    "system_b": systems[right],
                    "difference_a_minus_b": estimates[left] - estimates[right],
                    "ci95_low": low,
                    "ci95_high": high,
                    "a_sufficient_b_not": left_only,
                    "a_not_b_sufficient": right_only,
                    "mcnemar_exact_p": pvalue,
                }
            )
    adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in pair_rows])
    for row, adjusted_p in zip(pair_rows, adjusted, strict=True):
        row["p_holm"] = adjusted_p
        row["significant_holm_0_05"] = adjusted_p < 0.05
    write_csv(args.output_dir / "pairwise_comparisons.csv", pair_rows)

    omnibus = {
        "test": "Cochran's Q test for paired binary coverage outcomes",
        "statistic": q_statistic,
        "degrees_of_freedom": q_df,
        "p_value": q_pvalue,
        "n_questions": len(matrix),
        "n_systems": len(systems),
    }
    (args.output_dir / "omnibus_test.json").write_text(
        json.dumps(omnibus, indent=2) + "\n", encoding="utf-8"
    )

    significant = [row for row in pair_rows if row["significant_holm_0_05"]]
    lines = [
        "# MemoryAgentBench cross-system semantic-coverage significance",
        "",
        "Coverage is the full-audit binary semantic-sufficiency judgment. Cochran's Q tests the seven paired "
        "coverage proportions jointly. Pairwise exact McNemar p-values are Holm-corrected across all 21 system "
        "pairs. Source-stratified paired bootstrap intervals use 50,000 replicates.",
        "",
        "## Omnibus test",
        "",
        f"Cochran's Q({q_df}) = **{q_statistic:.3f}**, p = **{q_pvalue:.6g}**.",
        "",
        "## System estimates",
        "",
        "| System | Sufficient / N | Coverage | 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in sorted(system_rows, key=lambda item: float(item["semantic_evidence_coverage"]), reverse=True):
        lines.append(
            f"| {row['system']} | {row['semantically_sufficient_n']} / {row['n']} | "
            f"{float(row['semantic_evidence_coverage']) * 100:.2f}% | "
            f"[{float(row['ci95_low']) * 100:.2f}, {float(row['ci95_high']) * 100:.2f}]% |"
        )
    lines += [
        "",
        f"## Holm-significant pairwise differences ({len(significant)} / {len(pair_rows)})",
        "",
        "| A | B | A-B | 95% CI | discordant A/B | exact p | Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(significant, key=lambda item: float(item["p_holm"])):
        lines.append(
            f"| {row['system_a']} | {row['system_b']} | "
            f"{float(row['difference_a_minus_b']) * 100:+.2f} pp | "
            f"[{float(row['ci95_low']) * 100:+.2f}, {float(row['ci95_high']) * 100:+.2f}] | "
            f"{row['a_sufficient_b_not']} / {row['a_not_b_sufficient']} | "
            f"{float(row['mcnemar_exact_p']):.6g} | {float(row['p_holm']):.6g} |"
        )
    lines += [
        "",
        "The tests cover question-sampling uncertainty and paired binary differences in the saved semantic judgments; "
        "they do not cover semantic-judge rerun variance.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
