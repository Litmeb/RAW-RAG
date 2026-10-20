"""Compare seven-system result-to-path coverage loss on MEME Tr/Cas tasks.

The primary analysis excludes Del/Abs because their outputs are status answers
(non-disclosure/uncertainty), for which old-value visibility is not positive
result sufficiency.  Tr and Cas provide a clean nested estimand: result values
are complete, then task-specific chronology or causal propagation is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from evaluation.scripts.analyze_meme_task_specific_significance import (
    bootstrap_sign_pvalue,
    holm_adjust,
    omnibus_max_range,
    pairwise_rows,
)


EVAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DETERMINISTIC = EVAL_DIR / "analysis/meme_task_specific_evidence_v2_20260818/per_question.csv"
DEFAULT_LLM = EVAL_DIR / "analysis/meme_task_aware_llm_full_flash_20260818_v1/reasoning_audit.csv"
DEFAULT_FOUR = EVAL_DIR / "analysis/meme_four_audit_comparison_20260818/four_audit_by_system.csv"
DEFAULT_OUT = EVAL_DIR / "analysis/meme_result_path_gap_20260818"


def bool_value(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def prepare_frame(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    keys = ["system", "question_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate system/question keys in {path}")
    frame = frame[(frame["phase"] == "after") & frame["task_type"].isin(["Tr", "Cas"])].copy()
    path_column = "evidence_sufficient" if method == "deterministic" else "chain_sufficient"
    frame["value_complete"] = frame["value_complete"].map(bool_value)
    frame["path_sufficient"] = frame[path_column].map(bool_value)
    if frame[["value_complete", "path_sufficient"]].isna().any().any():
        raise ValueError(f"Invalid value/path labels in {path}")
    frame["value_complete"] = frame["value_complete"].astype(bool)
    frame["path_sufficient"] = frame["path_sufficient"].astype(bool)
    violations = frame[frame["path_sufficient"] & ~frame["value_complete"]]
    if not violations.empty:
        raise ValueError(f"Path-without-value violations in {method}: {len(violations)}")
    frame["method"] = method
    return frame[
        [
            "method", "system", "question_id", "conversation_id", "task_type",
            "value_complete", "path_sufficient",
        ]
    ]


def aligned_counts(
    frame: pd.DataFrame,
    systems: list[str],
    episodes: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    question_counts = np.zeros((len(episodes), len(systems)), dtype=np.int64)
    value_counts = np.zeros_like(question_counts)
    path_counts = np.zeros_like(question_counts)
    expected_by_episode: dict[str, list[str]] = {}
    for episode_index, episode in enumerate(episodes):
        episode_rows = frame[frame["conversation_id"] == episode]
        for system_index, system in enumerate(systems):
            rows = episode_rows[episode_rows["system"] == system].sort_values("question_id")
            questions = rows["question_id"].tolist()
            if episode not in expected_by_episode:
                expected_by_episode[episode] = questions
            elif questions != expected_by_episode[episode]:
                raise ValueError(f"Question alignment failed for {system}/{episode}")
            question_counts[episode_index, system_index] = len(rows)
            value_counts[episode_index, system_index] = int(rows["value_complete"].sum())
            path_counts[episode_index, system_index] = int(rows["path_sufficient"].sum())
    return question_counts, value_counts, path_counts


def bootstrap_metrics(
    counts_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    episode_count = next(iter(counts_by_method.values()))[0].shape[0]
    rng = np.random.default_rng(seed)
    output = {
        method: {
            metric: np.empty((samples, counts[0].shape[1]), dtype=float)
            for metric in ("value_coverage", "path_coverage", "gap", "retention", "loss")
        }
        for method, counts in counts_by_method.items()
    }
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, episode_count, size=(size, episode_count))
        for method, (questions, values, paths) in counts_by_method.items():
            sampled_questions = questions[indices].sum(axis=1, dtype=float)
            sampled_values = values[indices].sum(axis=1, dtype=float)
            sampled_paths = paths[indices].sum(axis=1, dtype=float)
            value_coverage = sampled_values / sampled_questions
            path_coverage = sampled_paths / sampled_questions
            retention = sampled_paths / sampled_values
            output[method]["value_coverage"][start : start + size] = value_coverage
            output[method]["path_coverage"][start : start + size] = path_coverage
            output[method]["gap"][start : start + size] = value_coverage - path_coverage
            output[method]["retention"][start : start + size] = retention
            output[method]["loss"][start : start + size] = 1 - retention
    return output


def estimates(counts: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    questions, values, paths = (array.sum(axis=0, dtype=float) for array in counts)
    value_coverage = values / questions
    path_coverage = paths / questions
    retention = paths / values
    return {
        "questions": questions,
        "values": values,
        "paths": paths,
        "value_coverage": value_coverage,
        "path_coverage": path_coverage,
        "gap": value_coverage - path_coverage,
        "retention": retention,
        "loss": 1 - retention,
    }


def system_rows(
    systems: list[str],
    method: str,
    point: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for index, system in enumerate(systems):
        row: dict[str, Any] = {
            "method": method,
            "system": system,
            "n": int(point["questions"][index]),
            "value_complete_n": int(point["values"][index]),
            "path_sufficient_n": int(point["paths"][index]),
        }
        for metric in ("value_coverage", "path_coverage", "gap", "retention", "loss"):
            low, high = np.quantile(draws[metric][:, index], [0.025, 0.975])
            row[metric] = float(point[metric][index])
            row[f"{metric}_ci95_low"] = float(low)
            row[f"{metric}_ci95_high"] = float(high)
        rows.append(row)
    return rows


def cross_method_rows(
    systems: list[str],
    points: dict[str, dict[str, np.ndarray]],
    draws: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    output = []
    for metric in ("gap", "retention"):
        family = []
        for index, system in enumerate(systems):
            differences = draws["llm"][metric][:, index] - draws["deterministic"][metric][:, index]
            low, high = np.quantile(differences, [0.025, 0.975])
            family.append(
                {
                    "metric": metric,
                    "system": system,
                    "deterministic": float(points["deterministic"][metric][index]),
                    "llm": float(points["llm"][metric][index]),
                    "llm_minus_deterministic": float(
                        points["llm"][metric][index] - points["deterministic"][metric][index]
                    ),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "p_bootstrap_two_sided": bootstrap_sign_pvalue(differences),
                }
            )
        adjusted = holm_adjust([float(row["p_bootstrap_two_sided"]) for row in family])
        for row, pvalue in zip(family, adjusted):
            row["p_holm"] = pvalue
            row["significant_holm_0_05"] = pvalue < 0.05
        output.extend(family)
    return output


def task_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    output = []
    for method, frame in frames.items():
        for (system, task), rows in frame.groupby(["system", "task_type"], sort=True):
            value_n = int(rows["value_complete"].sum())
            path_n = int(rows["path_sufficient"].sum())
            output.append(
                {
                    "method": method,
                    "system": system,
                    "task_type": task,
                    "n": len(rows),
                    "value_coverage": value_n / len(rows),
                    "path_coverage": path_n / len(rows),
                    "gap": (value_n - path_n) / len(rows),
                    "retention": path_n / value_n if value_n else "",
                    "value_complete_n": value_n,
                    "path_sufficient_n": path_n,
                }
            )
    return output


def correlation_rows(
    systems: list[str],
    points: dict[str, dict[str, np.ndarray]],
    four_audit_path: Path,
) -> list[dict[str, Any]]:
    accuracy_frame = pd.read_csv(four_audit_path).set_index("system")
    accuracy = np.asarray([accuracy_frame.loc[system, "accuracy"] for system in systems])
    output = []
    for method, point in points.items():
        for metric in ("gap", "retention"):
            for target_name, target in (
                ("overall_accuracy", accuracy),
                ("value_coverage", point["value_coverage"]),
            ):
                pearson = pearsonr(point[metric], target)
                spearman = spearmanr(point[metric], target)
                output.append(
                    {
                        "method": method,
                        "metric": metric,
                        "target": target_name,
                        "pearson_r": float(pearson.statistic),
                        "pearson_p": float(pearson.pvalue),
                        "spearman_rho": float(spearman.statistic),
                        "spearman_p": float(spearman.pvalue),
                        "systems": len(systems),
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(
    systems: list[str],
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    omnibus: dict[str, Any],
    cross_method: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
) -> str:
    indexed = {(row["method"], row["system"]): row for row in rows}
    task_index = {
        (row["method"], row["system"], row["task_type"]): row for row in tasks
    }
    lines = [
        "# MEME seven-system result-to-path coverage gap",
        "",
        "Primary estimand: Tr/Cas after-state questions only (264 per system). Del and Abs are "
        "excluded because their outputs are status answers rather than positive current values.",
        "",
        "## System gap and conditional path retention",
        "",
        "| System | Deterministic value→path | Det gap / retention | LLM value→path | LLM gap / retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for system in systems:
        det = indexed[("deterministic", system)]
        llm = indexed[("llm", system)]
        lines.append(
            f"| {system} | {pct(det['value_coverage'])}→{pct(det['path_coverage'])} | "
            f"{pct(det['gap'])} / {pct(det['retention'])} | "
            f"{pct(llm['value_coverage'])}→{pct(llm['path_coverage'])} | "
            f"{pct(llm['gap'])} / {pct(llm['retention'])} |"
        )
    lines.extend(
        [
            "",
            "Gap is the share of all Tr/Cas questions with complete result values but no complete path. "
            "Retention is `P(path sufficient | value complete)` and is the safer cross-system comparison "
            "when value coverage differs.",
            "",
            "## Cross-system tests",
            "",
        ]
    )
    for method in ("deterministic", "llm"):
        lines.append(
            f"- {method}: gap range {omnibus[method]['gap']['observed_range'] * 100:.2f} pp, "
            f"omnibus p={omnibus[method]['gap']['p_value']:.6g}; retention range "
            f"{omnibus[method]['retention']['observed_range'] * 100:.2f} pp, "
            f"omnibus p={omnibus[method]['retention']['p_value']:.6g}."
        )
    significant_gap = [
        row for row in cross_method if row["metric"] == "gap" and row["significant_holm_0_05"]
    ]
    significant_retention = [
        row
        for row in cross_method
        if row["metric"] == "retention" and row["significant_holm_0_05"]
    ]
    llm_retention_pairs = [
        row for row in pairwise if row["method"] == "llm" and row["metric"] == "retention"
    ]
    llm_retention_significant = [
        row for row in llm_retention_pairs if row["significant_holm_0_05"]
    ]
    llm_retention_accuracy = next(
        row
        for row in correlations
        if row["method"] == "llm"
        and row["metric"] == "retention"
        and row["target"] == "overall_accuracy"
    )
    lines.extend(
        [
            "",
            "## Method-sensitive systems",
            "",
            f"LLM vs deterministic gap differs after Holm correction for {len(significant_gap)}/7 systems: "
            + ", ".join(
                f"{row['system']} ({row['llm_minus_deterministic'] * 100:+.2f} pp)"
                for row in significant_gap
            )
            + ".",
            f"LLM vs deterministic retention differs for {len(significant_retention)}/7 systems: "
            + ", ".join(
                f"{row['system']} ({row['llm_minus_deterministic'] * 100:+.2f} pp)"
                for row in significant_retention
            )
            + ".",
            "",
            "## Task decomposition",
            "",
            "| System | Det Tr retention | LLM Tr retention | Det Cas retention | LLM Cas retention |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for system in systems:
        lines.append(
            f"| {system} | {pct(task_index[('deterministic', system, 'Tr')]['retention'])} | "
            f"{pct(task_index[('llm', system, 'Tr')]['retention'])} | "
            f"{pct(task_index[('deterministic', system, 'Cas')]['retention'])} | "
            f"{pct(task_index[('llm', system, 'Cas')]['retention'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- RAG and MemMachine have high value coverage and comparatively high path retention under both methods; "
            "they are the most robust high-coverage systems.",
            "- MemU is lowest under both methods, so its path bottleneck is not a verifier artifact; it combines low "
            "value coverage with very low conditional path retention.",
            "- Memos and Mem0 are method-sensitive. Deterministic proximity/cue rules grant much higher Cas retention "
            "than the LLM's semantic causal-link judgment, so claims that they preserve executable propagation should "
            "use the LLM estimate or present both.",
            "- Tr and Cas should not be averaged without decomposition: most systems preserve explicit Tr order once "
            "values are present, whereas Cas rule-trigger-target linking is the dominant structural loss.",
            "- A small absolute gap is not automatically good: it can arise from low value coverage. Report gap together "
            "with value coverage and conditional retention.",
            f"- LLM retention has {len(llm_retention_significant)}/{len(llm_retention_pairs)} Holm-significant "
            "system pairs. RAG and MemMachine are not significantly different; Memos and MemU are each below the "
            "higher-retention systems and also differ from one another.",
            f"- Across seven systems, LLM retention tracks Overall accuracy strongly "
            f"(Pearson {llm_retention_accuracy['pearson_r']:.3f}, Spearman "
            f"{llm_retention_accuracy['spearman_rho']:.3f}), but this n=7 correlation is exploratory.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, default=DEFAULT_DETERMINISTIC)
    parser.add_argument("--llm", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--four-audit-summary", type=Path, default=DEFAULT_FOUR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.bootstrap_samples < 1000:
        parser.error("--bootstrap-samples must be at least 1000")
    if args.output_dir.exists():
        manifest_path = args.output_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite unrecognized directory: {args.output_dir}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("version") != "meme-result-path-gap-v1":
            raise FileExistsError(f"Refusing to overwrite incompatible analysis: {args.output_dir}")

    frames = {
        "deterministic": prepare_frame(args.deterministic, "deterministic"),
        "llm": prepare_frame(args.llm, "llm"),
    }
    systems = sorted(frames["deterministic"]["system"].unique().tolist())
    episodes = sorted(frames["deterministic"]["conversation_id"].unique().tolist())
    expected_keys = set(map(tuple, frames["deterministic"][["system", "question_id"]].to_numpy()))
    llm_keys = set(map(tuple, frames["llm"][["system", "question_id"]].to_numpy()))
    if expected_keys != llm_keys:
        raise ValueError("Deterministic and LLM Tr/Cas rows are not aligned")
    counts = {
        method: aligned_counts(frame, systems, episodes) for method, frame in frames.items()
    }
    if not np.array_equal(counts["deterministic"][0], counts["llm"][0]):
        raise ValueError("Method question counts are not episode-aligned")
    points = {method: estimates(method_counts) for method, method_counts in counts.items()}
    draws = bootstrap_metrics(counts, args.bootstrap_samples, args.seed)

    rows = [
        row
        for method in ("deterministic", "llm")
        for row in system_rows(systems, method, points[method], draws[method])
    ]
    tasks = task_rows(frames)
    omnibus = {
        method: {
            metric: omnibus_max_range(points[method][metric], draws[method][metric])
            for metric in ("gap", "retention")
        }
        for method in ("deterministic", "llm")
    }
    pairwise = []
    for method in ("deterministic", "llm"):
        for metric in ("gap", "retention"):
            for row in pairwise_rows(systems, points[method][metric], draws[method][metric]):
                pairwise.append({"method": method, "metric": metric, **row})
    cross_method = cross_method_rows(systems, points, draws)
    correlations = correlation_rows(systems, points, args.four_audit_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "system_estimates.csv", rows)
    write_csv(args.output_dir / "task_breakdown.csv", tasks)
    write_csv(args.output_dir / "pairwise_comparisons.csv", pairwise)
    write_csv(args.output_dir / "cross_method_comparisons.csv", cross_method)
    write_csv(args.output_dir / "system_correlations.csv", correlations)
    (args.output_dir / "omnibus_tests.json").write_text(
        json.dumps(omnibus, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(
            systems,
            rows,
            tasks,
            omnibus,
            cross_method,
            pairwise,
            correlations,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": "meme-result-path-gap-v1",
                "scope": "Tr-after and Cas-after",
                "systems": systems,
                "episodes": len(episodes),
                "questions_per_system": int(points["deterministic"]["questions"][0]),
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote result-to-path gap analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
