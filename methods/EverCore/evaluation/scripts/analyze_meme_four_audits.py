"""Unify and compare the four MEME evidence audits on paired questions.

The four primary audits cross two constructs (surface answer-value visibility
and task-specific path completeness) with two methods (deterministic and LLM).
Additional controls isolate input truncation, compare methods on identical
evidence views, and inspect value-to-path attrition and component agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr

from evaluation.scripts.analyze_meme_answer_only_evidence import verify_answer_only
from evaluation.scripts.analyze_meme_deterministic_evidence import DATA, qid_parts
from evaluation.scripts.judge_answerability_spotcheck import build_answer_index, context_excerpt


EVAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER = EVAL_DIR / "analysis/meme_answer_only_evidence_20260818/per_question.csv"
DEFAULT_SEMANTIC_DIR = EVAL_DIR / "analysis/meme_seven_systems_full_flash_20260817"
DEFAULT_TASK = EVAL_DIR / "analysis/meme_task_specific_evidence_v2_20260818/per_question.csv"
DEFAULT_LLM_PATH_DIR = EVAL_DIR / "analysis/meme_task_aware_llm_full_flash_20260818_v1"
DEFAULT_OUT = EVAL_DIR / "analysis/meme_four_audit_comparison_20260818"
CHAIN_TASKS = {("Tr", "after"), ("Del", "after"), ("Cas", "after"), ("Abs", "after")}


def bool_value(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def cohen_kappa(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return float("nan")
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_bootstrap_delta(
    frame: pd.DataFrame,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = frame.groupby("conversation_id", sort=True)
    episodes = []
    for _, rows in grouped:
        episodes.append(
            (
                int(rows[left].sum()),
                int(rows[right].sum()),
                len(rows),
            )
        )
    counts = np.asarray(episodes, dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        size = min(1000, samples - start)
        indices = rng.integers(0, len(counts), size=(size, len(counts)))
        sampled = counts[indices].sum(axis=1)
        draws[start : start + size] = (sampled[:, 1] - sampled[:, 0]) / sampled[:, 2]
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(frame[right]) - np.mean(frame[left])), float(low), float(high)


def paired_stats(
    frame: pd.DataFrame,
    left: str,
    right: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    valid = frame[left].notna() & frame[right].notna()
    rows = frame.loc[valid, ["conversation_id", left, right]].copy()
    rows[left] = rows[left].astype(bool)
    rows[right] = rows[right].astype(bool)
    left_values = rows[left].to_numpy(dtype=bool)
    right_values = rows[right].to_numpy(dtype=bool)
    left_only = int(np.count_nonzero(left_values & ~right_values))
    right_only = int(np.count_nonzero(~left_values & right_values))
    both = int(np.count_nonzero(left_values & right_values))
    neither = len(rows) - left_only - right_only - both
    discordant = left_only + right_only
    delta, low, high = paired_bootstrap_delta(rows, left, right, samples, seed)
    return {
        "valid_n": len(rows),
        "left_rate": float(np.mean(left_values)),
        "right_rate": float(np.mean(right_values)),
        "right_minus_left": delta,
        "delta_ci95_low": low,
        "delta_ci95_high": high,
        "both_positive": both,
        "left_only": left_only,
        "right_only": right_only,
        "both_negative": neither,
        "agreement": float(np.mean(left_values == right_values)),
        "cohen_kappa": cohen_kappa(left_values, right_values),
        "mcnemar_exact_p": (
            float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
        ),
    }


def summarize_audits(frame: pd.DataFrame, keys: list[str]) -> list[dict[str, Any]]:
    metrics = {
        "det_result": "det_result_full",
        "llm_result": "llm_result_excerpt",
        "det_path": "det_path_full",
        "llm_path": "llm_path_combined",
    }
    output = []
    grouped = frame.groupby(keys, sort=True, dropna=False)
    for group, rows in grouped:
        group = group if isinstance(group, tuple) else (group,)
        record: dict[str, Any] = dict(zip(keys, group))
        record["n"] = len(rows)
        record["accuracy"] = float(rows["correct"].mean())
        for label, column in metrics.items():
            valid = rows[column].notna()
            sufficient = valid & rows[column].astype("boolean").fillna(False)
            record[f"{label}_valid_n"] = int(valid.sum())
            record[f"{label}_coverage"] = float(rows.loc[valid, column].mean()) if valid.any() else ""
            record[f"{label}_sufficient_n"] = int(sufficient.sum())
            record[f"{label}_utilization"] = (
                float(rows.loc[sufficient, "correct"].mean()) if sufficient.any() else ""
            )
        output.append(record)
    return output


def comparison_rows(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    contrasts = [
        (
            "result_method_same_excerpt",
            "det_result_excerpt",
            "llm_result_excerpt",
            frame.index == frame.index,
            "Same 9,000-character gold-localized excerpt; isolates audit method approximately.",
        ),
        (
            "path_method_full_chain",
            "det_path_full",
            "llm_path_combined",
            frame["chain_sensitive"],
            "Complete context on chain-sensitive tasks; primary path-method comparison.",
        ),
        (
            "det_construct_full_chain",
            "det_result_full",
            "det_path_full",
            frame["chain_sensitive"],
            "Same deterministic method and complete context; changes result visibility to path completeness.",
        ),
        (
            "llm_construct_mixed_input_chain",
            "llm_result_excerpt",
            "llm_path_combined",
            frame["chain_sensitive"],
            "Descriptive only: result used an excerpt while path used complete context.",
        ),
        (
            "det_result_truncation",
            "det_result_full",
            "det_result_excerpt",
            frame.index == frame.index,
            "Same deterministic rule; quantifies full-context to semantic-excerpt input loss.",
        ),
    ]
    scopes: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("system", ["system"]),
        ("task_phase", ["task_type", "phase"]),
    ]
    output = []
    for contrast_index, (name, left, right, mask, note) in enumerate(contrasts):
        selected = frame.loc[mask].copy()
        for scope_type, keys in scopes:
            groups = [((), selected)] if not keys else selected.groupby(keys, sort=True)
            family: list[dict[str, Any]] = []
            for group, rows in groups:
                group = group if isinstance(group, tuple) else (group,)
                stats = paired_stats(
                    rows,
                    left,
                    right,
                    samples,
                    seed + contrast_index * 1000 + len(family),
                )
                scope = "all" if not keys else "/".join(str(value) for value in group)
                family.append(
                    {
                        "contrast": name,
                        "scope_type": scope_type,
                        "scope": scope,
                        "left_metric": left,
                        "right_metric": right,
                        **stats,
                        "interpretation": note,
                    }
                )
            adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in family])
            for row, pvalue in zip(family, adjusted):
                row["mcnemar_holm_p"] = pvalue
            output.extend(family)
    return output


def value_path_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    selected = frame[frame["task_type"].isin(["Tr", "Cas"]) & frame["chain_sensitive"]]
    output = []
    methods = [
        ("deterministic", "det_value_complete", "det_path_full"),
        ("llm_same_call", "llm_value_complete", "llm_path_combined"),
    ]
    scopes = [("overall", []), ("system", ["system"]), ("task", ["task_type"])]
    for method, value_column, path_column in methods:
        for scope_type, keys in scopes:
            groups = [((), selected)] if not keys else selected.groupby(keys, sort=True)
            for group, rows in groups:
                group = group if isinstance(group, tuple) else (group,)
                valid = rows[value_column].notna() & rows[path_column].notna()
                values = rows.loc[valid, value_column].astype(bool)
                paths = rows.loc[valid, path_column].astype(bool)
                value_n = int(values.sum())
                path_n = int(paths.sum())
                output.append(
                    {
                        "method": method,
                        "scope_type": scope_type,
                        "scope": "all" if not keys else "/".join(str(value) for value in group),
                        "valid_n": int(valid.sum()),
                        "value_complete_n": value_n,
                        "value_coverage": float(values.mean()),
                        "path_sufficient_n": path_n,
                        "path_coverage": float(paths.mean()),
                        "value_minus_path": float(values.mean() - paths.mean()),
                        "path_retention_given_value": (
                            float((values & paths).sum() / value_n) if value_n else ""
                        ),
                        "path_without_value_n": int((~values & paths).sum()),
                    }
                )
    return output


def criterion_validity_rows(
    frame: pd.DataFrame,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics = [
        ("det_result", "det_result_full"),
        ("llm_result", "llm_result_excerpt"),
        ("det_path", "det_path_full"),
        ("llm_path", "llm_path_combined"),
    ]
    output = []
    for metric_index, (label, column) in enumerate(metrics):
        rows = frame[frame[column].notna()].copy()
        rows[column] = rows[column].astype(bool)
        covered = rows[column]
        covered_accuracy = float(rows.loc[covered, "correct"].mean())
        uncovered_accuracy = float(rows.loc[~covered, "correct"].mean())
        episode_counts = []
        for _, episode_rows in rows.groupby("conversation_id", sort=True):
            episode_covered = episode_rows[column]
            episode_counts.append(
                (
                    int(episode_rows.loc[episode_covered, "correct"].sum()),
                    int(episode_covered.sum()),
                    int(episode_rows.loc[~episode_covered, "correct"].sum()),
                    int((~episode_covered).sum()),
                )
            )
        counts = np.asarray(episode_counts, dtype=np.int64)
        rng = np.random.default_rng(seed + metric_index)
        draws = np.empty(samples, dtype=float)
        for start in range(0, samples, 1000):
            size = min(1000, samples - start)
            indices = rng.integers(0, len(counts), size=(size, len(counts)))
            sampled = counts[indices].sum(axis=1)
            covered_draw = np.divide(
                sampled[:, 0], sampled[:, 1], out=np.full(size, np.nan), where=sampled[:, 1] > 0
            )
            uncovered_draw = np.divide(
                sampled[:, 2], sampled[:, 3], out=np.full(size, np.nan), where=sampled[:, 3] > 0
            )
            draws[start : start + size] = covered_draw - uncovered_draw
        finite_draws = draws[np.isfinite(draws)]
        low, high = np.quantile(finite_draws, [0.025, 0.975])
        by_system = rows.groupby("system", sort=True).agg(
            coverage=(column, "mean"), accuracy=("correct", "mean")
        )
        correlation_defined = (
            by_system["coverage"].nunique() > 1 and by_system["accuracy"].nunique() > 1
        )
        pearson = pearsonr(by_system["coverage"], by_system["accuracy"]) if correlation_defined else None
        spearman = spearmanr(by_system["coverage"], by_system["accuracy"]) if correlation_defined else None
        output.append(
            {
                "audit": label,
                "valid_n": len(rows),
                "covered_accuracy": covered_accuracy,
                "uncovered_accuracy": uncovered_accuracy,
                "accuracy_lift": covered_accuracy - uncovered_accuracy,
                "accuracy_lift_ci95_low": float(low),
                "accuracy_lift_ci95_high": float(high),
                "system_pearson_r": float(pearson.statistic) if pearson else "",
                "system_pearson_p": float(pearson.pvalue) if pearson else "",
                "system_spearman_rho": float(spearman.statistic) if spearman else "",
                "system_spearman_p": float(spearman.pvalue) if spearman else "",
                "system_n": len(by_system),
            }
        )
    return output


def rank_correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = {
        "det_result": "det_result_full",
        "llm_result": "llm_result_excerpt",
        "det_path": "det_path_full",
        "llm_path": "llm_path_combined",
    }
    rates = frame.groupby("system", sort=True)[list(metrics.values())].mean()
    output = []
    labels = list(metrics)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            result = spearmanr(rates[metrics[left]], rates[metrics[right]])
            output.append(
                {
                    "left_audit": left,
                    "right_audit": right,
                    "spearman_rho": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "systems": len(rates),
                }
            )
    return output


def component_rows(reasoning: pd.DataFrame, deterministic: pd.DataFrame) -> list[dict[str, Any]]:
    merged = reasoning.merge(
        deterministic,
        on=["system", "question_id"],
        how="inner",
        validate="one_to_one",
    )
    mappings = [
        ("Tr", "value_complete", "det_value_complete", "llm_value_complete"),
        ("Tr", "temporal_chain", "det_order_support", "llm_temporal_chain"),
        ("Del", "deletion_targeted", "det_deletion_support", "llm_deletion_targeted"),
        ("Cas", "value_complete", "det_value_complete", "llm_value_complete"),
        ("Cas", "dependency_rule", "det_dependency_support", "llm_dependency_rule"),
        ("Cas", "dependency_linked", "det_dependency_support", "llm_linked"),
        ("Cas", "upstream_trigger", "det_trigger_support", "llm_trigger_support"),
        ("Abs", "value_complete", "det_value_complete", "llm_value_complete"),
        ("Abs", "dependency_rule", "det_dependency_support", "llm_dependency_rule"),
        ("Abs", "dependency_linked", "det_dependency_support", "llm_linked"),
        ("Abs", "upstream_trigger", "det_trigger_support", "llm_trigger_support"),
    ]
    output = []
    for task, component, left, right in mappings:
        task_rows = merged[merged["task_type"] == task]
        for scope, rows in [("all", task_rows), *task_rows.groupby("system", sort=True)]:
            valid = rows[left].notna() & rows[right].notna()
            left_values = rows.loc[valid, left].astype(bool).to_numpy()
            right_values = rows.loc[valid, right].astype(bool).to_numpy()
            left_only = int(np.count_nonzero(left_values & ~right_values))
            right_only = int(np.count_nonzero(~left_values & right_values))
            discordant = left_only + right_only
            output.append(
                {
                    "task_type": task,
                    "component": component,
                    "scope": scope,
                    "valid_n": len(left_values),
                    "deterministic_rate": float(np.mean(left_values)),
                    "llm_rate": float(np.mean(right_values)),
                    "agreement": float(np.mean(left_values == right_values)),
                    "cohen_kappa": cohen_kappa(left_values, right_values),
                    "deterministic_only": left_only,
                    "llm_only": right_only,
                    "mcnemar_exact_p": (
                        float(binomtest(left_only, discordant, 0.5).pvalue)
                        if discordant
                        else 1.0
                    ),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    return "n/a" if value == "" or pd.isna(value) else f"{float(value) * 100:.2f}%"


def build_report(
    systems: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    value_path: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    criterion: list[dict[str, Any]],
    rank_correlations: list[dict[str, Any]],
) -> str:
    primary = {
        row["contrast"]: row
        for row in comparisons
        if row["scope_type"] == "overall"
    }
    attrition = {
        row["method"]: row
        for row in value_path
        if row["scope_type"] == "overall"
    }
    lines = [
        "# MEME four-audit synthesis",
        "",
        "The four primary cells cross construct (answer-value visibility vs task-specific path "
        "completeness) with method (deterministic vs LLM). All seven systems use exactly aligned "
        "question contexts and generated answers.",
        "",
        "## Four-audit system summary",
        "",
        "| System | Det result | LLM result | Det path | LLM path |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in systems:
        lines.append(
            f"| {row['system']} | {pct(row['det_result_coverage'])} | "
            f"{pct(row['llm_result_coverage'])} | {pct(row['det_path_coverage'])} | "
            f"{pct(row['llm_path_coverage'])} |"
        )
    result_method = primary["result_method_same_excerpt"]
    path_method = primary["path_method_full_chain"]
    truncation = primary["det_result_truncation"]
    lines.extend(
        [
            "",
            "## Primary paired controls",
            "",
            f"- Result method effect on the same 9,000-character excerpts: LLM minus deterministic "
            f"**{result_method['right_minus_left'] * 100:+.2f} pp**, 95% CI "
            f"[{result_method['delta_ci95_low'] * 100:+.2f}, {result_method['delta_ci95_high'] * 100:+.2f}] pp; "
            f"agreement {pct(result_method['agreement'])}, kappa {result_method['cohen_kappa']:.3f}.",
            f"- Path method effect on identical complete contexts for 3,458 chain-sensitive rows: "
            f"LLM minus deterministic **{path_method['right_minus_left'] * 100:+.2f} pp**, 95% CI "
            f"[{path_method['delta_ci95_low'] * 100:+.2f}, {path_method['delta_ci95_high'] * 100:+.2f}] pp; "
            f"agreement {pct(path_method['agreement'])}, kappa {path_method['cohen_kappa']:.3f}.",
            f"- Deterministic excerpt sensitivity: excerpt minus complete-context result coverage "
            f"**{truncation['right_minus_left'] * 100:+.2f} pp**, 95% CI "
            f"[{truncation['delta_ci95_low'] * 100:+.2f}, {truncation['delta_ci95_high'] * 100:+.2f}] pp.",
            "",
            "## Same-context value-to-path attrition",
            "",
        ]
    )
    for method in ("deterministic", "llm_same_call"):
        row = attrition[method]
        lines.append(
            f"- {method}: value coverage {pct(row['value_coverage'])}, path coverage "
            f"{pct(row['path_coverage'])}, gap {float(row['value_minus_path']) * 100:.2f} pp, "
            f"path retention given value {pct(row['path_retention_given_value'])}, "
            f"path-without-value violations {row['path_without_value_n']}."
        )
    lines.extend(
        [
            "",
            "This attrition analysis excludes Del-after and Abs-after because their gold results are "
            "status answers (non-disclosure and uncertainty). Visibility of an old value is not a "
            "positive result-sufficiency condition for either task.",
            "",
            "## Criterion validity and ranking stability",
            "",
            "| Audit | Accuracy if covered | Accuracy if uncovered | Lift (95% episode-bootstrap CI) | Coverage-accuracy Pearson/Spearman across 7 systems |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in criterion:
        lines.append(
            f"| {row['audit']} | {pct(row['covered_accuracy'])} | {pct(row['uncovered_accuracy'])} | "
            f"{row['accuracy_lift'] * 100:+.2f} pp "
            f"[{row['accuracy_lift_ci95_low'] * 100:+.2f}, {row['accuracy_lift_ci95_high'] * 100:+.2f}] | "
            f"{row['system_pearson_r']:.3f}/{row['system_spearman_rho']:.3f} |"
        )
    strongest = max(rank_correlations, key=lambda row: abs(float(row["spearman_rho"])))
    weakest = min(rank_correlations, key=lambda row: abs(float(row["spearman_rho"])))
    lines.extend(
        [
            "",
            f"Across-system audit rankings are most aligned for {strongest['left_audit']} vs "
            f"{strongest['right_audit']} (Spearman {strongest['spearman_rho']:.3f}) and least aligned "
            f"for {weakest['left_audit']} vs {weakest['right_audit']} "
            f"(Spearman {weakest['spearman_rho']:.3f}); with only seven systems these are descriptive.",
            "",
            "## Validity and sensitivity checks",
            "",
            f"- Exact context alignment: {sum(int(row['same_context_n']) for row in alignment)} / "
            f"{sum(int(row['n']) for row in alignment)} rows; exact generated-answer alignment is "
            f"{sum(int(row['same_answer_n']) for row in alignment)} / "
            f"{sum(int(row['n']) for row in alignment)}.",
            f"- The original LLM-result audit has {sum(int(row['missing_n']) for row in missing)} "
            "missing judgments. Per-system lower/upper bounds are saved; no missing label is imputed.",
            "- LLM path coverage is the combined metric: 3,458 chain-sensitive rows were re-judged "
            "on complete contexts, while simple task/phase rows inherit the original semantic judgment.",
            "- The raw LLM-result vs LLM-path construct contrast remains input-confounded and is "
            "reported as descriptive only. The same-call LLM value-to-path analysis is the cleaner "
            "construct comparison for Tr/Cas/Abs.",
            "",
            "## Output guide",
            "",
            "- `four_audit_by_system.csv` and `four_audit_by_system_task_phase.csv`: coverage and conditional utilization.",
            "- `paired_contrasts.csv`: paired rates, bootstrap CIs, agreement, kappa, and exact McNemar tests.",
            "- `same_context_value_path_attrition.csv`: deterministic and same-call LLM structural attrition.",
            "- `component_agreement.csv`: value/order/deletion/dependency/trigger agreement.",
            "- `criterion_validity.csv` and `audit_rank_correlations.csv`: correctness separation and ranking stability.",
            "- `context_alignment.csv`, `semantic_missing_bounds.csv`, and `disagreement_sample.csv`: audit controls.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer-only", type=Path, default=DEFAULT_ANSWER)
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--task-specific", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--llm-path-dir", type=Path, default=DEFAULT_LLM_PATH_DIR)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    if args.output_dir.exists():
        existing_manifest = args.output_dir / "manifest.json"
        if not existing_manifest.exists():
            raise FileExistsError(f"Refusing to overwrite unrecognized directory: {args.output_dir}")
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("version") != "meme-four-audit-comparison-v1":
            raise FileExistsError(f"Refusing to overwrite incompatible analysis: {args.output_dir}")

    answer = pd.read_csv(args.answer_only)
    task = pd.read_csv(args.task_specific)
    combined = pd.read_csv(args.llm_path_dir / "combined_coverage.csv")
    semantic = pd.read_csv(args.semantic_dir / "llm_answerability_spotcheck.csv")
    reasoning = pd.read_csv(args.llm_path_dir / "reasoning_audit.csv")
    keys = ["system", "question_id"]
    for name, table in {
        "answer": answer,
        "task": task,
        "combined": combined,
        "semantic": semantic,
        "reasoning": reasoning,
    }.items():
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate system/question keys in {name}")

    frame = answer[
        [
            "system", "run_name", "question_id", "conversation_id", "phase", "task_type",
            "correct", "answer_only_sufficient", "required_values", "missing_values",
        ]
    ].rename(columns={"answer_only_sufficient": "det_result_full"})
    task_columns = task[
        [
            "system", "question_id", "evidence_sufficient", "value_complete", "order_support",
            "dependency_rule_support", "trigger_support", "deletion_event_support",
        ]
    ].rename(
        columns={
            "evidence_sufficient": "det_path_full",
            "value_complete": "det_value_complete",
            "order_support": "det_order_support",
            "dependency_rule_support": "det_dependency_support",
            "trigger_support": "det_trigger_support",
            "deletion_event_support": "det_deletion_support",
        }
    )
    frame = frame.merge(task_columns, on=keys, validate="one_to_one")
    frame = frame.merge(
        combined[keys + ["evidence_sufficient"]].rename(
            columns={"evidence_sufficient": "llm_path_combined"}
        ),
        on=keys,
        validate="one_to_one",
    )
    semantic["llm_result_excerpt"] = semantic["llm_answerable"].map(bool_value)
    frame = frame.merge(
        semantic[keys + ["llm_result_excerpt", "gold_values_preview", "llm_notes"]],
        on=keys,
        validate="one_to_one",
    )
    frame["chain_sensitive"] = [
        (task_type, phase) in CHAIN_TASKS
        for task_type, phase in zip(frame["task_type"], frame["phase"])
    ]
    for column in [
        "det_result_full", "det_path_full", "det_value_complete", "det_order_support",
        "det_dependency_support", "det_trigger_support", "det_deletion_support",
        "llm_path_combined",
    ]:
        frame[column] = frame[column].map(bool_value).astype("boolean")
    frame["llm_result_excerpt"] = frame["llm_result_excerpt"].astype("boolean")

    episodes = json.loads(args.data.read_text(encoding="utf-8"))
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    manifest = json.loads((args.semantic_dir / "run_manifest.json").read_text(encoding="utf-8"))
    answer_index = build_answer_index(manifest)
    semantic_runs = {row["system"]: Path(row["run_dirs"][0]) for row in manifest}
    deterministic_answers: dict[str, dict[str, dict[str, Any]]] = {}
    alignment = []
    for system, rows in frame.groupby("system", sort=True):
        run_names = rows["run_name"].unique().tolist()
        if len(run_names) != 1:
            raise ValueError(f"Multiple deterministic runs for {system}: {run_names}")
        run = EVAL_DIR / "results" / run_names[0]
        items = json.loads((run / "answer_results.json").read_text(encoding="utf-8"))
        deterministic_answers[system] = {str(item["question_id"]): item for item in items}
        same_context = 0
        same_answer = 0
        for _, row in rows.iterrows():
            semantic_item = answer_index[("meme", system, "meme", "meme", row["question_id"])]
            deterministic_item = deterministic_answers[system][row["question_id"]]
            same_context += (
                str(semantic_item.get("formatted_context") or "")
                == str(deterministic_item.get("formatted_context") or "")
            )
            same_answer += (
                str(semantic_item.get("generated_answer") or "")
                == str(deterministic_item.get("generated_answer") or "")
            )
        alignment.append(
            {
                "system": system,
                "deterministic_run": run.name,
                "semantic_run": semantic_runs[system].name,
                "n": len(rows),
                "same_context_n": same_context,
                "same_answer_n": same_answer,
            }
        )
    if any(row["same_context_n"] != row["n"] for row in alignment):
        raise ValueError("Four-audit contexts are not exactly aligned")

    excerpt_values = []
    for _, row in frame.iterrows():
        item = answer_index[("meme", row["system"], "meme", "meme", row["question_id"])]
        excerpt = context_excerpt(
            str(item.get("formatted_context") or ""),
            str(row.get("gold_values_preview") or ""),
            9000,
        )
        episode_id, phase, question_index = qid_parts(row["question_id"])
        question = episodes_by_id[episode_id][f"{phase}_questions"]["questions"][question_index]
        excerpt_values.append(verify_answer_only(excerpt, question).sufficient)
    frame["det_result_excerpt"] = pd.Series(excerpt_values, dtype="boolean")

    reasoning_components = reasoning[
        keys
        + ["task_type"]
        + [
            "value_complete", "temporal_chain", "deletion_targeted", "dependency_rule",
            "upstream_trigger_state", "linked", "chain_sufficient", "audit_notes",
        ]
    ].copy()
    for column in [
        "value_complete", "temporal_chain", "deletion_targeted", "dependency_rule",
        "upstream_trigger_state", "linked", "chain_sufficient",
    ]:
        reasoning_components[column] = reasoning_components[column].map(bool_value).astype("boolean")
    reasoning_components = reasoning_components.rename(
        columns={
            "value_complete": "llm_value_complete",
            "temporal_chain": "llm_temporal_chain",
            "deletion_targeted": "llm_deletion_targeted",
            "dependency_rule": "llm_dependency_rule",
            "upstream_trigger_state": "llm_trigger_support",
            "linked": "llm_linked",
            "chain_sufficient": "llm_chain_sufficient",
            "audit_notes": "llm_path_notes",
        }
    )
    frame = frame.merge(
        reasoning_components.drop(columns="task_type"),
        on=keys,
        how="left",
        validate="one_to_one",
    )

    system_summary = summarize_audits(frame, ["system"])
    task_summary = summarize_audits(frame, ["system", "task_type", "phase"])
    comparisons = comparison_rows(frame, args.bootstrap_samples, args.seed)
    value_path = value_path_rows(frame)
    deterministic_components = frame[
        keys
        + [
            "det_value_complete", "det_order_support", "det_dependency_support",
            "det_trigger_support", "det_deletion_support",
        ]
    ]
    component_agreement = component_rows(reasoning_components, deterministic_components)
    criterion = criterion_validity_rows(frame, args.bootstrap_samples, args.seed)
    rank_correlations = rank_correlation_rows(frame)

    missing_bounds = []
    for system, rows in frame.groupby("system", sort=True):
        valid = rows["llm_result_excerpt"].notna()
        positive = int(rows.loc[valid, "llm_result_excerpt"].sum())
        missing_n = int((~valid).sum())
        missing_bounds.append(
            {
                "system": system,
                "n": len(rows),
                "valid_n": int(valid.sum()),
                "missing_n": missing_n,
                "observed_valid_coverage": positive / int(valid.sum()),
                "all_missing_false_lower": positive / len(rows),
                "all_missing_true_upper": (positive + missing_n) / len(rows),
            }
        )

    samples = []
    sample_specs = [
        ("result_same_excerpt", "det_result_excerpt", "llm_result_excerpt", frame.index == frame.index),
        ("path_full_chain", "det_path_full", "llm_path_combined", frame["chain_sensitive"]),
    ]
    for comparison, left, right, mask in sample_specs:
        candidates = frame.loc[mask & frame[left].notna() & frame[right].notna()].copy()
        candidates = candidates[candidates[left] != candidates[right]]
        candidates["direction"] = np.where(candidates[left], "deterministic_only", "llm_only")
        for (task_type, direction), rows in candidates.groupby(["task_type", "direction"], sort=True):
            for _, row in rows.sort_values(keys).head(5).iterrows():
                samples.append(
                    {
                        "comparison": comparison,
                        "task_type": task_type,
                        "direction": direction,
                        "system": row["system"],
                        "question_id": row["question_id"],
                        "phase": row["phase"],
                        "deterministic": bool(row[left]),
                        "llm": bool(row[right]),
                        "required_values": row["required_values"],
                        "missing_values": row["missing_values"],
                        "llm_result_notes": row["llm_notes"],
                        "llm_path_notes": row.get("llm_path_notes", ""),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "four_audit_by_system.csv", system_summary)
    write_csv(args.output_dir / "four_audit_by_system_task_phase.csv", task_summary)
    write_csv(args.output_dir / "paired_contrasts.csv", comparisons)
    write_csv(args.output_dir / "same_context_value_path_attrition.csv", value_path)
    write_csv(args.output_dir / "component_agreement.csv", component_agreement)
    write_csv(args.output_dir / "criterion_validity.csv", criterion)
    write_csv(args.output_dir / "audit_rank_correlations.csv", rank_correlations)
    write_csv(args.output_dir / "context_alignment.csv", alignment)
    write_csv(args.output_dir / "semantic_missing_bounds.csv", missing_bounds)
    write_csv(args.output_dir / "disagreement_sample.csv", samples)
    write_csv(
        args.output_dir / "per_question_four_audits.csv",
        frame[
            [
                "system", "question_id", "conversation_id", "task_type", "phase", "correct",
                "det_result_full", "det_result_excerpt", "llm_result_excerpt", "det_path_full",
                "llm_path_combined", "chain_sensitive",
            ]
        ].to_dict("records"),
    )
    (args.output_dir / "report.md").write_text(
        build_report(
            system_summary,
            comparisons,
            value_path,
            alignment,
            missing_bounds,
            criterion,
            rank_correlations,
        ),
        encoding="utf-8",
    )
    manifest_out = {
        "version": "meme-four-audit-comparison-v1",
        "questions_per_system": 1188,
        "systems": 7,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "semantic_excerpt_chars": 9000,
        "chain_sensitive_rows": int(frame["chain_sensitive"].sum()),
        "llm_result_valid_rows": int(frame["llm_result_excerpt"].notna().sum()),
        "definitions": {
            "det_result_full": "all normalized entity_values visible in complete context",
            "llm_result_excerpt": "generic semantic answerability on gold-localized excerpt",
            "det_path_full": "deterministic task-specific path verifier on complete context",
            "llm_path_combined": "full-context LLM path judgment on chain tasks; inherited semantic judgment otherwise",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote four-audit synthesis for {len(frame)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
