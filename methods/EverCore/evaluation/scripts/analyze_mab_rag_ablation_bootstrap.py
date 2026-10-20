"""Paired bootstrap significance analysis for MemoryAgentBench RAG ablations.

The analysis uses saved per-question ``metric_value`` values and reproduces the
benchmark macro aggregation used by the RAG ablation table. Questions are
resampled *as pairs* within each source, source scores are aggregated with the
official task weights, and the four task scores are macro-averaged for Total.
No model or judge calls are made.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "analysis" / "mab_rag_ablation_bootstrap_20260818"


@dataclass(frozen=True)
class SourceSpec:
    split: str
    source: str
    path_source: str | None = None

    @property
    def directory_source(self) -> str:
        return self.path_source or self.source

    @property
    def key(self) -> str:
        return f"{self.split}/{self.source}"


SOURCES = (
    SourceSpec("Accurate_Retrieval", "eventqa_full"),
    SourceSpec("Accurate_Retrieval", "longmemeval_s*", "longmemeval_s_"),
    SourceSpec("Accurate_Retrieval", "ruler_qa1_197K"),
    SourceSpec("Accurate_Retrieval", "ruler_qa2_421K"),
    SourceSpec("Test_Time_Learning", "recsys_redial_full"),
    SourceSpec("Test_Time_Learning", "icl_banking77_5900shot_balance"),
    SourceSpec("Test_Time_Learning", "icl_clinic150_7050shot_balance"),
    SourceSpec("Test_Time_Learning", "icl_nlu_8296shot_balance"),
    SourceSpec("Test_Time_Learning", "icl_trec_coarse_6600shot_balance"),
    SourceSpec("Test_Time_Learning", "icl_trec_fine_6400shot_balance"),
    SourceSpec("Long_Range_Understanding", "detective_qa"),
    SourceSpec("Long_Range_Understanding", "infbench_sum_eng_shots2"),
    SourceSpec("Conflict_Resolution", "factconsolidation_mh_262k"),
    SourceSpec("Conflict_Resolution", "factconsolidation_sh_262k"),
)

BASELINE_SUFFIX = "normal-recover-rerank"
CONDITION_SUFFIXES: dict[str, tuple[str, ...]] = {
    "embedding_only_top15": ("embedding-only-topk15-20260815",),
    "topk_1": ("fixed-topk-1-20260808",),
    "topk_5": ("fixed-topk-5-20260808",),
    "topk_10": ("fixed-topk-10-20260808",),
    "topk_30": ("fixed-topk-30-20260808",),
    "topk_60": ("fixed-topk-60-20260815", "fixed-topk-60-20260809"),
    "topk_120": ("fixed-topk-120-20260815", "fixed-topk-120-20260809"),
    "chunk_128": ("128-16-128-15",),
    "chunk_256_no_overlap": ("256-0-128-15",),
    "chunk_512": ("512-64-128-15",),
    "chunk_1024": ("1024-128-128-15",),
    "display_turn_dedup": ("turn-dedup-topk15-20260815",),
}
CONDITION_LABELS = {
    "embedding_only_top15": "无 rerank：embedding-only top-15",
    "topk_1": "固定最终 top-k=1",
    "topk_5": "固定最终 top-k=5",
    "topk_10": "固定最终 top-k=10",
    "topk_30": "固定最终 top-k=30",
    "topk_60": "固定最终 top-k=60",
    "topk_120": "固定最终 top-k=120",
    "chunk_128": "Chunk=128（overlap=16）",
    "chunk_256_no_overlap": "Chunk=256（overlap=0）",
    "chunk_512": "Chunk=512（overlap=64）",
    "chunk_1024": "Chunk=1024（overlap=128）",
    "display_turn_dedup": "display turn dedup",
}
SCOPE_LABELS = {
    "Accurate_Retrieval": "Accurate Retrieval",
    "Test_Time_Learning": "Test Time Learning",
    "Long_Range_Understanding": "Long Range Understanding",
    "Conflict_Resolution": "Conflict Resolution",
}
CHUNK_CONDITIONS = {"chunk_128", "chunk_256_no_overlap", "chunk_512", "chunk_1024"}

# Each mapping reproduces the published MemoryAgentBench macro score. TTL gives
# ReDial half of the task score and the five ICL sources the other half.
SCOPE_WEIGHTS: dict[str, dict[str, float]] = {
    "Accurate_Retrieval": {
        source.key: 0.25 for source in SOURCES if source.split == "Accurate_Retrieval"
    },
    "Test_Time_Learning": {
        "Test_Time_Learning/recsys_redial_full": 0.5,
        **{
            source.key: 0.1
            for source in SOURCES
            if source.split == "Test_Time_Learning" and source.source.startswith("icl_")
        },
    },
    "Long_Range_Understanding": {
        source.key: 0.5
        for source in SOURCES
        if source.split == "Long_Range_Understanding"
    },
    "Conflict_Resolution": {
        source.key: 0.5 for source in SOURCES if source.split == "Conflict_Resolution"
    },
}
SCOPE_WEIGHTS["Total"] = {
    source_key: task_weight * 0.25
    for task in (
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    )
    for source_key, task_weight in SCOPE_WEIGHTS[task].items()
}
SCOPE_ORDER = (
    "Total",
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(flatten_results(item))
        return rows
    return [value] if isinstance(value, dict) else []


def score_from_item(item: dict[str, Any]) -> float:
    value = item.get("metric_value")
    if value is not None:
        return float(value)
    for key in ("score", "correct", "is_correct"):
        if item.get(key) is not None:
            return float(item[key])
    raise ValueError(f"Evaluation item has no numeric score: {item.get('question_id')}")


def load_score_map(run_dir: Path, expected_source: str) -> dict[str, float]:
    eval_path = run_dir / "eval_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json: {eval_path}")
    data = load_json(eval_path)
    rows = flatten_results(data.get("detailed_results"))
    scores: dict[str, float] = {}
    observed_sources = set()
    for item in rows:
        question_id = item.get("question_id")
        if not question_id:
            continue
        question_id = str(question_id)
        if question_id in scores:
            raise ValueError(f"Duplicate question_id {question_id!r} in {eval_path}")
        scores[question_id] = score_from_item(item)
        if item.get("source"):
            observed_sources.add(str(item["source"]))
    if not scores:
        raise ValueError(f"No per-question scores found in {eval_path}")
    if observed_sources and observed_sources != {expected_source}:
        raise ValueError(
            f"Unexpected source metadata in {eval_path}: {sorted(observed_sources)}; "
            f"expected {expected_source!r}"
        )
    return scores


def run_dir(results_dir: Path, source: SourceSpec, suffix: str) -> Path:
    name = (
        f"memoryagentbench-rag-{source.split}-{source.directory_source}-{suffix}"
    )
    return results_dir / name


def resolve_variant_dir(
    results_dir: Path, condition: str, source: SourceSpec
) -> tuple[Path, bool]:
    for suffix in CONDITION_SUFFIXES[condition]:
        candidate = run_dir(results_dir, source, suffix)
        if (candidate / "eval_results.json").exists():
            return candidate, False
    if condition in CHUNK_CONDITIONS and source.split == "Conflict_Resolution":
        # FactConsolidation uses its independent fact-level chunking override, so
        # the published chunk-size table carries the unchanged baseline score.
        return run_dir(results_dir, source, BASELINE_SUFFIX), True
    tried = [str(run_dir(results_dir, source, suffix)) for suffix in CONDITION_SUFFIXES[condition]]
    raise FileNotFoundError(
        f"No completed run for {condition} / {source.key}; tried: {tried}"
    )


def stable_condition_seed(base_seed: int, condition: str) -> int:
    digest = hashlib.sha256(condition.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def bootstrap_source_mean_differences(
    differences: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    batch_size: int = 1000,
) -> np.ndarray:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("differences must be a non-empty one-dimensional array")
    output = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        output[start:stop] = differences[indices].mean(axis=1)
    return output


def weighted_score(source_means: dict[str, float], weights: dict[str, float]) -> float:
    return sum(source_means[key] * weight for key, weight in weights.items())


def weighted_distribution(
    source_distributions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    return sum(
        (source_distributions[key] * weight for key, weight in weights.items()),
        np.zeros_like(next(iter(source_distributions.values()))),
    )


def bootstrap_sign_pvalue(distribution: np.ndarray) -> float:
    samples = len(distribution)
    lower = (int(np.count_nonzero(distribution <= 0.0)) + 1) / (samples + 1)
    upper = (int(np.count_nonzero(distribution >= 0.0)) + 1) / (samples + 1)
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    for scope in SCOPE_ORDER:
        scope_rows = [row for row in rows if row["scope"] == scope]
        ordered = sorted(scope_rows, key=lambda row: float(row["p_bootstrap_two_sided"]))
        running_max = 0.0
        total = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, float(row["p_bootstrap_two_sided"]) * (total - rank))
            running_max = max(running_max, adjusted)
            row["p_holm_within_scope"] = running_max
            row["significant_holm_0_05"] = running_max < 0.05


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f}"


def p_text(value: float, samples: int) -> str:
    resolution = 2.0 / (samples + 1)
    return f"<{resolution:.4f}" if value <= resolution else f"{value:.4f}"


def render_combined_table(summary_rows: list[dict[str, Any]], samples: int) -> str:
    indexed = {
        (str(row["condition"]), str(row["scope"])): row for row in summary_rows
    }
    first_condition = next(iter(CONDITION_SUFFIXES))
    baseline = {
        scope: float(indexed[(first_condition, scope)]["baseline_score"])
        for scope in SCOPE_ORDER
    }

    def score_cell(condition: str, scope: str) -> str:
        row = indexed[(condition, scope)]
        marker = " ✓" if row["significant_holm_0_05"] else ""
        inherited = "†" if condition in CHUNK_CONDITIONS and scope == "Conflict_Resolution" else ""
        return f"{pct(float(row['variant_score']))}{marker}{inherited}"

    lines = [
        "## Ablation 与统计显著性合并表",
        "",
        "| RAG 条件 | Accurate Retrieval | Test Time Learning | Long Range Understanding | Conflict Resolution | Total | Total 相对 baseline Δ (pp) | Paired 95% CI (pp) | Holm p | 统计结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| Baseline：chunk=256，overlap=32，embed@128 → rerank@15 | {pct(baseline['Accurate_Retrieval'])} | {pct(baseline['Test_Time_Learning'])} | {pct(baseline['Long_Range_Understanding'])} | {pct(baseline['Conflict_Resolution'])} | **{pct(baseline['Total'])}** | — | — | — | 参照组 |",
    ]
    for condition in CONDITION_SUFFIXES:
        total = indexed[(condition, "Total")]
        significant = bool(total["significant_holm_0_05"])
        delta = float(total["delta"])
        task_significant = [
            SCOPE_LABELS[scope]
            for scope in SCOPE_ORDER[1:]
            if indexed[(condition, scope)]["significant_holm_0_05"]
        ]
        if significant:
            conclusion = "显著上升" if delta > 0 else "显著下降"
        elif task_significant:
            conclusion = f"Total 不显著；{', '.join(task_significant)} 显著"
        else:
            conclusion = "不显著"
        lines.append(
            f"| {CONDITION_LABELS[condition]} | "
            f"{score_cell(condition, 'Accurate_Retrieval')} | "
            f"{score_cell(condition, 'Test_Time_Learning')} | "
            f"{score_cell(condition, 'Long_Range_Understanding')} | "
            f"{score_cell(condition, 'Conflict_Resolution')} | "
            f"**{score_cell(condition, 'Total')}** | {pp(delta)} | "
            f"[{pp(float(total['ci_low']))}, {pp(float(total['ci_high']))}] | "
            f"{p_text(float(total['p_holm_within_scope']), samples)} | {conclusion} |"
        )
    lines.extend(
        [
            "",
            "- `✓` 表示该列相对 baseline 经 Holm 校正后 `p < 0.05`。",
            "- `†` 表示 Conflict Resolution 沿用不变的 fact-level baseline，没有按通用 token-window chunk 配置重跑。",
            "",
        ]
    )
    return "\n".join(lines)


def render_report(summary_rows: list[dict[str, Any]], samples: int, seed: int) -> str:
    by_scope = {
        scope: [row for row in summary_rows if row["scope"] == scope]
        for scope in SCOPE_ORDER
    }
    lines = [
        "# MemoryAgentBench RAG Ablation Paired Bootstrap",
        "",
        "## Method",
        "",
        f"- Paired question-level bootstrap: **{samples:,}** resamples, seed **{seed}**.",
        "- Resampling is independent within each of the 14 benchmark sources; baseline and variant always use the same sampled question indices.",
        "- Source means are combined using the published MemoryAgentBench macro convention: Accurate Retrieval averages 4 sources; Test Time Learning is `mean(ReDial, mean(5 ICL sources))`; Long Range Understanding and Conflict Resolution each average 2 sources; Total averages the four task scores.",
        "- Scores use each evaluator's continuous `metric_value` (important for ReDial and InfBench).",
        f"- The 95% intervals are percentile paired-bootstrap intervals. `p_boot` is a two-sided bootstrap sign probability with a finite-sample correction. `p_Holm` adjusts the {len(CONDITION_SUFFIXES)} comparisons separately within each reported scope.",
        "- Chunk-size variants inherit the unchanged fact-level Conflict Resolution baseline because that split has an independent source-specific chunking override.",
        "",
        render_combined_table(summary_rows, samples),
        "## Total",
        "",
        "| Condition | Score | Δ vs baseline (pp) | Paired 95% CI (pp) | p_boot | p_Holm | Holm sig. | W / T / L |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in by_scope["Total"]:
        lines.append(
            f"| {row['condition']} | {pct(row['variant_score'])} | {pp(row['delta'])} | "
            f"[{pp(row['ci_low'])}, {pp(row['ci_high'])}] | "
            f"{p_text(float(row['p_bootstrap_two_sided']), samples)} | "
            f"{p_text(float(row['p_holm_within_scope']), samples)} | "
            f"{'yes' if row['significant_holm_0_05'] else 'no'} | "
            f"{row['wins']} / {row['ties']} / {row['losses']} |"
        )
    for scope in SCOPE_ORDER[1:]:
        lines.extend(
            [
                "",
                f"## {scope.replace('_', ' ')}",
                "",
                "| Condition | Baseline | Variant | Δ (pp) | Paired 95% CI (pp) | p_Holm | W / T / L |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in by_scope[scope]:
            lines.append(
                f"| {row['condition']} | {pct(row['baseline_score'])} | "
                f"{pct(row['variant_score'])} | {pp(row['delta'])} | "
                f"[{pp(row['ci_low'])}, {pp(row['ci_high'])}] | "
                f"{p_text(float(row['p_holm_within_scope']), samples)} | "
                f"{row['wins']} / {row['ties']} / {row['losses']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            f"A confidence interval excluding zero supports a directional difference under question-level resampling. Holm significance is the more conservative flag when interpreting all {len(CONDITION_SUFFIXES)} ablations together. Questions from the same long memory instance may be correlated, so these intervals do not capture model rerun/judge variance or between-dataset generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(results_dir: Path, output_dir: Path, samples: int, seed: int) -> None:
    baseline_scores: dict[str, dict[str, float]] = {}
    baseline_paths: dict[str, Path] = {}
    for source in SOURCES:
        path = run_dir(results_dir, source, BASELINE_SUFFIX)
        baseline_scores[source.key] = load_score_map(path, source.source)
        baseline_paths[source.key] = path

    summary_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "bootstrap_samples": samples,
        "seed": seed,
        "aggregation_weights": SCOPE_WEIGHTS,
        "conditions": {},
    }

    for condition in CONDITION_SUFFIXES:
        rng = np.random.default_rng(stable_condition_seed(seed, condition))
        baseline_means: dict[str, float] = {}
        variant_means: dict[str, float] = {}
        boot_differences: dict[str, np.ndarray] = {}
        pair_counts: dict[str, int] = {}
        source_outcomes: dict[str, tuple[int, int, int]] = {}
        inherited: dict[str, bool] = {}
        condition_manifest: dict[str, Any] = {}

        for source in SOURCES:
            variant_path, is_inherited = resolve_variant_dir(results_dir, condition, source)
            candidate = load_score_map(variant_path, source.source)
            baseline = baseline_scores[source.key]
            if set(candidate) != set(baseline):
                missing = sorted(set(baseline) - set(candidate))[:5]
                extra = sorted(set(candidate) - set(baseline))[:5]
                raise ValueError(
                    f"Question mismatch for {condition} / {source.key}: "
                    f"baseline={len(baseline)}, variant={len(candidate)}, "
                    f"missing={missing}, extra={extra}"
                )
            question_ids = sorted(baseline)
            baseline_array = np.asarray([baseline[qid] for qid in question_ids], dtype=float)
            variant_array = np.asarray([candidate[qid] for qid in question_ids], dtype=float)
            differences = variant_array - baseline_array
            wins = int(np.count_nonzero(differences > 1e-12))
            losses = int(np.count_nonzero(differences < -1e-12))
            ties = len(differences) - wins - losses

            baseline_means[source.key] = float(baseline_array.mean())
            variant_means[source.key] = float(variant_array.mean())
            boot_differences[source.key] = bootstrap_source_mean_differences(
                differences, samples, rng
            )
            pair_counts[source.key] = len(differences)
            source_outcomes[source.key] = (wins, ties, losses)
            inherited[source.key] = is_inherited
            condition_manifest[source.key] = {
                "baseline_run": str(baseline_paths[source.key]),
                "variant_run": str(variant_path),
                "paired_questions": len(differences),
                "inherited_baseline": is_inherited,
            }
            source_rows.append(
                {
                    "condition": condition,
                    "split": source.split,
                    "source": source.source,
                    "paired_questions": len(differences),
                    "baseline_score": baseline_means[source.key],
                    "variant_score": variant_means[source.key],
                    "delta": float(differences.mean()),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "inherited_baseline": is_inherited,
                    "baseline_run": str(baseline_paths[source.key]),
                    "variant_run": str(variant_path),
                }
            )

        manifest["conditions"][condition] = condition_manifest
        for scope in SCOPE_ORDER:
            weights = SCOPE_WEIGHTS[scope]
            distribution = weighted_distribution(boot_differences, weights)
            baseline_score = weighted_score(baseline_means, weights)
            variant_score = weighted_score(variant_means, weights)
            wins = sum(source_outcomes[key][0] for key in weights)
            ties = sum(source_outcomes[key][1] for key in weights)
            losses = sum(source_outcomes[key][2] for key in weights)
            ci_low, ci_high = np.quantile(distribution, [0.025, 0.975])
            summary_rows.append(
                {
                    "condition": condition,
                    "scope": scope,
                    "source_count": len(weights),
                    "paired_questions": sum(pair_counts[key] for key in weights),
                    "inherited_source_count": sum(bool(inherited[key]) for key in weights),
                    "baseline_score": baseline_score,
                    "variant_score": variant_score,
                    "delta": variant_score - baseline_score,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "p_bootstrap_two_sided": bootstrap_sign_pvalue(distribution),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                }
            )

    holm_adjust(summary_rows)
    for row in summary_rows:
        row["ci_excludes_zero"] = row["ci_low"] > 0.0 or row["ci_high"] < 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paired_bootstrap_summary.csv", summary_rows)
    write_csv(output_dir / "paired_source_details.csv", source_rows)
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        render_report(summary_rows, samples, seed), encoding="utf-8"
    )
    (output_dir / "combined_ablation_table.md").write_text(
        render_combined_table(summary_rows, samples), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1_000:
        raise ValueError("--bootstrap-samples must be at least 1000")
    analyze(args.results_dir.resolve(), args.output_dir.resolve(), args.bootstrap_samples, args.seed)
    print(f"Wrote paired bootstrap analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
