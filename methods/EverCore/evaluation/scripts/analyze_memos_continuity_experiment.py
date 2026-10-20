"""Analyze paired EventQA results for the Memos continuity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from scipy.stats import binomtest


EVALUATION_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_NATIVE = RESULTS_DIR / (
    "memoryagentbench-memos_local-Accurate_Retrieval-eventqa_full-fine"
)
DEFAULT_MEMORY = RESULTS_DIR / (
    "memoryagentbench-memos_local-Accurate_Retrieval-eventqa_full-memory-oracle-"
    "continuity-gold-dense-5000-20260716"
)
DEFAULT_RAW = RESULTS_DIR / (
    "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-raw-rag-native-"
    "5000-reader-control-20260716"
)
DEFAULT_OUTPUT = RESULTS_DIR / "memos_continuity_eventqa_report_20260716"
DEFAULT_DIAGNOSTIC_SUMMARY = DEFAULT_NATIVE / (
    "memos_diagnostics_20260716_094401_llm_audit/summary.json"
)
DEFAULT_ORACLE_AUDIT = DEFAULT_MEMORY / "continuity_mapping_audit.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def correctness(run_dir: Path) -> tuple[list[str], np.ndarray]:
    rows = load_json(run_dir / "eval_results.json")["detailed_results"]
    ids = [str(row["question_id"]) for row in rows]
    values = np.asarray(
        [float(row.get("metric_value") or 0) > 0 for row in rows], dtype=np.int8
    )
    return ids, values


def contexts(run_dir: Path) -> dict[str, str]:
    answer_path = run_dir / "answer_results.json"
    if answer_path.exists():
        return {
            str(row["question_id"]): str(row.get("formatted_context") or "")
            for row in load_json(answer_path)
        }
    return {
        str(row["question_id"]): str(
            (row.get("retrieval_metadata") or {}).get("formatted_context") or ""
        )
        for row in load_json(run_dir / "search_results.json")
    }


def paired_difference_ci(
    left: np.ndarray, right: np.ndarray, *, samples: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(left)
    differences = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        differences[start : start + count] = (
            left[indices].mean(axis=1) - right[indices].mean(axis=1)
        )
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(left.mean() - right.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def paired_test(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    discordant = left_only + right_only
    pvalue = (
        float(binomtest(left_only, discordant, 0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant": discordant,
        "mcnemar_exact_pvalue": pvalue,
    }


def token_summary(run_dir: Path, ids: list[str]) -> dict[str, float]:
    encoding = tiktoken.get_encoding("o200k_base")
    by_id = contexts(run_dir)
    lengths = [len(encoding.encode(by_id[question_id])) for question_id in ids]
    return {
        "min": min(lengths),
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
    }


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--memory-oracle", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--diagnostic-summary", type=Path, default=DEFAULT_DIAGNOSTIC_SUMMARY
    )
    parser.add_argument("--oracle-audit", type=Path, default=DEFAULT_ORACLE_AUDIT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    native_ids, native = correctness(args.native)
    memory_ids, memory = correctness(args.memory_oracle)
    raw_ids, raw = correctness(args.raw)
    if native_ids != memory_ids or native_ids != raw_ids:
        raise ValueError("Question IDs/order do not align across the three runs")

    native_accuracy = float(native.mean())
    memory_accuracy = float(memory.mean())
    raw_accuracy = float(raw.mean())
    original_gap = raw_accuracy - native_accuracy
    repair_gain = memory_accuracy - native_accuracy
    residual_gap = raw_accuracy - memory_accuracy
    recovery_ratio = repair_gain / original_gap if original_gap else 0.0
    diagnostic_counts = load_json(args.diagnostic_summary)["deterministic"]["counts"]
    oracle_audit = load_json(args.oracle_audit)
    average_available_provenance = sum(
        row["available_provenance_memories"] for row in oracle_audit
    ) / len(oracle_audit)
    average_selected_provenance = sum(
        row["selected_provenance_memories"] for row in oracle_audit
    ) / len(oracle_audit)

    raw_correct_memory_wrong = [
        question_id
        for question_id, raw_value, memory_value in zip(
            native_ids, raw, memory, strict=True
        )
        if raw_value and not memory_value
    ]
    raw_wrong_memory_correct = [
        question_id
        for question_id, raw_value, memory_value in zip(
            native_ids, raw, memory, strict=True
        )
        if not raw_value and memory_value
    ]

    report = {
        "experiment": "Memos EventQA continuity recovery",
        "questions": len(native_ids),
        "new_reranker_calls": 0,
        "new_embedding_api_calls": 0,
        "storage_diagnostics": {
            "stored_memories": diagnostic_counts["memos_memories"],
            "source_chunks": diagnostic_counts["converted_chunks"],
            "covered_source_chunks": diagnostic_counts["covered_chunks"],
            "missing_source_chunks": diagnostic_counts["missing_chunks"],
            "source_coverage": diagnostic_counts["covered_chunks"]
            / diagnostic_counts["converted_chunks"],
            "duplicate_extra_source_hits": diagnostic_counts[
                "duplicate_extra_source_hits"
            ],
            "average_available_raw_window_memories": average_available_provenance,
            "average_selected_raw_window_memories": average_selected_provenance,
            "questions_without_raw_window_memory": sum(
                row["available_provenance_memories"] == 0 for row in oracle_audit
            ),
        },
        "conditions": {
            "native_memos": {
                "run_dir": str(args.native.resolve()),
                "correct": int(native.sum()),
                "accuracy": native_accuracy,
                "context_tokens": token_summary(args.native, native_ids),
            },
            "memory_store_oracle": {
                "run_dir": str(args.memory_oracle.resolve()),
                "correct": int(memory.sum()),
                "accuracy": memory_accuracy,
                "context_tokens": token_summary(args.memory_oracle, native_ids),
                "advantages": [
                    "gold answer included in dense query",
                    "complete 9252-record Memos export available as candidate pool",
                    "all memories from raw evidence source window prioritized",
                    "selected memories restored to source_chunk_indices order",
                    "stored native Memos embeddings reused without reranking",
                ],
                "reader_exposure": "Memos memory text only; metadata.sources raw text excluded",
            },
            "continuity_preserving_raw_rag": {
                "run_dir": str(args.raw.resolve()),
                "correct": int(raw.sum()),
                "accuracy": raw_accuracy,
                "context_tokens": token_summary(args.raw, native_ids),
                "note": (
                    "Reuses the existing Mem0 continuity raw-reader control. Its historical "
                    "search artifact contains reranker output, but this experiment made no "
                    "new reranker calls."
                ),
            },
        },
        "effects": {
            "original_raw_minus_native_gap": original_gap,
            "memory_oracle_repair_gain": repair_gain,
            "residual_raw_minus_memory_oracle_gap": residual_gap,
            "recovery_ratio": recovery_ratio,
            "paired_bootstrap": {
                "memory_oracle_minus_native": paired_difference_ci(
                    memory, native, samples=args.bootstrap_samples, seed=args.seed
                ),
                "raw_minus_memory_oracle": paired_difference_ci(
                    raw, memory, samples=args.bootstrap_samples, seed=args.seed + 1
                ),
            },
            "paired_mcnemar": {
                "memory_oracle_vs_native": paired_test(memory, native),
                "raw_vs_memory_oracle": paired_test(raw, memory),
            },
        },
        "case_sets": {
            "raw_correct_memory_oracle_wrong_count": len(raw_correct_memory_wrong),
            "raw_correct_memory_oracle_wrong": raw_correct_memory_wrong,
            "raw_wrong_memory_oracle_correct_count": len(raw_wrong_memory_correct),
            "raw_wrong_memory_oracle_correct": raw_wrong_memory_correct,
        },
        "interpretation_limit": (
            "The memory condition is a strong gold-informed dense-retrieval oracle, "
            "not a mathematical exhaustive oracle over every possible memory subset."
        ),
    }

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_prefix.with_suffix(".json"), report)

    memory_ci = report["effects"]["paired_bootstrap"]["memory_oracle_minus_native"]
    residual_ci = report["effects"]["paired_bootstrap"]["raw_minus_memory_oracle"]
    memory_test = report["effects"]["paired_mcnemar"]["memory_oracle_vs_native"]
    residual_test = report["effects"]["paired_mcnemar"]["raw_vs_memory_oracle"]
    markdown = f"""# Memos EventQA Continuity Experiment

| Condition | Correct | Accuracy | Mean context tokens |
|---|---:|---:|---:|
| Native Memos | {int(native.sum())}/500 | {percentage(native_accuracy)} | {report['conditions']['native_memos']['context_tokens']['mean']:.1f} |
| Gold-informed Memos memory-store oracle | {int(memory.sum())}/500 | {percentage(memory_accuracy)} | {report['conditions']['memory_store_oracle']['context_tokens']['mean']:.1f} |
| Reused continuity-preserving raw RAG | {int(raw.sum())}/500 | {percentage(raw_accuracy)} | {report['conditions']['continuity_preserving_raw_rag']['context_tokens']['mean']:.1f} |

- Oracle repair gain over native Memos: **{repair_gain * 100:.2f} pp** (paired bootstrap 95% CI {memory_ci['ci95_low'] * 100:.2f} to {memory_ci['ci95_high'] * 100:.2f} pp; exact McNemar p={memory_test['mcnemar_exact_pvalue']:.6g}).
- Residual raw-vs-memory-oracle gap: **{residual_gap * 100:.2f} pp** (paired bootstrap 95% CI {residual_ci['ci95_low'] * 100:.2f} to {residual_ci['ci95_high'] * 100:.2f} pp; exact McNemar p={residual_test['mcnemar_exact_pvalue']:.6g}).
- Fraction of the original raw-vs-native gap recovered: **{recovery_ratio * 100:.2f}%**.
- Raw correct / memory oracle wrong: **{len(raw_correct_memory_wrong)}** questions.
- Raw wrong / memory oracle correct: **{len(raw_wrong_memory_correct)}** questions.
- New reranker calls: **0**. New embedding API calls: **0**.
- Memos source coverage: **{diagnostic_counts['covered_chunks']}/{diagnostic_counts['converted_chunks']} ({diagnostic_counts['covered_chunks'] / diagnostic_counts['converted_chunks'] * 100:.2f}%)**; missing chunks: **{diagnostic_counts['missing_chunks']}**; duplicate extra source hits: **{diagnostic_counts['duplicate_extra_source_hits']}**.
- The raw evidence window had **{average_available_provenance:.2f}** provenance memories per question on average, and the oracle included **{average_selected_provenance:.2f}** of them; questions with no provenance memory: **{sum(row['available_provenance_memories'] == 0 for row in oracle_audit)}**.

The oracle receives the gold answer in its dense query, can search the complete 9,252-record Memos export, prioritizes every memory derived from the raw evidence window, and restores selected memories to source order. The reader sees only Memos memory text, never the raw `metadata.sources` content.

The raw control is reused unchanged from the Mem0 experiment. Its historical retrieval artifact contains reranker output; no reranker was called in this Memos experiment.

This is a strong empirical upper-bound construction, not a mathematical exhaustive oracle over all possible subsets of stored memories.
"""
    output_prefix.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
