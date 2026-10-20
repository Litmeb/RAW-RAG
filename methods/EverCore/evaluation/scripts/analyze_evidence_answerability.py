"""Analyze evidence coverage and answerability proxies for existing runs.

This script is intentionally offline-first: it uses already generated
answer_results.json and eval_results.json files and does not rerun any
benchmark stages. The metrics are high-precision proxies:

- exact_value_hit / value_coverage: whether normalized gold answer values
  appear in the retrieved formatted_context.
- gold_token_recall: softer content-word overlap between the gold answer and
  formatted_context.
- error bucket: cross product of correctness and high-evidence proxy.

The goal is to separate likely retrieval/representation failures from likely
answer-stage or conflict-resolution failures.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = EVALUATION_DIR / "results"
DEFAULT_OUTPUT_DIR = EVALUATION_DIR / "analysis" / "evidence_answerability"


MAB_SOURCES: dict[str, list[str]] = {
    "Accurate_Retrieval": [
        "eventqa_full",
        "longmemeval_s_",
        "ruler_qa1_197K",
        "ruler_qa2_421K",
    ],
    "Test_Time_Learning": [
        "recsys_redial_full",
        "icl_banking77_5900shot_balance",
        "icl_clinic150_7050shot_balance",
        "icl_nlu_8296shot_balance",
        "icl_trec_coarse_6600shot_balance",
        "icl_trec_fine_6400shot_balance",
    ],
    "Long_Range_Understanding": [
        "detective_qa",
        "infbench_sum_eng_shots2",
    ],
    "Conflict_Resolution": [
        "factconsolidation_mh_262k",
        "factconsolidation_sh_262k",
    ],
}


MAB_RUNS: dict[str, str] = {
    "rag": "normal-recover-rerank",
    "emem": "normal",
    "mem0": "normal-simplified-prompt",
    "evermemos": "normal-new-converter-label-atomic-context",
    "memmachine": "normal",
    "memos": "normal-simplified-prompt",
}


MEME_RUNS: dict[str, list[str]] = {
    "rag": ["meme-rag-normal"],
    "emem": ["meme-emem-normal"],
    "mem0": [
        "meme-mem0-normal-0-50",
        "meme-mem0-normal-32-49",
        "meme-mem0-normal-50-100",
    ],
    "evermemos": ["meme-evermemos-normal-3"],
    "memmachine": ["meme-memmachine-normal"],
    "memos": ["meme-memos-time-filter"],
}


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "am",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    system: str
    split: str
    source: str
    run_dirs: tuple[Path, ...]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_detailed_results(detailed: Any) -> list[dict[str, Any]]:
    if isinstance(detailed, list):
        return [item for item in detailed if isinstance(item, dict)]
    if isinstance(detailed, dict):
        rows: list[dict[str, Any]] = []
        for value in detailed.values():
            if isinstance(value, list):
                rows.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                rows.append(value)
        return rows
    return []


def normalize_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: Any) -> list[str]:
    normalized = normalize_text(text)
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
    ]


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    word_count = len(re.findall(r"\S+", text))
    char_estimate = math.ceil(len(text) / 4)
    return max(word_count, char_estimate)


def scalar_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def score_from_eval_item(item: dict[str, Any]) -> float:
    phase = str(item.get("phase") or "")
    if phase == "after" and "filtered_u_pass" in item:
        return scalar_score(item.get("filtered_u_pass"))
    for key in ("metric_value", "u_score", "score", "u_pass", "filtered_u_pass"):
        if key in item:
            return scalar_score(item.get(key))
    if item.get("is_correct"):
        return 1.0
    return 0.0


def recursive_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        rows: list[str] = []
        for child in value.values():
            rows.extend(recursive_strings(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(recursive_strings(child))
        return rows
    return [str(value)]


def try_literal_list(text: str) -> list[str]:
    text = text.strip()
    if not text or text[0] not in "[(":
        return []
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def split_answer_values(answer: Any) -> list[str]:
    text = "" if answer is None else str(answer).strip()
    if not text:
        return []

    literal_values = try_literal_list(text)
    if literal_values:
        return literal_values

    cleaned = re.sub(r"^\s*(answer|final answer)\s*:\s*", "", text, flags=re.I)
    cleaned = cleaned.strip(" \t\r\n.")

    # Keep long prose answers whole; token recall handles softer evidence.
    if len(cleaned) > 240:
        return [cleaned]

    parts = re.split(r"\s*(?:\n|;|\||,\s+and\s+|,\s*|\s+/\s+)\s*", cleaned)
    parts = [part.strip(" \t\r\n.;:-") for part in parts if part.strip(" \t\r\n.;:-")]
    if len(parts) > 1:
        return parts
    return [cleaned]


def dedupe_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value).strip()
        norm = normalize_text(value)
        if not norm or norm in seen:
            continue
        if norm in {"unknown", "none", "null", "n a", "not known"}:
            continue
        seen.add(norm)
        result.append(value)
    return result


def gold_values_for_item(dataset: str, answer_item: dict[str, Any]) -> list[str]:
    metadata = answer_item.get("metadata") or {}
    values: list[str] = []
    values.extend(split_answer_values(answer_item.get("golden_answer")))

    if dataset == "memoryagentbench":
        values.extend(recursive_strings(metadata.get("gold_answers")))

    # MEME gold answers are more reliable than raw entity_values for deletion
    # and absence tasks, where old entity values are present but should not be
    # returned. Fall back to entity_values only if the answer yielded nothing.
    if dataset == "meme" and not values:
        values.extend(recursive_strings(metadata.get("entity_values")))

    return dedupe_values(values)


def value_coverage(values: list[str], context_norm: str) -> tuple[int, int, float, bool]:
    if not values:
        return 0, 0, 0.0, False

    hit = 0
    exact_full_hit = False
    for value in values:
        value_norm = normalize_text(value)
        if not value_norm:
            continue
        if value_norm in context_norm:
            hit += 1
            exact_full_hit = True
            continue

        # For longer values, allow all content words to appear in the context.
        tokens = content_tokens(value_norm)
        if len(tokens) >= 4 and all(token in context_norm for token in tokens):
            hit += 1

    total = len(values)
    return hit, total, hit / total if total else 0.0, exact_full_hit


def token_recall(answer: Any, context_norm: str) -> float:
    tokens = content_tokens(answer)
    if not tokens:
        return 0.0
    unique = sorted(set(tokens))
    hits = sum(1 for token in unique if token in context_norm)
    return hits / len(unique)


def bucket_name(score: float, evidence_score: float, threshold: float) -> str:
    correctish = score >= 0.5
    high_evidence = evidence_score >= threshold
    if correctish and high_evidence:
        return "correct_high_evidence"
    if correctish and not high_evidence:
        return "correct_low_evidence"
    if not correctish and high_evidence:
        return "wrong_high_evidence"
    return "wrong_low_evidence"


def iter_run_specs(
    results_root: Path,
    mab_runs: dict[str, str] | None = None,
    meme_runs: dict[str, str] | None = None,
    include_meme: bool = True,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for system, run_name in (mab_runs or MAB_RUNS).items():
        for split, sources in MAB_SOURCES.items():
            for source in sources:
                run_dir = results_root / f"memoryagentbench-{system}-{split}-{source}-{run_name}"
                specs.append(
                    RunSpec(
                        dataset="memoryagentbench",
                        system=system,
                        split=split,
                        source=source,
                        run_dirs=(run_dir,),
                    )
                )

    if include_meme:
        selected_meme_runs = (
            {system: [run_name] for system, run_name in meme_runs.items()}
            if meme_runs is not None
            else MEME_RUNS
        )
        for system, run_names in selected_meme_runs.items():
            run_dirs = tuple(results_root / run_name for run_name in run_names)
            specs.append(
                RunSpec(
                    dataset="meme",
                    system=system,
                    split="meme",
                    source="meme",
                    run_dirs=run_dirs,
                )
            )
    return specs


def parse_mab_run_specs(values: list[str]) -> dict[str, str]:
    """Parse repeatable ``system=run-name`` CLI values."""
    parsed: dict[str, str] = {}
    for value in values:
        system, separator, run_name = value.partition("=")
        if not separator or not system.strip() or not run_name.strip():
            raise ValueError(f"Invalid --mab-run {value!r}; expected SYSTEM=RUN_NAME")
        parsed[system.strip()] = run_name.strip()
    return parsed


def load_answers_and_scores(spec: RunSpec) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, Any]]:
    answers_by_qid: dict[str, dict[str, Any]] = {}
    scores_by_qid: dict[str, float] = {}
    manifest: dict[str, Any] = {
        "dataset": spec.dataset,
        "system": spec.system,
        "split": spec.split,
        "source": spec.source,
        "run_dirs": [str(path) for path in spec.run_dirs],
        "missing": [],
        "answer_items": 0,
        "eval_items": 0,
    }

    for run_dir in spec.run_dirs:
        answer_path = run_dir / "answer_results.json"
        eval_path = run_dir / "eval_results.json"
        if not answer_path.exists() or not eval_path.exists():
            manifest["missing"].append(str(run_dir))
            continue

        answers = load_json(answer_path)
        if isinstance(answers, list):
            for item in answers:
                if isinstance(item, dict) and item.get("question_id"):
                    answers_by_qid[str(item["question_id"])] = item

        eval_data = load_json(eval_path)
        if isinstance(eval_data, dict):
            for item in flatten_detailed_results(eval_data.get("detailed_results")):
                qid = item.get("question_id")
                if qid:
                    scores_by_qid[str(qid)] = score_from_eval_item(item)

    manifest["answer_items"] = len(answers_by_qid)
    manifest["eval_items"] = len(scores_by_qid)

    answers_sorted = sorted(answers_by_qid.values(), key=lambda row: str(row.get("question_id")))
    return answers_sorted, scores_by_qid, manifest


def record_from_item(
    spec: RunSpec,
    item: dict[str, Any],
    score: float,
    threshold: float,
) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    context = str(item.get("formatted_context") or "")
    context_norm = normalize_text(context)
    values = gold_values_for_item(spec.dataset, item)
    value_hits, value_total, coverage, exact_hit = value_coverage(values, context_norm)
    gold_recall = token_recall(item.get("golden_answer"), context_norm)
    evidence_score = max(coverage, gold_recall if value_total == 0 else 0.0)
    if value_total > 0:
        evidence_score = coverage

    phase = metadata.get("phase") or ""
    task_type = metadata.get("task_type") or item.get("category") or spec.source
    group = spec.source if spec.dataset == "memoryagentbench" else f"{phase}:{task_type}"

    return {
        "dataset": spec.dataset,
        "system": spec.system,
        "split": spec.split,
        "source": spec.source,
        "group": group,
        "question_id": item.get("question_id"),
        "conversation_id": item.get("conversation_id"),
        "category": item.get("category"),
        "phase": phase,
        "task_type": task_type,
        "score": score,
        "correctish": int(score >= 0.5),
        "context_chars": len(context),
        "context_tokens_est": approx_tokens(context),
        "gold_value_count": value_total,
        "gold_value_hits": value_hits,
        "value_coverage": coverage,
        "exact_value_hit": int(exact_hit),
        "gold_token_recall": gold_recall,
        "evidence_score": evidence_score,
        "high_evidence": int(evidence_score >= threshold),
        "bucket": bucket_name(score, evidence_score, threshold),
        "gold_values_preview": " | ".join(values[:5]),
        "golden_answer": item.get("golden_answer"),
        "generated_answer": item.get("answer") or item.get("generated_answer"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: list[dict[str, Any]], preferred: list[str] | None = None) -> list[str]:
    preferred = preferred or []
    fields = set(preferred)
    for row in rows:
        fields.update(row.keys())
    return preferred + sorted(field for field in fields if field not in preferred)


def aggregate(rows: list[dict[str, Any]], group_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)

    result: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        n = len(group_rows)
        bucket_counts = defaultdict(int)
        for row in group_rows:
            bucket_counts[row["bucket"]] += 1

        out = {field: value for field, value in zip(group_fields, key)}
        out.update(
            {
                "n": n,
                "accuracy": mean(row["score"] for row in group_rows),
                "high_evidence_rate": mean(row["high_evidence"] for row in group_rows),
                "exact_value_hit_rate": mean(row["exact_value_hit"] for row in group_rows),
                "mean_value_coverage": mean(row["value_coverage"] for row in group_rows),
                "mean_gold_token_recall": mean(row["gold_token_recall"] for row in group_rows),
                "mean_context_tokens_est": mean(row["context_tokens_est"] for row in group_rows),
                "correct_high_evidence": bucket_counts["correct_high_evidence"],
                "correct_low_evidence": bucket_counts["correct_low_evidence"],
                "wrong_high_evidence": bucket_counts["wrong_high_evidence"],
                "wrong_low_evidence": bucket_counts["wrong_low_evidence"],
                "wrong_high_evidence_rate": bucket_counts["wrong_high_evidence"] / n if n else 0,
                "wrong_low_evidence_rate": bucket_counts["wrong_low_evidence"] / n if n else 0,
            }
        )
        result.append(out)
    return result


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def num(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    lines = []
    headers = [label for _, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if key.endswith("rate") or key in {
                "accuracy",
                "high_evidence_rate",
                "exact_value_hit_rate",
                "mean_value_coverage",
                "mean_gold_token_recall",
            }:
                values.append(pct(value))
            elif key.endswith("tokens_est") or key == "n":
                values.append(num(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def select_rows(rows: list[dict[str, Any]], **filters: str) -> list[dict[str, Any]]:
    selected = rows
    for key, value in filters.items():
        selected = [row for row in selected if row.get(key) == value]
    return selected


def write_report(
    output_dir: Path,
    manifest_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    threshold: float,
) -> None:
    overall = aggregate(
        [
            {
                **row,
                "dataset_system": f"{row['dataset']}:{row['system']}",
            }
            for row in []
        ],
        ["dataset_system"],
    )
    del overall

    report_lines = [
        "# Evidence Coverage / Answerability Proxy Analysis",
        "",
        f"High-evidence threshold: value coverage >= {threshold:.2f}.",
        "",
        "This analysis uses existing `answer_results.json` and `eval_results.json` files only. "
        "It does not rerun benchmark stages. Evidence metrics are proxies based on whether "
        "normalized gold answer values appear in the retrieved `formatted_context`; they are "
        "most reliable for exact recall, classification labels, entity values, and structured "
        "MEME tasks, and less reliable for abstractive summaries.",
        "",
        "## Run Manifest",
        "",
        markdown_table(
            manifest_rows,
            [
                ("dataset", "Dataset"),
                ("system", "System"),
                ("split", "Split"),
                ("source", "Source"),
                ("answer_items", "Answers"),
                ("eval_items", "Eval Items"),
                ("missing_count", "Missing"),
            ],
            limit=120,
        ),
        "",
        "## Overall",
        "",
        markdown_table(
            summary_rows,
            [
                ("dataset", "Dataset"),
                ("system", "System"),
                ("n", "N"),
                ("accuracy", "Accuracy"),
                ("high_evidence_rate", "High Evidence"),
                ("exact_value_hit_rate", "Exact Hit"),
                ("mean_value_coverage", "Value Coverage"),
                ("mean_gold_token_recall", "Token Recall"),
                ("mean_context_tokens_est", "Ctx Tok"),
                ("wrong_high_evidence_rate", "Wrong+Evidence"),
                ("wrong_low_evidence_rate", "Wrong+No Evidence"),
            ],
        ),
        "",
        "## MemoryAgentBench By Split",
        "",
        markdown_table(
            [row for row in group_rows if row["dataset"] == "memoryagentbench" and row["group_level"] == "split"],
            [
                ("system", "System"),
                ("split", "Split"),
                ("n", "N"),
                ("accuracy", "Accuracy"),
                ("high_evidence_rate", "High Evidence"),
                ("mean_value_coverage", "Value Coverage"),
                ("mean_context_tokens_est", "Ctx Tok"),
                ("wrong_high_evidence_rate", "Wrong+Evidence"),
                ("wrong_low_evidence_rate", "Wrong+No Evidence"),
            ],
        ),
        "",
        "## MEME By Task",
        "",
        markdown_table(
            [row for row in group_rows if row["dataset"] == "meme" and row["group_level"] == "group"],
            [
                ("system", "System"),
                ("group", "Task"),
                ("n", "N"),
                ("accuracy", "Accuracy"),
                ("high_evidence_rate", "High Evidence"),
                ("mean_value_coverage", "Value Coverage"),
                ("mean_context_tokens_est", "Ctx Tok"),
                ("wrong_high_evidence_rate", "Wrong+Evidence"),
                ("wrong_low_evidence_rate", "Wrong+No Evidence"),
            ],
        ),
        "",
        "## Diagnostic Reading",
        "",
        "- `wrong_low_evidence` is the strongest proxy for retrieval or memory-representation failure: the answer value is not visible in the returned context.",
        "- `wrong_high_evidence` indicates the system likely retrieved/stored the answer value but still failed, often due to conflict resolution, ordering, aggregation, noisy context, or answer-stage selection.",
        "- Large context token counts with low value coverage indicate inefficient memory formatting: the system sends many tokens without surfacing the target evidence.",
        "- `correct_low_evidence` should be treated cautiously: it may be a proxy false negative, answer leakage through the question/options, or use of world/model priors.",
        "",
        "Generated files:",
        "- `sample_metrics.csv`: per-question evidence proxy rows.",
        "- `summary_by_system.csv`: dataset/system summary.",
        "- `summary_by_group.csv`: split/source/task-level summary.",
        "- `run_manifest.json`: exact result directories used.",
    ]

    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-evidence-threshold", type=float, default=0.8)
    parser.add_argument(
        "--mab-run",
        action="append",
        default=[],
        metavar="SYSTEM=RUN_NAME",
        help="Override the default MemoryAgentBench runs; repeat per system.",
    )
    parser.add_argument(
        "--meme-run",
        action="append",
        default=[],
        metavar="SYSTEM=RUN_NAME",
        help="Override the default MEME runs; repeat per system.",
    )
    parser.add_argument(
        "--memoryagentbench-only",
        action="store_true",
        help="Skip the fixed MEME run set.",
    )
    parser.add_argument(
        "--meme-only",
        action="store_true",
        help="Skip MemoryAgentBench runs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    try:
        mab_runs = parse_mab_run_specs(args.mab_run)
        meme_runs = parse_mab_run_specs(args.meme_run)
    except ValueError as exc:
        parser.error(str(exc))
    for spec in iter_run_specs(
        args.results_root,
        mab_runs=mab_runs or None,
        meme_runs=meme_runs or None,
        include_meme=not args.memoryagentbench_only,
    ):
        if args.meme_only and spec.dataset == "memoryagentbench":
            continue
        answers, scores_by_qid, manifest = load_answers_and_scores(spec)
        manifest["missing_count"] = len(manifest["missing"])
        manifest_rows.append(manifest)
        for item in answers:
            qid = str(item.get("question_id"))
            if qid not in scores_by_qid:
                continue
            records.append(record_from_item(spec, item, scores_by_qid[qid], args.high_evidence_threshold))

    sample_fields = [
        "dataset",
        "system",
        "split",
        "source",
        "group",
        "question_id",
        "conversation_id",
        "category",
        "phase",
        "task_type",
        "score",
        "correctish",
        "context_chars",
        "context_tokens_est",
        "gold_value_count",
        "gold_value_hits",
        "value_coverage",
        "exact_value_hit",
        "gold_token_recall",
        "evidence_score",
        "high_evidence",
        "bucket",
        "gold_values_preview",
        "golden_answer",
        "generated_answer",
    ]
    write_csv(output_dir / "sample_metrics.csv", records, sample_fields)

    summary_rows = aggregate(records, ["dataset", "system"])
    write_csv(
        output_dir / "summary_by_system.csv",
        summary_rows,
        union_fieldnames(summary_rows, ["dataset", "system"]) if summary_rows else [],
    )

    group_rows: list[dict[str, Any]] = []
    for group_fields, level in [
        (["dataset", "system", "split"], "split"),
        (["dataset", "system", "source"], "source"),
        (["dataset", "system", "group"], "group"),
    ]:
        for row in aggregate(records, group_fields):
            row["group_level"] = level
            group_rows.append(row)
    write_csv(
        output_dir / "summary_by_group.csv",
        group_rows,
        union_fieldnames(
            group_rows,
            ["dataset", "system", "group_level", "split", "source", "group"],
        )
        if group_rows
        else [],
    )

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(output_dir, manifest_rows, summary_rows, group_rows, args.high_evidence_threshold)

    print(f"Wrote {len(records)} sample rows to {output_dir}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
