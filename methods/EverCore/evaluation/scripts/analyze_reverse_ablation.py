"""Paired diagnostics for MemoryAgentBench reverse-ablation RAG variants.

The script is offline-first. It reads existing result directories and compares
each variant against a raw RAG baseline on matched question_id values.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
EVIDENCE_SCRIPT = SCRIPT_DIR / "analyze_evidence_answerability.py"


def load_evidence_module():
    spec = importlib.util.spec_from_file_location("evidence_answerability", EVIDENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {EVIDENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evidence_answerability"] = module
    spec.loader.exec_module(module)
    return module


EV = load_evidence_module()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def infer_split_source(run_dir: Path, answers: list[dict[str, Any]]) -> tuple[str, str]:
    for item in answers:
        metadata = item.get("metadata") or {}
        split = metadata.get("mab_split")
        source = metadata.get("source")
        if split and source:
            return str(split), str(source)

    match = re.search(
        r"memoryagentbench-[^-]+-(?P<split>[^-]+)-(?P<source>[^-]+)-",
        run_dir.name,
    )
    if match:
        return match.group("split"), match.group("source")
    return "unknown", "unknown"


def score_map(eval_data: dict[str, Any]) -> dict[str, float]:
    scores = {}
    for item in EV.flatten_detailed_results(eval_data.get("detailed_results")):
        qid = item.get("question_id")
        if qid:
            scores[str(qid)] = EV.score_from_eval_item(item)
    return scores


def load_run(label: str, run_dir: Path, threshold: float) -> dict[str, Any]:
    answer_path = run_dir / "answer_results.json"
    eval_path = run_dir / "eval_results.json"
    if not answer_path.exists():
        raise FileNotFoundError(f"Missing answer_results.json: {answer_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval_results.json: {eval_path}")

    answers = [item for item in load_json(answer_path) if isinstance(item, dict)]
    eval_data = load_json(eval_path)
    split, source = infer_split_source(run_dir, answers)
    spec = EV.RunSpec(
        dataset="memoryagentbench",
        system=label,
        split=split,
        source=source,
        run_dirs=(run_dir,),
    )
    scores = score_map(eval_data)
    records = []
    for item in answers:
        qid = str(item.get("question_id") or "")
        if not qid or qid not in scores:
            continue
        records.append(EV.record_from_item(spec, item, scores[qid], threshold))

    return {
        "label": label,
        "run_dir": run_dir,
        "split": split,
        "source": source,
        "answers": answers,
        "scores": scores,
        "records": records,
        "records_by_qid": {str(row["question_id"]): row for row in records},
        "eval": eval_data,
        "request_stats": request_log_stats(run_dir),
        "request_log_file_count": sum(1 for _ in (run_dir / "logs").glob("*.jsonl")),
        "report_elapsed_seconds": read_report_elapsed(run_dir),
    }


def read_report_elapsed(run_dir: Path) -> float | None:
    report_path = run_dir / "report.txt"
    if not report_path.exists():
        return None
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Time Elapsed:"):
            raw = line.split(":", 1)[1].strip().rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def usage_from_response(response: Any) -> dict[str, int]:
    if not isinstance(response, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def request_log_stats(run_dir: Path) -> dict[str, Any]:
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "latency_ms_sum": 0.0,
            "latency_ms_max": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_request_tokens": 0,
        }
    )
    log_files = sorted((run_dir / "logs").glob("*.jsonl"))
    for path in log_files:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = str(event.get("stage") or "unknown")
                call_type = str(event.get("call_type") or "unknown")
                row = stats[(stage, call_type)]
                row["calls"] += 1
                if event.get("error"):
                    row["errors"] += 1
                latency = float(event.get("latency_ms") or 0.0)
                row["latency_ms_sum"] += latency
                row["latency_ms_max"] = max(row["latency_ms_max"], latency)
                usage = usage_from_response(event.get("response"))
                for key, value in usage.items():
                    row[key] += value
                request_text = json.dumps(event.get("request") or {}, ensure_ascii=False)
                row["estimated_request_tokens"] += EV.approx_tokens(request_text)

    by_key = {}
    for (stage, call_type), row in stats.items():
        row = dict(row)
        row["latency_ms_avg"] = row["latency_ms_sum"] / row["calls"] if row["calls"] else 0.0
        by_key[f"{stage}:{call_type}"] = row
    return by_key


def resource_kind(call_type: str) -> str:
    """Map a logged request type to the resource whose usage we report.

    Embedding APIs expose input-token usage, rerankers are counted by request,
    and the remaining calls in these RAG runs are generative LLM calls.
    """
    normalized = call_type.strip().lower()
    if "embedding" in normalized:
        return "embedding"
    if "rerank" in normalized:
        return "reranker"
    return "llm"


def summarize_resource_usage(run: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": run["label"],
        "split": run["split"],
        "source": run["source"],
        "run_dir": str(run["run_dir"]),
        "report_elapsed_seconds": run["report_elapsed_seconds"],
        "request_logs_available": bool(run.get("request_log_file_count", 0)),
        "embedding_calls": 0,
        "embedding_errors": 0,
        "embedding_tokens": 0,
        "embedding_latency_seconds": 0.0,
        "reranker_calls": 0,
        "reranker_errors": 0,
        "reranker_latency_seconds": 0.0,
        "llm_calls": 0,
        "llm_errors": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_latency_seconds": 0.0,
    }
    for key, stats in run["request_stats"].items():
        _, _, call_type = key.partition(":")
        kind = resource_kind(call_type)
        summary[f"{kind}_calls"] += int(stats["calls"])
        summary[f"{kind}_errors"] += int(stats["errors"])
        summary[f"{kind}_latency_seconds"] += float(stats["latency_ms_sum"]) / 1000.0
        if kind == "embedding":
            summary["embedding_tokens"] += int(stats["total_tokens"])
        elif kind == "llm":
            summary["llm_prompt_tokens"] += int(stats["prompt_tokens"])
            summary["llm_completion_tokens"] += int(stats["completion_tokens"])
            summary["llm_total_tokens"] += int(stats["total_tokens"])
    return summary


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    records = run["records"]
    buckets = Counter(row["bucket"] for row in records)
    total = len(records)
    score_values = [float(row["score"]) for row in records]
    context_tokens = [float(row["context_tokens_est"]) for row in records]
    total_calls = sum(row["calls"] for row in run["request_stats"].values())
    total_tokens = sum(row["total_tokens"] for row in run["request_stats"].values())
    request_latency = sum(row["latency_ms_sum"] for row in run["request_stats"].values())
    return {
        "label": run["label"],
        "split": run["split"],
        "source": run["source"],
        "run_dir": str(run["run_dir"]),
        "n": total,
        "correct": int(round(sum(score_values))),
        "accuracy": safe_mean(score_values),
        "exact_value_hit_rate": safe_mean([float(row["exact_value_hit"]) for row in records]),
        "mean_value_coverage": safe_mean([float(row["value_coverage"]) for row in records]),
        "mean_gold_token_recall": safe_mean([float(row["gold_token_recall"]) for row in records]),
        "high_evidence_rate": safe_mean([float(row["high_evidence"]) for row in records]),
        "wrong_high_evidence": buckets["wrong_high_evidence"],
        "wrong_low_evidence": buckets["wrong_low_evidence"],
        "wrong_high_evidence_rate": buckets["wrong_high_evidence"] / total if total else 0.0,
        "wrong_low_evidence_rate": buckets["wrong_low_evidence"] / total if total else 0.0,
        "mean_context_tokens_est": safe_mean(context_tokens),
        "report_elapsed_seconds": run["report_elapsed_seconds"],
        "api_logged_calls": total_calls,
        "api_logged_total_tokens": total_tokens,
        "api_logged_latency_seconds": request_latency / 1000.0,
    }


def paired_diff(baseline: dict[str, Any], variant: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_records = baseline["records_by_qid"]
    variant_records = variant["records_by_qid"]
    common_qids = sorted(set(base_records) & set(variant_records))

    rows = []
    counts = Counter()
    for qid in common_qids:
        b = base_records[qid]
        v = variant_records[qid]
        b_correct = float(b["score"]) >= 0.5
        v_correct = float(v["score"]) >= 0.5
        if b_correct and v_correct:
            counts["both_correct"] += 1
            transition = "both_correct"
        elif b_correct and not v_correct:
            counts["rag_correct_variant_wrong"] += 1
            transition = "rag_correct_variant_wrong"
        elif not b_correct and v_correct:
            counts["rag_wrong_variant_correct"] += 1
            transition = "rag_wrong_variant_correct"
        else:
            counts["both_wrong"] += 1
            transition = "both_wrong"

        rows.append(
            {
                "variant": variant["label"],
                "question_id": qid,
                "conversation_id": v.get("conversation_id"),
                "category": v.get("category"),
                "transition": transition,
                "baseline_score": b["score"],
                "variant_score": v["score"],
                "baseline_bucket": b["bucket"],
                "variant_bucket": v["bucket"],
                "baseline_value_coverage": b["value_coverage"],
                "variant_value_coverage": v["value_coverage"],
                "baseline_exact_value_hit": b["exact_value_hit"],
                "variant_exact_value_hit": v["exact_value_hit"],
                "baseline_context_tokens_est": b["context_tokens_est"],
                "variant_context_tokens_est": v["context_tokens_est"],
                "context_token_delta": int(v["context_tokens_est"]) - int(b["context_tokens_est"]),
                "gold_values_preview": v.get("gold_values_preview"),
                "golden_answer": v.get("golden_answer"),
                "baseline_answer": b.get("generated_answer"),
                "variant_answer": v.get("generated_answer"),
            }
        )

    n = len(common_qids)
    base_acc = safe_mean([float(base_records[qid]["score"]) for qid in common_qids])
    var_acc = safe_mean([float(variant_records[qid]["score"]) for qid in common_qids])
    summary = {
        "variant": variant["label"],
        "baseline": baseline["label"],
        "split": variant["split"],
        "source": variant["source"],
        "paired_n": n,
        "baseline_accuracy_on_common": base_acc,
        "variant_accuracy_on_common": var_acc,
        "accuracy_delta_pp": (var_acc - base_acc) * 100,
        "rag_correct_variant_wrong": counts["rag_correct_variant_wrong"],
        "rag_wrong_variant_correct": counts["rag_wrong_variant_correct"],
        "both_correct": counts["both_correct"],
        "both_wrong": counts["both_wrong"],
        "net_correct_delta": counts["rag_wrong_variant_correct"] - counts["rag_correct_variant_wrong"],
        "mean_value_coverage_delta": safe_mean(
            [
                float(variant_records[qid]["value_coverage"]) - float(base_records[qid]["value_coverage"])
                for qid in common_qids
            ]
        ),
        "exact_value_hit_delta": safe_mean(
            [
                float(variant_records[qid]["exact_value_hit"]) - float(base_records[qid]["exact_value_hit"])
                for qid in common_qids
            ]
        ),
        "high_evidence_delta": safe_mean(
            [
                float(variant_records[qid]["high_evidence"]) - float(base_records[qid]["high_evidence"])
                for qid in common_qids
            ]
        ),
        "mean_context_token_delta": safe_mean(
            [
                float(variant_records[qid]["context_tokens_est"]) - float(base_records[qid]["context_tokens_est"])
                for qid in common_qids
            ]
        ),
    }
    return summary, rows


def request_stats_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for key, stats in sorted(run["request_stats"].items()):
            stage, _, call_type = key.partition(":")
            rows.append(
                {
                    "label": run["label"],
                    "stage": stage,
                    "call_type": call_type,
                    **stats,
                }
            )
    return rows


def write_markdown(
    output_dir: Path,
    baseline_summary: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    resource_summaries: list[dict[str, Any]],
    paired_summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# MemoryAgentBench Reverse Ablation Diagnostics",
        "",
        f"Baseline: `{baseline_summary['label']}`",
        f"Baseline run: `{baseline_summary['run_dir']}`",
        "",
        "## Run Summary",
        "",
        "| run | n | accuracy | correct | value coverage | exact hit | high evidence | wrong+evidence | wrong+low evidence | ctx tokens | calls | logged tokens | elapsed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summaries:
        lines.append(
            f"| {row['label']} | {row['n']} | {pct(row['accuracy'])} | {row['correct']} | "
            f"{pct(row['mean_value_coverage'])} | {pct(row['exact_value_hit_rate'])} | "
            f"{pct(row['high_evidence_rate'])} | {row['wrong_high_evidence']} | "
            f"{row['wrong_low_evidence']} | {row['mean_context_tokens_est']:.0f} | "
            f"{row['api_logged_calls']} | {row['api_logged_total_tokens']} | "
            f"{row['report_elapsed_seconds'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "## Resource Usage",
            "",
            "Embedding and LLM tokens use provider-reported response usage. "
            "Reranker usage is request count only.",
            "",
            "| run | request logs | embedding calls | embedding errors | embedding tokens | reranker calls | reranker errors | LLM calls | LLM errors | LLM prompt tokens | LLM completion tokens | LLM total tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in resource_summaries:
        lines.append(
            f"| {row['label']} | {'yes' if row['request_logs_available'] else 'no'} | "
            f"{row['embedding_calls']} | {row['embedding_errors']} | "
            f"{row['embedding_tokens']} | {row['reranker_calls']} | "
            f"{row['reranker_errors']} | {row['llm_calls']} | {row['llm_errors']} | "
            f"{row['llm_prompt_tokens']} | {row['llm_completion_tokens']} | "
            f"{row['llm_total_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Paired Diff vs Baseline",
            "",
            "| variant | paired n | baseline acc | variant acc | delta pp | RAG correct -> variant wrong | RAG wrong -> variant correct | net | value coverage delta | exact hit delta | high evidence delta | ctx token delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_summaries:
        lines.append(
            f"| {row['variant']} | {row['paired_n']} | {pct(row['baseline_accuracy_on_common'])} | "
            f"{pct(row['variant_accuracy_on_common'])} | {row['accuracy_delta_pp']:.2f} | "
            f"{row['rag_correct_variant_wrong']} | {row['rag_wrong_variant_correct']} | "
            f"{row['net_correct_delta']} | {row['mean_value_coverage_delta']:.3f} | "
            f"{row['exact_value_hit_delta']:.3f} | {row['high_evidence_delta']:.3f} | "
            f"{row['mean_context_token_delta']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Reading Guide",
            "",
            "- `RAG correct -> variant wrong` is the main damage count for reverse ablation.",
            "- `RAG wrong -> variant correct` is the offsetting gain count.",
            "- Lower value coverage or exact-hit rate suggests evidence damage or representation loss.",
            "- Higher context tokens with lower or flat coverage suggests context noise.",
            "- `wrong_high_evidence` means the answer value is visible but the model still failed, often pointing to ordering, conflict resolution, or answer selection.",
            "",
            "Generated files:",
            "- `run_summary.csv`",
            "- `paired_summary.csv`",
            "- `paired_details.csv`",
            "- `request_log_summary.csv`",
            "- `resource_usage_summary.csv`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_variant(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = Path(raw)
        return path.name, path
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="raw-rag-baseline")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant as label=DIR. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATION_DIR / "analysis" / "reverse_ablation",
    )
    parser.add_argument("--high-evidence-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_run(
        label=args.baseline_label,
        run_dir=args.baseline_dir.resolve(),
        threshold=args.high_evidence_threshold,
    )
    variants = [
        load_run(label=label, run_dir=path.resolve(), threshold=args.high_evidence_threshold)
        for label, path in (parse_variant(raw) for raw in args.variant)
    ]
    runs = [baseline, *variants]

    run_summaries = [summarize_run(run) for run in runs]
    resource_summaries = [summarize_resource_usage(run) for run in runs]
    paired_summaries = []
    paired_rows = []
    for variant in variants:
        summary, rows = paired_diff(baseline, variant)
        paired_summaries.append(summary)
        paired_rows.extend(rows)

    write_csv(output_dir / "run_summary.csv", run_summaries)
    write_csv(output_dir / "paired_summary.csv", paired_summaries)
    write_csv(output_dir / "paired_details.csv", paired_rows)
    write_csv(output_dir / "request_log_summary.csv", request_stats_rows(runs))
    write_csv(output_dir / "resource_usage_summary.csv", resource_summaries)
    write_markdown(
        output_dir,
        run_summaries[0],
        run_summaries,
        resource_summaries,
        paired_summaries,
    )

    manifest = {
        "baseline_dir": str(args.baseline_dir.resolve()),
        "variants": [
            {"label": label, "run_dir": str(path.resolve())}
            for label, path in (parse_variant(raw) for raw in args.variant)
        ],
        "high_evidence_threshold": args.high_evidence_threshold,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote reverse-ablation diagnostics to: {output_dir}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
