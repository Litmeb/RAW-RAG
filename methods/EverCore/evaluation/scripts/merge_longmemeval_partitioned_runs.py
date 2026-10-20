"""Merge the two completed LongMemEval partition runs into one result view."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SOURCE_NAMES = [
    "longmemeval-rag-256-32-128-15-turn-expand-time-filter-0-49",
    "longmemeval-rag-256-32-128-15-turn-expand-time-filter-50-499",
]
OUTPUT_NAME = "longmemeval-rag-256-32-128-15-turn-expand-time-filter"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def flatten_detailed_results(payload: dict) -> list[dict]:
    detailed = payload["detailed_results"]
    if isinstance(detailed, list):
        return detailed
    return [row for rows in detailed.values() for row in rows]


def assert_unique_and_aligned(result_sets: list[dict[str, list[dict]]]) -> None:
    all_ids: list[str] = []
    for source_name, result_set in zip(SOURCE_NAMES, result_sets, strict=True):
        id_sequences = {
            kind: [row["question_id"] for row in rows]
            for kind, rows in result_set.items()
        }
        if len({tuple(ids) for ids in id_sequences.values()}) != 1:
            raise ValueError(f"Question order is not aligned in {source_name}")
        ids = next(iter(id_sequences.values()))
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate question IDs within {source_name}")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Source runs contain overlapping question IDs")


def rebuild_eval(rows: list[dict], model: str, num_runs: int) -> dict:
    run_scores: list[float] = []
    category_counts: dict[str, dict] = defaultdict(
        lambda: {"correct": [0] * num_runs, "total": 0}
    )

    for run_index in range(num_runs):
        key = f"judgment_{run_index + 1}"
        judged = [row for row in rows if key in row.get("llm_judgments", {})]
        run_scores.append(
            sum(bool(row["llm_judgments"][key]) for row in judged) / len(judged)
        )
        for row in judged:
            category = row.get("category")
            if category is not None and row["llm_judgments"][key]:
                category_counts[str(category)]["correct"][run_index] += 1

    for row in rows:
        category = row.get("category")
        if category is not None:
            category_counts[str(category)]["total"] += 1

    category_accuracies = {}
    for category, stats in category_counts.items():
        accuracies = [count / stats["total"] for count in stats["correct"]]
        category_accuracies[category] = {
            "mean": mean(accuracies),
            "std": pstdev(accuracies),
            "individual_runs": accuracies,
            "total": stats["total"],
        }

    accuracy = mean(run_scores)
    return {
        "total_questions": len(rows),
        "correct": int(accuracy * len(rows)),
        "accuracy": accuracy,
        "detailed_results": {"default_group": rows},
        "metadata": {
            "model": model,
            "num_runs": num_runs,
            "mean_accuracy": accuracy,
            "std_accuracy": pstdev(run_scores),
            "run_scores": run_scores,
            "category_accuracies": category_accuracies,
        },
    }


def main() -> None:
    output_dir = RESULTS_DIR / OUTPUT_NAME
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    sources = [RESULTS_DIR / name for name in SOURCE_NAMES]
    source_evals = [load_json(path / "eval_results.json") for path in sources]
    result_sets = []
    for source, eval_payload in zip(sources, source_evals, strict=True):
        result_sets.append(
            {
                "search": load_json(source / "search_results.json"),
                "answer": load_json(source / "answer_results.json"),
                "eval": flatten_detailed_results(eval_payload),
            }
        )
    assert_unique_and_aligned(result_sets)

    models = {payload["metadata"]["model"] for payload in source_evals}
    run_counts = {payload["metadata"]["num_runs"] for payload in source_evals}
    if len(models) != 1 or len(run_counts) != 1:
        raise ValueError("Source evaluator configurations do not match")

    search_rows = [row for result_set in result_sets for row in result_set["search"]]
    answer_rows = [row for result_set in result_sets for row in result_set["answer"]]
    eval_rows = [row for result_set in result_sets for row in result_set["eval"]]
    eval_payload = rebuild_eval(eval_rows, models.pop(), run_counts.pop())
    created_at = datetime.now(timezone.utc).isoformat()

    output_dir.mkdir(parents=False)
    write_json(output_dir / "search_results.json", search_rows)
    write_json(output_dir / "answer_results.json", answer_rows)
    write_json(output_dir / "eval_results.json", eval_payload)
    write_json(
        output_dir / "checkpoint_default.json",
        {
            "run_name": "default",
            "last_updated": created_at,
            "completed_stages": ["search", "answer", "evaluate"],
            "search_results": search_rows,
            "answer_results": answer_rows,
            "eval_results": eval_payload,
        },
    )
    write_json(
        output_dir / "merge_manifest.json",
        {
            "merged_from": [
                {
                    "source_run": name,
                    "total_questions": payload["total_questions"],
                    "correct": payload["correct"],
                    "accuracy": payload["accuracy"],
                }
                for name, payload in zip(SOURCE_NAMES, source_evals, strict=True)
            ],
            "merge_created_at": created_at,
            "merge_note": (
                "Merged complementary LongMemEval question partitions into a "
                "synthetic complete result view. Per-partition rag_index.json files "
                "were intentionally not copied."
            ),
        },
    )

    report = "\n".join(
        [
            "=" * 60,
            "📊 Evaluation Report",
            "=" * 60,
            "",
            "System: RAG",
            "Time Elapsed: 1306.56s (sum of source reports)",
            "",
            f"Total Questions: {eval_payload['total_questions']}",
            f"Correct: {eval_payload['correct']}",
            f"Accuracy: {eval_payload['accuracy']:.2%}",
            "",
            "=" * 60,
        ]
    )
    (output_dir / "report.txt").write_text(report, encoding="utf-8")
    (output_dir / "pipeline.log").write_text(
        "Merged from complementary LongMemEval partition runs; no pipeline was "
        "executed for this synthetic complete run.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
