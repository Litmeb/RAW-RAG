"""Summarize MEME subset accuracy from evaluation result directories.

Examples:
    python scripts/meme_subset_accuracy.py meme-rag-normal
    python scripts/meme_subset_accuracy.py --best-by-method
    python scripts/meme_subset_accuracy.py --best-by-method --format csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = EVALUATION_DIR / "results"
DEFAULT_SYSTEMS_DIR = EVALUATION_DIR / "config" / "systems"
DEFAULT_FULL_TOTAL = 1188


def load_eval(result_dir: Path) -> dict[str, Any]:
    eval_path = result_dir / "eval_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"missing eval_results.json: {eval_path}")
    with eval_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_result_dir(run: str, results_root: Path) -> Path:
    candidate = Path(run)
    if candidate.exists():
        return candidate.resolve()

    candidate = results_root / run
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"result directory not found: {run}")


def flatten_detailed_results(detailed_results: Any) -> list[dict[str, Any]]:
    if isinstance(detailed_results, dict):
        flattened: list[dict[str, Any]] = []
        for value in detailed_results.values():
            if isinstance(value, list):
                flattened.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                flattened.append(value)
        return flattened
    if isinstance(detailed_results, list):
        return [item for item in detailed_results if isinstance(item, dict)]
    return []


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_from_item(item: dict[str, Any], metric: str) -> float:
    phase = str(item.get("phase") or "")
    if metric == "filtered" and phase == "after" and "filtered_u_pass" in item:
        return to_float(item.get("filtered_u_pass"))
    if "u_score" in item:
        return to_float(item.get("u_score"))
    if "score" in item:
        return to_float(item.get("score"))
    return 1.0 if item.get("is_correct") or item.get("u_pass") else 0.0


def subset_key(item: dict[str, Any]) -> str:
    phase = str(item.get("phase") or "unknown")
    task_type = str(item.get("task_type") or item.get("category") or "unknown")
    return f"{phase}:{task_type}"


def subset_counts_from_details(data: dict[str, Any], metric: str) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = defaultdict(lambda: {"pass": 0.0, "total": 0.0})
    for item in flatten_detailed_results(data.get("detailed_results")):
        key = subset_key(item)
        counts[key]["pass"] += score_from_item(item, metric)
        counts[key]["total"] += 1.0
    return dict(counts)


def subset_counts_from_metadata(data: dict[str, Any], metric: str) -> dict[str, dict[str, float]]:
    metadata = data.get("metadata") or {}
    raw_counts = metadata.get("per_phase_task_counts")
    if not isinstance(raw_counts, dict):
        return {}

    counts = {
        str(key): {
            "pass": to_float(value.get("pass") if isinstance(value, dict) else None),
            "total": to_float(value.get("total") if isinstance(value, dict) else None),
        }
        for key, value in raw_counts.items()
    }

    if metric == "filtered":
        filtered_counts = metadata.get("per_phase_task_filtered_counts")
        if isinstance(filtered_counts, dict):
            for key, value in filtered_counts.items():
                key = str(key)
                if key.startswith("after:") and isinstance(value, dict):
                    counts[key] = {
                        "pass": to_float(value.get("pass")),
                        "total": to_float(value.get("total")),
                    }

    return counts


def get_subset_counts(data: dict[str, Any], metric: str) -> dict[str, dict[str, float]]:
    counts = subset_counts_from_metadata(data, metric)
    if counts:
        return counts
    return subset_counts_from_details(data, metric)


def sort_subset_key(key: str) -> tuple[int, str]:
    phase, _, task = key.partition(":")
    phase_order = {"before": 0, "after": 1}
    task_order = {"Tr": 0, "Cas": 1, "Abs": 2, "Del": 3, "Agg": 4, "ER": 5}
    return (phase_order.get(phase, 9), f"{task_order.get(task, 99):02d}:{task}")


def subset_rows(data: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    rows = []
    for key, counts in sorted(get_subset_counts(data, metric).items(), key=lambda item: sort_subset_key(item[0])):
        total = to_float(counts.get("total"))
        correct = to_float(counts.get("pass"))
        rows.append(
            {
                "subset": key,
                "correct": correct,
                "total": int(total) if total.is_integer() else total,
                "accuracy": correct / total if total else 0.0,
            }
        )
    return rows


def format_number(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_markdown(rows: list[dict[str, Any]], columns: list[str]) -> None:
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if column.endswith("accuracy") or column.startswith("accuracy") or ":" in column:
                value = format_percent(to_float(value)) if value != "" else ""
            elif column == "correct":
                value = format_number(value)
            values.append(str(value))
        print("| " + " | ".join(values) + " |")


def print_csv_rows(rows: list[dict[str, Any]], columns: list[str], delimiter: str) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=columns, delimiter=delimiter, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})


def print_table(rows: list[dict[str, Any]], columns: list[str], fmt: str) -> None:
    if fmt == "markdown":
        print_markdown(rows, columns)
    elif fmt == "csv":
        print_csv_rows(rows, columns, ",")
    elif fmt == "tsv":
        print_csv_rows(rows, columns, "\t")
    else:
        raise ValueError(f"unsupported format: {fmt}")


def load_system_names(systems_dir: Path) -> list[str]:
    if not systems_dir.exists():
        return []
    return sorted((path.stem for path in systems_dir.glob("*.yaml")), key=len, reverse=True)


def method_from_run_name(run_name: str, system_names: list[str]) -> str:
    rest = run_name.removeprefix("meme-")
    for system_name in system_names:
        if rest == system_name or rest.startswith(f"{system_name}-"):
            return system_name
    return rest.split("-", 1)[0]


def iter_meme_runs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith("meme-") and (path / "eval_results.json").exists()
    )


def best_by_method_rows(
    results_root: Path,
    systems_dir: Path,
    metric: str,
    include_incomplete: bool,
    full_total: int,
) -> list[dict[str, Any]]:
    system_names = load_system_names(systems_dir)
    best: dict[str, tuple[Path, dict[str, Any]]] = {}

    for result_dir in iter_meme_runs(results_root):
        data = load_eval(result_dir)
        total = int(data.get("total_questions") or 0)
        if not include_incomplete and total != full_total:
            continue

        method = method_from_run_name(result_dir.name, system_names)
        current = best.get(method)
        if current is None or to_float(data.get("accuracy")) > to_float(current[1].get("accuracy")):
            best[method] = (result_dir, data)

    rows = []
    for method, (result_dir, data) in sorted(best.items()):
        row: dict[str, Any] = {
            "method": method,
            "run": result_dir.name,
            "accuracy": to_float(data.get("accuracy")),
            "correct": to_float(data.get("correct")),
            "total": data.get("total_questions"),
        }
        for subset_row in subset_rows(data, metric):
            row[subset_row["subset"]] = subset_row["accuracy"]
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MEME subset accuracy.")
    parser.add_argument("run", nargs="?", help="Run directory path or name under evaluation/results.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--systems-dir", type=Path, default=DEFAULT_SYSTEMS_DIR)
    parser.add_argument("--metric", choices=["filtered", "raw"], default="filtered")
    parser.add_argument("--format", choices=["markdown", "csv", "tsv"], default="markdown")
    parser.add_argument("--best-by-method", action="store_true", help="Summarize the best full MEME run for each method.")
    parser.add_argument("--include-incomplete", action="store_true", help="Include smoke/partial runs in --best-by-method.")
    parser.add_argument("--full-total", type=int, default=DEFAULT_FULL_TOTAL)
    args = parser.parse_args()

    if args.best_by_method:
        rows = best_by_method_rows(
            results_root=args.results_root,
            systems_dir=args.systems_dir,
            metric=args.metric,
            include_incomplete=args.include_incomplete,
            full_total=args.full_total,
        )
        if not rows:
            print("No MEME runs found.", file=sys.stderr)
            return 1
        subset_columns = sorted(
            {column for row in rows for column in row if ":" in column},
            key=sort_subset_key,
        )
        print_table(rows, ["method", "run", "accuracy", "correct", "total", *subset_columns], args.format)
        return 0

    if not args.run:
        parser.error("provide a run name/path or use --best-by-method")

    result_dir = resolve_result_dir(args.run, args.results_root)
    data = load_eval(result_dir)
    rows = subset_rows(data, args.metric)
    print(f"# {result_dir.name}")
    print()
    print(f"- metric: {args.metric}")
    print(f"- overall accuracy: {format_percent(to_float(data.get('accuracy')))}")
    print(f"- correct/total: {format_number(to_float(data.get('correct')))}/{data.get('total_questions')}")
    print()
    print_table(rows, ["subset", "accuracy", "correct", "total"], args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
