"""Analyze LongMemEval accuracy after excluding *_abs questions from one run."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        help="Run name under methods/EverCore/evaluation/results/",
    )
    parser.add_argument(
        "--run-dir",
        help="Absolute or relative path to a run directory containing eval_results.json",
    )
    parser.add_argument(
        "--suffix",
        default="_abs",
        help="Question-id suffix to exclude. Default: _abs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the text report",
    )
    return parser.parse_args()


def resolve_run_dir(args) -> Path:
    if bool(args.run_name) == bool(args.run_dir):
        raise SystemExit("Provide exactly one of --run-name or --run-dir.")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = Path("methods/EverCore/evaluation/results") / args.run_name

    run_dir = run_dir.resolve()
    eval_path = run_dir / "eval_results.json"
    if not eval_path.is_file():
        raise SystemExit(f"eval_results.json not found: {eval_path}")
    return run_dir


def load_detailed_results(eval_path: Path):
    with eval_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    grouped = payload.get("detailed_results", {})
    flattened = []
    if isinstance(grouped, dict):
        for items in grouped.values():
            if isinstance(items, list):
                flattened.extend(items)
    elif isinstance(grouped, list):
        flattened.extend(grouped)
    return flattened


def compute_stats(results):
    run_keys = []
    if results:
        sample_keys = results[0].get("llm_judgments", {}).keys()
        run_keys = sorted(
            sample_keys,
            key=lambda key: int(key.split("_")[-1]) if key.split("_")[-1].isdigit() else key,
        )

    run_scores = []
    category_totals = defaultdict(int)
    category_correct = defaultdict(lambda: [0] * len(run_keys))

    for item in results:
        category = item.get("category", "unknown")
        category_totals[category] += 1

    for run_idx, run_key in enumerate(run_keys):
        correct = 0
        total = 0
        for item in results:
            judgment = item.get("llm_judgments", {}).get(run_key)
            category = item.get("category", "unknown")
            if judgment is None:
                continue
            total += 1
            if judgment:
                correct += 1
                category_correct[category][run_idx] += 1
        run_scores.append((correct / total) if total else 0.0)

    category_stats = {}
    for category in sorted(category_totals):
        total = category_totals[category]
        per_run = [
            (category_correct[category][idx] / total) if total else 0.0
            for idx in range(len(run_keys))
        ]
        category_stats[category] = {
            "total": total,
            "run_accuracies": per_run,
            "mean_accuracy": mean(per_run) if per_run else 0.0,
            "std_accuracy": pstdev(per_run) if per_run else 0.0,
        }

    return {
        "total_questions": len(results),
        "run_keys": run_keys,
        "run_accuracies": run_scores,
        "mean_accuracy": mean(run_scores) if run_scores else 0.0,
        "std_accuracy": pstdev(run_scores) if run_scores else 0.0,
        "categories": category_stats,
    }


def render_text(run_dir: Path, suffix: str, kept_results, removed_results, stats):
    print("=" * 72)
    print(f"LongMemEval no-{suffix} analysis: {run_dir.name}")
    print("=" * 72)
    print(f"Run dir: {run_dir}")
    print(f"Excluded suffix: {suffix}")
    print(f"Kept questions: {len(kept_results)}")
    print(f"Removed questions: {len(removed_results)}")
    print()
    print(
        f"Overall: {stats['mean_accuracy']:.4f} +/- {stats['std_accuracy']:.4f} "
        f"({stats['mean_accuracy'] * 100:.2f}% +/- {stats['std_accuracy'] * 100:.2f}%)"
    )
    print("Run accuracies: " + ", ".join(f"{x:.4f}" for x in stats["run_accuracies"]))
    print()
    print("By category:")
    for category, category_stats in stats["categories"].items():
        print(
            f"  {category:<24} "
            f"{category_stats['mean_accuracy']:.4f} +/- {category_stats['std_accuracy']:.4f} "
            f"(n={category_stats['total']}, runs={[round(x, 4) for x in category_stats['run_accuracies']]})"
        )


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args)
    detailed_results = load_detailed_results(run_dir / "eval_results.json")
    removed_results = [
        item for item in detailed_results if str(item.get("question_id", "")).endswith(args.suffix)
    ]
    kept_results = [
        item for item in detailed_results if not str(item.get("question_id", "")).endswith(args.suffix)
    ]
    stats = compute_stats(kept_results)

    output = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "excluded_suffix": args.suffix,
        "kept_questions": len(kept_results),
        "removed_questions": len(removed_results),
        **stats,
    }

    if args.json:
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    render_text(run_dir, args.suffix, kept_results, removed_results, stats)


if __name__ == "__main__":
    main()
