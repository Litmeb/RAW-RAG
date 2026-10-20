"""Recompute official MEME scores from an existing eval_results.json.

This is intentionally judge-free: it only reapplies the aggregation and
trivial-pass rules to already judged detailed results.
"""

import argparse
import json
from pathlib import Path

from evaluation.src.evaluators.meme_judge import summarize_meme_results


TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")


def recompute(result_dir: Path) -> dict:
    eval_path = result_dir / "eval_results.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    official = summarize_meme_results(data["detailed_results"])
    metadata = data.setdefault("metadata", {})
    metadata["official_accuracy"] = official["overall_accuracy"]
    metadata["task_scores"] = official["task_scores"]
    metadata["trivial_analysis"] = official["trivial_analysis"]
    data["accuracy"] = official["overall_accuracy"]
    eval_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    system = data.get("metadata", {}).get("system", result_dir.name)
    lines = [
        "=" * 60,
        "MEME Evaluation Report",
        "=" * 60,
        "",
        f"System: {system}",
        "Scoring: mean of ER, Agg, Tr, Del, Cas, Abs",
        "",
        "Task Accuracy (official MEME scoring):",
    ]
    for task in TASKS:
        score = official["task_scores"][task]
        lines.append(
            f"  {task}: {score['pass']:.2f}/{score['total']} "
            f"({score['accuracy']:.2%})"
        )
    lines.extend(["", f"Accuracy (MEME Overall): {official['overall_accuracy']:.2%}", "", "=" * 60])
    (result_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    return official


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for result_dir in args.result_dirs:
        official = recompute(result_dir)
        print(f"{result_dir}: {official['overall_accuracy']:.6f}")


if __name__ == "__main__":
    main()
