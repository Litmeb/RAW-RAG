"""Estimate evidence-conditioned reader utilization from a spot-check.

The estimator reweights semantic answerability rates sampled within the four
deterministic evidence buckets by their full-run bucket counts.  It reports
the probability that a system's answer is correct conditional on its retrieved
context being judged sufficient for the question and gold answer.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis" / "evidence_answerability"
BUCKETS = (
    "correct_high_evidence",
    "correct_low_evidence",
    "wrong_high_evidence",
    "wrong_low_evidence",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "system", "n", "estimated_semantic_evidence_coverage",
        "estimated_sufficient_and_correct", "estimated_sufficient_and_wrong",
        "estimated_reader_utilization", "estimated_reader_failure_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()

    full_rows = load_csv(args.analysis_dir / "summary_by_system.csv")
    spot_rows = load_csv(args.analysis_dir / "llm_answerability_spotcheck_summary.csv")
    rates = {
        (row["dataset"], row["system"], row["bucket"]): float(row["llm_answerable_rate"])
        for row in spot_rows
        if int(row["valid_n"]) > 0
    }
    estimates: list[dict[str, Any]] = []
    for row in full_rows:
        key = (row["dataset"], row["system"])
        missing = [bucket for bucket in BUCKETS if key + (bucket,) not in rates]
        if missing:
            raise SystemExit(f"Missing semantic spot-check rates for {key}: {', '.join(missing)}")
        sufficient_correct = sum(
            float(row[bucket]) * rates[key + (bucket,)]
            for bucket in BUCKETS
            if bucket.startswith("correct_")
        )
        sufficient_wrong = sum(
            float(row[bucket]) * rates[key + (bucket,)]
            for bucket in BUCKETS
            if bucket.startswith("wrong_")
        )
        sufficient = sufficient_correct + sufficient_wrong
        n = float(row["n"])
        utilization = sufficient_correct / sufficient if sufficient else 0.0
        estimates.append(
            {
                "dataset": key[0],
                "system": key[1],
                "n": int(n),
                "estimated_semantic_evidence_coverage": sufficient / n if n else 0.0,
                "estimated_sufficient_and_correct": sufficient_correct,
                "estimated_sufficient_and_wrong": sufficient_wrong,
                "estimated_reader_utilization": utilization,
                "estimated_reader_failure_rate": 1.0 - utilization,
            }
        )
    estimates.sort(key=lambda row: (row["dataset"], row["system"]))
    output = args.analysis_dir / "reader_utilization_estimate.csv"
    write_csv(output, estimates)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
