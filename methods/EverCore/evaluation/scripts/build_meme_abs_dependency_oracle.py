"""Build an Abs-only MEME oracle with a value-free dependency-validity rule.

This is an answer-readout ablation over an existing answer-location oracle. It
does not change raw evidence or add entity values/gold answers. Only Abs
questions receive a generic rule for invalidating a dependent fact after its
upstream dependency changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from evaluation.scripts.build_meme_continuity_oracle_runs import (
    DEFAULT_DATA,
    load_json,
    question_specs,
)


EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RUN = (
    EVALUATION_DIR
    / "results/meme-rag-answer-location-oracle-20260731-deepseek-v4-flash"
)
DEPENDENCY_VALIDITY_RULE = """<DEPENDENCY_VALIDITY_RULE>
For a current-answer question, a fact explicitly stated to depend on another
fact becomes stale if the context later says that dependency changed. Do not
report that earlier dependent value as the current answer. If no explicit new
value for the dependent fact is stated after the change, answer exactly: I
don't have that information.
</DEPENDENCY_VALIDITY_RULE>
"""


def abs_question_ids(episodes: list[dict[str, Any]]) -> set[str]:
    return {
        spec.question_id
        for spec in question_specs(episodes, 0, len(episodes))
        if spec.task_type == "Abs"
    }


def build_rows(
    source_rows: list[dict[str, Any]], question_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_question_id = {str(row.get("question_id")): row for row in source_rows}
    missing = sorted(question_ids - set(by_question_id))
    if missing:
        raise ValueError(f"Source oracle is missing Abs questions: {missing[:3]}")

    rows = []
    audit = []
    for question_id in sorted(question_ids):
        row = copy.deepcopy(by_question_id[question_id])
        metadata = row.setdefault("retrieval_metadata", {})
        original_context = str(metadata.get("formatted_context") or "")
        metadata["formatted_context"] = DEPENDENCY_VALIDITY_RULE + "\n" + original_context
        metadata["abs_dependency_validity_rule"] = True
        metadata["abs_rule_contains_gold_values"] = False
        rows.append(row)
        audit.append(
            {
                "question_id": question_id,
                "rule": "dependency_validity_rule",
                "context_characters_before": len(original_context),
                "context_characters_after": len(metadata["formatted_context"]),
            }
        )
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {output_dir}")
    episodes = load_json(args.data.resolve())
    source_rows = load_json(args.source_run.resolve() / "search_results.json")
    rows, audit = build_rows(source_rows, abs_question_ids(episodes))
    output_dir.mkdir(parents=True)
    (output_dir / "search_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "abs_dependency_rule_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Built Abs dependency-rule oracle: {output_dir} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
