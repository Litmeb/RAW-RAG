"""Audit MEME evidence with deterministic, task-specific state verifiers.

The verifier deliberately distinguishes answer-bearing evidence from state
transition evidence.  It is stricter than bag-of-strings matching for Tr/Cas
and phase-aware for Del/Abs.  The output is diagnostic rather than a replacement
for human semantic annotation: exact synthetic entity values anchor every rule,
while relation, temporal, and deletion cues are deterministic proxies.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from evaluation.scripts.analyze_meme_deterministic_evidence import (
    DATA,
    RESULTS,
    RUNS,
    load_scores,
    norm,
    qid_parts,
    strings,
)
from evaluation.scripts.build_meme_continuity_oracle_runs import task_for_question


EVAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = EVAL_DIR / "analysis/meme_task_specific_evidence_v2_20260818"
DEFAULT_SEMANTIC = (
    EVAL_DIR
    / "analysis/meme_seven_systems_full_flash_20260817"
    / "llm_answerability_spotcheck.csv"
)

RELATION_CUES = (
    "depend", "determin", "influenc", "affect", "because", "due to",
    "if ", "when ", "conditional", "based on", "linked", "lead to",
    "result in", "change", "switch",
)
TEMPORAL_CUES = (
    " then ", " later ", " subsequently ", " afterward", " before ",
    " after ", " previously ", " formerly ", " now ", " currently ",
    " recently ", " switched ", " changed ", " moved from ", " replaced ",
)
DELETION_CUES = (
    "remove", "removed", "delete", "deleted", "forget", "forgot",
    "erase", "erased", "no longer remember", "don't remember",
    "do not remember", "explicitly removed",
)

ISO_DATE_RE = re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b")
NORMALIZED_ISO_DATE_RE = re.compile(
    r"\b(20\d{2})\s+(\d{1,2})\s+(\d{1,2})(?=\s|t|$)"
)
MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"),
            ("april", "apr"), ("may",), ("june", "jun"),
            ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"),
            ("october", "oct"), ("november", "nov"), ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
DAY_MONTH_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_RE})\s*,?\s*(20\d{{2}})\b",
    re.I,
)
MONTH_DAY_RE = re.compile(
    rf"\b({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(20\d{{2}})\b",
    re.I,
)


@dataclass(frozen=True)
class Verification:
    sufficient: bool
    value_complete: bool = False
    order_support: bool = False
    dependency_rule_support: bool = False
    trigger_support: bool = False
    deletion_event_support: bool = False
    non_disclosure: bool = False
    reason: str = ""


def unique_normalized(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(value for item in values for value in [norm(item)] if value))


def value_positions(context: str, value: str) -> list[int]:
    return [match.start() for match in re.finditer(re.escape(value), context)]


def date_markers(context: str) -> list[tuple[int, datetime]]:
    markers: list[tuple[int, datetime]] = []
    for pattern in (ISO_DATE_RE, NORMALIZED_ISO_DATE_RE):
        for match in pattern.finditer(context):
            try:
                date = datetime(*map(int, match.groups()))
            except ValueError:
                continue
            markers.append((match.start(), date))
    for match in DAY_MONTH_RE.finditer(context):
        day, month, year = match.groups()
        markers.append((match.start(), datetime(int(year), MONTHS[month.lower()], int(day))))
    for match in MONTH_DAY_RE.finditer(context):
        month, day, year = match.groups()
        markers.append((match.start(), datetime(int(year), MONTHS[month.lower()], int(day))))
    return sorted(set(markers))


def occurrence_dates(context: str, value: str, markers: list[tuple[int, datetime]]) -> list[datetime]:
    dates = []
    for position in value_positions(context, value):
        preceding = [(p, date) for p, date in markers if p <= position and position - p <= 700]
        if preceding:
            dates.append(max(preceding, key=lambda item: item[0])[1])
    return sorted(set(dates))


def has_dated_chain(context: str, values: list[str]) -> bool:
    markers = date_markers(context)
    choices = [occurrence_dates(context, value, markers) for value in values]
    if not values or any(not dates for dates in choices):
        return False
    previous: datetime | None = None
    for dates in choices:
        candidates = [date for date in dates if previous is None or date > previous]
        if not candidates:
            return False
        previous = min(candidates)
    return True


def has_cued_textual_chain(context: str, values: list[str]) -> bool:
    if len(values) < 2:
        return bool(values and values[0] in context)
    positions = [value_positions(context, value) for value in values]
    if any(not found for found in positions):
        return False
    previous = -1
    chosen: list[int] = []
    for candidates in positions:
        later = [position for position in candidates if position > previous]
        if not later:
            return False
        previous = min(later)
        chosen.append(previous)
    for left, right in zip(chosen, chosen[1:]):
        window = f" {context[max(0, left - 120):min(len(context), right + 180)]} "
        if not any(cue in window for cue in TEMPORAL_CUES):
            return False
    return True


def linked_by_relation(context: str, target_values: list[str], source_signals: list[str]) -> bool:
    for target in target_values:
        for position in value_positions(context, target):
            window = f" {context[max(0, position - 500):position + len(target) + 500]} "
            if any(source in window for source in source_signals) and any(
                cue in window for cue in RELATION_CUES
            ):
                return True
    return False


def linked_deletion(context: str, target_values: list[str], target_entities: list[str]) -> bool:
    anchors = [*target_values, *target_entities]
    for anchor in anchors:
        for position in value_positions(context, anchor):
            window = f" {context[max(0, position - 350):position + len(anchor) + 350]} "
            if any(cue in window for cue in DELETION_CUES):
                return True
    return False


def entity_state(episode: dict[str, Any], entity: str, phase: str) -> list[str]:
    state = (episode.get("entities") or {}).get(entity, {})
    key = "before" if phase == "before" else "after"
    return unique_normalized(strings(state.get(key)))


def verify(
    context_raw: str,
    episode: dict[str, Any],
    question: dict[str, Any],
    phase: str,
) -> Verification:
    context = norm(context_raw)
    task = str(question.get("task_type") or "")
    task_spec = task_for_question(episode, question)
    target_entities = [str(value) for value in task_spec.get("target_entities") or []]
    question_values = unique_normalized(strings(question.get("entity_values")))
    value_complete = bool(question_values) and all(value in context for value in question_values)

    if task == "ER":
        return Verification(value_complete, value_complete=value_complete, reason="verbatim value")

    if task == "Agg":
        return Verification(value_complete, value_complete=value_complete, reason="all target entity values")

    if task == "Tr":
        dated = value_complete and has_dated_chain(context, question_values)
        cued = value_complete and has_cued_textual_chain(context, question_values)
        ordered = dated or cued
        return Verification(
            value_complete and ordered,
            value_complete=value_complete,
            order_support=ordered,
            reason="complete ordered history" if ordered else "missing chronology",
        )

    target_before = unique_normalized(
        value
        for entity in target_entities
        for value in strings((episode.get("entities") or {}).get(entity, {}).get("before"))
    )
    before_complete = bool(target_before) and all(value in context for value in target_before)

    if task == "Del":
        if phase == "before":
            return Verification(before_complete, value_complete=before_complete, reason="pre-deletion value")
        entity_labels = unique_normalized(entity.replace("_", " ") for entity in target_entities)
        deletion = linked_deletion(context, target_before, entity_labels)
        non_disclosure = all(value not in context for value in target_before)
        return Verification(
            deletion,
            value_complete=before_complete,
            deletion_event_support=deletion,
            non_disclosure=non_disclosure,
            reason="targeted deletion event" if deletion else "missing targeted deletion event",
        )

    cascade_source = str(task_spec.get("cascade_source") or "")
    source = (episode.get("entities") or {}).get(cascade_source, {})
    source_before = unique_normalized(strings(source.get("before")))
    source_after = unique_normalized(strings(source.get("after")))
    source_labels = unique_normalized([cascade_source.replace("_", " ")])
    source_signals = list(dict.fromkeys([*source_labels, *source_before, *source_after]))
    trigger = bool(source_after) and all(value in context for value in source_after)

    if task == "Cas":
        if phase == "before":
            return Verification(before_complete, value_complete=before_complete, reason="pre-cascade value")
        target_after = unique_normalized(
            value
            for entity in target_entities
            for value in strings((episode.get("entities") or {}).get(entity, {}).get("after"))
        )
        after_complete = bool(target_after) and all(value in context for value in target_after)
        dependency = linked_by_relation(context, target_after, source_signals)
        sufficient = after_complete and dependency and trigger
        return Verification(
            sufficient,
            value_complete=after_complete,
            dependency_rule_support=dependency,
            trigger_support=trigger,
            reason="cascade rule, trigger, and replacement" if sufficient else "incomplete cascade path",
        )

    if task == "Abs":
        if phase == "before":
            return Verification(before_complete, value_complete=before_complete, reason="pre-invalidation value")
        dependency = linked_by_relation(context, target_before, source_signals)
        sufficient = before_complete and dependency and trigger
        return Verification(
            sufficient,
            value_complete=before_complete,
            dependency_rule_support=dependency,
            trigger_support=trigger,
            reason="dependency and invalidating trigger" if sufficient else "incomplete invalidation path",
        )

    raise ValueError(f"Unsupported MEME task: {task}")


def load_semantic(path: Path) -> dict[tuple[str, str], bool]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get("llm_answerable", "")).lower()
            if value in {"true", "false"}:
                result[(row["system"], row["question_id"])] = value == "true"
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for group, rows in sorted(grouped.items()):
        sufficient = [row for row in rows if row["evidence_sufficient"]]
        output.append(
            {
                **dict(zip(keys, group)),
                "n": len(rows),
                "accuracy": mean(row["correct"] for row in rows),
                "evidence_sufficient_rate": mean(row["evidence_sufficient"] for row in rows),
                "sufficient_n": len(sufficient),
                "reader_utilization": mean(row["correct"] for row in sufficient) if sufficient else "",
                "sufficient_correct_n": sum(row["correct"] for row in sufficient),
                "value_complete_rate": mean(row["value_complete"] for row in rows),
                "order_support_rate": mean(row["order_support"] for row in rows),
                "dependency_rule_support_rate": mean(row["dependency_rule_support"] for row in rows),
                "trigger_support_rate": mean(row["trigger_support"] for row in rows),
                "deletion_event_support_rate": mean(row["deletion_event_support"] for row in rows),
                "non_disclosure_rate": mean(row["non_disclosure"] for row in rows),
            }
        )
    return output


def comparison_rows(
    records: list[dict[str, Any]], semantic: dict[tuple[str, str], bool], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if (row["system"], row["question_id"]) in semantic:
            grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for group, rows in sorted(grouped.items()):
        pairs = [
            (bool(row["evidence_sufficient"]), semantic[(row["system"], row["question_id"])])
            for row in rows
        ]
        tp = sum(predicted and actual for predicted, actual in pairs)
        fp = sum(predicted and not actual for predicted, actual in pairs)
        fn = sum(not predicted and actual for predicted, actual in pairs)
        tn = len(pairs) - tp - fp - fn
        output.append(
            {
                **dict(zip(keys, group)),
                "valid_n": len(pairs),
                "task_specific_rate": mean(int(predicted) for predicted, _ in pairs),
                "semantic_rate": mean(int(actual) for _, actual in pairs),
                "precision_vs_semantic": tp / (tp + fp) if tp + fp else "",
                "recall_vs_semantic": tp / (tp + fn) if tp + fn else "",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return output


def parse_run_spec(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Run must be LABEL=PATH")
    return label.strip(), path.strip()


def parse_decomposition_spec(value: str) -> tuple[str, tuple[str, str, str]]:
    family, separator, labels = value.partition("=")
    parts = tuple(part.strip() for part in labels.split(","))
    if not separator or not family.strip() or len(parts) != 3 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "Decomposition must be FAMILY=RAW_LABEL,STORE_LABEL,NATIVE_LABEL"
        )
    return family.strip(), parts


def representation_decomposition(
    records: list[dict[str, Any]],
    specs: list[tuple[str, tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    """Summarize Raw→Store→Native coverage loss and native readout loss."""
    by_label: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        by_label[str(row["system"])][str(row["question_id"])] = row
    output = []
    for family, (raw_label, store_label, native_label) in specs:
        label_maps = [by_label[label] for label in (raw_label, store_label, native_label)]
        question_ids = set(label_maps[0])
        if any(set(rows) != question_ids for rows in label_maps[1:]):
            raise ValueError(f"Question alignment failed for decomposition {family}")
        ordered = [label_maps[index][qid] for index in range(3) for qid in sorted(question_ids)]
        del ordered  # Alignment is validated above; aggregate each condition directly.
        rates = []
        for rows in label_maps:
            values = list(rows.values())
            coverage = mean(row["evidence_sufficient"] for row in values)
            sufficient = [row for row in values if row["evidence_sufficient"]]
            utilization = mean(row["correct"] for row in sufficient) if sufficient else 0.0
            rates.append((coverage, utilization, mean(row["correct"] for row in values)))
        raw, store, native = rates
        output.append(
            {
                "family": family,
                "n": len(question_ids),
                "raw_label": raw_label,
                "store_label": store_label,
                "native_label": native_label,
                "raw_coverage": raw[0],
                "store_coverage": store[0],
                "native_coverage": native[0],
                "representation_loss_raw_minus_store": raw[0] - store[0],
                "retrieval_loss_store_minus_native": store[0] - native[0],
                "raw_reader_utilization": raw[1],
                "store_reader_utilization": store[1],
                "native_reader_utilization": native[1],
                "raw_readout_loss_sufficient_but_wrong": raw[0] * (1 - raw[1]),
                "store_readout_loss_sufficient_but_wrong": store[0] * (1 - store[1]),
                "native_readout_loss_sufficient_but_wrong": native[0] * (1 - native[1]),
                "raw_accuracy": raw[2],
                "store_accuracy": store[2],
                "native_accuracy": native[2],
            }
        )
    return output


def render_report(
    summaries: list[dict[str, Any]], decompositions: list[dict[str, Any]]
) -> str:
    def pct(value: Any) -> str:
        return f"{float(value) * 100:.2f}%"

    lines = [
        "# MEME task-specific evidence verifier v2",
        "",
        "Coverage is the deterministic task-specific evidence-sufficiency rate. Reader "
        "utilization is correctness conditional on sufficient evidence.",
        "",
        "## Condition comparison",
        "",
        "| Condition | Accuracy | Coverage | Reader utilization |",
        "|---|---:|---:|---:|",
    ]
    for row in summaries:
        utilization = row["reader_utilization"]
        lines.append(
            f"| {row['system']} | {pct(row['accuracy'])} | "
            f"{pct(row['evidence_sufficient_rate'])} | "
            f"{pct(utilization) if utilization != '' else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Raw → Store → Native decomposition",
            "",
            "| Family | Raw/Store/Native coverage | Representation loss | Retrieval loss | Raw/Store/Native utilization | Raw/Store/Native readout loss |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in decompositions:
        lines.append(
            f"| {row['family']} | {pct(row['raw_coverage'])}/"
            f"{pct(row['store_coverage'])}/{pct(row['native_coverage'])} | "
            f"{pct(row['representation_loss_raw_minus_store'])} | "
            f"{pct(row['retrieval_loss_store_minus_native'])} | "
            f"{pct(row['raw_reader_utilization'])}/"
            f"{pct(row['store_reader_utilization'])}/"
            f"{pct(row['native_reader_utilization'])} | "
            f"{pct(row['raw_readout_loss_sufficient_but_wrong'])}/"
            f"{pct(row['store_readout_loss_sufficient_but_wrong'])}/"
            f"{pct(row['native_readout_loss_sufficient_but_wrong'])} |"
        )
    lines.extend(
        [
            "",
            "Representation loss is Raw coverage minus complete-store coverage; retrieval "
            "loss is complete-store minus native coverage. Readout loss is the share of all "
            "questions with sufficient evidence but an incorrect answer, reported per condition.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--semantic-csv", type=Path, default=DEFAULT_SEMANTIC)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        default=None,
        help="Repeatable LABEL=PATH override; replaces the built-in seven-system run set.",
    )
    parser.add_argument(
        "--decomposition",
        action="append",
        type=parse_decomposition_spec,
        default=[],
        help="Repeatable FAMILY=RAW_LABEL,STORE_LABEL,NATIVE_LABEL comparison.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite analysis directory: {args.output_dir}")

    episodes = json.loads(args.data.read_text(encoding="utf-8"))
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    records: list[dict[str, Any]] = []
    configured_runs = (
        [
            (system, run_name)
            for system, run_names in RUNS.items()
            for run_name in run_names
        ]
        if args.run is None
        else args.run
    )
    for system, run_path in configured_runs:
        run = Path(run_path)
        if not run.is_absolute():
            run = args.results_root / run
        run_name = run.name
        answers = json.loads((run / "answer_results.json").read_text(encoding="utf-8"))
        scores = load_scores(run / "eval_results.json")
        for item in answers:
            question_id = str(item.get("question_id"))
            if question_id not in scores:
                continue
            episode_id, phase, question_index = qid_parts(question_id)
            episode = episodes_by_id[episode_id]
            question = episode[f"{phase}_questions"]["questions"][question_index]
            result = verify(
                str(item.get("formatted_context") or ""), episode, question, phase
            )
            records.append(
                {
                    "system": system,
                    "run_name": run_name,
                    "question_id": question_id,
                    "conversation_id": item.get("conversation_id"),
                    "phase": phase,
                    "task_type": question.get("task_type"),
                    "score": scores[question_id],
                    "correct": int(scores[question_id] >= 0.5),
                    "evidence_sufficient": int(result.sufficient),
                    "value_complete": int(result.value_complete),
                    "order_support": int(result.order_support),
                    "dependency_rule_support": int(result.dependency_rule_support),
                    "trigger_support": int(result.trigger_support),
                    "deletion_event_support": int(result.deletion_event_support),
                    "non_disclosure": int(result.non_disclosure),
                    "reason": result.reason,
                }
            )

    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_question.csv", records)
    system_summary = aggregate(records, ("system",))
    write_csv(args.output_dir / "summary_by_system.csv", system_summary)
    write_csv(
        args.output_dir / "summary_by_system_task_phase.csv",
        aggregate(records, ("system", "task_type", "phase")),
    )
    semantic = load_semantic(args.semantic_csv)
    write_csv(
        args.output_dir / "comparison_by_system.csv",
        comparison_rows(records, semantic, ("system",)),
    )
    write_csv(
        args.output_dir / "comparison_by_task_phase.csv",
        comparison_rows(records, semantic, ("task_type", "phase")),
    )
    decomposition = representation_decomposition(records, args.decomposition)
    write_csv(args.output_dir / "representation_decomposition.csv", decomposition)
    (args.output_dir / "report.md").write_text(
        render_report(system_summary, decomposition), encoding="utf-8"
    )
    manifest = {
        "version": "meme-task-specific-evidence-v2",
        "runs": [{"label": label, "path": path} for label, path in configured_runs],
        "decompositions": decomposition,
        "records": len(records),
        "semantic_comparison": str(args.semantic_csv),
        "rules": {
            "ER": "all verbatim target values",
            "Agg": "all target-entity values",
            "Tr": "all history values plus dated or explicitly cued chronological chain",
            "Del_before": "old value",
            "Del_after": "targeted deletion event; non-disclosure reported separately",
            "Cas_before": "pre-cascade target value",
            "Cas_after": "replacement value plus dependency rule plus upstream after-state trigger",
            "Abs_before": "pre-invalidation target value",
            "Abs_after": "old target plus dependency rule plus upstream after-state trigger",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} task-specific MEME rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
