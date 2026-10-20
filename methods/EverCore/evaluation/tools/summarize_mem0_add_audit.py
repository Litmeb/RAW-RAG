"""Summarize and validate ``mem0_add_audit.jsonl`` from a Mem0 Local run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable


MUTATION_EVENTS = {"ADD", "UPDATE", "DELETE"}


def _event(value: Any) -> str:
    event = str(value or "MISSING").strip().upper()
    return event or "MISSING"


def _decision_actions(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for call in record.get("llm_calls", []):
        if call.get("stage") != "memory_action_decision":
            continue
        actions = call.get("proposed_actions")
        if actions is None:
            parsed = call.get("parsed_response")
            actions = parsed.get("memory", []) if isinstance(parsed, dict) else []
        for action in actions or []:
            if isinstance(action, dict):
                yield action


def _returned_actions(record: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    result = record.get("add_result")
    if not isinstance(result, dict):
        return
    for action in result.get("results", []) or []:
        if isinstance(action, dict):
            yield action


def _counter_dict(counter: Counter) -> Dict[str, int]:
    return dict(sorted(counter.items()))


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def summarize_audit(path: Path) -> Dict[str, Any]:
    statuses: Counter = Counter()
    decision_all: Counter = Counter()
    decision_success: Counter = Counter()
    returned_events: Counter = Counter()
    persisted_events: Counter = Counter()
    conversations: dict[str, Counter] = defaultdict(Counter)
    batch_attempts: Counter = Counter()
    mismatches = []
    malformed_lines = []
    llm_stages: Counter = Counter()
    totals: Counter = Counter()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                malformed_lines.append(
                    {
                        "line_number": line_number,
                        "error": str(error),
                        "prefix": line[:200],
                    }
                )
                continue

            totals["records"] += 1
            status = str(record.get("status", "missing"))
            statuses[status] += 1
            context = record.get("context") or {}
            conversation_id = str(context.get("conversation_id", "unknown"))
            batch_key = "|".join(
                [
                    conversation_id,
                    str(context.get("speaker", "")),
                    str(context.get("batch_index", "")),
                ]
            )
            batch_attempts[batch_key] += 1
            conversations[conversation_id]["attempts"] += 1
            conversations[conversation_id][f"status_{status}"] += 1

            for call in record.get("llm_calls", []):
                llm_stages[str(call.get("stage", "unknown"))] += 1
                if call.get("stage") == "fact_extraction":
                    totals["raw_extracted_facts"] += len(
                        call.get("extracted_facts") or []
                    )
            totals["normalized_facts_searched"] += len(
                [
                    fact
                    for fact in record.get("normalized_facts_searched", [])
                    if fact is not None
                ]
            )
            totals["vector_searches"] += len(record.get("vector_searches", []))
            totals["retrieved_old_memories"] += sum(
                len(search.get("results", []) or [])
                for search in record.get("vector_searches", [])
            )

            proposed_mutations: Counter = Counter()
            for action in _decision_actions(record):
                event = _event(action.get("event"))
                decision_all[event] += 1
                conversations[conversation_id][f"decision_{event}"] += 1
                if status == "success":
                    decision_success[event] += 1
                if event in MUTATION_EVENTS:
                    proposed_mutations[event] += 1

            for action in _returned_actions(record):
                returned_events[_event(action.get("event"))] += 1

            committed: Counter = Counter()
            for row in record.get("persisted_history_rows", []):
                event = _event(row.get("event"))
                persisted_events[event] += 1
                conversations[conversation_id][f"persisted_{event}"] += 1
                committed[event] += 1
                if event == "UPDATE":
                    old_memory = row.get("old_memory")
                    new_memory = row.get("new_memory")
                    if not isinstance(old_memory, str) or not isinstance(
                        new_memory, str
                    ):
                        totals["persisted_updates_missing_text"] += 1
                    elif old_memory == new_memory:
                        totals["persisted_updates_exact_duplicate"] += 1
                    else:
                        totals["persisted_updates_effective"] += 1

            if proposed_mutations != committed:
                mismatches.append(
                    {
                        "line_number": line_number,
                        "operation_id": record.get("operation_id"),
                        "status": status,
                        "context": context,
                        "proposed_mutations": _counter_dict(proposed_mutations),
                        "persisted_mutations": _counter_dict(committed),
                        "error": record.get("error"),
                    }
                )

    retried_batches = {
        key: attempts for key, attempts in batch_attempts.items() if attempts > 1
    }
    decision_total = sum(decision_all.values())
    proposed_updates = decision_all["UPDATE"]
    proposed_deletes = decision_all["DELETE"]
    persisted_updates = persisted_events["UPDATE"]
    persisted_deletes = persisted_events["DELETE"]
    effective_updates = totals["persisted_updates_effective"]
    effective_mutations = effective_updates + persisted_deletes
    return {
        "schema_version": 1,
        "audit_file": str(path.resolve()),
        "records": {
            "total": totals["records"],
            "status": _counter_dict(statuses),
            "valid_json_lines": totals["records"],
            "malformed_json_lines": len(malformed_lines),
        },
        "coverage": {
            "unique_batches": len(batch_attempts),
            "retried_batches": len(retried_batches),
            "max_attempts_for_one_batch": max(batch_attempts.values(), default=0),
            "llm_calls_by_stage": _counter_dict(llm_stages),
            "raw_extracted_facts": totals["raw_extracted_facts"],
            "normalized_facts_searched": totals["normalized_facts_searched"],
            "vector_searches": totals["vector_searches"],
            "retrieved_old_memories": totals["retrieved_old_memories"],
        },
        "events": {
            "decision_all_attempts": _counter_dict(decision_all),
            "decision_successful_attempts": _counter_dict(decision_success),
            "returned_by_memory_add": _counter_dict(returned_events),
            "persisted_to_history": _counter_dict(persisted_events),
            "decision_total": decision_total,
            "decision_shares_percent": {
                event: _percentage(count, decision_total)
                for event, count in sorted(decision_all.items())
            },
            "proposed_update_delete": {
                "count": proposed_updates + proposed_deletes,
                "share_percent": _percentage(
                    proposed_updates + proposed_deletes, decision_total
                ),
            },
            "update_validity": {
                "proposed": proposed_updates,
                "persisted": persisted_updates,
                "unpersisted_or_invalid_references": max(
                    proposed_updates - persisted_updates, 0
                ),
                "exact_old_new_duplicates": totals[
                    "persisted_updates_exact_duplicate"
                ],
                "persisted_missing_old_or_new": totals[
                    "persisted_updates_missing_text"
                ],
                "effective_text_changes": effective_updates,
                "effective_share_of_proposed_percent": _percentage(
                    effective_updates, proposed_updates
                ),
                "effective_share_of_all_decisions_percent": _percentage(
                    effective_updates, decision_total
                ),
            },
            "delete_validity": {
                "proposed": proposed_deletes,
                "persisted": persisted_deletes,
                "unpersisted_or_invalid_references": max(
                    proposed_deletes - persisted_deletes, 0
                ),
            },
            "effective_update_delete": {
                "count": effective_mutations,
                "share_percent": _percentage(effective_mutations, decision_total),
            },
        },
        "consistency": {
            "operations_with_proposed_vs_persisted_mismatch": len(mismatches),
            "mismatches": mismatches,
            "retried_batch_attempts": dict(sorted(retried_batches.items())),
            "malformed_lines": malformed_lines,
        },
        "per_conversation": {
            conversation_id: _counter_dict(counts)
            for conversation_id, counts in sorted(conversations.items())
        },
    }


def _markdown(summary: Dict[str, Any]) -> str:
    records = summary["records"]
    coverage = summary["coverage"]
    events = summary["events"]
    consistency = summary["consistency"]
    lines = [
        "# Mem0 Add audit summary",
        "",
        f"- Audit file: `{summary['audit_file']}`",
        f"- Valid records: `{records['total']}`",
        f"- Status: `{json.dumps(records['status'], ensure_ascii=False)}`",
        f"- Unique batches: `{coverage['unique_batches']}`",
        f"- Retried batches: `{coverage['retried_batches']}`",
        f"- Raw extracted facts: `{coverage['raw_extracted_facts']}`",
        f"- Normalized facts searched: `{coverage['normalized_facts_searched']}`",
        f"- Retrieved old memories: `{coverage['retrieved_old_memories']}`",
        "",
        "## Event counts",
        "",
        f"- Decision, all attempts: `{json.dumps(events['decision_all_attempts'], ensure_ascii=False)}`",
        f"- Decision, successful attempts: `{json.dumps(events['decision_successful_attempts'], ensure_ascii=False)}`",
        f"- Returned by `Memory.add`: `{json.dumps(events['returned_by_memory_add'], ensure_ascii=False)}`",
        f"- Persisted to history: `{json.dumps(events['persisted_to_history'], ensure_ascii=False)}`",
        f"- Decision shares: `{json.dumps(events['decision_shares_percent'], ensure_ascii=False)}`",
        f"- Proposed UPDATE+DELETE: `{events['proposed_update_delete']['count']}` "
        f"(`{events['proposed_update_delete']['share_percent']:.4f}%`)",
        f"- Effective UPDATEs (persisted, old != new): "
        f"`{events['update_validity']['effective_text_changes']}` / "
        f"`{events['update_validity']['proposed']}` proposed "
        f"(`{events['update_validity']['effective_share_of_proposed_percent']:.4f}%`)",
        f"- Exact old/new duplicate UPDATEs: "
        f"`{events['update_validity']['exact_old_new_duplicates']}`",
        f"- Unpersisted/invalid UPDATE references: "
        f"`{events['update_validity']['unpersisted_or_invalid_references']}`",
        f"- Effective UPDATE + persisted DELETE: "
        f"`{events['effective_update_delete']['count']}` "
        f"(`{events['effective_update_delete']['share_percent']:.4f}%` of decisions)",
        "",
        "## Consistency",
        "",
        f"- Proposed/persisted mismatches: `{consistency['operations_with_proposed_vs_persisted_mismatch']}`",
        f"- Malformed JSONL lines: `{records['malformed_json_lines']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-file",
        default="mem0_add_audit.jsonl",
        help="Audit JSONL filename inside --run-dir.",
    )
    parser.add_argument(
        "--output-prefix",
        default="mem0_add_audit_summary",
        help="Output filename prefix inside --run-dir.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    audit_path = run_dir / args.audit_file
    if not audit_path.is_file():
        raise FileNotFoundError(f"Audit file not found: {audit_path}")

    summary = summarize_audit(audit_path)
    json_path = run_dir / f"{args.output_prefix}.json"
    markdown_path = run_dir / f"{args.output_prefix}.md"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    markdown_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
