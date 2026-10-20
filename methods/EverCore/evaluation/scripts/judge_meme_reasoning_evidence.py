"""Task-aware LLM evidence audit for MEME reasoning-chain questions.

The existing semantic answerability audit is deliberately preserved.  This
audit re-judges only questions whose evidence contract requires more than the
answer value itself: Tr (both phases), and Del/Cas/Abs after questions.  The
remaining rows inherit the existing semantic judgment in the combined output.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation.scripts.analyze_meme_deterministic_evidence import DATA, qid_parts, strings
from evaluation.scripts.build_meme_continuity_oracle_runs import task_for_question
from evaluation.scripts.judge_answerability_spotcheck import (
    build_answer_index,
    extract_json,
    load_dotenv,
)


EVAL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EVAL_DIR / "analysis/meme_seven_systems_full_flash_20260817"
DEFAULT_ENV = EVAL_DIR.parent / ".env"
REASONING_TASKS = {("Tr", "before"), ("Tr", "after"), ("Del", "after"),
                   ("Cas", "after"), ("Abs", "after")}
COMPONENT_FIELDS = (
    "value_complete",
    "temporal_chain",
    "deletion_targeted",
    "dependency_rule",
    "upstream_trigger_state",
    "linked",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def unique_strings(value: Any) -> list[str]:
    return list(dict.fromkeys(text.strip() for text in strings(value) if text.strip()))


def evidence_contract(episode: dict[str, Any], question: dict[str, Any], phase: str) -> dict[str, Any]:
    """Build hidden-gold evidence anchors; anchors never count as retrieved evidence."""
    task = str(question.get("task_type") or "")
    spec = task_for_question(episode, question)
    entities = episode.get("entities") or {}
    targets = [str(value) for value in spec.get("target_entities") or []]
    target_before = unique_strings([
        value for entity in targets for value in strings((entities.get(entity) or {}).get("before"))
    ])
    target_after = unique_strings([
        value for entity in targets for value in strings((entities.get(entity) or {}).get("after"))
    ])
    source_entity = str(spec.get("cascade_source") or "")
    source = entities.get(source_entity) or {}
    contract = {
        "task_type": task,
        "phase": phase,
        "target_entities": targets,
        "history_values": unique_strings(question.get("entity_values")),
        "old_target_values": target_before,
        "new_target_values": target_after,
        "upstream_entity": source_entity,
        "upstream_before_values": unique_strings(source.get("before")),
        "upstream_after_values": unique_strings(source.get("after")),
    }
    if task == "Tr":
        contract["sufficiency_rule"] = (
            "Every history value must appear, and the context must establish their chronological "
            "order using dates or explicit temporal language such as then/later/after."
        )
    elif task == "Del":
        contract["sufficiency_rule"] = (
            "An explicit deletion instruction/event must point to the target entity or old value. "
            "Merely failing to retrieve the old value is not deletion evidence."
        )
    elif task == "Cas":
        contract["sufficiency_rule"] = (
            "The new target value, the dependency rule, and the upstream post-trigger state must "
            "all appear and be linked into one executable causal chain."
        )
    elif task == "Abs":
        contract["sufficiency_rule"] = (
            "The old target value, its dependency rule, and the upstream state that invalidates it "
            "must all appear and be linked into one executable causal chain."
        )
    else:
        raise ValueError(f"No reasoning-chain contract for {task}:{phase}")
    return contract


def build_prompt(question_text: str, contract: dict[str, Any], context: str) -> str:
    return f"""You are a strict MEME retrieval-evidence auditor.

Judge whether the RETRIEVED CONTEXT satisfies the task-specific evidence contract. The hidden-gold
contract tells you what to look for, but it is NOT evidence. Only statements in RETRIEVED CONTEXT
count. Do not use the question, contract, world knowledge, or assumptions to fill a missing link.
For cascade/invalidation, isolated mentions are insufficient: the dependency rule, trigger state,
and target state must be related. For deletion, absence of an old value is never proof of deletion.

Return one JSON object with exactly these keys:
{{"sufficient": true/false, "value_complete": true/false,
 "temporal_chain": true/false, "deletion_targeted": true/false,
 "dependency_rule": true/false, "upstream_trigger_state": true/false,
 "linked": true/false, "notes": "brief evidence-based reason"}}
Use false for component fields that do not apply.

QUESTION:
{question_text}

HIDDEN-GOLD EVIDENCE CONTRACT (requirements only; not evidence):
{json.dumps(contract, ensure_ascii=False, indent=2)}

RETRIEVED CONTEXT:
{context}
"""


def parse_judgment(text: str) -> dict[str, Any]:
    parsed = extract_json(text)
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        # Do not accept the old answerability schema accidentally.
        return {"valid": False, "raw": text[:500]}
    result = {"valid": True, "sufficient": sufficient, "notes": str(parsed.get("notes") or "")}
    for field in COMPONENT_FIELDS:
        result[field] = parsed.get(field) if isinstance(parsed.get(field), bool) else False
    return result


def judge_one(
    base_row: dict[str, str], item: dict[str, Any], contract: dict[str, Any], model: str,
    base_url: str, api_key: str, max_context_chars: int, max_tokens: int, timeout: int,
) -> dict[str, Any]:
    context = str(item.get("formatted_context") or "")
    truncated = max_context_chars > 0 and len(context) > max_context_chars
    if truncated:
        context = context[:max_context_chars]
    prompt = build_prompt(str(item.get("question") or ""), contract, context)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You audit complete reasoning-chain evidence and return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    last_text = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
            message = body["choices"][0]["message"]
            last_text = str(message.get("content") or message.get("reasoning_content") or "")
            judgment = parse_judgment(last_text)
            if judgment.get("valid"):
                return {
                    **base_row,
                    "audit_model": model,
                    "context_chars_audited": len(context),
                    "context_truncated": int(truncated),
                    "chain_sufficient": judgment["sufficient"],
                    **{field: judgment[field] for field in COMPONENT_FIELDS},
                    "audit_status": "valid",
                    "audit_notes": judgment["notes"],
                }
            last_error = ValueError("response lacks a boolean sufficient field")
        except (urllib.error.URLError, http.client.HTTPException, OSError, socket.timeout,
                TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    return {
        **base_row,
        "audit_model": model,
        "context_chars_audited": len(context),
        "context_truncated": int(truncated),
        "chain_sufficient": "",
        **{field: "" for field in COMPONENT_FIELDS},
        "audit_status": "parse_error" if last_text else "error",
        "audit_notes": (last_text or str(last_error))[:500],
    }


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["system"]), str(row["question_id"])


def is_valid(row: dict[str, Any]) -> bool:
    return str(row.get("chain_sufficient", "")).lower() in {"true", "false"}


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for group, items in sorted(grouped.items()):
        valid = [item for item in items if str(item.get(value, "")).lower() in {"true", "false"}]
        output.append({
            **dict(zip(keys, group)), "n": len(items), "valid_n": len(valid),
            "coverage": mean(str(item[value]).lower() == "true" for item in valid) if valid else "",
        })
    return output


def render_report(model: str, reasoning: list[dict[str, Any]], combined: list[dict[str, Any]]) -> str:
    chain_summary = summarize(reasoning, ("system",), "chain_sufficient")
    combined_summary = summarize(combined, ("system",), "evidence_sufficient")
    combined_by_system = {row["system"]: row for row in combined_summary}
    lines = [
        "# MEME task-aware LLM evidence audit", "", f"Model: `{model}`", "",
        "Only `Tr`, `Del-after`, `Cas-after`, and `Abs-after` are re-judged. Other task/phase rows",
        "retain the original semantic answerability judgment. Retrieved contexts are complete unless",
        "`context_truncated=1` is explicitly recorded.", "",
        "| System | Re-audited valid/N | Chain coverage | Combined coverage |", "|---|---:|---:|---:|",
    ]
    for row in chain_summary:
        combined_row = combined_by_system[row["system"]]
        chain = "n/a" if row["coverage"] == "" else f"{100 * float(row['coverage']):.2f}%"
        total = "n/a" if combined_row["coverage"] == "" else f"{100 * float(combined_row['coverage']):.2f}%"
        lines.append(f"| {row['system']} | {row['valid_n']}/{row['n']} | {chain} | {total} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-analysis-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context-chars", type=int, default=0,
                        help="0 audits the complete context; positive values explicitly truncate.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-samples", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite audit directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(args.env_path)
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY (or LLM_BASE_URL/LLM_API_KEY) is missing")

    semantic_rows = load_csv(args.input_analysis_dir / "llm_answerability_spotcheck.csv")
    manifest = json.loads((args.input_analysis_dir / "run_manifest.json").read_text(encoding="utf-8"))
    answer_index = build_answer_index(manifest)
    episodes = json.loads(args.data.read_text(encoding="utf-8"))
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}

    tasks: list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]] = []
    for row in semantic_rows:
        task_phase = (str(row.get("task_type")), str(row.get("phase")))
        if task_phase not in REASONING_TASKS:
            continue
        item = answer_index.get((row["dataset"], row["system"], row["split"], row["source"], row["question_id"]))
        if item is None:
            continue
        episode_id, phase, question_index = qid_parts(row["question_id"])
        episode = episodes_by_id[episode_id]
        question = episode[f"{phase}_questions"]["questions"][question_index]
        base = {key: row.get(key, "") for key in (
            "dataset", "system", "split", "source", "question_id", "conversation_id",
            "phase", "task_type", "score", "correctish", "llm_answerable",
        )}
        tasks.append((base, item, evidence_contract(episode, question, phase)))

    output_csv = args.output_dir / "reasoning_audit.csv"
    retained: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and output_csv.exists():
        retained = {row_key(row): row for row in load_csv(output_csv) if is_valid(row)}
    pending = [task for task in tasks if row_key(task[0]) not in retained]
    if args.max_new_samples > 0:
        pending = pending[:args.max_new_samples]
    results: list[dict[str, Any]] = list(retained.values())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(
            judge_one, base, item, contract, args.model, base_url, api_key,
            args.max_context_chars, args.max_tokens, args.timeout,
        ) for base, item, contract in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0:
                results.sort(key=row_key)
                write_csv(output_csv, results)
                print(f"Judged {completed}/{len(pending)} new reasoning-chain rows", flush=True)
    results.sort(key=row_key)
    write_csv(output_csv, results)

    audited = {row_key(row): row for row in results if is_valid(row)}
    combined = []
    for row in semantic_rows:
        key = row_key(row)
        task_phase = (str(row.get("task_type")), str(row.get("phase")))
        if task_phase in REASONING_TASKS:
            evidence = audited.get(key, {}).get("chain_sufficient", "")
            source = "reasoning_chain_llm" if evidence != "" else "pending_reasoning_audit"
        else:
            evidence = row.get("llm_answerable", "")
            source = "original_semantic_llm"
        combined.append({
            "system": row["system"], "question_id": row["question_id"],
            "conversation_id": row.get("conversation_id", ""),
            "phase": row.get("phase", ""), "task_type": row.get("task_type", ""),
            "score": row.get("score", ""),
            "correct": int(float(row.get("score", 0) or 0) >= 0.5),
            "correctish": row.get("correctish", ""),
            "evidence_sufficient": evidence, "coverage_source": source,
        })
    write_csv(args.output_dir / "combined_coverage.csv", combined)
    write_csv(args.output_dir / "summary_reasoning_by_system_task_phase.csv",
              summarize(results, ("system", "task_type", "phase"), "chain_sufficient"))
    write_csv(args.output_dir / "summary_combined_by_system.csv",
              summarize(combined, ("system",), "evidence_sufficient"))
    (args.output_dir / "report.md").write_text(render_report(args.model, results, combined), encoding="utf-8")
    metadata = {
        "version": "meme-task-aware-llm-evidence-v1", "model": args.model,
        "input_analysis_dir": str(args.input_analysis_dir), "data": str(args.data),
        "complete_context": args.max_context_chars == 0,
        "max_context_chars": args.max_context_chars, "target_rows": len(tasks),
        "valid_rows": len(audited), "pending_rows": len(tasks) - len(audited),
        "rejudged_conditions": sorted(f"{task}:{phase}" for task, phase in REASONING_TASKS),
        "inherited_conditions": "all other task/phase rows from original semantic LLM audit",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Valid reasoning judgments: {len(audited)}/{len(tasks)}; output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
