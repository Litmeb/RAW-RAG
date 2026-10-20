"""Build an EverMemOS MEME answer-only run with conditional modality restored.

The intervention is deliberately narrow:

* keep native retrieval results, ordering, scores, and every unrelated context byte;
* consider every MEME Cascade question, without consulting its answer/eval outcome;
* when the summary sentence containing the annotated replacement value uses a
  weak modality (would/plan/potential), restore explicit ``if ... will``: use a
  token-only modal rewrite when possible, otherwise replace only that sentence
  with the matching original user clause.

The output is a fresh run directory containing ``search_results.json`` plus a
complete replacement audit.  It is intended for the CLI answer fast path.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WEAK_MODAL_RE = re.compile(
    r"\b(?:would|plan|plans|planned|planning|potential|potentially)\b", re.I
)
IF_WILL_RE = re.compile(r"\bif\b.*\bwill\b", re.I | re.S)
MEMORY_SEPARATOR_RE = re.compile(r"(\n\s*---\s*\n)")
SENTENCE_RE = re.compile(r".*?(?:[.!?](?=\s|$)|$)", re.S)
ORACLE_FOLLOWUP_RE = re.compile(
    r"\s+(?:Any tips or things to watch out for\?|Do you have any thoughts on that\?).*$",
    re.I | re.S,
)


def normalized(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def contains_value(text: str, value: str) -> bool:
    needle = normalized(value)
    return bool(needle and needle in normalized(text))


def extract_original_clause(row: dict[str, Any], value: str) -> str | None:
    """Find the original user-authored conditional clause for ``value``."""
    candidates: list[str] = []
    source = row.get("official_source") or {}
    for session in source.get("sessions") or []:
        for message in session.get("conversation") or []:
            if str(message.get("role", "")).lower() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if not contains_value(content, value) or not IF_WILL_RE.search(content):
                continue
            clause = ORACLE_FOLLOWUP_RE.sub("", content).strip()
            if clause:
                candidates.append(clause)
    if not candidates:
        return None
    # Prefer the shortest exact rule when a session repeats the same value.
    return min(candidates, key=len)


@dataclass(frozen=True)
class Replacement:
    value: str
    old_sentence: str
    new_sentence: str
    original_clause: str
    block_index: int
    rewrite_mode: str


def _memory_payload(block: str) -> tuple[str, str]:
    """Separate the global context preamble from the first memory payload."""
    match = re.match(
        r"(?s)^(Episodes memories for conversation between .*?:\s*\n\s*)(.*)$",
        block,
    )
    if match:
        return match.group(1), match.group(2)
    return "", block


def _replace_in_block(
    block: str, value: str, original_clause: str, block_index: int
) -> tuple[str, Replacement | None]:
    prefix, payload = _memory_payload(block)
    # EverMemOS formats each block as ``<memory title>: <summary body>``.
    # Keep the title out of sentence matching so it is never overwritten.
    title_match = re.match(r"(?s)^(\s*[^\n]*?:\s*)(.*)$", payload)
    title = title_match.group(1) if title_match else ""
    body = title_match.group(2) if title_match else payload
    candidates: list[tuple[int, int, str]] = []
    for match in SENTENCE_RE.finditer(body):
        sentence = match.group(0)
        if contains_value(sentence, value) and WEAK_MODAL_RE.search(sentence):
            candidates.append((match.start(), match.end(), sentence))
    if not candidates:
        return block, None

    # Prefer a sentence explicitly attributed to the user, then the shortest
    # sentence. This avoids replacing a later assistant paraphrase.
    start, end, old_sentence = min(
        candidates,
        key=lambda item: (
            0 if re.search(r"\b(?:the )?user\b", item[2], re.I) else 1,
            len(item[2]),
        ),
    )
    leading = old_sentence[: len(old_sentence) - len(old_sentence.lstrip())]
    trailing = old_sentence[len(old_sentence.rstrip()) :]
    stripped = old_sentence.strip()

    # If the summary retained the condition itself, change only the weakened
    # modal token. Otherwise restoring ``if ... will`` requires the original
    # user clause; replacement remains confined to the one matching sentence.
    restored = re.sub(r"\bwould\b", "will", stripped, flags=re.I)
    restored = re.sub(
        r"\b(?:plan|plans|planned)\s+to\b", "will", restored, flags=re.I
    )
    token_rewrite = IF_WILL_RE.search(restored) is not None
    token_rewrite = token_rewrite and not re.search(
        r"\b(?:potential|potentially|planning)\b", stripped, re.I
    )
    if token_rewrite and restored != stripped:
        replacement_text = restored
        rewrite_mode = "modal_token_only"
    else:
        replacement_text = f'The User stated: "{original_clause}"'
        rewrite_mode = "original_if_will_clause"
    new_sentence = f"{leading}{replacement_text}{trailing}"
    new_body = body[:start] + new_sentence + body[end:]
    return prefix + title + new_body, Replacement(
        value=value,
        old_sentence=old_sentence.strip(),
        new_sentence=new_sentence.strip(),
        original_clause=original_clause,
        block_index=block_index,
        rewrite_mode=rewrite_mode,
    )


def restore_value_modality(
    context: str, value: str, original_clause: str
) -> tuple[str, Replacement | None]:
    parts = MEMORY_SEPARATOR_RE.split(context)
    block_number = 0
    for index in range(0, len(parts), 2):
        updated, replacement = _replace_in_block(
            parts[index], value, original_clause, block_number
        )
        block_number += 1
        if replacement is not None:
            parts[index] = updated
            return "".join(parts), replacement
    return context, None


def target_after_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    metadata = row.get("meme_metadata") or {}
    for state in (metadata.get("target_states") or {}).values():
        if isinstance(state, dict) and state.get("after"):
            value = str(state["after"])
            if value not in values:
                values.append(value)
    return values


def build_run(
    source_search: Path, diagnostics: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    search_rows = json.loads(source_search.read_text(encoding="utf-8"))
    diagnostic_rows = json.loads(diagnostics.read_text(encoding="utf-8"))
    diagnostics_by_id = {row["question_id"]: row for row in diagnostic_rows}

    output_rows = copy.deepcopy(search_rows)
    audit_rows: list[dict[str, Any]] = []
    counters = {
        "search_rows": len(search_rows),
        "cascade_rows": 0,
        "candidate_values": 0,
        "replacements": 0,
        "missing_original_clause": 0,
        "value_not_in_context": 0,
        "no_weak_summary_sentence": 0,
    }

    for search_row in output_rows:
        question_id = search_row["question_id"]
        diagnostic = diagnostics_by_id.get(question_id)
        if not diagnostic or diagnostic.get("task_type") != "Cas":
            continue
        counters["cascade_rows"] += 1
        metadata = search_row.setdefault("retrieval_metadata", {})
        context = str(metadata.get("formatted_context") or "")

        for value in target_after_values(diagnostic):
            counters["candidate_values"] += 1
            if not contains_value(context, value):
                counters["value_not_in_context"] += 1
                continue
            original_clause = extract_original_clause(diagnostic, value)
            if original_clause is None:
                counters["missing_original_clause"] += 1
                continue
            updated_context, replacement = restore_value_modality(
                context, value, original_clause
            )
            if replacement is None:
                counters["no_weak_summary_sentence"] += 1
                continue
            context = updated_context
            counters["replacements"] += 1
            audit_rows.append(
                {
                    "question_id": question_id,
                    "phase": diagnostic.get("phase"),
                    "task_type": diagnostic.get("task_type"),
                    "target_value": replacement.value,
                    "block_index": replacement.block_index,
                    "old_sentence": replacement.old_sentence,
                    "new_sentence": replacement.new_sentence,
                    "original_clause": replacement.original_clause,
                    "rewrite_mode": replacement.rewrite_mode,
                }
            )
        metadata["formatted_context"] = context

    output_dir.mkdir(parents=True)
    (output_dir / "search_results.json").write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "intervention": "restore_conditional_modality_from_summary_to_original_if_will",
        "source_search_results": str(source_search.resolve()),
        "source_diagnostics": str(diagnostics.resolve()),
        "selection_uses_answer_or_eval_outcome": False,
        "counters": counters,
        "replacements": audit_rows,
    }
    (output_dir / "conditional_modality_restore_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-search", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_run(args.source_search, args.diagnostics, args.output_dir)
    print(json.dumps(audit["counters"], ensure_ascii=False, indent=2))
    print(f"Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
