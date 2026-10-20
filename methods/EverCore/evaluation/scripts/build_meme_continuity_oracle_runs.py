"""Build MEME gold-evidence continuity controls for several memory systems.

Unlike the older EventQA experiment, MEME supplies an episode-level list of
official evidence sessions and question-level target entities, entity values,
and gold answers.  This script uses those fields directly:

* ``raw_gold_evidence`` exposes every official evidence session visible at the
  question's before/after cutoff, in source order.
* each ``*_memory_oracle`` exposes only memory-object text previously returned
  anywhere for that conversation by the corresponding system.  The pool is
  deduplicated, phase-safe, and selected deterministically using the MEME gold
  fields.  It is deliberately called an *observed-pool* oracle rather than a
  complete-store oracle.

The generated directories contain search_results.json and are intended for the
evaluation CLI's ``--stages answer evaluate --conv-range`` fast path.  Source
runs are read-only and output directories are never reused.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken


EVALUATION_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_DATA = EVALUATION_DIR / "data/meme/meme_filler32k.json"
DEFAULT_TAG = "meme-continuity-gold-fields-task-readout-conv0-5-v5-20260717"
DEFAULT_SYSTEM_RUNS = {
    "mem0": "meme-mem0-normal-merged",
    "memos": "meme-memos-time-filter",
    "emem": "meme-emem-normal",
    "evermemos": "meme-evermemos-normal-3",
}
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QuestionSpec:
    question_id: str
    conversation_id: str
    episode_index: int
    phase: str
    question: str
    task_type: str
    target_entities: tuple[str, ...]
    entity_values: dict[str, Any]
    gold_answer: str
    cascade_source: str
    cutoff: int
    support_values: tuple[str, ...]


@dataclass(frozen=True)
class MemoryObject:
    content: str
    score: float
    metadata: dict[str, Any]
    first_seen: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(flatten_strings(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_strings(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def normalized_text(value: str) -> str:
    return " ".join(WORD_RE.findall(str(value).lower()))


def tokens(value: str) -> set[str]:
    return set(WORD_RE.findall(str(value).lower()))


def task_for_question(episode: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    entities = tuple(question.get("entity") or [])
    matches = [
        task
        for task in episode.get("tasks", [])
        if str(task.get("type")) == str(question.get("task_type"))
        and tuple(task.get("target_entities") or []) == entities
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one MEME task for type={question.get('task_type')} "
            f"entities={entities}, found {len(matches)}"
        )
    return matches[0]


def question_specs(episodes: list[dict[str, Any]], start: int, end: int) -> list[QuestionSpec]:
    specs = []
    for episode_index in range(start, end):
        episode = episodes[episode_index]
        episode_id = str(episode["episode_id"])
        for phase in ("before", "after"):
            block = episode[f"{phase}_questions"]
            cutoff = int(block["position_after_session"])
            for question_index, question in enumerate(block.get("questions", [])):
                task = task_for_question(episode, question)
                gold_answer = str(
                    question.get("expected_answer", question.get("gold_answer", ""))
                )
                target_entities = tuple(task.get("target_entities") or question.get("entity") or [])
                if tuple(question.get("entity") or []) != target_entities:
                    raise ValueError(f"Target-entity mismatch in {episode_id} {phase} qa{question_index}")

                support = []
                for entity in target_entities:
                    state = (episode.get("entities") or {}).get(entity, {})
                    support.extend(flatten_strings(state.get("before")))
                    if phase == "after":
                        support.extend(flatten_strings(state.get("after")))
                        support.extend(flatten_strings(state.get("history")))
                cascade_source = str(task.get("cascade_source") or "")
                if cascade_source:
                    state = (episode.get("entities") or {}).get(cascade_source, {})
                    support.extend(flatten_strings(state.get("before")))
                    if phase == "after":
                        support.extend(flatten_strings(state.get("after")))
                support.extend(flatten_strings(question.get("entity_values") or {}))

                specs.append(
                    QuestionSpec(
                        question_id=f"meme_{episode_id}_{phase}_qa{question_index}",
                        conversation_id=f"meme_{episode_index}",
                        episode_index=episode_index,
                        phase=phase,
                        question=str(question.get("question") or ""),
                        task_type=str(question.get("task_type") or ""),
                        target_entities=target_entities,
                        entity_values=dict(question.get("entity_values") or {}),
                        gold_answer=gold_answer,
                        cascade_source=cascade_source,
                        cutoff=cutoff,
                        support_values=tuple(dict.fromkeys(value for value in support if value)),
                    )
                )
    return specs


def memory_pool(
    search_rows: list[dict[str, Any]], conversation_id: str, phase: str
) -> list[MemoryObject]:
    """Return a deduplicated, phase-safe pool observed across native searches."""
    candidates: dict[str, MemoryObject] = {}
    seen_index = 0
    for row in search_rows:
        if str(row.get("conversation_id")) != conversation_id:
            continue
        row_phase = "before" if "_before_" in str(row.get("question_id")) else "after"
        if phase == "before" and row_phase != "before":
            continue
        for result in row.get("results") or []:
            content = str(result.get("content") or "").strip()
            key = normalized_text(content)
            if not content or not key:
                continue
            score = float(result.get("score") or 0.0)
            existing = candidates.get(key)
            if existing is None or score > existing.score:
                candidates[key] = MemoryObject(
                    content=content,
                    score=score,
                    metadata=dict(result.get("metadata") or {}),
                    first_seen=seen_index if existing is None else existing.first_seen,
                )
            seen_index += 1
    return sorted(candidates.values(), key=lambda item: item.first_seen)


def coverage_signals(spec: QuestionSpec) -> tuple[list[str], set[str]]:
    phrases = list(spec.support_values)
    phrases.extend(flatten_strings(spec.entity_values))
    # Gold answers are useful for ER/Agg/Tr but include evaluator language such
    # as "uncertain" for Abs and "explicitly removed" for Del.  Those state
    # words still help locate memories that preserved deletion/uncertainty.
    phrases.append(spec.gold_answer)
    phrases.extend(entity.replace("_", " ") for entity in spec.target_entities)
    if spec.cascade_source:
        phrases.append(spec.cascade_source.replace("_", " "))
    unique_phrases = list(dict.fromkeys(normalized_text(value) for value in phrases))
    unique_phrases = [value for value in unique_phrases if value]
    query_tokens = tokens(spec.question)
    return unique_phrases, query_tokens


def rank_memory_objects(
    pool: list[MemoryObject], spec: QuestionSpec
) -> tuple[list[tuple[MemoryObject, float, list[str]]], dict[str, bool]]:
    phrases, query_tokens = coverage_signals(spec)
    phrase_present = {phrase: False for phrase in phrases}
    ranked = []
    for memory in pool:
        text = normalized_text(memory.content)
        memory_tokens = set(text.split())
        hits = [phrase for phrase in phrases if phrase in text]
        for phrase in hits:
            phrase_present[phrase] = True
        overlap = len(memory_tokens & query_tokens) / max(1, len(query_tokens))
        # Exact gold-field phrases dominate native similarity.  Query overlap
        # breaks ties and helps with paraphrased entity labels.
        exact_weight = sum(4.0 + min(4.0, len(phrase.split()) / 3.0) for phrase in hits)
        score = exact_weight + 2.0 * overlap + 0.05 * memory.score
        if hits or overlap > 0:
            ranked.append((memory, score, hits))
    ranked.sort(key=lambda item: (item[1], -item[0].first_seen), reverse=True)
    return ranked, phrase_present


def pack_memory_context(
    ranked: list[tuple[MemoryObject, float, list[str]]],
    encoding: Any,
    budget: int | None,
) -> tuple[list[dict[str, Any]], str]:
    selected = []
    segments = []
    used = 0
    for memory, oracle_score, hits in ranked:
        segment = f"<MEMORY>\n{memory.content}\n</MEMORY>\n\n"
        cost = len(encoding.encode(segment))
        if budget is not None and not selected and cost > budget:
            segment = encoding.decode(encoding.encode(segment)[:budget])
            cost = budget
        elif budget is not None and used + cost > budget:
            continue
        selected.append(
            {
                "content": memory.content,
                "score": oracle_score,
                "metadata": {
                    "oracle_exact_hits": hits,
                    "native_score": memory.score,
                    "native_metadata": memory.metadata,
                },
            }
        )
        segments.append(segment)
        used += cost
        if budget is not None and used >= budget:
            break
    return selected, "".join(segments)


def raw_session_text(session: dict[str, Any], index: int) -> str:
    messages = []
    for message in session.get("conversation") or []:
        role = str(message.get("role") or "").capitalize()
        messages.append(f"{role}: {str(message.get('content') or '').strip()}")
    return (
        f"<RAW_EVIDENCE session={index} time={session.get('timestamp', '')}>\n"
        + "\n".join(messages)
        + "\n</RAW_EVIDENCE>\n\n"
    )


def task_readout_policy(spec: QuestionSpec) -> str:
    """Return task semantics without exposing entity values or the gold answer."""
    target = ", ".join(spec.target_entities)
    lines = [
        "<MEME_READOUT_POLICY>",
        f"Question phase: {spec.phase}.",
        f"Task type: {spec.task_type}.",
        f"Target entities only: {target}.",
        "Treat the evidence as a chronological state log. Later applicable state "
        "updates override earlier values; do not combine superseded and current values.",
    ]
    if spec.phase == "before" and spec.task_type in {"Cas", "Abs"}:
        lines.append(
            "Before-phase rule: the upstream change has not happened yet. Any "
            "conditional replacement is hypothetical and must not be activated; "
            "return the current pre-change target value."
        )
    elif spec.phase == "after" and spec.task_type == "Cas":
        lines.append(
            "Cascade rule: find the stated dependency between the target and its "
            "upstream entity, then apply it when the later upstream change occurs. "
            "Return only the resulting current target value."
        )
    elif spec.phase == "after" and spec.task_type == "Abs":
        lines.append(
            "Absence rule: the upstream entity changed but no replacement target "
            "value was declared. State that the current target is uncertain; a "
            "previous value may be identified only as previous, never as current."
        )
    elif spec.phase == "after" and spec.task_type == "Del":
        lines.append(
            "Deletion rule: a request to remove the target makes it unavailable as "
            "current memory. Say that it was explicitly removed and do not reveal "
            "the deleted value."
        )
    elif spec.task_type == "Tr":
        lines.append(
            "Tracking rule: list every target value in chronological order, earliest "
            "to latest, without adding unrelated items."
        )
    elif spec.task_type == "Agg":
        lines.append(
            "Aggregation rule: return one value for every listed target entity and "
            "exclude facts belonging to other entities."
        )
    elif spec.task_type == "ER":
        lines.append(
            "Exact-recall rule: reproduce the target value verbatim, preserving its "
            "wording and order."
        )
    else:
        lines.append("Return the latest phase-visible value for the target entity.")
    lines.extend(
        [
            "Use only the evidence below. Do not infer a value that the evidence and "
            "the stated transition rule do not support.",
            "</MEME_READOUT_POLICY>",
            "",
        ]
    )
    return "\n".join(lines)


def add_readout_policy(context: str, spec: QuestionSpec, enabled: bool) -> str:
    if not enabled:
        return context
    return context + "\n" + task_readout_policy(spec)


def evidence_budget(
    encoding: Any, total_budget: int | None, spec: QuestionSpec, enabled: bool
) -> int | None:
    if total_budget is None:
        return None
    if not enabled:
        return total_budget
    policy_tokens = len(encoding.encode(task_readout_policy(spec)))
    return max(1, total_budget - policy_tokens)


def pack_raw_context(
    episode: dict[str, Any], spec: QuestionSpec, encoding: Any, budget: int | None
) -> tuple[list[dict[str, Any]], str, list[int]]:
    visible = [
        int(index)
        for index in episode.get("evidence_session_indices") or []
        if int(index) <= spec.cutoff
    ]
    phrases, _query_tokens = coverage_signals(spec)
    scored_sessions = []
    for index in visible:
        session_text = raw_session_text(episode["sessions"][index], index)
        normalized = normalized_text(session_text)
        hits = [phrase for phrase in phrases if phrase in normalized]
        if hits:
            scored_sessions.append((index, hits))
    # MEME's entity values normally identify the exact evidence sessions.  The
    # fallback keeps the official evidence set if a generator paraphrased every
    # supplied value beyond deterministic matching.
    selected_sessions = scored_sessions or [(index, []) for index in visible]
    selected_sessions.sort(key=lambda item: item[0])
    rows = []
    segments = []
    used = 0
    for index, hits in selected_sessions:
        session = episode["sessions"][index]
        segment = raw_session_text(session, index)
        cost = len(encoding.encode(segment))
        if budget is not None and not rows and cost > budget:
            segment = encoding.decode(encoding.encode(segment)[:budget])
            cost = budget
        elif budget is not None and used + cost > budget:
            continue
        content = "\n".join(
            f"{str(item.get('role') or '').capitalize()}: {str(item.get('content') or '').strip()}"
            for item in session.get("conversation") or []
        )
        rows.append(
            {
                "content": content,
                "score": 1.0,
                "metadata": {
                    "source": "official_meme_evidence_session",
                    "session_index": index,
                    "timestamp": session.get("timestamp"),
                    "oracle_exact_hits": hits,
                },
            }
        )
        segments.append(segment)
        used += cost
    return rows, "".join(segments), visible


def base_row(spec: QuestionSpec, results: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": spec.question_id,
        "query": spec.question,
        "conversation_id": spec.conversation_id,
        "results": results,
        "retrieval_metadata": metadata,
    }


def create_output(path: Path, rows: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {path}")
    path.mkdir(parents=True)
    dump_json(path / "search_results.json", rows)
    dump_json(path / "meme_continuity_oracle_audit.json", audit)


def build(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    encoding = tiktoken.get_encoding("o200k_base")

    raw_rows = []
    raw_audit = []
    for spec in specs:
        results, context, visible = pack_raw_context(
            episodes[spec.episode_index],
            spec,
            encoding,
            evidence_budget(
                encoding, args.token_budget, spec, args.task_aware_readout
            ),
        )
        context = add_readout_policy(context, spec, args.task_aware_readout)
        metadata = {
            "system": "meme_raw_gold_evidence_continuity_oracle",
            "formatted_context": context,
            "top_k": len(results),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_source": "official MEME evidence_session_indices",
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
        }
        raw_rows.append(base_row(spec, results, metadata))
        raw_audit.append(
            {
                "question_id": spec.question_id,
                "phase": spec.phase,
                "task_type": spec.task_type,
                "visible_official_evidence_sessions": visible,
                "selected_sessions": [row["metadata"]["session_index"] for row in results],
                "context_tokens": len(encoding.encode(context)),
            }
        )
    raw_dir = RESULTS_DIR / f"meme-raw-{args.tag}"
    if not args.skip_raw:
        create_output(raw_dir, raw_rows, raw_audit)
        print(f"Built raw oracle: {raw_dir} ({len(raw_rows)} questions)")

    for system, run_name in args.system_run:
        source_dir = RESULTS_DIR / run_name
        searches = load_json(source_dir / "search_results.json")
        rows = []
        audit = []
        for spec in specs:
            pool = memory_pool(searches, spec.conversation_id, spec.phase)
            ranked, phrase_present = rank_memory_objects(pool, spec)
            selected, context = pack_memory_context(
                ranked,
                encoding,
                evidence_budget(
                    encoding, args.token_budget, spec, args.task_aware_readout
                ),
            )
            context = add_readout_policy(context, spec, args.task_aware_readout)
            support_phrases = [normalized_text(value) for value in spec.support_values]
            support_phrases = [value for value in support_phrases if value]
            support_covered = {
                phrase: any(phrase in normalized_text(item.content) for item in pool)
                for phrase in support_phrases
            }
            metadata = {
                "system": f"{system}_meme_gold_fields_observed_pool_oracle",
                "formatted_context": context,
                "top_k": len(selected),
                "context_token_budget": args.token_budget,
                "context_tokens_after_cap": len(encoding.encode(context)),
                "oracle_target_entities": list(spec.target_entities),
                "oracle_entity_values": spec.entity_values,
                "oracle_gold_answer_used_for_selection": True,
                "oracle_complete_store_candidate_access": False,
                "oracle_candidate_scope": "deduplicated objects observed across native searches",
                "oracle_phase_safe": True,
                "question_phase": spec.phase,
                "question_cutoff_session": spec.cutoff,
                "task_aware_readout_policy": args.task_aware_readout,
            }
            rows.append(base_row(spec, selected, metadata))
            audit.append(
                {
                    "question_id": spec.question_id,
                    "phase": spec.phase,
                    "task_type": spec.task_type,
                    "target_entities": list(spec.target_entities),
                    "entity_values": spec.entity_values,
                    "gold_answer": spec.gold_answer,
                    "cascade_source": spec.cascade_source,
                    "candidate_pool_size": len(pool),
                    "ranked_candidate_count": len(ranked),
                    "selected_count": len(selected),
                    "context_tokens": len(encoding.encode(context)),
                    "selection_phrase_coverage": phrase_present,
                    "support_value_coverage": support_covered,
                    "all_support_values_observed": all(support_covered.values()),
                    "any_support_value_observed": any(support_covered.values()),
                }
            )
        output_dir = RESULTS_DIR / f"meme-{system}-{args.tag}"
        create_output(output_dir, rows, audit)
        covered = sum(row["all_support_values_observed"] for row in audit)
        print(
            f"Built {system} observed-pool oracle: {output_dir} "
            f"({len(rows)} questions; all-support coverage {covered}/{len(audit)})"
        )


def parse_system_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--system-run must use SYSTEM=RUN_DIR_NAME")
    system, run_name = value.split("=", 1)
    if not system or not run_name:
        raise argparse.ArgumentTypeError("--system-run must use SYSTEM=RUN_DIR_NAME")
    return system, run_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--end-conv", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=5000)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--task-aware-readout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix evidence with task semantics that do not contain gold values.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Build only requested system memory oracles, leaving raw artifacts untouched.",
    )
    parser.add_argument(
        "--system-run",
        action="append",
        type=parse_system_run,
        default=None,
        help="Repeatable SYSTEM=existing-run-directory-name mapping.",
    )
    args = parser.parse_args()
    if args.system_run is None:
        args.system_run = list(DEFAULT_SYSTEM_RUNS.items())
    build(args)


if __name__ == "__main__":
    main()
