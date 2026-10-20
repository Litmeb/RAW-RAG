"""Deterministic, question-aligned MEME propagation diagnostics.

The diagnostic uses MEME's official evidence sessions and structured task
metadata.  Coverage is an exact normalized signal audit, not an LLM judgment:
every persisted object is scanned before the matching examples are capped in
the output artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
SCRIPTS_DIR = EVALUATION_DIR / "scripts"
for candidate in (EVERCORE_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_meme_complete_store_oracle_runs import (  # noqa: E402
    MemoryObject,
    load_evermemos_store,
    load_mem0_store,
    load_memos_store,
    phase_safe_pool,
    session_timestamp_maps,
)
from build_meme_continuity_oracle_runs import (  # noqa: E402
    QuestionSpec,
    flatten_strings,
    load_json,
    normalized_text,
    question_specs,
    task_for_question,
)
DEFAULT_DATA = EVALUATION_DIR / "data/meme/meme_filler32k.json"
DEFAULT_RUNS = {
    "mem0_local": EVALUATION_DIR / "results/meme-mem0_local-normal",
    "memos_local": EVALUATION_DIR / "results/meme-memos_local-fine",
    "evermemos": EVALUATION_DIR / "results/meme-evermemos-normal-3",
}
WORD_RE = re.compile(r"[a-z0-9]+")
MEME_TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")
ABSTRACT_STATE_MARKERS = {"none", "deleted", "null", "n a", "unknown"}


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def index_rows(value: Any) -> dict[str, dict[str, Any]]:
    """Index rows at any envelope depth (MEME eval groups rows by task)."""
    rows: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            if node.get("question_id"):
                rows.append(node)
            else:
                for child in node.values():
                    visit(child)

    visit(value)
    return {
        str(row["question_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("question_id")
    }


def evaluator_correct(row: dict[str, Any]) -> bool:
    """Use MEME's per-question semantic result, before trivial-pass filtering."""
    for key in ("u_pass", "is_correct", "correct"):
        if key in row:
            return bool(row[key])
    for key in ("u_score", "metric_value"):
        if row.get(key) is not None:
            return float(row[key]) > 0
    return False


def official_credit(row: dict[str, Any], phase: str) -> float | None:
    """Return official MEME credit; before questions are diagnostic-only."""
    if phase != "after":
        return None
    if row.get("filtered_u_pass") is not None:
        return float(row["filtered_u_pass"])
    return 1.0 if evaluator_correct(row) else 0.0


def unique_values(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for text in flatten_strings(value):
            key = normalized_text(text)
            if key and key not in seen:
                seen.add(key)
                result.append(text)
    return result


def question_signal_spec(
    episode: dict[str, Any], spec: QuestionSpec, question: dict[str, Any]
) -> dict[str, Any]:
    """Derive phase-visible value and dependency signals from official metadata."""
    task = task_for_question(episode, question)
    target_entities = list(task.get("target_entities") or question.get("entity") or [])
    target_states = {
        entity: dict((episode.get("entities") or {}).get(entity) or {})
        for entity in target_entities
    }
    cascade_source = str(task.get("cascade_source") or "")
    cascade_state = dict((episode.get("entities") or {}).get(cascade_source) or {})
    edges = [
        dict(edge)
        for edge in episode.get("dependency_edges_used") or []
        if str(edge.get("target") or "") in target_entities
    ]

    required: list[dict[str, str]] = []

    def add(kind: str, entity: str, value: Any) -> None:
        for text in unique_values([value]):
            key = normalized_text(text)
            if (
                key
                and key not in ABSTRACT_STATE_MARKERS
                and not any(item["normalized"] == key for item in required)
            ):
                required.append(
                    {"kind": kind, "entity": entity, "value": text, "normalized": key}
                )

    task_type = spec.task_type
    if task_type == "Tr":
        for entity, state in target_states.items():
            add("target_history", entity, state.get("history") or question.get("entity_values"))
    else:
        for entity, value in (question.get("entity_values") or {}).items():
            add("question_value", str(entity), value)

    if spec.phase == "after" and task_type in {"Del", "Cas", "Abs"}:
        for entity, state in target_states.items():
            add("target_before", entity, state.get("before"))
            # ``deleted`` is an abstract dataset state, not a phrase that must
            # occur in the source or memory text.  The prior value remains a
            # deterministic Del signal; deletion semantics stay in metadata.
            if task_type == "Cas":
                add("target_after", entity, state.get("after"))
    if spec.phase == "after" and task_type in {"Cas", "Abs"}:
        add("cascade_source_before", cascade_source, cascade_state.get("before"))
        add("cascade_source_after", cascade_source, cascade_state.get("after"))

    semantic_expectations = {
        "requires_relation_semantics": task_type in {"Cas", "Abs"},
        "requires_deletion_semantics": task_type == "Del",
        "abstract_target_states": {
            entity: {
                key: value
                for key, value in state.items()
                if key in {"before", "after"}
                and normalized_text(str(value or "")) in ABSTRACT_STATE_MARKERS
            }
            for entity, state in target_states.items()
            if any(
                normalized_text(str(state.get(key) or "")) in ABSTRACT_STATE_MARKERS
                for key in ("before", "after")
            )
        },
    }

    return {
        "task": task,
        "target_entities": target_entities,
        "target_states": target_states,
        "cascade_source": cascade_source,
        "cascade_source_state": cascade_state,
        "dependency_edges": edges,
        "root": episode.get("root"),
        "root_change": episode.get("root_change"),
        "required_value_signals": required,
        "semantic_expectations": semantic_expectations,
    }


def session_text(session: dict[str, Any]) -> str:
    messages = [
        f"{str(item.get('role') or '').capitalize()}: {str(item.get('content') or '')}"
        for item in session.get("conversation") or []
    ]
    facts = [str(item) for item in session.get("gold_facts") or []]
    return "\n".join([*messages, *(f"GOLD_FACT: {fact}" for fact in facts)])


def official_source_layer(
    episode: dict[str, Any], spec: QuestionSpec
) -> tuple[list[dict[str, Any]], str]:
    indices = [
        int(index)
        for index in episode.get("evidence_session_indices") or []
        if int(index) <= spec.cutoff
    ]
    rows = []
    for index in indices:
        session = episode["sessions"][index]
        rows.append(
            {
                "session_index": index,
                "session_id": session.get("session_id"),
                "timestamp": session.get("timestamp"),
                "evidence_type": session.get("evidence_type"),
                "gold_facts": session.get("gold_facts") or [],
                "conversation": session.get("conversation") or [],
                "text": session_text(session),
            }
        )
    return rows, "\n\n".join(row["text"] for row in rows)


def retrieval_texts(search: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(row.get("content") or ""), row)
        for row in search.get("results") or []
        if isinstance(row, dict) and str(row.get("content") or "").strip()
    ]


def scan_signals(
    signals: list[dict[str, str]],
    texts: list[tuple[str, dict[str, Any]]],
    *,
    max_examples: int,
) -> dict[str, Any]:
    matches = []
    covered = 0
    for signal in signals:
        hits = [
            metadata
            for text, metadata in texts
            if signal["normalized"] in normalized_text(text)
        ]
        if hits:
            covered += 1
        matches.append(
            {
                **signal,
                "covered": bool(hits),
                "match_count": len(hits),
                "match_examples": hits[:max_examples],
                "examples_capped": len(hits) > max_examples,
            }
        )
    total = len(signals)
    status = "all" if covered == total else "none" if covered == 0 else "partial"
    return {
        "status": status,
        "covered": covered,
        "total": total,
        "coverage": covered / total if total else 1.0,
        "signals": matches,
    }


def memory_texts(objects: list[MemoryObject]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            memory.content,
            {
                "content": memory.content,
                "memory_id": memory.metadata.get("memory_id"),
                "memory_kind": memory.metadata.get("memory_kind"),
                "source_session_indices": memory.metadata.get("source_session_indices"),
            },
        )
        for memory in objects
    ]


def infer_first_signal_gap(
    source: dict[str, Any],
    store: dict[str, Any],
    retrieval: dict[str, Any],
    context: dict[str, Any],
    correct: bool,
) -> str:
    """Classify the first exact-value gap without claiming causal attribution."""
    if source["status"] != "all":
        return "official_source_signal_gap"
    for layer, stage in (
        (store, "persisted_store_signal_gap"),
        (retrieval, "native_retrieval_signal_gap"),
        (context, "reader_context_signal_gap"),
    ):
        if layer["status"] != "all":
            return stage
    return "no_signal_gap" if correct else "answer_mismatch_with_all_value_signals"


# Backward-compatible import for older notebooks/tests.  New artifacts use the
# explicitly non-causal name above.
infer_error_stage = infer_first_signal_gap


def answer_signal_coverage(answer: str, gold: str) -> dict[str, Any]:
    signals = [
        {"kind": "gold_answer", "entity": "", "value": value, "normalized": normalized_text(value)}
        for value in unique_values([gold])
    ]
    return scan_signals(signals, [(answer, {"answer": answer})], max_examples=1)


StoreLoader = Callable[
    [argparse.Namespace, list[dict[str, Any]], dict[str, list[tuple[str, int]]]],
    tuple[dict[str, list[MemoryObject]], dict[str, Any]],
]


def load_mem0_backend(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    del episodes
    return load_mem0_store(args.run_dir.resolve(), timestamp_maps)


def load_memos_backend(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    store, audit, exported = load_memos_store(
        args.memos_api_url, args.memos_prefix, conversation_ids, timestamp_maps
    )
    if args.memory_export_path:
        dump_json(args.memory_export_path.resolve(), exported)
    del episodes
    return store, audit


def load_evermemos_backend(
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    del episodes
    return load_evermemos_store(args.run_dir.resolve(), conversation_ids, timestamp_maps)


LOADERS: dict[str, StoreLoader] = {
    "mem0_local": load_mem0_backend,
    "memos_local": load_memos_backend,
    "evermemos": load_evermemos_backend,
}


def question_lookup(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        for phase in ("before", "after"):
            for index, question in enumerate(episode[f"{phase}_questions"].get("questions") or []):
                result[f"meme_{episode_id}_{phase}_qa{index}"] = question
    return result


def build_diagnostics(
    *,
    system: str,
    args: argparse.Namespace,
    episodes: list[dict[str, Any]],
    specs: list[QuestionSpec],
    store: dict[str, list[MemoryObject]],
    searches: dict[str, dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    evaluations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    questions = question_lookup(episodes)
    rows = []
    for spec in specs:
        search = searches[spec.question_id]
        answer = answers[spec.question_id]
        evaluation = evaluations[spec.question_id]
        question = questions[spec.question_id]
        episode = episodes[spec.episode_index]
        signal_spec = question_signal_spec(episode, spec, question)
        signals = signal_spec["required_value_signals"]
        source_rows, source_text = official_source_layer(episode, spec)
        phase_store = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
        native = retrieval_texts(search)
        metadata = search.get("retrieval_metadata") or {}
        context_source = "search.retrieval_metadata.formatted_context"
        reader_context = str(metadata.get("formatted_context") or "")
        if not reader_context:
            context_source = "answer.formatted_context"
            reader_context = str(answer.get("formatted_context") or "")
        if not reader_context:
            context_source = "reconstructed_from_native_results"
            reader_context = "\n\n".join(text for text, _row in native)

        source_cov = scan_signals(
            signals,
            [(source_text, {"official_session_indices": [row["session_index"] for row in source_rows]})],
            max_examples=args.max_match_examples,
        )
        store_cov = scan_signals(
            signals, memory_texts(phase_store), max_examples=args.max_match_examples
        )
        retrieval_cov = scan_signals(signals, native, max_examples=args.max_match_examples)
        context_cov = scan_signals(
            signals,
            [(reader_context, {"context_chars": len(reader_context)})],
            max_examples=args.max_match_examples,
        )
        generated = str(answer.get("answer") or evaluation.get("generated_answer") or "")
        correct = evaluator_correct(evaluation)
        credit = official_credit(evaluation, spec.phase)
        gap = infer_first_signal_gap(
            source_cov, store_cov, retrieval_cov, context_cov, correct
        )
        rows.append(
            {
                "system": system,
                "question_id": spec.question_id,
                "conversation_id": spec.conversation_id,
                "episode_index": spec.episode_index,
                "phase": spec.phase,
                "cutoff_session": spec.cutoff,
                "task_type": spec.task_type,
                "question": spec.question,
                "gold_answer": spec.gold_answer,
                "generated_answer": generated,
                "final_correct": correct,
                "official_meme_credit": credit,
                "meme_pass_type": evaluation.get("pass_type"),
                "meme_metadata": signal_spec,
                "official_source": {
                    "evidence_session_indices": [row["session_index"] for row in source_rows],
                    "sessions": source_rows,
                    "coverage": source_cov,
                },
                "persisted_store_snapshot": {
                    "phase_visible_object_count": len(phase_store),
                    "coverage": store_cov,
                },
                "native_retrieval": {
                    "result_count": len(native),
                    "results": [row for _text, row in native],
                    "coverage": retrieval_cov,
                },
                "reader_context": {
                    "chars": len(reader_context),
                    "source": context_source,
                    "text": reader_context,
                    "coverage": context_cov,
                },
                "answer": {
                    "correct": correct,
                    "gold_exact_coverage": answer_signal_coverage(generated, spec.gold_answer),
                },
                "first_signal_gap": gap,
                "requires_semantic_inspection": bool(
                    signal_spec["semantic_expectations"]["requires_relation_semantics"]
                    or signal_spec["semantic_expectations"]["requires_deletion_semantics"]
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    layers = (
        "official_source",
        "persisted_store_snapshot",
        "native_retrieval",
        "reader_context",
    )
    task_scores: dict[str, Any] = {}
    for task in MEME_TASKS:
        task_rows = [
            row
            for row in rows
            if row["task_type"] == task and row["phase"] == "after"
        ]
        credit = sum(float(row["official_meme_credit"] or 0.0) for row in task_rows)
        task_scores[task] = {
            "pass": credit,
            "total": len(task_rows),
            "accuracy": credit / len(task_rows) if task_rows else 0.0,
        }
    available_tasks = [task for task in MEME_TASKS if task_scores[task]["total"]]
    available_task_macro = (
        sum(task_scores[task]["accuracy"] for task in available_tasks)
        / len(available_tasks)
        if available_tasks
        else 0.0
    )
    official_overall = available_task_macro if len(available_tasks) == len(MEME_TASKS) else None
    return {
        "questions": len(rows),
        "raw_correct": sum(row["final_correct"] for row in rows),
        "raw_question_accuracy": (
            sum(row["final_correct"] for row in rows) / len(rows) if rows else 0.0
        ),
        "official_meme": {
            "overall_accuracy": official_overall,
            "available_task_macro_accuracy": available_task_macro,
            "is_complete_six_task_evaluation": len(available_tasks) == len(MEME_TASKS),
            "task_scores": task_scores,
            "scoring": (
                "after-phase only; Cas/Abs/Del use filtered_u_pass; overall is "
                "the macro average of the six task accuracies"
            ),
        },
        "first_signal_gap_counts": dict(
            Counter(row["first_signal_gap"] for row in rows)
        ),
        "coverage_status_counts": {
            layer: dict(Counter(row[layer]["coverage"]["status"] for row in rows))
            for layer in layers
        },
        "by_task": {
            task: {
                "questions": len(task_rows),
                "correct": sum(row["final_correct"] for row in task_rows),
                "first_signal_gap_counts": dict(
                    Counter(row["first_signal_gap"] for row in task_rows)
                ),
            }
            for task in sorted({row["task_type"] for row in rows})
            if (task_rows := [row for row in rows if row["task_type"] == task])
        },
    }


def render_markdown(system: str, run_dir: Path, summary: dict[str, Any]) -> str:
    official_accuracy = summary["official_meme"]["overall_accuracy"]
    official_display = f"{official_accuracy:.4f}" if official_accuracy is not None else "N/A (subset)"
    lines = [
        f"# {system} MEME deterministic propagation audit",
        "",
        f"Run: `{run_dir}`",
        f"Questions: {summary['questions']}",
        f"Raw all-question accuracy: {summary['raw_question_accuracy']:.4f}",
        f"Official MEME six-task macro accuracy: {official_display}",
        "",
        "Coverage is normalized exact-value signal coverage, paired with official dependency metadata. It is a reproducible diagnostic proxy, not a semantic sufficiency judgment; Cas/Abs/Del relation semantics require inspection of the emitted evidence.",
        "",
        "The first-signal-gap label is non-causal. Mutable final stores are not historical cutoff snapshots, and Cas/Abs/Del state semantics require inspection of the emitted metadata and evidence.",
        "",
        "## Official MEME task scores",
        "",
        "| task | pass | total | accuracy |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {task} | {score['pass']:.0f} | {score['total']} | {score['accuracy']:.4f} |"
        for task, score in summary["official_meme"]["task_scores"].items()
    )
    lines.extend([
        "",
        "## First exact-value signal gap (non-causal)",
        "",
        "| stage | count |",
        "|---|---:|",
    ])
    lines.extend(
        f"| {stage} | {count} |"
        for stage, count in sorted(summary["first_signal_gap_counts"].items())
    )
    lines.extend(["", "## Coverage by layer", ""])
    lines.extend(
        f"- `{layer}`: `{counts}`"
        for layer, counts in summary["coverage_status_counts"].items()
    )
    return "\n".join(lines) + "\n"


def parse_args(system: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic MEME source-to-answer propagation diagnostic"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS[system])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--end-conv", type=int, default=100)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--max-match-examples", type=int, default=5)
    parser.add_argument("--output-dir", type=Path)
    if system == "memos_local":
        parser.add_argument("--memos-api-url", default="http://localhost:8000")
        parser.add_argument("--memos-prefix", default="fine")
        parser.add_argument("--memory-export-path", type=Path)
    else:
        parser.set_defaults(memory_export_path=None)
    return parser.parse_args()


def run(system: str) -> int:
    args = parse_args(system)
    run_dir = args.run_dir.resolve()
    data_path = args.data.resolve()
    if run_dir != DEFAULT_RUNS[system].resolve() and system == "evermemos":
        raise ValueError(
            "EverMemOS propagation audit is pinned to "
            f"{DEFAULT_RUNS[system].resolve()}, got {run_dir}"
        )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    episodes = load_json(data_path)
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    if args.question_id:
        selected = set(args.question_id)
        specs = [spec for spec in specs if spec.question_id in selected]
        missing = selected - {spec.question_id for spec in specs}
        if missing:
            raise ValueError(f"Unknown/out-of-range question IDs: {sorted(missing)}")

    searches = index_rows(load_json(run_dir / "search_results.json"))
    answers = index_rows(load_json(run_dir / "answer_results.json"))
    evaluations = index_rows(load_json(run_dir / "eval_results.json"))
    if system == "memos_local":
        observed_prefixes = {
            str(user_id).split("__", 1)[0]
            for row in searches.values()
            for user_id in (row.get("retrieval_metadata") or {}).get("user_ids") or []
            if "__" in str(user_id)
        }
        if observed_prefixes and observed_prefixes != {args.memos_prefix}:
            raise ValueError(
                "Memos store namespace does not match the native run: "
                f"run prefixes={sorted(observed_prefixes)}, requested={args.memos_prefix!r}"
            )
    timestamp_maps = session_timestamp_maps(episodes)
    store, store_audit = LOADERS[system](args, episodes, timestamp_maps)
    expected = {spec.question_id for spec in specs}
    for name, index in (("search", searches), ("answer", answers), ("evaluation", evaluations)):
        missing = expected - set(index)
        if missing:
            raise ValueError(f"{name} artifact is missing {len(missing)} selected questions")

    rows = build_diagnostics(
        system=system,
        args=args,
        episodes=episodes,
        specs=specs,
        store=store,
        searches=searches,
        answers=answers,
        evaluations=evaluations,
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / f"meme_propagation_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    summary = summarize(rows)
    dump_json(output_dir / "question_diagnostics.json", rows)
    with (output_dir / "question_diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_json(
        output_dir / "question_manifest.json",
        [
            {
                "question_id": row["question_id"],
                "conversation_id": row["conversation_id"],
                "phase": row["phase"],
                "task_type": row["task_type"],
            }
            for row in rows
        ],
    )
    payload = {
        "system": system,
        "run_dir": str(run_dir),
        "data": str(data_path),
        "output_dir": str(output_dir),
        "method": "deterministic normalized exact-value scan with official dependency metadata",
        "limitations": (
            "Exact-value coverage does not by itself prove that Cas/Abs/Del dependency, "
            "invalidation, or deletion semantics were preserved. Full official source, "
            "matching persisted-snapshot objects, native results, and reader context are "
            "emitted for inspection. Persisted mutable stores are not treated as historical "
            "cutoff reconstructions."
        ),
        "store_audit": store_audit,
        **summary,
    }
    dump_json(output_dir / "summary.json", payload)
    (output_dir / "summary.md").write_text(
        render_markdown(system, run_dir, payload), encoding="utf-8"
    )
    print(f"Diagnostics written to: {output_dir}")
    return 0
