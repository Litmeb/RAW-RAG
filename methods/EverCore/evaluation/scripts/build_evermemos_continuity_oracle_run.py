"""Build a zero-API, no-reranker EverMemOS continuity recovery oracle.

The reader sees only EverMemOS episode and atomic-fact text, never raw
``original_data``.  For each EventQA question, the oracle prioritizes memories
from MemCells overlapping the raw evidence window, fills the remaining
5000-token budget from the complete atomic-fact store using a gold-informed
dense query, and restores selected memories to source-chunk order.

Existing 1024-dimensional EverMemOS ``text-embedding-3-small`` fact vectors are
reused.  The existing 1536-dimensional gold-query cache from the Mem0 experiment
is shortened to its first 1024 dimensions and renormalized, matching the
embedding-v3 dimensions-shortening convention.  No API or reranker is called.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken


EVALUATION_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_SOURCE = RESULTS_DIR / (
    "memoryagentbench-evermemos-Accurate_Retrieval-eventqa_full-normal-new-converter"
)
DEFAULT_MEM0_NATIVE = RESULTS_DIR / (
    "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-normal"
)
DEFAULT_MEM0_ORACLE = RESULTS_DIR / (
    "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-memory-oracle-"
    "continuity-rag-anchor-memory-oracle-5000-v3-20260716"
)
DEFAULT_GOLD_EMBEDDINGS = RESULTS_DIR / (
    "mem0_eventqa_gold_query_embeddings_continuity-prev-anchor-embedding-"
    "oracle-5000-v2-20260716.json"
)
DEFAULT_OUTPUT = RESULTS_DIR / (
    "memoryagentbench-evermemos-Accurate_Retrieval-eventqa_full-memory-oracle-"
    "continuity-gold-hybrid-5000-v2-20260717"
)
DEFAULT_DATA = EVALUATION_DIR / (
    "data/memoryagentbench/Accurate_Retrieval/eventqa_full"
)


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    event_id: str
    kind: str
    source_chunks: tuple[int, ...]
    within_event_index: int
    timestamp: str
    subject: str
    text: str
    vector: np.ndarray | None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def conversation_index(value: Any) -> int:
    match = re.search(r"(\d+)(?:\D*)$", str(value or ""))
    if not match:
        raise ValueError(f"Cannot parse conversation index from {value!r}")
    return int(match.group(1))


def parse_locomo_timestamp(value: str) -> datetime:
    normalized = " ".join(str(value).strip().split())
    normalized = re.sub(r"\b(am|pm)\b", lambda match: match.group(1).upper(), normalized)
    return datetime.strptime(normalized, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=timezone.utc
    )


def normalize_timestamp(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        parsed = (
            parse_locomo_timestamp(text)
            if " on " in text and "," in text
            else datetime.fromisoformat(text.replace("Z", "+00:00"))
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.replace("session_", "")),
    )


def timestamp_maps(data_dir: Path) -> dict[tuple[int, str], int]:
    converted = load_json(data_dir / "memoryagentbench_locomo_style.json")
    result = {}
    for conv, sample in enumerate(converted):
        conversation = sample["conversation"]
        for key in session_keys(conversation):
            source_index = int(key.replace("session_", ""))
            timestamp = normalize_timestamp(conversation[f"{key}_date_time"])
            result[(conv, timestamp)] = source_index
    return result


def original_source_chunks(
    doc: dict[str, Any], conv: int, by_timestamp: dict[tuple[int, str], int]
) -> tuple[int, ...]:
    chunks = set()
    for item in doc.get("original_data") or []:
        message = item.get("message", item) if isinstance(item, dict) else {}
        if not isinstance(message, dict) or not message.get("timestamp"):
            continue
        timestamp = normalize_timestamp(message["timestamp"])
        source_index = by_timestamp.get((conv, timestamp))
        if source_index is not None:
            chunks.add(source_index)
    return tuple(sorted(chunks))


def normalize_vector(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def load_store(
    source_dir: Path,
    conv: int,
    by_timestamp: dict[tuple[int, str], int],
) -> tuple[list[MemoryItem], list[MemoryItem], np.ndarray]:
    path = source_dir / "vectors" / f"embedding_index_conv_{conv}.pkl"
    with path.open("rb") as handle:
        rows = pickle.load(handle)  # noqa: S301 - trusted local evaluation artifact

    episodes = []
    facts = []
    for row in rows:
        doc = row["doc"]
        event_id = str(doc.get("event_id") or "")
        source_chunks = original_source_chunks(doc, conv, by_timestamp)
        timestamp = str(doc.get("timestamp") or "")
        subject = str(doc.get("subject") or "")
        episode = str(doc.get("episode") or "").strip()
        if episode:
            episodes.append(
                MemoryItem(
                    item_id=f"{event_id}:episode",
                    event_id=event_id,
                    kind="episode",
                    source_chunks=source_chunks,
                    within_event_index=-1,
                    timestamp=timestamp,
                    subject=subject,
                    text=episode,
                    vector=None,
                )
            )

        atomic = doc.get("atomic_fact") or {}
        fact_texts = atomic.get("atomic_fact") or [] if isinstance(atomic, dict) else []
        fact_vectors = row.get("embeddings", {}).get("atomic_facts") or []
        if len(fact_texts) != len(fact_vectors):
            raise ValueError(
                f"Atomic fact/vector mismatch in conversation {conv}, event {event_id}: "
                f"{len(fact_texts)} vs {len(fact_vectors)}"
            )
        for fact_index, (text, vector) in enumerate(zip(fact_texts, fact_vectors, strict=True)):
            text = str(text).strip()
            if not text:
                continue
            facts.append(
                MemoryItem(
                    item_id=f"{event_id}:fact:{fact_index}",
                    event_id=event_id,
                    kind="atomic_fact",
                    source_chunks=source_chunks,
                    within_event_index=fact_index,
                    timestamp=timestamp,
                    subject=subject,
                    text=text,
                    vector=normalize_vector(vector),
                )
            )
    matrix = np.vstack([item.vector for item in facts])
    return episodes, facts, matrix


def source_key(item: MemoryItem) -> tuple[int, int, str]:
    first_chunk = min(item.source_chunks) if item.source_chunks else 10**9
    return first_chunk, item.within_event_index, item.item_id


def lexical_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))
        if len(token) > 2
    }


def lexical_relevance(
    item_tokens: set[str], gold_tokens: set[str], previous_tokens: set[str]
) -> float:
    gold_recall = len(item_tokens & gold_tokens) / max(1, len(gold_tokens))
    previous_recall = len(item_tokens & previous_tokens) / max(1, len(previous_tokens))
    return 4.0 * gold_recall + 1.5 * previous_recall


def item_segment(item: MemoryItem, score: float | None, origin: str) -> str:
    chunks = ",".join(str(value) for value in item.source_chunks) or "unknown"
    score_text = "" if score is None else f" score={score:.6f}"
    return (
        f"<{item.kind.upper()} source_chunks={chunks} time={item.timestamp} "
        f"origin={origin}{score_text}>\n"
        f"Title: {item.subject}\n{item.text}\n</{item.kind.upper()}>\n\n"
    )


def pack_oracle(
    episodes: list[MemoryItem],
    facts: list[MemoryItem],
    fact_scores: np.ndarray,
    provenance_chunks: set[int],
    gold: str,
    previous: str,
    item_tokens: dict[str, set[str]],
    encoding: Any,
    budget: int,
) -> tuple[list[tuple[MemoryItem, float | None, str]], str, int]:
    provenance_event_ids = {
        item.event_id
        for item in episodes
        if provenance_chunks.intersection(item.source_chunks)
    }
    event_dense: dict[str, float] = {}
    for index, item in enumerate(facts):
        event_dense[item.event_id] = max(
            event_dense.get(item.event_id, -1.0), float(fact_scores[index])
        )
    gold_tokens = lexical_tokens(gold)
    previous_tokens = lexical_tokens(previous)

    ranked_items: list[tuple[float, MemoryItem, float | None, str]] = []
    for item in episodes:
        dense = event_dense.get(item.event_id, 0.0)
        provenance = item.event_id in provenance_event_ids
        relevance = lexical_relevance(
            item_tokens[item.item_id], gold_tokens, previous_tokens
        ) + dense
        if provenance:
            relevance += 0.75
        ranked_items.append(
            (
                relevance,
                item,
                dense,
                "raw_window_provenance" if provenance else "global_gold_hybrid",
            )
        )
    for index, item in enumerate(facts):
        dense = float(fact_scores[index])
        provenance = item.event_id in provenance_event_ids
        relevance = lexical_relevance(
            item_tokens[item.item_id], gold_tokens, previous_tokens
        ) + dense
        if provenance:
            relevance += 0.75
        ranked_items.append(
            (
                relevance,
                item,
                dense,
                "raw_window_provenance" if provenance else "global_gold_hybrid",
            )
        )
    ranked_items.sort(key=lambda value: (value[0], value[1].item_id), reverse=True)
    available_provenance = sum(
        origin == "raw_window_provenance" for _, _, _, origin in ranked_items
    )

    chosen = []
    used = 0
    for _relevance, item, score, origin in ranked_items:
        segment = item_segment(item, score, origin)
        cost = len(encoding.encode(segment))
        if not chosen and cost > budget:
            truncated = encoding.decode(encoding.encode(segment)[:budget])
            return [(item, score, origin)], truncated, available_provenance
        if used + cost <= budget:
            chosen.append((item, score, origin))
            used += cost
        if used >= budget:
            break

    chosen.sort(key=lambda value: source_key(value[0]))
    context = "".join(item_segment(item, score, origin) for item, score, origin in chosen)
    if len(encoding.encode(context)) > budget:
        raise AssertionError("Packed EverMemOS context exceeds token budget")
    return chosen, context, available_provenance


def build(args: argparse.Namespace) -> None:
    source_dir = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {output_dir}")

    source_searches = load_json(source_dir / "search_results.json")
    source_answers = load_json(source_dir / "answer_results.json")
    answer_by_question = {
        str(row["question_id"]): row for row in source_answers
    }
    mem0_searches = load_json(args.mem0_native.resolve() / "search_results.json")
    cached_gold = np.asarray(load_json(args.gold_embeddings.resolve()), dtype=np.float32)
    if cached_gold.shape[0] != len(mem0_searches):
        raise ValueError("Gold-query embedding cache does not align with Mem0 questions")
    gold_by_question = {
        str(row["question_id"]): cached_gold[index]
        for index, row in enumerate(mem0_searches)
    }
    mapping_rows = load_json(
        args.mem0_oracle.resolve() / "continuity_mapping_audit.json"
    )
    mapping_by_question = {str(row["question_id"]): row for row in mapping_rows}
    by_timestamp = timestamp_maps(args.data_dir.resolve())

    searches_by_conv: dict[int, list[dict[str, Any]]] = {}
    for row in source_searches:
        searches_by_conv.setdefault(conversation_index(row["conversation_id"]), []).append(row)

    encoding = tiktoken.get_encoding("o200k_base")
    output_by_question = {}
    audit_rows = []
    store_counts = {}
    embedding_dimension = None
    for conv in sorted(searches_by_conv):
        episodes, facts, fact_matrix = load_store(source_dir, conv, by_timestamp)
        item_tokens = {
            item.item_id: lexical_tokens(item.text) for item in episodes + facts
        }
        embedding_dimension = fact_matrix.shape[1]
        store_counts[str(conv)] = {"episodes": len(episodes), "atomic_facts": len(facts)}
        questions = searches_by_conv[conv]
        query_matrix = np.vstack(
            [gold_by_question[str(row["question_id"])][:embedding_dimension] for row in questions]
        )
        query_matrix = np.vstack([normalize_vector(vector) for vector in query_matrix])
        score_matrix = query_matrix @ fact_matrix.T

        for question_index, source in enumerate(questions):
            question_id = str(source["question_id"])
            provenance_chunks = {
                int(value)
                for value in mapping_by_question[question_id]["raw_source_sessions"]
            }
            answer_row = answer_by_question[question_id]
            gold = str(answer_row.get("golden_answer") or "")
            previous = str((answer_row.get("metadata") or {}).get("previous_event") or "")
            selected, context, available_provenance = pack_oracle(
                episodes,
                facts,
                score_matrix[question_index],
                provenance_chunks,
                gold,
                previous,
                item_tokens,
                encoding,
                args.token_budget,
            )
            context_tokens = len(encoding.encode(context))
            results = [
                {
                    "content": item.text,
                    "score": score,
                    "metadata": {
                        "memory_type": item.kind,
                        "item_id": item.item_id,
                        "event_id": item.event_id,
                        "subject": item.subject,
                        "timestamp": item.timestamp,
                        "source_chunk_indices": list(item.source_chunks),
                        "oracle_origin": origin,
                    },
                }
                for item, score, origin in selected
            ]
            selected_provenance = sum(
                origin == "raw_window_provenance" for _, _, origin in selected
            )
            output_by_question[question_id] = {
                "question_id": question_id,
                "query": source["query"],
                "conversation_id": source["conversation_id"],
                "results": results,
                "retrieval_metadata": {
                    "system": "evermemos_continuity_memory_dense_oracle",
                    "formatted_context": context,
                    "context_token_budget": args.token_budget,
                    "context_tokens_after_cap": context_tokens,
                    "oracle_uses_gold_answer": True,
                    "oracle_complete_store_candidate_access": True,
                    "oracle_reranker_used": False,
                    "new_reranker_calls": 0,
                    "new_embedding_api_calls": 0,
                    "stored_native_fact_embeddings_reused": True,
                    "gold_embedding_shortened_to_dimension": embedding_dimension,
                    "oracle_selection": (
                        "gold lexical coverage + previous-event coverage + dense score + "
                        "provenance bonus"
                    ),
                    "raw_original_data_exposed": False,
                    "raw_control_reused": True,
                    "provenance_source_chunks": sorted(provenance_chunks),
                    "available_provenance_items": available_provenance,
                    "selected_memory_count": len(selected),
                    "selected_provenance_memory_count": selected_provenance,
                },
            }
            audit_rows.append(
                {
                    "question_id": question_id,
                    "conversation_index": conv,
                    "provenance_source_chunks": sorted(provenance_chunks),
                    "available_conversation_episodes": len(episodes),
                    "available_conversation_atomic_facts": len(facts),
                    "available_provenance_items": available_provenance,
                    "selected_memories": len(selected),
                    "selected_provenance_memories": selected_provenance,
                    "context_tokens": context_tokens,
                }
            )
        print(
            f"Built EverMemOS oracle contexts for conversation {conv}: "
            f"{len(questions)} questions"
        )

    output_rows = [output_by_question[str(row["question_id"])] for row in source_searches]
    if len(output_rows) != 500:
        raise AssertionError(f"Expected 500 questions, got {len(output_rows)}")
    output_dir.mkdir(parents=True)
    dump_json(output_dir / "search_results.json", output_rows)
    dump_json(output_dir / "continuity_mapping_audit.json", audit_rows)
    summary = {
        "experiment": "EverMemOS continuity recovery, gold-informed memory-store oracle",
        "questions": len(output_rows),
        "source_run": str(source_dir),
        "stored_by_conversation": store_counts,
        "stored_episodes_total": sum(value["episodes"] for value in store_counts.values()),
        "stored_atomic_facts_total": sum(
            value["atomic_facts"] for value in store_counts.values()
        ),
        "stored_embedding_dimension": embedding_dimension,
        "gold_query_embedding_source_dimension": cached_gold.shape[1],
        "gold_query_embedding_shortening": "first stored_dimension components + L2 normalize",
        "output_dir": str(output_dir),
        "token_budget": args.token_budget,
        "new_reranker_calls": 0,
        "new_embedding_api_calls": 0,
        "raw_original_data_exposed": False,
        "raw_control_reused": True,
        "context_tokens": {
            "min": min(row["context_tokens"] for row in audit_rows),
            "mean": sum(row["context_tokens"] for row in audit_rows) / len(audit_rows),
            "max": max(row["context_tokens"] for row in audit_rows),
        },
        "selected_memories": {
            "min": min(row["selected_memories"] for row in audit_rows),
            "mean": sum(row["selected_memories"] for row in audit_rows) / len(audit_rows),
            "max": max(row["selected_memories"] for row in audit_rows),
        },
    }
    dump_json(output_dir / "continuity_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mem0-native", type=Path, default=DEFAULT_MEM0_NATIVE)
    parser.add_argument("--mem0-oracle", type=Path, default=DEFAULT_MEM0_ORACLE)
    parser.add_argument("--gold-embeddings", type=Path, default=DEFAULT_GOLD_EMBEDDINGS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-budget", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
