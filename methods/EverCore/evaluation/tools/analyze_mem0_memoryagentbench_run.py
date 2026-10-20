"""Diagnose error propagation in a local Mem0 MemoryAgentBench run.

The tool is read-only with respect to benchmark and run artifacts. It links
source chunks to extracted Mem0 memories through ``conversation_id`` and
``batch_index``, then audits two observable transitions:

1. source chunk -> stored memories
2. stored memories -> retrieved top-k -> final answer

Every invocation writes a fresh diagnostics directory under the selected run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
SRC_DIR = EVERCORE_DIR / "src"
for candidate in (EVERCORE_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


DEFAULT_RUN_NAME = (
    "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-normal"
)
DEFAULT_DATA_REL = Path("data/memoryagentbench/Accurate_Retrieval/eventqa_full")
DEFAULT_RUN_REL = Path("results") / DEFAULT_RUN_NAME

STORAGE_LABELS = {
    "faithful",
    "omission",
    "hallucination",
    "over_compressed",
    "mixed_issue",
    "unclear",
}
SUPPORT_LABELS = {"sufficient", "partial", "none", "contradictory", "unclear"}
ERROR_STAGE_LABELS = {
    "no_error",
    "source_ambiguity",
    "storage_loss",
    "retrieval_loss",
    "answer_error",
    "compound_error",
    "unclear",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "about",
    "after",
    "already",
    "answer",
    "based",
    "below",
    "choose",
    "event",
    "events",
    "from",
    "happens",
    "have",
    "include",
    "list",
    "only",
    "possible",
    "response",
    "subsequent",
    "task",
    "that",
    "these",
    "which",
    "with",
    "your",
}


@dataclass(frozen=True)
class ChunkRecord:
    conv: int
    chunk: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    conv: int
    chunk: int
    text: str
    created_at: str
    user_id: str
    metadata: dict[str, Any]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def truncate_text(value: str, limit: int) -> str:
    value = str(value or "")
    if len(value) <= limit:
        return value
    half = max(1, (limit - 40) // 2)
    return value[:half] + "\n...[truncated]...\n" + value[-half:]


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.replace("session_", "")),
    )


def load_chunks(data_dir: Path) -> list[list[ChunkRecord]]:
    converted = load_json(data_dir / "memoryagentbench_locomo_style.json")
    chunks_by_conv: list[list[ChunkRecord]] = []
    for conv_idx, item in enumerate(converted):
        conversation = item["conversation"]
        chunks = []
        for chunk_idx, key in enumerate(session_keys(conversation)):
            messages = conversation[key]
            text = "\n".join(str(message.get("text", "")) for message in messages)
            chunks.append(
                ChunkRecord(
                    conv=conv_idx,
                    chunk=chunk_idx,
                    timestamp=str(conversation.get(f"{key}_date_time", "")),
                    text=text,
                )
            )
        chunks_by_conv.append(chunks)
    return chunks_by_conv


def conversation_index(value: Any) -> int:
    match = re.search(r"(\d+)$", str(value or ""))
    if not match:
        raise ValueError(f"Cannot parse conversation index from {value!r}")
    return int(match.group(1))


def choose_memory_collection(client: Any, requested: str | None) -> str:
    names = [item.name for item in client.get_collections().collections]
    if requested:
        if requested not in names:
            raise ValueError(
                f"Qdrant collection {requested!r} not found; available: {names}"
            )
        return requested
    candidates = [name for name in names if not name.endswith("_entities")]
    if "mem0_local" in candidates:
        return "mem0_local"
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Could not infer the Mem0 memory collection. "
        f"Use --collection-name; available: {names}"
    )


def load_memories(
    run_dir: Path, collection_name: str | None
) -> tuple[list[MemoryRecord], dict[str, Any]]:
    from qdrant_client import QdrantClient

    qdrant_path = run_dir / "mem0_qdrant"
    if not qdrant_path.exists():
        raise FileNotFoundError(f"Mem0 Qdrant directory not found: {qdrant_path}")

    client = QdrantClient(path=str(qdrant_path))
    try:
        selected = choose_memory_collection(client, collection_name)
        collection_names = [item.name for item in client.get_collections().collections]
        entity_collection = f"{selected}_entities"
        entity_count = (
            client.count(collection_name=entity_collection, exact=True).count
            if entity_collection in collection_names
            else 0
        )

        records: list[MemoryRecord] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=selected,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                metadata = dict(payload)
                records.append(
                    MemoryRecord(
                        memory_id=str(point.id),
                        conv=conversation_index(payload.get("conversation_id")),
                        chunk=int(payload.get("batch_index", -1)),
                        text=str(payload.get("data") or payload.get("memory") or ""),
                        created_at=str(payload.get("created_at") or ""),
                        user_id=str(payload.get("user_id") or ""),
                        metadata=metadata,
                    )
                )
            if offset is None:
                break
    finally:
        client.close()

    return records, {
        "path": str(qdrant_path),
        "collection": selected,
        "memory_points": len(records),
        "entity_collection": entity_collection if entity_count else None,
        "entity_points": entity_count,
    }


def load_history_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "mem0_history.db"
    if not path.exists():
        return {"available": False, "path": str(path)}

    connection = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "history" not in tables:
            return {"available": True, "path": str(path), "history_table": False}
        event_counts = dict(
            connection.execute(
                "SELECT COALESCE(event, ''), COUNT(*) FROM history GROUP BY event"
            ).fetchall()
        )
        total, unique_ids, old_nonnull, new_nonnull, deleted = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT memory_id),
                   SUM(CASE WHEN old_memory IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN new_memory IS NOT NULL THEN 1 ELSE 0 END),
                   SUM(CASE WHEN is_deleted = 1 THEN 1 ELSE 0 END)
            FROM history
            """
        ).fetchone()
        return {
            "available": True,
            "path": str(path),
            "history_table": True,
            "rows": total,
            "unique_memory_ids": unique_ids,
            "event_counts": event_counts,
            "old_memory_nonnull": old_nonnull or 0,
            "new_memory_nonnull": new_nonnull or 0,
            "deleted_rows": deleted or 0,
        }
    finally:
        connection.close()


@lru_cache(maxsize=None)
def _normalize_text(value: str) -> str:
    return " ".join(TOKEN_RE.findall(value.lower()))


def normalize_text(value: Any) -> str:
    return _normalize_text(str(value or ""))


@lru_cache(maxsize=None)
def _content_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in TOKEN_RE.findall(value.lower())
        if len(token) > 2 and token not in STOPWORDS
    )


def content_tokens(value: Any) -> frozenset[str]:
    return _content_tokens(str(value or ""))


def token_recall(needle: set[str] | frozenset[str], text: str) -> float:
    if not needle:
        return 0.0
    return len(needle & content_tokens(text)) / len(needle)


def relevance_score(target_tokens: set[str] | frozenset[str], text: str) -> float:
    text_tokens = content_tokens(text)
    if not target_tokens or not text_tokens:
        return 0.0
    overlap = len(target_tokens & text_tokens)
    return overlap / math.sqrt(len(target_tokens) * len(text_tokens))


def stratified_sample(
    rows: list[dict[str, Any]], sample_size: int, seed: int, strata: tuple[str, ...]
) -> list[dict[str, Any]]:
    if sample_size <= 0 or not rows:
        return []
    if len(rows) <= sample_size:
        return rows[:]

    rng = random.Random(seed)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in strata)].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=str):
        group = groups[key]
        quota = max(1, round(sample_size * len(group) / len(rows)))
        selected.extend(rng.sample(group, min(quota, len(group))))
    if len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)
    selected_ids = {id(row) for row in selected}
    remaining = [row for row in rows if id(row) not in selected_ids]
    while len(selected) < sample_size and remaining:
        picked = rng.randrange(len(remaining))
        selected.append(remaining.pop(picked))
    return selected


def index_eval_results(eval_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("question_id")): row
        for row in eval_results.get("detailed_results", [])
        if row.get("question_id")
    }


def is_correct(eval_row: dict[str, Any] | None, answer_row: dict[str, Any]) -> bool:
    if eval_row is not None:
        if "metric_value" in eval_row:
            return float(eval_row.get("metric_value") or 0) > 0
        if "correct" in eval_row:
            return bool(eval_row.get("correct"))
    generated = normalize_text(answer_row.get("answer"))
    golden = normalize_text(answer_row.get("golden_answer"))
    return bool(golden and golden in generated)


def build_indexes(
    chunks_by_conv: list[list[ChunkRecord]], memories: list[MemoryRecord]
) -> tuple[dict[tuple[int, int], list[MemoryRecord]], dict[int, list[MemoryRecord]]]:
    by_chunk: dict[tuple[int, int], list[MemoryRecord]] = defaultdict(list)
    by_conv: dict[int, list[MemoryRecord]] = defaultdict(list)
    for memory in memories:
        by_chunk[(memory.conv, memory.chunk)].append(memory)
        by_conv[memory.conv].append(memory)
    for conv_idx, chunks in enumerate(chunks_by_conv):
        for chunk in chunks:
            by_chunk.setdefault((conv_idx, chunk.chunk), [])
        by_conv.setdefault(conv_idx, [])
    return by_chunk, by_conv


def deterministic_summary(
    chunks_by_conv: list[list[ChunkRecord]],
    memories: list[MemoryRecord],
    by_chunk: dict[tuple[int, int], list[MemoryRecord]],
    qdrant_summary: dict[str, Any],
    history_summary: dict[str, Any],
    searches: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    eval_results: dict[str, Any],
) -> dict[str, Any]:
    coverage_rows = []
    expected_keys = {
        (conv_idx, chunk.chunk)
        for conv_idx, chunks in enumerate(chunks_by_conv)
        for chunk in chunks
    }
    observed_keys = {(memory.conv, memory.chunk) for memory in memories}
    for conv_idx, chunks in enumerate(chunks_by_conv):
        counts = [len(by_chunk[(conv_idx, chunk.chunk)]) for chunk in chunks]
        missing = [idx for idx, count in enumerate(counts) if count == 0]
        coverage_rows.append(
            {
                "conv": conv_idx,
                "expected_chunks": len(chunks),
                "represented_chunks": sum(count > 0 for count in counts),
                "missing_chunks": len(missing),
                "missing_chunk_indices": missing,
                "memories": sum(counts),
                "memories_per_chunk": dict(Counter(counts)),
            }
        )

    search_by_id = {
        str(row.get("question_id")): row for row in searches if row.get("question_id")
    }
    answer_by_id = {
        str(row.get("question_id")): row for row in answers if row.get("question_id")
    }
    eval_by_id = index_eval_results(eval_results)
    search_ids = set(search_by_id)
    answer_ids = set(answer_by_id)
    eval_ids = set(eval_by_id)
    memory_ids = {memory.memory_id for memory in memories}
    retrieved = [item for row in searches for item in row.get("results", [])]
    retrieved_ids = {
        str(item.get("metadata", {}).get("id") or "") for item in retrieved
    }
    unknown_ids = sorted(memory_id for memory_id in retrieved_ids if memory_id and memory_id not in memory_ids)
    correct = sum(
        is_correct(eval_by_id.get(question_id), row)
        for question_id, row in answer_by_id.items()
    )

    return {
        "counts": {
            "conversations": len(chunks_by_conv),
            "source_chunks": len(expected_keys),
            "stored_memories": len(memories),
            "represented_source_chunks": len(expected_keys & observed_keys),
            "missing_source_chunks": len(expected_keys - observed_keys),
            "out_of_range_memory_chunk_keys": len(observed_keys - expected_keys),
            "search_questions": len(searches),
            "answer_questions": len(answers),
            "eval_questions": int(eval_results.get("total_questions") or len(eval_ids)),
            "correct_answers": correct,
            "accuracy": correct / len(answers) if answers else 0.0,
            "retrieved_results": len(retrieved),
            "unique_retrieved_memory_ids": len(retrieved_ids - {""}),
        },
        "coverage": {
            "by_conversation": coverage_rows,
            "missing_chunk_keys": sorted(expected_keys - observed_keys),
            "out_of_range_chunk_keys": sorted(observed_keys - expected_keys),
        },
        "result_alignment": {
            "search_without_answer": sorted(search_ids - answer_ids),
            "answer_without_search": sorted(answer_ids - search_ids),
            "answer_without_eval": sorted(answer_ids - eval_ids),
            "eval_without_answer": sorted(eval_ids - answer_ids),
            "retrieved_unknown_memory_ids": unknown_ids,
            "results_per_question": dict(
                Counter(len(row.get("results", [])) for row in searches)
            ),
        },
        "qdrant": qdrant_summary,
        "history": history_summary,
    }


def source_candidates(
    chunks: list[ChunkRecord], question: str, golden_answer: str, limit: int
) -> list[ChunkRecord]:
    gold_tokens = content_tokens(golden_answer)
    query_tokens = content_tokens(question)
    target = gold_tokens | query_tokens
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            relevance_score(gold_tokens, chunk.text),
            relevance_score(target, chunk.text),
            -chunk.chunk,
        ),
        reverse=True,
    )

    selected_indices: list[int] = []
    for chunk in ranked[: max(2, limit // 2)]:
        for index in (chunk.chunk - 1, chunk.chunk, chunk.chunk + 1):
            if 0 <= index < len(chunks) and index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= limit:
                break
        if len(selected_indices) >= limit:
            break
    return [chunks[index] for index in selected_indices]


def top_memories(
    memories: Iterable[MemoryRecord], target: str, limit: int
) -> list[MemoryRecord]:
    tokens = content_tokens(target)
    return sorted(
        memories,
        key=lambda memory: (relevance_score(tokens, memory.text), memory.memory_id),
        reverse=True,
    )[:limit]


def format_chunks(chunks: Iterable[ChunkRecord], max_chars: int) -> str:
    parts = [
        f"[chunk {chunk.chunk}; {chunk.timestamp}]\n{chunk.text}" for chunk in chunks
    ]
    return truncate_text("\n\n".join(parts), max_chars)


def format_memories(memories: Iterable[MemoryRecord], max_chars: int) -> str:
    parts = [
        f"[memory {memory.memory_id}; source chunk {memory.chunk}; "
        f"created_at {memory.created_at}]\n{memory.text}"
        for memory in memories
    ]
    return truncate_text("\n\n".join(parts), max_chars)


def format_retrieval(results: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {}) or {}
        source_metadata = metadata.get("metadata", {}) or {}
        parts.append(
            f"[rank {rank}; memory {metadata.get('id', '')}; "
            f"source chunk {source_metadata.get('batch_index', '')}; "
            f"score {result.get('score', 0)}]\n{result.get('content', '')}"
        )
    return truncate_text("\n\n".join(parts), max_chars)


def build_storage_samples(
    chunks_by_conv: list[list[ChunkRecord]],
    by_chunk: dict[tuple[int, int], list[MemoryRecord]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for conv_idx, chunks in enumerate(chunks_by_conv):
        for chunk in chunks:
            memories = by_chunk[(conv_idx, chunk.chunk)]
            rows.append(
                {
                    "sample_id": f"storage-c{conv_idx}-k{chunk.chunk}",
                    "category": "storage_fidelity",
                    "conv": conv_idx,
                    "chunk": chunk.chunk,
                    "memory_count": len(memories),
                    "coverage_bucket": "empty" if not memories else "represented",
                    "source_text": chunk.text,
                    "stored_memories": [asdict(memory) for memory in memories],
                }
            )

    missing = [row for row in rows if row["memory_count"] == 0]
    represented = [row for row in rows if row["memory_count"] > 0]
    remaining_size = max(0, sample_size - len(missing))
    selected = missing[:sample_size]
    selected.extend(
        stratified_sample(
            represented,
            remaining_size,
            seed,
            strata=("conv", "coverage_bucket"),
        )
    )
    return selected[:sample_size]


def build_question_rows(
    chunks_by_conv: list[list[ChunkRecord]],
    by_chunk: dict[tuple[int, int], list[MemoryRecord]],
    by_conv: dict[int, list[MemoryRecord]],
    searches: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    eval_results: dict[str, Any],
    source_candidate_limit: int,
    stored_candidate_limit: int,
) -> list[dict[str, Any]]:
    search_by_id = {
        str(row.get("question_id")): row for row in searches if row.get("question_id")
    }
    eval_by_id = index_eval_results(eval_results)
    rows = []
    for answer in answers:
        question_id = str(answer.get("question_id") or "")
        search = search_by_id.get(question_id, {})
        conv = conversation_index(answer.get("conversation_id"))
        question = str(answer.get("question") or search.get("query") or "")
        golden = str(answer.get("golden_answer") or "")
        candidates = source_candidates(
            chunks_by_conv[conv], question, golden, source_candidate_limit
        )
        candidate_keys = {(conv, chunk.chunk) for chunk in candidates}
        linked_memories = [
            memory for key in candidate_keys for memory in by_chunk.get(key, [])
        ]
        semantic_memories = top_memories(
            by_conv[conv], f"{golden}\n{question}", stored_candidate_limit
        )
        selected_memories = []
        seen_memory_ids = set()
        for memory in linked_memories + semantic_memories:
            if memory.memory_id in seen_memory_ids:
                continue
            seen_memory_ids.add(memory.memory_id)
            selected_memories.append(memory)
            if len(selected_memories) >= stored_candidate_limit:
                break

        retrieved = list(search.get("results", []))
        gold_tokens = content_tokens(golden)
        storage_text = "\n".join(memory.text for memory in by_conv[conv])
        retrieval_text = "\n".join(str(item.get("content") or "") for item in retrieved)
        correct = is_correct(eval_by_id.get(question_id), answer)
        rows.append(
            {
                "sample_id": f"question-{question_id}",
                "category": "question_propagation",
                "question_id": question_id,
                "conv": conv,
                "correct": correct,
                "question": question,
                "golden_answer": golden,
                "generated_answer": str(answer.get("answer") or ""),
                "source_candidates": [asdict(chunk) for chunk in candidates],
                "stored_candidates": [asdict(memory) for memory in selected_memories],
                "retrieved_results": retrieved,
                "deterministic_proxies": {
                    "gold_exact_in_any_stored_memory": normalize_text(golden)
                    in normalize_text(storage_text),
                    "gold_exact_in_retrieval": normalize_text(golden)
                    in normalize_text(retrieval_text),
                    "max_stored_gold_token_recall": max(
                        (token_recall(gold_tokens, memory.text) for memory in by_conv[conv]),
                        default=0.0,
                    ),
                    "max_retrieved_gold_token_recall": max(
                        (
                            token_recall(gold_tokens, str(item.get("content") or ""))
                            for item in retrieved
                        ),
                        default=0.0,
                    ),
                },
            }
        )
    return rows


def build_question_samples(
    question_rows: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    return stratified_sample(
        question_rows, sample_size, seed, strata=("conv", "correct")
    )


def storage_prompt(sample: dict[str, Any]) -> str:
    memory_records = [MemoryRecord(**row) for row in sample["stored_memories"]]
    return f"""You are auditing Mem0's transformation of one source chunk into stored memories.

Judge whether the stored memories preserve the major events, entities, relations, and ordering cues that a future question may need. Concise paraphrasing is acceptable. Do not require every stylistic detail. An empty memory list is an omission when the source contains meaningful information.

SOURCE CHUNK:
{truncate_text(sample['source_text'], 10000)}

STORED MEMORIES:
{format_memories(memory_records, 9000) or '[none]'}

Return only JSON:
{{
  "label": "faithful|omission|hallucination|over_compressed|mixed_issue|unclear",
  "is_issue": true,
  "confidence": 0.0,
  "preserved_major_facts": 0,
  "missing_major_facts": 0,
  "unsupported_claims": 0,
  "rationale": "short evidence-based reason"
}}
"""


def question_prompt(sample: dict[str, Any]) -> str:
    source_records = [ChunkRecord(**row) for row in sample["source_candidates"]]
    memory_records = [MemoryRecord(**row) for row in sample["stored_candidates"]]
    return f"""You are tracing an answer through a Mem0 pipeline.

Evaluate each layer independently against the GOLD ANSWER. "Sufficient" means that layer contains enough evidence to choose the gold answer from the options without relying on prior knowledge. The source section contains the most relevant source candidates selected from the full document; use "unclear" rather than inventing evidence if those candidates are inadequate.

Attribution rules for evidence propagation, regardless of whether the final
answer happens to be correct:
- storage_loss: source is sufficient, but stored memories lose the necessary evidence.
- retrieval_loss: stored memories are sufficient, but retrieved top-k loses the evidence.
- answer_error: retrieved top-k is sufficient, but the final answer is wrong.
- compound_error: more than one transition materially contributes.
- source_ambiguity: even the source candidates do not establish the gold answer.
- no_error: all observable layers preserve sufficient evidence and the final answer is correct.

A correct final answer does not erase an observable upstream loss: it may be a
lucky guess or recovery from partial evidence. Assign the corresponding loss
stage when the support labels show one, even when FINAL ANSWER CORRECT is true.

QUESTION:
{truncate_text(sample['question'], 5000)}

GOLD ANSWER:
{sample['golden_answer']}

GENERATED ANSWER:
{sample['generated_answer']}

FINAL ANSWER CORRECT: {str(bool(sample['correct'])).lower()}

SOURCE CANDIDATES:
{format_chunks(source_records, 12000)}

STORED MEMORY CANDIDATES:
{format_memories(memory_records, 10000) or '[none]'}

RETRIEVED TOP-K:
{format_retrieval(sample['retrieved_results'], 12000) or '[none]'}

Return only JSON:
{{
  "source_support": "sufficient|partial|none|contradictory|unclear",
  "storage_support": "sufficient|partial|none|contradictory|unclear",
  "retrieval_support": "sufficient|partial|none|contradictory|unclear",
  "error_stage": "no_error|source_ambiguity|storage_loss|retrieval_loss|answer_error|compound_error|unclear",
  "confidence": 0.0,
  "rationale": "short evidence-based reason"
}}
"""


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
    raise ValueError("No valid JSON object found")


def normalize_judgment(sample: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    category = sample["category"]
    if category == "storage_fidelity":
        label = str(parsed.get("label") or "unclear")
        if label not in STORAGE_LABELS:
            label = "unclear"
        parsed["label"] = label
        parsed["is_issue"] = label != "faithful"
        return parsed

    for key in ("source_support", "storage_support", "retrieval_support"):
        value = str(parsed.get(key) or "unclear")
        parsed[key] = value if value in SUPPORT_LABELS else "unclear"
    stage = str(parsed.get("error_stage") or "unclear")
    if stage not in ERROR_STAGE_LABELS:
        stage = "unclear"
    if not sample.get("correct") and stage == "no_error":
        stage = "unclear"
    parsed["error_stage"] = stage
    return parsed


async def judge_one(
    sample: dict[str, Any],
    llm: Any,
    retries: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt = (
        storage_prompt(sample)
        if sample["category"] == "storage_fidelity"
        else question_prompt(sample)
    )
    raw = ""
    last_error = ""
    async with semaphore:
        for _attempt in range(1, retries + 1):
            try:
                raw = await llm.generate(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                judgment = normalize_judgment(sample, extract_json_object(raw))
                return {
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "ok": True,
                    "judgment": judgment,
                    "raw_response": raw,
                }
            except Exception as exc:  # Preserve audit failures in the artifact.
                last_error = f"{type(exc).__name__}: {exc}"
        return {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "ok": False,
            "error": last_error,
            "judgment": normalize_judgment(sample, {"rationale": last_error}),
            "raw_response": raw,
        }


async def run_judging(
    samples: list[dict[str, Any]],
    output_path: Path,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    concurrency: int,
    retries: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    from memory_layer.llm.llm_provider import LLMProvider

    llm = LLMProvider(
        provider_type=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = []

    async def run_and_record(sample: dict[str, Any]) -> None:
        result = await judge_one(sample, llm, retries, max_tokens, semaphore)
        append_jsonl(output_path, result)
        results.append(result)
        if len(results) % 25 == 0 or len(results) == len(samples):
            print(f"Judged {len(results)}/{len(samples)} samples")

    await asyncio.gather(*(run_and_record(sample) for sample in samples))
    return sorted(results, key=lambda row: row["sample_id"])


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return {
        "low": (centre - margin) / denominator,
        "high": (centre + margin) / denominator,
    }


def summarize_judgments(results: list[dict[str, Any]]) -> dict[str, Any]:
    storage_rows = [row for row in results if row["category"] == "storage_fidelity"]
    question_rows = [
        row for row in results if row["category"] == "question_propagation"
    ]
    storage_labels = Counter(
        str(row.get("judgment", {}).get("label", "unclear")) for row in storage_rows
    )
    storage_issues = len(storage_rows) - storage_labels.get("faithful", 0)

    stages = Counter(
        str(row.get("judgment", {}).get("error_stage", "unclear"))
        for row in question_rows
    )
    supports = {
        key: dict(
            Counter(
                str(row.get("judgment", {}).get(key, "unclear"))
                for row in question_rows
            )
        )
        for key in ("source_support", "storage_support", "retrieval_support")
    }
    return {
        "storage_fidelity": {
            "total": len(storage_rows),
            "parse_failures": sum(not row.get("ok") for row in storage_rows),
            "labels": dict(storage_labels),
            "issues": storage_issues,
            "issue_rate": storage_issues / len(storage_rows) if storage_rows else 0.0,
            "issue_rate_wilson_95": wilson_interval(storage_issues, len(storage_rows)),
        },
        "question_propagation": {
            "total": len(question_rows),
            "parse_failures": sum(not row.get("ok") for row in question_rows),
            "error_stage_counts": dict(stages),
            "support_counts": supports,
        },
    }


def representative_examples(
    samples: list[dict[str, Any]], results: list[dict[str, Any]], limit: int = 5
) -> dict[str, list[dict[str, Any]]]:
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        judgment = result.get("judgment", {})
        if result["category"] == "storage_fidelity":
            label = str(judgment.get("label", "unclear"))
            if label == "faithful" or len(examples[label]) >= limit:
                continue
            sample = samples_by_id[result["sample_id"]]
            examples[label].append(
                {
                    "sample_id": result["sample_id"],
                    "conv": sample.get("conv"),
                    "chunk": sample.get("chunk"),
                    "rationale": judgment.get("rationale"),
                }
            )
        else:
            stage = str(judgment.get("error_stage", "unclear"))
            if stage == "no_error" or len(examples[stage]) >= limit:
                continue
            sample = samples_by_id[result["sample_id"]]
            examples[stage].append(
                {
                    "sample_id": result["sample_id"],
                    "question_id": sample.get("question_id"),
                    "golden_answer": sample.get("golden_answer"),
                    "generated_answer": sample.get("generated_answer"),
                    "rationale": judgment.get("rationale"),
                }
            )
    return dict(examples)


def proxy_summary(question_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for proxy in (
        "gold_exact_in_any_stored_memory",
        "gold_exact_in_retrieval",
    ):
        groups: dict[bool, list[dict[str, Any]]] = defaultdict(list)
        for row in question_rows:
            groups[bool(row["deterministic_proxies"][proxy])].append(row)
        summary[proxy] = {
            str(value).lower(): {
                "questions": len(rows),
                "wrong": sum(not row["correct"] for row in rows),
                "wrong_rate": (
                    sum(not row["correct"] for row in rows) / len(rows) if rows else 0.0
                ),
            }
            for value, rows in sorted(groups.items())
        }
    for proxy in (
        "max_stored_gold_token_recall",
        "max_retrieved_gold_token_recall",
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in question_rows:
            value = float(row["deterministic_proxies"][proxy])
            if value < 0.25:
                bucket = "[0.00,0.25)"
            elif value < 0.5:
                bucket = "[0.25,0.50)"
            elif value < 0.75:
                bucket = "[0.50,0.75)"
            else:
                bucket = "[0.75,1.00]"
            groups[bucket].append(row)
        summary[proxy] = {
            bucket: {
                "questions": len(rows),
                "wrong": sum(not row["correct"] for row in rows),
                "wrong_rate": sum(not row["correct"] for row in rows) / len(rows),
            }
            for bucket, rows in sorted(groups.items())
        }
    return summary


def compact_question_rows(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep full evidence in sampled rows while making the all-question index small."""
    compact = []
    for row in question_rows:
        compact.append(
            {
                "question_id": row["question_id"],
                "conv": row["conv"],
                "correct": row["correct"],
                "golden_answer": row["golden_answer"],
                "generated_answer": row["generated_answer"],
                "source_candidate_chunks": [
                    chunk["chunk"] for chunk in row["source_candidates"]
                ],
                "stored_candidate_memory_ids": [
                    memory["memory_id"] for memory in row["stored_candidates"]
                ],
                "retrieved_memory_ids": [
                    str(result.get("metadata", {}).get("id") or "")
                    for result in row["retrieved_results"]
                ],
                "deterministic_proxies": row["deterministic_proxies"],
            }
        )
    return compact


def write_markdown(
    path: Path,
    deterministic: dict[str, Any],
    proxy: dict[str, Any],
    sample_counts: dict[str, int],
    judgments: dict[str, Any] | None,
    examples: dict[str, list[dict[str, Any]]] | None,
    output_dir: Path,
    model: str,
) -> None:
    counts = deterministic["counts"]
    lines = [
        "# MemoryAgentBench Mem0 Diagnostic Summary",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Deterministic Checks",
        "",
        f"- Conversations: {counts['conversations']}",
        f"- Source chunks: {counts['source_chunks']}",
        f"- Stored memories: {counts['stored_memories']}",
        f"- Represented source chunks: {counts['represented_source_chunks']}/{counts['source_chunks']}",
        f"- Missing source chunks: {counts['missing_source_chunks']}",
        f"- Questions: {counts['answer_questions']}",
        f"- Retrieved results: {counts['retrieved_results']}",
        f"- Accuracy: {counts['correct_answers']}/{counts['answer_questions']} ({counts['accuracy'] * 100:.2f}%)",
        "",
        "### Coverage By Conversation",
        "",
        "| conv | chunks | represented | missing | memories |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in deterministic["coverage"]["by_conversation"]:
        lines.append(
            f"| {row['conv']} | {row['expected_chunks']} | {row['represented_chunks']} | "
            f"{row['missing_chunks']} | {row['memories']} |"
        )

    lines.extend(["", "### Deterministic Propagation Proxies", ""])
    for proxy_name, buckets in proxy.items():
        for value, row in buckets.items():
            lines.append(
                f"- `{proxy_name}={value}`: {row['wrong']}/{row['questions']} wrong "
                f"({row['wrong_rate'] * 100:.1f}%)"
            )
    lines.extend(
        [
            "",
            "Source candidates use gold-aware lexical ranking. LLM source-support "
            "judgments are conditional on this candidate-selection recall.",
        ]
    )

    lines.extend(
        [
            "",
            "## Samples",
            "",
            f"- Storage fidelity: {sample_counts.get('storage_fidelity', 0)}",
            f"- Question propagation: {sample_counts.get('question_propagation', 0)}",
        ]
    )
    if judgments is None:
        lines.extend(["", "## LLM Audit", "", "Not run (`--dry-run` or `--skip-llm`)."])
    else:
        storage = judgments["storage_fidelity"]
        propagation = judgments["question_propagation"]
        lines.extend(
            [
                "",
                "## LLM Audit",
                "",
                f"Judge model: `{model}`",
                "",
                f"- Storage issues: {storage['issues']}/{storage['total']} "
                f"({storage['issue_rate'] * 100:.1f}%)",
                f"- Storage labels: `{storage['labels']}`",
                f"- Error stages: `{propagation['error_stage_counts']}`",
                f"- Support labels: `{propagation['support_counts']}`",
                f"- Parse failures: {storage['parse_failures'] + propagation['parse_failures']}",
            ]
        )
        if examples:
            lines.extend(["", "## Representative Issues", ""])
            for label, rows in sorted(examples.items()):
                lines.append(f"### {label}")
                for row in rows:
                    lines.append(
                        f"- `{row['sample_id']}`: {row.get('rationale', '')}"
                    )
                lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def create_output_dir(run_dir: Path, no_llm: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if no_llm else "llm_audit"
    output_dir = run_dir / f"mem0_diagnostics_{timestamp}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=EVALUATION_DIR / DEFAULT_RUN_REL)
    parser.add_argument("--data-dir", type=Path, default=EVALUATION_DIR / DEFAULT_DATA_REL)
    parser.add_argument("--collection-name")
    parser.add_argument("--sample-size", type=int, default=100, help="Samples per audit category.")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--source-candidates", type=int, default=8)
    parser.add_argument("--stored-candidates", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--judge-provider", default="deepseek")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--judge-base-url-env", default="DEEPSEEK_BASE_URL")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    load_env_file(EVERCORE_DIR / ".env")
    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative")

    no_llm = args.dry_run or args.skip_llm
    api_key = ""
    base_url = ""
    if not no_llm:
        api_key = os.environ.get(args.judge_api_key_env, "")
        base_url = os.environ.get(args.judge_base_url_env, "")
        if not api_key or not base_url:
            raise ValueError(
                f"Missing {args.judge_api_key_env} or {args.judge_base_url_env}"
            )

    chunks_by_conv = load_chunks(data_dir)
    memories, qdrant_summary = load_memories(run_dir, args.collection_name)
    by_chunk, by_conv = build_indexes(chunks_by_conv, memories)
    searches = load_json(run_dir / "search_results.json")
    answers = load_json(run_dir / "answer_results.json")
    eval_results = load_json(run_dir / "eval_results.json")
    history_summary = load_history_summary(run_dir)

    deterministic = deterministic_summary(
        chunks_by_conv,
        memories,
        by_chunk,
        qdrant_summary,
        history_summary,
        searches,
        answers,
        eval_results,
    )
    question_rows = build_question_rows(
        chunks_by_conv,
        by_chunk,
        by_conv,
        searches,
        answers,
        eval_results,
        source_candidate_limit=max(1, args.source_candidates),
        stored_candidate_limit=max(1, args.stored_candidates),
    )
    storage_samples = build_storage_samples(
        chunks_by_conv, by_chunk, args.sample_size, args.seed
    )
    question_samples = build_question_samples(
        question_rows, args.sample_size, args.seed + 1
    )
    samples = storage_samples + question_samples
    sample_counts = dict(Counter(sample["category"] for sample in samples))
    proxy = proxy_summary(question_rows)

    output_dir = create_output_dir(run_dir, no_llm=no_llm)
    dump_json(
        output_dir / "deterministic_question_rows.json",
        compact_question_rows(question_rows),
    )
    dump_json(output_dir / "samples.json", samples)
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    results = None
    judgments = None
    examples = None
    if no_llm:
        (output_dir / "judge_results.jsonl").write_text("", encoding="utf-8")
    else:
        results = await run_judging(
            samples,
            output_dir / "judge_results.jsonl",
            provider=args.judge_provider,
            model=args.judge_model,
            api_key=api_key,
            base_url=base_url,
            concurrency=args.judge_concurrency,
            retries=args.judge_retries,
            max_tokens=max(256, args.judge_max_tokens),
        )
        dump_json(output_dir / "judge_results.json", results)
        judgments = summarize_judgments(results)
        examples = representative_examples(samples, results)

    summary = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "mode": "dry_run" if no_llm else "llm_audit",
        "judge_model": None if no_llm else args.judge_model,
        "seed": args.seed,
        "sample_counts": sample_counts,
        "deterministic": deterministic,
        "propagation_proxies": proxy,
        "judgments": judgments,
        "examples": examples,
    }
    dump_json(output_dir / "summary.json", summary)
    write_markdown(
        output_dir / "summary.md",
        deterministic,
        proxy,
        sample_counts,
        judgments,
        examples,
        output_dir,
        args.judge_model,
    )

    print(f"Diagnostics written to: {output_dir}")
    print(json.dumps(deterministic["counts"], ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
