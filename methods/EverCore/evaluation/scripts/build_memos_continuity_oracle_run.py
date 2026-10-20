"""Build a no-reranker, gold-informed Memos memory continuity oracle.

The oracle may expose only Memos memory text.  It receives all memories whose
recorded source chunks overlap the raw evidence window used in the Mem0
continuity experiment, then fills the remaining 5000-token budget with memories
selected by dense similarity to ``previous_event + gold answer``.  Selected
memories are restored to source-chunk order before reader evaluation.

The script reads the embeddings already stored in an existing Memos diagnostic
export and never calls the Memos search endpoint, an embedding API, or a
reranker.  Only the extracted ``memory`` text is exposed to the reader;
``metadata.sources`` is used solely as hidden provenance metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_MEMOS_RUN = RESULTS_DIR / (
    "memoryagentbench-memos_local-Accurate_Retrieval-eventqa_full-fine"
)
DEFAULT_MEMORY_RECORDS = DEFAULT_MEMOS_RUN / (
    "memos_diagnostics_20260716_094401_llm_audit/memory_records.json"
)
DEFAULT_MEMORY_EXPORT = DEFAULT_MEMOS_RUN / (
    "memos_diagnostics_20260716_094401_llm_audit/memory_export.json"
)
DEFAULT_MEM0_RUN = RESULTS_DIR / (
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
    "memoryagentbench-memos_local-Accurate_Retrieval-eventqa_full-memory-oracle-"
    "continuity-gold-dense-5000-20260716"
)
DEFAULT_MEMORY_EMBEDDINGS = RESULTS_DIR / (
    "memos_eventqa_memory_embeddings_text-embedding-3-small_20260716.npz"
)


@dataclass(frozen=True)
class MemoryRecord:
    conv: int
    cube_id: str
    memory_id: str
    text: str
    memory_type: str
    source_chunks: tuple[int, ...]
    source_timestamps: tuple[str, ...]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def conversation_index(value: Any) -> int:
    match = re.search(r"(\d+)(?:\D*)$", str(value or ""))
    if not match:
        raise ValueError(f"Cannot parse conversation index from {value!r}")
    return int(match.group(1))


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def load_memory_records(path: Path) -> list[MemoryRecord]:
    rows = load_json(path)
    records = []
    for row in rows:
        text = str(row.get("memory") or "").strip()
        conv = row.get("conv")
        if conv is None or not text:
            continue
        records.append(
            MemoryRecord(
                conv=int(conv),
                cube_id=str(row.get("cube_id") or ""),
                memory_id=str(row.get("memory_id") or ""),
                text=text,
                memory_type=str(row.get("memory_type") or ""),
                source_chunks=tuple(int(item) for item in row.get("source_chunk_indices") or []),
                source_timestamps=tuple(str(item) for item in row.get("source_timestamps") or []),
            )
        )
    return records


def load_export_embeddings(
    memory_export_path: Path, memory_ids: np.ndarray
) -> np.ndarray:
    exported = load_json(memory_export_path)
    by_id = {}
    for row in exported:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        embedding = metadata.get("embedding")
        if embedding:
            by_id[str(row.get("id") or "")] = embedding
    missing = [str(memory_id) for memory_id in memory_ids if str(memory_id) not in by_id]
    if missing:
        raise ValueError(
            f"Memos export is missing embeddings for {len(missing)} memories; "
            f"first missing ID: {missing[0]}"
        )
    matrix = np.asarray([by_id[str(memory_id)] for memory_id in memory_ids], dtype=np.float32)
    return normalize_matrix(matrix)


def memory_segment(memory: MemoryRecord, score: float, origin: str) -> str:
    chunks = ",".join(str(item) for item in memory.source_chunks) or "unknown"
    timestamp = memory.source_timestamps[0] if memory.source_timestamps else ""
    return (
        f"<MEMORY source_chunks={chunks} time={timestamp} origin={origin} "
        f"score={score:.6f}>\n{memory.text}\n</MEMORY>\n\n"
    )


def source_sort_key(memory: MemoryRecord) -> tuple[int, str, str]:
    chunk = min(memory.source_chunks) if memory.source_chunks else 10**9
    timestamp = memory.source_timestamps[0] if memory.source_timestamps else ""
    return chunk, timestamp, memory.memory_id


def pack_oracle(
    memories: list[MemoryRecord],
    scores: np.ndarray,
    provenance_chunks: set[int],
    encoding: Any,
    budget: int,
) -> tuple[list[tuple[MemoryRecord, float, str]], str]:
    provenance_indices = [
        index
        for index, memory in enumerate(memories)
        if provenance_chunks.intersection(memory.source_chunks)
    ]
    provenance_set = set(provenance_indices)
    provenance_indices.sort(key=lambda index: source_sort_key(memories[index]))
    global_indices = [index for index in range(len(memories)) if index not in provenance_set]
    global_indices.sort(key=lambda index: (float(scores[index]), memories[index].memory_id), reverse=True)

    chosen: list[tuple[MemoryRecord, float, str]] = []
    used = 0
    for index in provenance_indices + global_indices:
        memory = memories[index]
        origin = "raw_window_provenance" if index in provenance_set else "global_gold_dense"
        score = float(scores[index])
        segment = memory_segment(memory, score, origin)
        cost = len(encoding.encode(segment))
        if not chosen and cost > budget:
            chosen.append((memory, score, origin))
            break
        if used + cost <= budget:
            chosen.append((memory, score, origin))
            used += cost
        if used >= budget:
            break

    chosen.sort(key=lambda item: source_sort_key(item[0]))
    context = "".join(
        memory_segment(memory, score, origin) for memory, score, origin in chosen
    )
    if len(encoding.encode(context)) > budget:
        raise AssertionError("Packed Memos context exceeds token budget")
    return chosen, context


def build(args: argparse.Namespace) -> None:
    memos_run = args.memos_run.resolve()
    memory_records_path = args.memory_records.resolve()
    memory_export_path = args.memory_export.resolve()
    mem0_run = args.mem0_run.resolve()
    mem0_oracle = args.mem0_oracle.resolve()
    gold_embeddings_path = args.gold_embeddings.resolve()
    output_dir = args.output_dir.resolve()
    memory_embeddings_path = args.memory_embeddings.resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {output_dir}")

    load_env(EVERCORE_DIR / ".env")
    memories = load_memory_records(memory_records_path)
    if not memories:
        raise ValueError("No Memos memories loaded")

    memos_searches = load_json(memos_run / "search_results.json")
    mem0_searches = load_json(mem0_run / "search_results.json")
    gold_embeddings = normalize_matrix(np.asarray(load_json(gold_embeddings_path), dtype=np.float32))
    if len(mem0_searches) != len(gold_embeddings):
        raise ValueError("Gold embedding cache does not align with Mem0 questions")
    gold_by_question = {
        str(row["question_id"]): gold_embeddings[index]
        for index, row in enumerate(mem0_searches)
    }

    mapping_rows = load_json(mem0_oracle / "continuity_mapping_audit.json")
    mapping_by_question = {str(row["question_id"]): row for row in mapping_rows}

    memory_ids = np.asarray([memory.memory_id for memory in memories])
    if memory_embeddings_path.exists():
        cached = np.load(memory_embeddings_path)
        cached_ids = cached["memory_ids"]
        if not np.array_equal(cached_ids, memory_ids):
            raise ValueError("Memos memory embedding cache IDs do not match export")
        memory_embeddings = normalize_matrix(cached["embeddings"])
        print(f"Reusing Memos memory embedding cache: {memory_embeddings_path.name}")
    else:
        print(f"Loading stored Memos embeddings from: {memory_export_path}")
        memory_embeddings = load_export_embeddings(memory_export_path, memory_ids)
        np.savez_compressed(
            memory_embeddings_path,
            memory_ids=memory_ids,
            embeddings=memory_embeddings,
        )

    if memory_embeddings.shape[1] != gold_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: memories={memory_embeddings.shape[1]} "
            f"queries={gold_embeddings.shape[1]}"
        )

    indices_by_conv: dict[int, list[int]] = defaultdict(list)
    for index, memory in enumerate(memories):
        indices_by_conv[memory.conv].append(index)

    encoding = tiktoken.get_encoding("o200k_base")
    output_rows = []
    audit_rows = []
    for question_index, source in enumerate(memos_searches):
        question_id = str(source["question_id"])
        conv = conversation_index(source["conversation_id"])
        global_indices = indices_by_conv[conv]
        conv_memories = [memories[index] for index in global_indices]
        conv_matrix = memory_embeddings[global_indices]
        query_vector = gold_by_question[question_id]
        scores = conv_matrix @ query_vector

        # The Mem0 audit records source sessions as one-based indices; the Memos
        # diagnostic export records the same converted chunks as zero-based indices.
        raw_sessions = mapping_by_question[question_id]["raw_source_sessions"]
        provenance_chunks = {int(item) - 1 for item in raw_sessions}
        selected, context = pack_oracle(
            conv_memories,
            scores,
            provenance_chunks,
            encoding,
            args.token_budget,
        )
        context_tokens = len(encoding.encode(context))
        results = [
            {
                "content": memory.text,
                "score": score,
                "user_id": memory.cube_id,
                "metadata": {
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type,
                    "source_chunk_indices": list(memory.source_chunks),
                    "source_timestamps": list(memory.source_timestamps),
                    "oracle_origin": origin,
                },
            }
            for memory, score, origin in selected
        ]
        output_rows.append(
            {
                "question_id": question_id,
                "query": source["query"],
                "conversation_id": source["conversation_id"],
                "results": results,
                "retrieval_metadata": {
                    "system": "memos_continuity_memory_dense_oracle",
                    "formatted_context": context,
                    "context_token_budget": args.token_budget,
                    "context_tokens_after_cap": context_tokens,
                    "oracle_uses_gold_answer": True,
                    "oracle_complete_store_candidate_access": True,
                    "oracle_reranker_used": False,
                    "new_reranker_calls": 0,
                    "raw_control_reused": True,
                    "provenance_source_chunks": sorted(provenance_chunks),
                    "selected_memory_count": len(selected),
                    "selected_provenance_memory_count": sum(
                        origin == "raw_window_provenance" for _, _, origin in selected
                    ),
                },
            }
        )
        audit_rows.append(
            {
                "question_id": question_id,
                "conversation_index": conv,
                "provenance_source_chunks": sorted(provenance_chunks),
                "available_conversation_memories": len(conv_memories),
                "available_provenance_memories": sum(
                    bool(provenance_chunks.intersection(memory.source_chunks))
                    for memory in conv_memories
                ),
                "selected_memories": len(selected),
                "selected_provenance_memories": sum(
                    origin == "raw_window_provenance" for _, _, origin in selected
                ),
                "context_tokens": context_tokens,
            }
        )
        if (question_index + 1) % 50 == 0:
            print(f"Built Memos oracle contexts: {question_index + 1}/{len(memos_searches)}")

    if len(output_rows) != 500:
        raise AssertionError(f"Expected 500 questions, got {len(output_rows)}")
    output_dir.mkdir(parents=True)
    dump_json(output_dir / "search_results.json", output_rows)
    dump_json(output_dir / "continuity_mapping_audit.json", audit_rows)
    native_eval = load_json(memos_run / "eval_results.json")
    summary = {
        "experiment": "Memos continuity recovery, gold-informed dense memory oracle",
        "questions": len(output_rows),
        "stored_memories": len(memories),
        "native_memos_run": str(memos_run),
        "native_memos_accuracy": native_eval.get("accuracy"),
        "memory_records": str(memory_records_path),
        "memory_export": str(memory_export_path),
        "memory_embedding_cache": str(memory_embeddings_path),
        "gold_query_embedding_cache": str(gold_embeddings_path),
        "output_dir": str(output_dir),
        "token_budget": args.token_budget,
        "new_reranker_calls": 0,
        "new_embedding_api_calls": 0,
        "memos_search_endpoint_called": False,
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
    parser.add_argument("--memos-run", type=Path, default=DEFAULT_MEMOS_RUN)
    parser.add_argument("--memory-records", type=Path, default=DEFAULT_MEMORY_RECORDS)
    parser.add_argument("--memory-export", type=Path, default=DEFAULT_MEMORY_EXPORT)
    parser.add_argument("--mem0-run", type=Path, default=DEFAULT_MEM0_RUN)
    parser.add_argument("--mem0-oracle", type=Path, default=DEFAULT_MEM0_ORACLE)
    parser.add_argument("--gold-embeddings", type=Path, default=DEFAULT_GOLD_EMBEDDINGS)
    parser.add_argument("--memory-embeddings", type=Path, default=DEFAULT_MEMORY_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-budget", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
