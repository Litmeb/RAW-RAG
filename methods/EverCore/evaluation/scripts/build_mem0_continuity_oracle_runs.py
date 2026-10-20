"""Build embedding-only continuity oracle runs from an existing Mem0 local run.

The experiment compares two gold-informed upper bounds without calling a
reranker:

1. ``raw_oracle`` selects an answer-bearing raw RAG chunk with a gold query,
   maps it back to the original MemoryAgentBench source session, and exposes a
   contiguous source window.
2. ``memory_oracle`` may only expose Mem0 memory objects.  It receives every
   memory derived from the raw-oracle source window, fills unused context with
   the most gold-query-similar memories from the complete store, and restores
   source order using ``batch_index`` provenance.

Both variants use the same query embedding and context-token budget.  The
resulting directories contain ``search_results.json`` and can be passed to the
evaluation CLI's answer fast path.  Existing source artifacts are read-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tiktoken
from openai import AsyncOpenAI


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
DEFAULT_MEM0_RUN = (
    EVALUATION_DIR
    / "results"
    / "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-normal"
)
DEFAULT_RAG_RUN = (
    EVALUATION_DIR
    / "results"
    / "memoryagentbench-rag-Accurate_Retrieval-eventqa_full-normal-recover-rerank"
)
DEFAULT_DATA_DIR = (
    EVALUATION_DIR
    / "data"
    / "memoryagentbench"
    / "Accurate_Retrieval"
    / "eventqa_full"
)
DEFAULT_SUFFIX = "continuity-rag-anchor-memory-oracle-5000-v3-20260716"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class SourceSession:
    conversation_index: int
    session_index: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class StoredMemory:
    memory_id: str
    conversation_index: int
    batch_index: int
    timestamp: str
    text: str
    vector: np.ndarray
    payload: dict[str, Any]


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
    match = re.search(r"(\d+)$", str(value or ""))
    if not match:
        raise ValueError(f"Cannot parse conversation index from {value!r}")
    return int(match.group(1))


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ),
        key=lambda key: int(key.replace("session_", "")),
    )


def load_source_sessions(data_dir: Path) -> list[list[SourceSession]]:
    converted = load_json(data_dir / "memoryagentbench_locomo_style.json")
    result: list[list[SourceSession]] = []
    for conv_idx, item in enumerate(converted):
        conversation = item["conversation"]
        sessions = []
        for key in session_keys(conversation):
            session_idx = int(key.replace("session_", ""))
            text = "\n".join(
                str(message.get("text") or "") for message in conversation[key]
            )
            sessions.append(
                SourceSession(
                    conversation_index=conv_idx,
                    session_index=session_idx,
                    timestamp=str(conversation.get(f"{key}_date_time") or ""),
                    text=text,
                )
            )
        result.append(sessions)
    return result


def choose_collection(client: Any) -> str:
    names = [item.name for item in client.get_collections().collections]
    if "mem0_local" in names:
        return "mem0_local"
    candidates = [name for name in names if not name.endswith("_entities")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot infer Mem0 Qdrant collection from {names}")


def as_vector(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        # Mem0 v1.1 stores the dense embedding under the empty/default name and
        # may also store a sparse ``bm25`` vector.  This experiment deliberately
        # uses dense embeddings only, both for cost and cross-representation
        # comparability with the raw RAG index.
        if "" in value:
            value = value[""]
        else:
            dense_values = [
                item
                for name, item in value.items()
                if name.lower() not in {"bm25", "sparse"}
            ]
            if len(dense_values) != 1:
                raise ValueError(f"Cannot infer dense vector from {list(value)}")
            value = dense_values[0]
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array
    return array / norm


def load_stored_memories(run_dir: Path) -> list[StoredMemory]:
    from qdrant_client import QdrantClient

    qdrant_path = run_dir / "mem0_qdrant"
    client = QdrantClient(path=str(qdrant_path))
    records: list[StoredMemory] = []
    try:
        collection = choose_collection(client)
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = dict(point.payload or {})
                records.append(
                    StoredMemory(
                        memory_id=str(point.id),
                        conversation_index=conversation_index(
                            payload.get("conversation_id")
                        ),
                        batch_index=int(payload.get("batch_index", -1)),
                        timestamp=str(payload.get("created_at") or ""),
                        text=str(payload.get("data") or payload.get("memory") or ""),
                        vector=as_vector(point.vector),
                        payload=payload,
                    )
                )
            if offset is None:
                break
    finally:
        client.close()
    return records


def normalized_matrix(rows: Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(list(rows), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


async def embed_gold_queries(
    texts: list[str], model: str, api_key: str, base_url: str, batch_size: int
) -> np.ndarray:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
        print(f"Embedded gold queries: {min(start + len(batch), len(texts))}/{len(texts)}")
    return normalized_matrix(vectors)


def lexical_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(text or "")) if len(token) > 2}


def map_raw_chunk_to_session(
    raw_chunk: str, sessions: list[SourceSession], session_tokens: list[set[str]]
) -> tuple[int, float]:
    chunk_tokens = lexical_tokens(raw_chunk)
    if not chunk_tokens:
        return 0, 0.0
    scores = [len(chunk_tokens & tokens) / len(chunk_tokens) for tokens in session_tokens]
    best = int(np.argmax(scores))
    return sessions[best].session_index, float(scores[best])


def token_count(encoding: Any, text: str) -> int:
    return len(encoding.encode(text))


def raw_segment(session: SourceSession) -> str:
    return (
        f"<RAW_SOURCE session={session.session_index} time={session.timestamp}>\n"
        f"{session.text}\n"
        "</RAW_SOURCE>\n\n"
    )


def memory_segment(memory: StoredMemory, score: float) -> str:
    return (
        f"<MEMORY source_session={memory.batch_index} time={memory.timestamp} "
        ">\n"
        f"{memory.text}\n"
        "</MEMORY>\n\n"
    )


def centered_raw_window(
    sessions: list[SourceSession],
    anchor: int,
    encoding: Any,
    budget: int,
    segment_cache: dict[tuple[int, int], str],
    cost_cache: dict[tuple[int, int], int],
) -> tuple[list[SourceSession], str]:
    by_index = {session.session_index: session for session in sessions}
    if anchor not in by_index:
        raise KeyError(f"Source session {anchor} does not exist")

    # EventQA asks for the event immediately after ``previous_event``.  Prefer
    # forward source sessions, then add the preceding session only when budget
    # remains.  This preserves the local reading order while avoiding a
    # gold-answer-dependent raw locator.
    order = [anchor]
    for radius in range(1, len(sessions) + 1):
        left, right = anchor - radius, anchor + radius
        if right in by_index:
            order.append(right)
        if left in by_index:
            order.append(left)
        if len(order) >= len(sessions):
            break

    selected: list[SourceSession] = []
    used = 0
    for index in order:
        session = by_index[index]
        key = (session.conversation_index, session.session_index)
        segment = segment_cache[key]
        cost = cost_cache[key]
        if not selected and cost > budget:
            ids = encoding.encode(segment)[:budget]
            return [session], encoding.decode(ids)
        if used + cost <= budget:
            selected.append(session)
            used += cost

    selected.sort(key=lambda item: item.session_index)
    context = "".join(
        segment_cache[(session.conversation_index, session.session_index)]
        for session in selected
    )
    return selected, context


def pack_memory_oracle(
    memories: list[StoredMemory],
    scores: np.ndarray,
    provenance_sessions: set[int],
    encoding: Any,
    budget: int,
    segment_cache: dict[str, str],
    cost_cache: dict[str, int],
) -> tuple[list[tuple[StoredMemory, float, str]], str]:
    indexed = list(enumerate(memories))
    provenance = [
        (idx, memory)
        for idx, memory in indexed
        if memory.batch_index in provenance_sessions
    ]
    provenance.sort(key=lambda pair: (pair[1].batch_index, pair[1].memory_id))
    remaining = [
        (idx, memory)
        for idx, memory in indexed
        if memory.batch_index not in provenance_sessions
    ]
    remaining.sort(key=lambda pair: (float(scores[pair[0]]), pair[1].memory_id), reverse=True)

    chosen: list[tuple[int, StoredMemory, float, str]] = []
    used = 0
    for idx, memory in provenance + remaining:
        origin = "provenance" if memory.batch_index in provenance_sessions else "global_gold"
        segment = segment_cache[memory.memory_id]
        cost = cost_cache[memory.memory_id]
        if not chosen and cost > budget:
            ids = encoding.encode(segment)[:budget]
            segment = encoding.decode(ids)
            chosen.append((idx, memory, float(scores[idx]), origin))
            used = budget
            break
        if used + cost <= budget:
            chosen.append((idx, memory, float(scores[idx]), origin))
            used += cost
        if used >= budget:
            break

    # Restore the strongest source ordering available after selection.
    chosen.sort(key=lambda item: (item[1].batch_index, item[1].timestamp, item[1].memory_id))
    context = "".join(segment_cache[memory.memory_id] for _, memory, _, _ in chosen)
    return [(memory, score, origin) for _, memory, score, origin in chosen], context


def raw_search_record(
    source: dict[str, Any],
    selected: list[SourceSession],
    context: str,
    anchor_chunk: dict[str, Any],
    anchor_session: int,
    mapping_score: float,
    context_tokens: int,
    budget: int,
) -> dict[str, Any]:
    results = [
        {
            "content": raw_segment(session).strip(),
            "score": 1.0 if session.session_index == anchor_session else 0.0,
            "metadata": {
                "memory_type": "raw_source",
                "session_index": session.session_index,
                "timestamp": session.timestamp,
            },
        }
        for session in selected
    ]
    return {
        "question_id": source["question_id"],
        "query": source["query"],
        "conversation_id": source["conversation_id"],
        "results": results,
        "retrieval_metadata": {
            "system": "mem0_continuity_raw_embedding_oracle",
            "formatted_context": context,
            "context_token_budget": budget,
            "context_tokens_after_cap": context_tokens,
            "oracle_uses_gold_answer": False,
            "raw_locator_reuses_existing_rag_top1": True,
            "existing_rag_rerank_artifact_reused": True,
            "new_reranker_calls": 0,
            "oracle_reranker_used": False,
            "raw_anchor_chunk_id": anchor_chunk.get("chunk_id"),
            "raw_anchor_chunk_index": anchor_chunk.get("chunk_index"),
            "raw_anchor_source_session": anchor_session,
            "raw_anchor_mapping_recall": mapping_score,
            "selected_source_sessions": [item.session_index for item in selected],
        },
    }


def memory_search_record(
    source: dict[str, Any],
    selected: list[tuple[StoredMemory, float, str]],
    context: str,
    provenance_sessions: set[int],
    context_tokens: int,
    budget: int,
) -> dict[str, Any]:
    results = [
        {
            "content": memory_segment(memory, score).strip(),
            "score": score,
            "user_id": str(memory.payload.get("user_id") or ""),
            "metadata": {
                "id": memory.memory_id,
                "memory_type": "mem0_extracted_memory",
                "batch_index": memory.batch_index,
                "oracle_origin": origin,
                "raw": memory.payload,
            },
        }
        for memory, score, origin in selected
    ]
    return {
        "question_id": source["question_id"],
        "query": source["query"],
        "conversation_id": source["conversation_id"],
        "results": results,
        "retrieval_metadata": {
            "system": "mem0_continuity_memory_embedding_oracle",
            "formatted_context": context,
            "context_token_budget": budget,
            "context_tokens_after_cap": context_tokens,
            "oracle_uses_gold_answer": True,
            "oracle_reranker_used": False,
            "oracle_complete_store_candidate_access": True,
            "provenance_source_sessions": sorted(provenance_sessions),
            "selected_memory_count": len(selected),
            "selected_provenance_memory_count": sum(
                origin == "provenance" for _, _, origin in selected
            ),
            "selected_global_gold_memory_count": sum(
                origin == "global_gold" for _, _, origin in selected
            ),
        },
    }


def assert_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to reuse experiment output directory: {path}. Choose a new suffix."
        )


async def build(args: argparse.Namespace) -> None:
    load_env(EVERCORE_DIR / ".env")
    mem0_run = Path(args.mem0_run).resolve()
    rag_run = Path(args.rag_run).resolve()
    data_dir = Path(args.data_dir).resolve()
    result_root = EVALUATION_DIR / "results"
    raw_dir = result_root / (
        "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-"
        f"raw-oracle-{args.suffix}"
    )
    memory_dir = result_root / (
        "memoryagentbench-mem0_local-Accurate_Retrieval-eventqa_full-"
        f"memory-oracle-{args.suffix}"
    )
    assert_new_directory(raw_dir)
    assert_new_directory(memory_dir)

    mem0_searches = load_json(mem0_run / "search_results.json")
    mem0_answers = load_json(mem0_run / "answer_results.json")
    answer_by_id = {row["question_id"]: row for row in mem0_answers}
    if len(mem0_searches) != 500 or len(answer_by_id) != 500:
        raise ValueError("Expected the complete 500-question EventQA run")

    print("Loading source sessions and persisted vector stores...")
    sessions_by_conv = load_source_sessions(data_dir)
    memories = load_stored_memories(mem0_run)
    memories_by_conv: dict[int, list[StoredMemory]] = defaultdict(list)
    for memory in memories:
        memories_by_conv[memory.conversation_index].append(memory)

    print("Loading raw RAG index (large file)...")
    rag_index = load_json(rag_run / "rag_index.json")
    rag_conversations = rag_index["conversations"]
    rag_search_by_id = {
        row["question_id"]: row
        for row in load_json(rag_run / "search_results.json")
    }

    gold_queries = []
    for search in mem0_searches:
        answer = answer_by_id[search["question_id"]]
        previous = str((answer.get("metadata") or {}).get("previous_event") or "")
        gold = str(answer.get("golden_answer") or "")
        gold_queries.append(f"{previous}\nCorrect next event: {gold}")

    cache_path = result_root / f"mem0_eventqa_gold_query_embeddings_{args.suffix}.json"
    reusable_caches = sorted(
        result_root.glob("mem0_eventqa_gold_query_embeddings_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if cache_path.exists():
        gold_embeddings = normalized_matrix(load_json(cache_path))
    elif reusable_caches:
        cache_path = reusable_caches[0]
        gold_embeddings = normalized_matrix(load_json(cache_path))
        print(f"Reusing gold-query embedding cache: {cache_path.name}")
    else:
        model = os.environ.get("VECTORIZE_MODEL", "text-embedding-3-small")
        api_key = os.environ.get("VECTORIZE_API_KEY") or os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("VECTORIZE_BASE_URL") or os.environ.get(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )
        gold_embeddings = await embed_gold_queries(
            gold_queries,
            model=model,
            api_key=api_key,
            base_url=base_url,
            batch_size=args.embedding_batch_size,
        )
        dump_json(cache_path, gold_embeddings.tolist())

    if gold_embeddings.shape[0] != len(mem0_searches):
        raise ValueError("Gold-query embedding count does not match questions")

    encoding = tiktoken.get_encoding("o200k_base")
    raw_segment_cache = {
        (session.conversation_index, session.session_index): raw_segment(session)
        for sessions in sessions_by_conv
        for session in sessions
    }
    raw_cost_cache = {
        key: token_count(encoding, segment)
        for key, segment in raw_segment_cache.items()
    }
    memory_segment_cache = {
        memory.memory_id: memory_segment(memory, 0.0) for memory in memories
    }
    memory_cost_cache = {
        key: token_count(encoding, segment)
        for key, segment in memory_segment_cache.items()
    }
    raw_records = []
    memory_records = []
    audit_rows = []
    memory_matrices: dict[int, np.ndarray] = {}
    source_token_sets = [
        [lexical_tokens(session.text) for session in sessions]
        for sessions in sessions_by_conv
    ]

    for question_idx, source_search in enumerate(mem0_searches):
        conv_idx = conversation_index(source_search["conversation_id"])
        query_vector = gold_embeddings[question_idx]
        rag_conv = rag_conversations[source_search["conversation_id"]]
        rag_search = rag_search_by_id[source_search["question_id"]]
        if not rag_search.get("results"):
            raise ValueError(f"RAG source has no results for {source_search['question_id']}")
        rag_top = rag_search["results"][0]
        raw_anchor_idx = int((rag_top.get("metadata") or {}).get("chunk_index", -1))
        raw_anchor = rag_conv["chunks"][raw_anchor_idx]
        anchor_session, mapping_recall = map_raw_chunk_to_session(
            raw_anchor["content"],
            sessions_by_conv[conv_idx],
            source_token_sets[conv_idx],
        )
        selected_sessions, raw_context = centered_raw_window(
            sessions_by_conv[conv_idx],
            anchor_session,
            encoding,
            args.token_budget,
            raw_segment_cache,
            raw_cost_cache,
        )
        provenance_sessions = {item.session_index for item in selected_sessions}

        conv_memories = memories_by_conv[conv_idx]
        if conv_idx not in memory_matrices:
            memory_matrices[conv_idx] = np.vstack(
                [memory.vector for memory in conv_memories]
            )
        memory_scores = memory_matrices[conv_idx] @ query_vector
        selected_memories, memory_context = pack_memory_oracle(
            conv_memories,
            memory_scores,
            provenance_sessions,
            encoding,
            args.token_budget,
            memory_segment_cache,
            memory_cost_cache,
        )

        raw_tokens = token_count(encoding, raw_context)
        memory_tokens = token_count(encoding, memory_context)
        raw_records.append(
            raw_search_record(
                source_search,
                selected_sessions,
                raw_context,
                raw_anchor,
                anchor_session,
                mapping_recall,
                raw_tokens,
                args.token_budget,
            )
        )
        memory_records.append(
            memory_search_record(
                source_search,
                selected_memories,
                memory_context,
                provenance_sessions,
                memory_tokens,
                args.token_budget,
            )
        )
        audit_rows.append(
            {
                "question_id": source_search["question_id"],
                "conversation_index": conv_idx,
                "raw_anchor_chunk_index": raw_anchor.get("chunk_index"),
                "raw_anchor_source_session": anchor_session,
                "mapping_recall": mapping_recall,
                "raw_source_sessions": sorted(provenance_sessions),
                "raw_context_tokens": raw_tokens,
                "memory_context_tokens": memory_tokens,
                "provenance_memories_available": sum(
                    memory.batch_index in provenance_sessions for memory in conv_memories
                ),
                "selected_memories": len(selected_memories),
                "selected_provenance_memories": sum(
                    origin == "provenance" for _, _, origin in selected_memories
                ),
            }
        )
        if (question_idx + 1) % 50 == 0:
            print(f"Built oracle contexts: {question_idx + 1}/{len(mem0_searches)}")

    if len(raw_records) != 500 or len(memory_records) != 500:
        raise AssertionError("Oracle output is incomplete")
    for record in raw_records + memory_records:
        tokens = token_count(
            encoding, record["retrieval_metadata"]["formatted_context"]
        )
        if tokens > args.token_budget:
            raise AssertionError(
                f"Context budget exceeded for {record['question_id']}: {tokens}"
            )

    raw_dir.mkdir(parents=True)
    memory_dir.mkdir(parents=True)
    dump_json(raw_dir / "search_results.json", raw_records)
    dump_json(memory_dir / "search_results.json", memory_records)

    native_eval = load_json(mem0_run / "eval_results.json")
    mapping_values = [row["mapping_recall"] for row in audit_rows]
    summary = {
        "experiment": "Mem0 continuity irrecoverability, embedding-only oracles",
        "new_reranker_calls": 0,
        "raw_locator_reuses_existing_reranked_rag_artifact": True,
        "memory_oracle_reranker_used": False,
        "token_budget": args.token_budget,
        "questions": len(audit_rows),
        "source_mem0_run": str(mem0_run),
        "source_rag_index": str(rag_run / "rag_index.json"),
        "native_mem0": {
            "accuracy": native_eval.get("accuracy"),
            "correct": native_eval.get("correct"),
            "total_questions": native_eval.get("total_questions"),
        },
        "raw_oracle_dir": str(raw_dir),
        "memory_oracle_dir": str(memory_dir),
        "gold_query_embedding_cache": str(cache_path),
        "stored_memories": len(memories),
        "mapping_recall": {
            "min": min(mapping_values),
            "mean": sum(mapping_values) / len(mapping_values),
            "below_0_5": sum(value < 0.5 for value in mapping_values),
            "below_0_8": sum(value < 0.8 for value in mapping_values),
        },
        "raw_context_tokens": {
            "min": min(row["raw_context_tokens"] for row in audit_rows),
            "mean": sum(row["raw_context_tokens"] for row in audit_rows) / len(audit_rows),
            "max": max(row["raw_context_tokens"] for row in audit_rows),
        },
        "memory_context_tokens": {
            "min": min(row["memory_context_tokens"] for row in audit_rows),
            "mean": sum(row["memory_context_tokens"] for row in audit_rows) / len(audit_rows),
            "max": max(row["memory_context_tokens"] for row in audit_rows),
        },
        "method": {
            "raw": "reuse existing strong-RAG top-1 anchor -> source session -> forward-contiguous 5000-token window (zero new reranker calls)",
            "memory": "all memories from raw window + global gold-query embedding fill -> provenance order",
        },
    }
    dump_json(raw_dir / "continuity_build_summary.json", summary)
    dump_json(memory_dir / "continuity_build_summary.json", summary)
    dump_json(raw_dir / "continuity_mapping_audit.json", audit_rows)
    dump_json(memory_dir / "continuity_mapping_audit.json", audit_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem0-run", default=str(DEFAULT_MEM0_RUN))
    parser.add_argument("--rag-run", default=str(DEFAULT_RAG_RUN))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--token-budget", type=int, default=5000)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(build(parse_args()))
