"""Build a zero-API, no-reranker EMem continuity recovery oracle.

The reader sees only EMem EDU text.  For each EventQA question, the oracle first
includes every EDU extracted from the raw evidence window used by the completed
Mem0 experiment, then fills a 5000-token budget from the complete conversation
EDU store using a gold-informed dense query.  Existing EMem EDU embeddings and
the existing gold-query embedding cache are reused; no embedding API or
reranker is called.  Selected EDUs are restored to source-session order.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import tiktoken


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
REPO_ROOT = EVERCORE_DIR.parents[1]
RESULTS_DIR = EVALUATION_DIR / "results"
DEFAULT_NATIVE = RESULTS_DIR / (
    "memoryagentbench-emem-Accurate_Retrieval-eventqa_full-normal"
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
    "memoryagentbench-emem-Accurate_Retrieval-eventqa_full-memory-oracle-"
    "continuity-gold-dense-5000-20260716"
)


@dataclass(frozen=True)
class EduRecord:
    edu_id: str
    session_index: int
    within_session_index: int
    date: str
    speakers: tuple[str, ...]
    text: str


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


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def emem_paths(run_dir: Path, conv: int) -> tuple[Path, Path]:
    sample_dir = run_dir / "emem_outputs" / f"sample_memoryagentbench_{conv}__00_full"
    openie_paths = sorted(sample_dir.glob("openie_results_ner_*.pkl"))
    parquet_paths = sorted(sample_dir.glob("*/edu_embeddings/vdb_edu.parquet"))
    if len(openie_paths) != 1 or len(parquet_paths) != 1:
        raise ValueError(
            f"Expected one EMem OpenIE and EDU parquet for conversation {conv}; "
            f"found {openie_paths} and {parquet_paths}"
        )
    return openie_paths[0], parquet_paths[0]


def load_emem_edu_store(run_dir: Path, conv: int) -> tuple[list[EduRecord], np.ndarray]:
    os.environ.setdefault("SUPPORT_JSON_SCHEMA", "false")
    emem_src = REPO_ROOT / "EMem" / "src"
    if str(emem_src) not in sys.path:
        sys.path.insert(0, str(emem_src))

    openie_path, parquet_path = emem_paths(run_dir, conv)
    with open(openie_path, "rb") as handle:
        openie = pickle.load(handle)  # noqa: S301 - trusted local experiment artifact

    records: list[EduRecord] = []
    for fallback_session, item in enumerate(openie.get("sessions") or []):
        session = item.get("session")
        session_index = int(getattr(session, "session_id", fallback_session))
        for edu_index, edu in enumerate(item.get("edus") or []):
            edu_id = str(getattr(edu, "edu_id", ""))
            text = str(getattr(edu, "edu_text", "")).strip()
            if not edu_id or not text:
                continue
            records.append(
                EduRecord(
                    edu_id=edu_id,
                    session_index=session_index,
                    within_session_index=edu_index,
                    date=str(getattr(edu, "date", "") or getattr(session, "date_time", "")),
                    speakers=tuple(str(value) for value in getattr(edu, "source_speakers", []) or []),
                    text=text,
                )
            )

    table = pq.read_table(parquet_path, columns=["hash_id", "embedding"])
    parquet_ids = [str(value) for value in table.column("hash_id").to_pylist()]
    parquet_vectors = np.asarray(table.column("embedding").to_pylist(), dtype=np.float32)
    row_by_id = {edu_id: index for index, edu_id in enumerate(parquet_ids)}
    missing = [record.edu_id for record in records if record.edu_id not in row_by_id]
    if missing:
        raise ValueError(f"{len(missing)} EMem EDUs are missing embeddings for conversation {conv}")
    embeddings = np.vstack([parquet_vectors[row_by_id[record.edu_id]] for record in records])
    return records, normalize_matrix(embeddings)


def edu_segment(record: EduRecord, score: float, origin: str) -> str:
    speakers = ", ".join(record.speakers)
    return (
        f"<EMEM_EDU source_session={record.session_index} time={record.date} "
        f"speakers={speakers} origin={origin} score={score:.6f}>\n"
        f"{record.text}\n</EMEM_EDU>\n\n"
    )


def source_key(record: EduRecord) -> tuple[int, int, str]:
    return record.session_index, record.within_session_index, record.edu_id


def pack_oracle(
    records: list[EduRecord],
    scores: np.ndarray,
    provenance_sessions: set[int],
    encoding: Any,
    budget: int,
) -> tuple[list[tuple[EduRecord, float, str]], str]:
    provenance = [
        index for index, record in enumerate(records) if record.session_index in provenance_sessions
    ]
    provenance_set = set(provenance)
    provenance.sort(key=lambda index: source_key(records[index]))
    global_dense = [index for index in range(len(records)) if index not in provenance_set]
    global_dense.sort(
        key=lambda index: (float(scores[index]), records[index].edu_id), reverse=True
    )

    chosen: list[tuple[EduRecord, float, str]] = []
    used = 0
    for index in provenance + global_dense:
        record = records[index]
        score = float(scores[index])
        origin = "raw_window_provenance" if index in provenance_set else "global_gold_dense"
        segment = edu_segment(record, score, origin)
        cost = len(encoding.encode(segment))
        if not chosen and cost > budget:
            truncated = encoding.decode(encoding.encode(segment)[:budget])
            return [(record, score, origin)], truncated
        if used + cost <= budget:
            chosen.append((record, score, origin))
            used += cost
        if used >= budget:
            break

    chosen.sort(key=lambda item: source_key(item[0]))
    context = "".join(edu_segment(record, score, origin) for record, score, origin in chosen)
    if len(encoding.encode(context)) > budget:
        raise AssertionError("Packed EMem context exceeds token budget")
    return chosen, context


def build(args: argparse.Namespace) -> None:
    native = args.native.resolve()
    mem0_native = args.mem0_native.resolve()
    mem0_oracle = args.mem0_oracle.resolve()
    gold_embeddings_path = args.gold_embeddings.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {output_dir}")

    native_searches = load_json(native / "search_results.json")
    mem0_searches = load_json(mem0_native / "search_results.json")
    gold_embeddings = normalize_matrix(
        np.asarray(load_json(gold_embeddings_path), dtype=np.float32)
    )
    if len(mem0_searches) != len(gold_embeddings):
        raise ValueError("Gold-query embeddings do not align with Mem0 questions")
    gold_by_question = {
        str(row["question_id"]): gold_embeddings[index]
        for index, row in enumerate(mem0_searches)
    }
    mapping_rows = load_json(mem0_oracle / "continuity_mapping_audit.json")
    mapping_by_question = {str(row["question_id"]): row for row in mapping_rows}

    native_by_conv: dict[int, list[dict[str, Any]]] = {}
    for row in native_searches:
        native_by_conv.setdefault(conversation_index(row["conversation_id"]), []).append(row)

    encoding = tiktoken.get_encoding("o200k_base")
    output_by_question: dict[str, dict[str, Any]] = {}
    audit_rows = []
    store_counts = {}
    for conv in sorted(native_by_conv):
        records, embeddings = load_emem_edu_store(native, conv)
        store_counts[str(conv)] = len(records)
        questions = native_by_conv[conv]
        query_matrix = np.vstack([gold_by_question[str(row["question_id"])] for row in questions])
        if embeddings.shape[1] != query_matrix.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch for conversation {conv}: "
                f"EDUs={embeddings.shape[1]}, queries={query_matrix.shape[1]}"
            )
        score_matrix = query_matrix @ embeddings.T

        for question_index, source in enumerate(questions):
            question_id = str(source["question_id"])
            provenance_sessions = {
                int(value)
                for value in mapping_by_question[question_id]["raw_source_sessions"]
            }
            selected, context = pack_oracle(
                records,
                score_matrix[question_index],
                provenance_sessions,
                encoding,
                args.token_budget,
            )
            context_tokens = len(encoding.encode(context))
            results = [
                {
                    "content": record.text,
                    "score": score,
                    "metadata": {
                        "source": "edu",
                        "edu_id": record.edu_id,
                        "source_session": record.session_index,
                        "within_session_index": record.within_session_index,
                        "date": record.date,
                        "source_speakers": list(record.speakers),
                        "oracle_origin": origin,
                    },
                }
                for record, score, origin in selected
            ]
            output_by_question[question_id] = {
                "question_id": question_id,
                "query": source["query"],
                "conversation_id": source["conversation_id"],
                "results": results,
                "retrieval_metadata": {
                    "system": "emem_continuity_memory_dense_oracle",
                    "formatted_context": context,
                    "context_token_budget": args.token_budget,
                    "context_tokens_after_cap": context_tokens,
                    "oracle_uses_gold_answer": True,
                    "oracle_complete_store_candidate_access": True,
                    "oracle_reranker_used": False,
                    "new_reranker_calls": 0,
                    "new_embedding_api_calls": 0,
                    "stored_native_emem_embeddings_reused": True,
                    "raw_control_reused": True,
                    "provenance_source_sessions": sorted(provenance_sessions),
                    "selected_edu_count": len(selected),
                    "selected_provenance_edu_count": sum(
                        origin == "raw_window_provenance" for _, _, origin in selected
                    ),
                },
            }
            available_provenance = sum(
                record.session_index in provenance_sessions for record in records
            )
            audit_rows.append(
                {
                    "question_id": question_id,
                    "conversation_index": conv,
                    "provenance_source_sessions": sorted(provenance_sessions),
                    "available_conversation_edus": len(records),
                    "available_provenance_edus": available_provenance,
                    "selected_edus": len(selected),
                    "selected_provenance_edus": sum(
                        origin == "raw_window_provenance" for _, _, origin in selected
                    ),
                    "context_tokens": context_tokens,
                }
            )
        print(f"Built EMem oracle contexts for conversation {conv}: {len(questions)} questions")

    output_rows = [output_by_question[str(row["question_id"])] for row in native_searches]
    if len(output_rows) != 500:
        raise AssertionError(f"Expected 500 questions, got {len(output_rows)}")
    output_dir.mkdir(parents=True)
    dump_json(output_dir / "search_results.json", output_rows)
    dump_json(output_dir / "continuity_mapping_audit.json", audit_rows)
    native_eval = load_json(native / "eval_results.json")
    summary = {
        "experiment": "EMem continuity recovery, gold-informed dense EDU oracle",
        "questions": len(output_rows),
        "native_emem_run": str(native),
        "native_emem_accuracy": native_eval.get("accuracy"),
        "stored_edus_by_conversation": store_counts,
        "stored_edus_total": sum(store_counts.values()),
        "gold_query_embedding_cache": str(gold_embeddings_path),
        "output_dir": str(output_dir),
        "token_budget": args.token_budget,
        "new_reranker_calls": 0,
        "new_embedding_api_calls": 0,
        "raw_control_reused": True,
        "context_tokens": {
            "min": min(row["context_tokens"] for row in audit_rows),
            "mean": sum(row["context_tokens"] for row in audit_rows) / len(audit_rows),
            "max": max(row["context_tokens"] for row in audit_rows),
        },
        "selected_edus": {
            "min": min(row["selected_edus"] for row in audit_rows),
            "mean": sum(row["selected_edus"] for row in audit_rows) / len(audit_rows),
            "max": max(row["selected_edus"] for row in audit_rows),
        },
    }
    dump_json(output_dir / "continuity_build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--mem0-native", type=Path, default=DEFAULT_MEM0_NATIVE)
    parser.add_argument("--mem0-oracle", type=Path, default=DEFAULT_MEM0_ORACLE)
    parser.add_argument("--gold-embeddings", type=Path, default=DEFAULT_GOLD_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--token-budget", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
