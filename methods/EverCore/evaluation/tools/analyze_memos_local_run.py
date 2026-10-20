"""
Diagnose a MemoryAgentBench memos_local run.

The EverMemOS diagnostic script audits MemCells, episodes, atomic facts, and
clusters. Memos exposes a different internal surface through the local product
API, so this tool audits the layers Memos actually returns:

- source coverage: benchmark chunks -> Memos memory source timestamps
- memory source fidelity: Memos memory text vs metadata.sources
- retrieval answerability: top retrieved memories vs question/gold answer

The tool is read-only for benchmark artifacts and Memos state. It writes a fresh
diagnostics directory under the selected run directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
SRC_DIR = EVERCORE_DIR / "src"
for candidate in (EVERCORE_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


KNOWN_MAB_SPLITS = (
    "Accurate_Retrieval",
    "Conflict_Resolution",
    "Long_Range_Understanding",
    "Test_Time_Learning",
)

MEMORY_SOURCE_LABELS = {
    "faithful",
    "omission",
    "hallucination",
    "over_compressed",
    "mixed_source",
    "unclear",
}
RETRIEVAL_LABELS = {
    "answerable",
    "partially_answerable",
    "unanswerable",
    "misleading",
    "unclear",
}


@dataclass(frozen=True)
class ChunkRecord:
    conv: int
    chunk: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class MemoryRecord:
    conv: int | None
    cube_id: str
    memory_id: str
    memory: str
    memory_type: str
    status: str
    created_at: str
    updated_at: str
    source_timestamps: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    source_text: str
    source_count: int
    raw: dict[str, Any]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def replace_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: replace_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_env_vars(item) for item in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(key, default)

    return re.sub(r"\$\{([^:}]+)(?::([^}]*))?\}", repl, value)


def load_yaml_light(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return replace_env_vars(yaml.safe_load(f) or {})


def resolve_path(path: Path, default_base: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (default_base / path).resolve()


def truncate_text(value: str, limit: int) -> str:
    value = value or ""
    if len(value) <= limit:
        return value
    half = max(1, (limit - 40) // 2)
    return value[:half] + "\n...[truncated]...\n" + value[-half:]


def parse_locomo_timestamp(value: str) -> datetime:
    normalized = " ".join(str(value).strip().split())
    normalized = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), normalized)
    return datetime.strptime(normalized, "%I:%M %p on %d %B, %Y").replace(
        tzinfo=timezone.utc
    )


def normalize_timestamp(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return ""
        if " on " in text and "," in text:
            dt = parse_locomo_timestamp(text)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or item.get("text") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        [
            key
            for key in conversation
            if key.startswith("session_") and not key.endswith("_date_time")
        ],
        key=lambda key: int(key.replace("session_", "")),
    )


def load_chunks(data_dir: Path) -> tuple[list[list[ChunkRecord]], dict[tuple[int, str], int]]:
    converted = load_json(data_dir / "memoryagentbench_locomo_style.json")
    chunks_by_conv: list[list[ChunkRecord]] = []
    timestamp_to_chunk: dict[tuple[int, str], int] = {}
    for conv_idx, item in enumerate(converted):
        conversation = item["conversation"]
        chunks: list[ChunkRecord] = []
        for chunk_idx, key in enumerate(session_keys(conversation)):
            timestamp = normalize_timestamp(conversation[f"{key}_date_time"])
            messages = conversation[key]
            text = "\n".join(str(msg.get("text", "")) for msg in messages)
            chunks.append(
                ChunkRecord(
                    conv=conv_idx,
                    chunk=chunk_idx,
                    timestamp=timestamp,
                    text=text,
                )
            )
            timestamp_to_chunk[(conv_idx, timestamp)] = chunk_idx
        chunks_by_conv.append(chunks)
    return chunks_by_conv, timestamp_to_chunk


def infer_split_source_run_name(run_dir: Path) -> tuple[str | None, str | None, str | None]:
    name = run_dir.name
    prefix = "memoryagentbench-memos_local-"
    if not name.startswith(prefix):
        return None, None, None
    tail = name[len(prefix) :]
    for split in KNOWN_MAB_SPLITS:
        split_prefix = f"{split}-"
        if not tail.startswith(split_prefix):
            continue
        remainder = tail[len(split_prefix) :]
        data_split_dir = EVALUATION_DIR / "data" / "memoryagentbench" / split
        sources = []
        if data_split_dir.exists():
            sources = sorted(
                [path.name for path in data_split_dir.iterdir() if path.is_dir()],
                key=len,
                reverse=True,
            )
        for source in sources:
            source_prefix = f"{source}-"
            if remainder == source:
                return split, source, None
            if remainder.startswith(source_prefix):
                return split, source, remainder[len(source_prefix) :]
        if "-" in remainder:
            source, run_name = remainder.rsplit("-", 1)
            return split, source, run_name
        return split, remainder, None
    return None, None, None


def conv_index_from_cube(cube_id: str) -> int | None:
    match = re.search(r"(?:^|__)context_(\d+)_", cube_id)
    if not match:
        match = re.search(r"context_(\d+)_", cube_id)
    return int(match.group(1)) if match else None


def build_cube_ids(
    run_dir: Path,
    source: str,
    run_name: str | None,
    chunks_by_conv: list[list[ChunkRecord]],
    search_results: list[dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    cube_ids: list[str] = []
    for item in search_results:
        metadata = item.get("retrieval_metadata") or {}
        for user_id in metadata.get("user_ids") or []:
            if user_id not in seen:
                seen.add(user_id)
                cube_ids.append(str(user_id))
        for result in item.get("results") or []:
            user_id = result.get("user_id")
            if user_id and user_id not in seen:
                seen.add(str(user_id))
                cube_ids.append(str(user_id))
    if cube_ids:
        return cube_ids

    prefix = run_name or run_dir.name.rsplit("-", 1)[-1]
    return [f"{prefix}__context_{idx}_{source}" for idx in range(len(chunks_by_conv))]


def http_json(
    url: str,
    payload: Any | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    retries: int = 3,
) -> Any:
    headers = {"Content-Type": "application/json", **(headers or {})}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if payload is None else "POST"
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"HTTP request failed for {url}: {last_error}")


def flatten_memos_memory_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    for group_key in ("text_mem", "pref_mem", "tool_mem", "skill_mem"):
        for bucket in data.get(group_key) or []:
            if not isinstance(bucket, dict):
                continue
            cube_id = str(bucket.get("cube_id") or "")
            for memory in bucket.get("memories") or []:
                if isinstance(memory, dict):
                    item = dict(memory)
                    item["_cube_id"] = cube_id
                    item["_group"] = group_key
                    memories.append(item)
    return memories


def export_memos_memories(api_url: str, cube_ids: list[str]) -> list[dict[str, Any]]:
    api_url = api_url.rstrip("/")
    exported: list[dict[str, Any]] = []
    for cube_id in cube_ids:
        payload = {
            "mem_cube_id": cube_id,
            "user_id": cube_id,
            "include_preference": True,
            "include_tool_memory": True,
            "include_skill_memory": True,
            "page": None,
            "page_size": None,
        }
        response = http_json(f"{api_url}/product/get_memory", payload)
        data = response.get("data") or {}
        exported.extend(flatten_memos_memory_response(data))
    deduped: dict[str, dict[str, Any]] = {}
    for item in exported:
        memory_id = str(item.get("id") or f"{item.get('_cube_id')}:{len(deduped)}")
        deduped[memory_id] = item
    return list(deduped.values())


def source_text_from_metadata(metadata: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    parts: list[str] = []
    timestamps: list[str] = []
    for source in metadata.get("sources") or []:
        if not isinstance(source, dict):
            continue
        role = source.get("role") or source.get("type") or "source"
        text = str(source.get("content") or "")
        timestamp = normalize_timestamp(source.get("chat_time")) if source.get("chat_time") else ""
        if timestamp:
            timestamps.append(timestamp)
        if text:
            prefix = f"[{timestamp}] {role}: " if timestamp else f"{role}: "
            parts.append(prefix + text)
    return "\n".join(parts), tuple(timestamps)


def build_memory_records(
    raw_memories: list[dict[str, Any]],
    timestamp_to_chunk: dict[tuple[int, str], int],
) -> list[MemoryRecord]:
    records: list[MemoryRecord] = []
    for item in raw_memories:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        cube_id = str(item.get("_cube_id") or metadata.get("user_id") or metadata.get("mem_cube_id") or "")
        conv = conv_index_from_cube(cube_id)
        source_text, source_timestamps = source_text_from_metadata(metadata)
        chunk_indices = []
        if conv is not None:
            for timestamp in source_timestamps:
                idx = timestamp_to_chunk.get((conv, timestamp))
                if idx is not None:
                    chunk_indices.append(idx)
        records.append(
            MemoryRecord(
                conv=conv,
                cube_id=cube_id,
                memory_id=str(item.get("id") or ""),
                memory=str(item.get("memory") or item.get("memory_value") or ""),
                memory_type=str(metadata.get("memory_type") or metadata.get("type") or item.get("_group") or ""),
                status=str(metadata.get("status") or ""),
                created_at=str(metadata.get("created_at") or ""),
                updated_at=str(metadata.get("updated_at") or ""),
                source_timestamps=tuple(source_timestamps),
                source_chunk_indices=tuple(sorted(set(chunk_indices))),
                source_text=source_text,
                source_count=len(metadata.get("sources") or []),
                raw=item,
            )
        )
    return records


def compact_memory_record(record: MemoryRecord) -> dict[str, Any]:
    return {
        "conv": record.conv,
        "cube_id": record.cube_id,
        "memory_id": record.memory_id,
        "memory": record.memory,
        "memory_type": record.memory_type,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "source_timestamps": list(record.source_timestamps),
        "source_chunk_indices": list(record.source_chunk_indices),
        "source_count": record.source_count,
        "source_text": record.source_text,
    }


def deterministic_summary(
    chunks_by_conv: list[list[ChunkRecord]],
    memory_records: list[MemoryRecord],
    search_results: list[dict[str, Any]],
    eval_results: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_by_conv = {idx: len(chunks) for idx, chunks in enumerate(chunks_by_conv)}
    covered_by_conv: dict[int, set[int]] = defaultdict(set)
    duplicate_hits_by_conv: dict[int, Counter[int]] = defaultdict(Counter)
    source_span_sizes = []
    source_counts = []
    for record in memory_records:
        source_counts.append(record.source_count)
        if record.source_chunk_indices:
            source_span_sizes.append(
                max(record.source_chunk_indices) - min(record.source_chunk_indices) + 1
            )
        if record.conv is None:
            continue
        for idx in record.source_chunk_indices:
            covered_by_conv[record.conv].add(idx)
            duplicate_hits_by_conv[record.conv][idx] += 1

    coverage_rows = []
    total_expected = sum(expected_by_conv.values())
    total_seen = 0
    duplicate_extra = 0
    for conv, expected in expected_by_conv.items():
        seen = len(covered_by_conv.get(conv, set()))
        total_seen += seen
        duplicate_extra += sum(
            max(0, count - 1) for count in duplicate_hits_by_conv.get(conv, {}).values()
        )
        missing = sorted(set(range(expected)) - covered_by_conv.get(conv, set()))
        coverage_rows.append(
            {
                "conv": conv,
                "expected": expected,
                "seen": seen,
                "missing": len(missing),
                "duplicate_extra_source_hits": sum(
                    max(0, count - 1)
                    for count in duplicate_hits_by_conv.get(conv, {}).values()
                ),
                "missing_ranges": ranges(missing),
            }
        )

    detail_results = (eval_results or {}).get("detailed_results") or []
    correct = sum(1 for row in detail_results if float(row.get("metric_value") or 0) > 0)
    total_questions = len(detail_results) or len(search_results)
    return {
        "counts": {
            "conversations": len(chunks_by_conv),
            "converted_chunks": total_expected,
            "memos_memories": len(memory_records),
            "memories_with_sources": sum(1 for record in memory_records if record.source_count),
            "covered_chunks": total_seen,
            "missing_chunks": total_expected - total_seen,
            "duplicate_extra_source_hits": duplicate_extra,
            "search_questions": len(search_results),
            "eval_questions": len(detail_results),
            "correct": correct if detail_results else None,
            "accuracy": (correct / total_questions) if detail_results and total_questions else None,
        },
        "coverage_by_conversation": coverage_rows,
        "source_count": describe_numbers(source_counts),
        "source_span_chunks": describe_numbers(source_span_sizes),
    }


def ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    result = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        result.append([start, prev])
        start = prev = value
    result.append([start, prev])
    return result


def describe_numbers(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    sorted_values = sorted(values)

    def percentile(p: float) -> int:
        idx = min(len(sorted_values) - 1, max(0, math.ceil(p * len(sorted_values)) - 1))
        return sorted_values[idx]

    return {
        "count": len(values),
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "max": sorted_values[-1],
        "mean": sum(values) / len(values),
    }


def eval_by_question(eval_results: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows = (eval_results or {}).get("detailed_results") or []
    return {str(row.get("question_id")): row for row in rows if row.get("question_id")}


def build_memory_source_samples(
    memory_records: list[MemoryRecord], sample_size: int, rng: random.Random
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in memory_records
        if record.memory and (record.source_text or record.source_timestamps)
    ]
    candidates.sort(
        key=lambda record: (
            len(record.source_chunk_indices),
            len(record.memory),
            record.memory_id,
        ),
        reverse=True,
    )
    anchors = candidates[: max(0, min(len(candidates), sample_size // 4))]
    anchor_ids = {record.memory_id for record in anchors}
    rest = [record for record in candidates if record.memory_id not in anchor_ids]
    sampled = anchors + rng.sample(rest, k=min(len(rest), max(0, sample_size - len(anchors))))
    rows = []
    for idx, record in enumerate(sampled[:sample_size]):
        rows.append(
            {
                "sample_id": f"memory_source_{idx:04d}",
                "type": "memory_source",
                "category": "memory_source",
                "conv": record.conv,
                "cube_id": record.cube_id,
                "memory_id": record.memory_id,
                "memory_type": record.memory_type,
                "source_count": record.source_count,
                "source_chunk_indices": list(record.source_chunk_indices),
                "source_timestamps": list(record.source_timestamps),
                "source_text": record.source_text,
                "memory": record.memory,
            }
        )
    return rows


def build_retrieval_samples(
    search_results: list[dict[str, Any]],
    eval_lookup: dict[str, dict[str, Any]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = []
    for item in search_results:
        question_id = str(item.get("question_id") or "")
        eval_row = eval_lookup.get(question_id, {})
        metric = eval_row.get("metric_value")
        is_correct = bool(metric is not None and float(metric or 0) > 0)
        candidates.append(
            {
                "item": item,
                "eval": eval_row,
                "is_correct": is_correct,
            }
        )

    wrong = [row for row in candidates if row["eval"] and not row["is_correct"]]
    correct = [row for row in candidates if row["is_correct"]]
    no_eval = [row for row in candidates if not row["eval"]]
    rng.shuffle(wrong)
    rng.shuffle(correct)
    rng.shuffle(no_eval)
    selected = wrong[: math.ceil(sample_size * 0.7)]
    remaining = sample_size - len(selected)
    selected.extend(correct[: max(0, remaining // 2)])
    remaining = sample_size - len(selected)
    selected.extend(no_eval[:remaining])
    remaining = sample_size - len(selected)
    if remaining > 0:
        leftovers = [
            row
            for row in candidates
            if id(row) not in {id(selected_row) for selected_row in selected}
        ]
        selected.extend(leftovers[:remaining])

    rows = []
    for idx, row in enumerate(selected[:sample_size]):
        item = row["item"]
        eval_row = row["eval"]
        contexts = []
        for result_idx, result in enumerate((item.get("results") or [])[:10], start=1):
            metadata = result.get("metadata") or {}
            contexts.append(
                {
                    "rank": result_idx,
                    "memory_id": metadata.get("memory_id"),
                    "score": result.get("score"),
                    "created_at": metadata.get("created_at"),
                    "content": result.get("content") or "",
                    "source_timestamps": [
                        str(source.get("chat_time"))
                        for source in (
                            ((metadata.get("raw") or {}).get("metadata") or {}).get("sources")
                            or []
                        )
                        if isinstance(source, dict) and source.get("chat_time")
                    ],
                }
            )
        rows.append(
            {
                "sample_id": f"retrieval_answerability_{idx:04d}",
                "type": "retrieval_answerability",
                "category": "retrieval_answerability",
                "question_id": item.get("question_id"),
                "conversation_id": item.get("conversation_id"),
                "query": item.get("query"),
                "gold_answer": eval_row.get("golden_answer") or item.get("golden_answer"),
                "generated_answer": eval_row.get("generated_answer"),
                "metric_value": eval_row.get("metric_value"),
                "is_correct": row["is_correct"],
                "top_contexts": contexts,
            }
        )
    return rows


def build_samples(
    memory_records: list[MemoryRecord],
    search_results: list[dict[str, Any]],
    eval_results: dict[str, Any] | None,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    eval_lookup = eval_by_question(eval_results)
    memory_samples = build_memory_source_samples(memory_records, sample_size, rng)
    retrieval_samples = (
        build_retrieval_samples(search_results, eval_lookup, sample_size, rng)
        if search_results
        else []
    )
    samples = memory_samples + retrieval_samples
    return samples, {
        "sample_counts": dict(Counter(sample["category"] for sample in samples)),
        "seed": seed,
        "sample_size_per_category": sample_size,
    }


def build_prompt(sample: dict[str, Any]) -> str:
    category = sample["category"]
    if category == "memory_source":
        return f"""
You are auditing a Memos local memory object for source fidelity.

Compare the source messages with the stored memory. Decide whether the memory
faithfully preserves the source information needed for later QA.

Labels:
- faithful: supported by source, no important distortion.
- omission: source has important concrete details missing from memory.
- hallucination: memory adds unsupported facts.
- over_compressed: memory is supported but too compressed to preserve useful QA details.
- mixed_source: memory merges unrelated source messages or topics in a confusing way.
- unclear: cannot judge.

Return strict JSON:
{{
  "label": "faithful|omission|hallucination|over_compressed|mixed_source|unclear",
  "is_issue": true,
  "confidence": 0.0,
  "rationale": "brief reason"
}}

Source messages:
{truncate_text(sample.get('source_text', ''), 5000)}

Stored memory:
{truncate_text(sample.get('memory', ''), 3000)}
""".strip()

    if category == "retrieval_answerability":
        contexts = []
        for ctx in sample.get("top_contexts") or []:
            contexts.append(
                f"[{ctx.get('rank')}] memory_id={ctx.get('memory_id')} "
                f"score={ctx.get('score')}\n{truncate_text(ctx.get('content', ''), 1200)}"
            )
        return f"""
You are auditing retrieved Memos memories for answerability.

Question:
{sample.get('query', '')}

Gold answer:
{sample.get('gold_answer', '')}

Generated answer:
{sample.get('generated_answer', '')}

Top retrieved memories:
{chr(10).join(contexts)}

Decide whether the retrieved memories contain enough evidence to answer the
question with the gold answer.

Labels:
- answerable: retrieved memories directly contain enough evidence.
- partially_answerable: some evidence is present but key details are missing.
- unanswerable: retrieved memories do not contain the answer evidence.
- misleading: retrieved memories point toward a wrong answer or stale/conflicting answer.
- unclear: cannot judge.

Return strict JSON:
{{
  "label": "answerable|partially_answerable|unanswerable|misleading|unclear",
  "is_issue": true,
  "confidence": 0.0,
  "rationale": "brief reason"
}}
""".strip()

    raise ValueError(f"Unsupported sample category: {category}")


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))


def normalize_judgment(category: str, raw: dict[str, Any]) -> dict[str, Any]:
    labels = MEMORY_SOURCE_LABELS if category == "memory_source" else RETRIEVAL_LABELS
    label = str(raw.get("label") or "unclear")
    if label not in labels:
        label = "unclear"
    if category == "memory_source":
        is_issue = label != "faithful"
    else:
        is_issue = label not in {"answerable", "unclear"}
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "label": label,
        "is_issue": is_issue,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": str(raw.get("rationale") or ""),
    }


async def judge_sample(
    sample: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    retries: int,
) -> dict[str, Any]:
    prompt = build_prompt(sample)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful diagnostic judge. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = ""
    for attempt in range(retries):
        try:
            response = await asyncio.to_thread(
                http_json,
                url,
                payload,
                headers=headers,
                timeout=180,
                retries=1,
            )
            content = response["choices"][0]["message"]["content"]
            raw_judgment = extract_json_object(content)
            return {
                "sample_id": sample["sample_id"],
                "category": sample["category"],
                "ok": True,
                "judgment": normalize_judgment(sample["category"], raw_judgment),
                "raw_response": content,
            }
        except Exception as exc:  # noqa: BLE001 - diagnostic tool records failures.
            last_error = str(exc)
            if attempt + 1 < retries:
                await asyncio.sleep(2**attempt)
    return {
        "sample_id": sample["sample_id"],
        "category": sample["category"],
        "ok": False,
        "error": last_error,
    }


async def judge_samples(
    samples: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    retries: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async def run(sample: dict[str, Any]) -> None:
        async with semaphore:
            result = await judge_sample(
                sample,
                base_url=base_url,
                api_key=api_key,
                model=model,
                retries=retries,
            )
            results.append(result)
            print(f"Judged {len(results)}/{len(samples)} samples", flush=True)

    await asyncio.gather(*(run(sample) for sample in samples))
    results.sort(key=lambda row: row["sample_id"])
    return results


def summarize_judgments(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result.get("category", "unknown")].append(result)

    summary = {}
    for category, rows in sorted(by_category.items()):
        ok_rows = [row for row in rows if row.get("ok")]
        labels = Counter(row["judgment"]["label"] for row in ok_rows)
        issues = sum(1 for row in ok_rows if row["judgment"].get("is_issue"))
        total = len(ok_rows)
        summary[category] = {
            "total": total,
            "failed": len(rows) - total,
            "issues": issues,
            "issue_rate": issues / total if total else None,
            "labels": dict(labels),
        }
    return summary


def representative_examples(
    samples: list[dict[str, Any]], results: list[dict[str, Any]], limit: int = 5
) -> dict[str, list[dict[str, Any]]]:
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if not result.get("ok") or not result.get("judgment", {}).get("is_issue"):
            continue
        category = result["category"]
        if len(examples[category]) >= limit:
            continue
        sample = sample_by_id.get(result["sample_id"], {})
        item = {
            "sample_id": result["sample_id"],
            "label": result["judgment"]["label"],
            "rationale": result["judgment"].get("rationale", ""),
        }
        if category == "memory_source":
            item.update(
                {
                    "conv": sample.get("conv"),
                    "memory_id": sample.get("memory_id"),
                    "source_chunk_indices": sample.get("source_chunk_indices"),
                }
            )
        if category == "retrieval_answerability":
            item.update(
                {
                    "question_id": sample.get("question_id"),
                    "metric_value": sample.get("metric_value"),
                    "gold_answer": sample.get("gold_answer"),
                }
            )
        examples[category].append(item)
    return dict(examples)


def render_summary_md(
    *,
    output_dir: Path,
    run_dir: Path,
    data_dir: Path,
    deterministic: dict[str, Any],
    sample_meta: dict[str, Any],
    judgment_summary: dict[str, Any] | None,
    examples: dict[str, Any] | None,
    llm_model: str | None,
    llm_base_url: str | None,
) -> str:
    counts = deterministic["counts"]
    lines = [
        "# MemoryAgentBench Memos Local Diagnostic Summary",
        "",
        f"Output directory: `{output_dir}`",
        f"Run directory: `{run_dir}`",
        f"Data directory: `{data_dir}`",
        "",
        "## Deterministic Checks",
        "",
        f"- Conversations: {counts['conversations']}",
        f"- Converted chunks: {counts['converted_chunks']}",
        f"- Memos memories exported: {counts['memos_memories']}",
        f"- Memories with sources: {counts['memories_with_sources']}",
        f"- Covered converted chunks: {counts['covered_chunks']}/{counts['converted_chunks']}",
        f"- Missing converted chunks: {counts['missing_chunks']}/{counts['converted_chunks']}",
        f"- Duplicate extra source hits: {counts['duplicate_extra_source_hits']}",
        f"- Search questions: {counts['search_questions']}",
        f"- Eval questions: {counts['eval_questions']}",
    ]
    if counts.get("accuracy") is not None:
        lines.append(
            f"- Answer accuracy: {counts['correct']}/{counts['eval_questions']} "
            f"({counts['accuracy']:.2%})"
        )
    lines.extend(
        [
            "",
            "### Coverage By Conversation",
            "",
            "| conv | expected | seen | missing | duplicate extra source hits | missing ranges |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in deterministic["coverage_by_conversation"]:
        lines.append(
            f"| {row['conv']} | {row['expected']} | {row['seen']} | "
            f"{row['missing']} | {row['duplicate_extra_source_hits']} | "
            f"`{row['missing_ranges']}` |"
        )

    lines.extend(
        [
            "",
            "## Samples",
            "",
        ]
    )
    for category, count in sorted((sample_meta.get("sample_counts") or {}).items()):
        lines.append(f"- {category}: {count}")

    if judgment_summary is None:
        lines.extend(
            [
                "",
                "## LLM Audit",
                "",
                "Not run. Re-run without `--dry-run` / `--skip-llm` to audit samples.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## LLM Audit",
                "",
                f"- Model: `{llm_model}`",
                f"- Base URL: `{llm_base_url}`",
                "",
                "| category | total | issues | issue rate | failed | labels |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for category, item in sorted(judgment_summary.items()):
            rate = item["issue_rate"]
            rate_text = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(
                f"| {category} | {item['total']} | {item['issues']} | "
                f"{rate_text} | {item['failed']} | `{item['labels']}` |"
            )

        if examples:
            lines.extend(["", "## Representative Issue Examples", ""])
            for category, rows in sorted(examples.items()):
                lines.extend([f"### {category}", ""])
                for row in rows:
                    lines.append(
                        f"- `{row['sample_id']}` `{row['label']}`: "
                        f"{row.get('rationale', '')} ({row})"
                    )
                lines.append("")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `memory_source` audits Memos memory text against `metadata.sources`.",
            "- `retrieval_answerability` audits whether retrieved memories contain gold-answer evidence.",
            "- Missing source coverage means a converted benchmark chunk was not found in any exported memory source timestamp.",
            "- This tool does not inspect Docker files; it uses the local MemOS product API only.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--system-config",
        type=Path,
        default=EVALUATION_DIR / "config" / "systems" / "memos_local.yaml",
    )
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--run-name", default=None, help="Memos user-id prefix. Inferred from run dir when omitted.")
    parser.add_argument("--mab-split", default=None)
    parser.add_argument("--mab-source", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--judge-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument(
        "--memory-export-path",
        type=Path,
        help="Reuse a previously exported memory_export.json instead of calling MemOS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(EVERCORE_DIR / ".env")

    run_dir = args.run_dir
    run_dir = resolve_path(run_dir, EVALUATION_DIR)

    inferred_split, inferred_source, inferred_run_name = infer_split_source_run_name(run_dir)
    split = args.mab_split or inferred_split
    source = args.mab_source or inferred_source
    run_name = args.run_name or inferred_run_name
    if not split or not source:
        raise SystemExit(
            "Could not infer MemoryAgentBench split/source from run dir. "
            "Pass --mab-split and --mab-source."
        )

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = EVALUATION_DIR / "data" / "memoryagentbench" / split / source
    else:
        data_dir = resolve_path(data_dir, EVALUATION_DIR)

    config = load_yaml_light(args.system_config)
    api_url = (args.api_url or config.get("api_url") or "http://localhost:8000").rstrip("/")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "dry_run" if args.dry_run or args.skip_llm else "llm_audit"
    output_dir = run_dir / f"memos_diagnostics_{timestamp}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)

    chunks_by_conv, timestamp_to_chunk = load_chunks(data_dir)
    search_path = run_dir / "search_results.json"
    answer_path = run_dir / "answer_results.json"
    eval_path = run_dir / "eval_results.json"
    search_results = load_json(search_path) if search_path.exists() else []
    eval_results = load_json(eval_path) if eval_path.exists() else None

    cube_ids = build_cube_ids(run_dir, source, run_name, chunks_by_conv, search_results)
    memory_export_path = output_dir / "memory_export.json"
    if args.memory_export_path:
        source_export = resolve_path(args.memory_export_path, Path.cwd())
        raw_memories = load_json(source_export)
        dump_json(memory_export_path, raw_memories)
    else:
        raw_memories = export_memos_memories(api_url, cube_ids)
        dump_json(memory_export_path, raw_memories)

    memory_records = build_memory_records(raw_memories, timestamp_to_chunk)
    deterministic = deterministic_summary(
        chunks_by_conv=chunks_by_conv,
        memory_records=memory_records,
        search_results=search_results,
        eval_results=eval_results,
    )
    samples, sample_meta = build_samples(
        memory_records=memory_records,
        search_results=search_results,
        eval_results=eval_results,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    dump_json(output_dir / "memory_records.json", [compact_memory_record(record) for record in memory_records])
    dump_json(output_dir / "samples.json", samples)
    write_jsonl(output_dir / "samples.jsonl", samples)

    judgment_summary = None
    examples = None
    judge_base_url = args.judge_base_url or os.environ.get("DEEPSEEK_BASE_URL")
    judge_model = args.judge_model
    if not args.dry_run and not args.skip_llm:
        judge_api_key = os.environ.get(args.judge_api_key_env)
        if not judge_api_key:
            raise SystemExit(f"Missing judge API key env var: {args.judge_api_key_env}")
        if not judge_base_url:
            raise SystemExit("Missing judge base URL. Set DEEPSEEK_BASE_URL or pass --judge-base-url.")
        results = asyncio.run(
            judge_samples(
                samples,
                base_url=judge_base_url,
                api_key=judge_api_key,
                model=judge_model,
                concurrency=args.judge_concurrency,
                retries=args.judge_retries,
            )
        )
        dump_json(output_dir / "judge_results.json", results)
        write_jsonl(output_dir / "judge_results.jsonl", results)
        judgment_summary = summarize_judgments(results)
        examples = representative_examples(samples, results)

    summary = {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "split": split,
        "source": source,
        "run_name": run_name,
        "api_url": api_url,
        "cube_ids": cube_ids,
        "files": {
            "search_results": str(search_path) if search_path.exists() else None,
            "answer_results": str(answer_path) if answer_path.exists() else None,
            "eval_results": str(eval_path) if eval_path.exists() else None,
        },
        "deterministic": deterministic,
        "samples": sample_meta,
        "judgments": judgment_summary,
        "examples": examples,
        "judge": {
            "model": judge_model if judgment_summary is not None else None,
            "base_url": judge_base_url if judgment_summary is not None else None,
            "api_key_env": args.judge_api_key_env if judgment_summary is not None else None,
        },
    }
    dump_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_summary_md(
            output_dir=output_dir,
            run_dir=run_dir,
            data_dir=data_dir,
            deterministic=deterministic,
            sample_meta=sample_meta,
            judgment_summary=judgment_summary,
            examples=examples,
            llm_model=judge_model,
            llm_base_url=judge_base_url,
        ),
        encoding="utf-8",
    )

    print(f"Wrote diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
