"""
Diagnose a MemoryAgentBench EverMemOS run with deterministic checks and optional
LLM-audited samples.

This tool is intentionally read-only for benchmark artifacts. It writes a fresh
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


DEFAULT_RUN_NAME = "memoryagentbench-evermemos-Accurate_Retrieval-eventqa_full-normal-new-converter"
DEFAULT_DATA_REL = Path("data/memoryagentbench/Accurate_Retrieval/eventqa_full")
DEFAULT_RUN_REL = Path("results") / DEFAULT_RUN_NAME

BOUNDARY_LABELS = {
    "correct_split",
    "false_split",
    "correct_merge",
    "missed_split",
    "unclear",
}
EPISODE_LABELS = {
    "faithful",
    "omission",
    "hallucination",
    "irrelevant",
    "over_compressed",
    "unclear",
}
ATOMIC_LABELS = {
    "accurate",
    "hallucinated",
    "unsupported_inference",
    "too_fine",
    "too_coarse",
    "missing_major_fact",
    "redundant",
    "unclear",
}
CLUSTER_LABELS = {"same_cluster_correct", "mixed_cluster", "unclear"}


@dataclass(frozen=True)
class ChunkRecord:
    conv: int
    chunk: int
    timestamp: str
    text: str


@dataclass(frozen=True)
class MemCellRecord:
    conv: int
    index: int
    event_id: str
    timestamp: str
    subject: str
    summary: str
    episode: str
    original_text: str
    original_timestamps: tuple[str, ...]
    original_chunk_indices: tuple[int, ...]
    atomic_facts: tuple[str, ...]
    cluster_id: str | None


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def dump_json(path: Path, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


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
        if " on " in text and "," in text:
            dt = parse_locomo_timestamp(text)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "+00:00")


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


def original_item_to_message(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        msg = item.get("message", item)
        if isinstance(msg, dict):
            return msg
    return {}


def memcell_original_text(memcell: dict[str, Any]) -> str:
    parts = []
    for item in memcell.get("original_data") or []:
        msg = original_item_to_message(item)
        text = content_to_text(msg.get("content"))
        if text:
            ts = msg.get("timestamp", "")
            parts.append(f"[{ts}] {msg.get('sender_name', 'Unknown')}: {text}")
    return "\n\n".join(parts)


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


def load_cluster_assignments(run_dir: Path, conv: int) -> dict[str, str]:
    path = run_dir / "memcells" / "clusters" / f"conv_{conv}" / f"cluster_map_conv_{conv}.json"
    if not path.exists():
        return {}
    data = load_json(path)
    return data.get("assignments", {}) or {}


def resolve_memcell_run_dir(run_dir: Path) -> Path:
    """Resolve MemCell artifacts for both native and enriched-context runs."""
    direct = run_dir / "memcells"
    if direct.is_dir():
        return run_dir

    manifest_path = run_dir / "enriched_context_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        source = manifest.get("source_run_dir")
        if source:
            source_dir = Path(str(source)).expanduser()
            if not source_dir.is_absolute():
                source_dir = (run_dir / source_dir).resolve()
            if (source_dir / "memcells").is_dir():
                return source_dir

    raise FileNotFoundError(
        f"MemCell artifacts not found under {run_dir}. Expected a memcells directory "
        "or an enriched_context_manifest.json with a valid source_run_dir."
    )


def load_memcells(
    run_dir: Path, chunks_by_conv: list[list[ChunkRecord]], timestamp_to_chunk: dict[tuple[int, str], int]
) -> list[list[MemCellRecord]]:
    all_records: list[list[MemCellRecord]] = []
    for conv_idx, _chunks in enumerate(chunks_by_conv):
        cluster_map = load_cluster_assignments(run_dir, conv_idx)
        raw_memcells = load_json(run_dir / "memcells" / f"memcell_list_conv_{conv_idx}.json")
        records: list[MemCellRecord] = []
        for mem_idx, memcell in enumerate(raw_memcells):
            timestamps: list[str] = []
            chunk_indices: list[int] = []
            for item in memcell.get("original_data") or []:
                msg = original_item_to_message(item)
                timestamp = normalize_timestamp(msg.get("timestamp")) if msg.get("timestamp") else ""
                if timestamp:
                    timestamps.append(timestamp)
                    if (conv_idx, timestamp) in timestamp_to_chunk:
                        chunk_indices.append(timestamp_to_chunk[(conv_idx, timestamp)])
            atomic = memcell.get("atomic_fact") or {}
            facts = atomic.get("atomic_fact") if isinstance(atomic, dict) else []
            records.append(
                MemCellRecord(
                    conv=conv_idx,
                    index=mem_idx,
                    event_id=str(memcell.get("event_id") or ""),
                    timestamp=normalize_timestamp(memcell.get("timestamp"))
                    if memcell.get("timestamp")
                    else "",
                    subject=str(memcell.get("subject") or ""),
                    summary=str(memcell.get("summary") or ""),
                    episode=str(memcell.get("episode") or ""),
                    original_text=memcell_original_text(memcell),
                    original_timestamps=tuple(timestamps),
                    original_chunk_indices=tuple(chunk_indices),
                    atomic_facts=tuple(str(fact) for fact in facts or []),
                    cluster_id=cluster_map.get(str(memcell.get("event_id") or "")),
                )
            )
        all_records.append(records)
    return all_records


def ranges(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    values = sorted(values)
    out: list[list[int]] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
        else:
            out.append([start, prev])
            start = prev = value
    out.append([start, prev])
    return out


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return {"low": (centre - margin) / denom, "high": (centre + margin) / denom}


def deterministic_summary(
    run_dir: Path, chunks_by_conv: list[list[ChunkRecord]], memcells_by_conv: list[list[MemCellRecord]]
) -> dict[str, Any]:
    answer_results = load_json(run_dir / "answer_results.json")
    search_results = load_json(run_dir / "search_results.json")
    eval_results = load_json(run_dir / "eval_results.json")

    coverage = []
    total_expected = total_seen = total_missing = total_duplicates = 0
    for conv_idx, chunks in enumerate(chunks_by_conv):
        expected_indices = set(range(len(chunks)))
        seen_indices: list[int] = []
        for memcell in memcells_by_conv[conv_idx]:
            seen_indices.extend(memcell.original_chunk_indices)
        counts = Counter(seen_indices)
        missing = sorted(expected_indices - set(seen_indices))
        duplicates = sorted(idx for idx, count in counts.items() if count > 1)
        total_expected += len(chunks)
        total_seen += len(seen_indices)
        total_missing += len(missing)
        total_duplicates += sum(counts[idx] - 1 for idx in duplicates)
        coverage.append(
            {
                "conv": conv_idx,
                "expected_chunks": len(chunks),
                "memcells": len(memcells_by_conv[conv_idx]),
                "seen_original_items": len(seen_indices),
                "unique_seen_chunks": len(counts),
                "missing_count": len(missing),
                "missing_ranges": ranges(missing),
                "duplicate_extra_count": sum(counts[idx] - 1 for idx in duplicates),
                "duplicate_indices": duplicates,
            }
        )

    facts = [len(memcell.atomic_facts) for conv in memcells_by_conv for memcell in conv]
    memcell_sizes = [
        len(memcell.original_chunk_indices) for conv in memcells_by_conv for memcell in conv
    ]
    return {
        "counts": {
            "conversations": len(chunks_by_conv),
            "converted_chunks": total_expected,
            "memcells": sum(len(conv) for conv in memcells_by_conv),
            "memcell_original_items": total_seen,
            "search_results": len(search_results),
            "answer_results": len(answer_results),
            "eval_total_questions": eval_results.get("total_questions"),
            "eval_correct": eval_results.get("correct"),
            "eval_accuracy": eval_results.get("accuracy"),
            "atomic_facts": sum(facts),
            "memcells_with_atomic_facts": sum(1 for value in facts if value > 0),
        },
        "coverage": {
            "total_expected_chunks": total_expected,
            "total_seen_original_items": total_seen,
            "total_missing_chunks": total_missing,
            "total_duplicate_extra_items": total_duplicates,
            "by_conversation": coverage,
        },
        "memcell_size": {
            "min": min(memcell_sizes) if memcell_sizes else 0,
            "max": max(memcell_sizes) if memcell_sizes else 0,
            "avg": sum(memcell_sizes) / len(memcell_sizes) if memcell_sizes else 0,
        },
        "atomic_fact_count": {
            "min": min(facts) if facts else 0,
            "max": max(facts) if facts else 0,
            "avg": sum(facts) / len(facts) if facts else 0,
        },
    }


def stratified_sample(
    items: list[dict[str, Any]], sample_size: int, rng: random.Random, key: str = "conv"
) -> list[dict[str, Any]]:
    if len(items) <= sample_size:
        return items[:]
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[item.get(key)].append(item)
    selected: list[dict[str, Any]] = []
    for group_items in groups.values():
        quota = max(1, round(sample_size * len(group_items) / len(items)))
        selected.extend(rng.sample(group_items, min(quota, len(group_items))))
    if len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)
    remaining = [item for item in items if item not in selected]
    while len(selected) < sample_size and remaining:
        item = rng.choice(remaining)
        remaining.remove(item)
        selected.append(item)
    return selected


def build_boundary_candidates(
    chunks_by_conv: list[list[ChunkRecord]], memcells_by_conv: list[list[MemCellRecord]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for conv_idx, memcells in enumerate(memcells_by_conv):
        chunks = chunks_by_conv[conv_idx]
        owners: dict[int, list[MemCellRecord]] = defaultdict(list)
        for memcell in memcells:
            for chunk_idx in set(memcell.original_chunk_indices):
                if 0 <= chunk_idx < len(chunks):
                    owners[chunk_idx].append(memcell)

        # A boundary decision is only defined between consecutive source chunks.
        # Deriving it from adjacent MemCell records can compare distant chunks when
        # MemCells are not stored chronologically or contain non-contiguous sources.
        for left_idx in range(len(chunks) - 1):
            right_idx = left_idx + 1
            left_owners = owners.get(left_idx, [])
            right_owners = owners.get(right_idx, [])
            if len(left_owners) != 1 or len(right_owners) != 1:
                continue
            left_memcell = left_owners[0]
            right_memcell = right_owners[0]
            merged = left_memcell.index == right_memcell.index
            cross_day = chunks[left_idx].timestamp[:10] != chunks[right_idx].timestamp[:10]
            candidate = {
                "type": "boundary",
                "category": "boundary",
                "conv": conv_idx,
                "left_memcell": left_memcell.index,
                "right_memcell": right_memcell.index,
                "left_chunk": left_idx,
                "right_chunk": right_idx,
                "left_timestamp": chunks[left_idx].timestamp,
                "right_timestamp": chunks[right_idx].timestamp,
                "observed_decision": "merged" if merged else "split",
                "cross_day": cross_day,
                "left_text": chunks[left_idx].text,
                "right_text": chunks[right_idx].text,
            }
            if merged:
                candidate.update(
                    {
                        "memcell": left_memcell.index,
                        "memcell_subject": left_memcell.subject,
                    }
                )
            else:
                candidate["memcell_subjects"] = [
                    left_memcell.subject,
                    right_memcell.subject,
                ]
            candidates.append(candidate)
    return candidates


def build_episode_candidates(memcells_by_conv: list[list[MemCellRecord]]) -> list[dict[str, Any]]:
    out = []
    for conv in memcells_by_conv:
        for memcell in conv:
            out.append(
                {
                    "type": "episode",
                    "category": "episode",
                    "conv": memcell.conv,
                    "memcell": memcell.index,
                    "event_id": memcell.event_id,
                    "size": len(memcell.original_chunk_indices),
                    "original_text": memcell.original_text,
                    "subject": memcell.subject,
                    "summary": memcell.summary,
                    "episode": memcell.episode,
                }
            )
    return out


def build_atomic_candidates(memcells_by_conv: list[list[MemCellRecord]]) -> list[dict[str, Any]]:
    out = []
    for conv in memcells_by_conv:
        for memcell in conv:
            if not memcell.atomic_facts:
                continue
            out.append(
                {
                    "type": "atomic",
                    "category": "atomic",
                    "conv": memcell.conv,
                    "memcell": memcell.index,
                    "event_id": memcell.event_id,
                    "size": len(memcell.original_chunk_indices),
                    "fact_count": len(memcell.atomic_facts),
                    "original_text": memcell.original_text,
                    "episode": memcell.episode,
                    "atomic_facts": list(memcell.atomic_facts),
                }
            )
    return out


def build_cluster_candidates(memcells_by_conv: list[list[MemCellRecord]], rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for conv in memcells_by_conv:
        by_cluster: dict[str, list[MemCellRecord]] = defaultdict(list)
        for memcell in conv:
            if memcell.cluster_id:
                by_cluster[memcell.cluster_id].append(memcell)
        for cluster_id, members in by_cluster.items():
            if len(members) < 2:
                continue
            # Generate a bounded set of random same-cluster pairs per cluster.
            pair_budget = min(20, len(members) * 2)
            seen = set()
            attempts = 0
            while len(seen) < pair_budget and attempts < pair_budget * 10:
                attempts += 1
                left, right = rng.sample(members, 2)
                pair = tuple(sorted((left.index, right.index)))
                if pair in seen:
                    continue
                seen.add(pair)
                out.append(
                    {
                        "type": "cluster",
                        "category": "cluster",
                        "conv": left.conv,
                        "cluster_id": cluster_id,
                        "left_memcell": left.index,
                        "right_memcell": right.index,
                        "left_subject": left.subject,
                        "right_subject": right.subject,
                        "left_episode": left.episode,
                        "right_episode": right.episode,
                    }
                )
    return out


def jaccard_tokens(text_a: str, text_b: str) -> float:
    def toks(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower()))

    a = toks(text_a)
    b = toks(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_cluster_recall_probe(memcells_by_conv: list[list[MemCellRecord]], limit: int = 50) -> list[dict[str, Any]]:
    probes = []
    for conv in memcells_by_conv:
        for left, right in zip(conv, conv[1:]):
            if left.cluster_id and right.cluster_id and left.cluster_id != right.cluster_id:
                probes.append(
                    {
                        "conv": left.conv,
                        "left_memcell": left.index,
                        "right_memcell": right.index,
                        "left_cluster": left.cluster_id,
                        "right_cluster": right.cluster_id,
                        "similarity": jaccard_tokens(left.episode, right.episode),
                        "left_subject": left.subject,
                        "right_subject": right.subject,
                    }
                )
    return sorted(probes, key=lambda item: item["similarity"], reverse=True)[:limit]


def build_samples(
    chunks_by_conv: list[list[ChunkRecord]],
    memcells_by_conv: list[list[MemCellRecord]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    boundary = stratified_sample(build_boundary_candidates(chunks_by_conv, memcells_by_conv), sample_size, rng)
    episode = stratified_sample(build_episode_candidates(memcells_by_conv), sample_size, rng)
    atomic_candidates = build_atomic_candidates(memcells_by_conv)
    if len(atomic_candidates) <= sample_size:
        atomic = atomic_candidates[:]
    else:
        # Ensure high/low fact-count representation before regular stratified fill.
        atomic_sorted = sorted(atomic_candidates, key=lambda item: item["fact_count"])
        anchor_quota = max(1, sample_size // 5)
        anchor_map = {
            (item["conv"], item["memcell"]): item
            for item in (atomic_sorted[:anchor_quota] + atomic_sorted[-anchor_quota:])
        }
        anchors = list(anchor_map.values())
        atomic_rest = [
            item
            for item in atomic_candidates
            if (item["conv"], item["memcell"]) not in anchor_map
        ]
        atomic = anchors + stratified_sample(
            atomic_rest, max(0, sample_size - len(anchors)), rng
        )
        atomic = atomic[:sample_size]
    cluster = stratified_sample(build_cluster_candidates(memcells_by_conv, rng), sample_size, rng)

    samples = []
    for category, rows in [
        ("boundary", boundary),
        ("episode", episode),
        ("atomic", atomic),
        ("cluster", cluster),
    ]:
        for idx, row in enumerate(rows):
            row = dict(row)
            row["sample_id"] = f"{category}_{idx:04d}"
            samples.append(row)

    meta = {
        "sample_size_requested": sample_size,
        "sample_counts": Counter(sample["category"] for sample in samples),
        "cluster_recall_probe": build_cluster_recall_probe(memcells_by_conv),
    }
    meta["sample_counts"] = dict(meta["sample_counts"])
    return samples, meta


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("No valid JSON object found")


def judge_prompt(sample: dict[str, Any]) -> tuple[str, set[str]]:
    category = sample["category"]
    if category == "boundary":
        observed_decision = sample["observed_decision"]
        if observed_decision == "split":
            allowed_labels = {"correct_split", "false_split"}
            label_instruction = (
                '- Return "correct_split" if a boundary is justified; otherwise return '
                '"false_split".'
            )
            label_schema = "correct_split|false_split"
        elif observed_decision == "merged":
            allowed_labels = {"correct_merge", "missed_split"}
            label_instruction = (
                '- Return "correct_merge" if the chunks form one coherent episode; otherwise '
                'return "missed_split".'
            )
            label_schema = "correct_merge|missed_split"
        else:
            raise ValueError(f"Unsupported boundary decision: {observed_decision}")
        prompt = f"""
You are auditing episodic-memory boundary detection for a converted MemoryAgentBench context.

Observed decision: {observed_decision}
Cross-day synthetic timestamp boundary: {sample['cross_day']}
Left timestamp: {sample.get('left_timestamp', '')}
Right timestamp: {sample.get('right_timestamp', '')}

Decision rule:
{label_instruction}

Judge only the boundary between these two consecutive source chunks. Use a high
threshold for splitting: a paragraph break, change of speaker, point-of-view shift,
or small focus change inside a continuous narrative is not enough by itself. A split
requires a clear discontinuity in time, place, participants, or core event. Cross-day
splits are expected by the production prompt.

LEFT CHUNK:
{truncate_text(sample['left_text'], 3500)}

RIGHT CHUNK:
{truncate_text(sample['right_text'], 3500)}

Return only JSON:
{{
  "label": "{label_schema}",
  "is_issue": true/false,
  "confidence": 0.0,
  "rationale": "short reason"
}}
"""
        return prompt, allowed_labels
    if category == "episode":
        prompt = f"""
You are auditing whether an extracted episodic memory is faithful to its source text.

SOURCE TEXT:
{sample['original_text']}

EXTRACTED SUBJECT:
{sample['subject']}

EXTRACTED SUMMARY:
{sample['summary']}

EXTRACTED EPISODE:
{sample['episode']}

Return only JSON:
{{
  "label": "faithful|omission|hallucination|irrelevant|over_compressed|unclear",
  "is_issue": true/false,
  "confidence": 0.0,
  "rationale": "short reason"
}}
"""
        return prompt, EPISODE_LABELS
    if category == "atomic":
        facts_text = "\n".join(
            f"{idx + 1}. {fact}" for idx, fact in enumerate(sample["atomic_facts"][:80])
        )
        prompt = f"""
You are auditing atomic facts extracted from source text. The SOURCE TEXT is
the authoritative evidence. The EPISODE is an auxiliary, potentially lossy
summary and is not the ground truth.

SOURCE TEXT:
{sample['original_text']}

EPISODE:
{sample['episode']}

ATOMIC FACTS:
{facts_text}

Evaluate every atomic fact directly against SOURCE TEXT. A fact supported by
SOURCE TEXT is accurate even when it is absent from EPISODE. Do not label a fact
hallucinated merely because the EPISODE omitted it. Classify the atomic-fact set
as a whole, not the quality of the EPISODE. Use missing_major_fact at the
memcell level when important source information is absent from all facts.
Fact-level counts must sum to the number of listed atomic facts; because
missing_major_fact describes an absent fact, its fact-level count must be zero.

Return only JSON:
{{
  "label": "accurate|hallucinated|unsupported_inference|too_fine|too_coarse|missing_major_fact|redundant|unclear",
  "is_issue": true/false,
  "confidence": 0.0,
  "fact_label_counts": {{
    "accurate": 0,
    "hallucinated": 0,
    "unsupported_inference": 0,
    "too_fine": 0,
    "too_coarse": 0,
    "missing_major_fact": 0,
    "redundant": 0,
    "unclear": 0
  }},
  "rationale": "short reason"
}}
"""
        return prompt, ATOMIC_LABELS
    if category == "cluster":
        prompt = f"""
You are auditing whether two episodic memories belong in the same narrative/topic cluster.

LEFT SUBJECT:
{sample['left_subject']}

LEFT EPISODE:
{truncate_text(sample['left_episode'], 3000)}

RIGHT SUBJECT:
{sample['right_subject']}

RIGHT EPISODE:
{truncate_text(sample['right_episode'], 3000)}

Return only JSON:
{{
  "label": "same_cluster_correct|mixed_cluster|unclear",
  "is_issue": true/false,
  "confidence": 0.0,
  "rationale": "short reason"
}}
"""
        return prompt, CLUSTER_LABELS
    raise ValueError(f"Unsupported sample category: {category}")


def label_is_issue(category: str, label: str) -> bool:
    if category == "boundary":
        return label not in {"correct_split", "correct_merge"}
    if category == "episode":
        return label != "faithful"
    if category == "atomic":
        return label != "accurate"
    if category == "cluster":
        return label != "same_cluster_correct"
    return True


async def judge_one(
    sample: dict[str, Any],
    llm: Any,
    retries: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    prompt, allowed_labels = judge_prompt(sample)
    async with semaphore:
        last_error = ""
        raw = ""
        for attempt in range(1, retries + 1):
            try:
                raw = await llm.generate(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                parsed = extract_json_object(raw)
                label = str(parsed.get("label", "unclear"))
                if label not in allowed_labels:
                    label = "unclear"
                if sample["category"] == "atomic":
                    fact_counts = parsed.get("fact_label_counts") or {}
                    normalized_counts = {}
                    for fact_label in ATOMIC_LABELS:
                        try:
                            count = int(fact_counts.get(fact_label, 0))
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"Invalid fact count for {fact_label}"
                            ) from exc
                        if count < 0:
                            raise ValueError(f"Negative fact count for {fact_label}")
                        normalized_counts[fact_label] = count
                    expected_count = len(sample["atomic_facts"][:80])
                    actual_count = sum(normalized_counts.values())
                    if actual_count != expected_count:
                        raise ValueError(
                            f"Fact label counts sum to {actual_count}, expected "
                            f"{expected_count}"
                        )
                    parsed["fact_label_counts"] = normalized_counts
                parsed["label"] = label
                parsed["is_issue"] = label_is_issue(sample["category"], label)
                result = {
                    "sample_id": sample["sample_id"],
                    "category": sample["category"],
                    "ok": True,
                    "judgment": parsed,
                    "raw_response": raw,
                }
                if sample["category"] == "boundary":
                    result["observed_decision"] = sample["observed_decision"]
                return result
            except Exception as exc:  # noqa: BLE001 - preserve audit errors in output
                last_error = f"{type(exc).__name__}: {exc}"
        result = {
            "sample_id": sample["sample_id"],
            "category": sample["category"],
            "ok": False,
            "error": last_error,
            "raw_response": raw,
            "judgment": {"label": "unclear", "is_issue": True, "confidence": 0, "rationale": last_error},
        }
        if sample["category"] == "boundary":
            result["observed_decision"] = sample["observed_decision"]
        return result


async def run_judging(
    samples: list[dict[str, Any]],
    output_path: Path,
    llm_config: dict[str, Any],
    concurrency: int,
    retries: int,
    existing_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    from memory_layer.llm.llm_provider import LLMProvider

    llm = LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=0,
        max_tokens=llm_config.get("max_tokens", 4096),
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results = list(existing_results or [])
    completed_ids = {str(row.get("sample_id")) for row in results}
    pending_samples = [
        sample for sample in samples if sample["sample_id"] not in completed_ids
    ]
    initial_count = len(results)

    async def run_and_record(sample: dict[str, Any]) -> None:
        result = await judge_one(
            sample,
            llm,
            retries,
            int(llm_config.get("max_tokens", 4096)),
            semaphore,
        )
        append_jsonl(output_path, [result])
        results.append(result)
        if len(results) % 25 == 0:
            print(f"Judged {len(results)}/{len(samples)} samples")

    if initial_count:
        print(
            f"Resuming with {initial_count}/{len(samples)} completed samples; "
            f"{len(pending_samples)} remaining"
        )
    await asyncio.gather(*(run_and_record(sample) for sample in pending_samples))
    return sorted(results, key=lambda row: row["sample_id"])


def summarize_judgments(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result)

    summary: dict[str, Any] = {}
    for category, rows in by_category.items():
        labels = Counter(str(row.get("judgment", {}).get("label", "unclear")) for row in rows)
        issues = sum(
            1
            for row in rows
            if label_is_issue(
                category, str(row.get("judgment", {}).get("label", "unclear"))
            )
        )
        item = {
            "total": len(rows),
            "parse_failures": sum(1 for row in rows if not row.get("ok")),
            "issues": issues,
            "issue_rate": issues / len(rows) if rows else 0,
            "issue_rate_wilson_95": wilson_interval(issues, len(rows)),
            "labels": dict(labels),
        }
        if category == "atomic":
            fact_counts: Counter[str] = Counter()
            for row in rows:
                counts = row.get("judgment", {}).get("fact_label_counts") or {}
                for label, count in counts.items():
                    try:
                        fact_counts[label] += int(count)
                    except (TypeError, ValueError):
                        pass
            total_fact_labels = sum(fact_counts.values())
            item["fact_label_counts"] = dict(fact_counts)
            item["fact_issue_rate"] = (
                (total_fact_labels - fact_counts.get("accurate", 0)) / total_fact_labels
                if total_fact_labels
                else 0
            )
        if category == "boundary":
            decision_summary: dict[str, Any] = {}
            for decision in ("split", "merged"):
                decision_rows = [
                    row for row in rows if row.get("observed_decision") == decision
                ]
                decision_labels = Counter(
                    str(row.get("judgment", {}).get("label", "unclear"))
                    for row in decision_rows
                )
                decision_issues = sum(
                    count
                    for label, count in decision_labels.items()
                    if label_is_issue(category, label)
                )
                decision_summary[decision] = {
                    "total": len(decision_rows),
                    "issues": decision_issues,
                    "issue_rate": (
                        decision_issues / len(decision_rows) if decision_rows else 0
                    ),
                    "labels": dict(decision_labels),
                }
            item["by_observed_decision"] = decision_summary
        summary[category] = item
    return summary


def representative_examples(
    samples: list[dict[str, Any]], results: list[dict[str, Any]], limit: int = 5
) -> dict[str, list[dict[str, Any]]]:
    by_id = {sample["sample_id"]: sample for sample in samples}
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        judgment = result.get("judgment", {})
        if not label_is_issue(result["category"], str(judgment.get("label", "unclear"))):
            continue
        category = result["category"]
        if len(out[category]) >= limit:
            continue
        sample = by_id.get(result["sample_id"], {})
        item = {
            "sample_id": result["sample_id"],
            "label": judgment.get("label"),
            "rationale": judgment.get("rationale"),
            "conv": sample.get("conv"),
            "memcell": sample.get("memcell"),
        }
        if category == "boundary":
            item.update(
                {
                    "observed_decision": sample.get("observed_decision"),
                    "left_chunk": sample.get("left_chunk"),
                    "right_chunk": sample.get("right_chunk"),
                    "cross_day": sample.get("cross_day"),
                }
            )
        if category == "cluster":
            item.update(
                {
                    "cluster_id": sample.get("cluster_id"),
                    "left_memcell": sample.get("left_memcell"),
                    "right_memcell": sample.get("right_memcell"),
                }
            )
        out[category].append(item)
    return dict(out)


def write_markdown(
    path: Path,
    deterministic: dict[str, Any],
    sample_meta: dict[str, Any],
    judgment_summary: dict[str, Any] | None,
    examples: dict[str, list[dict[str, Any]]] | None,
    output_dir: Path,
    judge_config: dict[str, Any] | None = None,
) -> None:
    counts = deterministic["counts"]
    coverage = deterministic["coverage"]
    lines = [
        "# MemoryAgentBench EverMemOS Diagnostic Summary",
        "",
        f"Output directory: `{output_dir}`",
        "",
        "## Deterministic Checks",
        "",
        f"- Conversations: {counts['conversations']}",
        f"- Converted chunks: {counts['converted_chunks']}",
        f"- MemCells: {counts['memcells']}",
        f"- MemCell original items: {counts['memcell_original_items']}",
        f"- Atomic facts: {counts['atomic_facts']}",
        f"- Answer accuracy: {counts['eval_correct']}/{counts['eval_total_questions']} ({float(counts['eval_accuracy']) * 100:.2f}%)",
        f"- Missing converted chunks: {coverage['total_missing_chunks']}/{coverage['total_expected_chunks']}",
        f"- Duplicate extra original items: {coverage['total_duplicate_extra_items']}",
        "",
        "### Coverage By Conversation",
        "",
        "| conv | expected | seen | missing | duplicate extra | missing ranges |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in coverage["by_conversation"]:
        lines.append(
            f"| {row['conv']} | {row['expected_chunks']} | {row['seen_original_items']} | "
            f"{row['missing_count']} | {row['duplicate_extra_count']} | `{row['missing_ranges']}` |"
        )

    lines.extend(["", "## Samples", ""])
    for category, count in sample_meta["sample_counts"].items():
        lines.append(f"- {category}: {count}")

    if judgment_summary is None:
        lines.extend(
            [
                "",
                "## LLM Audit",
                "",
                "Not run. Use the same command without `--dry-run` / `--skip-llm` to audit samples.",
            ]
        )
    else:
        lines.extend(["", "## LLM Audit", ""])
        if judge_config:
            lines.extend(
                [
                    f"- Provider: `{judge_config.get('provider')}`",
                    f"- Model: `{judge_config.get('model')}`",
                    f"- Base URL: `{judge_config.get('base_url')}`",
                    f"- Max output tokens: {judge_config.get('max_tokens')}",
                    "",
                ]
            )
        lines.append("| category | total | issues | issue rate | 95% CI | labels |")
        lines.append("|---|---:|---:|---:|---|---|")
        for category, item in sorted(judgment_summary.items()):
            ci = item["issue_rate_wilson_95"]
            lines.append(
                f"| {category} | {item['total']} | {item['issues']} | "
                f"{item['issue_rate'] * 100:.1f}% | "
                f"{ci['low'] * 100:.1f}%-{ci['high'] * 100:.1f}% | `{item['labels']}` |"
            )
            if category == "atomic" and "fact_label_counts" in item:
                lines.append(
                    f"| atomic fact-level | {sum(item['fact_label_counts'].values())} |  | "
                    f"{item['fact_issue_rate'] * 100:.1f}% |  | `{item['fact_label_counts']}` |"
                )
            if category == "boundary":
                for decision, decision_item in item.get(
                    "by_observed_decision", {}
                ).items():
                    lines.append(
                        f"| boundary/{decision} | {decision_item['total']} | "
                        f"{decision_item['issues']} | "
                        f"{decision_item['issue_rate'] * 100:.1f}% |  | "
                        f"`{decision_item['labels']}` |"
                    )

        lines.extend(["", "## Representative Issue Examples", ""])
        for category, rows in sorted((examples or {}).items()):
            lines.append(f"### {category}")
            if not rows:
                lines.append("- No issue examples captured.")
            for row in rows:
                lines.append(f"- `{row['sample_id']}` `{row['label']}`: {row['rationale']} ({row})")
            lines.append("")

    recall_probe = sample_meta.get("cluster_recall_probe") or []
    lines.extend(["", "## Exploratory Cluster Recall Probe", ""])
    lines.append("Top adjacent different-cluster pairs by lexical Jaccard similarity:")
    for row in recall_probe[:10]:
        lines.append(
            f"- conv {row['conv']} memcells {row['left_memcell']}->{row['right_memcell']} "
            f"clusters {row['left_cluster']}->{row['right_cluster']}, sim={row['similarity']:.3f}: "
            f"{row['left_subject']} / {row['right_subject']}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_output_dir(run_dir: Path, dry_run: bool) -> Path:
    suffix = "dry_run" if dry_run else "llm_audit"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / f"diagnostics_{timestamp}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=EVALUATION_DIR / DEFAULT_RUN_REL)
    parser.add_argument("--data-dir", type=Path, default=EVALUATION_DIR / DEFAULT_DATA_REL)
    parser.add_argument("--system-config", type=Path, default=EVALUATION_DIR / "config/systems/evermemos.yaml")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--dry-run", action="store_true", help="Only build deterministic summary and samples.")
    parser.add_argument("--skip-llm", action="store_true", help="Build samples but do not call the judge LLM.")
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--judge-provider")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--judge-base-url-env")
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--resume-output-dir",
        type=Path,
        help="Resume an interrupted audit from an existing diagnostics directory.",
    )
    parser.add_argument(
        "--rerun-category",
        action="append",
        choices=("boundary", "episode", "atomic", "cluster"),
        default=[],
        help="With --resume-output-dir, discard and rejudge this category.",
    )
    parser.add_argument(
        "--rerun-sample-id",
        action="append",
        default=[],
        help="With --resume-output-dir, discard and rejudge this sample ID.",
    )
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

    if args.resume_output_dir:
        output_dir = args.resume_output_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory not found: {output_dir}")
        if args.dry_run or args.skip_llm:
            raise ValueError("--resume-output-dir cannot be combined with no-LLM modes")
    else:
        output_dir = create_output_dir(run_dir, dry_run=args.dry_run or args.skip_llm)
    if (args.rerun_category or args.rerun_sample_id) and not args.resume_output_dir:
        raise ValueError("Rerun options require --resume-output-dir")
    chunks_by_conv, timestamp_to_chunk = load_chunks(data_dir)
    memcell_run_dir = resolve_memcell_run_dir(run_dir)
    memcells_by_conv = load_memcells(memcell_run_dir, chunks_by_conv, timestamp_to_chunk)
    deterministic = deterministic_summary(run_dir, chunks_by_conv, memcells_by_conv)
    samples, sample_meta = build_samples(
        chunks_by_conv=chunks_by_conv,
        memcells_by_conv=memcells_by_conv,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    if args.resume_output_dir:
        existing_samples = load_json(output_dir / "samples.json")
        if existing_samples != samples:
            raise ValueError(
                "Resume samples do not match the requested run/sample-size/seed"
            )
    else:
        dump_json(output_dir / "samples.json", samples)
        append_jsonl(output_dir / "samples.jsonl", samples)

    results: list[dict[str, Any]] | None = None
    judgment_summary = None
    examples = None
    judge_config_summary = None
    if not args.dry_run and not args.skip_llm:
        system_config = load_yaml_light(args.system_config)
        llm_config = dict(system_config.get("llm", {}))
        if args.judge_provider:
            llm_config["provider"] = args.judge_provider
        if args.judge_model:
            llm_config["model"] = args.judge_model
        if args.judge_api_key_env:
            llm_config["api_key"] = os.environ.get(args.judge_api_key_env, "")
        if args.judge_base_url_env:
            llm_config["base_url"] = os.environ.get(args.judge_base_url_env, "")
        if args.judge_max_tokens is not None:
            llm_config["max_tokens"] = args.judge_max_tokens
        if not llm_config.get("api_key") or not llm_config.get("base_url"):
            raise ValueError("Judge API key and base URL must be configured")
        judge_config_summary = {
            "provider": llm_config.get("provider"),
            "model": llm_config.get("model"),
            "base_url": llm_config.get("base_url"),
            "max_tokens": llm_config.get("max_tokens", 4096),
        }
        existing_results = (
            load_jsonl(output_dir / "judge_results.jsonl")
            if args.resume_output_dir
            else []
        )
        if args.rerun_category:
            rerun_categories = set(args.rerun_category)
            existing_results = [
                row
                for row in existing_results
                if row.get("category") not in rerun_categories
            ]
        if args.rerun_sample_id:
            known_sample_ids = {sample["sample_id"] for sample in samples}
            unknown_ids = set(args.rerun_sample_id) - known_sample_ids
            if unknown_ids:
                raise ValueError(f"Unknown rerun sample IDs: {sorted(unknown_ids)}")
            rerun_sample_ids = set(args.rerun_sample_id)
            existing_results = [
                row
                for row in existing_results
                if row.get("sample_id") not in rerun_sample_ids
            ]
        if args.rerun_category or args.rerun_sample_id:
            write_jsonl(output_dir / "judge_results.jsonl", existing_results)
        duplicate_ids = [
            sample_id
            for sample_id, count in Counter(
                str(row.get("sample_id")) for row in existing_results
            ).items()
            if count > 1
        ]
        if duplicate_ids:
            raise ValueError(f"Duplicate resumed sample IDs: {duplicate_ids[:10]}")
        results = await run_judging(
            samples=samples,
            output_path=output_dir / "judge_results.jsonl",
            llm_config=llm_config,
            concurrency=args.judge_concurrency,
            retries=args.judge_retries,
            existing_results=existing_results,
        )
        dump_json(output_dir / "judge_results.json", results)
        judgment_summary = summarize_judgments(results)
        examples = representative_examples(samples, results)
    else:
        # Create an empty file so downstream tooling can rely on its presence.
        (output_dir / "judge_results.jsonl").write_text("", encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "memcell_run_dir": str(memcell_run_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "deterministic": deterministic,
        "samples": sample_meta,
        "judgments": judgment_summary,
        "examples": examples,
        "judge": judge_config_summary,
        "mode": "dry_run" if args.dry_run else ("skip_llm" if args.skip_llm else "llm_audit"),
    }
    dump_json(output_dir / "summary.json", summary)
    write_markdown(
        output_dir / "summary.md",
        deterministic=deterministic,
        sample_meta=sample_meta,
        judgment_summary=judgment_summary,
        examples=examples,
        output_dir=output_dir,
        judge_config=judge_config_summary,
    )

    print(f"Diagnostics written to: {output_dir}")
    print(
        "Counts: "
        + json.dumps(deterministic["counts"], ensure_ascii=False, sort_keys=True)
    )
    print(
        "Coverage: "
        + json.dumps(deterministic["coverage"], ensure_ascii=False, sort_keys=True)
    )
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
