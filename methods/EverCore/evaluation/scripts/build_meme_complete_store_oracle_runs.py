"""Build phase-safe MEME oracles from complete persisted memory stores.

This complements ``build_meme_continuity_oracle_runs.py``.  The older script
uses the union of objects observed in native search results; this script reads
the backend's persisted store and therefore does not depend on native recall.

Backends are added and validated one at a time.  Mem0 Local is currently
supported through its persisted Qdrant ``mem0_local`` collection.  The entity
graph collection is intentionally excluded because native memory search
returns memory objects, not graph nodes as reader evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import pickle
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import tiktoken

try:
    from .build_meme_continuity_oracle_runs import (
        DEFAULT_DATA,
        MemoryObject,
        QuestionSpec,
        RESULTS_DIR,
        add_readout_policy,
        base_row,
        coverage_signals,
        create_output,
        evidence_budget,
        flatten_strings,
        load_json,
        normalized_text,
        pack_memory_context,
        question_specs,
        rank_memory_objects,
    )
except ImportError:  # Direct script execution from evaluation/scripts.
    from build_meme_continuity_oracle_runs import (
        DEFAULT_DATA,
        MemoryObject,
        QuestionSpec,
        RESULTS_DIR,
        add_readout_policy,
        base_row,
        coverage_signals,
        create_output,
        evidence_budget,
        flatten_strings,
        load_json,
        normalized_text,
        pack_memory_context,
        question_specs,
        rank_memory_objects,
    )


DEFAULT_MEM0_RUN = RESULTS_DIR / "meme-mem0_local-normal"
DEFAULT_MEMOS_RUN = RESULTS_DIR / "meme-memos_local-fine"
DEFAULT_MEMU_RUN = RESULTS_DIR / "meme-memu_local-normal"
DEFAULT_TAG = "meme-continuity-complete-store-task-readout-full-v1-20260717"
DEFAULT_MEMOS_API_URL = "http://localhost:8000"
DEFAULT_MEMOS_PREFIX = "fine"
DEFAULT_MEMU_DATABASE_URL = os.environ.get(
    "MEMU_DATABASE_DSN",
    "postgresql://memu:memu@127.0.0.1:5433/memu",
).replace("postgresql+psycopg://", "postgresql://", 1)
DEFAULT_EMEM_RUN = RESULTS_DIR / "meme-emem-normal"
DEFAULT_EVERMEMOS_RUN = RESULTS_DIR / "meme-evermemos-normal-3"
DEFAULT_RAG_RUN = RESULTS_DIR / "meme-rag-normal-15"
DEFAULT_MEMMACHINE_API_URL = "http://localhost:8080/api/v2"
DEFAULT_MEMMACHINE_ORG_ID = "ragforlocomo"
DEFAULT_MEMMACHINE_PROJECT_PREFIX = "eval"
DEFAULT_MEMMACHINE_RUN_NAME = "normal"
SPECIAL_TOKEN_RE = re.compile(r"<\|([^|<>]+)\|>")
MEMU_RESOURCE_RE = re.compile(
    r"^(?P<conversation>meme_\d+)-part-(?P<part>\d{4})(?:_#segment_\d+)?\.json$"
)


def canonical_timestamp(value: Any) -> str:
    """Normalize ISO-like timestamps to a timezone-independent wall time."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # Raw MEME sessions use ``2023/03/01 (Wed) 09:00`` while local memory
        # backends persist the same wall time in ISO form.
        without_weekday = re.sub(r"\s+\([^)]+\)", "", text)
        try:
            parsed = datetime.strptime(without_weekday, "%Y/%m/%d %H:%M")
        except ValueError:
            # Keep a normalized fallback so validation reports the mismatch.
            return " ".join(text.lower().split())
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.replace(microsecond=0).isoformat()


def session_timestamp_maps(
    episodes: list[dict[str, Any]],
) -> dict[str, list[tuple[str, int]]]:
    result: dict[str, list[tuple[str, int]]] = {}
    for episode_index, episode in enumerate(episodes):
        mapping: dict[str, int] = {}
        for session_index, session in enumerate(episode.get("sessions") or []):
            timestamp = canonical_timestamp(session.get("timestamp"))
            if not timestamp:
                continue
            if timestamp in mapping:
                raise ValueError(
                    f"Duplicate session timestamp in meme_{episode_index}: {timestamp}"
                )
            mapping[timestamp] = session_index
        result[f"meme_{episode_index}"] = sorted(
            (timestamp, index) for timestamp, index in mapping.items()
        )
    return result


def source_session_for_timestamp(
    boundaries: list[tuple[str, int]], timestamp: str
) -> int | None:
    """Map a per-message/batch timestamp to the containing MEME session."""
    if not boundaries or not timestamp:
        return None
    position = bisect.bisect_right(
        [boundary for boundary, _index in boundaries], timestamp
    ) - 1
    return boundaries[position][1] if position >= 0 else None


def iter_pretty_json_object_items(path: Path, object_key: str) -> Iterator[tuple[str, Any]]:
    """Stream members of a pretty-printed top-level JSON object.

    RAG indexes persist embeddings and can approach a gigabyte.  The evaluation
    writer uses ``json.dump(..., indent=2)``, so parsing one conversation member
    at a time avoids retaining every embedding vector in memory.
    """
    object_header = f'  "{object_key}": {{'
    member_re = re.compile(r'^    "((?:[^"\\]|\\.)+)": (\{.*)$')
    inside = False
    member_name: str | None = None
    member_lines: list[str] = []
    depth = 0
    in_string = False
    escaped = False

    def update_depth(text: str) -> None:
        nonlocal depth, in_string, escaped
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\r\n")
            if not inside:
                if stripped == object_header:
                    inside = True
                continue
            if member_name is None:
                if stripped == "  }" or stripped == "  },":
                    return
                match = member_re.match(stripped)
                if not match:
                    continue
                member_name = json.loads(f'"{match.group(1)}"')
                member_lines = [match.group(2)]
                depth = 0
                in_string = False
                escaped = False
                update_depth(match.group(2))
            else:
                member_lines.append(stripped)
                update_depth(stripped)

            if member_name is not None and depth == 0:
                raw = "\n".join(member_lines).rstrip().removesuffix(",")
                yield member_name, json.loads(raw)
                member_name = None
                member_lines = []

    if not inside:
        raise ValueError(f"Top-level object {object_key!r} not found in {path}")
    if member_name is not None:
        raise ValueError(f"Truncated JSON member {member_name!r} in {path}")


def rag_index_settings(path: Path) -> dict[str, Any]:
    """Read the small metadata object without loading the conversation index."""
    # Current indexes place the small scalar metadata mapping before the large
    # conversations object. Extract only the settings needed for audit.
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(16_384)
    result: dict[str, Any] = {}
    for key in ("chunk_size", "chunk_overlap"):
        match = re.search(rf'"{key}"\s*:\s*(-?\d+)', prefix)
        if match:
            result[key] = int(match.group(1))
    return result


def load_rag_store(
    run_dir: Path,
    timestamp_maps: dict[str, list[tuple[str, int]]],
    conversation_ids: list[str],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    """Load complete persisted RAG chunks while discarding stored embeddings."""
    index_path = run_dir / "rag_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"RAG index not found: {index_path}")
    wanted = set(conversation_ids)
    store: dict[str, list[MemoryObject]] = {}
    conversation_audit: dict[str, Any] = {}
    total_chunks = 0
    cross_session_chunks = 0
    unmapped: list[dict[str, Any]] = []

    for conversation_id, conversation in iter_pretty_json_object_items(
        index_path, "conversations"
    ):
        if conversation_id not in wanted:
            continue
        turn_context = conversation.get("turn_context") or {}
        turn_timestamps = list(turn_context.get("turn_timestamps") or [])
        boundaries = timestamp_maps.get(conversation_id, [])
        objects: list[MemoryObject] = []
        for fallback_index, chunk in enumerate(conversation.get("chunks") or []):
            content = str(chunk.get("content") or "").strip()
            start_turn = int(chunk.get("start_turn", 0))
            end_turn = int(chunk.get("end_turn", start_turn))
            covered_timestamps = turn_timestamps[start_turn:end_turn]
            source_sessions = sorted(
                {
                    session
                    for timestamp in covered_timestamps
                    if (
                        session := source_session_for_timestamp(
                            boundaries, canonical_timestamp(timestamp)
                        )
                    )
                    is not None
                }
            )
            if content and covered_timestamps and not source_sessions:
                unmapped.append(
                    {
                        "conversation_id": conversation_id,
                        "chunk_index": chunk.get("chunk_index", fallback_index),
                        "start_turn": start_turn,
                        "end_turn": end_turn,
                    }
                )
                continue
            if not content or not source_sessions:
                continue
            chunk_index = int(chunk.get("chunk_index", fallback_index))
            metadata = {
                "source_backend": "rag_index_chunk",
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk_index,
                "start_token": int(chunk.get("start_token", 0)),
                "end_token": int(chunk.get("end_token", 0)),
                "start_turn": start_turn,
                "end_turn": end_turn,
                "source_session_indices": source_sessions,
            }
            objects.append(
                MemoryObject(
                    content=content,
                    score=0.0,
                    metadata=metadata,
                    first_seen=chunk_index,
                )
            )
            total_chunks += 1
            cross_session_chunks += len(source_sessions) > 1
        objects.sort(key=lambda item: item.first_seen)
        store[conversation_id] = objects
        conversation_audit[conversation_id] = {
            "chunks": len(objects),
            "turns": len(turn_timestamps),
        }
        if set(store) == wanted:
            break

    missing = sorted(wanted - set(store))
    if missing:
        raise ValueError(f"RAG index is missing conversations: {missing[:5]}")
    if unmapped:
        raise ValueError(
            f"RAG chunk provenance validation failed for {len(unmapped)} chunks. "
            f"Sample: {json.dumps(unmapped[:3], ensure_ascii=False)}"
        )
    settings = rag_index_settings(index_path)
    return store, {
        "backend": "rag",
        "run_dir": str(run_dir.resolve()),
        "index_path": str(index_path.resolve()),
        "conversation_count": len(store),
        "total_objects": total_chunks,
        "chunk_size": settings.get("chunk_size"),
        "chunk_overlap": settings.get("chunk_overlap"),
        "cross_session_chunks": cross_session_chunks,
        "candidate_scope": "all persisted fixed-token chunks in rag_index.json",
        "phase_mapping": "chunk turn range -> turn timestamp -> MEME session",
        "reader_exposure": "complete persisted chunk text; chunks are never cropped",
        "conversations": conversation_audit,
    }


def qdrant_locked(error: Exception) -> bool:
    message = str(error).lower()
    return "already accessed by another instance" in message or "lock" in message


@contextmanager
def open_qdrant_readonly(run_dir: Path) -> Iterator[Any]:
    """Open QdrantLocal, snapshotting it when another process holds its lock."""
    from qdrant_client import QdrantClient

    source = run_dir / "mem0_qdrant"
    if not source.is_dir():
        raise FileNotFoundError(f"Mem0 Qdrant directory not found: {source}")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    client = None
    try:
        try:
            client = QdrantClient(path=str(source))
            # QdrantLocal initializes lazily; force the lock/schema check here.
            client.get_collections()
        except Exception as error:
            if not qdrant_locked(error):
                raise
            if client is not None:
                client.close()
            temp_dir = tempfile.TemporaryDirectory(prefix="meme_mem0_qdrant_read_")
            snapshot = Path(temp_dir.name) / "qdrant"
            shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns(".lock"))
            client = QdrantClient(path=str(snapshot))
        yield client
    finally:
        if client is not None:
            client.close()
        if temp_dir is not None:
            temp_dir.cleanup()


def choose_mem0_collection(client: Any) -> str:
    names = [item.name for item in client.get_collections().collections]
    if "mem0_local" in names:
        return "mem0_local"
    candidates = [name for name in names if not name.endswith("_entities")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Cannot infer Mem0 memory collection from {names}")


def load_mem0_store(
    run_dir: Path,
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    """Load every Mem0 memory object and map it to its source MEME session."""
    by_conversation: dict[str, list[MemoryObject]] = defaultdict(list)
    unmapped: list[dict[str, str]] = []
    total = 0
    collection = ""

    with open_qdrant_readonly(run_dir) as client:
        collection = choose_mem0_collection(client)
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                total += 1
                payload = dict(point.payload or {})
                conversation_id = str(
                    payload.get("conversation_id") or payload.get("run_id") or ""
                )
                content = str(payload.get("data") or payload.get("memory") or "").strip()
                timestamp = canonical_timestamp(
                    payload.get("created_at") or payload.get("timestamp")
                )
                source_session = source_session_for_timestamp(
                    timestamp_maps.get(conversation_id, []), timestamp
                )
                if not conversation_id or not content or source_session is None:
                    unmapped.append(
                        {
                            "memory_id": str(point.id),
                            "conversation_id": conversation_id,
                            "timestamp": timestamp,
                            "reason": (
                                "missing conversation/content"
                                if not conversation_id or not content
                                else "timestamp not present in MEME sessions"
                            ),
                        }
                    )
                    continue
                metadata = {
                    "memory_id": str(point.id),
                    "source_backend": "mem0_local_qdrant",
                    "source_collection": collection,
                    "source_session_indices": [source_session],
                    "source_timestamp": timestamp,
                    "store_payload": payload,
                }
                by_conversation[conversation_id].append(
                    MemoryObject(
                        content=content,
                        score=0.0,
                        metadata=metadata,
                        first_seen=total - 1,
                    )
                )
            if offset is None:
                break

    if unmapped:
        sample = json.dumps(unmapped[:3], ensure_ascii=False)
        raise ValueError(
            f"Mem0 complete-store validation failed: {len(unmapped)}/{total} "
            f"objects cannot be mapped phase-safely. Sample: {sample}"
        )
    expected = set(timestamp_maps)
    missing_conversations = sorted(expected - set(by_conversation))
    if missing_conversations:
        raise ValueError(
            "Mem0 store has no memory objects for MEME conversations: "
            + ", ".join(missing_conversations)
        )
    return dict(by_conversation), {
        "backend": "mem0_local",
        "run_dir": str(run_dir.resolve()),
        "collection": collection,
        "total_objects": total,
        "mapped_objects": total - len(unmapped),
        "conversation_count": len(by_conversation),
        "phase_mapping": "payload created_at -> containing MEME session time interval",
        "snapshot_semantics": (
            "final mutable Qdrant snapshot filtered by object creation provenance; "
            "not a historical reconstruction of the store at each cutoff"
        ),
        "historical_cutoff_reconstruction": False,
    }


def http_json(
    url: str, payload: dict[str, Any], timeout: float = 60.0, retries: int = 3
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"HTTP request failed for {url}: {last_error}")


def memmachine_project_id(prefix: str, run_name: str, conversation_id: str) -> str:
    return f"{prefix}__{run_name}__{conversation_id}"


def list_memmachine_memories(
    api_url: str,
    org_id: str,
    project_id: str,
    memory_type: str,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    """Enumerate a MemMachine project without depending on native search recall."""
    endpoint = f"{api_url.rstrip('/')}/memories/list"
    rows: list[dict[str, Any]] = []
    page_num = 0
    response_key = f"{memory_type}_memory"
    while True:
        response = http_json(
            endpoint,
            {
                "org_id": org_id,
                "project_id": project_id,
                "page_size": page_size,
                "page_num": page_num,
                "type": memory_type,
            },
            timeout=180.0,
        )
        if int(response.get("status", 0)) != 0:
            raise ValueError(
                f"MemMachine list failed for {project_id}/{memory_type}: {response}"
            )
        page = (response.get("content") or {}).get(response_key) or []
        if not isinstance(page, list):
            raise TypeError(
                f"MemMachine returned a non-list {response_key} for {project_id}"
            )
        rows.extend(item for item in page if isinstance(item, dict))
        if len(page) < page_size:
            return rows
        page_num += 1


def memmachine_session_index(
    metadata: dict[str, Any],
    created_at: Any,
    boundaries: list[tuple[str, int]],
) -> int | None:
    """Recover a source MEME session from native episode provenance."""
    session = str(metadata.get("session") or "")
    match = re.fullmatch(r"session_(\d+)", session)
    if match:
        index = int(match.group(1))
        if index in {value for _timestamp, value in boundaries}:
            return index
    timestamp = canonical_timestamp(
        metadata.get("source_timestamp")
        or metadata.get("session_datetime")
        or created_at
    )
    return source_session_for_timestamp(boundaries, timestamp)


async def _load_memmachine_postgres_metadata(
    database_url: str,
    feature_ids: list[int],
    session_keys: list[str],
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load feature citations and working summaries omitted by the REST models."""
    import asyncpg

    connection = await asyncpg.connect(database_url)
    try:
        citation_rows = await connection.fetch(
            """
            SELECT c.feature_id,
                   e.id AS episode_id,
                   e.created_at,
                   e.metadata
            FROM citations AS c
            JOIN episodestore AS e ON e.id::text = c.history_id
            WHERE c.feature_id = ANY($1::int[])
            ORDER BY c.feature_id, e.created_at, e.id
            """,
            feature_ids,
        )
        summary_rows = await connection.fetch(
            """
            SELECT session_key, summary, last_seq, episode_num, timestamp
            FROM short_term_memory_data
            WHERE session_key = ANY($1::varchar[])
            ORDER BY session_key
            """,
            session_keys,
        )
    finally:
        await connection.close()

    citations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in citation_rows:
        raw_metadata = row["metadata"]
        if isinstance(raw_metadata, str):
            raw_metadata = json.loads(raw_metadata)
        citations[int(row["feature_id"])].append(
            {
                "episode_id": int(row["episode_id"]),
                "created_at": str(row["created_at"] or ""),
                "metadata": dict(raw_metadata or {}),
            }
        )
    summaries = {
        str(row["session_key"]): {
            "summary": str(row["summary"] or "").strip(),
            "last_seq": row["last_seq"],
            "episode_num": row["episode_num"],
            "timestamp": row["timestamp"],
        }
        for row in summary_rows
    }
    return dict(citations), summaries


def load_memmachine_store(
    api_url: str,
    database_url: str,
    org_id: str,
    project_prefix: str,
    run_name: str,
    conversation_ids: list[str],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any], dict[str, Any]]:
    """Export all MemMachine episodic, semantic, and working-summary text."""
    if not database_url:
        raise ValueError(
            "MemMachine complete-store export requires --memmachine-database-url "
            "or MEMMACHINE_DATABASE_URL so semantic citations can be phase-mapped."
        )

    exports: dict[str, Any] = {}
    feature_ids: list[int] = []
    session_keys: list[str] = []
    for position, conversation_id in enumerate(conversation_ids, start=1):
        project_id = memmachine_project_id(
            project_prefix, run_name, conversation_id
        )
        episodic = list_memmachine_memories(
            api_url, org_id, project_id, "episodic"
        )
        semantic = list_memmachine_memories(
            api_url, org_id, project_id, "semantic"
        )
        if not episodic:
            raise ValueError(f"MemMachine returned no episodic store for {project_id}")
        for item in semantic:
            metadata = item.get("metadata") or {}
            feature_id = metadata.get("id") if isinstance(metadata, dict) else None
            if feature_id is None:
                raise ValueError(f"MemMachine semantic feature lacks id: {item}")
            feature_ids.append(int(feature_id))
        session_key = f"{org_id}/{project_id}"
        session_keys.append(session_key)
        exports[conversation_id] = {
            "org_id": org_id,
            "project_id": project_id,
            "session_key": session_key,
            "episodic_memory": episodic,
            "semantic_memory": semantic,
        }
        if position % 10 == 0 or position == len(conversation_ids):
            print(
                f"Exported MemMachine projects: {position}/{len(conversation_ids)} "
                f"({sum(len(row['episodic_memory']) for row in exports.values())} "
                f"episodic, {len(feature_ids)} semantic)"
            )

    citations, summaries = asyncio.run(
        _load_memmachine_postgres_metadata(
            database_url, sorted(set(feature_ids)), session_keys
        )
    )

    store: dict[str, list[MemoryObject]] = {}
    conversation_audit: list[dict[str, Any]] = []
    total_episodic = 0
    total_semantic = 0
    total_summaries = 0
    semantic_without_citations = 0
    escaped_control_tokens = 0
    unmapped: list[dict[str, str]] = []
    empty_objects: list[dict[str, str]] = []
    first_seen = 0

    for conversation_id in conversation_ids:
        exported = exports[conversation_id]
        boundaries = timestamp_maps.get(conversation_id, [])
        all_sessions = sorted(index for _timestamp, index in boundaries)
        objects: list[MemoryObject] = []
        episodic_count = 0
        semantic_count = 0

        for episode in exported["episodic_memory"]:
            raw_metadata = episode.get("metadata") or {}
            metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            source_session = memmachine_session_index(
                metadata, episode.get("created_at"), boundaries
            )
            raw_content = str(episode.get("content") or "").strip()
            content, escaped_count = escape_model_control_tokens(raw_content)
            escaped_control_tokens += escaped_count
            if not content:
                empty_objects.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": str(episode.get("uid") or ""),
                        "memory_kind": "episodic",
                    }
                )
                continue
            if source_session is None:
                unmapped.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": str(episode.get("uid") or ""),
                        "reason": "missing source session",
                    }
                )
                continue
            timestamp = str(
                metadata.get("source_timestamp") or episode.get("created_at") or ""
            )
            speaker = str(
                metadata.get("source_speaker") or episode.get("producer_id") or ""
            )
            prefix_parts = []
            if timestamp:
                prefix_parts.append(f"[{timestamp}]")
            if speaker:
                prefix_parts.append(f"{speaker}:")
            formatted = f"{' '.join(prefix_parts)} {content}".strip()
            objects.append(
                MemoryObject(
                    content=formatted,
                    score=0.0,
                    metadata={
                        "memory_id": str(episode.get("uid") or ""),
                        "memory_kind": "episodic",
                        "source_backend": "memmachine_memories_list_api",
                        "source_session_indices": [source_session],
                        "source_timestamp": timestamp,
                        "store_metadata": metadata,
                    },
                    first_seen=first_seen,
                )
            )
            first_seen += 1
            episodic_count += 1

        for feature in exported["semantic_memory"]:
            raw_metadata = feature.get("metadata") or {}
            feature_metadata = (
                dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
            )
            feature_id = int(feature_metadata["id"])
            feature_citations = citations.get(feature_id, [])
            source_sessions = sorted(
                {
                    session
                    for citation in feature_citations
                    if (
                        session := memmachine_session_index(
                            citation["metadata"],
                            citation.get("created_at"),
                            boundaries,
                        )
                    )
                    is not None
                }
            )
            conservative_unknown_provenance = not source_sessions
            if conservative_unknown_provenance:
                # The current consolidated feature exists in the complete store,
                # but without citations it cannot safely be exposed before every
                # source session is visible.  Treat it as depending on the whole
                # conversation rather than leaking final state into before queries.
                source_sessions = all_sessions
                semantic_without_citations += 1
            value = str(feature.get("value") or "").strip()
            if not value:
                empty_objects.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": str(feature_id),
                        "memory_kind": "semantic",
                    }
                )
                continue
            if not source_sessions:
                unmapped.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": str(feature_id),
                        "reason": "missing phase provenance",
                    }
                )
                continue
            category = str(feature.get("category") or "semantic")
            name = str(
                feature.get("feature_name") or feature.get("tag") or "fact"
            )
            raw_content = f"{category} / {name}: {value}"
            content, escaped_count = escape_model_control_tokens(raw_content)
            escaped_control_tokens += escaped_count
            objects.append(
                MemoryObject(
                    content=content,
                    score=0.0,
                    metadata={
                        "memory_id": str(feature_id),
                        "memory_kind": "semantic",
                        "source_backend": "memmachine_memories_list_api",
                        "source_session_indices": source_sessions,
                        "citation_episode_ids": [
                            row["episode_id"] for row in feature_citations
                        ],
                        "conservative_unknown_provenance": (
                            conservative_unknown_provenance
                        ),
                        "set_id": feature.get("set_id"),
                        "tag": feature.get("tag"),
                    },
                    first_seen=first_seen,
                )
            )
            first_seen += 1
            semantic_count += 1

        summary_row = summaries.get(exported["session_key"], {})
        summary = str(summary_row.get("summary") or "").strip()
        summary_count = 0
        if summary:
            content, escaped_count = escape_model_control_tokens(summary)
            escaped_control_tokens += escaped_count
            objects.append(
                MemoryObject(
                    content=content,
                    score=0.0,
                    metadata={
                        "memory_id": f"{exported['session_key']}:summary",
                        "memory_kind": "episodic_summary",
                        "source_backend": "memmachine_short_term_memory_data",
                        "source_session_indices": all_sessions,
                        "conservative_unknown_provenance": True,
                        "summary_metadata": summary_row,
                    },
                    first_seen=first_seen,
                )
            )
            first_seen += 1
            summary_count = 1

        if not objects:
            raise ValueError(f"MemMachine store has no usable objects for {conversation_id}")
        store[conversation_id] = objects
        total_episodic += episodic_count
        total_semantic += semantic_count
        total_summaries += summary_count
        conversation_audit.append(
            {
                "conversation_id": conversation_id,
                "project_id": exported["project_id"],
                "episodic": episodic_count,
                "semantic": semantic_count,
                "working_summaries": summary_count,
                "total_objects": len(objects),
            }
        )
        exported["working_memory_summary"] = summary_row

    if unmapped:
        sample = json.dumps(unmapped[:3], ensure_ascii=False)
        raise ValueError(
            f"MemMachine complete-store validation failed: {len(unmapped)} objects "
            f"cannot be mapped phase-safely. Sample: {sample}"
        )
    total_objects = total_episodic + total_semantic + total_summaries
    return store, {
        "backend": "memmachine",
        "api_url": api_url,
        "org_id": org_id,
        "project_prefix": project_prefix,
        "run_name": run_name,
        "conversation_count": len(store),
        "episodic": total_episodic,
        "semantic": total_semantic,
        "working_summaries": total_summaries,
        "total_objects": total_objects,
        "exported_objects": (
            sum(len(row["episodic_memory"]) for row in exports.values())
            + sum(len(row["semantic_memory"]) for row in exports.values())
            + total_summaries
        ),
        "empty_store_objects": empty_objects,
        "semantic_without_citations": semantic_without_citations,
        "escaped_model_control_tokens": escaped_control_tokens,
        "candidate_scope": (
            "all episodic and semantic objects from /memories/list plus persisted "
            "working-memory summaries"
        ),
        "phase_mapping": (
            "episodic source metadata and semantic citation episode IDs -> MEME sessions; "
            "uncited consolidated features/summaries conservatively depend on all sessions"
        ),
        "reader_exposure": (
            "native-formatted episodic, semantic-feature, and summary text only"
        ),
        "snapshot_semantics": (
            "current mutable MemMachine store filtered by complete citation provenance; "
            "historical feature versions and summaries are not reconstructed"
        ),
        "historical_cutoff_reconstruction": False,
        "conversations": conversation_audit,
    }, exports


def flatten_memos_response(data: dict[str, Any]) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    for group in ("text_mem", "pref_mem", "tool_mem", "skill_mem"):
        for bucket in data.get(group) or []:
            if not isinstance(bucket, dict):
                continue
            cube_id = str(bucket.get("cube_id") or "")
            for item in bucket.get("memories") or []:
                if isinstance(item, dict):
                    memory = dict(item)
                    memory["_cube_id"] = cube_id
                    memory["_group"] = group
                    memories.append(memory)
    return memories


def escape_model_control_tokens(text: str) -> tuple[str, int]:
    """Make stored model delimiters ordinary reader-visible text."""
    matches = SPECIAL_TOKEN_RE.findall(text)
    return SPECIAL_TOKEN_RE.sub(lambda match: f"[MODEL_{match.group(1)}]", text), len(matches)


def validate_native_namespace(
    run_dir: Path, expected_prefix: str, *, separator: str
) -> dict[str, Any]:
    """Verify that a native run actually queried the store being exported."""
    search_path = run_dir / "search_results.json"
    if not search_path.is_file():
        raise FileNotFoundError(f"Native search results not found: {search_path}")
    expected = f"{expected_prefix}{separator}meme_"
    namespace_re = re.compile(r'"user_id"\s*:\s*"([^"]+)"')
    observed: set[str] = set()
    with search_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            observed.update(namespace_re.findall(line))
    mismatched = sorted(value for value in observed if not value.startswith(expected))
    if not observed or mismatched:
        raise ValueError(
            f"Native namespace validation failed for {run_dir}: expected {expected}*, "
            f"observed={sorted(observed)[:5]}, mismatched={mismatched[:5]}"
        )
    return {
        "run_dir": str(run_dir.resolve()),
        "expected_prefix": expected_prefix,
        "separator": separator,
        "observed_user_ids": len(observed),
    }


def load_memos_store(
    api_url: str,
    prefix: str,
    conversation_ids: list[str],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any], list[dict[str, Any]]]:
    """Export all Memos objects while using raw sources only for phase safety."""
    by_conversation: dict[str, list[MemoryObject]] = defaultdict(list)
    exported: list[dict[str, Any]] = []
    unmapped: list[dict[str, str]] = []
    group_counts: dict[str, int] = defaultdict(int)
    escaped_control_tokens = 0
    first_seen = 0
    endpoint = f"{api_url.rstrip('/')}/product/get_memory"

    for position, conversation_id in enumerate(conversation_ids, start=1):
        match = re.search(r"(\d+)$", conversation_id)
        if not match:
            raise ValueError(f"Cannot parse MEME conversation id: {conversation_id}")
        cube_id = f"{prefix}__meme_{match.group(1)}_User"
        payload = {
            "mem_cube_id": cube_id,
            "user_id": cube_id,
            "include_preference": True,
            "include_tool_memory": True,
            "include_skill_memory": True,
            "page": None,
            "page_size": None,
        }
        response = http_json(endpoint, payload)
        memories = flatten_memos_response(response.get("data") or {})
        if not memories:
            raise ValueError(f"Memos returned no stored objects for {cube_id}")
        for item in memories:
            exported.append(item)
            group_counts[str(item.get("_group") or "unknown")] += 1
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            sources = [
                source
                for source in metadata.get("sources") or []
                if isinstance(source, dict)
            ]
            source_timestamps = [
                canonical_timestamp(source.get("chat_time"))
                for source in sources
                if source.get("chat_time")
            ]
            source_sessions = sorted(
                {
                    session
                    for timestamp in source_timestamps
                    if (
                        session := source_session_for_timestamp(
                            timestamp_maps.get(conversation_id, []), timestamp
                        )
                    )
                    is not None
                }
            )
            raw_content = str(item.get("memory") or item.get("memory_value") or "").strip()
            content, escaped_count = escape_model_control_tokens(raw_content)
            escaped_control_tokens += escaped_count
            memory_id = str(item.get("id") or f"{cube_id}:{first_seen}")
            if not content or not source_timestamps or len(source_sessions) == 0:
                unmapped.append(
                    {
                        "memory_id": memory_id,
                        "conversation_id": conversation_id,
                        "reason": (
                            "missing memory text"
                            if not content
                            else "missing or unmappable metadata.sources[].chat_time"
                        ),
                    }
                )
                first_seen += 1
                continue
            by_conversation[conversation_id].append(
                MemoryObject(
                    content=content,
                    score=0.0,
                    metadata={
                        "memory_id": memory_id,
                        "source_backend": "memos_local_product_api",
                        "source_group": item.get("_group"),
                        "source_cube_id": cube_id,
                        "source_session_indices": source_sessions,
                        "source_timestamps": source_timestamps,
                        # Preserve store metadata for auditing, but formatted
                        # context exposes only ``content``.
                        "store_metadata": metadata,
                    },
                    first_seen=first_seen,
                )
            )
            first_seen += 1
        if position % 10 == 0 or position == len(conversation_ids):
            print(
                f"Exported Memos cubes: {position}/{len(conversation_ids)} "
                f"({len(exported)} objects)"
            )

    if unmapped:
        sample = json.dumps(unmapped[:3], ensure_ascii=False)
        raise ValueError(
            f"Memos complete-store validation failed: {len(unmapped)}/{len(exported)} "
            f"objects cannot be mapped phase-safely. Sample: {sample}"
        )
    return dict(by_conversation), {
        "backend": "memos_local",
        "api_url": api_url,
        "cube_prefix": prefix,
        "total_objects": len(exported),
        "mapped_objects": len(exported) - len(unmapped),
        "conversation_count": len(by_conversation),
        "group_counts": dict(group_counts),
        "escaped_model_control_tokens": escaped_control_tokens,
        "phase_mapping": (
            "metadata.sources[].chat_time -> containing MEME session time interval"
        ),
        "reader_exposure": "memory text only; raw metadata source content excluded",
        "snapshot_semantics": (
            "current mutable Memos product-store export filtered by source provenance; "
            "not a historical reconstruction of the store at each cutoff"
        ),
        "historical_cutoff_reconstruction": False,
    }, exported


def parse_memu_resource_name(value: Any) -> tuple[str, int] | None:
    """Parse both original and memU-generated ``#segment`` resource URLs."""
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    match = MEMU_RESOURCE_RE.fullmatch(name)
    if not match:
        return None
    return match.group("conversation"), int(match.group("part"))


def memu_resource_provenance(
    resources: list[dict[str, Any]],
    run_dir: Path,
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Map persisted resources to source parts without exposing source text."""
    provenance: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    message_cache: dict[tuple[str, int], tuple[str, int | None]] = {}
    for row in resources:
        resource_id = str(row.get("id") or "")
        user_id = str(row.get("user_id") or "")
        parsed = parse_memu_resource_name(row.get("url") or row.get("local_path"))
        if not resource_id or parsed is None:
            errors.append({"resource_id": resource_id, "reason": "unparseable resource URL"})
            continue
        conversation_id, part_index = parsed
        if user_id != f"normal:{conversation_id}":
            errors.append(
                {
                    "resource_id": resource_id,
                    "reason": "namespace/conversation mismatch",
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
            )
            continue
        cache_key = (conversation_id, part_index)
        if cache_key not in message_cache:
            source_path = run_dir / "memu_resources" / (
                f"{conversation_id}-part-{part_index:04d}.json"
            )
            try:
                messages = load_json(source_path)
                timestamp = canonical_timestamp(messages[0].get("created_at"))
            except (FileNotFoundError, IndexError, AttributeError, TypeError, json.JSONDecodeError):
                timestamp = ""
            session = source_session_for_timestamp(
                timestamp_maps.get(conversation_id, []), timestamp
            )
            message_cache[cache_key] = (timestamp, session)
        timestamp, session = message_cache[cache_key]
        if not timestamp or session is None:
            errors.append(
                {
                    "resource_id": resource_id,
                    "reason": "missing or unmappable source part timestamp",
                    "conversation_id": conversation_id,
                    "part_index": part_index,
                }
            )
            continue
        provenance[resource_id] = {
            "conversation_id": conversation_id,
            "part_index": part_index,
            "source_timestamp": timestamp,
            "source_session_indices": [session],
            "segmented_resource": "#segment_" in str(row.get("url") or ""),
        }
    return provenance, errors


def memu_rows_to_store(
    resources: list[dict[str, Any]],
    items: list[dict[str, Any]],
    run_dir: Path,
    timestamp_maps: dict[str, list[tuple[str, int]]],
    categories: list[dict[str, Any]] | None = None,
    category_items: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any], list[dict[str, Any]]]:
    """Convert complete MemU PostgreSQL rows into phase-aware memory objects."""
    provenance, errors = memu_resource_provenance(resources, run_dir, timestamp_maps)
    by_conversation: dict[str, list[MemoryObject]] = defaultdict(list)
    exports: list[dict[str, Any]] = []
    type_counts: dict[str, int] = defaultdict(int)
    escaped_control_tokens = 0
    ordered = sorted(
        items,
        key=lambda row: (
            provenance.get(str(row.get("resource_id") or ""), {}).get("conversation_id", ""),
            provenance.get(str(row.get("resource_id") or ""), {}).get("part_index", -1),
            str(row.get("created_at") or ""),
            str(row.get("id") or ""),
        ),
    )
    item_provenance: dict[str, dict[str, Any]] = {}
    for first_seen, row in enumerate(ordered):
        item_id = str(row.get("id") or "")
        resource_id = str(row.get("resource_id") or "")
        source = provenance.get(resource_id)
        content, escaped_count = escape_model_control_tokens(
            str(row.get("summary") or "").strip()
        )
        escaped_control_tokens += escaped_count
        if source is None or not item_id or not content:
            errors.append(
                {
                    "memory_id": item_id,
                    "resource_id": resource_id,
                    "reason": "missing provenance, id, or summary",
                }
            )
            continue
        conversation_id = source["conversation_id"]
        if str(row.get("user_id") or "") != f"normal:{conversation_id}":
            errors.append(
                {
                    "memory_id": item_id,
                    "reason": "item namespace/conversation mismatch",
                }
            )
            continue
        memory_type = str(row.get("memory_type") or "unknown")
        type_counts[memory_type] += 1
        metadata = {
            "memory_id": item_id,
            "resource_id": resource_id,
            "memory_type": memory_type,
            "source_backend": "memu_local_postgres_memory_items",
            **source,
        }
        item_provenance[item_id] = metadata
        by_conversation[conversation_id].append(
            MemoryObject(content=content, score=0.0, metadata=metadata, first_seen=first_seen)
        )
        exports.append({"object_kind": "memory_item", "content": content, "metadata": metadata})

    links_by_category: dict[str, list[str]] = defaultdict(list)
    for link in category_items or []:
        links_by_category[str(link.get("category_id") or "")].append(
            str(link.get("item_id") or "")
        )
    category_count = 0
    empty_category_count = 0
    for category in sorted(
        categories or [], key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or ""))
    ):
        category_id = str(category.get("id") or "")
        linked = [
            item_provenance[item_id]
            for item_id in links_by_category.get(category_id, [])
            if item_id in item_provenance
        ]
        content, escaped_count = escape_model_control_tokens(
            str(category.get("summary") or "").strip()
        )
        escaped_control_tokens += escaped_count
        user_id = str(category.get("user_id") or "")
        conversation_id = user_id.removeprefix("normal:")
        if not content:
            empty_category_count += 1
            exports.append(
                {
                    "object_kind": "memory_category",
                    "content": "",
                    "metadata": {
                        "memory_id": category_id,
                        "category_name": category.get("name"),
                        "description": category.get("description"),
                        "user_id": user_id,
                        "excluded_reason": "empty reader-visible summary",
                    },
                }
            )
            continue
        if (
            not category_id
            or not linked
            or user_id != f"normal:{conversation_id}"
        ):
            errors.append(
                {"category_id": category_id, "reason": "missing summary or linked item provenance"}
            )
            continue
        source_sessions = sorted(
            {
                int(session)
                for metadata in linked
                for session in metadata["source_session_indices"]
            }
        )
        source_parts = sorted({int(metadata["part_index"]) for metadata in linked})
        metadata = {
            "memory_id": category_id,
            "memory_type": "category_summary",
            "category_name": category.get("name"),
            "source_backend": "memu_local_postgres_memory_categories",
            "conversation_id": conversation_id,
            "source_session_indices": source_sessions,
            "source_part_indices": source_parts,
            "linked_item_count": len(linked),
        }
        first_seen = len(ordered) + category_count
        category_count += 1
        by_conversation[conversation_id].append(
            MemoryObject(content=content, score=0.0, metadata=metadata, first_seen=first_seen)
        )
        exports.append(
            {"object_kind": "memory_category", "content": content, "metadata": metadata}
        )
    if errors:
        sample = json.dumps(errors[:3], ensure_ascii=False)
        raise ValueError(
            f"MemU complete-store validation failed: {len(errors)} invalid rows. Sample: {sample}"
        )
    return dict(by_conversation), {
        "backend": "memu_local_postgres",
        "run_dir": str(run_dir.resolve()),
        "total_resources": len(resources),
        "segmented_resources": sum(
            "#segment_" in str(row.get("url") or "") for row in resources
        ),
        "memory_items": len(items),
        "memory_categories": len(categories or []),
        "reader_visible_categories": category_count,
        "empty_category_summaries": empty_category_count,
        "category_item_links": len(category_items or []),
        "total_objects": len(items) + len(categories or []),
        "mapped_objects": len(items) + category_count,
        "conversation_count": len(by_conversation),
        "memory_type_counts": dict(type_counts),
        "escaped_model_control_tokens": escaped_control_tokens,
        "phase_mapping": "resource URL -> source part JSON timestamp -> MEME session",
        "reader_exposure": "memory_items.summary only; resource source text excluded",
        "candidate_scope": (
            "all memory_items and linked memory_categories in exact normal:meme_* namespaces"
        ),
        "category_phase_mapping": (
            "category_items links -> linked item source sessions; visible only after all sources"
        ),
        "historical_cutoff_reconstruction": False,
    }, exports


def load_memu_store(
    database_url: str,
    run_dir: Path,
    conversation_ids: list[str],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any], list[dict[str, Any]]]:
    """Enumerate every persisted MemU item for the exact native namespaces."""
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError("Run MemU store export in the memu-py313 environment") from exc

    users = [f"normal:{conversation_id}" for conversation_id in conversation_ids]
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, user_id, url, local_path FROM resources "
                "WHERE user_id = ANY(%s)",
                (users,),
            )
            resources = list(cursor.fetchall())
            cursor.execute(
                "SELECT id, user_id, resource_id, memory_type, summary, created_at "
                "FROM memory_items WHERE user_id = ANY(%s)",
                (users,),
            )
            items = list(cursor.fetchall())
            cursor.execute(
                "SELECT id, user_id, name, description, summary, created_at "
                "FROM memory_categories "
                "WHERE user_id = ANY(%s)",
                (users,),
            )
            categories = list(cursor.fetchall())
            cursor.execute(
                "SELECT category_id, item_id, user_id FROM category_items "
                "WHERE user_id = ANY(%s)",
                (users,),
            )
            category_items = list(cursor.fetchall())
    observed = {str(row["user_id"]) for row in resources}
    missing = sorted(set(users) - observed)
    if missing:
        raise ValueError(f"MemU store is missing requested namespaces: {missing[:5]}")
    return memu_rows_to_store(
        resources,
        items,
        run_dir,
        timestamp_maps,
        categories=categories,
        category_items=category_items,
    )


def load_emem_store(
    run_dir: Path, conversation_ids: list[str]
) -> tuple[dict[tuple[str, str], list[MemoryObject]], dict[str, Any]]:
    """Load every EDU from EMem's persisted before/after phase snapshots."""
    import pyarrow.parquet as pq

    root = run_dir / "emem_outputs"
    if not root.is_dir():
        raise FileNotFoundError(f"EMem output directory not found: {root}")
    store: dict[tuple[str, str], list[MemoryObject]] = {}
    snapshot_audit: list[dict[str, Any]] = []
    total_objects = 0
    escaped_control_tokens = 0
    objects_without_session_provenance = 0
    first_seen = 0

    for conversation_id in conversation_ids:
        match = re.search(r"(\d+)$", conversation_id)
        if not match:
            raise ValueError(f"Cannot parse MEME conversation id: {conversation_id}")
        conv = int(match.group(1))
        for phase_index, phase in enumerate(("before", "after")):
            sample_dirs = sorted(root.glob(f"sample_meme_{conv}__{phase_index:02d}_*"))
            if len(sample_dirs) != 1:
                raise ValueError(
                    f"Expected one EMem {phase} snapshot for {conversation_id}, "
                    f"found {sample_dirs}"
                )
            parquet_paths = sorted(sample_dirs[0].glob("*/edu_embeddings/vdb_edu.parquet"))
            if not parquet_paths:
                raise FileNotFoundError(f"No EMem EDU parquet under {sample_dirs[0]}")
            candidates = []
            for path in parquet_paths:
                table = pq.read_table(path, columns=["hash_id", "content"])
                candidates.append((table.num_rows, path, table))
            # One interrupted early artifact exists for meme_0/before.  The
            # completed native run retrieved an EDU only present in the larger
            # 380-row snapshot, so prefer the most complete parquet.
            row_count, selected_path, table = max(candidates, key=lambda item: item[0])
            attributes_path = selected_path.parents[1] / "conversation_graph_attributes.pkl"
            if not attributes_path.is_file():
                raise FileNotFoundError(
                    f"No EMem graph attributes beside selected parquet: {attributes_path}"
                )
            with attributes_path.open("rb") as handle:
                graph_attributes = pickle.load(handle)  # noqa: S301 - trusted run artifact
            edu_to_sessions = graph_attributes.get("edu_id_to_session_ids") or {}
            date_to_session = graph_attributes.get("date_to_session_id") or {}
            session_order = {
                session_id: index
                for index, (_date, session_id) in enumerate(
                    sorted(date_to_session.items(), key=lambda item: item[0])
                )
            }
            session_dates = {
                session_id: canonical_timestamp(date)
                for date, session_id in date_to_session.items()
            }
            objects: list[MemoryObject] = []
            missing_content = 0
            for row in table.to_pylist():
                raw_content = str(row.get("content") or "").strip()
                if not raw_content:
                    missing_content += 1
                    continue
                content, escaped_count = escape_model_control_tokens(raw_content)
                escaped_control_tokens += escaped_count
                memory_id = str(row.get("hash_id") or "")
                source_session_ids = sorted(
                    edu_to_sessions.get(memory_id) or (),
                    key=lambda value: session_order.get(value, 10**9),
                )
                source_session_indices = sorted(
                    {
                        session_order[session_id]
                        for session_id in source_session_ids
                        if session_id in session_order
                    }
                )
                if not source_session_indices:
                    objects_without_session_provenance += 1
                objects.append(
                    MemoryObject(
                        content=content,
                        score=0.0,
                        metadata={
                            "memory_id": memory_id,
                            "source_backend": "emem_edu_parquet",
                            "source_phase_snapshot": phase,
                            "source_snapshot_dir": str(sample_dirs[0]),
                            "source_parquet": str(selected_path),
                            "source_session_ids": source_session_ids,
                            "source_session_indices": source_session_indices,
                            "source_timestamps": [
                                session_dates[session_id]
                                for session_id in source_session_ids
                                if session_id in session_dates
                            ],
                        },
                        first_seen=first_seen,
                    )
                )
                first_seen += 1
            if missing_content:
                raise ValueError(
                    f"EMem snapshot {selected_path} contains {missing_content} empty EDUs"
                )
            if not objects:
                raise ValueError(f"EMem snapshot has no EDUs: {selected_path}")
            store[(conversation_id, phase)] = objects
            total_objects += len(objects)
            snapshot_audit.append(
                {
                    "conversation_id": conversation_id,
                    "phase": phase,
                    "selected_parquet": str(selected_path.resolve()),
                    "object_count": len(objects),
                    "parquet_candidates": [
                        {"path": str(path.resolve()), "rows": count}
                        for count, path, _table in candidates
                    ],
                }
            )

    return store, {
        "backend": "emem",
        "run_dir": str(run_dir.resolve()),
        "snapshot_count": len(store),
        "total_snapshot_objects": total_objects,
        "escaped_model_control_tokens": escaped_control_tokens,
        "objects_without_session_provenance": objects_without_session_provenance,
        "candidate_scope": "all EDUs in each persisted phase snapshot",
        "phase_mapping": (
            "native before/after visible-until snapshot boundary; EDU chronology from "
            "conversation_graph_attributes edu_id_to_session_ids/date_to_session_id"
        ),
        "snapshots": snapshot_audit,
    }


def emem_core_phrases(spec: Any) -> list[str]:
    """Gold-field phrases whose coverage, rather than query overlap, defines an oracle."""
    values = list(spec.support_values)
    values.extend(flatten_strings(spec.entity_values))
    values.append(spec.gold_answer)
    return list(
        dict.fromkeys(value for value in map(normalized_text, values) if value)
    )


def focus_emem_ranked_candidates(
    ranked: list[tuple[MemoryObject, float, list[str]]], spec: Any
) -> list[tuple[MemoryObject, float, list[str]]]:
    """Keep a compact gold-covering EDU set and discard complete-store query noise.

    The previous complete-store builder admitted every EDU with even one question-word
    overlap.  Candidate-pool expansion could therefore fill the reader budget with
    generic passages.  A gold oracle should first cover phrases that are actually in
    the benchmark state, while retaining a small amount of relation evidence for
    cascade/absence questions.
    """
    # Exact recall benefits from corroborating summary/detail EDUs, while deletion
    # is frequently represented only by disappearance from the snapshot.  Preserve
    # the full legacy candidate ranking for those tasks and change only presentation
    # chronology; compacting them around the old literal value is counterproductive.
    if spec.task_type in {"ER", "Del"}:
        return ranked

    core = emem_core_phrases(spec)
    present = {
        phrase
        for phrase in core
        if any(phrase in normalized_text(item.content) for item, _score, _hits in ranked)
    }
    selected: list[tuple[MemoryObject, float, list[str]]] = []
    selected_ids: set[int] = set()
    coverage_goal = 2 if spec.task_type == "Agg" else 1
    coverage_counts = {phrase: 0 for phrase in present}
    for entry in ranked:
        memory = entry[0]
        covered = {
            phrase
            for phrase in present
            if coverage_counts[phrase] < coverage_goal
            and phrase in normalized_text(memory.content)
        }
        if not covered:
            continue
        selected.append(entry)
        selected_ids.add(id(memory))
        for phrase in covered:
            coverage_counts[phrase] += 1
        if all(count >= coverage_goal for count in coverage_counts.values()):
            break

    # Conditional state transitions can be expressed without repeating a literal
    # value.  Retain at most two high-scoring relation EDUs mentioning the target or
    # cascade-source label, but do not re-admit the unbounded query-overlap tail.
    relation_phrases = [
        normalized_text(entity.replace("_", " ")) for entity in spec.target_entities
    ]
    if spec.cascade_source:
        relation_phrases.append(normalized_text(spec.cascade_source.replace("_", " ")))
    relation_added = 0
    for entry in ranked:
        memory = entry[0]
        if id(memory) in selected_ids:
            continue
        text = normalized_text(memory.content)
        if any(phrase and phrase in text for phrase in relation_phrases):
            selected.append(entry)
            selected_ids.add(id(memory))
            relation_added += 1
            if relation_added >= 2:
                break

    # If extraction destroyed every gold phrase, preserve a small diagnostic
    # fallback instead of producing an empty context or flooding it with noise.
    if not selected:
        selected = ranked[:3]
    return selected


def pack_emem_context(
    ranked: list[tuple[MemoryObject, float, list[str]]],
    encoding: Any,
    budget: int | None,
) -> tuple[list[dict[str, Any]], str]:
    """Select by oracle relevance, then present selected EDUs chronologically."""
    selected_tuples: list[tuple[MemoryObject, float, list[str]]] = []
    used = 0
    for memory, oracle_score, hits in ranked:
        segment = f"<MEMORY>\n{memory.content}\n</MEMORY>\n\n"
        cost = len(encoding.encode(segment))
        if budget is not None and used + cost > budget:
            continue
        selected_tuples.append((memory, oracle_score, hits))
        used += cost

    def chronology_key(
        entry: tuple[MemoryObject, float, list[str]],
    ) -> tuple[int, int, int]:
        memory = entry[0]
        indices = memory.metadata.get("source_session_indices") or []
        if indices:
            return (0, min(int(value) for value in indices), memory.first_seen)
        return (1, 10**9, memory.first_seen)

    selected_tuples.sort(key=chronology_key)
    selected = [
        {
            "content": memory.content,
            "score": oracle_score,
            "metadata": {
                "oracle_exact_hits": hits,
                "native_score": memory.score,
                "native_metadata": memory.metadata,
            },
        }
        for memory, oracle_score, hits in selected_tuples
    ]
    context = "".join(
        f"<MEMORY>\n{memory.content}\n</MEMORY>\n\n"
        for memory, _oracle_score, _hits in selected_tuples
    )
    return selected, context


def load_evermemos_store(
    run_dir: Path,
    conversation_ids: list[str],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, list[MemoryObject]], dict[str, Any]]:
    """Load all EverMemOS episode and atomic-fact representations."""
    vector_dir = run_dir / "vectors"
    if not vector_dir.is_dir():
        raise FileNotFoundError(f"EverMemOS vector directory not found: {vector_dir}")
    store: dict[str, list[MemoryObject]] = {}
    conversation_audit: list[dict[str, Any]] = []
    total_episodes = 0
    total_facts = 0
    escaped_control_tokens = 0
    unmapped: list[dict[str, str]] = []
    first_seen = 0

    for position, conversation_id in enumerate(conversation_ids, start=1):
        match = re.search(r"(\d+)$", conversation_id)
        if not match:
            raise ValueError(f"Cannot parse MEME conversation id: {conversation_id}")
        path = vector_dir / f"embedding_index_conv_{int(match.group(1))}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"EverMemOS store not found: {path}")
        with path.open("rb") as handle:
            rows = pickle.load(handle)  # noqa: S301 - trusted local run artifact

        objects: list[MemoryObject] = []
        episode_count = 0
        fact_count = 0
        for row_index, row in enumerate(rows):
            doc = row.get("doc") if isinstance(row, dict) else None
            if not isinstance(doc, dict):
                unmapped.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": f"row:{row_index}",
                        "reason": "missing doc",
                    }
                )
                continue
            event_id = str(doc.get("event_id") or f"row:{row_index}")
            source_timestamps = []
            for original in doc.get("original_data") or []:
                if not isinstance(original, dict):
                    continue
                message = original.get("message", original)
                if isinstance(message, dict) and message.get("timestamp"):
                    source_timestamps.append(canonical_timestamp(message["timestamp"]))
            source_sessions = sorted(
                {
                    session
                    for timestamp in source_timestamps
                    if (
                        session := source_session_for_timestamp(
                            timestamp_maps.get(conversation_id, []), timestamp
                        )
                    )
                    is not None
                }
            )
            if not source_timestamps or not source_sessions:
                unmapped.append(
                    {
                        "conversation_id": conversation_id,
                        "memory_id": event_id,
                        "reason": "missing or unmappable original_data timestamp",
                    }
                )
                continue
            common_metadata = {
                "event_id": event_id,
                "source_backend": "evermemos_vector_store",
                "source_store_path": str(path),
                "source_session_indices": source_sessions,
                "source_timestamps": source_timestamps,
                "timestamp": str(doc.get("timestamp") or ""),
                "subject": str(doc.get("subject") or ""),
            }
            raw_episode = str(doc.get("episode") or "").strip()
            if raw_episode:
                episode, escaped_count = escape_model_control_tokens(raw_episode)
                escaped_control_tokens += escaped_count
                objects.append(
                    MemoryObject(
                        content=episode,
                        score=0.0,
                        metadata={
                            **common_metadata,
                            "memory_id": f"{event_id}:episode",
                            "memory_kind": "episode",
                        },
                        first_seen=first_seen,
                    )
                )
                first_seen += 1
                episode_count += 1

            atomic = doc.get("atomic_fact") or {}
            fact_texts = atomic.get("atomic_fact") or [] if isinstance(atomic, dict) else []
            for fact_index, raw_fact in enumerate(fact_texts):
                raw_fact = str(raw_fact).strip()
                if not raw_fact:
                    continue
                fact, escaped_count = escape_model_control_tokens(raw_fact)
                escaped_control_tokens += escaped_count
                objects.append(
                    MemoryObject(
                        content=fact,
                        score=0.0,
                        metadata={
                            **common_metadata,
                            "memory_id": f"{event_id}:fact:{fact_index}",
                            "memory_kind": "atomic_fact",
                            "within_event_index": fact_index,
                        },
                        first_seen=first_seen,
                    )
                )
                first_seen += 1
                fact_count += 1
        if not objects:
            raise ValueError(f"EverMemOS store has no usable objects for {conversation_id}")
        store[conversation_id] = objects
        total_episodes += episode_count
        total_facts += fact_count
        conversation_audit.append(
            {
                "conversation_id": conversation_id,
                "store_path": str(path.resolve()),
                "source_events": len(rows),
                "episodes": episode_count,
                "atomic_facts": fact_count,
            }
        )
        if position % 10 == 0 or position == len(conversation_ids):
            print(
                f"Loaded EverMemOS stores: {position}/{len(conversation_ids)} "
                f"({total_episodes} episodes, {total_facts} atomic facts)"
            )

    if unmapped:
        sample = json.dumps(unmapped[:3], ensure_ascii=False)
        raise ValueError(
            f"EverMemOS complete-store validation failed: {len(unmapped)} events "
            f"cannot be mapped phase-safely. Sample: {sample}"
        )
    return store, {
        "backend": "evermemos",
        "run_dir": str(run_dir.resolve()),
        "conversation_count": len(store),
        "episodes": total_episodes,
        "atomic_facts": total_facts,
        "total_objects": total_episodes + total_facts,
        "escaped_model_control_tokens": escaped_control_tokens,
        "candidate_scope": "all episode and atomic-fact text in persisted vector stores",
        "phase_mapping": (
            "doc.original_data message timestamps -> containing MEME sessions"
        ),
        "reader_exposure": "episode/atomic-fact text only; original_data content excluded",
        "snapshot_semantics": (
            "persisted EverMemOS vector representations filtered by source provenance; "
            "immutability and historical cutoff reconstruction are not assumed"
        ),
        "historical_cutoff_reconstruction": False,
        "conversations": conversation_audit,
    }


def phase_safe_pool(
    objects: list[MemoryObject], cutoff: int
) -> list[MemoryObject]:
    deduped: dict[str, MemoryObject] = {}
    for memory in objects:
        source_sessions = memory.metadata.get("source_session_indices") or []
        if not source_sessions or max(int(value) for value in source_sessions) > cutoff:
            continue
        key = normalized_text(memory.content)
        if key and key not in deduped:
            deduped[key] = memory
    return sorted(deduped.values(), key=lambda item: item.first_seen)


def deduplicated_pool(objects: list[MemoryObject]) -> list[MemoryObject]:
    deduped: dict[str, MemoryObject] = {}
    for memory in objects:
        key = normalized_text(memory.content)
        if key and key not in deduped:
            deduped[key] = memory
    return sorted(deduped.values(), key=lambda item: item.first_seen)


def rank_rag_chunks(
    pool: list[MemoryObject], spec: QuestionSpec, official_evidence_sessions: set[int]
) -> tuple[list[tuple[MemoryObject, float, list[str]]], dict[str, bool]]:
    """Rank whole RAG chunks with gold fields and official-location provenance."""
    phrases, query_tokens = coverage_signals(spec)
    phrase_present = {phrase: False for phrase in phrases}
    ranked: list[tuple[MemoryObject, float, list[str]]] = []
    for chunk in pool:
        text = normalized_text(chunk.content)
        chunk_tokens = set(text.split())
        hits = [phrase for phrase in phrases if phrase in text]
        for phrase in hits:
            phrase_present[phrase] = True
        overlap = len(chunk_tokens & query_tokens) / max(1, len(query_tokens))
        source_sessions = {
            int(value) for value in chunk.metadata.get("source_session_indices") or []
        }
        evidence_location = bool(source_sessions & official_evidence_sessions)
        exact_weight = sum(
            4.0 + min(4.0, len(phrase.split()) / 3.0) for phrase in hits
        )
        score = exact_weight + 2.0 * overlap + (1.0 if evidence_location else 0.0)
        if hits or overlap > 0 or evidence_location:
            ranked.append((chunk, score, hits))
    ranked.sort(key=lambda item: (item[1], -item[0].first_seen), reverse=True)
    return ranked, phrase_present


def rag_token_count(encoding: Any, text: str) -> int:
    """Count persisted chunk text while treating model-control literals as text."""
    return len(encoding.encode(text, disallowed_special=()))


def pack_rag_chunk_context(
    ranked: list[tuple[MemoryObject, float, list[str]]],
    encoding: Any,
    budget: int | None,
) -> tuple[list[dict[str, Any]], str, int]:
    """Greedily cover gold signals, then render selected whole chunks in source order."""
    prepared = []
    for item in ranked:
        chunk, _score, _hits = item
        segment = (
            f'<RAG_CHUNK index={chunk.metadata.get("chunk_index")} '
            f'start_token={chunk.metadata.get("start_token")} '
            f'end_token={chunk.metadata.get("end_token")}>\n'
            f"{chunk.content}\n</RAG_CHUNK>\n\n"
        )
        prepared.append((item, segment, rag_token_count(encoding, segment)))
    remaining = list(prepared)
    chosen: list[tuple[MemoryObject, float, list[str], int]] = []
    covered: set[str] = set()
    used = 0
    skipped_oversize = sum(
        budget is not None and cost > budget for _item, _segment, cost in remaining
    )

    # With no cap every admitted chunk is retained; the final source order is
    # independent of the greedy coverage selection order.
    if budget is None and len({
        int(item[0].metadata.get("start_token", 0)) for item in ranked
    }) == len(ranked):
        chosen = [
            (chunk, score, hits, cost)
            for (chunk, score, hits), _segment, cost in prepared
        ]
        remaining = []

    while remaining:
        fitting = []
        for item, _segment, cost in remaining:
            _chunk, _score, hits = item
            if budget is not None and cost > budget:
                continue
            if budget is None or used + cost <= budget:
                fitting.append((item, cost, len(set(hits) - covered)))
        if not fitting:
            break
        best, cost, _new_hits = max(
            fitting,
            key=lambda value: (
                value[2] > 0,
                value[2],
                value[0][1],
                -value[0][0].first_seen,
            ),
        )
        remaining = [entry for entry in remaining if entry[0] is not best]
        chunk, score, hits = best
        chosen.append((chunk, score, hits, cost))
        covered.update(hits)
        used += cost

    chosen.sort(key=lambda item: int(item[0].metadata.get("start_token", 0)))
    selected = []
    segments = []
    for chunk, oracle_score, hits, _cost in chosen:
        segments.append(
            f'<RAG_CHUNK index={chunk.metadata.get("chunk_index")} '
            f'start_token={chunk.metadata.get("start_token")} '
            f'end_token={chunk.metadata.get("end_token")}>\n'
            f"{chunk.content}\n</RAG_CHUNK>\n\n"
        )
        selected.append(
            {
                "content": chunk.content,
                "score": oracle_score,
                "metadata": {
                    "oracle_exact_hits": hits,
                    "native_metadata": chunk.metadata,
                },
            }
        )
    return selected, "".join(segments), skipped_oversize


def build_rag(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    store, store_audit = load_rag_store(
        args.rag_run.resolve(), timestamp_maps, conversation_ids
    )
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        all_chunks = store.get(spec.conversation_id, [])
        # Chunk identity and overlap are part of the persisted RAG
        # representation, so unlike extracted memory objects these candidates
        # are not text-deduplicated.
        pool = [
            item
            for item in all_chunks
            if item.metadata.get("source_session_indices")
            and max(item.metadata["source_session_indices"]) <= spec.cutoff
        ]
        visible_evidence = {
            int(index)
            for index in episodes[spec.episode_index].get("evidence_session_indices") or []
            if int(index) <= spec.cutoff
        }
        ranked, phrase_present = rank_rag_chunks(pool, spec, visible_evidence)
        selected, context, skipped_oversize = pack_rag_chunk_context(
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
        cross_cutoff = sum(
            bool(item.metadata.get("source_session_indices"))
            and min(item.metadata["source_session_indices"]) <= spec.cutoff
            and max(item.metadata["source_session_indices"]) > spec.cutoff
            for item in all_chunks
        )
        metadata = {
            "system": "rag_meme_gold_fields_complete_chunk_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": rag_token_count(encoding, context),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_official_evidence_locations_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": "all phase-visible persisted fixed-token RAG chunks",
            "oracle_phase_safe": True,
            "oracle_whole_chunks_only": True,
            "oracle_selected_render_order": "ascending source start_token",
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_run_dir": str(args.rag_run.resolve()),
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
                "complete_store_conversation_size": len(all_chunks),
                "phase_visible_candidate_pool_size": len(pool),
                "visible_official_evidence_sessions": sorted(visible_evidence),
                "cross_cutoff_chunks_excluded": cross_cutoff,
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "selected_chunk_indices": [
                    item["metadata"]["native_metadata"]["chunk_index"]
                    for item in selected
                ],
                "oversize_chunks_skipped": skipped_oversize,
                "context_tokens": rag_token_count(encoding, context),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-rag-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built RAG complete chunk-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} chunks; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def build_mem0(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    store, store_audit = load_mem0_store(args.mem0_run.resolve(), timestamp_maps)
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
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
            "system": "mem0_local_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": "all phase-visible objects in Mem0 Local main memory collection",
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_run_dir": str(args.mem0_run.resolve()),
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
                "complete_store_conversation_size": len(
                    store.get(spec.conversation_id, [])
                ),
                "phase_visible_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-mem0_local-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built Mem0 Local complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} store objects; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def build_memos(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    native_namespace = validate_native_namespace(
        args.memos_native_run.resolve(), args.memos_prefix, separator="__"
    )
    store, store_audit, exported = load_memos_store(
        args.memos_api_url,
        args.memos_prefix,
        conversation_ids,
        timestamp_maps,
    )
    store_audit["native_namespace_validation"] = native_namespace
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
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
            "system": "memos_local_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": (
                "all phase-visible objects exported by Memos Local product/get_memory"
            ),
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_cube_prefix": args.memos_prefix,
            "native_run_dir": str(args.memos_native_run.resolve()),
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
                "complete_store_conversation_size": len(
                    store.get(spec.conversation_id, [])
                ),
                "phase_visible_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-memos_local-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    with (output_dir / "memos_complete_store_export.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exported, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built Memos Local complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} store objects; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def build_memu(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    native_namespace = validate_native_namespace(
        args.memu_run.resolve(), "normal", separator=":"
    )
    store, store_audit, exported = load_memu_store(
        args.memu_database_url,
        args.memu_run.resolve(),
        conversation_ids,
        timestamp_maps,
    )
    store_audit["native_namespace_validation"] = native_namespace
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
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
            "system": "memu_local_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": (
                "all phase-visible memory_items in exact normal:meme_* namespaces"
            ),
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_run_dir": str(args.memu_run.resolve()),
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
                "complete_store_conversation_size": len(
                    store.get(spec.conversation_id, [])
                ),
                "phase_visible_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-memu_local-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    with (output_dir / "memu_complete_store_export.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exported, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built MemU Local complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} store objects; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def build_emem(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    store, store_audit = load_emem_store(args.emem_run.resolve(), conversation_ids)
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = deduplicated_pool(store.get((spec.conversation_id, spec.phase), []))
        ranked, phrase_present = rank_memory_objects(pool, spec)
        budget = evidence_budget(
            encoding, args.token_budget, spec, args.task_aware_readout
        )
        if spec.task_type in {"Agg", "Del"}:
            # These tasks do not require temporal derivation.  Relevance-ordered
            # corroborating EDUs help the small reader identify membership or the
            # absence of a retained value; chronological compaction regressed both.
            focused = ranked
            selected, context = pack_memory_context(focused, encoding, budget)
            packing_policy = "legacy_relevance_order_full_ranked"
        else:
            focused = focus_emem_ranked_candidates(ranked, spec)
            selected, context = pack_emem_context(focused, encoding, budget)
            packing_policy = "gold_coverage_then_session_chronology"
        context = add_readout_policy(context, spec, args.task_aware_readout)
        support_phrases = [normalized_text(value) for value in spec.support_values]
        support_phrases = [value for value in support_phrases if value]
        support_covered = {
            phrase: any(phrase in normalized_text(item.content) for item in pool)
            for phrase in support_phrases
        }
        metadata = {
            "system": "emem_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": "all EDUs in the native phase-visible EMem snapshot",
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_run_dir": str(args.emem_run.resolve()),
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
                "phase_snapshot_store_size": len(
                    store.get((spec.conversation_id, spec.phase), [])
                ),
                "deduplicated_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "focused_candidate_count": len(focused),
                "selected_count": len(selected),
                "packing_policy": packing_policy,
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-emem-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built EMem complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_snapshot_objects']} "
        f"snapshot objects; all-support coverage {covered}/{len(audit)})"
    )


def build_evermemos(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    store, store_audit = load_evermemos_store(
        args.evermemos_run.resolve(), conversation_ids, timestamp_maps
    )
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
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
            "system": "evermemos_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": (
                "all phase-visible episode and atomic-fact text in EverMemOS stores"
            ),
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_run_dir": str(args.evermemos_run.resolve()),
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
                "complete_store_conversation_size": len(
                    store.get(spec.conversation_id, [])
                ),
                "phase_visible_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-evermemos-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built EverMemOS complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} store objects; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def build_memmachine(args: argparse.Namespace) -> None:
    episodes = load_json(args.data.resolve())
    if not 0 <= args.start_conv < args.end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    timestamp_maps = session_timestamp_maps(episodes)
    conversation_ids = [f"meme_{index}" for index in range(args.start_conv, args.end_conv)]
    store, store_audit, exported = load_memmachine_store(
        args.memmachine_api_url,
        args.memmachine_database_url,
        args.memmachine_org_id,
        args.memmachine_project_prefix,
        args.memmachine_run_name,
        conversation_ids,
        timestamp_maps,
    )
    encoding = tiktoken.get_encoding("o200k_base")

    rows = []
    audit = []
    for spec in specs:
        pool = phase_safe_pool(store.get(spec.conversation_id, []), spec.cutoff)
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
        selected_kind_counts: dict[str, int] = defaultdict(int)
        for item in selected:
            native_metadata = (item.get("metadata") or {}).get("native_metadata") or {}
            selected_kind_counts[str(native_metadata.get("memory_kind") or "unknown")] += 1
        metadata = {
            "system": "memmachine_meme_gold_fields_complete_store_oracle",
            "formatted_context": context,
            "top_k": len(selected),
            "context_token_budget": args.token_budget,
            "context_tokens_after_cap": len(encoding.encode(context)),
            "oracle_target_entities": list(spec.target_entities),
            "oracle_entity_values": spec.entity_values,
            "oracle_gold_answer_used_for_selection": True,
            "oracle_complete_store_candidate_access": True,
            "oracle_candidate_scope": (
                "all phase-visible episodic, semantic, and working-summary text "
                "in the persisted MemMachine project"
            ),
            "oracle_phase_safe": True,
            "question_phase": spec.phase,
            "question_cutoff_session": spec.cutoff,
            "task_aware_readout_policy": args.task_aware_readout,
            "store_namespace": {
                "org_id": args.memmachine_org_id,
                "project_prefix": args.memmachine_project_prefix,
                "run_name": args.memmachine_run_name,
            },
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
                "complete_store_conversation_size": len(
                    store.get(spec.conversation_id, [])
                ),
                "phase_visible_candidate_pool_size": len(pool),
                "ranked_candidate_count": len(ranked),
                "selected_count": len(selected),
                "selected_kind_counts": dict(selected_kind_counts),
                "context_tokens": len(encoding.encode(context)),
                "selection_phrase_coverage": phrase_present,
                "support_value_coverage": support_covered,
                "all_support_values_observed": all(support_covered.values()),
                "any_support_value_observed": any(support_covered.values()),
            }
        )

    output_dir = RESULTS_DIR / f"meme-memmachine-{args.tag}"
    create_output(output_dir, rows, audit)
    with (output_dir / "complete_store_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(store_audit, handle, ensure_ascii=False, indent=2)
    with (output_dir / "memmachine_complete_store_export.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(exported, handle, ensure_ascii=False, indent=2)
    covered = sum(row["all_support_values_observed"] for row in audit)
    print(
        f"Built MemMachine complete-store oracle: {output_dir} "
        f"({len(rows)} questions; {store_audit['total_objects']} store objects; "
        f"all-support coverage {covered}/{len(audit)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system",
        choices=("rag", "mem0", "memos", "memu", "emem", "evermemos", "memmachine"),
        default="mem0",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rag-run", type=Path, default=DEFAULT_RAG_RUN)
    parser.add_argument("--mem0-run", type=Path, default=DEFAULT_MEM0_RUN)
    parser.add_argument("--memos-native-run", type=Path, default=DEFAULT_MEMOS_RUN)
    parser.add_argument("--memos-api-url", default=DEFAULT_MEMOS_API_URL)
    parser.add_argument("--memos-prefix", default=DEFAULT_MEMOS_PREFIX)
    parser.add_argument("--memu-run", type=Path, default=DEFAULT_MEMU_RUN)
    parser.add_argument("--memu-database-url", default=DEFAULT_MEMU_DATABASE_URL)
    parser.add_argument("--emem-run", type=Path, default=DEFAULT_EMEM_RUN)
    parser.add_argument(
        "--evermemos-run", type=Path, default=DEFAULT_EVERMEMOS_RUN
    )
    parser.add_argument("--memmachine-api-url", default=DEFAULT_MEMMACHINE_API_URL)
    parser.add_argument("--memmachine-org-id", default=DEFAULT_MEMMACHINE_ORG_ID)
    parser.add_argument(
        "--memmachine-project-prefix", default=DEFAULT_MEMMACHINE_PROJECT_PREFIX
    )
    parser.add_argument("--memmachine-run-name", default=DEFAULT_MEMMACHINE_RUN_NAME)
    parser.add_argument(
        "--memmachine-database-url",
        default=os.environ.get("MEMMACHINE_DATABASE_URL", ""),
    )
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--end-conv", type=int, default=100)
    parser.add_argument("--token-budget", type=int, default=5000)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument(
        "--task-aware-readout", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.system == "rag":
        build_rag(args)
    elif args.system == "mem0":
        build_mem0(args)
    elif args.system == "memos":
        build_memos(args)
    elif args.system == "memu":
        build_memu(args)
    elif args.system == "emem":
        build_emem(args)
    elif args.system == "evermemos":
        build_evermemos(args)
    else:
        build_memmachine(args)


if __name__ == "__main__":
    main()
