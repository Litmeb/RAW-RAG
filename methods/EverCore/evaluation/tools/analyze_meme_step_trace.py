"""Question-level, multi-error MEME step tracing for agentic memory systems.

Unlike the older first-gap diagnostic, this tool judges every observable output
independently and records propagation, recovery, and newly introduced errors.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


EVALUATION_DIR = Path(__file__).resolve().parents[1]
EVERCORE_DIR = EVALUATION_DIR.parent
SCRIPTS_DIR = EVALUATION_DIR / "scripts"
for candidate in (EVERCORE_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from build_meme_complete_store_oracle_runs import (  # noqa: E402
    MemoryObject,
    canonical_timestamp,
    open_qdrant_readonly,
    phase_safe_pool,
    session_timestamp_maps,
    source_session_for_timestamp,
)
from build_meme_continuity_oracle_runs import (  # noqa: E402
    load_json,
    normalized_text,
    question_specs,
)
from evaluation.tools.meme_question_propagation_common import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_RUNS,
    LOADERS,
    MEME_TASKS,
    build_diagnostics,
    index_rows,
)


PROMPT_VERSION = "meme-step-trace-v5-mem0-entity-activation"
SYSTEM_LABELS = {
    "mem0_local": "Mem0 Local",
    "memos_local": "MemOS Local fine",
    "evermemos": "EverMemOS",
}
TERMINAL_STATES = {
    "pass",
    "fail",
    "not_observable",
    "replayable",
    "not_applicable",
    "judge_error",
}
SEMANTIC_STEPS = {
    "mem0_local": [
        ("official_source", "Official evidence is sufficient at the cutoff"),
        ("fact_store", "Mem0 extracted and persisted facts preserve the needed semantics"),
        (
            "entity_link_task_support",
            "Persisted entity-to-memory links adequately represent the task-relevant entities used by entity boosting",
        ),
        ("native_retrieval", "Mem0 fused retrieval after time filtering is sufficient"),
        ("reader_context", "The exact answer-reader context is sufficient"),
    ],
    "memos_local": [
        ("official_source", "Official evidence is sufficient at the cutoff"),
        (
            "multimodal_extraction_fidelity",
            "Persisted MultiModalStruct fine outputs faithfully preserve the task-relevant semantics in their source windows",
        ),
        (
            "state_invalidation_support",
            "For state-change tasks, the active store correctly applies update, cascade, abstraction, or deletion semantics",
        ),
        ("fine_memory_store", "The active MemOS fine store is sufficient for the official question"),
        ("native_retrieval", "Fast retrieval after MMR/rerank/time filtering is sufficient"),
        ("reader_context", "The exact answer-reader context is sufficient"),
    ],
    "evermemos": [
        ("official_source", "Official evidence is sufficient at the cutoff"),
        ("memcell_partition_quality", "The saved MemCell partition keeps task-dependent evidence together without harmful unrelated contamination"),
        ("episode_fidelity", "Episode extraction faithfully preserves the task-relevant semantics present in its MemCell input"),
        ("episode_task_support", "The saved episode representations are sufficient for the official question"),
        ("atomic_fact_fidelity", "Atomic-fact extraction faithfully preserves the task-relevant semantics present in its MemCell input"),
        ("atomic_fact_task_support", "The saved atomic-fact representations are sufficient for the official question"),
        ("query_refinement", "When executed, refined queries are semantically useful for recovering the missing answer evidence"),
        ("native_retrieval", "Final agentic retrieval is sufficient"),
        ("reader_context", "The exact answer-reader context is sufficient"),
    ],
}
EVERMEMOS_COMMON_PREFIX = [
    "official_source",
    "raw_boundary_decision_trace",
    "memcell_message_integrity",
    "memcell_cutoff_visibility",
    "memcell_partition_quality",
]
EVERMEMOS_RETRIEVAL_SUFFIX = [
    "round1_hybrid_candidates",
    "round1_rerank",
    "sufficiency_decision",
    "query_refinement",
    "round2_hybrid_rrf",
    "final_merge_rerank",
    "native_retrieval",
    "reader_context",
    "answer",
]
EVERMEMOS_PATHS = {
    "episode_path": [
        *EVERMEMOS_COMMON_PREFIX,
        "episode_fidelity",
        "episode_task_support",
        "episode_index_integrity",
        *EVERMEMOS_RETRIEVAL_SUFFIX,
    ],
    "atomic_fact_path": [
        *EVERMEMOS_COMMON_PREFIX,
        "atomic_fact_fidelity",
        "atomic_fact_task_support",
        "atomic_fact_index_integrity",
        *EVERMEMOS_RETRIEVAL_SUFFIX,
    ],
}


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def lexical_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalized_text(value)))


def memory_payload(memory: MemoryObject) -> dict[str, Any]:
    return {
        "memory_id": memory.metadata.get("memory_id"),
        "memory_kind": memory.metadata.get("memory_kind"),
        "source_session_indices": memory.metadata.get("source_session_indices") or [],
        "content": memory.content,
    }


def select_store_candidates(
    row: dict[str, Any], objects: list[MemoryObject], *, lexical_limit: int
) -> tuple[list[MemoryObject], dict[str, Any]]:
    """Select all official-source-linked objects plus lexical fallbacks.

    Provenance-linked objects make the semantic store check independent of
    whether the memory paraphrases the gold value. Lexical fallbacks cover
    objects whose source mapping spans or misses an evidence-session boundary.
    """
    evidence = set(row["official_source"]["evidence_session_indices"])
    linked = [
        obj
        for obj in objects
        if evidence.intersection(obj.metadata.get("source_session_indices") or [])
    ]
    query_parts = [row["question"], row["gold_answer"]]
    query_parts.extend(
        signal.get("value", "")
        for signal in row["meme_metadata"].get("required_value_signals") or []
    )
    query_parts.extend(row["meme_metadata"].get("target_entities") or [])
    query_tokens = lexical_tokens(" ".join(query_parts))
    linked_ids = {id(obj) for obj in linked}
    ranked = sorted(
        (obj for obj in objects if id(obj) not in linked_ids),
        key=lambda obj: len(query_tokens & lexical_tokens(obj.content)),
        reverse=True,
    )
    fallbacks = [
        obj
        for obj in ranked[:lexical_limit]
        if query_tokens & lexical_tokens(obj.content)
    ]
    selected = linked + fallbacks
    return selected, {
        "phase_visible_objects": len(objects),
        "provenance_linked_objects": len(linked),
        "lexical_fallback_objects": len(fallbacks),
        "selected_objects": len(selected),
        "selection": (
            "all objects whose source_session_indices intersect official evidence; "
            "plus gold/question/entity lexical fallbacks"
        ),
    }


def clip_items(items: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], bool]:
    """Cap store candidates by whole objects, never by head/tail text slicing."""
    result: list[dict[str, Any]] = []
    used = 0
    for item in items:
        size = len(json.dumps(item, ensure_ascii=False))
        if result and used + size > max_chars:
            return result, True
        result.append(item)
        used += size
    return result, len(result) < len(items)


def _conversation_number(conversation_id: str) -> int:
    match = re.search(r"(\d+)$", conversation_id)
    if not match:
        raise ValueError(f"Cannot parse MEME conversation id: {conversation_id}")
    return int(match.group(1))


def _message_dict(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    message = item.get("message", item)
    return message if isinstance(message, dict) else {}


def _message_role(message: dict[str, Any]) -> str:
    role = str(
        message.get("role")
        or message.get("sender_name")
        or message.get("user_name")
        or message.get("sender_id")
        or ""
    ).lower()
    if "assistant" in role:
        return "assistant"
    if "user" in role:
        return "user"
    return role


def _message_key(message: dict[str, Any]) -> tuple[str, str, str]:
    return (
        canonical_timestamp(message.get("timestamp") or message.get("chat_time")),
        _message_role(message),
        str(message.get("content") or ""),
    )


def _official_message_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for session in row["official_source"]["sessions"]:
        base_text = canonical_timestamp(session.get("timestamp"))
        base = datetime.fromisoformat(base_text) if base_text else None
        for position, message in enumerate(session.get("conversation") or []):
            timestamp = (
                (base + timedelta(seconds=position * 30)).isoformat()
                if base is not None
                else ""
            )
            record = {
                "session_index": int(session["session_index"]),
                "position": position,
                "role": str(message.get("role") or "").lower(),
                "content": str(message.get("content") or ""),
                "timestamp": timestamp,
            }
            record["key"] = _message_key(record)
            record["message"] = {
                key: record[key]
                for key in ("session_index", "position", "role", "content", "timestamp")
            }
            records.append(record)
    return records


def load_mem0_step_artifacts(
    run_dir: Path,
    store: dict[str, list[MemoryObject]],
    selected_conversations: set[str],
) -> dict[str, Any]:
    """Load persisted Mem0 history and entity links without replaying the SDK.

    The history table contains post-embedding/post-hash-dedup records, not the
    raw extraction response. Qdrant contains the final memory and entity
    collections. Keeping these views separate lets the audit expose what is
    genuinely recorded without pretending to recover rejected candidates.
    """
    history_path = run_dir / "mem0_history.db"
    if not history_path.is_file():
        raise FileNotFoundError(f"Mem0 history database not found: {history_path}")
    connection = sqlite3.connect(
        f"file:{history_path.resolve().as_posix()}?mode=ro", uri=True
    )
    try:
        history_rows = connection.execute(
            "SELECT memory_id, old_memory, new_memory, event, created_at, "
            "updated_at, is_deleted FROM history"
        ).fetchall()
    finally:
        connection.close()

    add_records = {
        str(memory_id): {
            "memory_id": str(memory_id),
            "memory": str(new_memory or ""),
            "created_at": canonical_timestamp(created_at),
        }
        for memory_id, _old, new_memory, event, created_at, _updated, _deleted in history_rows
        if str(event).upper() == "ADD"
    }
    deleted_ids = {
        str(memory_id)
        for memory_id, _old, _new, event, _created, _updated, _deleted in history_rows
        if str(event).upper() == "DELETE"
    }
    active_history_ids = set(add_records) - deleted_ids

    memory_by_id: dict[str, MemoryObject] = {}
    for conversation_id, memories in store.items():
        if conversation_id not in selected_conversations:
            continue
        for memory in memories:
            memory_id = str(memory.metadata.get("memory_id") or "")
            if memory_id:
                memory_by_id[memory_id] = memory
    current_ids = set(memory_by_id)

    entity_by_memory: dict[str, list[dict[str, str]]] = defaultdict(list)
    broken_entity_links: list[dict[str, str]] = []
    entity_count = 0
    entity_link_count = 0
    with open_qdrant_readonly(run_dir) as client:
        collection_names = [item.name for item in client.get_collections().collections]
        entity_collections = [
            name for name in collection_names if name.endswith("_entities")
        ]
        if len(entity_collections) != 1:
            raise ValueError(
                f"Expected one Mem0 entity collection, found {entity_collections}"
            )
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=entity_collections[0],
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                run_id = str(payload.get("run_id") or "")
                if run_id and run_id not in selected_conversations:
                    continue
                entity_count += 1
                entity = {
                    "entity_id": str(point.id),
                    "text": str(payload.get("data") or ""),
                    "entity_type": str(payload.get("entity_type") or ""),
                }
                for linked_id in payload.get("linked_memory_ids") or []:
                    memory_id = str(linked_id)
                    entity_link_count += 1
                    entity_by_memory[memory_id].append(entity)
                    if memory_id not in current_ids:
                        broken_entity_links.append(
                            {
                                "entity_id": entity["entity_id"],
                                "entity": entity["text"],
                                "memory_id": memory_id,
                            }
                        )
            if offset is None:
                break

    duplicate_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for memory_id, memory in memory_by_id.items():
        payload = memory.metadata.get("store_payload") or {}
        memory_hash = str(payload.get("hash") or "")
        user_id = str(payload.get("user_id") or "")
        if memory_hash:
            duplicate_groups[(user_id, memory_hash)].append(memory_id)
    residual_duplicates = {
        key: ids for key, ids in duplicate_groups.items() if len(ids) > 1
    }

    return {
        "history_add_records": add_records,
        "history_deleted_ids": deleted_ids,
        "active_history_ids": active_history_ids,
        "memory_by_id": memory_by_id,
        "entity_by_memory": dict(entity_by_memory),
        "broken_entity_links": broken_entity_links,
        "residual_duplicate_groups": residual_duplicates,
        "audit": {
            "history_path": str(history_path.resolve()),
            "history_add_events": len(add_records),
            "history_delete_events": len(deleted_ids),
            "selected_current_memories": len(current_ids),
            "selected_qdrant_ids_without_add_history": len(current_ids - set(add_records)),
            # History has no conversation/user metadata, so an active ADD that
            # is absent from Qdrant cannot be assigned safely to one question.
            "global_active_history_ids_without_current_vector": len(
                active_history_ids
                - {
                    str(memory.metadata.get("memory_id"))
                    for memories in store.values()
                    for memory in memories
                    if memory.metadata.get("memory_id")
                }
            ),
            "entity_objects": entity_count,
            "entity_links": entity_link_count,
            "broken_entity_links": len(broken_entity_links),
            "memories_with_entity_links": len(current_ids & set(entity_by_memory)),
            "memories_without_entity_links": len(current_ids - set(entity_by_memory)),
            "residual_exact_duplicate_groups": len(residual_duplicates),
            "residual_exact_duplicate_points_beyond_first": sum(
                len(ids) - 1 for ids in residual_duplicates.values()
            ),
        },
    }


def build_mem0_question_observation(
    row: dict[str, Any],
    conversation_objects: list[MemoryObject],
    selected_store: list[MemoryObject],
    artifact: dict[str, Any],
    timestamp_map: list[tuple[str, int]],
) -> dict[str, Any]:
    """Build deterministic Mem0 steps from history, Qdrant, and entity links."""
    from mem0.utils.entity_extraction import extract_entities

    query_entities = [
        {"entity_type": entity_type, "text": entity_text}
        for entity_type, entity_text in extract_entities(row["question"])
    ]
    selected_ids = {
        str(memory.metadata.get("memory_id"))
        for memory in selected_store
        if memory.metadata.get("memory_id")
    }
    add_records = artifact["history_add_records"]
    missing_history_ids = sorted(selected_ids - set(add_records))
    history_lineage = _deterministic_step(
        "post_dedup_history_lineage",
        "Every task-relevant Qdrant memory has a post-dedup ADD history record",
        not missing_history_ids,
        {
            "task_relevant_qdrant_memories": len(selected_ids),
            "missing_add_history_ids": missing_history_ids[:20],
            "missing_add_history_ids_capped": len(missing_history_ids) > 20,
            "semantics": (
                "history ADD is written from records after extraction, embedding, and "
                "exact-hash filtering; it is not the raw LLM response"
            ),
        },
    )

    evidence_sessions = set(row["official_source"]["evidence_session_indices"])
    required_signals = [
        str(signal.get("normalized") or "")
        for signal in row["meme_metadata"].get("required_value_signals") or []
        if signal.get("normalized")
    ]

    def task_relevant(memory: MemoryObject) -> bool:
        sessions = set(memory.metadata.get("source_session_indices") or [])
        content = normalized_text(memory.content)
        return bool(evidence_sessions & sessions) or any(
            signal in content for signal in required_signals
        )

    visible_by_id = {
        str(memory.metadata.get("memory_id")): memory
        for memory in conversation_objects
        if memory.metadata.get("memory_id")
        and max(memory.metadata.get("source_session_indices") or [-1])
        <= row["cutoff_session"]
    }
    relevant_duplicate_groups = []
    for (user_id, memory_hash), ids in artifact["residual_duplicate_groups"].items():
        visible_ids = [memory_id for memory_id in ids if memory_id in visible_by_id]
        if len(visible_ids) < 2 or not any(
            task_relevant(visible_by_id[memory_id]) for memory_id in visible_ids
        ):
            continue
        relevant_duplicate_groups.append(
            {
                "user_id": user_id,
                "hash": memory_hash,
                "memory_ids": visible_ids,
                "content": visible_by_id[visible_ids[0]].content,
            }
        )
    residual_dedup = _deterministic_step(
        "task_relevant_residual_exact_duplicates",
        "No task-relevant exact-hash duplicate survives in the cutoff-visible store",
        not relevant_duplicate_groups,
        {
            "duplicate_groups": len(relevant_duplicate_groups),
            "duplicate_points_beyond_first": sum(
                len(group["memory_ids"]) - 1 for group in relevant_duplicate_groups
            ),
            "examples": relevant_duplicate_groups[:10],
            "scope": (
                "same user and MD5 hash; at least one duplicate is linked to an "
                "official evidence session or contains a required value signal"
            ),
        },
    )

    entity_by_memory = artifact["entity_by_memory"]
    relevant_entity_links = {
        memory_id: entity_by_memory.get(memory_id, []) for memory_id in selected_ids
    }
    broken_relevant_links = [
        item
        for item in artifact["broken_entity_links"]
        if item["memory_id"] in selected_ids
    ]
    entity_integrity = _deterministic_step(
        "entity_link_integrity",
        "Task-relevant persisted entity links point to existing memories",
        not broken_relevant_links,
        {
            "task_relevant_memories": len(selected_ids),
            "memories_with_entity_links": sum(
                bool(entities) for entities in relevant_entity_links.values()
            ),
            "memories_without_entity_links": sum(
                not entities for entities in relevant_entity_links.values()
            ),
            "entity_links": sum(len(items) for items in relevant_entity_links.values()),
            "broken_relevant_links": broken_relevant_links[:20],
        },
    )

    cross_cutoff_results = []
    unknown_timestamp_results = 0
    for result in row["native_retrieval"]["results"]:
        raw = ((result.get("metadata") or {}).get("raw") or {})
        timestamp = canonical_timestamp(
            raw.get("created_at")
            or (result.get("metadata") or {}).get("created_at")
        )
        session = source_session_for_timestamp(timestamp_map, timestamp)
        if session is None:
            unknown_timestamp_results += 1
        elif session > row["cutoff_session"]:
            cross_cutoff_results.append(
                {
                    "memory_id": raw.get("id")
                    or (result.get("metadata") or {}).get("id"),
                    "source_session_index": session,
                    "created_at": timestamp,
                    "content": str(result.get("content") or "")[:300],
                }
            )
    temporal_integrity = _deterministic_step(
        "temporal_filter_integrity",
        "Every native result is at or before the official cutoff",
        not cross_cutoff_results,
        {
            "native_results": len(row["native_retrieval"]["results"]),
            "cross_cutoff_results": len(cross_cutoff_results),
            "unknown_timestamp_results": unknown_timestamp_results,
            "cross_cutoff_examples": cross_cutoff_results[:5],
        },
    )

    semantic_entities = []
    for memory in selected_store:
        memory_id = str(memory.metadata.get("memory_id") or "")
        semantic_entities.append(
            {
                **memory_payload(memory),
                "entities": relevant_entity_links.get(memory_id, []),
            }
        )
    return {
        "deterministic_steps": [
            history_lineage,
            residual_dedup,
            entity_integrity,
            temporal_integrity,
        ],
        "semantic_artifact": {
            "query_entities": query_entities,
            "entity_boost_executed": bool(query_entities),
            "task_relevant_memories_with_entities": semantic_entities,
            "step_views": {
                "entity_link_task_support": (
                    "Judge only whether the saved entity labels and links adequately "
                    "represent target/query entities for entity-boost retrieval. Do not "
                    "use the memory text itself to excuse a missing or wrong link."
                )
            },
        },
        "audit": {
            "selected_memory_ids": sorted(selected_ids),
            "missing_add_history_ids": missing_history_ids,
            "task_relevant_duplicate_groups": relevant_duplicate_groups,
            "query_entities": query_entities,
            "entity_boost_executed": bool(query_entities),
        },
    }


def _memos_metadata(memory: MemoryObject) -> dict[str, Any]:
    metadata = memory.metadata.get("store_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _memos_sources(memory: MemoryObject) -> list[dict[str, Any]]:
    result = []
    for source in _memos_metadata(memory).get("sources") or []:
        if isinstance(source, dict):
            result.append(source)
        elif isinstance(source, str):
            try:
                parsed = json.loads(source)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                result.append(parsed)
    return result


def load_memos_qdrant_ids(
    base_url: str,
    collection: str,
    conversation_ids: set[str],
    prefix: str,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Read every Qdrant ID for the selected live MemOS cubes."""
    endpoint = f"{base_url.rstrip('/')}/collections/{collection}/points/scroll"
    result: dict[str, set[str]] = {}
    total = 0
    for position, conversation_id in enumerate(sorted(conversation_ids), start=1):
        number = _conversation_number(conversation_id)
        cube_id = f"{prefix}__meme_{number}_User"
        ids: set[str] = set()
        offset: Any = None
        while True:
            payload: dict[str, Any] = {
                "filter": {"must": [{"key": "user_name", "match": {"value": cube_id}}]},
                "limit": 1000,
                "with_payload": False,
                "with_vector": False,
            }
            if offset is not None:
                payload["offset"] = offset
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            page = (body.get("result") or {}).get("points") or []
            ids.update(str(point.get("id")) for point in page if point.get("id"))
            offset = (body.get("result") or {}).get("next_page_offset")
            if offset is None:
                break
        result[conversation_id] = ids
        total += len(ids)
        if position % 10 == 0 or position == len(conversation_ids):
            print(
                f"Read Memos Qdrant IDs: {position}/{len(conversation_ids)} "
                f"({total} points)"
            )
    return result, {
        "base_url": base_url,
        "collection": collection,
        "conversation_count": len(result),
        "points": total,
        "method": "Qdrant scroll filtered by exact user_name; payload/vector bodies omitted",
    }


def _memos_window_groups(objects: list[MemoryObject]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for memory in objects:
        metadata = _memos_metadata(memory)
        sources = _memos_sources(memory)
        signature = json.dumps(
            {
                "sources": sources,
                "background": metadata.get("background") or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        group = groups.setdefault(
            signature,
            {
                "sources": sources,
                "background": metadata.get("background") or "",
                "outputs": [],
            },
        )
        group["outputs"].append(
            {
                "memory_id": memory.metadata.get("memory_id"),
                "memory": memory.content,
                "key": metadata.get("key") or "",
                "tags": metadata.get("tags") or [],
                "memory_type": metadata.get("memory_type") or metadata.get("type") or "",
                "status": metadata.get("status") or "",
            }
        )
    return list(groups.values())


def build_memos_question_observation(
    row: dict[str, Any],
    conversation_objects: list[MemoryObject],
    selected_store: list[MemoryObject],
    qdrant_ids: set[str],
    timestamp_map: list[tuple[str, int]],
    *,
    max_store_chars: int,
) -> dict[str, Any]:
    """Build deterministic and semantic views from the live MemOS Docker stores."""
    expected_records = _official_message_records(row)
    actual_keys: Counter[tuple[str, str, str]] = Counter()
    for memory in conversation_objects:
        actual_keys.update(_message_key(source) for source in _memos_sources(memory))
    missing_messages = [
        record for record in expected_records if actual_keys[record["key"]] == 0
    ]
    duplicate_references = sum(
        max(0, actual_keys[record["key"]] - 1) for record in expected_records
    )
    source_step = _deterministic_step(
        "multimodal_window_provenance",
        "Official evidence messages survive in persisted MultiModalStruct source windows",
        not missing_messages,
        {
            "expected_evidence_messages": len(expected_records),
            "missing_evidence_messages": len(missing_messages),
            "duplicate_source_references": duplicate_references,
            "missing_examples": [record["message"] for record in missing_messages[:5]],
            "runtime_pipeline": (
                "multimodal_struct default; per-message fast parse; 1600-token windows "
                "with 200-token overlap; window-level four-way fine extraction plus "
                "per-source transfer"
            ),
        },
    )

    relevant_ids = {
        str(memory.metadata.get("memory_id"))
        for memory in selected_store
        if memory.metadata.get("memory_id")
    }
    missing_vector_ids = sorted(relevant_ids - qdrant_ids)
    vector_step = _deterministic_step(
        "graph_vector_persistence",
        "Evidence-linked Neo4j memory IDs have matching Qdrant points",
        not missing_vector_ids,
        {
            "evidence_linked_graph_objects": len(relevant_ids),
            "qdrant_points_in_conversation": len(qdrant_ids),
            "missing_vector_ids": missing_vector_ids[:20],
            "missing_vector_ids_capped": len(missing_vector_ids) > 20,
        },
    )

    statuses = Counter()
    history_nonempty = 0
    evolve_nonempty = 0
    for memory in conversation_objects:
        metadata = _memos_metadata(memory)
        statuses[str(metadata.get("status") or "unknown")] += 1
        history_nonempty += bool(metadata.get("history"))
        evolve_nonempty += bool(metadata.get("evolve_to"))
    evolution_used = bool(history_nonempty or evolve_nonempty or statuses.get("archived"))
    evolution_step = {
        "step_id": "state_evolution_mechanism",
        "label": "Merge/archive/version-history mechanism",
        "status": "pass" if evolution_used else "not_applicable",
        "reason": (
            "evolution telemetry is present in the live store"
            if evolution_used
            else "this run has no archived memories, history, or evolve_to links; the normal MultiModalStruct merge helper returns the extraction unchanged"
        ),
        "details": {
            "conversation_objects": len(conversation_objects),
            "status_counts": dict(statuses),
            "history_nonempty": history_nonempty,
            "evolve_to_nonempty": evolve_nonempty,
        },
    }

    cross_cutoff_results = []
    unknown_timestamp_results = 0
    for result in row["native_retrieval"]["results"]:
        raw = ((result.get("metadata") or {}).get("raw") or {})
        raw_metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        source_sessions = []
        for source in raw_metadata.get("sources") or []:
            if not isinstance(source, dict) or not source.get("chat_time"):
                continue
            session = source_session_for_timestamp(
                timestamp_map, canonical_timestamp(source.get("chat_time"))
            )
            if session is not None:
                source_sessions.append(session)
        if not source_sessions:
            unknown_timestamp_results += 1
        elif max(source_sessions) > row["cutoff_session"]:
            cross_cutoff_results.append(
                {
                    "memory_id": (result.get("metadata") or {}).get("memory_id"),
                    "source_session_indices": sorted(set(source_sessions)),
                    "content": str(result.get("content") or "")[:300],
                }
            )
    temporal_step = _deterministic_step(
        "temporal_filter_integrity",
        "Every native result is fully supported by sources at or before the cutoff",
        not cross_cutoff_results,
        {
            "native_results": len(row["native_retrieval"]["results"]),
            "cross_cutoff_results": len(cross_cutoff_results),
            "unknown_timestamp_results": unknown_timestamp_results,
            "cross_cutoff_examples": cross_cutoff_results[:5],
            "strict_rule": "max(metadata.sources[].chat_time session) <= cutoff_session",
        },
    )

    windows = _memos_window_groups(selected_store)
    clipped_windows, windows_clipped = clip_items(windows, max_store_chars)
    state_task = row["task_type"] in {"Del", "Cas", "Abs"}
    return {
        "deterministic_steps": [source_step, vector_step, evolution_step, temporal_step],
        "semantic_artifact": {
            "relevant_extraction_groups": clipped_windows,
            "groups_before_clip": len(windows),
            "groups_clipped": windows_clipped,
            "state_change_task": state_task,
            "step_views": {
                "multimodal_extraction_fidelity": (
                    "Compare each group's sources with its persisted outputs; judge omission, "
                    "hallucination, entity/time/relation corruption, and harmful duplication."
                ),
                "state_invalidation_support": (
                    "For Del/Cas/Abs only, judge active outputs under official before/after and dependency semantics."
                ),
                "fine_memory_store": (
                    "Ignore source text and judge only persisted outputs plus their metadata."
                ),
            },
        },
        "audit": {
            "evidence_linked_memory_ids": sorted(relevant_ids),
            "qdrant_missing_ids": missing_vector_ids,
            "state_change_task": state_task,
        },
    }


def _doc_map(rows: list[Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        doc = row.get("doc") if isinstance(row, dict) and isinstance(row.get("doc"), dict) else row
        if isinstance(doc, dict) and doc.get("event_id") is not None:
            result[str(doc["event_id"])] = doc
    return result


def load_evermemos_step_artifacts(
    run_dir: Path,
    conversation_ids: set[str],
    timestamp_maps: dict[str, list[tuple[str, int]]],
) -> dict[str, dict[str, Any]]:
    """Load final MemCells plus both persisted retrieval indexes.

    These are trusted artifacts produced by the selected local run. Only
    conversations present in the question manifest are loaded, which avoids
    reading every 30–40 MB MemCell JSON for a small audit.
    """
    result = {}
    for conversation_id in sorted(conversation_ids, key=_conversation_number):
        number = _conversation_number(conversation_id)
        memcell_path = run_dir / "memcells" / f"memcell_list_conv_{number}.json"
        vector_path = run_dir / "vectors" / f"embedding_index_conv_{number}.pkl"
        bm25_path = run_dir / "bm25_index" / f"bm25_index_conv_{number}.pkl"
        for path in (memcell_path, vector_path, bm25_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing EverMemOS step artifact: {path}")
        memcells = load_json(memcell_path)
        if not isinstance(memcells, list):
            raise ValueError(f"EverMemOS MemCell artifact must be a list: {memcell_path}")
        with vector_path.open("rb") as handle:
            vector_rows = pickle.load(handle)  # noqa: S301 - trusted local run artifact
        with bm25_path.open("rb") as handle:
            bm25_payload = pickle.load(handle)  # noqa: S301 - trusted local run artifact
        bm25_rows = bm25_payload.get("docs") if isinstance(bm25_payload, dict) else None
        if not isinstance(vector_rows, list) or not isinstance(bm25_rows, list):
            raise ValueError(f"Malformed EverMemOS index artifact for {conversation_id}")

        enriched = []
        for cell in memcells:
            messages = [_message_dict(item) for item in cell.get("original_data") or []]
            messages = [message for message in messages if message]
            source_sessions = sorted(
                {
                    session
                    for message in messages
                    if (
                        session := source_session_for_timestamp(
                            timestamp_maps.get(conversation_id, []),
                            canonical_timestamp(message.get("timestamp")),
                        )
                    )
                    is not None
                }
            )
            atomic = cell.get("atomic_fact") or {}
            facts = atomic.get("atomic_fact") or [] if isinstance(atomic, dict) else []
            enriched.append(
                {
                    "event_id": str(cell.get("event_id") or ""),
                    "source_session_indices": source_sessions,
                    "messages": messages,
                    "subject": str(cell.get("subject") or ""),
                    "summary": str(cell.get("summary") or ""),
                    "episode": str(cell.get("episode") or ""),
                    "atomic_facts": [str(fact) for fact in facts if str(fact).strip()],
                    "timestamp": str(cell.get("timestamp") or ""),
                    "atomic_metadata": {
                        key: str(atomic.get(key) or "")
                        for key in ("timestamp", "time")
                    }
                    if isinstance(atomic, dict)
                    else {},
                }
            )
        result[conversation_id] = {
            "memcells": enriched,
            "vector_docs": _doc_map(vector_rows),
            "bm25_docs": _doc_map(bm25_rows),
            "paths": {
                "memcells": str(memcell_path.resolve()),
                "vector": str(vector_path.resolve()),
                "bm25": str(bm25_path.resolve()),
            },
        }
    return result


def _compact_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": _message_role(message),
        "timestamp": canonical_timestamp(message.get("timestamp")),
        "content": str(message.get("content") or ""),
    }


def _deterministic_step(
    step_id: str, label: str, passed: bool, details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "label": label,
        "status": "pass" if passed else "fail",
        "reason": "deterministic artifact comparison",
        "details": details,
    }


def _index_integrity(
    relevant_cells: list[dict[str, Any]],
    index_docs: dict[str, dict[str, Any]],
    kind: str,
) -> dict[str, Any]:
    missing_events = []
    mismatched_events = []
    checked = 0
    for cell in relevant_cells:
        event_id = cell["event_id"]
        expected_episode = tuple(
            normalized_text(cell[key]) for key in ("subject", "summary", "episode")
        )
        expected_facts = Counter(normalized_text(fact) for fact in cell["atomic_facts"])
        if kind == "episode" and not any(expected_episode):
            continue
        if kind == "atomic_fact" and not expected_facts:
            continue
        checked += 1
        doc = index_docs.get(event_id)
        if doc is None:
            missing_events.append(event_id)
            continue
        if kind == "episode":
            observed_episode = tuple(
                normalized_text(str(doc.get(key) or ""))
                for key in ("subject", "summary", "episode")
            )
            mismatch = observed_episode != expected_episode
        else:
            atomic = doc.get("atomic_fact") or {}
            facts = atomic.get("atomic_fact") or [] if isinstance(atomic, dict) else []
            observed_metadata = {
                key: normalized_text(str(atomic.get(key) or ""))
                for key in ("timestamp", "time")
            } if isinstance(atomic, dict) else {}
            expected_metadata = {
                key: normalized_text(value)
                for key, value in cell["atomic_metadata"].items()
            }
            mismatch = (
                Counter(normalized_text(str(fact)) for fact in facts) != expected_facts
                or observed_metadata != expected_metadata
            )
        if mismatch:
            mismatched_events.append(event_id)
    return {
        "checked_events": checked,
        "missing_event_ids": missing_events[:20],
        "missing_event_count": len(missing_events),
        "content_mismatch_event_ids": mismatched_events[:20],
        "content_mismatch_event_count": len(mismatched_events),
        "pass": not missing_events and not mismatched_events,
    }


def build_evermemos_question_observation(
    row: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    expected_records = _official_message_records(row)
    expected_keys = Counter(
        (record["timestamp"], record["role"], record["content"])
        for record in expected_records
    )
    all_cells = artifact["memcells"]
    all_occurrences: Counter[tuple[str, str, str]] = Counter()
    visible_occurrences: Counter[tuple[str, str, str]] = Counter()
    relevant_cells = []
    visible_relevant_cells = []
    cross_cutoff_cells = []
    expected_key_set = set(expected_keys)
    for cell in all_cells:
        keys = [_message_key(message) for message in cell["messages"]]
        all_occurrences.update(keys)
        is_relevant = bool(expected_key_set.intersection(keys))
        is_visible = bool(cell["source_session_indices"]) and max(
            cell["source_session_indices"]
        ) <= row["cutoff_session"]
        if is_visible:
            visible_occurrences.update(keys)
        if is_relevant:
            relevant_cells.append(cell)
            if is_visible:
                visible_relevant_cells.append(cell)
            else:
                cross_cutoff_cells.append(cell["event_id"])

    missing = list((expected_keys - all_occurrences).elements())
    duplicate = [
        key for key, count in all_occurrences.items() if key in expected_keys and count > expected_keys[key]
    ]
    invisible = list((expected_keys - visible_occurrences).elements())
    message_integrity = _deterministic_step(
        "memcell_message_integrity",
        "Official evidence messages preserved exactly once in saved MemCells",
        not missing and not duplicate,
        {
            "expected_evidence_messages": sum(expected_keys.values()),
            "missing_count": len(missing),
            "duplicate_count": len(duplicate),
            "missing_examples": [list(item) for item in missing[:5]],
            "duplicate_examples": [list(item) for item in duplicate[:5]],
        },
    )
    cutoff_visibility = _deterministic_step(
        "memcell_cutoff_visibility",
        "Official evidence remains visible after MemCell-level time filtering",
        not invisible,
        {
            "expected_evidence_messages": sum(expected_keys.values()),
            "invisible_message_count": len(invisible),
            "cross_cutoff_memcell_ids": cross_cutoff_cells[:20],
            "cross_cutoff_memcell_count": len(cross_cutoff_cells),
        },
    )

    vector_episode = _index_integrity(relevant_cells, artifact["vector_docs"], "episode")
    bm25_episode = _index_integrity(relevant_cells, artifact["bm25_docs"], "episode")
    vector_atomic = _index_integrity(relevant_cells, artifact["vector_docs"], "atomic_fact")
    bm25_atomic = _index_integrity(relevant_cells, artifact["bm25_docs"], "atomic_fact")
    episode_index = _deterministic_step(
        "episode_index_integrity",
        "Relevant episodes persisted identically in vector and BM25 indexes",
        vector_episode["pass"] and bm25_episode["pass"],
        {"vector": vector_episode, "bm25": bm25_episode},
    )
    atomic_index = _deterministic_step(
        "atomic_fact_index_integrity",
        "Relevant atomic facts persisted identically in vector and BM25 indexes",
        vector_atomic["pass"] and bm25_atomic["pass"],
        {"vector": vector_atomic, "bm25": bm25_atomic},
    )

    semantic_cells = [
        {
            "event_id": cell["event_id"],
            "source_session_indices": cell["source_session_indices"],
            "visible_at_cutoff": cell in visible_relevant_cells,
            "memcell_timestamp": cell["timestamp"],
            "original_messages": [_compact_message(message) for message in cell["messages"]],
            "subject": cell["subject"],
            "summary": cell["summary"],
            "episode": cell["episode"],
            "atomic_facts": cell["atomic_facts"],
            "atomic_metadata": cell["atomic_metadata"],
        }
        for cell in relevant_cells
    ]
    return {
        "deterministic_steps": [
            message_integrity,
            cutoff_visibility,
            episode_index,
            atomic_index,
        ],
        "semantic_artifact": {
            "relevant_memcells": semantic_cells,
            "partition_metrics": {
                "relevant_memcell_count": len(relevant_cells),
                "visible_relevant_memcell_count": len(visible_relevant_cells),
                "evidence_messages": len(expected_records),
                "cross_cutoff_memcell_count": len(cross_cutoff_cells),
            },
            "step_views": {
                "memcell_partition_quality": "Judge grouping using relevant_memcells[].original_messages and visibility.",
                "episode_fidelity": "Compare each relevant_memcell's original_messages with its episode.",
                "episode_task_support": "Judge the visible_at_cutoff episode outputs jointly.",
                "atomic_fact_fidelity": "Compare each relevant_memcell's original_messages with its atomic_facts.",
                "atomic_fact_task_support": "Judge the visible_at_cutoff atomic_facts jointly.",
            },
        },
        "audit": {
            "relevant_memcell_ids": [cell["event_id"] for cell in relevant_cells],
            "visible_relevant_memcell_ids": [cell["event_id"] for cell in visible_relevant_cells],
            "artifact_paths": artifact["paths"],
        },
    }


def build_judge_payload(
    system: str,
    row: dict[str, Any],
    selected_store: list[MemoryObject],
    search: dict[str, Any],
    *,
    max_store_chars: int,
    evermemos_observation: dict[str, Any] | None = None,
    memos_observation: dict[str, Any] | None = None,
    mem0_observation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    official = row["official_source"]["sessions"]
    common = {
        "official_source": official,
        "native_retrieval": row["native_retrieval"]["results"],
        "reader_context": row["reader_context"]["text"],
    }
    objects = [memory_payload(obj) for obj in selected_store]
    clipped = False
    if system == "evermemos":
        if evermemos_observation is None:
            raise ValueError("EverMemOS semantic observation is required")
        semantic_artifact = evermemos_observation["semantic_artifact"]
        common["memcell_write_artifact"] = {
            **semantic_artifact,
            # Every official-evidence-linked MemCell is retained. Cutting this
            # list would recreate the incomplete-evidence flaw this audit is
            # intended to avoid.
            "artifact_clipped": False,
        }
        clipped = False
    elif system == "memos_local":
        if memos_observation is None:
            raise ValueError("MemOS Docker observation is required")
        semantic_artifact = memos_observation["semantic_artifact"]
        common["memos_multimodal_write_artifact"] = semantic_artifact
        common["fine_memory_store"], store_clipped = clip_items(objects, max_store_chars)
        clipped = bool(semantic_artifact["groups_clipped"] or store_clipped)
    elif system == "mem0_local":
        if mem0_observation is None:
            raise ValueError("Mem0 artifact observation is required")
        common["fact_store"], clipped = clip_items(objects, max_store_chars)
        common["mem0_entity_link_artifact"] = mem0_observation[
            "semantic_artifact"
        ]
    else:
        raise ValueError(f"Unsupported system: {system}")
    payload = {
        "benchmark": {
            "question_id": row["question_id"],
            "phase": row["phase"],
            "cutoff_session": row["cutoff_session"],
            "task_type": row["task_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "meme_metadata": row["meme_metadata"],
        },
        "steps": common,
    }
    retrieval_metadata = search.get("retrieval_metadata") or {}
    agent_metadata = {
        key: retrieval_metadata.get(key)
        for key in (
            "is_multi_round",
            "is_sufficient",
            "reasoning",
            "missing_info",
            "key_information_found",
            "refined_queries",
            "query_strategy",
            "round1_count",
            "round2_count",
            "final_count",
        )
        if key in retrieval_metadata
    }
    if system == "evermemos":
        common["query_refinement"] = agent_metadata
    return payload, {
        "store_candidates_clipped": clipped,
        "official_source_chars": len(json.dumps(official, ensure_ascii=False)),
        "judge_payload_chars": len(json.dumps(payload, ensure_ascii=False)),
        "agent_metadata": agent_metadata,
    }


def judge_prompt(system: str, payload: dict[str, Any]) -> str:
    step_lines = "\n".join(
        f'- `{step_id}`: {description}' for step_id, description in SEMANTIC_STEPS[system]
    )
    return f"""You are auditing a MEME benchmark run of {SYSTEM_LABELS[system]}.

Judge EVERY listed step independently. Do not stop after the first failure and do not assume that an upstream failure forces a downstream failure. A step passes only if its own output contains enough correct information to derive the official gold answer under the supplied phase, cutoff, entity-state, deletion, cascade, abstraction, and dependency semantics. Paraphrases count. Merely mentioning isolated values does not prove a relation. Contradictory or stale state that prevents a reliable answer is a failure.

Respect each step's declared view. Fidelity steps may compare original messages with their derived representation. Task-support steps must ignore original_messages and judge only the named saved output plus its persisted timestamp/source metadata. Do not use episode text to rescue atomic_fact_task_support or atomic facts to rescue episode_task_support.

For MemOS, `multimodal_extraction_fidelity` may compare each extraction group's source messages with its outputs. `fine_memory_store` must ignore source messages and judge only the persisted memory outputs. `state_invalidation_support` is meaningful only when the supplied artifact marks a state-change task.

For Mem0, `entity_link_task_support` must judge the persisted entity labels and memory links as an entity-boost representation. The memory text may identify what should have been linked, but must not be used to rescue a missing or incorrect entity link.

Steps:
{step_lines}

Return JSON only:
{{"steps":[{{"step_id":"...","status":"pass|fail","reason":"concise evidence-based reason","missing_or_wrong":"concise, empty when none","confidence":0.0}}]}}

Return exactly one entry for every step ID above. The official evidence is complete and must not be treated as a memory-system output.

INPUT:
{json.dumps(payload, ensure_ascii=False)}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(value[start : end + 1])


def call_judge(client: Any, model: str, system: str, payload: dict[str, Any], retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": judge_prompt(system, payload)}],
            )
            raw = response.choices[0].message.content or ""
            parsed = parse_json_response(raw)
            expected = [step_id for step_id, _ in SEMANTIC_STEPS[system]]
            entries = parsed.get("steps") or []
            by_id = {str(entry.get("step_id")): entry for entry in entries if isinstance(entry, dict)}
            if set(by_id) != set(expected):
                raise ValueError(f"judge step IDs {sorted(by_id)} != expected {sorted(expected)}")
            for entry in by_id.values():
                if entry.get("status") not in {"pass", "fail"}:
                    raise ValueError(f"invalid judge status: {entry.get('status')!r}")
            return {"steps": [by_id[step_id] for step_id in expected], "raw_response": raw}
        except Exception as exc:  # API and parse failures are preserved in artifacts.
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    assert last_error is not None
    return {"error": f"{type(last_error).__name__}: {last_error}", "steps": []}


def system_scaffold(
    system: str,
    search: dict[str, Any],
    evermemos_observation: dict[str, Any] | None = None,
    memos_observation: dict[str, Any] | None = None,
    mem0_observation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    def hidden(step_id: str, label: str, reason: str) -> dict[str, Any]:
        return {"step_id": step_id, "label": label, "status": "not_observable", "reason": reason}

    def replayable(step_id: str, label: str, reason: str) -> dict[str, Any]:
        return {
            "step_id": step_id,
            "label": label,
            "status": "replayable",
            "reason": reason,
        }

    if system == "mem0_local":
        if mem0_observation is None:
            raise ValueError("Mem0 deterministic observation is required")
        entity_boost_executed = bool(
            mem0_observation["semantic_artifact"]["entity_boost_executed"]
        )
        return [
            hidden(
                "write_existing_candidates_original",
                "Original write-time top-10 existing-memory candidates",
                "the local existing_results list and prompt were not persisted",
            ),
            replayable(
                "write_existing_candidates_replay",
                "Audit-time replay of write-time existing-memory retrieval",
                "the MEME batches, batch indices, stored vectors, and search implementation survive; replay must be labeled counterfactual because the original query vector and exact historical store snapshot were not saved",
            ),
            hidden(
                "raw_llm_extraction_response",
                "Raw additive-extraction LLM response",
                "raw JSON, linked_memory_ids, parse failures, embedding failures, and rejected candidates were not persisted",
            ),
            hidden(
                "dedup_rejected_candidates",
                "Candidates rejected by exact-hash dedup",
                "the pre-dedup extraction list and debug skip records were not persisted",
            ),
            *mem0_observation["deterministic_steps"],
            (
                hidden(
                    "entity_link_task_support",
                    "Entity-link support for entity boosting",
                    "semantic quality requires LLM judging",
                )
                if entity_boost_executed
                else {
                    "step_id": "entity_link_task_support",
                    "label": "Entity-link support for entity boosting",
                    "status": "not_applicable",
                    "reason": "query entity extraction returned no entities, so the entity-boost branch did not execute",
                    "details": {"query_entities": []},
                }
            ),
            hidden(
                "retrieval_components_original",
                "Original semantic, keyword, and entity-boost component candidates",
                "search_results retain only final combined scores; query entities, per-branch candidates, and score_details were not persisted",
            ),
            replayable(
                "retrieval_components_replay",
                "Audit-time retrieval replay with score decomposition",
                "the final dense/sparse indexes, entity collection, queries, and SDK explain path survive; replay can be compared with original final IDs but is not original telemetry",
            ),
        ]
    if system == "memos_local":
        if memos_observation is None:
            raise ValueError("MemOS deterministic observation is required")
        state_change_task = bool(memos_observation["semantic_artifact"]["state_change_task"])
        return [
            *memos_observation["deterministic_steps"],
            hidden(
                "raw_fine_extraction_response",
                "Raw fine-mode LLM response and parse failures",
                "successful parsed fields survive in Neo4j, but the original LLM text and rejected/failed outputs were lost with the rotated ingestion logs",
            ),
            (
                hidden(
                    "state_invalidation_support",
                    "State invalidation/deletion semantic support",
                    "semantic quality requires LLM judging",
                )
                if state_change_task
                else {
                    "step_id": "state_invalidation_support",
                    "label": "State invalidation/deletion semantic support",
                    "status": "not_applicable",
                    "reason": "the MEME task does not require Del/Cas/Abs state invalidation semantics",
                }
            ),
            hidden(
                "hybrid_candidate_generation",
                "Graph/vector/BM25 hybrid candidate generation",
                "the original per-branch and merged candidate lists were not persisted; live Docker replay is counterfactual unless instrumented",
            ),
            hidden(
                "internal_cosine_rerank",
                "Internal per-path cosine rerank",
                "the original pre/post lists were not persisted; rerank toggles change upstream scores and trimming",
            ),
            hidden(
                "handler_mmr_dedup",
                "SearchHandler MMR selection",
                "the original pre/post MMR lists were not persisted; the live hook point can capture them during an instrumented replay",
            ),
        ]
    metadata = search.get("retrieval_metadata") or {}
    multi = bool(metadata.get("is_multi_round"))
    if evermemos_observation is None:
        raise ValueError("EverMemOS deterministic observation is required")
    return [
        hidden(
            "raw_boundary_decision_trace",
            "Raw per-call LLM boundary decision trace",
            "final MemCell partitions are saved, but prompts, responses, should_wait, retries, and split trigger types were not persisted",
        ),
        *evermemos_observation["deterministic_steps"],
        hidden("round1_hybrid_candidates", "Round-one hybrid candidates", "candidate list was not persisted"),
        hidden("round1_rerank", "Round-one rerank", "pre/post rerank lists were not persisted"),
        {
            "step_id": "sufficiency_decision",
            "label": "LLM sufficiency decision telemetry",
            "status": "not_observable",
            "reason": "decision telemetry persisted, but the exact round-one reranked document list needed to judge the decision was not persisted" if "is_sufficient" in metadata else "decision telemetry missing",
            "telemetry": {k: metadata.get(k) for k in ("is_sufficient", "reasoning", "missing_info", "key_information_found") if k in metadata},
        },
        {
            "step_id": "query_refinement",
            "label": "LLM query refinement telemetry",
            "status": "not_observable" if multi and metadata.get("refined_queries") else "not_applicable" if not multi else "fail",
            "reason": "refined queries persisted for semantic judging" if multi and metadata.get("refined_queries") else "single-round path" if not multi else "multi-round path has no refined queries",
            "telemetry": metadata.get("refined_queries") or [],
        },
        hidden("round2_hybrid_rrf", "Round-two hybrid retrieval and RRF", "intermediate document list was not persisted") if multi else {"step_id": "round2_hybrid_rrf", "label": "Round-two hybrid retrieval and RRF", "status": "not_applicable", "reason": "single-round path"},
        hidden("final_merge_rerank", "Final merge, event dedup, and rerank", "pre/post lists were not persisted") if multi else {"step_id": "final_merge_rerank", "label": "Final merge, event dedup, and rerank", "status": "not_applicable", "reason": "single-round path"},
    ]


def ordered_steps(system: str, judged: list[dict[str, Any]], scaffold: list[dict[str, Any]], final_correct: bool) -> list[dict[str, Any]]:
    judged_by_id = {entry["step_id"]: entry for entry in judged}
    scaff_by_id = {entry["step_id"]: entry for entry in scaffold}
    order = {
        "mem0_local": [
            "official_source",
            "write_existing_candidates_original",
            "write_existing_candidates_replay",
            "raw_llm_extraction_response",
            "dedup_rejected_candidates",
            "post_dedup_history_lineage",
            "task_relevant_residual_exact_duplicates",
            "fact_store",
            "entity_link_integrity",
            "entity_link_task_support",
            "retrieval_components_original",
            "retrieval_components_replay",
            "temporal_filter_integrity",
            "native_retrieval",
            "reader_context",
        ],
        "memos_local": [
            "official_source",
            "multimodal_window_provenance",
            "raw_fine_extraction_response",
            "multimodal_extraction_fidelity",
            "graph_vector_persistence",
            "state_evolution_mechanism",
            "state_invalidation_support",
            "fine_memory_store",
            "hybrid_candidate_generation",
            "internal_cosine_rerank",
            "handler_mmr_dedup",
            "temporal_filter_integrity",
            "native_retrieval",
            "reader_context",
        ],
        "evermemos": [
            "official_source",
            "raw_boundary_decision_trace",
            "memcell_message_integrity",
            "memcell_cutoff_visibility",
            "memcell_partition_quality",
            "episode_fidelity",
            "episode_task_support",
            "atomic_fact_fidelity",
            "atomic_fact_task_support",
            "episode_index_integrity",
            "atomic_fact_index_integrity",
            "round1_hybrid_candidates",
            "round1_rerank",
            "sufficiency_decision",
            "query_refinement",
            "round2_hybrid_rrf",
            "final_merge_rerank",
            "native_retrieval",
            "reader_context",
        ],
    }[system]
    result = []
    for step_id in order:
        scaffold = scaff_by_id.get(step_id)
        # A branch that did not execute remains N/A even though the fixed judge
        # schema contains an entry for it.
        if scaffold and scaffold.get("status") == "not_applicable":
            entry = dict(scaffold)
        else:
            entry = dict(judged_by_id.get(step_id) or scaffold)
        entry.setdefault("label", dict(SEMANTIC_STEPS[system]).get(step_id, step_id))
        result.append(entry)
    result.append({"step_id": "answer", "label": "Official MEME answer evaluation", "status": "pass" if final_correct else "fail", "reason": "official per-question evaluator"})
    return result


def add_transitions(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous: dict[str, Any] | None = None
    skipped = 0
    transitions = []
    for step in steps:
        if step["status"] not in {"pass", "fail"}:
            skipped += 1
            continue
        if previous is not None:
            pair = (previous["status"], step["status"])
            kind = {("pass", "pass"): "preserved", ("pass", "fail"): "new_error", ("fail", "fail"): "propagated", ("fail", "pass"): "recovered"}[pair]
            transitions.append({"from": previous["step_id"], "to": step["step_id"], "kind": kind, "unobserved_steps_between": skipped})
        previous = step
        skipped = 0
    return transitions


def add_system_transitions(
    system: str, steps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build causal transitions without linearizing EverMemOS's two branches."""
    if system != "evermemos":
        return add_transitions(steps)
    by_id = {step["step_id"]: step for step in steps}
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path_name, step_ids in EVERMEMOS_PATHS.items():
        path_steps = [by_id[step_id] for step_id in step_ids]
        for transition in add_transitions(path_steps):
            key = (transition["from"], transition["to"], transition["kind"])
            if key not in merged:
                merged[key] = {**transition, "paths": [path_name]}
            else:
                merged[key]["paths"].append(path_name)
                merged[key]["unobserved_steps_between"] = min(
                    merged[key]["unobserved_steps_between"],
                    transition["unobserved_steps_between"],
                )
    return list(merged.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[str, Counter[str]] = defaultdict(Counter)
    transitions: Counter[str] = Counter()
    failed_histogram: Counter[int] = Counter()
    final_by_failures: dict[int, Counter[str]] = defaultdict(Counter)
    cumulative_ever_failed: Counter[str] = Counter()
    cumulative_by_path: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        failed = 0
        ever_failed = False
        for step in row["steps"]:
            by_step[step["step_id"]][step["status"]] += 1
            if step["status"] == "fail":
                failed += 1
                ever_failed = True
            if ever_failed and row.get("system") != "evermemos":
                cumulative_ever_failed[step["step_id"]] += 1
        if row.get("system") == "evermemos":
            statuses = {step["step_id"]: step["status"] for step in row["steps"]}
            for path_name, step_ids in EVERMEMOS_PATHS.items():
                path_failed = False
                for step_id in step_ids:
                    if statuses[step_id] == "fail":
                        path_failed = True
                    if path_failed:
                        cumulative_by_path[path_name][step_id] += 1
        failed_histogram[failed] += 1
        final_by_failures[failed]["questions"] += 1
        final_by_failures[failed]["correct"] += int(row["final_correct"])
        transitions.update(item["kind"] for item in row["transitions"])
    return {
        "questions": len(rows),
        "step_status_counts": {step: dict(counts) for step, counts in by_step.items()},
        "step_error_rates": {
            step: counts["fail"] / (counts["pass"] + counts["fail"])
            if counts["pass"] + counts["fail"] else None
            for step, counts in by_step.items()
        },
        "transition_counts": dict(transitions),
        "failed_observable_steps_per_question": dict(sorted(failed_histogram.items())),
        "final_accuracy_by_failed_steps": {
            str(n): {**counts, "accuracy": counts["correct"] / counts["questions"]}
            for n, counts in sorted(final_by_failures.items())
        },
        "cumulative_questions_ever_failed_by_step": dict(cumulative_ever_failed),
        "cumulative_questions_ever_failed_by_path_and_step": {
            path: dict(counts) for path, counts in cumulative_by_path.items()
        },
    }


def render_markdown(system: str, summary: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        f"# {SYSTEM_LABELS[system]} MEME step trace",
        "",
        f"Questions: {summary['questions']}",
        f"Judge: `{metadata['judge_model'] or 'disabled'}` (`{PROMPT_VERSION}`)",
        "",
        "Every observable step is counted independently. `not_observable` is not treated as success and is excluded from error-rate denominators.",
        "",
        "## Per-step outcomes",
        "",
        "| step | pass | fail | not observable | replayable | N/A | judge error | error rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step, counts in summary["step_status_counts"].items():
        rate = summary["step_error_rates"][step]
        display = "N/A" if rate is None else f"{rate:.4f}"
        lines.append(f"| {step} | {counts.get('pass', 0)} | {counts.get('fail', 0)} | {counts.get('not_observable', 0)} | {counts.get('replayable', 0)} | {counts.get('not_applicable', 0)} | {counts.get('judge_error', 0)} | {display} |")
    lines.extend(["", "## Observable transitions", ""])
    for kind in ("preserved", "new_error", "propagated", "recovered"):
        lines.append(f"- `{kind}`: {summary['transition_counts'].get(kind, 0)}")
    lines.extend(["", "## Final accuracy by number of failed observable steps", "", "| failed steps | questions | correct | accuracy |", "|---:|---:|---:|---:|"])
    for failed, counts in summary["final_accuracy_by_failed_steps"].items():
        lines.append(f"| {failed} | {counts['questions']} | {counts['correct']} | {counts['accuracy']:.4f} |")
    if system == "evermemos":
        lines.extend(
            [
                "",
                "## Cumulative failure by representation path",
                "",
                "Episode and atomic-fact transformations are parallel branches; cumulative counts and transitions never cross from one branch to the other.",
                "",
                "| path | step | questions failed at or before step |",
                "|---|---|---:|",
            ]
        )
        for path, counts in summary[
            "cumulative_questions_ever_failed_by_path_and_step"
        ].items():
            for step_id, count in counts.items():
                lines.append(f"| {path} | {step_id} | {count} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(DEFAULT_RUNS), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--end-conv", type=int, default=100)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--per-task", type=int, default=0, help="Select N after-phase questions per MEME task")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument("--lexical-store-candidates", type=int, default=20)
    parser.add_argument("--max-store-chars", type=int, default=60000)
    parser.add_argument("--judge-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--memos-api-url", default="http://localhost:8000")
    parser.add_argument("--memos-prefix", default="fine")
    parser.add_argument("--memos-qdrant-url", default="http://localhost:6333")
    parser.add_argument("--memos-qdrant-collection", default="neo4j_vec_db")
    parser.add_argument("--memory-export-path", type=Path)
    parser.add_argument("--max-match-examples", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(EVERCORE_DIR / ".env")
    system = args.system
    run_dir = (args.run_dir or DEFAULT_RUNS[system]).resolve()
    # Shared store loaders consume the argparse namespace directly.
    args.run_dir = run_dir
    if system == "evermemos" and run_dir != DEFAULT_RUNS[system].resolve():
        raise ValueError(f"EverMemOS audit is pinned to {DEFAULT_RUNS[system].resolve()}")
    episodes = load_json(args.data.resolve())
    specs = question_specs(episodes, args.start_conv, args.end_conv)
    if args.question_id:
        selected = set(args.question_id)
        specs = [spec for spec in specs if spec.question_id in selected]
    if args.per_task:
        counts: Counter[str] = Counter()
        chosen = []
        for spec in specs:
            if spec.phase == "after" and spec.task_type in MEME_TASKS and counts[spec.task_type] < args.per_task:
                chosen.append(spec)
                counts[spec.task_type] += 1
        specs = chosen
    if args.max_questions:
        specs = specs[: args.max_questions]
    if not specs:
        raise ValueError("No questions selected")

    searches = index_rows(load_json(run_dir / "search_results.json"))
    answers = index_rows(load_json(run_dir / "answer_results.json"))
    evaluations = index_rows(load_json(run_dir / "eval_results.json"))
    expected = {spec.question_id for spec in specs}
    for name, rows in (("search", searches), ("answer", answers), ("evaluation", evaluations)):
        missing = expected - set(rows)
        if missing:
            raise ValueError(f"{name} artifact missing {len(missing)} selected questions")
    if system == "memos_local":
        prefixes = {
            str(user_id).split("__", 1)[0]
            for row in searches.values()
            for user_id in (row.get("retrieval_metadata") or {}).get("user_ids") or []
            if "__" in str(user_id)
        }
        if prefixes and prefixes != {args.memos_prefix}:
            raise ValueError(f"Memos namespace mismatch: run={sorted(prefixes)}, requested={args.memos_prefix}")

    timestamp_maps = session_timestamp_maps(episodes)
    store, store_audit = LOADERS[system](args, episodes, timestamp_maps)
    selected_conversations = {spec.conversation_id for spec in specs}
    mem0_artifacts = (
        load_mem0_step_artifacts(run_dir, store, selected_conversations)
        if system == "mem0_local"
        else {}
    )
    evermemos_artifacts = (
        load_evermemos_step_artifacts(
            run_dir,
            selected_conversations,
            timestamp_maps,
        )
        if system == "evermemos"
        else {}
    )
    memos_qdrant_ids: dict[str, set[str]] = {}
    memos_qdrant_audit: dict[str, Any] | None = None
    if system == "memos_local":
        memos_qdrant_ids, memos_qdrant_audit = load_memos_qdrant_ids(
            args.memos_qdrant_url,
            args.memos_qdrant_collection,
            selected_conversations,
            args.memos_prefix,
        )
    base_rows = build_diagnostics(
        system=system,
        args=args,
        episodes=episodes,
        specs=specs,
        store=store,
        searches=searches,
        answers=answers,
        evaluations=evaluations,
    )
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / f"meme_step_trace_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=False)

    prepared = []
    for row in base_rows:
        visible = phase_safe_pool(store.get(row["conversation_id"], []), row["cutoff_session"])
        selected, selection = select_store_candidates(row, visible, lexical_limit=args.lexical_store_candidates)
        evermemos_observation = (
            build_evermemos_question_observation(
                row, evermemos_artifacts[row["conversation_id"]]
            )
            if system == "evermemos"
            else None
        )
        memos_observation = (
            build_memos_question_observation(
                row,
                store.get(row["conversation_id"], []),
                selected,
                memos_qdrant_ids[row["conversation_id"]],
                timestamp_maps.get(row["conversation_id"], []),
                max_store_chars=args.max_store_chars,
            )
            if system == "memos_local"
            else None
        )
        mem0_observation = (
            build_mem0_question_observation(
                row,
                store.get(row["conversation_id"], []),
                selected,
                mem0_artifacts,
                timestamp_maps.get(row["conversation_id"], []),
            )
            if system == "mem0_local"
            else None
        )
        payload, payload_meta = build_judge_payload(
            system,
            row,
            selected,
            searches[row["question_id"]],
            max_store_chars=args.max_store_chars,
            evermemos_observation=evermemos_observation,
            memos_observation=memos_observation,
            mem0_observation=mem0_observation,
        )
        prepared.append(
            (
                row,
                payload,
                {**selection, **payload_meta},
                evermemos_observation,
                memos_observation,
                mem0_observation,
            )
        )

    judgments: dict[str, dict[str, Any]] = {}
    if not args.no_llm:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL")
        if not api_key or not base_url:
            raise ValueError("DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL are required")
        client = OpenAI(api_key=api_key, base_url=base_url)
        with ThreadPoolExecutor(max_workers=args.judge_workers) as pool:
            futures = {
                pool.submit(call_judge, client, args.judge_model, system, payload, args.judge_retries): row["question_id"]
                for row, payload, _meta, _evermemos_observation, _memos_observation, _mem0_observation in prepared
            }
            for index, future in enumerate(as_completed(futures), start=1):
                question_id = futures[future]
                judgments[question_id] = future.result()
                print(f"Judged {index}/{len(futures)}: {question_id}")

    output_rows = []
    for (
        row,
        _payload,
        selection,
        evermemos_observation,
        memos_observation,
        mem0_observation,
    ) in prepared:
        judgment = judgments.get(row["question_id"], {})
        if args.no_llm:
            judged = []
            for step_id, label in SEMANTIC_STEPS[system]:
                if step_id in {
                    "query_refinement",
                    "memcell_partition_quality",
                    "episode_fidelity",
                    "episode_task_support",
                    "atomic_fact_fidelity",
                    "atomic_fact_task_support",
                    "multimodal_extraction_fidelity",
                    "state_invalidation_support",
                    "entity_link_task_support",
                }:
                    judged.append({"step_id": step_id, "label": label, "status": "not_observable", "reason": "semantic quality requires LLM judging"})
                    continue
                if step_id == "official_source":
                    layer = row["official_source"]
                elif step_id in {
                    "fact_store",
                    "fine_memory_store",
                }:
                    layer = row["persisted_store_snapshot"]
                elif step_id == "native_retrieval":
                    layer = row["native_retrieval"]
                else:
                    layer = row["reader_context"]
                status = "pass" if layer["coverage"]["status"] == "all" else "fail"
                judged.append({"step_id": step_id, "label": label, "status": status, "reason": "deterministic exact-signal proxy (--no-llm)"})
        elif judgment.get("error"):
            judged = [{"step_id": step_id, "label": label, "status": "judge_error", "reason": judgment["error"]} for step_id, label in SEMANTIC_STEPS[system]]
        else:
            judged = judgment["steps"]
        steps = ordered_steps(
            system,
            judged,
            system_scaffold(
                system,
                searches[row["question_id"]],
                evermemos_observation,
                memos_observation,
                mem0_observation,
            ),
            row["final_correct"],
        )
        output_rows.append({
            "system": system,
            "question_id": row["question_id"],
            "conversation_id": row["conversation_id"],
            "phase": row["phase"],
            "task_type": row["task_type"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": row["generated_answer"],
            "final_correct": row["final_correct"],
            "candidate_selection": selection,
            "retrieval_agent_metadata": selection.get("agent_metadata") or {},
            "evermemos_memcell_audit": (
                evermemos_observation["audit"]
                if evermemos_observation is not None
                else None
            ),
            "memos_docker_audit": (
                memos_observation["audit"] if memos_observation is not None else None
            ),
            "mem0_artifact_audit": (
                mem0_observation["audit"] if mem0_observation is not None else None
            ),
            "steps": steps,
            "transitions": add_system_transitions(system, steps),
            "judge_raw_response": judgment.get("raw_response"),
            "judge_error": judgment.get("error"),
        })

    summary = summarize(output_rows)
    metadata = {
        "system": system,
        "run_dir": str(run_dir),
        "data": str(args.data.resolve()),
        "output_dir": str(output_dir),
        "prompt_version": PROMPT_VERSION,
        "judge_model": None if args.no_llm else args.judge_model,
        "judge_temperature": 0,
        "store_audit": store_audit,
        "mem0_artifact_audit": (
            mem0_artifacts.get("audit") if system == "mem0_local" else None
        ),
        "memos_qdrant_audit": memos_qdrant_audit,
        "limitations": (
            "Unpersisted internal lists are not_observable; replayable marks a "
            "counterfactual audit path, not original telemetry. Mem0 history ADD "
            "records, residual hashes, final entity links, and temporal outputs are "
            "audited directly; raw extraction and original retrieval components remain "
            "unavailable. MemOS live Neo4j source "
            "windows and Qdrant IDs are audited directly; raw ingestion LLM text and "
            "original pre/post rerank/MMR lists are unavailable without instrumented "
            "replay. EverMemOS final MemCell "
            "partitions and both indexes are audited directly, but raw per-call boundary "
            "prompts/responses and retrieval intermediate lists remain unavailable. "
            "Audit-time store candidate selection is provenance-first with lexical fallback; "
            "clipping is recorded per question."
        ),
    }
    dump_json(output_dir / "question_step_traces.json", output_rows)
    with (output_dir / "question_step_traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dump_json(output_dir / "summary.json", {**metadata, **summary})
    (output_dir / "summary.md").write_text(render_markdown(system, summary, metadata), encoding="utf-8")
    dump_json(output_dir / "question_manifest.json", [{key: row[key] for key in ("question_id", "conversation_id", "phase", "task_type")} for row in output_rows])
    print(f"Step trace written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
