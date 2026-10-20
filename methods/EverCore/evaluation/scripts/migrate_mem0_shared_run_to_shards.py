"""Migrate a partial Mem0 QdrantLocal run into per-conversation shards."""

from __future__ import annotations

import argparse
import json
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient, models

from evaluation.src.adapters.mem0_adapter import Mem0Adapter


CHECKPOINT_NAME = "mem0_local_add_resume.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-run-dir", type=Path, required=True)
    parser.add_argument("--dst-run-dir", type=Path, required=True)
    parser.add_argument("--dst-run-name", required=True)
    parser.add_argument("--scroll-batch-size", type=int, default=128)
    return parser.parse_args()


def _new_user_id(old_user_id: str, run_name: str) -> str:
    prefix = Mem0Adapter._sanitize_user_id_component(run_name)
    suffix = old_user_id.split("__", 1)[-1]
    return f"{prefix}__{suffix}" if prefix else suffix


def _create_collections(source: QdrantClient, destination: QdrantClient) -> List[str]:
    collection_names = [item.name for item in source.get_collections().collections]
    for name in collection_names:
        params = source.get_collection(name).config.params
        destination.create_collection(
            collection_name=name,
            vectors_config=params.vectors,
            sparse_vectors_config=params.sparse_vectors,
        )
    return collection_names


def _copy_collection_for_user(
    source: QdrantClient,
    destination: QdrantClient,
    collection_name: str,
    old_user_id: str,
    new_user_id: str,
    batch_size: int,
) -> List[str]:
    offset = None
    copied_ids: List[str] = []
    while True:
        points, offset = source.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=old_user_id),
                    )
                ]
            ),
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        migrated_points = []
        for point in points:
            payload = dict(point.payload or {})
            payload["user_id"] = new_user_id
            migrated_points.append(
                models.PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            )
            copied_ids.append(str(point.id))
        if migrated_points:
            destination.upsert(collection_name, migrated_points, wait=True)
        if offset is None:
            break
    return copied_ids


def _read_sqlite_source(source_db: Path) -> Dict[str, Any]:
    connection = sqlite3.connect(source_db)
    try:
        schemas = {
            name: sql
            for name, sql in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name IN ('history', 'messages')"
            )
        }
        history_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(history)")
        ]
        message_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(messages)")
        ]
        return {
            "schemas": schemas,
            "history_columns": history_columns,
            "message_columns": message_columns,
            "history": connection.execute("SELECT * FROM history").fetchall(),
            "messages": connection.execute("SELECT * FROM messages").fetchall(),
        }
    finally:
        connection.close()


def _write_sqlite_shard(
    destination_db: Path,
    sqlite_source: Dict[str, Any],
    current_memory_ids: set[str],
    old_user_id: str,
    new_user_id: str,
) -> Dict[str, int]:
    destination_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination_db)
    try:
        for table_name in ("history", "messages"):
            connection.execute(sqlite_source["schemas"][table_name])

        history_columns = sqlite_source["history_columns"]
        memory_id_index = history_columns.index("memory_id")
        history_rows = [
            row
            for row in sqlite_source["history"]
            if str(row[memory_id_index]) in current_memory_ids
        ]
        history_placeholders = ",".join("?" for _ in history_columns)
        connection.executemany(
            f"INSERT INTO history VALUES ({history_placeholders})", history_rows
        )

        message_columns = sqlite_source["message_columns"]
        session_scope_index = message_columns.index("session_scope")
        message_rows = []
        for row in sqlite_source["messages"]:
            session_scope = str(row[session_scope_index])
            if f"user_id={old_user_id}" not in session_scope:
                continue
            migrated = list(row)
            migrated[session_scope_index] = session_scope.replace(
                f"user_id={old_user_id}", f"user_id={new_user_id}"
            )
            message_rows.append(tuple(migrated))
        message_placeholders = ",".join("?" for _ in message_columns)
        connection.executemany(
            f"INSERT INTO messages VALUES ({message_placeholders})", message_rows
        )
        connection.commit()
        return {"history": len(history_rows), "messages": len(message_rows)}
    finally:
        connection.close()


def migrate(args: argparse.Namespace) -> Dict[str, Any]:
    source_dir = args.src_run_dir.resolve()
    destination_dir = args.dst_run_dir.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source run directory does not exist: {source_dir}")
    if destination_dir.exists():
        raise FileExistsError(
            f"Destination already exists; choose a unique run directory: {destination_dir}"
        )
    if args.scroll_batch_size < 1:
        raise ValueError("--scroll-batch-size must be positive")

    checkpoint_path = source_dir / CHECKPOINT_NAME
    source_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if source_checkpoint.get("storage_layout", "shared") != "shared":
        raise ValueError("Source checkpoint is not a shared-storage Mem0 run")

    source_qdrant_path = source_dir / "mem0_qdrant"
    source_history_path = source_dir / "mem0_history.db"
    if not source_qdrant_path.is_dir() or not source_history_path.is_file():
        raise FileNotFoundError("Source run is missing mem0_qdrant or mem0_history.db")

    temporary_dir = destination_dir.with_name(f"{destination_dir.name}.migrating")
    if temporary_dir.exists():
        raise FileExistsError(
            f"Temporary migration directory already exists: {temporary_dir}"
        )
    temporary_dir.mkdir(parents=True)

    sqlite_source = _read_sqlite_source(source_history_path)
    source_client = QdrantClient(path=str(source_qdrant_path))
    migrated_checkpoint = deepcopy(source_checkpoint)
    migrated_checkpoint.update(
        {
            "version": 2,
            "run_name": args.dst_run_name,
            "storage_layout": "conversation_shards",
            "migrated_from": str(source_dir),
            "migrated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    report: Dict[str, Any] = {
        "source": str(source_dir),
        "destination": str(destination_dir),
        "run_name": args.dst_run_name,
        "conversations": {},
    }

    try:
        source_collections = [
            item.name for item in source_client.get_collections().collections
        ]
        if "mem0_local" not in source_collections:
            raise ValueError("Source Qdrant does not contain mem0_local")

        for conversation_id, conversation_state in migrated_checkpoint[
            "conversations"
        ].items():
            speakers = conversation_state.get("speakers", {})
            if set(speakers) != {"speaker_a"}:
                raise ValueError(
                    "Migration currently expects one MemoryAgentBench speaker per "
                    f"conversation; got {sorted(speakers)} for {conversation_id}"
                )
            speaker_state = speakers["speaker_a"]
            completed = sorted(map(int, speaker_state.get("completed_batches", [])))
            if completed != list(range(len(completed))):
                raise ValueError(
                    f"Non-contiguous checkpoint for {conversation_id}: {completed}"
                )

            old_user_id = str(speaker_state["user_id"])
            new_user_id = _new_user_id(old_user_id, args.dst_run_name)
            speaker_state["user_id"] = new_user_id
            shard_key = Mem0Adapter._conversation_shard_key(conversation_id)
            shard_dir = temporary_dir / "mem0_shards" / shard_key
            shard_client = QdrantClient(path=str(shard_dir / "qdrant"))
            try:
                _create_collections(source_client, shard_client)
                collection_counts = {}
                current_memory_ids: set[str] = set()
                for collection_name in source_collections:
                    copied_ids = _copy_collection_for_user(
                        source_client,
                        shard_client,
                        collection_name,
                        old_user_id,
                        new_user_id,
                        args.scroll_batch_size,
                    )
                    collection_counts[collection_name] = len(copied_ids)
                    if collection_name == "mem0_local":
                        current_memory_ids.update(copied_ids)
            finally:
                shard_client.close()

            sqlite_counts = _write_sqlite_shard(
                shard_dir / "history.db",
                sqlite_source,
                current_memory_ids,
                old_user_id,
                new_user_id,
            )
            report["conversations"][conversation_id] = {
                "old_user_id": old_user_id,
                "new_user_id": new_user_id,
                "completed_batches": len(completed),
                "total_batches": speaker_state.get("total_batches"),
                "collections": collection_counts,
                "sqlite": sqlite_counts,
            }
    finally:
        source_client.close()

    (temporary_dir / CHECKPOINT_NAME).write_text(
        json.dumps(migrated_checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (temporary_dir / "mem0_shard_migration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_dir.replace(destination_dir)
    return report


def main() -> None:
    report = migrate(_parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
