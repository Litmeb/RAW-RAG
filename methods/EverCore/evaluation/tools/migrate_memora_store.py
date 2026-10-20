"""Migrate a legacy shared Memora Chroma store into conversation shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from evaluation.src.adapters.memora_adapter import MemoraAdapter


class _LegacyEmbeddingFunction:
    """Persist Chroma's legacy/custom EF marker without making API calls."""

    @staticmethod
    def name() -> str:
        return NotImplemented

    def get_config(self) -> dict[str, Any]:
        return NotImplemented

    def __call__(self, input: list[str]) -> list[list[float]]:
        raise RuntimeError(
            "Migration always supplies existing embeddings; embedding must not run"
        )


def _collection_names(client: Any) -> set[str]:
    names = set()
    for item in client.list_collections():
        names.add(item if isinstance(item, str) else item.name)
    return names


def _batches(values: list[Any], size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, len(values), size):
        yield start, min(start + size, len(values))


def migrate_memora_run(
    source_run: Path,
    target_run: Path,
    *,
    persist_dir: str = "memora_store",
    resume_file: str = "memora_add_resume.json",
    collection_base: str = "memora",
    batch_size: int = 500,
) -> dict[str, Any]:
    """Copy checkpointed conversation collections without recomputing embeddings."""
    import chromadb

    source_run = source_run.resolve()
    target_run = target_run.resolve()
    source_store = source_run / persist_dir
    source_resume = source_run / resume_file
    target_store = target_run / persist_dir
    target_resume = target_run / resume_file

    if source_run == target_run:
        raise ValueError("Source and target run directories must differ")
    if not source_store.exists() or not source_resume.exists():
        raise FileNotFoundError(
            f"Legacy Memora store/checkpoint not found under {source_run}"
        )
    if target_resume.exists():
        raise FileExistsError(
            f"Target checkpoint already exists: {target_resume}. "
            "Choose a fresh unique run directory."
        )

    state = json.loads(source_resume.read_text(encoding="utf-8"))
    conversations = state.get("conversations", {})
    if not isinstance(conversations, dict) or not conversations:
        raise ValueError(f"No conversations found in {source_resume}")

    target_run.mkdir(parents=True, exist_ok=True)
    target_store.mkdir(parents=True, exist_ok=True)
    source_client = chromadb.PersistentClient(path=str(source_store))
    available = _collection_names(source_client)
    migrated = []

    for conversation_id in conversations:
        user_id = MemoraAdapter._user_id(conversation_id)
        collection_name = f"{collection_base}_{user_id}"
        if collection_name not in available:
            raise RuntimeError(
                f"Checkpoint references {conversation_id}, but legacy collection "
                f"{collection_name!r} is missing"
            )

        source_collection = source_client.get_collection(collection_name)
        source_count = source_collection.count()
        shard_path = target_store / user_id
        shard_path.mkdir(parents=True, exist_ok=True)
        target_client = chromadb.PersistentClient(path=str(shard_path))
        target_collection = target_client.get_or_create_collection(
            collection_name,
            metadata=source_collection.metadata or {"hnsw:space": "cosine"},
            # Preserve the source's custom/legacy embedding-function marker.
            # A bare collection persists Chroma's local default EF and then
            # conflicts when Memora reopens it with its OpenAI-compatible EF.
            embedding_function=_LegacyEmbeddingFunction(),
        )

        payload = source_collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        ids = list(payload.get("ids") or [])
        documents = list(payload.get("documents") or [])
        metadatas = list(payload.get("metadatas") or [])
        embeddings = payload.get("embeddings")
        embeddings = list(embeddings) if embeddings is not None else []
        if not (
            len(ids) == len(documents) == len(metadatas) == len(embeddings)
            == source_count
        ):
            raise RuntimeError(
                f"Incomplete legacy read for {conversation_id}: count={source_count}, "
                f"ids={len(ids)}, documents={len(documents)}, "
                f"metadatas={len(metadatas)}, embeddings={len(embeddings)}"
            )

        if ids:
            for start, end in _batches(ids, max(1, batch_size)):
                target_collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )
        target_count = target_collection.count()
        if target_count != source_count:
            raise RuntimeError(
                f"Migration count mismatch for {conversation_id}: "
                f"source={source_count}, target={target_count}"
            )
        migrated.append(
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "collection": collection_name,
                "count": source_count,
            }
        )

    shutil.copy2(source_resume, target_resume)
    manifest = {
        "version": 1,
        "layout": "conversation_shards_v1",
        "source_run": str(source_run),
        "target_run": str(target_run),
        "resume_file": resume_file,
        "conversations": migrated,
        "total_records": sum(item["count"] for item in migrated),
    }
    manifest_path = target_run / "memora_migration_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument("--persist-dir", default="memora_store")
    parser.add_argument("--resume-file", default="memora_add_resume.json")
    parser.add_argument("--collection-base", default="memora")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    manifest = migrate_memora_run(
        args.source_run,
        args.target_run,
        persist_dir=args.persist_dir,
        resume_file=args.resume_file,
        collection_base=args.collection_base,
        batch_size=args.batch_size,
    )
    print(
        f"Migrated {len(manifest['conversations'])} conversations / "
        f"{manifest['total_records']} records to {manifest['target_run']}"
    )


if __name__ == "__main__":
    main()
