from __future__ import annotations

import json
from pathlib import Path

import tiktoken

from evaluation.scripts.build_meme_complete_store_oracle_runs import (
    load_rag_store,
    pack_rag_chunk_context,
)
from evaluation.scripts.build_meme_continuity_oracle_runs import MemoryObject


def write_index(path: Path) -> None:
    value = {
        "metadata": {"chunk_size": 256, "chunk_overlap": 32},
        "conversations": {
            "meme_0": {
                "conversation_id": "meme_0",
                "chunks": [
                    {
                        "chunk_id": "meme_0_chunk_0",
                        "chunk_index": 0,
                        "content": "old state",
                        "start_turn": 0,
                        "end_turn": 1,
                        "start_token": 0,
                        "end_token": 256,
                    },
                    {
                        "chunk_id": "meme_0_chunk_1",
                        "chunk_index": 1,
                        "content": "old state and new state",
                        "start_turn": 0,
                        "end_turn": 2,
                        "start_token": 224,
                        "end_token": 480,
                    },
                    {
                        "chunk_id": "meme_0_chunk_2",
                        "chunk_index": 2,
                        "content": "new state",
                        "start_turn": 1,
                        "end_turn": 2,
                        "start_token": 448,
                        "end_token": 704,
                    },
                ],
                "turn_context": {
                    "turn_spans": [[10, 200], [200, 500]],
                    "turn_timestamps": [
                        "2023-03-01T09:00:00",
                        "2023-03-02T09:00:00",
                    ],
                    "turn_session_dates": ["2023-03-01", "2023-03-02"],
                },
                "embeddings": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }
        },
    }
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_load_rag_store_preserves_chunks_and_maps_sessions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_index(run_dir / "rag_index.json")

    store, audit = load_rag_store(
        run_dir,
        {
            "meme_0": [
                ("2023-03-01T09:00:00", 0),
                ("2023-03-02T09:00:00", 1),
            ]
        },
        ["meme_0"],
    )

    assert [item.metadata["chunk_index"] for item in store["meme_0"]] == [0, 1, 2]
    assert [item.metadata["source_session_indices"] for item in store["meme_0"]] == [
        [0],
        [0, 1],
        [1],
    ]
    assert audit["chunk_size"] == 256
    assert audit["chunk_overlap"] == 32
    assert audit["cross_session_chunks"] == 1


def test_pack_rag_context_keeps_whole_chunks_and_restores_source_order() -> None:
    encoding = tiktoken.get_encoding("o200k_base")
    early = MemoryObject(
        content="early supporting value",
        score=0.0,
        metadata={"chunk_index": 1, "start_token": 224, "end_token": 480},
        first_seen=1,
    )
    late = MemoryObject(
        content="late gold answer",
        score=0.0,
        metadata={"chunk_index": 2, "start_token": 448, "end_token": 704},
        first_seen=2,
    )
    ranked = [(late, 10.0, ["gold answer"]), (early, 5.0, ["supporting value"])]

    selected, context, skipped = pack_rag_chunk_context(ranked, encoding, 200)

    assert [row["metadata"]["native_metadata"]["chunk_index"] for row in selected] == [1, 2]
    assert context.index("early supporting value") < context.index("late gold answer")
    assert skipped == 0

    selected, context, skipped = pack_rag_chunk_context(ranked, encoding, 1)
    assert selected == []
    assert context == ""
    assert skipped == 2
