from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.scripts import build_meme_complete_store_oracle_runs as builder


TIMESTAMPS = {
    "meme_0": [
        ("2023-03-01T09:00:00", 0),
        ("2023-03-02T09:00:00", 1),
    ]
}


def test_native_namespace_validation_rejects_old_memos_store(tmp_path: Path) -> None:
    run = tmp_path / "meme-memos_local-fine"
    run.mkdir()
    (run / "search_results.json").write_text(
        json.dumps([{"results": [{"user_id": "fine__meme_0_User"}]}]),
        encoding="utf-8",
    )
    audit = builder.validate_native_namespace(run, "fine", separator="__")
    assert audit["observed_user_ids"] == 1

    with pytest.raises(ValueError, match="Native namespace validation failed"):
        builder.validate_native_namespace(run, "normal", separator="__")


def test_memos_fine_export_uses_exact_cube_and_maps_provenance(monkeypatch) -> None:
    calls = []

    def fake_http_json(url, payload, timeout=60.0, retries=3):
        del url, timeout, retries
        calls.append(payload)
        return {
            "data": {
                "text_mem": [
                    {
                        "cube_id": payload["mem_cube_id"],
                        "memories": [
                            {
                                "id": "m1",
                                "memory": "reader-visible fact",
                                "metadata": {
                                    "sources": [
                                        {"chat_time": "2023-03-02T09:03:00"}
                                    ]
                                },
                            }
                        ],
                    }
                ]
            }
        }

    monkeypatch.setattr(builder, "http_json", fake_http_json)
    store, audit, exported = builder.load_memos_store(
        "http://memos", "fine", ["meme_0"], TIMESTAMPS
    )

    assert calls[0]["mem_cube_id"] == "fine__meme_0_User"
    assert store["meme_0"][0].metadata["source_session_indices"] == [1]
    assert audit["cube_prefix"] == "fine"
    assert len(exported) == 1


def write_memu_part(run: Path, part: int, timestamp: str) -> None:
    resources = run / "memu_resources"
    resources.mkdir(exist_ok=True)
    (resources / f"meme_0-part-{part:04d}.json").write_text(
        json.dumps([{"created_at": timestamp, "content": f"source {part}"}]),
        encoding="utf-8",
    )


def test_memu_parser_enumeration_phase_cutoff_and_order(tmp_path: Path) -> None:
    run = tmp_path / "meme-memu_local-normal"
    run.mkdir()
    write_memu_part(run, 0, "2023-03-01T09:05:00")
    write_memu_part(run, 1, "2023-03-02T09:05:00")
    resources = [
        {
            "id": "r1",
            "user_id": "normal:meme_0",
            "url": "meme_0-part-0001_#segment_7.json",
        },
        {
            "id": "r0",
            "user_id": "normal:meme_0",
            "url": str(run / "memu_resources" / "meme_0-part-0000.json"),
        },
    ]
    items = [
        {
            "id": "late",
            "user_id": "normal:meme_0",
            "resource_id": "r1",
            "memory_type": "event",
            "summary": "late state",
            "created_at": "2026-01-02T00:00:00Z",
        },
        {
            "id": "early-b",
            "user_id": "normal:meme_0",
            "resource_id": "r0",
            "memory_type": "profile",
            "summary": "early second",
            "created_at": "2026-01-01T00:00:02Z",
        },
        {
            "id": "early-a",
            "user_id": "normal:meme_0",
            "resource_id": "r0",
            "memory_type": "profile",
            "summary": "early first",
            "created_at": "2026-01-01T00:00:01Z",
        },
    ]

    store, audit, exported = builder.memu_rows_to_store(
        resources,
        items,
        run,
        TIMESTAMPS,
        categories=[
            {
                "id": "c0",
                "user_id": "normal:meme_0",
                "name": "profile",
                "summary": "combined category state",
                "created_at": "2026-01-03T00:00:00Z",
            },
            {
                "id": "c-empty",
                "user_id": "normal:meme_0",
                "name": "empty",
                "description": "schema-only category",
                "summary": None,
                "created_at": "2026-01-04T00:00:00Z",
            },
        ],
        category_items=[
            {"category_id": "c0", "item_id": "early-a"},
            {"category_id": "c0", "item_id": "late"},
        ],
    )

    assert builder.parse_memu_resource_name(resources[0]["url"]) == ("meme_0", 1)
    assert [item.content for item in store["meme_0"]] == [
        "early first",
        "early second",
        "late state",
        "combined category state",
    ]
    assert [item.content for item in builder.phase_safe_pool(store["meme_0"], 0)] == [
        "early first",
        "early second",
    ]
    assert audit["total_resources"] == 2
    assert audit["segmented_resources"] == 1
    assert audit["memory_items"] == 3
    assert audit["memory_categories"] == 2
    assert audit["reader_visible_categories"] == 1
    assert audit["empty_category_summaries"] == 1
    assert audit["total_objects"] == 5
    assert len(exported) == 5


def test_memu_parser_rejects_cross_namespace_resource(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    write_memu_part(run, 0, "2023-03-01T09:05:00")
    resources = [
        {
            "id": "r0",
            "user_id": "other:meme_0",
            "url": "meme_0-part-0000.json",
        }
    ]
    with pytest.raises(ValueError, match="validation failed"):
        builder.memu_rows_to_store(resources, [], run, TIMESTAMPS)
