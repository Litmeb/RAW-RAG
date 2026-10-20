from __future__ import annotations

from evaluation.scripts import build_meme_complete_store_oracle_runs as builder


def test_memmachine_project_id_matches_native_namespace() -> None:
    assert (
        builder.memmachine_project_id("eval", "normal", "meme_17")
        == "eval__normal__meme_17"
    )


def test_memmachine_session_index_prefers_native_session_metadata() -> None:
    boundaries = [
        ("2023-03-01T09:00:00", 0),
        ("2023-03-02T09:00:00", 1),
    ]
    assert (
        builder.memmachine_session_index(
            {"session": "session_1", "source_timestamp": "wrong"},
            "",
            boundaries,
        )
        == 1
    )
    assert (
        builder.memmachine_session_index(
            {"source_timestamp": "2023-03-01T09:03:00+00:00"},
            "",
            boundaries,
        )
        == 0
    )


def test_list_memmachine_memories_paginates(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_http_json(url, payload, timeout=60.0, retries=3):
        del url, timeout, retries
        calls.append(payload)
        page = int(payload["page_num"])
        rows = [{"uid": str(page * 2 + offset)} for offset in range(2)]
        if page == 1:
            rows = rows[:1]
        return {"status": 0, "content": {"episodic_memory": rows}}

    monkeypatch.setattr(builder, "http_json", fake_http_json)
    rows = builder.list_memmachine_memories(
        "http://localhost:8080/api/v2",
        "ragforlocomo",
        "eval__normal__meme_0",
        "episodic",
        page_size=2,
    )

    assert [row["uid"] for row in rows] == ["0", "1", "2"]
    assert [call["page_num"] for call in calls] == [0, 1]

