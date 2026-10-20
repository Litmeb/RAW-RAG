import asyncio
import json
import sqlite3
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from evaluation.src.adapters.mem0_adapter import (
    Mem0Adapter,
    _AsyncLocalMemoryClient,
)
from evaluation.src.adapters.mem0_add_audit import (
    Mem0AddAuditWriteError,
    append_jsonl_fsynced,
    run_audited_add,
)
from evaluation.tools.summarize_mem0_add_audit import summarize_audit


class FakeHistory:
    def __init__(self):
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.execute(
            """
            CREATE TABLE history (
                id TEXT,
                memory_id TEXT,
                old_memory TEXT,
                new_memory TEXT,
                event TEXT,
                created_at TEXT,
                updated_at TEXT,
                is_deleted INTEGER,
                actor_id TEXT,
                role TEXT
            )
            """
        )

    def add(self, event: str, *, old_memory=None, new_memory=None):
        self.connection.execute(
            """
            INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"history-{event}",
                f"memory-{event}",
                old_memory,
                new_memory,
                event,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:01+00:00",
                int(event == "DELETE"),
                None,
                None,
            ),
        )
        self.connection.commit()


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate_response(self, *args, **kwargs):
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeVectorStore:
    def search(self, query, vectors, limit, filters):
        return [
            SimpleNamespace(
                id="old-memory-1",
                score=0.91,
                payload={"data": "Old preference", "api_key": "must-hide"},
            )
        ]


class FakeMemory:
    def __init__(self, responses):
        self.llm = FakeLLM(responses)
        self.vector_store = FakeVectorStore()
        self.db = FakeHistory()

    def add(self, **kwargs):
        extraction = self.llm.generate_response(
            messages=[
                {"role": "system", "content": "extract facts"},
                {"role": "user", "content": kwargs["messages"][0]["content"]},
            ],
            response_format={"type": "json_object"},
        )
        facts = json.loads(extraction)["facts"]
        for fact in facts:
            self.vector_store.search(
                query=fact,
                vectors=[0.1, 0.2],
                limit=5,
                filters={"user_id": kwargs["user_id"]},
            )
        decision = self.llm.generate_response(
            messages=[{"role": "user", "content": "choose actions"}],
            response_format={"type": "json_object"},
        )
        actions = json.loads(decision)["memory"]
        returned = []
        for action in actions:
            if action["event"] == "ADD":
                self.db.add("ADD", new_memory=action["text"])
                returned.append(
                    {
                        "id": "memory-ADD",
                        "memory": action["text"],
                        "event": "ADD",
                    }
                )
        return {"results": returned}


def _success_memory():
    return FakeMemory(
        [
            json.dumps({"facts": ["Likes green tea"]}),
            json.dumps(
                {
                    "memory": [
                        {"id": "new", "text": "Likes green tea", "event": "ADD"},
                        {"id": "0", "text": "Old preference", "event": "NONE"},
                    ]
                }
            ),
        ]
    )


def test_run_audited_add_captures_full_decision_and_persistence():
    memory = _success_memory()
    original_generate = memory.llm.generate_response
    original_search = memory.vector_store.search
    record = {}

    result = run_audited_add(
        memory,
        add_kwargs={
            "messages": [{"role": "user", "content": "I like green tea"}],
            "user_id": "user-1",
            "run_id": "conversation-1",
            "metadata": {"batch_index": 2},
        },
        context={"conversation_id": "conversation-1", "batch_index": 2},
        record=record,
    )

    assert result["results"][0]["event"] == "ADD"
    assert record["status"] == "success"
    assert [call["stage"] for call in record["llm_calls"]] == [
        "fact_extraction",
        "memory_action_decision",
    ]
    assert record["llm_calls"][0]["extracted_facts"] == ["Likes green tea"]
    assert [
        action["event"]
        for action in record["llm_calls"][1]["proposed_actions"]
    ] == ["ADD", "NONE"]
    assert record["normalized_facts_searched"] == ["Likes green tea"]
    assert record["vector_searches"][0]["results"][0]["payload"]["api_key"] == (
        "<redacted>"
    )
    assert record["vector_searches"][0]["request"]["vector"]["dimension"] == 2
    assert "vectors" not in record["vector_searches"][0]["request"]["kwargs"]
    assert [row["event"] for row in record["persisted_history_rows"]] == ["ADD"]
    assert memory.llm.generate_response == original_generate
    assert memory.vector_store.search == original_search


def test_add_only_policy_rewrites_all_mutating_v1_actions():
    response = """```json
    {"memory": [
      {"id": "0", "text": "New preference", "old_memory": "Old preference", "event": "UPDATE"},
      {"id": "1", "text": "Obsolete fact", "event": "DELETE"},
      {"id": "2", "text": "Already known", "event": "NONE"},
      {"text": "Fresh fact", "event": "ADD"}
    ]}
    ```"""

    rewritten = Mem0Adapter._rewrite_mem0_actions_add_only(response)
    actions = json.loads(rewritten)["memory"]

    assert [action["event"] for action in actions] == ["ADD", "NONE", "NONE", "ADD"]
    assert all("id" not in action for action in actions)
    assert all("old_memory" not in action for action in actions)


def test_async_client_persists_error_record_before_reraising():
    memory = FakeMemory([RuntimeError("model unavailable")])
    client = _AsyncLocalMemoryClient(memory)
    records = []

    async def capture(record):
        records.append(record)

    async def run():
        with pytest.raises(RuntimeError, match="model unavailable"):
            await client.add(
                messages=[{"role": "user", "content": "hello"}],
                user_id="user-1",
                _audit_context={"conversation_id": "conversation-1"},
                _audit_callback=capture,
            )

    asyncio.run(run())

    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["error"]["type"] == "RuntimeError"
    assert records[0]["llm_calls"][0]["status"] == "error"


def test_audit_write_error_is_not_retried():
    adapter = object.__new__(Mem0Adapter)
    adapter.max_retries = 3
    adapter.console = Console(file=StringIO())
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise Mem0AddAuditWriteError("disk full")

    async def run():
        with pytest.raises(Mem0AddAuditWriteError, match="disk full"):
            await adapter._call_with_retry("audited add", operation)

    asyncio.run(run())
    assert attempts == 1


def test_content_filter_add_is_skipped_without_retry():
    adapter = object.__new__(Mem0Adapter)
    adapter.max_retries = 3
    adapter.console = Console(file=StringIO())
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Error code: 400 - {'code': 'content_filter'}")

    result = asyncio.run(adapter._call_with_retry("add batch 1/1", operation))

    assert result is None
    assert attempts == 1


def test_content_filter_search_is_not_silently_skipped():
    adapter = object.__new__(Mem0Adapter)
    adapter.max_retries = 2
    adapter.console = Console(file=StringIO())
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Error code: 400 - {'code': 'content_filter'}")

    async def run():
        with pytest.raises(RuntimeError, match="content_filter"):
            await adapter._call_with_retry("search", operation)

    asyncio.run(run())
    assert attempts == 2


def test_jsonl_writer_redacts_credentials_and_summary_counts_none(tmp_path: Path):
    audit_path = tmp_path / "mem0_add_audit.jsonl"
    first = {
        "operation_id": "op-1",
        "status": "success",
        "context": {
            "conversation_id": "conversation-1",
            "speaker": "speaker_a",
            "batch_index": 0,
        },
        "add_input": {"metadata": {"authorization": "Bearer secret"}},
        "llm_calls": [
            {
                "stage": "fact_extraction",
                "extracted_facts": ["fact"],
            },
            {
                "stage": "memory_action_decision",
                "proposed_actions": [
                    {"event": "ADD", "text": "fact"},
                    {"event": "NONE", "text": "old"},
                ],
            },
        ],
        "normalized_facts_searched": ["fact"],
        "vector_searches": [{"results": [{"id": "old"}]}],
        "add_result": {"results": [{"event": "ADD"}]},
        "persisted_history_rows": [{"event": "ADD"}],
    }
    second = {
        **first,
        "operation_id": "op-2",
        "status": "error",
        "add_result": None,
        "persisted_history_rows": [],
    }
    append_jsonl_fsynced(audit_path, first, fsync=True)
    append_jsonl_fsynced(audit_path, second, fsync=True)

    rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["add_input"]["metadata"]["authorization"] == "<redacted>"

    summary = summarize_audit(audit_path)
    assert summary["records"]["status"] == {"error": 1, "success": 1}
    assert summary["coverage"]["unique_batches"] == 1
    assert summary["coverage"]["retried_batches"] == 1
    assert summary["events"]["decision_all_attempts"] == {"ADD": 2, "NONE": 2}
    assert summary["events"]["persisted_to_history"] == {"ADD": 1}
    assert (
        summary["consistency"][
            "operations_with_proposed_vs_persisted_mismatch"
        ]
        == 1
    )
