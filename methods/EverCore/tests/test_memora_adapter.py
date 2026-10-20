import asyncio
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from evaluation.src.adapters.memora_adapter import (
    MemoraAdapter,
    MemoraProcessRecycleRequested,
)
from evaluation.src.adapters.registry import list_adapters
from evaluation.src.core.data_models import Conversation, Message
from evaluation.src.utils.config import load_yaml


class FakeEntry:
    def __init__(self, value, timestamp, score=0.8):
        self.value = value
        self.timestamp = timestamp
        self.score = score
        self.index = value
        self.memory_type = "factual"
        self.episodic_memory_ids = []

    def get_memory_value(self, return_history=False):
        return self.value


class FakeClient:
    instances = {}
    active_store_operations = 0
    max_active_store_operations = 0
    store_operation_lock = threading.Lock()

    def __init__(self, cfg, user_id):
        self.cfg = cfg
        self.user_id = user_id
        self.clear_calls = 0
        self.add_calls = []
        self.query_calls = []
        self.memories = []
        self.instances[user_id] = self

    def clear(self):
        self._store_operation(lambda: setattr(self, "clear_calls", self.clear_calls + 1))

    def add(self, messages, type, metadata):
        self._store_operation(lambda: self.add_calls.append((messages, type, metadata)))
        return [SimpleNamespace(index="stored")]

    @classmethod
    def _store_operation(cls, operation):
        with cls.store_operation_lock:
            cls.active_store_operations += 1
            cls.max_active_store_operations = max(
                cls.max_active_store_operations, cls.active_store_operations
            )
        try:
            time.sleep(0.01)
            operation()
        finally:
            with cls.store_operation_lock:
                cls.active_store_operations -= 1

    def advance_query(self, query, top_k, query_type):
        self.query_calls.append((query, top_k, query_type))
        return self.memories

    def get(self, key):
        return None


class FakeSegmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = 2

    def segment_conversation(self, messages):
        return [
            {
                "indices": list(range(start, min(start + self.batch_size, len(messages)))),
                "topic": f"batch-{start // self.batch_size}",
            }
            for start in range(0, len(messages), self.batch_size)
        ]


def make_config(memora_root):
    return {
        "name": "memora",
        "adapter": "memora",
        "num_workers": 2,
        "memora": {
            "root": str(memora_root),
            "batch_size": 2,
            "segment_workers": 1,
            "enable_segmentation": False,
            "store_chunk_chars": 10000,
            "min_store_chunk_chars": 1000,
            "llm": {
                "model": "gpt-4o-mini",
                "api_key": "test",
                "base_url": "https://llm.invalid/v1",
                "temperature": 0,
            },
            "embedding": {
                "model": "text-embedding-3-small",
                "api_key": "test",
                "base_url": "https://embedding.invalid/v1",
            },
            "memory": {"return_history": True},
        },
        "retrieval": {"strategy": "prompt"},
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test",
            "base_url": "https://llm.invalid/v1",
        },
        "search": {
            "top_k": 20,
            "temporal_fetch_multiplier": 5,
            "temporal_fetch_cap": 100,
        },
    }


def make_adapter(tmp_path, monkeypatch):
    repo_root = MemoraAdapter._repo_root()
    adapter = MemoraAdapter(
        make_config(repo_root / "external" / "Memora"), output_dir=tmp_path
    )

    def fake_load():
        adapter._client_class = FakeClient
        adapter._segmenter_class = FakeSegmenter
        adapter._omega_conf = OmegaConf

    monkeypatch.setattr(adapter, "_load_upstream", fake_load)
    FakeClient.instances.clear()
    FakeClient.active_store_operations = 0
    FakeClient.max_active_store_operations = 0
    return adapter


def make_conversation(dataset_name):
    return Conversation(
        conversation_id=f"{dataset_name}_0",
        messages=[
            Message(
                sender_id="u",
                sender_name="User",
                content="Alice moved to Seattle.",
                timestamp=datetime(2024, 1, 1, 9, 0),
                metadata={"session": "session_0"},
            ),
            Message(
                sender_id="a",
                sender_name="Assistant",
                content="I will remember that.",
                timestamp=datetime(2024, 1, 1, 9, 1),
                metadata={"session": "session_0"},
            ),
            Message(
                sender_id="u",
                sender_name="User",
                content="Alice later moved to Paris.",
                timestamp=datetime(2024, 2, 1, 9, 0),
                metadata={"session": "session_1"},
            ),
        ],
    )


def test_memora_is_registered_and_config_is_fair_to_mem0_local():
    assert "memora" in list_adapters()
    memora = load_yaml("evaluation/config/systems/memora.yaml")
    mem0 = load_yaml("evaluation/config/systems/mem0_local.yaml")
    assert memora["search"]["top_k"] == mem0["search"]["top_k"] == 20
    assert memora["llm"]["model"] == mem0["llm"]["model"]
    assert memora["llm"]["base_url"] == mem0["llm"]["base_url"]
    assert memora["answer"]["prompt_key"] == mem0["answer"]["prompt_key"]
    assert memora["comparison_protocol"]["native_memory_construction_and_retrieval"]
    assert memora["memora"]["timeout_seconds"] == 600
    assert memora["memora"]["sdk_max_retries"] == 3
    assert memora["memora"]["segment_workers"] == 2
    assert memora["memora"]["store_chunk_chars"] == 10000
    assert memora["memora"]["min_store_chunk_chars"] == 1000
    assert (
        memora["dataset_overrides"]["memoryagentbench"]["memora"][
            "recycle_every_sessions"
        ]
        == 100
    )


def test_long_store_input_is_losslessly_chunked_and_resumable(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path, monkeypatch)
    adapter.store_chunk_chars = 32
    conversation = make_conversation("memoryagentbench")
    conversation.messages = [
        Message(
            sender_id="u",
            sender_name="User",
            content="First sentence has facts. Second sentence has more facts. Third one.",
            timestamp=datetime(2024, 1, 1, 9, 0),
            metadata={"session": "session_0"},
        )
    ]

    asyncio.run(adapter.add([conversation]))
    client = adapter._get_client(conversation.conversation_id)
    stored = "".join(
        message["content"]
        for call_messages, _type, _metadata in client.add_calls
        for message in call_messages
    )
    assert stored == "User: " + conversation.messages[0].content
    assert len(client.add_calls) > 1
    assert all(
        len(message["content"]) <= adapter.store_chunk_chars
        for call_messages, _type, _metadata in client.add_calls
        for message in call_messages
    )

    state = adapter._load_resume_state()
    completed = state["conversations"][conversation.conversation_id]["completed"]
    assert completed == [f"0:0:{index}" for index in range(len(client.add_calls))]
    assert [call[2]["segment_index"] for call in client.add_calls] == [
        f"0.{index}" for index in range(len(client.add_calls))
    ]

    asyncio.run(adapter.add([conversation]))
    assert len(client.add_calls) == len(completed)


def test_length_overflow_is_cleaned_split_and_resumed(tmp_path, monkeypatch):
    class LengthFinishReasonError(Exception):
        pass

    class LengthLimitedClient(FakeClient):
        def __init__(self, cfg, user_id):
            super().__init__(cfg, user_id)
            self.failed_episodic = set()
            self.delete_calls = []

        def add(self, messages, type, metadata):
            content_length = sum(len(message["content"]) for message in messages)
            if content_length > 15:
                key = MemoraAdapter._episodic_key(metadata)
                self.failed_episodic.add(key)
                raise LengthFinishReasonError("structured output was truncated")
            return super().add(messages, type, metadata)

        def get(self, key):
            if key in self.failed_episodic:
                return SimpleNamespace(index=key)
            return super().get(key)

        def delete(self, key):
            self.delete_calls.append(key)
            self.failed_episodic.discard(key)

    adapter = make_adapter(tmp_path, monkeypatch)
    adapter._load_upstream()
    adapter._client_class = LengthLimitedClient
    monkeypatch.setattr(adapter, "_load_upstream", lambda: None)
    adapter.store_chunk_chars = 100
    adapter.min_store_chunk_chars = 5
    conversation = make_conversation("memoryagentbench")
    conversation.messages = conversation.messages[:1]

    asyncio.run(adapter.add([conversation]))
    client = adapter._get_client(conversation.conversation_id)
    stored = "".join(
        message["content"]
        for call_messages, _type, _metadata in client.add_calls
        for message in call_messages
    )
    assert stored == "User: " + conversation.messages[0].content
    assert client.delete_calls
    assert client.failed_episodic == set()
    assert all(
        sum(len(message["content"]) for message in call_messages) <= 15
        for call_messages, _type, _metadata in client.add_calls
    )

    calls_after_first_run = len(client.add_calls)
    completed = adapter._load_resume_state()["conversations"][
        conversation.conversation_id
    ]["completed"]
    assert any(key.endswith(":split") for key in completed)
    asyncio.run(adapter.add([conversation]))
    assert len(client.add_calls) == calls_after_first_run


@pytest.mark.parametrize("dataset_name", ["locomo", "meme", "memoryagentbench"])
def test_add_supports_all_requested_datasets_and_resumes(
    tmp_path, monkeypatch, dataset_name
):
    adapter = make_adapter(tmp_path, monkeypatch)
    conversation = make_conversation(dataset_name)

    first = asyncio.run(adapter.add([conversation]))
    client = adapter._get_client(conversation.conversation_id)
    assert first["conversation_ids"] == [conversation.conversation_id]
    assert client.clear_calls == 1
    assert len(client.add_calls) == 2
    assert client.add_calls[0][0][0]["content"].startswith("User:")
    assert client.add_calls[0][2]["timestamp"] == "2024-01-01T09:00:00"

    asyncio.run(adapter.add([conversation]))
    assert client.clear_calls == 1
    assert len(client.add_calls) == 2

    state = adapter._load_resume_state()
    assert state["conversations"][conversation.conversation_id]["complete"] is True


def test_complete_conversation_resume_skips_segmentation_and_store(
    tmp_path, monkeypatch
):
    adapter = make_adapter(tmp_path, monkeypatch)
    conversation = make_conversation("meme")
    signature = adapter._conversation_signature(conversation)
    adapter.resume_path.write_text(
        json.dumps(
            {
                "version": 1,
                "conversations": {
                    conversation.conversation_id: {
                        "signature": signature,
                        "completed": ["0:0", "1:0"],
                        "complete": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def unexpected_segmentation(_messages):
        raise AssertionError("completed conversation should not be segmented again")

    monkeypatch.setattr(adapter, "_segments_for_session", unexpected_segmentation)
    result = asyncio.run(adapter.add([conversation]))
    client = adapter._get_client(conversation.conversation_id)

    assert result["add_results"][0]["resumed_complete"] is True
    assert client.clear_calls == 0
    assert client.add_calls == []


def test_incremental_add_appends_without_signature_clear(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path, monkeypatch)
    source = make_conversation("beam10m")
    context_id = source.conversation_id

    for sequence_index, message in enumerate(source.messages[:2], start=1):
        incremental = Conversation(
            conversation_id=context_id,
            messages=[message],
            metadata=source.metadata,
        )
        asyncio.run(
            adapter.add_incremental_message(
                incremental, sequence_index=sequence_index
            )
        )
    asyncio.run(adapter.finalize_incremental_conversation(context_id))

    client = adapter._get_client(context_id)
    assert client.clear_calls == 0
    assert len(client.add_calls) == 2
    assert [call[2]["incremental_sequence_index"] for call in client.add_calls] == [
        1,
        2,
    ]
    state = adapter._load_resume_state()["conversations"][context_id]
    assert state["mode"] == "incremental_v1"
    assert state["complete"] is True


def test_add_requests_worker_recycle_after_configured_new_sessions(
    tmp_path, monkeypatch
):
    conversation = make_conversation("memoryagentbench")
    conversation.messages = [
        Message(
            sender_id="u",
            sender_name="User",
            content=f"MemoryAgentBench session {index}.",
            timestamp=datetime(2024, 1, 1, 9 + index, 0),
            metadata={"session": f"session_{index}"},
        )
        for index in range(3)
    ]

    first = make_adapter(tmp_path, monkeypatch)
    first.recycle_every_sessions = 2
    with pytest.raises(MemoraProcessRecycleRequested):
        asyncio.run(first.add([conversation]))

    first_state = first._load_resume_state()["conversations"][
        conversation.conversation_id
    ]
    assert first_state["completed"] == ["0:0", "1:0"]
    assert first_state["complete"] is False

    second = make_adapter(tmp_path, monkeypatch)
    second.recycle_every_sessions = 2
    with pytest.raises(MemoraProcessRecycleRequested):
        asyncio.run(second.add([conversation]))

    second_state = second._load_resume_state()["conversations"][
        conversation.conversation_id
    ]
    assert second_state["completed"] == ["0:0", "1:0", "2:0"]
    assert second_state["complete"] is True

    third = make_adapter(tmp_path, monkeypatch)
    third.recycle_every_sessions = 2
    result = asyncio.run(third.add([conversation]))
    assert result["add_results"][0]["resumed_complete"] is True


def test_add_uses_distinct_shards_and_parallel_store_writes(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path, monkeypatch)
    conversations = [make_conversation("locomo"), make_conversation("meme")]

    asyncio.run(adapter.add(conversations))

    locomo_client = adapter._get_client("locomo_0")
    meme_client = adapter._get_client("meme_0")
    assert locomo_client.cfg.memory.persist_path != meme_client.cfg.memory.persist_path
    assert Path(locomo_client.cfg.memory.persist_path).parent == adapter.persist_path
    assert Path(meme_client.cfg.memory.persist_path).parent == adapter.persist_path
    assert FakeClient.max_active_store_operations == 2


def test_upstream_delete_tolerates_stale_missing_cue(tmp_path):
    """A deleted/promoted cue must not abort a resumable Memora add stage."""
    source_root = MemoraAdapter._repo_root() / "external" / "Memora" / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from memora.core.memory import AgentMemory
    from memora.core.memory_entry import MemoryEntry

    class Store:
        def __init__(self):
            self.deleted = []

        def get(self, key):
            return None

        def delete(self, key):
            self.deleted.append(key)

    memory = AgentMemory.__new__(AgentMemory)
    memory._store = Store()
    entry = MemoryEntry(
        index="Malintzin's intelligence and resourcefulness",
        value="Malintzin was intelligent and resourceful.",
        cue_indices="Malintzin intelligence||Malintzin resourcefulness",
    )

    memory._delete_primary_memory(entry)

    assert memory._store.deleted == [entry.index]


def test_search_uses_native_strategy_and_filters_future_memory(tmp_path, monkeypatch):
    adapter = make_adapter(tmp_path, monkeypatch)
    conversation = make_conversation("meme")
    client = adapter._get_client(conversation.conversation_id)
    client.memories = [
        FakeEntry("past", "2024-01-01T09:00:00", 0.9),
        FakeEntry("future", "2024-02-01T09:00:00", 0.8),
    ]

    result = asyncio.run(
        adapter.search(
            "Where did Alice move?",
            conversation.conversation_id,
            None,
            question_id="meme_0_before_qa0",
            question_date="2024-01-15T00:00:00",
        )
    )

    assert client.query_calls == [("Where did Alice move?", 100, "prompt")]
    assert [item["content"] for item in result.results] == [
        "Details: 2024-01-01T09:00:00: past"
    ]
    assert "future" not in result.retrieval_metadata["formatted_context"]
    assert result.retrieval_metadata["future_filtered_result_count"] == 1
    assert result.question_id == "meme_0_before_qa0"
