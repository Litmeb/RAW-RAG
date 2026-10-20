import asyncio
import json
import threading
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rich.console import Console

from evaluation.src.adapters.mem0_adapter import Mem0Adapter, _AsyncLocalMemoryClient
from evaluation.src.core.data_models import Conversation, Message


class FakeClient:
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.add_calls = []
        self.search_calls = []

    async def add(self, **kwargs):
        self.add_calls.append(kwargs)
        return {"results": []}

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"results": []}


def test_local_worker_finishes_before_cancellation_releases_operation_lock():
    started = threading.Event()
    release = threading.Event()

    def blocking_operation():
        started.set()
        release.wait(timeout=5)

    client = _AsyncLocalMemoryClient(SimpleNamespace())

    async def run():
        task = asyncio.create_task(client._run_serialized(blocking_operation))
        while not started.is_set():
            await asyncio.sleep(0.01)

        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()
        assert client._operation_lock.locked()

        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected the cancelled operation to stay cancelled")

        assert not client._operation_lock.locked()

    asyncio.run(run())


def test_resume_batch_count_ignores_messages_that_add_will_drop(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    adapter.config["dataset_name"] = "longmemeval"
    adapter.batch_size = 5
    messages = [{"content": f"message-{index}"} for index in range(5)]
    messages.append({"content": "   "})

    assert adapter._get_batch_count(messages) == 1


def test_timeout_batch_is_recursively_split_until_sub_batches_succeed(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    adapter.add_timeout_split_enabled = True
    attempts = []

    class APITimeoutError(Exception):
        pass

    async def call_with_retry(_description, operation, max_attempts=None):
        return await operation()

    async def add_items(items, split_path):
        attempts.append((list(items), split_path))
        if len(items) > 1:
            raise APITimeoutError("Request timed out")

    adapter._call_with_retry = call_with_retry
    asyncio.run(
        adapter._add_with_timeout_split(
            "add batch 85/97",
            [0, 1, 2, 3, 4],
            add_items,
        )
    )

    successful_items = [items[0] for items, _path in attempts if len(items) == 1]
    assert successful_items == [0, 1, 2, 3, 4]
    assert ([0, 1], "L") in attempts
    assert ([2, 3, 4], "R") in attempts


def test_timeout_split_resume_skips_durably_completed_branch(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    adapter.add_timeout_split_enabled = True
    attempts = []
    recorded = []

    async def call_with_retry(_description, operation, max_attempts=None):
        return await operation()

    async def add_items(items, split_path):
        attempts.append((list(items), split_path))

    async def record_split(split_path):
        recorded.append(split_path)

    adapter._call_with_retry = call_with_retry
    asyncio.run(
        adapter._add_with_timeout_split(
            "add batch 85/97",
            [0, 1, 2, 3, 4],
            add_items,
            completed_split_paths={"L"},
            record_completed_split=record_split,
        )
    )

    assert attempts == [([2, 3, 4], "R")]
    assert recorded == ["R"]


def make_sharded_adapter(tmp_path: Path) -> Mem0Adapter:
    adapter = object.__new__(Mem0Adapter)
    adapter.config = {
        "client_mode": "local",
        "dataset_name": "memoryagentbench",
        "memoryagentbench": {"source": "longmemeval_s*"},
        "run_name": "shard-test",
        "local": {"memory": {}},
    }
    adapter.output_dir = tmp_path
    adapter.client_mode = "local"
    adapter.client = None
    adapter._custom_instructions = None
    adapter._local_shard_by_conversation = True
    adapter._local_shard_clients = {}
    adapter._local_shard_client_tasks = {}
    adapter._local_http_timeout_seconds = 90.0
    adapter._local_http_max_retries = 0
    adapter.search_operation_max_attempts = 1
    adapter.debug_search_results = False
    adapter.console = Console(file=StringIO())
    return adapter


def make_conversation(conversation_id: str = "memoryagentbench_0") -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        messages=[Message(sender_id="user", sender_name="User", content="hello")],
        metadata={"speaker_a": "User"},
    )


def test_local_memory_config_uses_isolated_paths_per_conversation(tmp_path):
    adapter = make_sharded_adapter(tmp_path)

    first = adapter._build_local_memory_config(adapter.config, "memoryagentbench_0")
    second = adapter._build_local_memory_config(adapter.config, "memoryagentbench_1")

    first_vector_path = Path(first["vector_store"]["config"]["path"])
    second_vector_path = Path(second["vector_store"]["config"]["path"])
    assert first_vector_path != second_vector_path
    assert first_vector_path.parent.parent == tmp_path / "mem0_shards"
    assert second_vector_path.parent.parent == tmp_path / "mem0_shards"
    assert Path(first["history_db_path"]) == first_vector_path.parent / "history.db"
    assert Path(second["history_db_path"]) == second_vector_path.parent / "history.db"


def test_default_shard_root_does_not_repeat_relative_output_dir():
    adapter = make_sharded_adapter(Path("evaluation/results/test-run"))

    shard_dir = adapter._local_shard_dir("memoryagentbench_0")

    assert shard_dir.parent == Path("evaluation/results/test-run/mem0_shards")


def test_client_router_reuses_one_client_per_conversation(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    created = []

    def create_client(_config, conversation_id):
        created.append(conversation_id)
        return FakeClient(conversation_id)

    adapter._create_local_mem0_client = create_client

    async def run():
        first_a, first_b, second = await asyncio.gather(
            adapter._client_for_conversation("memoryagentbench_0"),
            adapter._client_for_conversation("memoryagentbench_0"),
            adapter._client_for_conversation("memoryagentbench_1"),
        )
        return first_a, first_b, second

    first_a, first_b, second = asyncio.run(run())

    assert first_a is first_b
    assert first_a is not second
    assert sorted(created) == ["memoryagentbench_0", "memoryagentbench_1"]


def test_add_and_search_route_to_conversation_shard(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    clients = {
        conversation_id: FakeClient(conversation_id)
        for conversation_id in ("memoryagentbench_0", "memoryagentbench_1")
    }
    adapter._create_local_mem0_client = lambda _config, conversation_id: clients[
        conversation_id
    ]
    adapter.max_retries = 1
    adapter.search_interval = 0
    adapter.mem0_search_rerank = False
    adapter._temporal_fetch_top_k = lambda top_k, _question_date: top_k
    adapter._filter_results_for_question_date = (
        lambda results, _question_date, **_kwargs: (results, {})
    )
    adapter._remember_temporal_filter_metadata = lambda *_args: None

    async def call_with_retry(_description, operation, max_attempts=None):
        return await operation()

    adapter._call_with_retry = call_with_retry

    async def run():
        await adapter._add_messages_to_client(
            messages=[{"role": "user", "content": "first"}],
            source_timestamp=None,
            user_id="user-0",
            run_id="memoryagentbench_0",
            metadata={},
        )
        await adapter._search_single_user(
            "query", "memoryagentbench_1", "user-1", 5
        )

    asyncio.run(run())

    assert len(clients["memoryagentbench_0"].add_calls) == 1
    assert not clients["memoryagentbench_0"].search_calls
    assert len(clients["memoryagentbench_1"].search_calls) == 1
    assert not clients["memoryagentbench_1"].add_calls
    search_call = clients["memoryagentbench_1"].search_calls[0]
    assert search_call["limit"] == 5
    assert "top_k" not in search_call
    assert search_call["user_id"] == "user-1"


def test_local_openai_compatible_clients_receive_bounded_timeout(tmp_path):
    adapter = make_sharded_adapter(tmp_path)

    class ConfigurableClient:
        def __init__(self):
            self.calls = []

        def with_options(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(configured=kwargs)

    embedding_client = ConfigurableClient()
    llm_client = ConfigurableClient()
    memory = SimpleNamespace(
        embedding_model=SimpleNamespace(client=embedding_client),
        llm=SimpleNamespace(client=llm_client),
        reranker=None,
    )

    adapter._configure_local_http_clients(memory)

    expected = {"timeout": 90.0, "max_retries": 0}
    assert embedding_client.calls == [expected]
    assert llm_client.calls == [expected]
    assert memory.embedding_model.client.configured == expected
    assert memory.llm.client.configured == expected


def test_checkpoint_flushes_on_interval_and_force_flush(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    adapter.add_resume_enabled = True
    adapter.add_resume_flush_every_batches = 3
    adapter._add_resume_state_path = tmp_path / "resume.json"
    adapter._add_resume_state = {
        "version": 2,
        "storage_layout": "conversation_shards",
        "conversations": {},
    }
    adapter._add_resume_state_lock = asyncio.Lock()
    adapter._add_resume_dirty_batches = 0
    adapter._write_add_resume_state_locked = AsyncMock()
    conversation = make_conversation()

    async def run():
        for batch_index in range(2):
            await adapter._record_completed_batch(
                conversation,
                user_id="user-0",
                speaker="speaker_a",
                batch_index=batch_index,
                total_batches=4,
                batch_start=batch_index,
                batch_size=1,
            )
        assert adapter._write_add_resume_state_locked.await_count == 0

        await adapter._record_completed_batch(
            conversation,
            user_id="user-0",
            speaker="speaker_a",
            batch_index=2,
            total_batches=4,
            batch_start=2,
            batch_size=1,
        )
        assert adapter._write_add_resume_state_locked.await_count == 1

        await adapter._record_completed_batch(
            conversation,
            user_id="user-0",
            speaker="speaker_a",
            batch_index=3,
            total_batches=4,
            batch_start=3,
            batch_size=1,
        )
        await adapter._save_add_resume_state()

    asyncio.run(run())

    assert adapter._write_add_resume_state_locked.await_count == 2
    assert adapter._add_resume_dirty_batches == 0


def test_checkpoint_rejects_shared_layout_for_sharded_run(tmp_path):
    adapter = make_sharded_adapter(tmp_path)
    adapter.batch_size = 2
    adapter.add_resume_state_file = "resume.json"
    (tmp_path / "resume.json").write_text(
        json.dumps({"version": 1, "conversations": {}}), encoding="utf-8"
    )

    try:
        adapter._init_add_resume_state()
    except ValueError as error:
        assert "storage layout mismatch" in str(error)
    else:
        raise AssertionError("Expected the incompatible checkpoint to be rejected")
