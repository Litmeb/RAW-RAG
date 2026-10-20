import asyncio
from io import StringIO
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from evaluation.src.adapters.mem0_adapter import Mem0Adapter
from evaluation.src.adapters.memos_adapter import MemosAdapter
from evaluation.src.core.data_models import Conversation, Message


class RecordingProgress:
    def __init__(self) -> None:
        self.advances = []

    def update(self, task_id, *, advance) -> None:
        self.advances.append((task_id, advance))


def make_conversation(message_count: int) -> Conversation:
    return Conversation(
        conversation_id="test_0",
        messages=[
            Message(
                sender_id="user",
                sender_name="User",
                content=f"message {index}",
            )
            for index in range(message_count)
        ],
        metadata={"speaker_a": "User"},
    )


def formatted_messages(message_count: int):
    return [
        {"role": "user", "content": f"message {index}"}
        for index in range(message_count)
    ]


def make_mem0_adapter(*, memoryagentbench: bool) -> Mem0Adapter:
    adapter = object.__new__(Mem0Adapter)
    adapter.config = {
        "dataset_name": "memoryagentbench" if memoryagentbench else "locomo",
        "memoryagentbench": {"source": "test_source"},
        "run_name": "test_run",
    }
    adapter.batch_size = 2
    adapter.max_content_length = 8_000
    adapter.max_retries = 1
    adapter.add_interval = 0
    adapter.add_resume_enabled = False
    adapter._add_resume_state = None
    adapter.console = Console(file=StringIO())
    adapter._add_messages_to_client = AsyncMock(return_value=None)

    async def call_with_retry(_description, operation):
        return await operation()

    adapter._call_with_retry = call_with_retry
    return adapter


def make_memos_adapter(*, memoryagentbench: bool) -> MemosAdapter:
    adapter = object.__new__(MemosAdapter)
    adapter.config = {
        "dataset_name": "memoryagentbench" if memoryagentbench else "locomo",
        "memoryagentbench": {"source": "test_source"},
        "run_name": "test_run",
    }
    adapter.api_url = "http://test.invalid"
    adapter.api_flavor = "openmem"
    adapter.batch_size = 2
    adapter.console = Console(file=StringIO())
    adapter._get_session = AsyncMock(return_value=object())
    adapter._send_message_batch = AsyncMock(return_value=None)
    return adapter


@pytest.mark.parametrize("adapter_factory", [make_mem0_adapter, make_memos_adapter])
def test_standard_add_progress_advances_after_each_batch(adapter_factory):
    adapter = adapter_factory(memoryagentbench=False)
    progress = RecordingProgress()

    asyncio.run(
        adapter._add_user_messages(
            make_conversation(5),
            formatted_messages(5),
            speaker="speaker_a",
            progress=progress,
            task_id=7,
        )
    )

    assert progress.advances == [(7, 2), (7, 2), (7, 1)]


@pytest.mark.parametrize("adapter_factory", [make_mem0_adapter, make_memos_adapter])
def test_memoryagentbench_add_progress_advances_per_source_chunk(
    adapter_factory,
):
    adapter = adapter_factory(memoryagentbench=True)
    progress = RecordingProgress()

    asyncio.run(
        adapter._add_user_messages(
            make_conversation(3),
            formatted_messages(3),
            speaker="speaker_a",
            progress=progress,
            task_id=9,
        )
    )

    assert progress.advances == [(9, 1), (9, 1), (9, 1)]


def test_mem0_memoryagentbench_partial_resume_skips_completed_chunks():
    adapter = make_mem0_adapter(memoryagentbench=True)
    adapter.add_resume_enabled = True
    adapter.add_resume_flush_every_batches = 20
    adapter._add_resume_state_path = None
    adapter._add_resume_state_lock = asyncio.Lock()
    adapter._add_resume_dirty_batches = 0
    adapter._add_resume_state = {
        "conversations": {
            "test_0": {
                "speakers": {
                    "speaker_a": {
                        "completed_batches": [0, 1],
                    }
                }
            }
        }
    }
    progress = RecordingProgress()

    asyncio.run(
        adapter._add_user_messages(
            make_conversation(3),
            formatted_messages(3),
            speaker="speaker_a",
            progress=progress,
            task_id=11,
        )
    )

    assert adapter._add_messages_to_client.await_count == 1
    assert progress.advances == [(11, 1), (11, 1), (11, 1)]
