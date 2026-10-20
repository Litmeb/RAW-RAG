import asyncio

import pytest

from evaluation.src.adapters.memu_adapter import MemuAdapter
from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.core.data_models import Conversation, Message


def _conversation(index: int) -> Conversation:
    return Conversation(
        conversation_id=f"conversation_{index}",
        messages=[Message(sender_id="user", sender_name="User", content="hello")],
        metadata={"speaker_a": "User"},
    )


def test_memu_add_limits_conversation_concurrency_and_preserves_order():
    adapter = object.__new__(MemuAdapter)
    adapter.num_workers = 2

    active = 0
    max_active = 0
    started = []

    adapter._extract_conversation_info = lambda **kwargs: {
        "need_dual_perspective": False
    }
    adapter._get_format_type = lambda: "default"
    adapter._conversation_to_messages = lambda *args, **kwargs: [
        {"name": "User", "content": "hello"}
    ]

    async def add_user_messages(conv, messages, speaker, **kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        started.append(conv.conversation_id)
        await asyncio.sleep(0.01)
        active -= 1
        return f"task-{conv.conversation_id}"

    async def wait_for_tasks(task_results, **kwargs):
        return None

    async def post_add_process(add_results, **kwargs):
        return None

    adapter._add_user_messages = add_user_messages
    adapter._wait_for_conversation_tasks = wait_for_tasks
    adapter._post_add_process = post_add_process
    adapter._build_add_result = lambda conversation_ids, add_results, **kwargs: {
        "conversation_ids": conversation_ids,
        "task_ids": add_results,
    }

    conversations = [_conversation(index) for index in range(4)]
    result = asyncio.run(adapter.add(conversations))

    assert max_active == 2
    assert result["conversation_ids"] == [
        "conversation_0",
        "conversation_1",
        "conversation_2",
        "conversation_3",
    ]
    assert result["task_ids"] == [
        "task-conversation_0",
        "task-conversation_1",
        "task-conversation_2",
        "task-conversation_3",
    ]


def test_add_drains_other_conversation_tasks_before_propagating_failure():
    adapter = object.__new__(MemuAdapter)
    adapter.num_workers = 2
    peer_started = asyncio.Event()
    peer_drained = False

    adapter._extract_conversation_info = lambda **kwargs: {
        "need_dual_perspective": False
    }
    adapter._get_format_type = lambda: "default"
    adapter._conversation_to_messages = lambda *args, **kwargs: [
        {"name": "User", "content": "hello"}
    ]

    async def add_user_messages(conv, messages, speaker, **kwargs):
        nonlocal peer_drained
        if conv.conversation_id == "conversation_0":
            await peer_started.wait()
            raise RuntimeError("first add failed")
        if conv.conversation_id == "conversation_1":
            peer_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.01)
                peer_drained = True
                raise
        raise AssertionError("queued conversations must be cancelled")

    adapter._add_user_messages = add_user_messages

    async def run():
        with pytest.raises(RuntimeError, match="first add failed"):
            await OnlineAPIAdapter.add(
                adapter, [_conversation(index) for index in range(3)]
            )
        assert peer_drained

    asyncio.run(run())
