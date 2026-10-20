import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from evaluation.src.adapters.memu_local_adapter import MemuLocalAdapter
from evaluation.src.core.data_models import Conversation, Message


class FakeMemUService:
    def __init__(self, fail_on_call=None, delay=0):
        self.fail_on_call = fail_on_call
        self.delay = delay
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def memorize(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.delay:
            await asyncio.sleep(self.delay)
        self.active -= 1
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("simulated interruption")
        return {"items": [{"id": len(self.calls)}]}


def make_adapter(tmp_path: Path, service: FakeMemUService):
    adapter = object.__new__(MemuLocalAdapter)
    adapter.config = {
        "run_name": "resume-test",
        "add_resume": {
            "enabled": True,
            "state_file": "memu_local_add_resume.json",
        },
    }
    adapter.output_dir = tmp_path
    adapter.console = Console(file=StringIO())
    adapter.resource_dir = tmp_path / "memu_resources"
    adapter.resource_dir.mkdir(parents=True, exist_ok=True)
    adapter._resource_paths = {}
    adapter._service_lock = asyncio.Lock()
    adapter._add_resume_lock = asyncio.Lock()
    adapter.num_workers = 1
    adapter.add_resume_enabled = True
    adapter.add_resume_state_path = tmp_path / "memu_local_add_resume.json"
    adapter._add_resume_state = adapter._load_add_resume_state()
    adapter.service = service
    return adapter


def make_conversation(conversation_id="conversation_0"):
    return Conversation(
        conversation_id=conversation_id,
        messages=[
            Message(sender_id="u", sender_name="User", content="one"),
            Message(sender_id="u", sender_name="User", content="two"),
            Message(sender_id="u", sender_name="User", content="three"),
        ],
        metadata={"speaker_a": "User"},
    )


def test_memu_local_add_resumes_completed_chunks(tmp_path):
    conversation = make_conversation()
    interrupted = make_adapter(tmp_path, FakeMemUService(fail_on_call=2))

    with pytest.raises(RuntimeError, match="simulated interruption"):
        asyncio.run(interrupted.add([conversation]))

    first_state = interrupted._add_resume_state["conversations"]["conversation_0"]
    assert first_state["completed_parts"] == [0]

    resumed_service = FakeMemUService()
    resumed = make_adapter(tmp_path, resumed_service)
    asyncio.run(resumed.add([conversation]))

    assert len(resumed_service.calls) == 2
    assert resumed._add_resume_state["conversations"]["conversation_0"]["completed"]


def test_memu_local_resume_rejects_changed_input(tmp_path):
    conversation = make_conversation()
    adapter = make_adapter(tmp_path, FakeMemUService())
    asyncio.run(adapter.add([conversation]))

    changed = make_conversation()
    changed.messages[0].content = "changed"
    resumed = make_adapter(tmp_path, FakeMemUService())

    with pytest.raises(RuntimeError, match="does not match conversation"):
        asyncio.run(resumed.add([changed]))


def test_memu_local_add_limits_conversation_concurrency(tmp_path):
    service = FakeMemUService(delay=0.01)
    adapter = make_adapter(tmp_path, service)
    adapter.num_workers = 2
    conversations = [make_conversation(f"conversation_{index}") for index in range(4)]

    result = asyncio.run(adapter.add(conversations))

    assert service.max_active == 2
    assert [item["conversation_id"] for item in result["conversations"]] == [
        "conversation_0",
        "conversation_1",
        "conversation_2",
        "conversation_3",
    ]


def test_memu_local_memoryagentbench_user_ids_are_isolated_by_source(tmp_path):
    adapter = make_adapter(tmp_path, FakeMemUService())
    adapter.config.update(
        {
            "dataset_name": "memoryagentbench",
            "memoryagentbench": {
                "split": "Test_Time_Learning",
                "source": "recsys_redial_full",
            },
        }
    )

    assert adapter._user_id("memoryagentbench_0") == (
        "resume-test:Test_Time_Learning:recsys_redial_full:memoryagentbench_0"
    )

    adapter.config["memoryagentbench"]["source"] = "icl_banking77_5900shot_balance"
    assert adapter._user_id("memoryagentbench_0") == (
        "resume-test:Test_Time_Learning:"
        "icl_banking77_5900shot_balance:memoryagentbench_0"
    )
