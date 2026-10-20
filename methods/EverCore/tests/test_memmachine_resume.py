import asyncio
import json
from pathlib import Path

import pytest

from evaluation.src.adapters.memmachine_adapter import MemMachineAdapter
from evaluation.src.core.data_models import Conversation, Message


def _config(**overrides):
    config = {
        "base_url": "http://localhost:8080/api/v2",
        "org_id": "ragforlocomo",
        "project_prefix": "eval",
        "run_name": "resume-test",
        "memory_types": ["episodic"],
        "batch_size": 2,
        "llm": {
            "provider": "openai",
            "model": "test-model",
            "api_key": "test-api-key",
        },
    }
    config.update(overrides)
    return config


def _conversation() -> Conversation:
    return Conversation(
        conversation_id="longmemeval_0",
        messages=[
            Message(
                sender_id="user",
                sender_name="User",
                content=f"message {index}",
            )
            for index in range(5)
        ],
    )


def test_memmachine_add_resumes_after_each_successful_batch(tmp_path: Path):
    asyncio.run(_run_resume_scenario(tmp_path))


async def _run_resume_scenario(tmp_path: Path):
    first = MemMachineAdapter(_config(), output_dir=tmp_path)
    conversation = _conversation()
    messages = first._conversation_to_messages(conversation)
    first_payloads = []

    async def fail_on_second_batch(method, url, **kwargs):
        del method, url
        first_payloads.append(kwargs["json"])
        if len(first_payloads) == 2:
            raise RuntimeError("interrupted")
        return {}

    first._request_json_with_retry = fail_on_second_batch
    with pytest.raises(RuntimeError, match="interrupted"):
        await first._add_user_messages(conversation, messages, "speaker_a")

    checkpoint = json.loads(
        (tmp_path / "memmachine_add_resume.json").read_text(encoding="utf-8")
    )
    assert checkpoint["conversations"]["longmemeval_0"]["completed_messages"] == 2

    resumed = MemMachineAdapter(_config(), output_dir=tmp_path)
    resumed_payloads = []

    async def record_batch(method, url, **kwargs):
        del method, url
        resumed_payloads.append(kwargs["json"])
        return {}

    resumed._request_json_with_retry = record_batch
    await resumed._add_user_messages(conversation, messages, "speaker_a")

    assert [
        [message["content"] for message in payload["messages"]]
        for payload in resumed_payloads
    ] == [["message 2", "message 3"], ["message 4"]]
    checkpoint = json.loads(
        (tmp_path / "memmachine_add_resume.json").read_text(encoding="utf-8")
    )
    assert checkpoint["conversations"]["longmemeval_0"]["completed_messages"] == 5


def test_memmachine_add_exposes_episode_ids_for_async_barriers(tmp_path: Path):
    async def scenario():
        adapter = MemMachineAdapter(
            _config(add_resume={"enabled": False}), output_dir=tmp_path
        )
        conversation = _conversation()
        responses = iter(
            [
                {"results": [{"uid": "episode-1"}, {"uid": "episode-2"}]},
                {"results": [{"uid": "episode-3"}, {"uid": "episode-4"}]},
                {"results": [{"uid": "episode-5"}]},
            ]
        )

        async def request(_method, _url, **_kwargs):
            return next(responses)

        adapter._request_json_with_retry = request
        result = await adapter._add_user_messages(
            conversation,
            adapter._conversation_to_messages(conversation),
            "speaker_a",
        )
        assert result["episode_ids"] == [
            "episode-1",
            "episode-2",
            "episode-3",
            "episode-4",
            "episode-5",
        ]

    asyncio.run(scenario())


def test_memmachine_resume_rejects_changed_conversation(tmp_path: Path):
    adapter = MemMachineAdapter(_config(), output_dir=tmp_path)
    conversation = _conversation()
    entry = adapter._add_resume_entry(conversation, len(conversation.messages))
    entry["completed_messages"] = 2
    tmp_path.joinpath("memmachine_add_resume.json").write_text(
        json.dumps(adapter._add_resume_state), encoding="utf-8"
    )

    changed = _conversation()
    changed.messages[0].content = "changed"
    resumed = MemMachineAdapter(_config(), output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="does not match conversation"):
        resumed._add_resume_entry(changed, len(changed.messages))


def test_memmachine_resume_promotes_newer_pending_checkpoint(tmp_path: Path):
    adapter = MemMachineAdapter(_config(), output_dir=tmp_path)
    conversation = _conversation()
    entry = adapter._add_resume_entry(conversation, len(conversation.messages))
    entry["completed_messages"] = 2
    checkpoint_path = tmp_path / "memmachine_add_resume.json"
    checkpoint_path.write_text(
        json.dumps(adapter._add_resume_state), encoding="utf-8"
    )

    entry["completed_messages"] = 4
    pending_path = tmp_path / "memmachine_add_resume.json.tmp"
    pending_path.write_text(
        json.dumps(adapter._add_resume_state), encoding="utf-8"
    )

    resumed = MemMachineAdapter(_config(), output_dir=tmp_path)

    assert resumed._add_resume_state["conversations"]["longmemeval_0"][
        "completed_messages"
    ] == 4
    assert not pending_path.exists()
    promoted = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert promoted["conversations"]["longmemeval_0"]["completed_messages"] == 4
