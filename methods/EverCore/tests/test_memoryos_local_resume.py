import asyncio
from pathlib import Path

from evaluation.src.adapters.memoryos_adapter import MemoryOSAdapter
from evaluation.src.core.data_models import Conversation, Message


def make_adapter(tmp_path: Path, calls: list):
    adapter = object.__new__(MemoryOSAdapter)
    adapter.config = {"dataset_name": "locomo"}
    adapter.api_flavor = "memoryos"
    adapter.output_dir = tmp_path
    adapter.add_resume_enabled = True
    adapter.add_resume_state_path = tmp_path / "memoryos_add_resume.json"
    adapter._add_resume_lock = asyncio.Lock()
    adapter._add_resume_state = adapter._load_add_resume_state()
    adapter.batch_size = 1
    adapter._session = object()
    adapter._add_url = lambda: "http://test/memory/add"
    adapter._get_session = lambda: asyncio.sleep(0, result=adapter._session)
    adapter._send_message_batch = make_sender(calls)
    return adapter


def make_sender(calls):
    async def send(**kwargs):
        calls.append(kwargs["batch_messages"])

    return send


def make_conversation():
    return Conversation(
        conversation_id="conversation_0",
        messages=[
            Message(sender_id="u", sender_name="User", content="one"),
            Message(sender_id="u", sender_name="User", content="two"),
            Message(sender_id="u", sender_name="User", content="three"),
        ],
        metadata={"speaker_a": "User"},
    )


def test_memoryos_add_resumes_completed_requests(tmp_path):
    conversation = make_conversation()
    first_calls = []
    first = make_adapter(tmp_path, first_calls)

    async def interrupting_send(**kwargs):
        first_calls.append(kwargs["batch_messages"])
        if len(first_calls) == 2:
            raise RuntimeError("simulated interruption")

    first._send_message_batch = interrupting_send
    try:
        asyncio.run(
            first._add_user_messages(
                conversation,
                [{"role": "user", "content": message.content} for message in conversation.messages],
                "speaker_a",
            )
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated interruption"
    else:
        raise AssertionError("expected interruption")

    second_calls = []
    resumed = make_adapter(tmp_path, second_calls)
    asyncio.run(
        resumed._add_user_messages(
            conversation,
            [{"role": "user", "content": message.content} for message in conversation.messages],
            "speaker_a",
        )
    )

    assert len(second_calls) == 2
    assert [item[0]["content"] for item in second_calls] == ["two", "three"]
    assert resumed._add_resume_state["conversations"]["conversation_0::speaker_a"][
        "completed"
    ]


def test_memoryos_resume_rejects_changed_input(tmp_path):
    conversation = make_conversation()
    messages = [{"role": "user", "content": message.content} for message in conversation.messages]
    adapter = make_adapter(tmp_path, [])
    asyncio.run(adapter._add_user_messages(conversation, messages, "speaker_a"))

    changed = list(messages)
    changed[0] = {"role": "user", "content": "changed"}
    resumed = make_adapter(tmp_path, [])
    try:
        asyncio.run(resumed._add_user_messages(conversation, changed, "speaker_a"))
    except RuntimeError as exc:
        assert "does not match conversation" in str(exc)
    else:
        raise AssertionError("expected changed-input rejection")
