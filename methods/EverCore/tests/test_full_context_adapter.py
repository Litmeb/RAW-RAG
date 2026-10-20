import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace

import pytest

from evaluation.src.adapters.registry import create_adapter
from evaluation.src.core.data_models import Conversation, Message, QAPair
from evaluation.src.core.stages.answer_stage import run_answer_stage


def adapter(**answer):
    return create_adapter("full_context", {
        "adapter": "full_context", "llm": {"model": "gpt-4o-mini", "max_tokens": 100},
        "answer": {"context_window": 1800, "context_safety_margin": 32, **answer},
    })


@pytest.mark.parametrize("strategy", ["head", "tail", "head_tail"])
@pytest.mark.parametrize("prompt", ["answer_prompt_cot", "answer_prompt_meme_official", "answer_prompt_mab_qa"])
def test_complete_request_fits(strategy, prompt):
    a = adapter(truncation_strategy=strategy)
    history = "BEGIN " + "中文🙂 evidence <|endoftext|> " * 2000 + " END"
    context, audit = a.prepare_answer_context("question " * 50, history, prompt)
    assert audit["truncated"]
    assert audit["input_tokens"] + a.output_tokens + a.margin <= a.window
    assert "BEGIN" in context if strategy != "tail" else "END" in context
    if strategy == "head_tail":
        assert context.endswith(" END")


def test_short_history_and_impossible_question():
    a = adapter()
    text, audit = a.prepare_answer_context("Question?", "history")
    assert text == "history" and not audit["truncated"]
    with pytest.raises(ValueError, match="Question and prompt"):
        a.prepare_answer_context("question " * 5000, "history")
    with pytest.raises(ValueError, match="context_window"):
        adapter(context_window=50)


def test_unknown_tokenizer_is_conservative():
    a = adapter(tokenizer_model="unknown-model", context_window=6000)
    _, audit = a.prepare_answer_context("question", "中文🙂" * 2000)
    assert audit["token_count_method"] == "utf8_bytes"
    assert audit["input_tokens"] + a.output_tokens + a.margin <= a.window


def test_history_isolation_resume_and_cutoff():
    a = adapter()
    conv = Conversation("c1", [
        Message("u", "User", "old", datetime(2024, 1, 1)),
        Message("u", "User", "future", datetime(2025, 1, 1)),
        Message("u", "User", "unknown"),
    ])
    other = Conversation("c2", [Message("u", "User", "SECRET")])
    index = a.build_lazy_index([conv, other], None)
    sr = asyncio.run(a.search("q", "c1", index, question_date="2024-06-01"))
    assert "old" in sr.retrieval_metadata["formatted_context"]
    assert sr.retrieval_metadata["excluded_messages"] == 2
    assert "SECRET" not in sr.retrieval_metadata["formatted_context"]
    full = asyncio.run(a.search("q", "c1", asyncio.run(a.add([conv]))))
    assert all(text in full.retrieval_metadata["formatted_context"] for text in ("old", "future", "unknown"))


@pytest.mark.parametrize("dataset,prompt", [
    ("locomo", None), ("meme", "answer_prompt_meme_official"),
    ("longmemeval", None), ("memoryagentbench", "answer_prompt_mab_qa"),
])
def test_pipeline_saves_actual_bounded_context(dataset, prompt):
    a = adapter()
    a.config["dataset_name"] = dataset
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="FINAL ANSWER: yes"))])

    a._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    conv = Conversation("c", [Message("u", "User", "history " * 10000)])
    qa = QAPair("q", "question?", "yes", metadata={"prompt_key": prompt})
    sr = asyncio.run(a.search(qa.question, "c", asyncio.run(a.add([conv]))))
    results = asyncio.run(run_answer_stage(a, [qa], [sr], None, logging.getLogger(__name__)))
    result = results[0]
    assert result.answer == "yes"
    assert result.metadata["context_budget"]["truncated"]
    assert requests[0]["messages"] == a._messages(qa.question, result.formatted_context, prompt)
    assert a._request_tokens(requests[0]["messages"]) + requests[0]["max_tokens"] + a.margin <= a.window


def test_mab_child_preserves_window():
    from evaluation.cli import build_memoryagentbench_child_command
    args = SimpleNamespace(
        dataset="memoryagentbench", system="full_context", stages=None,
        smoke=False, smoke_messages=10, smoke_questions=3, conv_range=None,
        from_conv=0, to_conv=None, locomo_sample=None, sample_workers=None,
        run_name="cw8k", clean_groups=False, context_window=8192,
        mab_disable_official_chunking=True, mab_chunk_strategy="source_aware",
        mab_chunk_size=None, mab_chunk_overlap=None,
    )
    command = build_memoryagentbench_child_command(args, "Accurate_Retrieval", "eventqa_full", None)
    assert command[command.index("--context-window") + 1] == "8192"
    assert "--mab-disable-official-chunking" in command
