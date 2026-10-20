import asyncio
import logging

import tiktoken

from evaluation.src.core.stages.answer_stage import (
    run_answer_stage,
    truncate_context_to_token_budget,
)
from evaluation.src.core.data_models import QAPair, SearchResult


def test_truncate_context_to_token_budget_preserves_ranked_prefix():
    context = "first-ranked evidence " + ("additional evidence " * 200)

    truncated, counts = truncate_context_to_token_budget(
        context,
        max_context_tokens=100,
        model="gpt-4o-mini",
    )

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    assert truncated.startswith("first-ranked evidence")
    assert len(encoding.encode(truncated, disallowed_special=())) <= 100
    assert counts == (None, 100)


def test_truncate_context_to_token_budget_leaves_short_context_unchanged():
    context = "short evidence"

    unchanged, counts = truncate_context_to_token_budget(
        context,
        max_context_tokens=100,
        model="unknown-compatible-model",
    )

    assert unchanged == context
    assert counts[0] == counts[1]


def test_truncate_context_to_token_budget_can_be_disabled():
    context = "evidence " * 200

    unchanged, counts = truncate_context_to_token_budget(
        context,
        max_context_tokens=0,
    )

    assert unchanged == context
    assert counts is None


def test_truncate_context_avoids_tokenizing_an_unbounded_source():
    context = "ranked evidence " * 1000

    truncated, counts = truncate_context_to_token_budget(
        context,
        max_context_tokens=10,
        model="gpt-4o-mini",
    )

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    assert truncated.startswith("ranked evidence")
    assert len(encoding.encode(truncated, disallowed_special=())) <= 10
    assert counts[0] is None


def test_answer_stage_applies_budget_before_adapter_call():
    class RecordingAdapter:
        config = {
            "answer": {"max_context_tokens": 10, "num_workers": 1},
            "llm": {"model": "gpt-4o-mini"},
        }

        async def answer(self, query, context, **_kwargs):
            self.context = context
            return "answer"

    adapter = RecordingAdapter()
    qa = QAPair("q1", "question", "gold")
    search_result = SearchResult(
        query="question",
        conversation_id="c1",
        results=[],
        question_id="q1",
        retrieval_metadata={"formatted_context": "ranked evidence " * 1000},
    )

    results = asyncio.run(
        run_answer_stage(
            adapter=adapter,
            qa_pairs=[qa],
            search_results=[search_result],
            checkpoint_manager=None,
            logger=logging.getLogger(__name__),
        )
    )

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    assert len(encoding.encode(adapter.context, disallowed_special=())) <= 10
    assert results[0].formatted_context == adapter.context
