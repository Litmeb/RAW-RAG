"""Malformed judge output must not become a MEME correctness label."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evaluation.src.core.data_models import AnswerResult
from evaluation.src.evaluators.meme_judge import MEMEJudge


def make_judge(contents, *, num_runs=1, max_attempts=3):
    judge = MEMEJudge({
        "llm": {"api_key": "offline", "base_url": "http://127.0.0.1:1/v1"},
        "num_runs": num_runs,
        "max_attempts": max_attempts,
        "retry_delay_seconds": 0,
    })
    judge.client.chat.completions.create = AsyncMock(side_effect=[
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        for content in contents
    ])
    return judge


@pytest.mark.parametrize("task", ["Tr", "Del", "Cas", "Abs"])
@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize("label", [True, False])
def test_valid_boolean_labels_are_preserved(task, phase, label):
    judge = make_judge([json.dumps({"correct": label, "reason": "checked"})])
    result = asyncio.run(judge._judge_single("q", "gold", "answer", task, phase))
    assert result["u_pass"] is label
    assert result["u_reason"] == "checked"
    assert judge.client.chat.completions.create.await_count == 1


@pytest.mark.parametrize("content", [
    '{"correct": "false"}', '{"correct": "true"}',
    '{"correct": 0}', '{"correct": 1}', '{"correct": null}',
    '{"correct": []}', '{"correct": {}}', '{}',
    '[]', 'null', 'false', 'not JSON', '', None,
    '{"correct": true, "reason": false}',
])
def test_invalid_single_response_is_retried_not_scored(content):
    judge = make_judge([content, '{"correct": false}'])
    result = asyncio.run(judge._judge_single("q", "gold", "answer", "Del", "after"))
    assert result["u_pass"] is False
    assert judge.client.chat.completions.create.await_count == 2


@pytest.mark.parametrize("content", [
    '{"results": [{"present": "false"}, {"present": true}]}',
    '{"results": [{"present": 0}, {"present": true}]}',
    '{"results": [{"present": null}, {"present": true}]}',
    '{"results": [{}, {"present": true}]}',
    '{"results": [null, {"present": true}]}',
    '{"results": [{"present": true}]}',
    '{"results": [{"present": true}, {"present": true}, {"present": true}]}',
    '{"results": {}}', '{"results": null}', '{}', '[]',
])
def test_invalid_aggregation_is_retried_not_defaulted(content):
    valid = '{"results": [{"present": false}, {"present": true}]}'
    judge = make_judge([content, valid])
    result = asyncio.run(judge._judge_multi("q", {"a": "A", "b": "B"}, "answer"))
    assert result["u_pass"] is False
    assert result["u_pass_per_entity"] == {"a": False, "b": True}
    assert result["u_pass_count"] == 1
    assert result["u_pass_total"] == 2
    assert judge.client.chat.completions.create.await_count == 2


@pytest.mark.parametrize("content", [
    '{"correct": false}',
    '```json\n{"correct": false}\n```',
    'Result: {"correct": false} End.',
])
def test_existing_json_wrappers_remain_supported(content):
    judge = make_judge([content])
    assert asyncio.run(judge._call_json("q"))["correct"] is False


@pytest.mark.parametrize("task", ["Del", "Agg"])
def test_exhausted_invalid_responses_abort_evaluation(task):
    judge = make_judge(['{}'] * 3)
    answer = AnswerResult(
        "q1", "q", "answer", "gold", conversation_id="c1",
        metadata={"task_type": task, "phase": "after", "entity_values": {"e": "gold"}},
    )
    with pytest.raises(ValueError, match="Invalid MEME judge response after 3 attempts"):
        asyncio.run(judge.evaluate([answer]))
    assert judge.client.chat.completions.create.await_count == 3


def test_retries_do_not_change_number_of_independent_judgments():
    judge = make_judge([
        '{"correct": "false"}', '{"correct": false}',
        '{"correct": true}', '{"correct": false}',
    ], num_runs=3)
    answer = AnswerResult(
        "q1", "q", "answer", "gold", conversation_id="c1",
        metadata={"task_type": "Del", "phase": "after"},
    )
    result = asyncio.run(judge._evaluate_single_answer(answer))
    assert result["u_pass"] is False
    assert result["u_score"] == pytest.approx(1 / 3)
    assert len(result["judge_runs"]) == 3
    assert judge.client.chat.completions.create.await_count == 4


def test_exact_recall_still_avoids_model_calls():
    judge = make_judge([])
    result = asyncio.run(judge._judge_single("q", "gold", "gold", "ER", "after"))
    assert result["u_pass"] is True
    judge.client.chat.completions.create.assert_not_awaited()


def test_transport_errors_propagate_without_becoming_labels():
    judge = make_judge([])
    judge.client.chat.completions.create.side_effect = RuntimeError("service unavailable")
    with pytest.raises(RuntimeError, match="service unavailable"):
        asyncio.run(judge._call_json("q"))
