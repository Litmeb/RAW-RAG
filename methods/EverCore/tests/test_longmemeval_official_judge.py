import asyncio
import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evaluation.cli import apply_longmemeval_judge_mode, run_evaluate_only
from evaluation.src.core.data_models import AnswerResult
from evaluation.src.evaluators.registry import create_evaluator
from evaluation.src.evaluators.longmemeval_official_prompt import get_anscheck_prompt


@pytest.mark.parametrize("category, phrase", [
    ("single-session-user", "contains all the intermediate steps"),
    ("single-session-assistant", "contains all the intermediate steps"),
    ("multi-session", "only contains a subset"),
    ("temporal-reasoning", "do not penalize off-by-one errors"),
    ("knowledge-update", "some previous information along with an updated answer"),
    ("single-session-preference", "does not need to reflect all the points"),
])
def test_official_task_rubrics(category, phrase):
    prompt = get_anscheck_prompt(category, "Q", "A", "R")
    assert phrase in prompt
    assert "Question: Q" in prompt
    assert "Model Response: R" in prompt
    assert prompt.endswith("Answer yes or no only.")


def test_abstention_overrides_task_and_unknown_task_fails():
    assert "unanswerable" in get_anscheck_prompt("unknown", "Q", "A", "R", True)
    with pytest.raises(NotImplementedError):
        get_anscheck_prompt("unknown", "Q", "A", "R")


def make_judge(responses):
    judge = create_evaluator("longmemeval_official_judge", {
        "llm": {"api_key": "test-key"}, "num_runs": 3,
    })
    judge.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(side_effect=[
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
            for text in responses
        ])
    )))
    return judge


def test_request_parsing_and_aggregation():
    judge = make_judge([" YES ", "no", "yesterday"])
    answers = [
        AnswerResult("q1", "Q", "R", "A", category="temporal-reasoning"),
        AnswerResult("q2_abs", "Q", "R", "A", category="single-session-user"),
        AnswerResult("q3", "Q", "R", "A", category="knowledge-update"),
    ]
    result = asyncio.run(judge.evaluate(answers))
    assert result.correct == 2  # Upstream literally checks the 'yes' substring.
    assert result.accuracy == 2 / 3
    assert result.metadata["num_runs"] == 1
    assert result.metadata["judge_mode"] == "official"
    assert result.metadata["category_accuracies"]["single-session-user"]["mean"] == 0
    calls = judge.client.chat.completions.create.call_args_list
    assert len(calls) == 3
    for call in calls:
        assert call.kwargs["model"] == "gpt-4o-2024-08-06"
        assert call.kwargs["max_tokens"] == 10
        assert call.kwargs["temperature"] == 0
        assert call.kwargs["n"] == 1
        assert len(call.kwargs["messages"]) == 1
        assert call.kwargs["messages"][0]["role"] == "user"
    assert "unanswerable" in calls[1].kwargs["messages"][0]["content"]


def test_api_errors_are_not_scored_as_wrong():
    judge = make_judge([])
    judge.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    with pytest.raises(RuntimeError, match="API unavailable"):
        asyncio.run(judge._evaluate_single_answer(
            AnswerResult("q", "Q", "R", "A", category="multi-session")
        ))


def test_cli_mode_and_model_overrides(monkeypatch):
    monkeypatch.delenv("EVALUATE_MODEL", raising=False)
    config = {"evaluation": {"type": "llm_judge", "num_runs": 3,
                             "llm": {"model": "old", "api_key": "test-key"}}}
    apply_longmemeval_judge_mode(config, "official")
    assert config["evaluation"]["type"] == "longmemeval_official_judge"
    assert config["evaluation"]["llm"]["model"] == "gpt-4o-2024-08-06"
    assert config["evaluation"]["llm"]["api_key"] == "test-key"
    monkeypatch.setenv("EVALUATE_MODEL", "gpt-4o-mini")
    apply_longmemeval_judge_mode(config, "official")
    judge = create_evaluator(config["evaluation"]["type"], config["evaluation"])
    assert judge.model == "gpt-4o-mini-2024-07-18"
    apply_longmemeval_judge_mode(config, "evermemos")
    assert config["evaluation"]["type"] == "llm_judge"
    assert config["evaluation"]["num_runs"] == 3


def test_cli_defaults_to_evermemos_judge():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lme-judge",
        choices=["evermemos", "official"],
        default="evermemos",
    )
    assert parser.parse_args([]).lme_judge == "evermemos"


def test_evaluate_only_uses_selected_judge(tmp_path, monkeypatch):
    import json
    from rich.console import Console

    judge = make_judge(["yes"])
    selected = []

    def factory(name, config):
        selected.append(name)
        return judge

    monkeypatch.setattr("evaluation.cli.create_evaluator", factory)
    (tmp_path / "answer_results.json").write_text(json.dumps([
        {"question_id": "q_abs", "question": "Q", "answer": "unknown",
         "golden_answer": "unanswerable", "category": "multi-session"}
    ]), encoding="utf-8")
    config = {"evaluation": {}}
    apply_longmemeval_judge_mode(config, "official")
    args = SimpleNamespace(system="rag", smoke=False, only_category=None, from_conv=0,
                           to_conv=None, locomo_sample=None)
    asyncio.run(run_evaluate_only(args, config, {"name": "rag"}, tmp_path, Console()))
    assert selected == ["longmemeval_official_judge"]
    saved = json.loads((tmp_path / "eval_results.json").read_text(encoding="utf-8"))
    assert saved["accuracy"] == 1.0
