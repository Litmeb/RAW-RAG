"""Metric definitions, alignment guards and offline scoring CLI."""

import copy
import json
import math
import subprocess
import sys

import pytest

from evaluation.scripts.score_locomo_lexical import lexical_scores, score_records


def answer(qid="q1", category="4", text="red blue", gold="red green"):
    return {"question_id": qid, "category": category, "answer": text, "golden_answer": gold}


def judge(row):
    return {
        "question_id": row["question_id"], "category": row["category"],
        "generated_answer": row["answer"], "golden_answer": row["golden_answer"],
        "llm_judgments": {"judgment_1": True, "judgment_2": False, "judgment_3": True},
    }


@pytest.mark.parametrize("prediction,reference,f1,bleu", [
    ("RED red blue", "red blue", 100, 200 / 3),
    ("Hello, world!", "hello world", 100, 50),
    ("red", "red blue", 200 / 3, 100 / math.e),
    ("red", "blue", 0, 0),
    ("", "blue", 0, 0),
    ("", "", 0, 0),
    ("red-blue", "red blue", 0, 0),
    ("I can't.", "i can't.", 100, 100),
])
def test_metric_definitions(prediction, reference, f1, bleu):
    scores = lexical_scores(prediction, reference)
    assert scores["f1"] == pytest.approx(f1)
    assert scores["bleu1"] == pytest.approx(bleu)


def test_weighted_summary_and_exclusion():
    rows = [
        answer("q1", text="red", gold="red"),
        answer("q2", text="red", gold="red"),
        answer("q3", "1", text="blue", gold="red"),
        answer("q4", 5, text="ignore", gold="ignore"),
    ]
    report = score_records(rows, [judge(row) for row in rows], 3)
    assert report["overall"]["questions"] == 3
    assert report["overall"]["f1"] == pytest.approx(200 / 3)
    assert report["overall"]["j"] == pytest.approx(200 / 3)
    assert report["categories"]["4"]["f1"] == 100
    assert report["categories"]["1"]["f1"] == 0
    assert "5" not in report["categories"]


@pytest.mark.parametrize("mutation,message", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate"),
    (lambda rows: rows[0].update(category="unknown"), "invalid LoCoMo"),
    (lambda rows: rows[0].update(answer=None), "must be a string"),
    (lambda rows: rows[0].update(answer="Error: Failed to generate answer"), "generation failure"),
])
def test_invalid_answers(mutation, message):
    rows = [answer()]
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        score_records(rows)


@pytest.mark.parametrize("field,value,message", [
    ("generated_answer", "different", "does not match"),
    ("golden_answer", "different", "does not match"),
    ("category", "2", "does not match"),
    ("question_id", "missing", "sets differ"),
    ("llm_judgments", {"judgment_1": True}, "complete boolean"),
    ("llm_judgments", {"judgment_1": True, "judgment_2": 1, "judgment_3": False}, "complete boolean"),
])
def test_judge_alignment(field, value, message):
    row = answer()
    evaluation = judge(row)
    evaluation[field] = value
    with pytest.raises(ValueError, match=message):
        score_records([row], [evaluation])


def test_expected_count_and_empty_input():
    with pytest.raises(ValueError, match="Expected 1540"):
        score_records([answer()], expected_questions=1540)
    with pytest.raises(ValueError, match="No non-adversarial"):
        score_records([answer(category=5)])


def test_numeric_reference_matches_saved_judge():
    row = answer(text="2020", gold=2020)
    report = score_records([row], [judge(row)])
    assert report["overall"]["f1"] == 100
    assert report["overall"]["bleu1"] == 100


def test_cli_reports_and_preserves_inputs(tmp_path):
    row = answer(text="Blue.", gold="blue")
    for name, value in (
        ("answer_results.json", [row]),
        ("eval_results.json", {"detailed_results": {"conversation": [judge(row)]}}),
    ):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
    (tmp_path / "report.txt").write_text("original report", encoding="utf-8")
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.scripts.score_locomo_lexical",
         "--run-dir", str(tmp_path), "--with-judge", "--expected-questions", "1"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "lexical_metrics.json").read_text())
    assert report["overall"]["f1"] == 100
    assert report["overall"]["bleu1"] == 50
    assert report["protocol"]["judge_repetitions"] == 3
    assert set(report["inputs"]) == {"answer_results.json", "eval_results.json"}
    assert "Overall | 1" in (tmp_path / "lexical_report.txt").read_text()
    for name, content in originals.items():
        assert (tmp_path / name).read_bytes() == content


def test_cli_count_failure_does_not_write_reports(tmp_path):
    (tmp_path / "answer_results.json").write_text(json.dumps([answer()]))
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.scripts.score_locomo_lexical",
         "--run-dir", str(tmp_path), "--expected-questions", "1540"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "Expected 1540" in result.stderr
    assert not (tmp_path / "lexical_metrics.json").exists()


def test_missing_tokenizer_does_not_silently_change_metric(monkeypatch):
    def unavailable(*args, **kwargs):
        raise LookupError("punkt_tab missing")

    monkeypatch.setattr("evaluation.scripts.score_locomo_lexical.nltk.word_tokenize", unavailable)
    with pytest.raises(LookupError, match="punkt_tab"):
        lexical_scores("red", "red")
