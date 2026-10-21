"""Offline LoCoMo token-set F1 and sentence BLEU-1, optionally with saved J."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


CATEGORIES = {"4": "Single-hop", "1": "Multi-hop", "2": "Temporal", "3": "Open-domain"}


def lexical_scores(answer: str, reference: str) -> dict:
    def tokens(text):
        return set(text.lower().translate(str.maketrans(".,!?", "    ")).split())

    predicted, gold = tokens(answer), tokens(reference)
    overlap = len(predicted & gold)
    f1 = 2 * overlap / (len(predicted) + len(gold)) if overlap else 0.0
    bleu = sentence_bleu(
        [nltk.word_tokenize(reference.lower(), language="english")],
        nltk.word_tokenize(answer.lower(), language="english"),
        weights=(1, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1,
    )
    return {"f1": 100 * f1, "bleu1": 100 * bleu}


def indexed_rows(rows):
    if not isinstance(rows, list):
        raise ValueError("Expected a list of question records")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Question records must be objects")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in indexed:
            raise ValueError(f"Missing or duplicate question_id: {qid!r}")
        category = str(row.get("category"))
        if category not in {*CATEGORIES, "5"}:
            raise ValueError(f"{qid}: invalid LoCoMo category {category!r}")
        indexed[qid] = row
    return {qid: row for qid, row in indexed.items() if str(row["category"]) != "5"}


def score_records(answers, evaluations=None, expected_questions=None, expected_judgments=3):
    answers = indexed_rows(answers)
    if not answers:
        raise ValueError("No non-adversarial answers")
    if expected_questions is not None and len(answers) != expected_questions:
        raise ValueError(f"Expected {expected_questions} questions, found {len(answers)}")
    if evaluations is not None:
        evaluations = indexed_rows(evaluations)
        if answers.keys() != evaluations.keys():
            raise ValueError("Answer/judge question sets differ")
    scored = []
    for qid, row in answers.items():
        answer, gold = row.get("answer"), row.get("golden_answer")
        # LoCoMo includes numeric JSON references (for example, a year).
        valid_gold = isinstance(gold, str) or (
            type(gold) in (int, float) and math.isfinite(gold)
        )
        if not isinstance(answer, str) or not valid_gold:
            raise ValueError(f"{qid}: answer must be a string; golden_answer must be text or a finite number")
        if answer.lstrip().startswith("Error:"):
            raise ValueError(f"{qid}: answer-generation failure; repair before scoring")
        item = {
            "question_id": qid, "category": str(row["category"]),
            **lexical_scores(answer, str(gold)),
        }
        if evaluations is not None:
            evaluation = evaluations[qid]
            if (
                evaluation.get("generated_answer") != answer
                or evaluation.get("golden_answer") != gold
                or str(evaluation["category"]) != item["category"]
            ):
                raise ValueError(f"{qid}: saved judgment does not match the answer/reference/category")
            judgments = evaluation.get("llm_judgments", {})
            keys = {f"judgment_{i + 1}" for i in range(expected_judgments)}
            if (
                not isinstance(judgments, dict) or set(judgments) != keys
                or any(type(value) is not bool for value in judgments.values())
            ):
                raise ValueError(f"{qid}: expected {expected_judgments} complete boolean judgments")
            item["j"] = 100 * mean(judgments.values())
        scored.append(item)

    def summarize(rows):
        metrics = ("f1", "bleu1") if evaluations is None else ("j", "f1", "bleu1")
        return {"questions": len(rows), **{key: mean(row[key] for row in rows) for key in metrics}}

    return {
        "scale": "0-100", "excluded_category": "5",
        "overall": summarize(scored),
        "categories": {
            category: {"name": name, **summarize(selected)}
            for category, name in CATEGORIES.items()
            if (selected := [row for row in scored if row["category"] == category])
        },
        "per_question": scored,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--with-judge", action="store_true", help="Validate and include saved LLM judgments")
    parser.add_argument("--expected-questions", type=int)
    parser.add_argument("--expected-judgments", type=int, default=3)
    args = parser.parse_args()
    if args.expected_judgments < 1 or (
        args.expected_questions is not None and args.expected_questions < 1
    ):
        parser.error("Expected counts must be positive")
    run = args.run_dir.resolve()
    answer_path = run / "answer_results.json"
    judge_path = run / "eval_results.json"
    try:
        evaluations = None
        if args.with_judge:
            evaluations = read_json(judge_path)["detailed_results"]
            if isinstance(evaluations, dict):
                if any(not isinstance(rows, list) for rows in evaluations.values()):
                    raise ValueError("Judge groups must contain lists")
                evaluations = [row for rows in evaluations.values() for row in rows]
        report = score_records(
            read_json(answer_path), evaluations, args.expected_questions, args.expected_judgments,
        )
    except KeyError as exc:
        parser.exit(1, f"Cannot score LoCoMo: missing field {exc}\n")
    except LookupError as exc:
        parser.exit(1, f"NLTK tokenizer resources unavailable. Run: python -m nltk.downloader punkt_tab\n{exc}\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Cannot score LoCoMo: {exc}\n")
    report["protocol"] = {
        "f1": "lowercase; replace periods, commas, exclamation/question marks with spaces; split; token sets; empty overlap = 0",
        "bleu1": "English nltk.word_tokenize; lowercase; weights=(1,0,0,0); method1 smoothing",
        "aggregation": "question-weighted means; J averages all repetitions per question",
        "nltk_version": nltk.__version__,
        "judge_repetitions": args.expected_judgments if args.with_judge else None,
    }
    report["inputs"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ([answer_path, judge_path] if args.with_judge else [answer_path])
    }
    metrics = ("j", "f1", "bleu1") if args.with_judge else ("f1", "bleu1")
    lines = ["LoCoMo scores (0-100; category 5 excluded)", "Subset | N | " + " | ".join(metrics)]
    for name, summary in [
        ("Overall", report["overall"]),
        *((value["name"], value) for value in report["categories"].values()),
    ]:
        lines.append(f"{name} | {summary['questions']} | " + " | ".join(f"{summary[k]:.2f}" for k in metrics))
    # Separate reports leave original answers, judgments and report.txt untouched.
    (run / "lexical_metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (run / "lexical_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
