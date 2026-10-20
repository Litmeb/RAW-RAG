"""
MemoryAgentBench evaluator.

This evaluator routes each QA pair to the official metric family associated with
its source/task:
- substring exact match for EventQA / RULER / FactConsolidation
- exact-match style parsing for ICL / DetectiveQA
- Recall@5 for Recsys
- official LLM-as-judge prompt family for LongMemEval
- official HELMET-style fluency/recall/precision aggregation for InfBench
"""

from __future__ import annotations

import asyncio
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from openai import AsyncOpenAI

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator


def _normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _substring_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    pred = _normalize_answer(prediction)
    return any(_normalize_answer(gt) in pred for gt in ground_truths if gt)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    fenced = re.findall(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    for candidate in reversed(fenced):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _extract_choice_label(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\(([a-zA-Z])\)", text)
    if match:
        return f"({match.group(1).lower()})"
    match = re.search(r"\b([A-D])\.\s*", text)
    if match:
        return match.group(1).upper()
    match = re.search(r"\blabel\s*:\s*([0-9A-Za-z_-]+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return text.strip()


def _extract_detective_answer(text: str) -> str:
    payload = _extract_json_object(text)
    if payload and payload.get("answer"):
        return str(payload["answer"]).strip()
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"\b([A-D]\.\s*.+)", text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_longmemeval_label(text: str) -> bool:
    return "yes" in (text or "").strip().lower()


FLUENCY_PROMPT_BOOK = """Please act as an impartial judge and evaluate the fluency of the provided text. The text should be coherent, non-repetitive, fluent, and grammatically correct.

Below is your grading rubric:
- Score 0 (incoherent, repetitive, or incomplete): Incoherent sentences, repetitive sentences (even if not by exact words), incomplete answers, or gibberish.
- Score 1 (coherent, non-repetitive answer): Coherent, non-repetitive, fluent, grammatically correct answers.

Output valid JSON only:
{{"fluency": 0}}
or
{{"fluency": 1}}

Text: "{text}"
"""

RECALL_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel. It should discuss the plots and characters of the story. The text should contain all the given key points.

Evaluate whether each key point is present in the summary. Score equals the number of supported key points.

Output valid JSON only:
{{"recall": 0}}

Key points:
{keypoints}

Summary: "{summary}"
"""

PRECISION_PROMPT_BOOK = """Please act as an impartial judge and evaluate the quality of the provided summary of a novel.

Count how many factual summary sentences are supported by the expert summary, and also count the number of summary sentences.

Output valid JSON only:
{{"precision": 0, "sentence_count": 0}}

Expert summary: <start of summary>{expert_summary}<end of summary>

Provided summary: <start of summary>{summary}<end of summary>
"""


@register_evaluator("memoryagentbench")
class MemoryAgentBenchEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        )
        self.model = llm_config.get("model", "gpt-4o")
        self.num_runs = int(config.get("num_runs", 1))
        self.judge_max_attempts = max(1, int(config.get("judge_max_attempts", 3)))
        self.data_root = (
            Path(__file__).resolve().parents[2] / "data" / "memoryagentbench"
        )

    async def evaluate(self, answer_results: List[AnswerResult]) -> EvaluationResult:
        print("\n" + "=" * 60)
        print(f"Evaluation: MemoryAgentBench (model={self.model})")
        print("=" * 60)

        semaphore = asyncio.Semaphore(6)

        async def evaluate_one(answer_result: AnswerResult) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self._evaluate_single(answer_result)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  ⚠️ Skipping question {answer_result.question_id} after "
                        f"judge failures: {reason}"
                    )
                    metadata = answer_result.metadata or {}
                    return {
                        "question_id": answer_result.question_id,
                        "question": answer_result.question,
                        "golden_answer": answer_result.golden_answer,
                        "generated_answer": answer_result.answer or "",
                        "category": answer_result.category,
                        "source": str(metadata.get("source", "unknown")),
                        "evaluation_family": str(
                            metadata.get("evaluation_family", "unknown")
                        ),
                        "metric_value": 0.0,
                        "metric_name": "skipped",
                        "skipped": True,
                        "skip_reason": reason,
                    }

        detailed_results = await asyncio.gather(
            *(evaluate_one(result) for result in answer_results)
        )

        scored_results = [item for item in detailed_results if not item.get("skipped")]
        skipped_count = len(detailed_results) - len(scored_results)
        item_scores = [item["metric_value"] for item in scored_results]
        overall = float(np.mean(item_scores)) if item_scores else 0.0

        by_source: Dict[str, List[float]] = defaultdict(list)
        for item in scored_results:
            by_source[item["source"]].append(float(item["metric_value"]))

        source_metrics = {
            source: {
                "mean": float(np.mean(values)) if values else 0.0,
                "count": len(values),
            }
            for source, values in sorted(by_source.items())
        }

        print(f"   - Total questions: {len(answer_results)}")
        print(f"   - Scored questions: {len(scored_results)}")
        print(f"   - Skipped questions: {skipped_count}")
        print(f"   - Mean score: {overall:.4f}")
        for source, metrics in source_metrics.items():
            print(f"   - {source}: {metrics['mean']:.4f} (n={metrics['count']})")

        return EvaluationResult(
            total_questions=len(answer_results),
            correct=int(round(overall * len(scored_results))) if scored_results else 0,
            accuracy=overall,
            detailed_results=detailed_results,
            metadata={
                "evaluator": "memoryagentbench",
                "model": self.model,
                "source_metrics": source_metrics,
                "skipped_count": skipped_count,
            },
        )

    async def _evaluate_single(self, answer_result: AnswerResult) -> Dict[str, Any]:
        metadata = answer_result.metadata or {}
        source = str(metadata.get("source", "unknown"))
        family = str(metadata.get("evaluation_family", "substring_exact_match"))
        gold_answers = metadata.get("gold_answers") or [answer_result.golden_answer]
        gold_answers = [str(item) for item in gold_answers if item is not None]
        generated = answer_result.answer or ""

        if family == "substring_exact_match":
            score = 1.0 if _substring_exact_match(generated, gold_answers) else 0.0
            extra = {"metric_name": "substring_exact_match"}
        elif family == "icl_exact_match":
            score = self._evaluate_icl(generated, gold_answers)
            extra = {"metric_name": "exact_match"}
        elif family == "detective_qa":
            score = self._evaluate_detective(generated, gold_answers)
            extra = {"metric_name": "exact_match"}
        elif family == "recsys_recall":
            score, extra = self._evaluate_recsys(generated, gold_answers, metadata)
        elif family == "longmemeval":
            score = await self._evaluate_longmemeval(answer_result, gold_answers)
            extra = {"metric_name": "llm_judge"}
        elif family == "infbench_sum":
            score, extra = await self._evaluate_infbench(answer_result, gold_answers)
        else:
            raise ValueError(f"Unsupported MemoryAgentBench evaluation family '{family}'")

        return {
            "question_id": answer_result.question_id,
            "question": answer_result.question,
            "golden_answer": answer_result.golden_answer,
            "generated_answer": generated,
            "category": answer_result.category,
            "source": source,
            "evaluation_family": family,
            "metric_value": float(score),
            **extra,
        }

    def _evaluate_icl(self, generated: str, gold_answers: List[str]) -> float:
        parsed_generated = _extract_choice_label(generated)
        gold = str(gold_answers[0]) if gold_answers else ""
        return 1.0 if _normalize_answer(parsed_generated) == _normalize_answer(gold) else 0.0

    def _evaluate_detective(self, generated: str, gold_answers: List[str]) -> float:
        parsed_generated = _extract_detective_answer(generated)
        gold = str(gold_answers[0]) if gold_answers else ""
        return 1.0 if _normalize_answer(parsed_generated) == _normalize_answer(gold) else 0.0

    def _evaluate_recsys(
        self,
        generated: str,
        gold_answers: List[str],
        metadata: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        split_name = metadata.get("mab_split")
        source_name = metadata.get("source")
        entity_path = (
            self.data_root
            / str(split_name)
            / str(source_name)
            / "entity2id.json"
        )
        if not entity_path.exists():
            sanitized_split = re.sub(r"[^A-Za-z0-9._-]+", "_", str(split_name))
            sanitized_source = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source_name))
            entity_path = self.data_root / sanitized_split / sanitized_source / "entity2id.json"
        name_to_id = json.loads(entity_path.read_text(encoding="utf-8"))
        id_to_name = {int(v): self._extract_movie_name(k) for k, v in name_to_id.items()}

        predicted = self._extract_recommendations(generated)
        predicted_norm = [self._extract_movie_name(item) for item in predicted]

        gt_ids = [int(item) for item in gold_answers]
        gt_movies = [id_to_name[item] for item in gt_ids if item in id_to_name]

        recall_at_5 = (
            sum(movie in predicted_norm[:5] for movie in gt_movies) / len(gt_movies)
            if gt_movies
            else 0.0
        )
        return recall_at_5, {
            "metric_name": "Recall@5",
            "predicted_movies": predicted_norm[:5],
            "gold_movies": gt_movies,
        }

    async def _evaluate_longmemeval(
        self,
        answer_result: AnswerResult,
        gold_answers: List[str],
    ) -> float:
        metadata = answer_result.metadata or {}
        task = str(metadata.get("category") or answer_result.category or "")
        question_id = str(answer_result.question_id)
        abstention = "_abs" in question_id or "_abs" in str(metadata.get("qa_pair_id", ""))
        gold = gold_answers[0] if gold_answers else ""
        prompt = self._build_longmemeval_prompt(
            task=task,
            question=answer_result.question,
            answer=gold,
            response=answer_result.answer,
            abstention=abstention,
        )
        label = await self._complete_text(prompt, max_tokens=10)
        return 1.0 if _parse_longmemeval_label(label) else 0.0

    async def _evaluate_infbench(
        self,
        answer_result: AnswerResult,
        gold_answers: List[str],
    ) -> tuple[float, Dict[str, Any]]:
        metadata = answer_result.metadata or {}
        keypoints = metadata.get("keypoints") or []
        summary = answer_result.answer.strip()
        expert_summary = gold_answers[0] if gold_answers else answer_result.golden_answer

        fluency_raw = await self._complete_text(
            FLUENCY_PROMPT_BOOK.format(text=summary),
            max_tokens=64,
        )
        recall_raw = await self._complete_text(
            RECALL_PROMPT_BOOK.format(
                keypoints="\n".join(f"{idx+1}. {kp}" for idx, kp in enumerate(keypoints)),
                summary=summary,
            ),
            max_tokens=256,
        )
        precision_raw = await self._complete_text(
            PRECISION_PROMPT_BOOK.format(
                expert_summary=expert_summary,
                summary=summary,
            ),
            max_tokens=256,
        )

        fluency_json = _extract_json_object(fluency_raw) or {"fluency": 0}
        recall_json = _extract_json_object(recall_raw) or {"recall": 0}
        precision_json = _extract_json_object(precision_raw) or {
            "precision": 0,
            "sentence_count": 1,
        }

        recall = (
            float(recall_json.get("recall", 0)) / len(keypoints)
            if keypoints
            else 0.0
        )
        precision = float(precision_json.get("precision", 0)) / max(
            1, int(precision_json.get("sentence_count", 1))
        )
        fluency = float(fluency_json.get("fluency", 0))
        f1 = (
            fluency * 2 * (recall * precision) / (recall + precision)
            if recall + precision > 0
            else 0.0
        )

        return f1, {
            "metric_name": "F1",
            "fluency": fluency,
            "recall": recall,
            "precision": precision,
        }

    async def _complete_text(self, prompt: str, max_tokens: int) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.judge_max_attempts + 1):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                detail = str(exc)
                if attempt >= self.judge_max_attempts:
                    break
                wait_seconds = min(2 ** (attempt - 1), 30)
                print(
                    f"  ⚠️ Judge LLM call failed "
                    f"(attempt {attempt}/{self.judge_max_attempts}); "
                    f"retrying in {wait_seconds}s: {type(exc).__name__}: {detail[:300]}"
                )
                await asyncio.sleep(wait_seconds)

        assert last_error is not None
        raise last_error

    def _build_longmemeval_prompt(
        self,
        *,
        task: str,
        question: str,
        answer: str,
        response: str,
        abstention: bool,
    ) -> str:
        if abstention:
            return (
                "I will give you an unanswerable question, an explanation, and a response "
                "from a model. Please answer yes if the model correctly identifies the "
                "question as unanswerable. The model could say that the information is "
                "incomplete, or some other information is given but the asked information "
                "is not.\n\n"
                f"Question: {question}\n\n"
                f"Explanation: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Does the model correctly identify the question as unanswerable? "
                "Answer yes or no only."
            )
        if task == "temporal-reasoning":
            suffix = (
                " In addition, do not penalize off-by-one errors for the number of "
                "days. If the question asks for the number of days/weeks/months, etc., "
                "and the model makes off-by-one errors, the model's response is still correct."
            )
        elif task == "knowledge-update":
            suffix = (
                " If the response contains some previous information along with an updated "
                "answer, the response should be considered correct as long as the updated "
                "answer is the required answer."
            )
        elif task == "single-session-preference":
            return (
                "I will give you a question, a rubric for desired personalized response, "
                "and a response from a model. Please answer yes if the response satisfies "
                "the desired response. Otherwise, answer no. The model does not need to "
                "reflect all the points in the rubric. The response is correct as long as "
                "it recalls and utilizes the user's personal information correctly.\n\n"
                f"Question: {question}\n\n"
                f"Rubric: {answer}\n\n"
                f"Model Response: {response}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
        else:
            suffix = (
                " If the response is equivalent to the correct answer or contains all the "
                "intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the "
                "answer, answer no."
            )
        return (
            "I will give you a question, a correct answer, and a response from a model. "
            "Please answer yes if the response contains the correct answer. Otherwise, "
            f"answer no.{suffix}\n\n"
            f"Question: {question}\n\n"
            f"Correct Answer: {answer}\n\n"
            f"Model Response: {response}\n\n"
            "Is the model response correct? Answer yes or no only."
        )

    def _extract_movie_name(self, text: str) -> str:
        text = (text or "").split("/")[-1]
        text = text.replace("_", " ").replace("-", " ").replace(">", " ")
        text = re.sub(r"\([^()]*\)", "", text)
        text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _extract_recommendations(self, text: str) -> List[str]:
        try:
            _, recommendation_text = text.split("1.", maxsplit=1)
        except Exception:
            recommendation_text = text.replace(",", "\n")
        lines = [line.strip() for line in recommendation_text.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            line = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", line)
            line = re.sub(r"\([^()]*\)", "", line)
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                cleaned.append(line)
        return cleaned
