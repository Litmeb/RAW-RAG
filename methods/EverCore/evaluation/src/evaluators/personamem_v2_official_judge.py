"""
PersonaMem-v2 official-style open-ended evaluator.

This mirrors the PersonaMem-v2 narrow judge: evaluate the generated response
against the row's ground-truth preference, using a separate prompt for
"do not" / avoidance preferences, and aggregate 0.0-1.0 judge scores.
"""

import ast
import asyncio
import json
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Any, Dict, List

from openai import AsyncOpenAI

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator
from evaluation.src.utils.prompts import format_prompt


@register_evaluator("personamem_v2_official_judge")
class PersonaMemV2OfficialJudge(BaseEvaluator):
    """Official-style narrow judge for PersonaMem-v2 open-ended responses."""

    PROMPT_KEY = "personamem_v2_official_judge"

    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        )
        self.model = llm_config.get("model", "gpt-5-chat")
        self.temperature = float(llm_config.get("temperature", 0))
        self.max_tokens = int(llm_config.get("max_tokens", 2048))
        self.num_runs = int(config.get("num_runs", 3))
        self.max_concurrency = int(config.get("max_concurrency", 10))
        self.store_judge_responses = bool(config.get("store_judge_responses", True))
        self.score_threshold = float(config.get("score_threshold", 0.5))

    async def evaluate(self, answer_results: List[AnswerResult]) -> EvaluationResult:
        print(f"\n{'='*60}")
        print(
            "Evaluation: PersonaMem-v2 Official Judge "
            f"(model={self.model}, runs={self.num_runs})"
        )
        print(f"{'='*60}")

        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def evaluate_single(answer_result: AnswerResult) -> Dict[str, Any]:
            async with semaphore:
                return await self._evaluate_single_answer(answer_result)

        detailed_results = await asyncio.gather(
            *(evaluate_single(result) for result in answer_results)
        )

        item_scores = [float(item["score"]) for item in detailed_results]
        overall = mean(item_scores) if item_scores else 0.0

        run_scores = []
        for run_idx in range(self.num_runs):
            values = [
                float(item["judge_scores"][run_idx])
                for item in detailed_results
                if run_idx < len(item["judge_scores"])
            ]
            if values:
                run_scores.append(mean(values))

        category_stats = defaultdict(lambda: {"scores": [], "run_scores": [[] for _ in range(self.num_runs)]})
        for item in detailed_results:
            category = item.get("category")
            if category is None:
                continue
            category_stats[str(category)]["scores"].append(float(item["score"]))
            for run_idx, score in enumerate(item["judge_scores"]):
                if run_idx < self.num_runs:
                    category_stats[str(category)]["run_scores"][run_idx].append(float(score))

        category_accuracies = {}
        for category, stats in category_stats.items():
            scores = stats["scores"]
            individual_runs = [
                mean(values) for values in stats["run_scores"] if values
            ]
            category_accuracies[category] = {
                "mean": mean(scores) if scores else 0.0,
                "std": pstdev(scores) if len(scores) > 1 else 0.0,
                "individual_runs": individual_runs,
                "total": len(scores),
            }

        grouped_results = self._group_by_conversation(detailed_results)

        print("\n✅ Evaluation complete:")
        print(f"   - Total questions: {len(answer_results)}")
        print(f"   - Mean score: {overall:.4f}")
        print(f"   - Std deviation: {pstdev(item_scores) if len(item_scores) > 1 else 0.0:.4f}")
        print(f"   - Run scores: {[f'{score:.4f}' for score in run_scores]}")

        if category_accuracies:
            print("\n📊 Category statistics:")
            for category, stats in sorted(category_accuracies.items()):
                print(
                    f"   Category {category}: {stats['mean']:.4f} ± "
                    f"{stats['std']:.4f} (n={stats['total']})"
                )

        return EvaluationResult(
            total_questions=len(answer_results),
            correct=int(round(overall * len(answer_results))),
            accuracy=overall,
            detailed_results=grouped_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs,
                "mean_score": overall,
                "std_score": pstdev(item_scores) if len(item_scores) > 1 else 0.0,
                "run_scores": run_scores,
                "category_accuracies": category_accuracies,
                "score_threshold": self.score_threshold,
                "judge_style": "personamem_v2_official_narrow",
            },
        )

    async def _evaluate_single_answer(self, answer_result: AnswerResult) -> Dict[str, Any]:
        metadata = answer_result.metadata or {}
        user_query = self._extract_user_query(metadata.get("raw_user_query"), answer_result.question)
        preference = str(metadata.get("preference", "") or "")
        is_negative = preference.lower().startswith("do not")
        prompt_name = "negative_prompt" if is_negative else "positive_prompt"
        prompt = format_prompt(
            self.PROMPT_KEY,
            prompt_name,
            user_query=user_query,
            preference=preference,
            model_response=answer_result.answer or "",
        )

        judge_scores: List[float] = []
        judge_responses: List[str] = []

        for _ in range(max(1, self.num_runs)):
            response = await self._query_judge(prompt)
            score = self._extract_score(response)
            judge_scores.append(score)
            if self.store_judge_responses:
                judge_responses.append(response)

        score = mean(judge_scores) if judge_scores else 0.0
        result = {
            "question_id": answer_result.question_id,
            "conversation_id": answer_result.conversation_id,
            "question": answer_result.question,
            "user_query": user_query,
            "preference": preference,
            "preference_is_negative": is_negative,
            "golden_answer": answer_result.golden_answer,
            "generated_answer": answer_result.answer,
            "judge_scores": judge_scores,
            "score": score,
            "passed_threshold": score >= self.score_threshold,
            "category": answer_result.category,
        }
        if self.store_judge_responses:
            result["judge_responses"] = judge_responses
        return result

    async def _query_judge(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _extract_user_query(raw_user_query: Any, fallback: str) -> str:
        if isinstance(raw_user_query, dict):
            content = raw_user_query.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        if isinstance(raw_user_query, str) and raw_user_query.strip():
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(raw_user_query)
                    if isinstance(parsed, dict):
                        content = parsed.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                except Exception:
                    pass
            return raw_user_query.strip()

        return str(fallback or "").strip()

    @staticmethod
    def _extract_score(response: str) -> float:
        if not response:
            return 0.0

        boxed_patterns = [
            r"\\boxed\{([0-9]*\.?[0-9]+)\}",
            r"\$\\boxed\{([0-9]*\.?[0-9]+)\}\$",
            r"\\boxed\s*\{([0-9]*\.?[0-9]+)\}",
        ]
        for pattern in boxed_patterns:
            match = re.search(pattern, response)
            if match:
                return PersonaMemV2OfficialJudge._clamp_score(match.group(1))

        fallback_patterns = [
            r"score[:\s]+([0-9]*\.?[0-9]+)",
            r"rating[:\s]+([0-9]*\.?[0-9]+)",
            r"([0-9]*\.[0-9]+)\s*/\s*1\.?0?",
        ]
        for pattern in fallback_patterns:
            match = re.search(pattern, response, flags=re.IGNORECASE)
            if match:
                value = PersonaMemV2OfficialJudge._clamp_score(match.group(1))
                if 0.0 <= value <= 1.0:
                    return value

        print(
            "  ⚠️ PersonaMem-v2 judge: could not extract score from response: "
            f"{response[:120]}..."
        )
        return 0.0

    @staticmethod
    def _clamp_score(raw_value: str) -> float:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _group_by_conversation(detailed_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for result in detailed_results:
            group_key = result.get("conversation_id") or "default_group"
            grouped[group_key].append(result)
        return dict(grouped)
