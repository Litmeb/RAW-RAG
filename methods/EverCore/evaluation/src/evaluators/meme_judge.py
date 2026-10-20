"""
MEME evaluator - reproduce the benchmark's task-specific judging flow.
"""

import asyncio
import json
import re
from collections import defaultdict
from typing import Any, Dict, List

from openai import AsyncOpenAI

from evaluation.src.core.data_models import AnswerResult, EvaluationResult
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.evaluators.registry import register_evaluator


MEME_TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")
TRIVIAL_PASS_TASKS = {"Cas", "Abs", "Del"}


def summarize_meme_results(detailed_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute the official MEME six-task scores from judged answer rows.

    ER/Agg/Tr use after-phase accuracy.  Cas/Abs/Del use after-phase accuracy
    with trivial-pass filtering: an after pass only receives credit when the
    matching before-phase answer also passed.
    """
    grouped_results = defaultdict(list)
    for items in detailed_results.values():
        for item in items:
            grouped_results[item["conversation_id"]].append(item)

    task_counts = {
        task: {"pass": 0.0, "total": 0, "accuracy": 0.0}
        for task in MEME_TASKS
    }
    trivial_analysis = {
        task: {
            "total": 0,
            "real_pass": 0.0,
            "trivial_pass": 0.0,
            "knew_but_failed": 0,
            "never_knew": 0,
        }
        for task in TRIVIAL_PASS_TASKS
    }

    for items in grouped_results.values():
        before_map = {
            item.get("entity_key", ""): bool(item.get("u_pass", False))
            for item in items
            if item.get("phase") == "before"
        }
        for item in items:
            if item.get("phase") != "after":
                continue
            task = str(item.get("task_type", "")).split(" (")[0]
            if task not in MEME_TASKS:
                continue

            score = 1.0 if bool(item.get("u_pass", False)) else 0.0
            filtered_score = score
            if task in TRIVIAL_PASS_TASKS:
                before_score = before_map.get(item.get("entity_key", ""), False)
                analysis = trivial_analysis[task]
                analysis["total"] += 1
                if score > 0 and before_score > 0:
                    item["pass_type"] = "real"
                    analysis["real_pass"] += score
                elif score > 0:
                    item["pass_type"] = "trivial"
                    analysis["trivial_pass"] += score
                    filtered_score = 0.0
                elif before_score > 0:
                    item["pass_type"] = "knew_but_failed"
                    analysis["knew_but_failed"] += 1
                else:
                    item["pass_type"] = "never_knew"
                    analysis["never_knew"] += 1
            else:
                item["pass_type"] = "raw"
            item["filtered_u_pass"] = filtered_score

            task_counts[task]["total"] += 1
            task_counts[task]["pass"] += filtered_score

    for task in MEME_TASKS:
        total = task_counts[task]["total"]
        task_counts[task]["accuracy"] = (
            task_counts[task]["pass"] / total if total else 0.0
        )

    overall = sum(task_counts[task]["accuracy"] for task in MEME_TASKS) / len(MEME_TASKS)
    return {
        "task_scores": task_counts,
        "overall_accuracy": overall,
        "trivial_analysis": trivial_analysis,
    }


@register_evaluator("meme_judge")
class MEMEJudge(BaseEvaluator):
    """Task-specific evaluator aligned with MEME's public judge pipeline."""

    BEFORE_COMMON_PROMPT = """You are a judge evaluating whether an AI agent's answer is semantically correct.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the correct information matching the gold answer?

Rules:
- Focus on semantic equivalence, not exact wording
- If the gold value is present in the agent's answer, it is correct regardless of extra information
- "40 minutes" = "40 min"
- If agent says "I don't know" and gold is a specific value, it is incorrect

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    CASCADE_D_PROMPT = """You are a judge evaluating whether an AI agent correctly reports a CASCADE-DEPENDENT value after an upstream change.

QUESTION: {question}
GOLD (new cascaded value): {gold}
AGENT: {agent_answer}

Rules:
- Agent states the new value as the sole definitive answer -> YES
- Agent lists both old and new values together -> NO
- Agent states only the old value -> NO
- Agent hedges or uses future tense instead of confirming the current value -> NO
- "I don't know" -> NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    CASCADE_U_PROMPT = """You are a judge evaluating whether an AI agent appropriately handles uncertainty after an upstream entity changed.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Rules:
- Agent expresses uncertainty -> YES
- Agent acknowledges the upstream change and questions validity -> YES
- Agent confidently states the old value as definitive current fact -> NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    DELETION_PROMPT = """You are a judge evaluating whether an AI agent correctly handles deleted information.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Rules:
- Agent says the information is unavailable/unknown/removed without revealing the deleted value -> YES
- Agent reveals the deleted value in any way -> NO
- Agent returns the deleted value as if it still exists -> NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    MULTIHOP_PROMPT = """You are a judge evaluating whether an AI agent's answer contains specific values.

QUESTION: {question}
TARGET VALUES: {gold_values}
AGENT: {agent_answer}

For each target value, check whether it appears exactly or via a clear paraphrase.

Output ONLY JSON: {{"results": [{{"value": "...", "present": true/false}}], "reason": "one sentence"}}"""

    MULTIPLE_UPDATE_PROMPT = """You are a judge evaluating whether an AI agent correctly recalls a sequence of historical values for an entity that changed multiple times.

QUESTION: {question}
GOLD (ordered history values, comma-separated): {gold}
AGENT: {agent_answer}

Rules:
- The answer must contain all gold values in the exact chronological order
- Extra surrounding text is fine
- Missing any value or wrong order -> NO
- "I don't know" -> NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

    JUDGE_PROMPTS = {
        "Tr": MULTIPLE_UPDATE_PROMPT,
        "Cas": CASCADE_D_PROMPT,
        "Abs": CASCADE_U_PROMPT,
        "Del": DELETION_PROMPT,
        "Agg": MULTIHOP_PROMPT,
    }

    def __init__(self, config: dict):
        super().__init__(config)
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        )
        self.model = llm_config.get("model", "gpt-4o")
        self.num_runs = int(config.get("num_runs", 3))
        self.max_concurrency = int(config.get("max_concurrency", 10))

    async def evaluate(self, answer_results: List[AnswerResult]) -> EvaluationResult:
        print(f"\n{'='*60}")
        print(f"Evaluation: MEME Judge (model={self.model}, runs={self.num_runs})")
        print(f"{'='*60}")

        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def evaluate_single(answer_result: AnswerResult) -> Dict[str, Any]:
            async with semaphore:
                return await self._evaluate_single_answer(answer_result)

        judged_results = await asyncio.gather(
            *(evaluate_single(result) for result in answer_results)
        )

        detailed_results: Dict[str, List[Dict[str, Any]]] = {}
        before_pass = before_total = 0
        after_pass = after_total = 0
        filtered_after_pass = filtered_after_total = 0
        per_phase_task_counts = defaultdict(lambda: {"pass": 0, "total": 0})
        per_phase_task_filtered_counts = defaultdict(lambda: {"pass": 0, "total": 0})
        trivial_analysis_totals = {
            task: {
                "total": 0,
                "real_pass": 0,
                "trivial_pass": 0,
                "knew_but_failed": 0,
                "never_knew": 0,
            }
            for task in ("Cas", "Abs", "Del")
        }

        grouped_results = defaultdict(list)
        for result in judged_results:
            grouped_results[result["conversation_id"]].append(result)

        for conversation_id, items in grouped_results.items():
            items.sort(key=lambda item: item["question_id"])
            before_map = {
                item["entity_key"]: bool(item.get("u_pass", False))
                for item in items
                if item["phase"] == "before"
            }

            for item in items:
                task_type = item["task_type"]
                phase = item["phase"]
                score = 1.0 if bool(item.get("u_pass", False)) else 0.0
                task_key = f"{phase}:{task_type}"
                per_phase_task_counts[task_key]["total"] += 1
                per_phase_task_counts[task_key]["pass"] += score

                if phase == "before":
                    before_total += 1
                    before_pass += score
                    continue

                after_total += 1
                after_pass += score

                filtered_pass = score
                if task_type in TRIVIAL_PASS_TASKS:
                    trivial_analysis = trivial_analysis_totals[task_type]
                    trivial_analysis["total"] += 1
                    before_score = before_map.get(item["entity_key"], False)
                    if score > 0 and before_score > 0:
                        item["pass_type"] = "real"
                        trivial_analysis["real_pass"] += score
                    elif score > 0 and before_score <= 0:
                        item["pass_type"] = "trivial"
                        trivial_analysis["trivial_pass"] += score
                        filtered_pass = 0.0
                    elif score <= 0 and before_score > 0:
                        item["pass_type"] = "knew_but_failed"
                        trivial_analysis["knew_but_failed"] += 1
                    else:
                        item["pass_type"] = "never_knew"
                        trivial_analysis["never_knew"] += 1
                else:
                    item["pass_type"] = "raw"

                item["filtered_u_pass"] = filtered_pass
                filtered_after_total += 1
                filtered_task_key = f"{phase}:{task_type}"
                per_phase_task_filtered_counts[filtered_task_key]["total"] += 1
                filtered_after_pass += filtered_pass
                per_phase_task_filtered_counts[filtered_task_key]["pass"] += filtered_pass

            detailed_results[conversation_id] = items

        total_questions = len(answer_results)
        official = summarize_meme_results(detailed_results)
        raw_overall_pass = before_pass + after_pass
        filtered_overall_pass = before_pass + filtered_after_pass
        raw_accuracy = raw_overall_pass / total_questions if total_questions else 0.0
        filtered_accuracy = (
            filtered_overall_pass / total_questions if total_questions else 0.0
        )
        task_scores = official["task_scores"]
        official_accuracy = official["overall_accuracy"]

        print("\n✅ Evaluation complete:")
        print(
            f"   - Before: {before_pass}/{before_total} "
            f"({(before_pass / before_total * 100) if before_total else 0:.2f}%)"
        )
        print(
            f"   - After (raw): {after_pass}/{after_total} "
            f"({(after_pass / after_total * 100) if after_total else 0:.2f}%)"
        )
        print(
            f"   - After (trivial-pass filtered): {filtered_after_pass}/{filtered_after_total} "
            f"({(filtered_after_pass / filtered_after_total * 100) if filtered_after_total else 0:.2f}%)"
        )

        return EvaluationResult(
            total_questions=total_questions,
            correct=filtered_overall_pass,
            accuracy=official_accuracy,
            detailed_results=detailed_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs,
                "raw_accuracy": raw_accuracy,
                "filtered_accuracy": filtered_accuracy,
                "official_accuracy": official_accuracy,
                "task_scores": task_scores,
                "before_pass": before_pass,
                "before_total": before_total,
                "before_accuracy": before_pass / before_total if before_total else 0.0,
                "after_pass": after_pass,
                "after_total": after_total,
                "after_accuracy": after_pass / after_total if after_total else 0.0,
                "after_filtered_pass": filtered_after_pass,
                "after_filtered_total": filtered_after_total,
                "after_filtered_accuracy": (
                    filtered_after_pass / filtered_after_total
                    if filtered_after_total
                    else 0.0
                ),
                "per_phase_task_counts": dict(per_phase_task_counts),
                "per_phase_task_filtered_counts": dict(per_phase_task_filtered_counts),
                "trivial_analysis": official["trivial_analysis"],
            },
        )

    async def _evaluate_single_answer(
        self, answer_result: AnswerResult
    ) -> Dict[str, Any]:
        metadata = answer_result.metadata or {}
        task_type = str(metadata.get("task_type", ""))
        phase = str(metadata.get("phase", "after"))
        entity_values = metadata.get("entity_values", {}) or {}
        entity = metadata.get("entity")
        if isinstance(entity, list):
            entity = entity[0] if entity else ""

        if task_type == "Agg":
            judge_fn = self._judge_multi
            judge_kwargs = {
                "question": answer_result.question,
                "entity_values": entity_values,
                "agent_answer": answer_result.answer,
            }
        else:
            gold_value = self._extract_gold_value(entity_values, answer_result.golden_answer)
            judge_fn = self._judge_single
            judge_kwargs = {
                "question": answer_result.question,
                "gold_value": gold_value,
                "agent_answer": answer_result.answer,
                "task_type": task_type,
                "phase": phase,
            }

        judgments = [await judge_fn(**judge_kwargs) for _ in range(self.num_runs)]
        merged_judgment = self._merge_judgments(judgments)

        return {
            "question_id": answer_result.question_id,
            "conversation_id": answer_result.conversation_id,
            "question": answer_result.question,
            "golden_answer": answer_result.golden_answer,
            "generated_answer": answer_result.answer,
            "category": answer_result.category,
            "phase": phase,
            "task_type": task_type,
            "entity": entity,
            "entity_key": metadata.get("entity_key") or entity or "",
            "entity_values": entity_values,
            "hop": metadata.get("hop"),
            **merged_judgment,
        }

    def _extract_gold_value(
        self, entity_values: Dict[str, Any], fallback: str
    ) -> Any:
        if entity_values:
            return next(iter(entity_values.values()))
        return fallback

    async def _judge_single(
        self,
        question: str,
        gold_value: Any,
        agent_answer: str,
        task_type: str,
        phase: str,
    ) -> Dict[str, Any]:
        task_base = task_type.split(" (")[0]

        if task_base == "ER":
            gold_norm = " ".join(str(gold_value).lower().split())
            agent_norm = " ".join(str(agent_answer).lower().split())
            matched = gold_norm in agent_norm
            return {
                "u_pass": matched,
                "u_reason": (
                    "Gold value found in answer (substring match)"
                    if matched
                    else "Gold value not found in answer (substring match)"
                ),
            }

        if task_base == "Tr":
            history = gold_value if isinstance(gold_value, list) else [gold_value]
            prompt = self.MULTIPLE_UPDATE_PROMPT.format(
                question=question,
                gold=", ".join(str(value) for value in history),
                agent_answer=agent_answer,
            )
            result = await self._call_json(prompt)
            agent_lower = str(agent_answer).lower()
            positions = [agent_lower.find(str(value).lower()) for value in history]
            partial_pass = 0
            last_pos = -1
            for pos in positions:
                if pos > last_pos:
                    partial_pass += 1
                    last_pos = pos
            return {
                "u_pass": bool(result.get("correct", False)),
                "u_reason": result.get("reason", ""),
                "u_pass_count": partial_pass,
                "u_pass_total": len(history),
            }

        template = (
            self.BEFORE_COMMON_PROMPT
            if phase == "before" and task_base != "Agg"
            else self.JUDGE_PROMPTS[task_base]
        )
        result = await self._call_json(
            template.format(
                question=question,
                gold=gold_value,
                agent_answer=agent_answer,
            )
        )
        return {
            "u_pass": bool(result.get("correct", False)),
            "u_reason": result.get("reason", ""),
        }

    async def _judge_multi(
        self,
        question: str,
        entity_values: Dict[str, Any],
        agent_answer: str,
    ) -> Dict[str, Any]:
        entity_order = list(entity_values.keys())
        values_list = [entity_values[entity] for entity in entity_order]
        result = await self._call_json(
            self.MULTIHOP_PROMPT.format(
                question=question,
                gold_values=", ".join(f'"{value}"' for value in values_list),
                agent_answer=agent_answer,
            )
        )

        per_entity = {}
        results_list = result.get("results", [])
        for idx, entity in enumerate(entity_order):
            per_entity[entity] = (
                bool(results_list[idx].get("present", False))
                if idx < len(results_list)
                else False
            )

        pass_count = sum(1 for passed in per_entity.values() if passed)
        return {
            "u_pass": pass_count == len(per_entity),
            "u_pass_per_entity": per_entity,
            "u_pass_count": pass_count,
            "u_pass_total": len(per_entity),
            "u_reason": result.get("reason", ""),
        }

    async def _call_json(self, prompt: str) -> Dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            cleaned = content.strip()
            code_block_match = re.search(
                r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL
            )
            if code_block_match:
                return json.loads(code_block_match.group(1).strip())
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            raise

    def _merge_judgments(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not judgments:
            return {
                "u_pass": False,
                "u_score": 0.0,
                "judge_runs": [],
                "u_reason": "",
            }

        total_runs = len(judgments)
        score = sum(1.0 if bool(judgment.get("u_pass")) else 0.0 for judgment in judgments) / total_runs
        merged: Dict[str, Any] = {
            "u_pass": score >= 0.5,
            "u_score": score,
            "judge_runs": judgments,
            "u_reason": next(
                (str(judgment.get("u_reason", "")) for judgment in judgments if judgment.get("u_reason")),
                "",
            ),
        }

        if any("u_pass_count" in judgment for judgment in judgments):
            merged["u_pass_count"] = sum(
                float(judgment.get("u_pass_count", 0)) for judgment in judgments
            ) / total_runs
        if any("u_pass_total" in judgment for judgment in judgments):
            merged["u_pass_total"] = max(
                int(judgment.get("u_pass_total", 0)) for judgment in judgments
            )
        if any("u_pass_per_entity" in judgment for judgment in judgments):
            entity_keys = {
                entity
                for judgment in judgments
                for entity in judgment.get("u_pass_per_entity", {}).keys()
            }
            merged["u_pass_per_entity"] = {
                entity: (
                    sum(
                        1.0
                        if bool(judgment.get("u_pass_per_entity", {}).get(entity, False))
                        else 0.0
                        for judgment in judgments
                    )
                    / total_runs
                )
                for entity in sorted(entity_keys)
            }

        return merged
