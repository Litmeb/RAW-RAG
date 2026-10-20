"""Official LongMemEval judging with the framework's existing result schema."""

from evaluation.src.evaluators.llm_judge import LLMJudge
from evaluation.src.evaluators.longmemeval_official_prompt import get_anscheck_prompt
from evaluation.src.evaluators.registry import register_evaluator


@register_evaluator("longmemeval_official_judge")
class LongMemEvalOfficialJudge(LLMJudge):
    """One yes/no judgment per answer, with upstream task/abstention rubrics."""

    def __init__(self, config: dict):
        config = {**config, "num_runs": 1, "prompt_key": "longmemeval_official"}
        config["llm"] = {"model": "gpt-4o-2024-08-06", **config.get("llm", {})}
        super().__init__(config)
        self.model = {
            "gpt-4o": "gpt-4o-2024-08-06",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        }.get(self.model, self.model)

    async def _evaluate_single_answer(self, answer_result):
        abstention = "_abs" in answer_result.question_id
        prompt = get_anscheck_prompt(
            answer_result.category,
            answer_result.question,
            answer_result.golden_answer,
            answer_result.answer,
            abstention=abstention,
        )
        # Let API failures propagate: an incomplete evaluation must not silently
        # turn service errors into incorrect answers. The SDK retries API errors.
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=1,
            temperature=0,
            max_tokens=10,
        )
        content = response.choices[0].message.content.strip()
        label = "yes" in content.lower()
        return {
            "question_id": answer_result.question_id,
            "question": answer_result.question,
            "golden_answer": answer_result.golden_answer,
            "generated_answer": answer_result.answer,
            "category": answer_result.category,
            "abstention": abstention,
            "llm_judgments": {"judgment_1": label},
            "judge_response": content,
        }

    async def evaluate(self, answer_results):
        result = await super().evaluate(answer_results)
        # Count directly to avoid floating-point truncation in the shared helper.
        result.correct = sum(
            item["llm_judgments"]["judgment_1"]
            for group in result.detailed_results.values()
            for item in group
        )
        result.metadata.update(
            judge_mode="official", temperature=0, max_tokens=10,
            prompt_source="https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py",
        )
        return result
