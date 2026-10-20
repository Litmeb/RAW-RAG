"""Evaluate query-aware Mem0 writing on one complete MEME conversation.

The completed query-agnostic run is strictly read-only.  For every MEME
question in the selected conversation, this tool creates an isolated Mem0
store whose writer sees that question but never any gold field.  It then:

1. compares the historical baseline and query-aware stores with an LLM judge
   using MEME's target entities, entity values, and gold answer;
2. runs retrieval and answer generation against each query-aware store; and
3. judges both historical baseline answers and query-aware answers with the
   same MEME evaluator/model for a fair downstream comparison.

Per-question artifacts are written atomically enough for an interrupted run to
resume in the same output directory.  A completed output directory is never
reused.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

EVERCORE_ROOT = Path(__file__).resolve().parents[2]
EVERCORE_SRC = EVERCORE_ROOT / "src"
for import_root in (EVERCORE_ROOT, EVERCORE_SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.src.adapters.mem0_adapter import Mem0Adapter  # noqa: E402
from evaluation.src.core.data_models import AnswerResult  # noqa: E402
from evaluation.src.core.loaders import load_dataset  # noqa: E402
from evaluation.src.core.stages.answer_stage import (  # noqa: E402
    _build_answer_query,
    build_context,
)
from evaluation.src.evaluators.meme_judge import MEMEJudge  # noqa: E402
from evaluation.src.utils.config import load_yaml  # noqa: E402


EVALUATION_ROOT = EVERCORE_ROOT / "evaluation"
DEFAULT_SOURCE_RUN = EVALUATION_ROOT / "results" / "meme-mem0_local-normal"
DEFAULT_OUTPUT = (
    EVALUATION_ROOT
    / "results"
    / "meme-mem0_local-query-aware-add-full-conv0-v1-20260717"
)
DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
COMPLETION_MARKER = "COMPLETED.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--question-workers",
        type=int,
        default=1,
        help="Local Mem0/Qdrant is safest with one isolated store open at a time.",
    )
    parser.add_argument("--judge-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="Optional repeatable selector; default evaluates every question.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the protocol without making API calls.",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    dump = getattr(value, "model_dump", None)
    return _jsonable(dump()) if callable(dump) else repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")


class AuditedMem0Adapter(Mem0Adapter):
    """Mem0 adapter that persists the exact result of every add call."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.add_audit: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    async def _add_messages_to_client(
        self,
        *,
        messages: list[dict[str, Any]],
        source_timestamp: datetime | None,
        user_id: str,
        run_id: str,
        metadata: dict[str, Any],
    ) -> Any:
        response = await super()._add_messages_to_client(
            messages=messages,
            source_timestamp=source_timestamp,
            user_id=user_id,
            run_id=run_id,
            metadata=metadata,
        )
        self.add_audit.append(
            {
                "user_id": user_id,
                "run_id": run_id,
                "source_timestamp": (
                    source_timestamp.isoformat() if source_timestamp else None
                ),
                "metadata": _jsonable(metadata),
                "messages": _jsonable(messages),
                "response": _jsonable(response),
            }
        )
        return response


def _load_inputs(args: argparse.Namespace) -> tuple[Any, list[Any], str]:
    load_dotenv(EVERCORE_ROOT / ".env")
    dataset = load_dataset(
        "meme",
        str(EVALUATION_ROOT / "data" / "meme"),
        max_content_length=16000,
        dataset_format="meme",
    )
    if not 0 <= args.episode_index < len(dataset.conversations):
        raise ValueError(
            f"episode index {args.episode_index} is outside "
            f"[0, {len(dataset.conversations) - 1}]"
        )
    conversation = dataset.conversations[args.episode_index]
    questions = [
        qa
        for qa in dataset.qa_pairs
        if qa.metadata.get("conversation_id") == conversation.conversation_id
    ]
    if args.question_ids:
        selected = set(args.question_ids)
        unknown = selected - {qa.question_id for qa in questions}
        if unknown:
            raise ValueError(f"unknown question ids: {sorted(unknown)}")
        questions = [qa for qa in questions if qa.question_id in selected]
    if not questions:
        raise ValueError("no questions selected")
    prompts = load_yaml(str(EVALUATION_ROOT / "config" / "prompts.yaml"))
    instructions = prompts["add_stage"]["mem0"]["custom_instructions"]
    return conversation, questions, instructions


def _query_aware_instructions(baseline: str, question: str) -> str:
    return (
        baseline.rstrip()
        + "\n\nDownstream-query-aware writing requirement:\n"
        + f"- The future downstream question is: {json.dumps(question, ensure_ascii=False)}\n"
        + "- Prioritize facts that could help answer this question. Preserve exact "
        + "entities, values, negations, dependencies, deletions, and chronological "
        + "changes whenever relevant.\n"
        + "- Do not answer the question, use outside knowledge, or infer facts absent "
        + "from the messages. The question is retrieval intent, not evidence."
    )


def _question_config(
    *, qa: Any, instructions: str, question_dir: Path, top_k: int | None
) -> dict[str, Any]:
    config = load_yaml(str(EVALUATION_ROOT / "config" / "systems" / "mem0_local.yaml"))
    config["dataset_name"] = "meme"
    config["run_name"] = f"query_aware_{_safe_name(qa.question_id)}_20260717"
    config["custom_instructions"] = _query_aware_instructions(
        instructions, qa.question
    )
    config["num_workers"] = 1
    config["add_resume"] = {"enabled": False}
    config["local"] = dict(config.get("local", {}))
    config["local"]["shard_by_conversation"] = False
    config["local"]["telemetry_enabled"] = False
    memory = config["local"].setdefault("memory", {})
    vector_config = memory.setdefault("vector_store", {}).setdefault("config", {})
    vector_config["path"] = str(question_dir / "mem0_qdrant")
    memory["history_db_path"] = str(question_dir / "mem0_history.db")
    if top_k is not None:
        config.setdefault("search", {})["top_k"] = top_k
    return config


def _read_memories(store_path: Path, user_id: str) -> list[dict[str, Any]]:
    client = QdrantClient(path=str(store_path))
    try:
        records: list[dict[str, Any]] = []
        offset = None
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        )
        while True:
            points, offset = client.scroll(
                collection_name="mem0_local",
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(
                {"id": str(point.id), **_jsonable(point.payload or {})}
                for point in points
            )
            if offset is None:
                break
        return sorted(records, key=lambda row: row["id"])
    finally:
        client.close()


def _available_at_question(
    records: Iterable[dict[str, Any]], question_date: Any
) -> list[dict[str, Any]]:
    cutoff = Mem0Adapter._parse_temporal_datetime(question_date)
    if cutoff is None:
        return list(records)
    available = []
    for record in records:
        timestamp = Mem0Adapter._parse_temporal_datetime(record.get("created_at"))
        if timestamp is None or timestamp <= cutoff:
            available.append(record)
    return available


def _answer_result(qa: Any, answer: str, context: str, results: list[dict]) -> AnswerResult:
    return AnswerResult(
        question_id=qa.question_id,
        question=qa.question,
        answer=answer,
        golden_answer=qa.answer,
        category=qa.category,
        conversation_id=qa.metadata.get("conversation_id", ""),
        formatted_context=context,
        search_results=results,
        metadata=qa.metadata,
    )


async def _run_question(
    *,
    qa: Any,
    conversation: Any,
    baseline_instructions: str,
    output_dir: Path,
    top_k: int | None,
) -> dict[str, Any]:
    question_dir = output_dir / "questions" / _safe_name(qa.question_id)
    result_path = question_dir / "result.json"
    if result_path.exists():
        print(f"[resume] {qa.question_id}")
        return json.loads(result_path.read_text(encoding="utf-8"))

    # A missing result means this question did not finish. Its private store can
    # safely be rebuilt without touching any other question or the source run.
    resolved_question_dir = question_dir.resolve()
    expected_parent = (output_dir / "questions").resolve()
    if resolved_question_dir.parent != expected_parent:
        raise ValueError(
            f"refusing to reset question directory outside {expected_parent}: "
            f"{resolved_question_dir}"
        )
    if question_dir.exists():
        shutil.rmtree(question_dir)
    question_dir.mkdir(parents=True)
    config = _question_config(
        qa=qa,
        instructions=baseline_instructions,
        question_dir=question_dir,
        top_k=top_k,
    )
    adapter = AuditedMem0Adapter(config, output_dir=question_dir)
    user_id = adapter._extract_user_id(conversation, speaker="speaker_a")
    try:
        add_result = await adapter.add(
            conversations=[conversation],
            output_dir=question_dir,
            checkpoint_manager=None,
        )
        search_result = await adapter.search(
            query=qa.question,
            conversation_id=conversation.conversation_id,
            index=None,
            conversation=conversation,
            question_date=qa.metadata.get("question_date"),
        )
        context = build_context(search_result, qa)
        answer = await adapter.answer(
            query=_build_answer_query(qa, adapter),
            context=context,
            conversation_id=conversation.conversation_id,
            question_id=qa.question_id,
            prompt_key=qa.metadata.get("prompt_key"),
        )
    finally:
        await adapter.close()

    memories = _read_memories(question_dir / "mem0_qdrant", user_id)
    available = _available_at_question(memories, qa.metadata.get("question_date"))
    row = {
        "question_id": qa.question_id,
        "question": qa.question,
        "gold_answer": qa.answer,
        "target_entities": qa.metadata.get("entity", []),
        "entity_values": qa.metadata.get("entity_values", {}),
        "phase": qa.metadata.get("phase"),
        "task_type": qa.metadata.get("task_type"),
        "question_date": qa.metadata.get("question_date"),
        "run_name": config["run_name"],
        "user_id": user_id,
        "add_result": add_result,
        "add_operation_count": len(adapter.add_audit),
        "memory_count": len(memories),
        "available_memory_count": len(available),
        "available_memories": available,
        "answer_result": _answer_result(
            qa, answer, context, search_result.results
        ),
    }
    _write_json(question_dir / "add_operations.json", adapter.add_audit)
    _write_json(result_path, row)
    print(f"[complete] {qa.question_id}: {len(available)} memories")
    return _jsonable(row)


EVIDENCE_PROMPT = """You are evaluating whether a memory store preserved the correct evidence needed for a MEME benchmark question.

QUESTION DATE: {question_date}
PHASE: {phase}
TASK TYPE: {task_type}
QUESTION: {question}
TARGET ENTITIES: {target_entities}
OFFICIAL ENTITY VALUES: {entity_values}
GOLD ANSWER: {gold_answer}

STORE A (query-agnostic baseline):
{baseline_memories}

STORE B (query-aware writer):
{query_aware_memories}

Judge each store independently. A store passes only if its memories available at the question date contain sufficient, semantically correct evidence to derive the gold answer. Apply these task rules:
- Tr: preserve all historical values in chronological order.
- Cas: preserve the current cascaded value; stale values must not be asserted as current.
- Abs: preserve the upstream change and resulting uncertainty rather than confidently retaining the old value.
- Del: preserve that the information is deleted/unavailable; do not assert the deleted value as current.
- Agg: preserve every target entity/value needed for the multi-hop answer.
- ER: preserve the exact target value semantically.
For a before-phase question, assess the ordinary value at that earlier question date.
Ignore retrieval ranking and writing style. Do not give credit merely because a gold string occurs in stale or contradictory text.

Output ONLY JSON:
{{"baseline_pass": true/false, "query_aware_pass": true/false, "baseline_reason": "one sentence", "query_aware_reason": "one sentence"}}"""


def _memory_text(records: Iterable[dict[str, Any]]) -> str:
    lines = []
    for index, record in enumerate(records, start=1):
        created_at = record.get("created_at", "unknown-time")
        data = record.get("data", "")
        lines.append(f"{index}. [{created_at}] {data}")
    return "\n".join(lines) or "(empty store)"


async def _call_json(client: AsyncOpenAI, model: str, prompt: str) -> dict[str, Any]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def _judge_evidence(
    *,
    rows: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    model: str,
    workers: int,
) -> list[dict[str, Any]]:
    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
    )
    semaphore = asyncio.Semaphore(max(1, workers))

    async def judge(row: dict[str, Any]) -> dict[str, Any]:
        baseline_available = _available_at_question(
            baseline_records, row["question_date"]
        )
        prompt = EVIDENCE_PROMPT.format(
            question_date=row["question_date"],
            phase=row["phase"],
            task_type=row["task_type"],
            question=row["question"],
            target_entities=json.dumps(row["target_entities"], ensure_ascii=False),
            entity_values=json.dumps(row["entity_values"], ensure_ascii=False),
            gold_answer=row["gold_answer"],
            baseline_memories=_memory_text(baseline_available),
            query_aware_memories=_memory_text(row["available_memories"]),
        )
        async with semaphore:
            result = await _call_json(client, model, prompt)
        return {
            "question_id": row["question_id"],
            "phase": row["phase"],
            "task_type": row["task_type"],
            "baseline_available_memory_count": len(baseline_available),
            "query_aware_available_memory_count": len(row["available_memories"]),
            **result,
        }

    return await asyncio.gather(*(judge(row) for row in rows))


def _load_baseline_answers(source_run: Path, question_ids: set[str]) -> list[AnswerResult]:
    rows = json.loads((source_run / "answer_results.json").read_text(encoding="utf-8"))
    selected = [row for row in rows if row["question_id"] in question_ids]
    if len(selected) != len(question_ids):
        found = {row["question_id"] for row in selected}
        raise ValueError(f"baseline answers missing: {sorted(question_ids - found)}")
    return [AnswerResult(**row) for row in selected]


async def _judge_answers(
    answers: list[AnswerResult], model: str, workers: int
) -> dict[str, Any]:
    evaluator = MEMEJudge(
        {
            "llm": {
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "base_url": os.environ["DEEPSEEK_BASE_URL"],
                "model": model,
            },
            "num_runs": 1,
            "max_concurrency": max(1, workers),
        }
    )
    return _jsonable(await evaluator.evaluate(answers))


def _evidence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    baseline = sum(bool(row.get("baseline_pass")) for row in rows)
    aware = sum(bool(row.get("query_aware_pass")) for row in rows)
    transitions = Counter(
        (
            "pass" if row.get("baseline_pass") else "fail",
            "pass" if row.get("query_aware_pass") else "fail",
        )
        for row in rows
    )
    per_task = {}
    for task in sorted({str(row["task_type"]) for row in rows}):
        task_rows = [row for row in rows if row["task_type"] == task]
        per_task[task] = {
            "total": len(task_rows),
            "baseline_pass": sum(bool(row.get("baseline_pass")) for row in task_rows),
            "query_aware_pass": sum(
                bool(row.get("query_aware_pass")) for row in task_rows
            ),
        }
    return {
        "total": total,
        "baseline_pass": baseline,
        "baseline_accuracy": baseline / total if total else 0.0,
        "query_aware_pass": aware,
        "query_aware_accuracy": aware / total if total else 0.0,
        "delta_accuracy": (aware - baseline) / total if total else 0.0,
        "transitions": {f"{a}_to_{b}": count for (a, b), count in transitions.items()},
        "per_task": per_task,
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> None:
    evidence = report["evidence_summary"]
    baseline = report["downstream"]["baseline"]["metadata"]
    aware = report["downstream"]["query_aware"]["metadata"]
    lines = [
        "# MEME query-aware Mem0 writing — conversation0",
        "",
        f"- Judge: `{report['judge_model']}`",
        f"- Questions: {evidence['total']}",
        f"- Baseline source: `{report['source_run']}` (read-only)",
        "",
        "## Evidence preservation",
        "",
        f"- Baseline: {evidence['baseline_pass']}/{evidence['total']} "
        f"({evidence['baseline_accuracy']:.2%})",
        f"- Query-aware: {evidence['query_aware_pass']}/{evidence['total']} "
        f"({evidence['query_aware_accuracy']:.2%})",
        f"- Delta: {evidence['delta_accuracy']:+.2%}",
        f"- Transitions: `{evidence['transitions']}`",
        "",
        "## Downstream answer accuracy (same DeepSeek judge)",
        "",
        f"- Baseline official MEME accuracy: {baseline['official_accuracy']:.2%}",
        f"- Query-aware official MEME accuracy: {aware['official_accuracy']:.2%}",
        f"- Delta: {aware['official_accuracy'] - baseline['official_accuracy']:+.2%}",
        f"- Baseline raw all-question accuracy: {baseline['raw_accuracy']:.2%}",
        f"- Query-aware raw all-question accuracy: {aware['raw_accuracy']:.2%}",
        "",
        "## Per-task official scores",
        "",
        "| Task | Baseline | Query-aware | Delta |",
        "|---|---:|---:|---:|",
    ]
    for task in ("ER", "Agg", "Tr", "Del", "Cas", "Abs"):
        base_score = baseline["task_scores"][task]["accuracy"]
        aware_score = aware["task_scores"][task]["accuracy"]
        lines.append(
            f"| {task} | {base_score:.2%} | {aware_score:.2%} | "
            f"{aware_score - base_score:+.2%} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def _main() -> None:
    args = _parse_args()
    conversation, questions, baseline_instructions = _load_inputs(args)
    source_run = args.source_run.resolve()
    output_dir = args.output_dir.resolve()
    if not (source_run / "mem0_qdrant").exists():
        raise FileNotFoundError(f"baseline Mem0 store not found: {source_run}")
    if not (source_run / "answer_results.json").exists():
        raise FileNotFoundError(f"baseline answers not found: {source_run}")
    if (output_dir / COMPLETION_MARKER).exists():
        raise FileExistsError(f"refusing to reuse completed output: {output_dir}")

    protocol = {
        "conversation_id": conversation.conversation_id,
        "message_count": len(conversation.messages),
        "question_count": len(questions),
        "question_ids": [qa.question_id for qa in questions],
        "task_counts": dict(Counter(qa.metadata.get("task_type") for qa in questions)),
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "judge_model": args.judge_model,
        "gold_hidden_from_writers": True,
    }
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "protocol.json", protocol)
    semaphore = asyncio.Semaphore(max(1, args.question_workers))

    async def run_limited(qa: Any) -> dict[str, Any]:
        async with semaphore:
            return await _run_question(
                qa=qa,
                conversation=conversation,
                baseline_instructions=baseline_instructions,
                output_dir=output_dir,
                top_k=args.top_k,
            )

    question_rows = await asyncio.gather(*(run_limited(qa) for qa in questions))
    baseline_user_id = f"normal__{conversation.conversation_id}_User"
    baseline_records = _read_memories(source_run / "mem0_qdrant", baseline_user_id)
    if not baseline_records:
        raise ValueError(f"no baseline memories found for {baseline_user_id}")

    evidence_rows = await _judge_evidence(
        rows=question_rows,
        baseline_records=baseline_records,
        model=args.judge_model,
        workers=args.judge_workers,
    )
    question_ids = {qa.question_id for qa in questions}
    baseline_answers = _load_baseline_answers(source_run, question_ids)
    aware_answers = [
        AnswerResult(**row["answer_result"])
        for row in question_rows
    ]
    baseline_evaluation, aware_evaluation = await asyncio.gather(
        _judge_answers(baseline_answers, args.judge_model, args.judge_workers),
        _judge_answers(aware_answers, args.judge_model, args.judge_workers),
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **protocol,
        "baseline_memory_count": len(baseline_records),
        "evidence_summary": _evidence_summary(evidence_rows),
        "evidence_results": evidence_rows,
        "downstream": {
            "baseline": baseline_evaluation,
            "query_aware": aware_evaluation,
        },
        "question_results": question_rows,
    }
    _write_json(output_dir / "evaluation.json", report)
    _write_report(report, output_dir)
    _write_json(
        output_dir / COMPLETION_MARKER,
        {"created_at": report["created_at"], "evaluation": "evaluation.json"},
    )
    print(json.dumps(report["evidence_summary"], indent=2, ensure_ascii=False))
    print(f"Report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    asyncio.run(_main())
