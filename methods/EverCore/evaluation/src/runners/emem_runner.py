"""Runner integration for the external EMem project.

This runner intentionally bypasses the four-stage adapter pipeline. EMem owns
indexing, retrieval, and answering; EverCore owns dataset loading, filtering,
evaluation, and artifact layout.
"""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import local
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from rich.console import Console
from tqdm import tqdm

from evaluation.src.core.data_models import AnswerResult, Conversation as EvalConversation
from evaluation.src.core.data_models import Dataset, EvaluationResult, Message, QAPair, SearchResult
from evaluation.src.datasets.memoryagentbench import format_memoryagentbench_query
from evaluation.src.evaluators.base import BaseEvaluator
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.saver import ResultSaver
from memory_layer.llm.llm_provider import LLMProvider


class EMemIncrementalIngestor:
    """Keep one EMem graph live and append one unique session per message."""

    def __init__(self, system_config: Dict[str, Any], output_dir: Path):
        self.system_config = system_config
        self.output_dir = Path(output_dir)
        _configure_emem_environment(system_config)
        self.symbols = _load_emem_symbols(system_config)
        self.model = None
        self.conversation_id: Optional[str] = None
        self.date_format_type = str(
            (system_config.get("emem", {}) or {}).get("date_format_type") or "locomo"
        )

    async def start_incremental_conversation(self, conversation_id: str) -> None:
        await self.finalize_incremental_conversation(self.conversation_id)
        save_dir = (
            self.output_dir
            / "emem_outputs"
            / f"sample_{_safe_path_part(conversation_id)}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        config = _build_base_config(
            self.system_config, "beam_10m", str(save_dir), self.symbols
        )
        self.model = await asyncio.to_thread(
            self.symbols["EMem"],
            save_dir=str(save_dir),
            llm_model_name=self.system_config.get("llm", {}).get(
                "model", "gpt-4o-mini"
            ),
            llm_base_url=self.system_config.get("llm", {}).get("base_url"),
            embedding_model_name=_resolve_emem_embedding_model(self.system_config),
            embedding_base_url=self.system_config.get("embedding", {}).get("base_url"),
            global_config=config,
        )
        self.conversation_id = conversation_id

    async def add_incremental_message(
        self,
        conversation: EvalConversation,
        *,
        sequence_index: int,
        **_kwargs,
    ) -> Dict[str, Any]:
        if self.model is None or self.conversation_id != conversation.conversation_id:
            raise RuntimeError("EMem incremental conversation was not initialized")
        if len(conversation.messages) != 1:
            raise ValueError("EMem incremental ingestion requires one message")
        message = conversation.messages[0]
        message.metadata = {
            **message.metadata,
            "session": f"session_{sequence_index}",
            "dia_id": f"D{sequence_index}:0",
        }
        sample = _build_emem_sample(
            conversation,
            [],
            self.symbols,
            date_format_type=self.date_format_type,
        )
        await asyncio.to_thread(self.model.index_conversation, sample)
        return {
            "new_sessions": 1,
            "graph_nodes": int(self.model.graph.vcount()),
            "graph_edges": int(self.model.graph.ecount()),
        }

    async def finalize_incremental_conversation(
        self, conversation_id: Optional[str], **_kwargs
    ) -> None:
        if self.model is None:
            return
        # index_conversation persists graph/content stores after each session.
        # Dropping the model here releases per-conversation embedding/graph state.
        self.model = None
        self.conversation_id = None

    async def close(self) -> None:
        await self.finalize_incremental_conversation(self.conversation_id)


async def run_emem_evaluation(
    dataset: Dataset,
    dataset_config: Dict[str, Any],
    system_config: Dict[str, Any],
    evaluator: BaseEvaluator,
    output_dir: Path,
    *,
    smoke_test: bool = False,
    smoke_messages: int = 10,
    smoke_questions: int = 3,
    from_conv: int = 0,
    to_conv: Optional[int] = None,
    locomo_samples: Optional[List[int]] = None,
    filter_categories: Optional[List[Any]] = None,
    stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run EMem and save EverCore-compatible artifacts."""
    console = Console()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_stages = _normalize_emem_stages(stages)

    dataset = _prepare_dataset_scope(
        dataset=dataset,
        smoke_test=smoke_test,
        smoke_messages=smoke_messages,
        smoke_questions=smoke_questions,
        from_conv=from_conv,
        to_conv=to_conv,
        locomo_samples=locomo_samples,
        filter_categories=filter_categories or [],
        console=console,
    )
    dataset, system_config = _prepare_memoryagentbench_longmemeval_for_emem(
        dataset=dataset,
        system_config=system_config,
        console=console,
    )

    if not dataset.conversations:
        raise ValueError("No conversations selected for EMem runner.")

    console.print(f"\n{'=' * 60}", style="bold cyan")
    console.print("EMem Runner", style="bold cyan")
    console.print(f"{'=' * 60}", style="bold cyan")
    console.print(f"Dataset: {dataset.dataset_name}")
    console.print(f"System: {system_config.get('name', 'emem')}")
    console.print(f"Variant: {_variant_name(system_config)}")
    console.print(f"Output: {output_dir}")
    console.print(f"Conversations: {len(dataset.conversations)}")
    console.print(f"Questions: {len(dataset.qa_pairs)}\n")

    start_time = time.time()

    if _should_run_answer_only_from_saved_search(requested_stages):
        console.print("[cyan]EMem Answer-only mode: loading search_results.json[/cyan]")
        answer_results, search_results = await asyncio.to_thread(
            _run_emem_answer_only_from_saved_search,
            dataset=dataset,
            dataset_config=dataset_config,
            system_config=system_config,
            output_dir=output_dir,
            console=console,
        )

        saver = ResultSaver(output_dir)
        saver.save_json([_answer_result_to_dict(result) for result in answer_results], "answer_results.json")
        console.print(f"[green]Saved answer_results.json ({len(answer_results)} items)[/green]")

        eval_result = None
        if "evaluate" in requested_stages:
            eval_result = await evaluator.evaluate(answer_results)
            saver.save_json(_eval_result_to_dict(eval_result), "eval_results.json")
            _write_report(output_dir, system_config, eval_result, time.time() - start_time)

        return {
            "search_results": search_results,
            "answer_results": answer_results,
            "eval_result": eval_result,
            "output_dir": str(output_dir),
        }

    emem_symbols = _load_emem_symbols(system_config)
    _configure_emem_environment(system_config)

    if requested_stages and requested_stages <= {"add", "search"}:
        search_results, stage_timings = await asyncio.to_thread(
            _run_emem_add_search,
            dataset=dataset,
            system_config=system_config,
            output_dir=output_dir,
            emem_symbols=emem_symbols,
            console=console,
            run_search="search" in requested_stages,
        )
        saver = ResultSaver(output_dir)
        if "search" in requested_stages:
            saver.save_json(
                [_search_result_to_dict(result) for result in search_results],
                "search_results.json",
            )
            console.print(
                f"[green]Saved search_results.json ({len(search_results)} items)[/green]"
            )
        (output_dir / "stage_timings.json").write_text(
            json.dumps(stage_timings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "search_results": search_results,
            "answer_results": [],
            "eval_result": None,
            "output_dir": str(output_dir),
        }

    answer_results, search_results = await asyncio.to_thread(
        _run_emem_answers,
        dataset=dataset,
        dataset_config=dataset_config,
        system_config=system_config,
        output_dir=output_dir,
        emem_symbols=emem_symbols,
        console=console,
    )

    saver = ResultSaver(output_dir)
    saver.save_json([_search_result_to_dict(result) for result in search_results], "search_results.json")
    saver.save_json([_answer_result_to_dict(result) for result in answer_results], "answer_results.json")
    console.print(f"[green]Saved search_results.json ({len(search_results)} items)[/green]")
    console.print(f"[green]Saved answer_results.json ({len(answer_results)} items)[/green]")

    eval_result = await evaluator.evaluate(answer_results)
    saver.save_json(_eval_result_to_dict(eval_result), "eval_results.json")
    _write_report(output_dir, system_config, eval_result, time.time() - start_time)

    return {
        "search_results": search_results,
        "answer_results": answer_results,
        "eval_result": eval_result,
        "output_dir": str(output_dir),
    }


def _normalize_emem_stages(stages: Optional[List[str]]) -> Set[str]:
    if stages is None:
        return {"search", "answer", "evaluate"}
    return {str(stage) for stage in stages}


def _should_run_answer_only_from_saved_search(stages: Set[str]) -> bool:
    return "answer" in stages and "search" not in stages and "add" not in stages


def _prepare_memoryagentbench_longmemeval_for_emem(
    *,
    dataset: Dataset,
    system_config: Dict[str, Any],
    console: Console,
) -> Tuple[Dataset, Dict[str, Any]]:
    """Use raw LongMemEval sessions for EMem instead of MAB memorize chunks."""
    mab_config = system_config.get("memoryagentbench")
    if not isinstance(mab_config, dict):
        return dataset, system_config

    source = str(mab_config.get("source", ""))
    if not source.startswith("longmemeval"):
        return dataset, system_config

    data_dir = mab_config.get("data_dir")
    if not data_dir:
        return dataset, system_config

    raw_path = Path(data_dir) / "memoryagentbench_raw.json"
    if not raw_path.exists():
        console.print(
            f"[yellow]EMem native LongMemEval conversion skipped: missing {raw_path}[/yellow]"
        )
        return dataset, system_config

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    selected_ids = {conv.conversation_id for conv in dataset.conversations}
    native_conversations: List[EvalConversation] = []
    for conv in dataset.conversations:
        raw_idx = _memoryagentbench_conversation_index(conv.conversation_id)
        if raw_idx is None or raw_idx >= len(raw_samples):
            native_conversations.append(conv)
            continue
        native_conversations.append(
            _convert_memoryagentbench_longmemeval_raw_conversation(
                raw_samples[raw_idx],
                conversation_id=conv.conversation_id,
            )
        )

    prepared_dataset = Dataset(
        dataset_name=dataset.dataset_name,
        conversations=native_conversations,
        qa_pairs=[
            qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") in selected_ids
        ],
        metadata={
            **dataset.metadata,
            "emem_native_longmemeval": True,
            "emem_native_longmemeval_raw_file": str(raw_path),
        },
    )

    prepared_config = dict(system_config)
    emem_cfg = dict(prepared_config.get("emem", {}) or {})
    emem_cfg["date_format_type"] = "longmemeval"
    prepared_config["emem"] = emem_cfg

    console.print(
        "[cyan]EMem native LongMemEval conversion: using raw User/Assistant "
        f"sessions ({len(native_conversations)} conversation(s)); "
        "MemoryAgentBench memorize chunk wrappers bypassed.[/cyan]"
    )
    return prepared_dataset, prepared_config


def _memoryagentbench_conversation_index(conversation_id: str) -> Optional[int]:
    match = re.search(r"memoryagentbench_(\d+)$", str(conversation_id))
    if not match:
        return None
    return int(match.group(1))


def _convert_memoryagentbench_longmemeval_raw_conversation(
    sample: Dict[str, Any],
    *,
    conversation_id: str,
) -> EvalConversation:
    context = str(sample.get("context", ""))
    messages: List[Message] = []

    try:
        parsed = ast.literal_eval(context)
    except (SyntaxError, ValueError):
        parsed = []

    if isinstance(parsed, list):
        session_idx = 0
        item_idx = 0
        while item_idx < len(parsed):
            marker = parsed[item_idx]
            turns = parsed[item_idx + 1] if item_idx + 1 < len(parsed) else []
            if isinstance(marker, str) and marker.startswith("Chat Time:"):
                chat_time = _parse_memoryagentbench_longmemeval_chat_time(marker)
                if isinstance(turns, list):
                    for turn_idx, turn in enumerate(turns):
                        if not isinstance(turn, dict):
                            continue
                        role = _normalize_memoryagentbench_longmemeval_role(turn.get("role"))
                        content = str(turn.get("content") or "").strip()
                        if not content:
                            continue
                        timestamp = (
                            chat_time + timedelta(seconds=turn_idx)
                            if chat_time is not None
                            else None
                        )
                        messages.append(
                            Message(
                                sender_id=f"{role.lower()}_{conversation_id}",
                                sender_name=role,
                                content=content,
                                timestamp=timestamp,
                                metadata={
                                    "session": f"session_{session_idx}",
                                    "dia_id": f"D{session_idx}:{turn_idx + 1}",
                                    "mab_native_longmemeval": True,
                                    "mab_chat_time": marker,
                                    "mab_role": turn.get("role"),
                                    "timestamp_source": "mab_chat_time",
                                },
                            )
                        )
                session_idx += 1
                item_idx += 2
            else:
                item_idx += 1

    if not messages:
        messages = [
            Message(
                sender_id=f"user_{conversation_id}",
                sender_name="User",
                content=context,
                timestamp=None,
                metadata={
                    "session": "session_0",
                    "dia_id": "D0:1",
                    "mab_native_longmemeval_fallback": True,
                },
            )
        ]

    return EvalConversation(
        conversation_id=conversation_id,
        messages=messages,
        metadata={
            "speaker_a": "User",
            "speaker_b": "Assistant",
            "emem_native_longmemeval": True,
        },
    )


def _parse_memoryagentbench_longmemeval_chat_time(value: str) -> Optional[datetime]:
    match = re.search(
        r"Chat Time:\s*(\d{4}/\d{2}/\d{2})\s*\([^)]*\)\s*(\d{2}:\d{2})",
        str(value),
    )
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def _normalize_memoryagentbench_longmemeval_role(role: Any) -> str:
    normalized = str(role or "").strip().lower()
    if normalized == "assistant":
        return "Assistant"
    return "User"


def _load_emem_symbols(system_config: Dict[str, Any]) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[5]
    emem_root = Path(system_config.get("emem_root") or repo_root / "EMem").resolve()
    emem_src = emem_root / "src"
    if not emem_src.exists():
        raise FileNotFoundError(f"EMem source directory not found: {emem_src}")

    if str(emem_src) not in sys.path:
        sys.path.insert(0, str(emem_src))
    support_json_schema = system_config.get("support_json_schema", False)
    os.environ.setdefault("SUPPORT_JSON_SCHEMA", "true" if bool(support_json_schema) else "false")

    try:
        from emem import EMem
        from emem.utils.config_utils import BaseConfig
        from emem.utils.conversation_data_utils import (
            Conversation,
            EventSummary,
            LoCoMoSample,
            Observation,
            QA,
            Session,
            Turn,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import EMem dependencies. Install EMem requirements in the "
            "active environment, including python-igraph, before running "
            "`--system emem` or `--system emem_g`."
        ) from exc

    return {
        "EMem": EMem,
        "BaseConfig": BaseConfig,
        "Conversation": Conversation,
        "EventSummary": EventSummary,
        "LoCoMoSample": LoCoMoSample,
        "Observation": Observation,
        "QA": QA,
        "Session": Session,
        "Turn": Turn,
    }


def _configure_emem_environment(system_config: Dict[str, Any]) -> None:
    embedding_cfg = system_config.get("embedding", {})
    llm_cfg = system_config.get("llm", {})

    embedding_api_key = embedding_cfg.get("api_key") or llm_cfg.get("api_key")
    if embedding_api_key:
        os.environ["OPENAI_API_KEY"] = embedding_api_key
    elif llm_cfg.get("api_key") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = llm_cfg["api_key"]

    support_json_schema = system_config.get("support_json_schema")
    if support_json_schema is not None:
        os.environ["SUPPORT_JSON_SCHEMA"] = "true" if bool(support_json_schema) else "false"


def _run_emem_answers(
    dataset: Dataset,
    dataset_config: Dict[str, Any],
    system_config: Dict[str, Any],
    output_dir: Path,
    emem_symbols: Dict[str, Any],
    console: Console,
) -> Tuple[List[AnswerResult], List[SearchResult]]:
    qa_by_conversation: Dict[str, List[QAPair]] = defaultdict(list)
    for qa in dataset.qa_pairs:
        qa_by_conversation[qa.metadata.get("conversation_id", "")].append(qa)

    qa_lookup = {qa.question_id: qa for qa in dataset.qa_pairs}
    answers: Dict[str, AnswerResult] = _load_existing_answers(output_dir, qa_lookup)
    searches: Dict[str, SearchResult] = _load_existing_searches(output_dir, qa_lookup, answers)
    complete_conversations = _complete_answered_conversations(qa_by_conversation, answers)

    emem_outputs_dir = output_dir / "emem_outputs"
    emem_outputs_dir.mkdir(parents=True, exist_ok=True)

    config = _build_base_config(system_config, dataset.dataset_name, str(emem_outputs_dir), emem_symbols)
    llm_model_name = system_config.get("llm", {}).get("model", "gpt-4o-mini")
    embedding_model_name = _resolve_emem_embedding_model(system_config)
    console.print(
        f"[cyan]EMem embedding: model={embedding_model_name}, "
        f"base_url={system_config.get('embedding', {}).get('base_url') or 'https://api.openai.com/v1'}[/cyan]"
    )

    save_frequency = int(system_config.get("save_frequency") or 0)
    processed = 0
    saver = ResultSaver(output_dir)
    answer_llm_provider = _build_answer_llm_provider(system_config)
    answer_prompts = _load_answer_prompts()

    conversations_with_qas = [
        conv for conv in dataset.conversations if qa_by_conversation.get(conv.conversation_id)
    ]
    if complete_conversations:
        console.print(
            f"[green]EMem resume: skipping {len(complete_conversations)} conversation(s) "
            "with complete existing answers.[/green]"
        )
    for conv in tqdm(conversations_with_qas, desc="EMem samples", unit="sample"):
        if conv.conversation_id in complete_conversations:
            continue

        for qa in qa_by_conversation.get(conv.conversation_id, []):
            answers.pop(qa.question_id, None)
            searches.pop(qa.question_id, None)

        conv_qas = qa_by_conversation.get(conv.conversation_id, [])
        qa_groups = _build_emem_qa_groups(dataset.dataset_name, conv, conv_qas)

        for scope_idx, (scope_key, scoped_conv, scoped_qas, temporal_metadata) in enumerate(qa_groups):
            sample = _build_emem_sample(
                scoped_conv,
                scoped_qas,
                emem_symbols,
                date_format_type=config.date_format_type,
            )
            sample_save_dir = (
                emem_outputs_dir
                / f"sample_{_safe_path_part(sample.sample_id)}__{scope_idx:02d}_{_safe_path_part(scope_key)}"
            )
            mem_model = emem_symbols["EMem"](
                save_dir=str(sample_save_dir),
                llm_model_name=llm_model_name,
                llm_base_url=system_config.get("llm", {}).get("base_url"),
                embedding_model_name=embedding_model_name,
                embedding_base_url=system_config.get("embedding", {}).get("base_url"),
                global_config=config,
            )

            console.print(
                f"[cyan]Indexing EMem sample {sample.sample_id} "
                f"scope={scope_key} ({len(sample.qa)} questions, "
                f"{len(scoped_conv.messages)}/{len(conv.messages)} messages visible)[/cyan]"
            )
            mem_model.index_conversation(sample)
            qa_results = mem_model.rag_qa_conversation(queries=sample.qa, gold_docs=None, gold_answers=None)
            query_solutions, response_messages, metadata_items, retrieval_traces = _unpack_qa_results(qa_results)

            for idx, (emem_qa, solution) in enumerate(zip(sample.qa, query_solutions)):
                question_id = getattr(emem_qa, "question_id", None)
                if question_id is None:
                    question_id = _question_id_from_emem_qa(emem_qa)
                qa = qa_lookup[question_id]
                solution, temporal_context_metadata = _apply_post_retrieval_temporal_filter(
                    qa,
                    solution,
                )
                formatted_context, context_metadata, retrieval_items = _format_solution_context(
                    solution,
                    retrieval_traces,
                    idx,
                    config.date_format_type,
                    qa=qa,
                    conversation=scoped_conv,
                )
                metadata = dict(qa.metadata)
                fallback_answer = (solution.answer or "").strip()
                if temporal_context_metadata.get("emem_post_retrieval_temporal_filter_enabled"):
                    fallback_answer = "Error: Failed to generate temporally filtered answer"
                aligned_answer, answer_metadata = _generate_aligned_answer(
                    qa=qa,
                    dataset_name=dataset.dataset_name,
                    system_config=system_config,
                    llm_provider=answer_llm_provider,
                    prompts=answer_prompts,
                    context=formatted_context,
                    fallback_answer=fallback_answer,
                    console=console,
                )
                metadata.update(
                    {
                        "emem_variant": _variant_name(system_config),
                        "emem_sample_id": sample.sample_id,
                        "emem_scope_key": scope_key,
                        "emem_save_dir": str(sample_save_dir),
                        "emem_response_metadata": metadata_items[idx] if idx < len(metadata_items) else {},
                        "emem_raw_response": response_messages[idx] if idx < len(response_messages) else "",
                        "emem_internal_answer": (solution.answer or "").strip(),
                        **temporal_metadata,
                        **temporal_context_metadata,
                        **context_metadata,
                        **answer_metadata,
                    }
                )

                searches[qa.question_id] = SearchResult(
                    query=getattr(emem_qa, "question", qa.question),
                    conversation_id=qa.metadata.get("conversation_id", ""),
                    results=retrieval_items,
                    retrieval_metadata={
                        "system": "emem",
                        "formatted_context": formatted_context,
                        "question_id": qa.question_id,
                        **metadata,
                    },
                    question_id=qa.question_id,
                )
                answers[qa.question_id] = AnswerResult(
                    question_id=qa.question_id,
                    question=qa.question,
                    answer=aligned_answer,
                    golden_answer=qa.answer,
                    category=qa.category,
                    conversation_id=qa.metadata.get("conversation_id", ""),
                    formatted_context=formatted_context,
                    search_results=retrieval_items,
                    metadata=metadata,
                )
                processed += 1

            if save_frequency > 0 and processed % save_frequency == 0:
                saver.save_json([_answer_result_to_dict(answers[qa.question_id]) for qa in dataset.qa_pairs if qa.question_id in answers], "answer_results.partial.json")

    missing = [qa.question_id for qa in dataset.qa_pairs if qa.question_id not in answers]
    if missing:
        raise RuntimeError(f"EMem did not produce answers for {len(missing)} questions; first missing: {missing[0]}")

    return (
        [answers[qa.question_id] for qa in dataset.qa_pairs],
        [
            searches.get(qa.question_id)
            or _search_result_from_answer(answers[qa.question_id], qa)
            for qa in dataset.qa_pairs
        ],
    )


def _run_emem_add_search(
    dataset: Dataset,
    system_config: Dict[str, Any],
    output_dir: Path,
    emem_symbols: Dict[str, Any],
    console: Console,
    *,
    run_search: bool,
) -> Tuple[List[SearchResult], Dict[str, float]]:
    """Run EMem indexing and retrieval without its answer-generation step.

    EMem normally combines retrieval and QA in ``rag_qa_conversation``.  The
    speed benchmark needs a real boundary, so retrieval is invoked directly
    after ``index_conversation`` and one external query at a time.
    """
    invocation_start = time.perf_counter()
    add_seconds = 0.0
    search_seconds = 0.0
    searches: Dict[str, SearchResult] = {}
    qa_by_conversation: Dict[str, List[QAPair]] = defaultdict(list)
    for qa in dataset.qa_pairs:
        qa_by_conversation[qa.metadata.get("conversation_id", "")].append(qa)
    qa_lookup = {qa.question_id: qa for qa in dataset.qa_pairs}

    emem_outputs_dir = output_dir / "emem_outputs"
    emem_outputs_dir.mkdir(parents=True, exist_ok=True)
    config = _build_base_config(
        system_config, dataset.dataset_name, str(emem_outputs_dir), emem_symbols
    )
    llm_model_name = system_config.get("llm", {}).get("model", "gpt-4o-mini")
    embedding_model_name = _resolve_emem_embedding_model(system_config)

    conversations_with_qas = [
        conv
        for conv in dataset.conversations
        if qa_by_conversation.get(conv.conversation_id)
    ]
    for conv in tqdm(conversations_with_qas, desc="EMem samples", unit="sample"):
        conv_qas = qa_by_conversation[conv.conversation_id]
        qa_groups = _build_emem_qa_groups(dataset.dataset_name, conv, conv_qas)
        for scope_idx, (scope_key, scoped_conv, scoped_qas, temporal_metadata) in enumerate(
            qa_groups
        ):
            sample = _build_emem_sample(
                scoped_conv,
                scoped_qas,
                emem_symbols,
                date_format_type=config.date_format_type,
            )
            sample_save_dir = (
                emem_outputs_dir
                / f"sample_{_safe_path_part(sample.sample_id)}__{scope_idx:02d}_{_safe_path_part(scope_key)}"
            )
            mem_model = emem_symbols["EMem"](
                save_dir=str(sample_save_dir),
                llm_model_name=llm_model_name,
                llm_base_url=system_config.get("llm", {}).get("base_url"),
                embedding_model_name=embedding_model_name,
                embedding_base_url=system_config.get("embedding", {}).get("base_url"),
                global_config=config,
            )

            console.print(
                f"[cyan]Indexing EMem sample {sample.sample_id} scope={scope_key} "
                f"({len(scoped_conv.messages)}/{len(conv.messages)} messages visible)[/cyan]"
            )
            add_start = time.perf_counter()
            mem_model.index_conversation(sample)
            add_seconds += time.perf_counter() - add_start

            if not run_search:
                continue

            for emem_qa in sample.qa:
                question_id = getattr(emem_qa, "question_id", None)
                if question_id is None:
                    question_id = _question_id_from_emem_qa(emem_qa)
                qa = qa_lookup[question_id]
                query = getattr(emem_qa, "question", qa.question)
                search_start = time.perf_counter()
                solutions, retrieval_traces = (
                    mem_model.retrieve_structuring_conversation_enhanced_v2(
                        queries=[query],
                        gold_docs=None,
                        no_query_trace_saving=True,
                    )
                )
                search_seconds += time.perf_counter() - search_start
                solution, temporal_context_metadata = _apply_post_retrieval_temporal_filter(
                    qa, solutions[0]
                )
                formatted_context, context_metadata, retrieval_items = _format_solution_context(
                    solution,
                    retrieval_traces,
                    0,
                    config.date_format_type,
                    qa=qa,
                    conversation=scoped_conv,
                )
                metadata = {
                    "emem_variant": _variant_name(system_config),
                    "emem_sample_id": sample.sample_id,
                    "emem_scope_key": scope_key,
                    "emem_save_dir": str(sample_save_dir),
                    **temporal_metadata,
                    **temporal_context_metadata,
                    **context_metadata,
                }
                searches[qa.question_id] = SearchResult(
                    query=query,
                    conversation_id=qa.metadata.get("conversation_id", ""),
                    results=retrieval_items,
                    retrieval_metadata={
                        "system": "emem",
                        "formatted_context": formatted_context,
                        "question_id": qa.question_id,
                        **metadata,
                    },
                    question_id=qa.question_id,
                )

    if run_search:
        missing = [qa.question_id for qa in dataset.qa_pairs if qa.question_id not in searches]
        if missing:
            raise RuntimeError(
                f"EMem search did not produce {len(missing)} results; first missing: {missing[0]}"
            )

    timings = {
        "add_seconds": add_seconds,
        "ready_wait_seconds": 0.0,
        "add_to_ready_seconds": add_seconds,
        "search_seconds": search_seconds,
        "invocation_seconds": time.perf_counter() - invocation_start,
    }
    return [searches[qa.question_id] for qa in dataset.qa_pairs if qa.question_id in searches], timings


def _run_emem_answer_only_from_saved_search(
    dataset: Dataset,
    dataset_config: Dict[str, Any],
    system_config: Dict[str, Any],
    output_dir: Path,
    console: Console,
) -> Tuple[List[AnswerResult], List[SearchResult]]:
    del dataset_config  # Reserved for parity with the full EMem runner signature.

    qa_lookup = {qa.question_id: qa for qa in dataset.qa_pairs}
    searches = _load_saved_search_results_for_answer(output_dir, qa_lookup)
    missing = [qa.question_id for qa in dataset.qa_pairs if qa.question_id not in searches]
    if missing:
        raise RuntimeError(
            f"EMem answer-only mode is missing search results for {len(missing)} questions; "
            f"first missing: {missing[0]}"
        )

    answer_workers = _emem_answer_num_workers(system_config)
    answer_prompts: Optional[Dict[str, Any]] = None
    if any(
        _should_use_aligned_answer(_answer_prompt_key(qa, dataset.dataset_name, system_config))
        for qa in dataset.qa_pairs
    ):
        answer_prompts = _load_answer_prompts()

    console.print(f"[cyan]EMem answer-only concurrency: {answer_workers} workers[/cyan]")
    thread_state = local()

    def answer_one(qa: QAPair) -> AnswerResult:
        search = searches[qa.question_id]
        formatted_context = _formatted_context_from_search_result(search)
        metadata = _answer_metadata_from_saved_search(qa, search)
        fallback_answer = _fallback_answer_from_saved_search_metadata(metadata)
        prompt_key = _answer_prompt_key(qa, dataset.dataset_name, system_config)
        if _should_use_aligned_answer(prompt_key):
            llm_provider = getattr(thread_state, "answer_llm_provider", None)
            if llm_provider is None:
                llm_provider = _build_answer_llm_provider(system_config)
                thread_state.answer_llm_provider = llm_provider
            aligned_answer, answer_metadata = _generate_aligned_answer(
                qa=qa,
                dataset_name=dataset.dataset_name,
                system_config=system_config,
                llm_provider=llm_provider,
                prompts=answer_prompts or {},
                context=formatted_context,
                fallback_answer=fallback_answer,
                console=console,
            )
        else:
            aligned_answer = fallback_answer
            answer_metadata = {"emem_aligned_answer_used": False}
        metadata.update(answer_metadata)

        return AnswerResult(
            question_id=qa.question_id,
            question=qa.question,
            answer=aligned_answer,
            golden_answer=qa.answer,
            category=qa.category,
            conversation_id=search.conversation_id or qa.metadata.get("conversation_id", ""),
            formatted_context=formatted_context,
            search_results=search.results,
            metadata=metadata,
        )

    if answer_workers == 1:
        answer_results = [
            answer_one(qa)
            for qa in tqdm(dataset.qa_pairs, desc="EMem answer-only", unit="qa")
        ]
    else:
        with ThreadPoolExecutor(max_workers=answer_workers) as executor:
            answer_results = list(
                tqdm(
                    executor.map(answer_one, dataset.qa_pairs),
                    total=len(dataset.qa_pairs),
                    desc="EMem answer-only",
                    unit="qa",
                )
            )

    return [answer for answer in answer_results], [searches[qa.question_id] for qa in dataset.qa_pairs]


def _load_saved_search_results_for_answer(
    output_dir: Path,
    qa_lookup: Dict[str, QAPair],
) -> Dict[str, SearchResult]:
    path = output_dir / "search_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Search results not found: {path}. Please run EMem search/full evaluation first."
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid search_results.json: {path}") from exc

    if not isinstance(rows, list):
        raise ValueError(f"Expected search_results.json to contain a list: {path}")

    searches: Dict[str, SearchResult] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id") or "")
        qa = qa_lookup.get(question_id)
        if qa is None:
            continue
        searches[question_id] = SearchResult(
            query=str(row.get("query", qa.question)),
            conversation_id=str(row.get("conversation_id") or qa.metadata.get("conversation_id", "")),
            results=row.get("results") if isinstance(row.get("results"), list) else [],
            retrieval_metadata=(
                row.get("retrieval_metadata")
                if isinstance(row.get("retrieval_metadata"), dict)
                else {}
            ),
            question_id=question_id,
        )
    return searches


def _formatted_context_from_search_result(search: SearchResult) -> str:
    metadata = search.retrieval_metadata or {}
    if metadata.get("formatted_context") is not None:
        return str(metadata.get("formatted_context") or "")

    parts = []
    for idx, item in enumerate(search.results or []):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if content:
            parts.append(f"Memory {idx + 1}: {content}")
    return "\n".join(parts)


def _answer_metadata_from_saved_search(qa: QAPair, search: SearchResult) -> Dict[str, Any]:
    metadata = dict(qa.metadata or {})
    retrieval_metadata = search.retrieval_metadata or {}
    for key, value in retrieval_metadata.items():
        if key in {"formatted_context", "question_id", "system"}:
            continue
        metadata[key] = value

    for key in (
        "emem_aligned_answer_used",
        "emem_aligned_answer_prompt_key",
        "emem_aligned_answer_fallback_used",
    ):
        metadata.pop(key, None)
    return metadata


def _fallback_answer_from_saved_search_metadata(metadata: Dict[str, Any]) -> str:
    if metadata.get("emem_post_retrieval_temporal_filter_enabled"):
        return "Error: Failed to generate temporally filtered answer"

    internal_answer = metadata.get("emem_internal_answer")
    if isinstance(internal_answer, str):
        return internal_answer.strip()

    raw_response = metadata.get("emem_raw_response")
    if isinstance(raw_response, str):
        return raw_response.strip()

    return ""


def _emem_answer_num_workers(system_config: Dict[str, Any]) -> int:
    answer_cfg = system_config.get("answer", {}) or {}
    return max(1, int(answer_cfg.get("num_workers", 1)))


def _load_existing_answers(output_dir: Path, qa_lookup: Dict[str, QAPair]) -> Dict[str, AnswerResult]:
    path = output_dir / "answer_results.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    answers: Dict[str, AnswerResult] = {}
    if not isinstance(rows, list):
        return answers

    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id") or "")
        qa = qa_lookup.get(question_id)
        if qa is None or not _has_generated_answer(row):
            continue
        answers[question_id] = AnswerResult(
            question_id=question_id,
            question=str(row.get("question", qa.question)),
            answer=str(row.get("answer", "")),
            golden_answer=str(row.get("golden_answer", qa.answer)),
            category=row.get("category", qa.category),
            conversation_id=str(row.get("conversation_id") or qa.metadata.get("conversation_id", "")),
            formatted_context=str(row.get("formatted_context", "")),
            search_results=row.get("search_results") if isinstance(row.get("search_results"), list) else [],
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        )
    return answers


def _load_existing_searches(
    output_dir: Path,
    qa_lookup: Dict[str, QAPair],
    answers: Dict[str, AnswerResult],
) -> Dict[str, SearchResult]:
    path = output_dir / "search_results.json"
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    searches: Dict[str, SearchResult] = {}
    if not isinstance(rows, list):
        return searches

    for row in rows:
        if not isinstance(row, dict):
            continue
        question_id = str(row.get("question_id") or "")
        qa = qa_lookup.get(question_id)
        if qa is None or question_id not in answers:
            continue
        searches[question_id] = SearchResult(
            query=str(row.get("query", qa.question)),
            conversation_id=str(row.get("conversation_id") or qa.metadata.get("conversation_id", "")),
            results=row.get("results") if isinstance(row.get("results"), list) else [],
            retrieval_metadata=(
                row.get("retrieval_metadata")
                if isinstance(row.get("retrieval_metadata"), dict)
                else {}
            ),
            question_id=question_id,
        )
    return searches


def _has_generated_answer(row: Dict[str, Any]) -> bool:
    answer = row.get("answer")
    return isinstance(answer, str) and bool(answer.strip())


def _complete_answered_conversations(
    qa_by_conversation: Dict[str, List[QAPair]],
    answers: Dict[str, AnswerResult],
) -> set[str]:
    complete = set()
    for conversation_id, qas in qa_by_conversation.items():
        if qas and all(qa.question_id in answers for qa in qas):
            complete.add(conversation_id)
    return complete


def _search_result_from_answer(answer: AnswerResult, qa: QAPair) -> SearchResult:
    metadata = dict(answer.metadata or {})
    return SearchResult(
        query=qa.question,
        conversation_id=answer.conversation_id or qa.metadata.get("conversation_id", ""),
        results=answer.search_results or [],
        retrieval_metadata={
            "system": "emem",
            "formatted_context": answer.formatted_context,
            "question_id": qa.question_id,
            "resume_reconstructed_from_answer": True,
            **metadata,
        },
        question_id=qa.question_id,
    )


def _build_base_config(
    system_config: Dict[str, Any],
    dataset_name: str,
    save_dir: str,
    emem_symbols: Dict[str, Any],
) -> Any:
    cfg = system_config.get("emem", {})
    llm_cfg = system_config.get("llm", {})
    embedding_cfg = system_config.get("embedding", {})
    base_config = emem_symbols["BaseConfig"](
        max_new_tokens=int(llm_cfg.get("max_tokens", cfg.get("max_new_tokens", 8192))),
        seed=int(cfg.get("seed", 42)),
        temperature=float(llm_cfg.get("temperature", cfg.get("temperature", 0))),
        embedding_model_name=_resolve_emem_embedding_model(system_config),
        force_openie_from_scratch=bool(cfg.get("force_openie_from_scratch", False)),
        force_index_from_scratch=bool(cfg.get("force_index_from_scratch", False)),
        save_openie=bool(cfg.get("save_openie", True)),
        openie_mode=cfg.get("openie_mode", "edu_based_contextual_ee_online"),
        embedding_batch_size=int(embedding_cfg.get("batch_size", cfg.get("embedding_batch_size", 256))),
        embedding_return_as_normalized=bool(cfg.get("embedding_return_as_normalized", True)),
        synonymy_edge_sim_threshold=float(cfg.get("synonymy_edge_sim_threshold", 0.9)),
        linking_top_k=int(cfg.get("linking_top_k", 30)),
        qa_top_k=int(cfg.get("qa_top_k", 10)),
        date_format_type=cfg.get("date_format_type") or ("longmemeval" if dataset_name == "longmemeval" else "locomo"),
        save_dir=save_dir,
        skip_edu_context_gen=bool(cfg.get("skip_edu_context_gen", True)),
        skip_retrieval_ppr=bool(cfg.get("skip_retrieval_ppr", False)),
    )
    base_config.llm_base_url = llm_cfg.get("base_url")
    base_config.embedding_base_url = embedding_cfg.get("base_url")
    base_config.api_key = llm_cfg.get("api_key")
    return base_config


def _resolve_emem_embedding_model(system_config: Dict[str, Any]) -> str:
    env_override = os.environ.get("EMEM_EMBEDDING_MODEL")
    if env_override:
        return env_override

    configured_model = (system_config.get("embedding", {}) or {}).get("model")
    if configured_model:
        return str(configured_model)

    return os.environ.get("VECTORIZE_MODEL") or "text-embedding-3-small"


def _build_emem_sample(
    conv: EvalConversation,
    qa_pairs: List[QAPair],
    emem_symbols: Dict[str, Any],
    *,
    date_format_type: str = "locomo",
) -> Any:
    sessions = _build_emem_sessions(conv, emem_symbols, date_format_type=date_format_type)
    conversation = emem_symbols["Conversation"](
        speaker_a=conv.metadata.get("speaker_a") or "User",
        speaker_b=conv.metadata.get("speaker_b") or "Assistant",
        sessions=sessions,
    )
    return emem_symbols["LoCoMoSample"](
        sample_id=conv.conversation_id,
        qa=[_build_emem_qa(qa, emem_symbols) for qa in qa_pairs],
        conversation=conversation,
        event_summary=emem_symbols["EventSummary"](events={}),
        observation=emem_symbols["Observation"](observations={}),
        session_summary={},
    )


def _build_temporal_qa_groups(
    conv: EvalConversation, qa_pairs: List[QAPair]
) -> List[Tuple[str, EvalConversation, List[QAPair], Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[QAPair]] = {}
    group_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for qa in qa_pairs:
        question_date = qa.metadata.get("question_date")
        cutoff = _parse_temporal_datetime(question_date)
        if question_date and cutoff is not None:
            key = ("visible_until", cutoff.isoformat())
        elif question_date:
            key = ("unparsed_question_date", str(question_date))
        else:
            key = ("full", "")
        grouped.setdefault(key, []).append(qa)
        if key not in group_meta:
            group_meta[key] = {
                "question_date": question_date,
                "question_cutoff": cutoff,
            }

    groups: List[Tuple[str, EvalConversation, List[QAPair], Dict[str, Any]]] = []
    for key, scoped_qas in grouped.items():
        scope_type, scope_value = key
        question_date = group_meta[key]["question_date"]
        cutoff = group_meta[key]["question_cutoff"]
        scoped_conv, temporal_metadata = _filter_conversation_for_question_date(
            conv, question_date, cutoff
        )
        scope_key = scope_type if not scope_value else f"{scope_type}_{scope_value}"
        groups.append((scope_key, scoped_conv, scoped_qas, temporal_metadata))
    return groups


def _build_emem_qa_groups(
    dataset_name: str,
    conv: EvalConversation,
    qa_pairs: List[QAPair],
) -> List[Tuple[str, EvalConversation, List[QAPair], Dict[str, Any]]]:
    if _is_memoryagentbench_longmemeval_scope(dataset_name, qa_pairs):
        return [
            (
                "full",
                conv,
                qa_pairs,
                {
                    "emem_temporal_filter_enabled": False,
                    "emem_temporal_filter_skipped_reason": (
                        "memoryagentbench_longmemeval_memorize_once_query_many"
                    ),
                    "emem_original_message_count": len(conv.messages),
                    "emem_visible_message_count": len(conv.messages),
                    "emem_future_filtered_message_count": 0,
                    "emem_unknown_timestamp_message_count": 0,
                },
            )
        ]
    return _build_temporal_qa_groups(conv, qa_pairs)


def _is_memoryagentbench_longmemeval_scope(dataset_name: str, qa_pairs: List[QAPair]) -> bool:
    if not str(dataset_name).startswith("memoryagentbench") or not qa_pairs:
        return False
    return all(str(qa.metadata.get("source", "")).startswith("longmemeval") for qa in qa_pairs)


def _is_memoryagentbench_longmemeval_qa(qa: QAPair) -> bool:
    return str(qa.metadata.get("source", "")).startswith("longmemeval") and str(
        qa.metadata.get("prompt_key", "")
    ).startswith("answer_prompt_mab_")


def _apply_post_retrieval_temporal_filter(
    qa: QAPair,
    solution: Any,
) -> Tuple[Any, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "emem_post_retrieval_temporal_filter_enabled": False,
    }
    if not _is_memoryagentbench_longmemeval_qa(qa):
        return solution, metadata

    question_date = qa.metadata.get("question_date")
    cutoff = _parse_temporal_datetime(question_date)
    metadata["emem_post_retrieval_question_date"] = question_date
    if not question_date:
        metadata["emem_post_retrieval_temporal_filter_skipped_reason"] = "missing_question_date"
        return solution, metadata
    if cutoff is None:
        metadata["emem_post_retrieval_temporal_filter_error"] = "failed_to_parse_question_date"
        return solution, metadata

    metadata.update(
        {
            "emem_post_retrieval_temporal_filter_enabled": True,
            "emem_post_retrieval_cutoff": cutoff.isoformat(),
        }
    )

    edus = list(getattr(solution, "edus", None) or [])
    edu_scores = getattr(solution, "edu_scores", None)
    filtered_edus, filtered_edu_scores, edu_meta = _filter_temporal_items(
        edus,
        edu_scores,
        cutoff,
        _item_datetime,
    )
    if getattr(solution, "edus", None) is not None:
        solution.edus = filtered_edus
        solution.edu_scores = filtered_edu_scores
    metadata.update({f"emem_post_retrieval_edu_{key}": value for key, value in edu_meta.items()})

    docs = list(getattr(solution, "docs", None) or [])
    doc_scores = getattr(solution, "doc_scores", None)
    filtered_docs, filtered_doc_scores, doc_meta = _filter_temporal_items(
        docs,
        doc_scores,
        cutoff,
        _item_datetime,
    )
    if getattr(solution, "docs", None) is not None:
        solution.docs = filtered_docs
        solution.doc_scores = filtered_doc_scores
    metadata.update({f"emem_post_retrieval_session_{key}": value for key, value in doc_meta.items()})

    return solution, metadata


def _filter_temporal_items(
    items: List[Any],
    scores: Any,
    cutoff: datetime,
    get_item_datetime: Any,
) -> Tuple[List[Any], Any, Dict[str, int]]:
    kept_items: List[Any] = []
    kept_indices: List[int] = []
    future_count = 0
    unknown_count = 0

    for idx, item in enumerate(items):
        item_dt = get_item_datetime(item)
        if item_dt is None:
            unknown_count += 1
            kept_items.append(item)
            kept_indices.append(idx)
        elif item_dt <= cutoff:
            kept_items.append(item)
            kept_indices.append(idx)
        else:
            future_count += 1

    return (
        kept_items,
        _filter_scores_by_indices(scores, kept_indices),
        {
            "original_count": len(items),
            "visible_count": len(kept_items),
            "future_filtered_count": future_count,
            "unknown_timestamp_count": unknown_count,
        },
    )


def _filter_scores_by_indices(scores: Any, indices: List[int]) -> Any:
    if scores is None:
        return None
    try:
        return scores[indices]
    except Exception:
        try:
            return [scores[idx] for idx in indices]
        except Exception:
            return scores


def _item_datetime(item: Any) -> Optional[datetime]:
    for attr_name in ("date", "date_time", "timestamp"):
        value = getattr(item, attr_name, None)
        parsed = _parse_temporal_datetime(value)
        if parsed is not None:
            return parsed
    if isinstance(item, dict):
        for key in ("date", "date_time", "timestamp"):
            parsed = _parse_temporal_datetime(item.get(key))
            if parsed is not None:
                return parsed
    return None


def _filter_conversation_for_question_date(
    conv: EvalConversation,
    question_date: Any,
    cutoff: Optional[datetime],
) -> Tuple[EvalConversation, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "emem_temporal_filter_enabled": False,
        "emem_original_message_count": len(conv.messages),
        "emem_visible_message_count": len(conv.messages),
        "emem_future_filtered_message_count": 0,
        "emem_unknown_timestamp_message_count": 0,
    }
    if not question_date:
        return conv, metadata

    metadata["question_date"] = question_date
    if cutoff is None:
        metadata["emem_temporal_filter_error"] = "failed to parse question_date"
        return conv, metadata

    visible_messages: List[Message] = []
    future_count = 0
    unknown_count = 0
    for message in conv.messages:
        message_dt = _normalize_datetime(message.timestamp) if message.timestamp else None
        if message_dt is None:
            unknown_count += 1
            visible_messages.append(message)
        elif message_dt <= cutoff:
            visible_messages.append(message)
        else:
            future_count += 1

    metadata.update(
        {
            "emem_temporal_filter_enabled": True,
            "emem_temporal_filter_granularity": "message_timestamp",
            "emem_visible_message_count": len(visible_messages),
            "emem_future_filtered_message_count": future_count,
            "emem_unknown_timestamp_message_count": unknown_count,
            "emem_unknown_timestamp_visible": True,
        }
    )
    return (
        EvalConversation(
            conversation_id=conv.conversation_id,
            messages=visible_messages,
            metadata=dict(conv.metadata),
        ),
        metadata,
    )


def _build_emem_sessions(
    conv: EvalConversation,
    emem_symbols: Dict[str, Any],
    *,
    date_format_type: str = "locomo",
) -> Dict[int, Any]:
    grouped: Dict[int, List[Message]] = defaultdict(list)
    session_order: Dict[str, int] = {}
    for message in conv.messages:
        session_name = message.metadata.get("session") or "session_0"
        if session_name not in session_order:
            session_order[session_name] = _session_index(session_name, len(session_order))
        grouped[session_order[session_name]].append(message)

    if not grouped:
        grouped[0] = []

    sessions = {}
    for session_id in sorted(grouped):
        messages = grouped[session_id]
        date_time = _format_emem_datetime(
            messages[0].timestamp if messages else None,
            date_format_type=date_format_type,
        )
        turns = [
            emem_symbols["Turn"](
                speaker=message.sender_name,
                dia_id=message.metadata.get("dia_id") or f"D{session_id}:{idx}",
                text=message.content,
            )
            for idx, message in enumerate(messages)
        ]
        sessions[session_id] = emem_symbols["Session"](
            session_id=session_id,
            date_time=date_time,
            turns=turns,
        )
    return sessions


def _build_emem_qa(qa: QAPair, emem_symbols: Dict[str, Any]) -> Any:
    question = _build_emem_retrieval_query(qa)

    category = None
    if qa.category is not None:
        try:
            category = int(qa.category)
        except (TypeError, ValueError):
            category = qa.category

    emem_qa = emem_symbols["QA"](
        question=question,
        answer=qa.answer,
        evidence=qa.evidence,
        category=category,
        adversarial_answer=qa.metadata.get("adversarial_answer"),
    )
    setattr(emem_qa, "question_id", qa.question_id)
    return emem_qa


def _build_emem_retrieval_query(qa: QAPair) -> str:
    prompt_key = str(qa.metadata.get("prompt_key", ""))
    if prompt_key.startswith("answer_prompt_mab_"):
        return qa.question
    return qa.question


def _build_emem_question(qa: QAPair) -> str:
    source = qa.metadata.get("source")
    prompt_key = str(qa.metadata.get("prompt_key", ""))
    if source and prompt_key.startswith("answer_prompt_mab_"):
        question = format_memoryagentbench_query(
            str(source),
            qa.question,
            agent_kind="agentic_memory_agent",
        )
    elif "question_date" in qa.metadata:
        question_date = qa.metadata["question_date"]
        question = (
            f"Question date: {question_date}\n"
            "Treat this as the current time when resolving relative phrases such as "
            "\"recently\", \"last month\", \"previous\", \"current\", or \"now\". "
            "Only use memories dated at or before this question date.\n"
            f"Question: {qa.question}"
        )
    else:
        question = qa.question

    if "all_options" in qa.metadata:
        options = "\n".join(f"{key} {value}" for key, value in qa.metadata["all_options"].items())
        question = (
            f"{question}\n\nOPTIONS:\n{options}\n\n"
            "Return only the best option letter, such as (a), (b), (c), or (d)."
        )
    return question


def _build_answer_llm_provider(system_config: Dict[str, Any]) -> LLMProvider:
    llm_config = system_config.get("llm", {})
    return LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model", "gpt-4o-mini"),
        api_key=llm_config.get("api_key", ""),
        base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        temperature=llm_config.get("temperature", 0),
        max_tokens=llm_config.get("max_tokens", 8192),
    )


def _load_answer_prompts() -> Dict[str, Any]:
    evaluation_root = Path(__file__).resolve().parents[2]
    prompts_path = evaluation_root / "config" / "prompts.yaml"
    return load_yaml(str(prompts_path))


def _answer_prompt_key(
    qa: QAPair,
    dataset_name: str,
    system_config: Dict[str, Any],
) -> Optional[str]:
    prompt_key = qa.metadata.get("prompt_key")
    if prompt_key:
        return str(prompt_key)
    if str(dataset_name).startswith("meme"):
        return str(
            (system_config.get("answer", {}) or {}).get(
                "prompt_key", "answer_prompt_meme_official"
            )
        )
    return (system_config.get("answer", {}) or {}).get("prompt_key")


def _generate_aligned_answer(
    *,
    qa: QAPair,
    dataset_name: str,
    system_config: Dict[str, Any],
    llm_provider: LLMProvider,
    prompts: Dict[str, Any],
    context: str,
    fallback_answer: str,
    console: Console,
) -> Tuple[str, Dict[str, Any]]:
    prompt_key = _answer_prompt_key(qa, dataset_name, system_config)
    if not _should_use_aligned_answer(prompt_key):
        return fallback_answer, {"emem_aligned_answer_used": False}

    query = _build_emem_question(qa)
    try:
        if str(prompt_key).startswith("answer_prompt_mab_"):
            messages = _build_memoryagentbench_answer_messages(query, context)
        else:
            prompt_template = prompts["online_api"]["default"][str(prompt_key)]
            messages = [{"role": "user", "content": prompt_template.format(context=context, question=query)}]
        answer = asyncio.run(llm_provider.generate_messages(messages=messages, temperature=0))
        answer = _clean_answer_text(answer)
        if answer:
            return answer, {
                "emem_aligned_answer_used": True,
                "emem_aligned_answer_prompt_key": prompt_key,
                "emem_aligned_answer_fallback_used": False,
            }
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[yellow]EMem aligned answer generation failed for {qa.question_id}: {exc}; "
            "using EMem internal answer.[/yellow]"
        )

    return fallback_answer, {
        "emem_aligned_answer_used": True,
        "emem_aligned_answer_prompt_key": prompt_key,
        "emem_aligned_answer_fallback_used": True,
    }


def _should_use_aligned_answer(prompt_key: Optional[str]) -> bool:
    if not prompt_key:
        return False
    prompt_key = str(prompt_key)
    return prompt_key == "answer_prompt_meme_official" or prompt_key.startswith("answer_prompt_mab_")


def _build_memoryagentbench_answer_messages(query: str, context: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a helpful AI. Answer the question based on query and memories.\n"
        f"{context}\n"
    )
    user_content = f"{query}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _clean_answer_text(answer: str) -> str:
    answer = (answer or "").strip()
    if "FINAL ANSWER:" in answer:
        parts = answer.split("FINAL ANSWER:")
        answer = parts[1].strip() if len(parts) > 1 else answer.strip()
    return answer


def _prepare_dataset_scope(
    dataset: Dataset,
    smoke_test: bool,
    smoke_messages: int,
    smoke_questions: int,
    from_conv: int,
    to_conv: Optional[int],
    locomo_samples: Optional[List[int]],
    filter_categories: Iterable[Any],
    console: Console,
) -> Dataset:
    if locomo_samples:
        total = len(dataset.conversations)
        selected = sorted(set(locomo_samples))
        invalid = [idx for idx in selected if idx < 0 or idx >= total]
        if invalid:
            raise ValueError(f"Invalid --locomo-sample indices {invalid}; valid range is 0..{total - 1}")
        conversations = [conv for idx, conv in enumerate(dataset.conversations) if idx in selected]
        dataset = _replace_dataset_scope(dataset, conversations)
        console.print(f"[cyan]Applied LoCoMo sample filter: {selected}[/cyan]")

    if from_conv > 0 or to_conv is not None:
        conversations = dataset.conversations[from_conv:to_conv]
        dataset = _replace_dataset_scope(dataset, conversations)
        console.print(f"[cyan]Applied conversation range: [{from_conv}:{to_conv or 'end'}][/cyan]")

    if smoke_test:
        conversations = []
        qa_pairs = []
        for conv in dataset.conversations:
            if smoke_messages > 0:
                conv = EvalConversation(
                    conversation_id=conv.conversation_id,
                    messages=conv.messages[:smoke_messages],
                    metadata=dict(conv.metadata),
                )
            conversations.append(conv)
            conv_qas = [qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") == conv.conversation_id]
            qa_pairs.extend(conv_qas[:smoke_questions] if smoke_questions > 0 else conv_qas)
        dataset = Dataset(
            dataset_name=f"{dataset.dataset_name}_smoke",
            conversations=conversations,
            qa_pairs=qa_pairs,
            metadata={**dataset.metadata, "smoke_test": True},
        )
        console.print(f"[yellow]Applied smoke test: {len(conversations)} conversations, {len(qa_pairs)} questions[/yellow]")

    filter_set = {str(category) for category in filter_categories}
    if filter_set:
        before = len(dataset.qa_pairs)
        dataset.qa_pairs = [qa for qa in dataset.qa_pairs if str(qa.category) not in filter_set]
        removed = before - len(dataset.qa_pairs)
        if removed:
            console.print(f"[dim]Filtered out {removed} questions from categories: {', '.join(sorted(filter_set))}[/dim]")

    return dataset


def _replace_dataset_scope(dataset: Dataset, conversations: List[EvalConversation]) -> Dataset:
    conv_ids = {conv.conversation_id for conv in conversations}
    return Dataset(
        dataset_name=dataset.dataset_name,
        conversations=conversations,
        qa_pairs=[qa for qa in dataset.qa_pairs if qa.metadata.get("conversation_id") in conv_ids],
        metadata=dict(dataset.metadata),
    )


def _format_solution_context(
    solution: Any,
    retrieval_traces: List[Any],
    idx: int,
    date_format_type: str,
    *,
    qa: Optional[QAPair] = None,
    conversation: Optional[EvalConversation] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    parts = []
    retrieval_items = []
    edu_count = 0
    if getattr(solution, "edus", None):
        for edu_idx, edu in enumerate(solution.edus):
            edu_count += 1
            text = edu.to_context_string(date_format_type) if hasattr(edu, "to_context_string") else str(edu)
            parts.append(f"Memory {edu_idx + 1}: {text}")
            score = None
            if getattr(solution, "edu_scores", None) is not None and edu_idx < len(solution.edu_scores):
                score = float(solution.edu_scores[edu_idx])
            retrieval_items.append(
                {
                    "content": text,
                    "score": score,
                    "metadata": {
                        "source": "edu",
                        "rank": edu_idx + 1,
                    },
                }
            )
    if not parts and getattr(solution, "docs", None):
        for doc_idx, doc in enumerate(solution.docs):
            text = _format_emem_session_doc(doc, date_format_type)
            parts.append(f"Session {doc_idx + 1}: {text}")
            score = None
            if getattr(solution, "doc_scores", None) is not None and doc_idx < len(solution.doc_scores):
                score = float(solution.doc_scores[doc_idx])
            retrieval_items.append(
                {
                    "content": text,
                    "score": score,
                    "metadata": {
                        "source": "session",
                        "rank": doc_idx + 1,
                    },
                }
            )

    raw_mapping_context = _build_icl_raw_mapping_context(qa, conversation)
    if raw_mapping_context:
        parts.insert(0, raw_mapping_context)
        retrieval_items.insert(
            0,
            {
                "content": raw_mapping_context,
                "score": None,
                "metadata": {
                    "source": "icl_raw_mapping",
                    "rank": 0,
                },
            },
        )

    metadata = {
        "emem_retrieved_edus": edu_count,
        "emem_retrieved_sessions": len(getattr(solution, "docs", []) or []),
        "emem_no_retrieved_context": not bool(parts),
        "emem_icl_raw_mapping_context": bool(raw_mapping_context),
        "emem_icl_raw_mapping_chunks": _count_icl_raw_mapping_chunks(raw_mapping_context),
    }
    if idx < len(retrieval_traces):
        metadata["emem_retrieval_trace"] = _json_safe(retrieval_traces[idx])
    return "\n".join(parts), metadata, retrieval_items


def _is_memoryagentbench_icl_qa(qa: Optional[QAPair]) -> bool:
    if qa is None:
        return False
    source = str(qa.metadata.get("source") or "")
    prompt_key = str(qa.metadata.get("prompt_key") or "")
    return source.startswith("icl_") or prompt_key == "answer_prompt_mab_in_context_learning"


def _has_icl_label_mapping(text: str) -> bool:
    return (
        "examples I have learned" in text
        and len(re.findall(r"\blabel\s*:\s*[0-9A-Za-z_-]+\b", text, re.IGNORECASE)) >= 5
    )


def _content_tokens(text: str) -> set[str]:
    stopwords = {
        "label",
        "search",
        "archival",
        "memory",
        "provided",
        "mapping",
        "context",
        "numerical",
        "assign",
        "only",
        "output",
        "question",
    }
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def _extract_icl_label_chunks(conversation: Optional[EvalConversation]) -> List[str]:
    if conversation is None:
        return []

    chunks = []
    for message in conversation.messages:
        content = str(message.content or "")
        if _has_icl_label_mapping(content):
            chunks.append(content)
    return chunks


def _select_icl_label_chunks(
    query: str,
    conversation: Optional[EvalConversation],
    *,
    limit: int = 5,
) -> List[str]:
    query_tokens = _content_tokens(query)
    scored_chunks = []
    for chunk in _extract_icl_label_chunks(conversation):
        chunk_tokens = _content_tokens(chunk)
        overlap = len(query_tokens & chunk_tokens)
        scored_chunks.append((overlap, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in scored_chunks[:limit]]


def _build_icl_raw_mapping_context(
    qa: Optional[QAPair],
    conversation: Optional[EvalConversation],
) -> str:
    if not _is_memoryagentbench_icl_qa(qa):
        return ""

    chunks = _select_icl_label_chunks(qa.question, conversation)
    if not chunks:
        return ""

    formatted_chunks = [
        "[EXACT TRAINING EXAMPLES WITH GOLD LABELS "
        f"{chunk_idx}]\nUse these verbatim utterance-to-label mappings to answer "
        f"the query. Output only the matched numeric label.\n{chunk}"
        for chunk_idx, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(formatted_chunks)


def _count_icl_raw_mapping_chunks(context: str) -> int:
    if not context:
        return 0
    return context.count("[EXACT TRAINING EXAMPLES WITH GOLD LABELS ")


def _format_emem_session_doc(doc: Any, date_format_type: str) -> str:
    if hasattr(doc, "turns"):
        date_time = getattr(doc, "date_time", "")
        turns = []
        for turn in getattr(doc, "turns", []) or []:
            speaker = getattr(turn, "speaker", "")
            text = getattr(turn, "text", "")
            turns.append(f"{speaker}: {text}" if speaker else str(text))
        body = "\n".join(turns).strip()
        if date_time and body:
            return f"[Source conversation date: {date_time}]\n{body}"
        return body or str(getattr(doc, "session_id", doc))
    if hasattr(doc, "to_context_string"):
        return doc.to_context_string(date_format_type)
    return str(doc)


def _unpack_qa_results(qa_results: Any) -> Tuple[List[Any], List[str], List[Dict[str, Any]], List[Any]]:
    if len(qa_results) >= 4:
        return qa_results[0], qa_results[1], qa_results[2], qa_results[3]
    return qa_results[0], qa_results[1], qa_results[2], []


def _question_id_from_emem_qa(emem_qa: Any) -> str:
    question_id = getattr(emem_qa, "question_id", None)
    if not question_id:
        raise ValueError("EMem QA object is missing question_id mapping.")
    return question_id


def _session_index(session_name: str, fallback: int) -> int:
    try:
        return int(str(session_name).rsplit("_", 1)[1])
    except (IndexError, TypeError, ValueError):
        return fallback


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _parse_temporal_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, (int, float)):
        try:
            return _normalize_datetime(datetime.fromtimestamp(value, tz=timezone.utc))
        except (OSError, OverflowError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    for candidate in (text, text.upper(), text.title()):
        for fmt in (
            "%I:%M %p on %d %B, %Y",
            "%Y/%m/%d (%a) %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return _normalize_datetime(datetime.strptime(candidate, fmt))
            except ValueError:
                continue
    try:
        return _normalize_datetime(datetime.fromisoformat(text))
    except ValueError:
        return None


def _format_emem_datetime(value: Optional[datetime], *, date_format_type: str = "locomo") -> str:
    if value is None:
        value = datetime(2024, 1, 1, 0, 0)
    if date_format_type == "longmemeval":
        return value.strftime("%Y/%m/%d (%a) %H:%M")
    hour = value.hour % 12 or 12
    am_pm = "am" if value.hour < 12 else "pm"
    return f"{hour}:{value.minute:02d} {am_pm} on {value.day} {value.strftime('%B')}, {value.year}"


def _variant_name(system_config: Dict[str, Any]) -> str:
    return "EMem" if bool(system_config.get("emem", {}).get("skip_retrieval_ppr", False)) else "EMem-G"


def _safe_path_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _answer_result_to_dict(result: AnswerResult) -> Dict[str, Any]:
    return {
        "question_id": result.question_id,
        "question": result.question,
        "answer": result.answer,
        "golden_answer": result.golden_answer,
        "category": result.category,
        "conversation_id": result.conversation_id,
        "formatted_context": result.formatted_context,
        "search_results": result.search_results,
        "metadata": result.metadata,
    }


def _search_result_to_dict(result: SearchResult) -> Dict[str, Any]:
    return {
        "question_id": result.question_id,
        "query": result.query,
        "conversation_id": result.conversation_id,
        "results": result.results,
        "retrieval_metadata": result.retrieval_metadata,
    }


def _eval_result_to_dict(result: EvaluationResult) -> Dict[str, Any]:
    return {
        "total_questions": result.total_questions,
        "correct": result.correct,
        "accuracy": result.accuracy,
        "detailed_results": result.detailed_results,
        "metadata": result.metadata,
    }


def _write_report(
    output_dir: Path,
    system_config: Dict[str, Any],
    eval_result: EvaluationResult,
    elapsed_time: float,
) -> None:
    report = "\n".join(
        [
            "=" * 60,
            "Evaluation Report",
            "=" * 60,
            "",
            f"System: {system_config.get('name', 'emem')}",
            f"Variant: {_variant_name(system_config)}",
            f"Time Elapsed: {elapsed_time:.2f}s",
            "",
            f"Total Questions: {eval_result.total_questions}",
            f"Correct: {eval_result.correct}",
            f"Accuracy: {eval_result.accuracy:.2%}",
            "",
        ]
    )
    task_scores = (eval_result.metadata or {}).get("task_scores", {})
    if task_scores:
        report = report.replace(
            f"Accuracy: {eval_result.accuracy:.2%}",
            f"Accuracy (MEME Overall): {eval_result.accuracy:.2%}",
            1,
        )
        report += "Task Accuracy (official MEME scoring):\n"
        for task in ("ER", "Agg", "Tr", "Del", "Cas", "Abs"):
            score = task_scores.get(task, {})
            report += (
                f"  {task}: {score.get('pass', 0):.2f}/{score.get('total', 0)} "
                f"({score.get('accuracy', 0):.2%})\n"
            )
        report += "\n"
    report += "=" * 60
    with open(output_dir / "report.txt", "w", encoding="utf-8") as f:
        f.write(report)
