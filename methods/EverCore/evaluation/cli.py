"""
CLI entry point for the evaluation framework.

Usage:
    python -m evaluation.cli --dataset locomo --system evermemos
    python -m evaluation.cli --dataset locomo --system evermemos --smoke 10
    python -m evaluation.cli --dataset locomo --system evermemos --stages search answer evaluate
"""

import asyncio
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Environment initialization - must be done before importing EverCore components
# Reference: src/bootstrap.py initialization logic

# Add project paths
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables
from common_utils.load_env import setup_environment

from evaluation.src.evaluators.registry import create_evaluator
from evaluation.src.datasets.memoryagentbench import (
    prepare_memoryagentbench_dataset,
    sanitize_path_component,
    source_profile,
)
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.logger import get_console
from evaluation.src.utils.fingerprint import public_config
from evaluation.src.utils.run_identity import (
    guard_artifacts, prompt_fingerprint, ANSWER_ARTIFACTS, EVALUATION_ARTIFACTS,
)


MEMORA_RECYCLE_EXIT_CODE = 75


MAB_DEFAULT_BENCHES = {
    "Accurate_Retrieval": [
        "eventqa_full",
        "longmemeval_s*",
        "ruler_qa1_197K",
        "ruler_qa2_421K"
    ],
    "Test_Time_Learning": [
        "recsys_redial_full",
        "icl_banking77_5900shot_balance",
        "icl_clinic150_7050shot_balance",
        "icl_nlu_8296shot_balance",
        "icl_trec_coarse_6600shot_balance",
        "icl_trec_fine_6400shot_balance",
    ],
    "Long_Range_Understanding": [
        "detective_qa",
        "infbench_sum_eng_shots2",
    ],
    "Conflict_Resolution": [
        "factconsolidation_mh_262k",
        "factconsolidation_sh_262k",
    ],
}

VALID_STAGES = {"add", "search", "answer", "evaluate"}
MEME_JUDGE_DATASETS = {"meme", "meme_filler128k", "meme_nofiller"}


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    Deep merge configuration dictionaries.

    Args:
        base: Base configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_config(result[key], value)
        else:
            # Direct override
            result[key] = value
    return result


def apply_env_evaluation_overrides(dataset_name: str, dataset_config: dict, console) -> bool:
    """Apply narrow evaluator overrides from environment variables."""
    if dataset_name not in MEME_JUDGE_DATASETS:
        return True

    raw_num_runs = os.environ.get("MEME_JUDGE_NUM_RUNS")
    if not raw_num_runs:
        return True

    try:
        num_runs = int(raw_num_runs)
    except ValueError:
        console.print(
            f"[red]❌ MEME_JUDGE_NUM_RUNS must be an integer, got: {raw_num_runs}[/red]"
        )
        return False

    if num_runs < 1:
        console.print(
            f"[red]❌ MEME_JUDGE_NUM_RUNS must be >= 1, got: {num_runs}[/red]"
        )
        return False

    dataset_config.setdefault("evaluation", {})["num_runs"] = num_runs
    console.print(f"  ⚖️  MEME judge num_runs override: {num_runs}")
    return True


def flatten_mab_sources(raw_sources):
    """Normalize argparse's repeated/nargs source values into a flat list."""
    if not raw_sources:
        return []
    return [source for group in raw_sources for source in group]


def resolve_memoryagentbench_plan(args, parser):
    """Resolve MemoryAgentBench split/source CLI options into ordered run tasks."""
    if args.dataset != "memoryagentbench":
        return []

    sources = flatten_mab_sources(args.mab_source)
    if args.mab_split:
        if args.mab_split not in MAB_DEFAULT_BENCHES:
            parser.error(
                "--mab-split must be one of: "
                + ", ".join(MAB_DEFAULT_BENCHES.keys())
            )
        if sources:
            return [(args.mab_split, source) for source in sources]
        return [
            (args.mab_split, source)
            for source in MAB_DEFAULT_BENCHES[args.mab_split]
        ]

    if sources:
        parser.error("--mab-split is required when --mab-source is specified")

    return [
        (split, source)
        for split, sources_for_split in MAB_DEFAULT_BENCHES.items()
        for source in sources_for_split
    ]


def build_memoryagentbench_child_command(args, split, source, output_dir):
    """Build a single-source child CLI invocation preserving parent options."""
    command = [
        sys.executable,
        "-m",
        "evaluation.cli",
        "--dataset",
        args.dataset,
        "--system",
        args.system,
        "--mab-split",
        split,
        "--mab-source",
        source,
        "--mab-worker",
    ]

    if args.stages:
        command.extend(["--stages", *args.stages])
    if args.smoke:
        command.append("--smoke")
    command.extend(["--smoke-messages", str(args.smoke_messages)])
    command.extend(["--smoke-questions", str(args.smoke_questions)])
    if args.conv_range is not None:
        command.extend(["--conv-range", args.conv_range])
    else:
        if args.from_conv != 0:
            command.extend(["--from-conv", str(args.from_conv)])
        if args.to_conv is not None:
            command.extend(["--to-conv", str(args.to_conv)])
    if args.locomo_sample:
        command.extend(["--locomo-sample", *[str(i) for i in args.locomo_sample]])
    if args.sample_workers is not None:
        command.extend(["--sample-workers", str(args.sample_workers)])
    if getattr(args, "context_window", None) is not None:
        command.extend(["--context-window", str(args.context_window)])
    if args.run_name:
        command.extend(["--run-name", args.run_name])
    if output_dir is not None:
        command.extend(["--output-dir", str(output_dir)])
    if args.clean_groups:
        command.append("--clean-groups")
    if args.mab_disable_official_chunking:
        command.append("--mab-disable-official-chunking")
    if args.mab_chunk_strategy:
        command.extend(["--mab-chunk-strategy", args.mab_chunk_strategy])
    if args.mab_chunk_size is not None:
        command.extend(["--mab-chunk-size", str(args.mab_chunk_size)])
    if args.mab_chunk_overlap is not None:
        command.extend(["--mab-chunk-overlap", str(args.mab_chunk_overlap)])

    return command


def run_memoryagentbench_batch(args, plan, console):
    """Run multiple MemoryAgentBench split/source tasks sequentially."""
    console.print(
        f"\n[bold cyan]MemoryAgentBench batch plan: {len(plan)} run(s)[/bold cyan]"
    )
    base_output_dir = None
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir

    for index, (split, source) in enumerate(plan, start=1):
        suffix = f"{sanitize_path_component(split)}-{sanitize_path_component(source)}"
        output_dir = base_output_dir / suffix if base_output_dir else None
        console.print(
            f"\n[bold cyan]▶ MemoryAgentBench run {index}/{len(plan)}: "
            f"{split} / {source}[/bold cyan]"
        )
        command = build_memoryagentbench_child_command(
            args=args,
            split=split,
            source=source,
            output_dir=output_dir,
        )
        recycle_count = 0
        while True:
            completed = subprocess.run(command, cwd=project_root)
            if (
                args.system != "memora"
                or completed.returncode != MEMORA_RECYCLE_EXIT_CODE
            ):
                break
            recycle_count += 1
            console.print(
                "[yellow]Memora worker requested a controlled recycle after "
                f"persisting its session checkpoint; restarting "
                f"{split} / {source} (recycle {recycle_count}).[/yellow]"
            )
        if completed.returncode != 0:
            console.print(
                f"[red]❌ MemoryAgentBench batch stopped at {split} / {source} "
                f"(exit code {completed.returncode})[/red]"
            )
            return completed.returncode

    console.print("\n[bold green]✨ MemoryAgentBench batch completed![/bold green]\n")
    return 0


def read_memoryagentbench_accuracy(result_dir: Path):
    """Read accuracy fields from a MemoryAgentBench result directory."""
    eval_path = result_dir / "eval_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "total_questions": data.get("total_questions"),
            "correct": data.get("correct"),
            "accuracy": data.get("accuracy"),
            "source": "eval_results.json",
        }

    report_path = result_dir / "report.txt"
    if not report_path.exists():
        return None

    total_questions = None
    correct = None
    accuracy = None
    for line in report_path.read_text(encoding="utf-8").splitlines():
        key, _, raw_value = line.partition(":")
        value = raw_value.strip()
        if key == "Total Questions":
            total_questions = int(value)
        elif key == "Correct":
            correct = int(value)
        elif key == "Accuracy":
            accuracy = float(value.rstrip("%")) / 100.0

    if accuracy is None:
        return None

    return {
        "total_questions": total_questions,
        "correct": correct,
        "accuracy": accuracy,
        "source": "report.txt",
    }


def memoryagentbench_result_dir(args, split: str, source: str, evaluation_root: Path) -> Path:
    """Resolve the result directory for one MemoryAgentBench split/source."""
    suffix = f"{sanitize_path_component(split)}-{sanitize_path_component(source)}"
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
        if not base_output_dir.is_absolute():
            base_output_dir = Path.cwd() / base_output_dir
        return base_output_dir / suffix

    base_name = f"{args.dataset}-{args.system}-{suffix}"
    if args.run_name:
        base_name = f"{base_name}-{args.run_name}"
    return evaluation_root / "results" / base_name


def report_memoryagentbench_accuracies(args, console) -> int:
    """Print all MemoryAgentBench accuracies for one run name."""
    if args.dataset != "memoryagentbench":
        console.print("[red]❌ --mab-report-accuracy only supports --dataset memoryagentbench[/red]")
        return 1

    evaluation_root = Path(__file__).parent
    rows = []
    missing = []
    for split, sources_for_split in MAB_DEFAULT_BENCHES.items():
        for source in sources_for_split:
            result_dir = memoryagentbench_result_dir(args, split, source, evaluation_root)
            metrics = read_memoryagentbench_accuracy(result_dir)
            if metrics is None:
                missing.append((split, source, result_dir))
                continue
            rows.append((split, source, result_dir, metrics))

    run_label = args.run_name or "(no run-name)"
    console.print(
        f"\n[bold cyan]MemoryAgentBench accuracy summary[/bold cyan] "
        f"system={args.system} run={run_label}\n"
    )

    if rows:
        split_width = max(len("Split"), *(len(row[0]) for row in rows))
        source_width = max(len("Source"), *(len(row[1]) for row in rows))
        header = (
            f"{'Split':<{split_width}}  "
            f"{'Source':<{source_width}}  "
            f"{'Accuracy':>9}  {'Correct':>8}  {'Total':>7}"
        )
        console.print(header, soft_wrap=True)
        console.print("-" * len(header), soft_wrap=True)
        for split, source, _result_dir, metrics in rows:
            accuracy = metrics["accuracy"]
            accuracy_text = f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
            correct = metrics["correct"] if metrics["correct"] is not None else "-"
            total = (
                metrics["total_questions"]
                if metrics["total_questions"] is not None
                else "-"
            )
            console.print(
                f"{split:<{split_width}}  "
                f"{source:<{source_width}}  "
                f"{accuracy_text:>15}  {str(correct):>8}  {str(total):>7}",
                soft_wrap=True,
            )

        accuracies_by_split = {}
        for split, source, _result_dir, metrics in rows:
            accuracies_by_split.setdefault(split, {})[source] = metrics["accuracy"]

        subset_scores = {}
        test_time_details = None
        for split, sources_for_split in MAB_DEFAULT_BENCHES.items():
            source_scores = accuracies_by_split.get(split, {})
            if not source_scores:
                continue

            if split == "Test_Time_Learning":
                icl_scores = [
                    source_scores[source]
                    for source in sources_for_split
                    if source.startswith("icl_") and source in source_scores
                ]
                mcc_accuracy = sum(icl_scores) / len(icl_scores) if icl_scores else None
                recsys_accuracy = source_scores.get("recsys_redial_full")
                component_scores = [
                    score
                    for score in (recsys_accuracy, mcc_accuracy)
                    if score is not None
                ]
                if not component_scores:
                    continue
                subset_scores[split] = sum(component_scores) / len(component_scores)
                test_time_details = {
                    "mcc_accuracy": mcc_accuracy,
                    "mcc_count": len(icl_scores),
                    "recsys_accuracy": recsys_accuracy,
                }
            else:
                ordered_scores = [
                    source_scores[source]
                    for source in sources_for_split
                    if source in source_scores
                ]
                subset_scores[split] = sum(ordered_scores) / len(ordered_scores)

        if subset_scores:
            console.print("\nSubset macro accuracies:")
            for split in MAB_DEFAULT_BENCHES:
                if split not in subset_scores:
                    continue
                accuracy = subset_scores[split]
                console.print(f"  - {split}: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                if split == "Test_Time_Learning" and test_time_details:
                    mcc_accuracy = test_time_details["mcc_accuracy"]
                    recsys_accuracy = test_time_details["recsys_accuracy"]
                    mcc_text = (
                        f"{mcc_accuracy:.4f} ({mcc_accuracy * 100:.2f}%)"
                        if mcc_accuracy is not None
                        else "-"
                    )
                    recsys_text = (
                        f"{recsys_accuracy:.4f} ({recsys_accuracy * 100:.2f}%)"
                        if recsys_accuracy is not None
                        else "-"
                    )
                    console.print(
                        f"    MCC accuracy: {mcc_text} "
                        f"(from {test_time_details['mcc_count']} icl_* sources)"
                    )
                    console.print(f"    recsys_redial_full: {recsys_text}")

            total_accuracy = sum(subset_scores.values()) / len(subset_scores)
            console.print(
                f"\nTotal macro accuracy over found subsets: {total_accuracy:.4f} "
                f"({total_accuracy * 100:.2f}%)"
            )

    if missing:
        console.print(f"\n[yellow]Missing result directories/files: {len(missing)}[/yellow]")
        for split, source, result_dir in missing:
            console.print(f"  - {split} / {source}: {result_dir}")

    return 0 if rows else 1


def is_evaluate_only(stages) -> bool:
    """Return True when the requested pipeline contains only Evaluate."""
    return stages is not None and set(stages) == {"evaluate"}


def is_answer_fast_path(stages) -> bool:
    """Return True when stages can be run from saved search results only."""
    if stages is None or "answer" not in stages:
        return False
    return set(stages).issubset({"answer", "evaluate"})


def resolve_output_dir(args, evaluation_root: Path, memoryagentbench_selection=None) -> Path:
    """Resolve the output directory using the same naming rules as the full pipeline."""
    if args.output_dir:
        return Path(args.output_dir)

    if memoryagentbench_selection:
        suffix = (
            f"{sanitize_path_component(memoryagentbench_selection['split'])}-"
            f"{sanitize_path_component(memoryagentbench_selection['source'])}"
        )
        base_name = f"{args.dataset}-{args.system}-{suffix}"
        if args.run_name:
            return evaluation_root / "results" / f"{base_name}-{args.run_name}"
        return evaluation_root / "results" / base_name

    if args.run_name:
        return evaluation_root / "results" / f"{args.dataset}-{args.system}-{args.run_name}"

    return evaluation_root / "results" / f"{args.dataset}-{args.system}"


def memoryagentbench_selection_from_args(args):
    """Build lightweight MemoryAgentBench output metadata without preparing data."""
    if args.dataset != "memoryagentbench" or not args.mab_split:
        return None
    return {
        "split": args.mab_split,
        "source": args.mab_source or "default",
    }


def dict_to_answer_result(data: dict):
    """Convert saved answer JSON into an AnswerResult while tolerating extra keys."""
    from dataclasses import fields
    from evaluation.src.core.data_models import AnswerResult

    valid_keys = {field.name for field in fields(AnswerResult)}
    return AnswerResult(**{key: value for key, value in data.items() if key in valid_keys})


def answer_result_to_dict(answer_result) -> dict:
    """Convert AnswerResult to JSON-serializable dict."""
    return {
        "question_id": answer_result.question_id,
        "question": answer_result.question,
        "answer": answer_result.answer,
        "golden_answer": answer_result.golden_answer,
        "category": answer_result.category,
        "conversation_id": answer_result.conversation_id,
        "formatted_context": answer_result.formatted_context,
        "metadata": answer_result.metadata,
    }


def dict_to_search_result(data: dict):
    """Convert saved search JSON into a SearchResult."""
    from evaluation.src.core.data_models import SearchResult

    return SearchResult(**data)


def load_saved_search_results(output_dir: Path):
    """Load search_results.json for answer fast path."""
    search_path = output_dir / "search_results.json"
    if not search_path.exists():
        raise FileNotFoundError(
            f"Search results not found: {search_path}. "
            "Please run 'search' stage first."
        )

    with open(search_path, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    return [dict_to_search_result(item) for item in search_data], search_data


def align_search_results_to_qa_pairs(search_results, qa_pairs):
    """Filter and reorder search results to match selected QA pairs."""
    by_question_id = {
        sr.question_id: sr
        for sr in search_results
        if getattr(sr, "question_id", "")
    }
    by_conversation_and_query = {
        (sr.conversation_id, sr.query): sr for sr in search_results
    }

    aligned = []
    missing = []
    for qa in qa_pairs:
        conv_id = qa.metadata.get("conversation_id", "")
        sr = by_question_id.get(qa.question_id)
        if sr is None:
            sr = by_conversation_and_query.get((conv_id, qa.question))
            if sr is not None and not sr.question_id:
                sr.question_id = qa.question_id
        if sr is None:
            missing.append(qa.question_id)
        else:
            aligned.append(sr)

    if missing:
        raise ValueError(
            f"Missing search results for {len(missing)} selected questions; "
            f"first missing question_id: {missing[0]}"
        )

    return aligned


def apply_fast_path_dataset_scope(dataset, args, console):
    """Apply the same QA subset controls used by the normal pipeline."""
    if args.locomo_sample:
        total_convs = len(dataset.conversations)
        selected_indices = sorted(set(args.locomo_sample))
        invalid = [idx for idx in selected_indices if idx < 0 or idx >= total_convs]
        if invalid:
            raise ValueError(
                f"Invalid --locomo-sample indices {invalid}; "
                f"valid range is 0..{total_convs - 1}"
            )
        selected_convs = [
            conv for idx, conv in enumerate(dataset.conversations) if idx in selected_indices
        ]
        selected_conv_ids = {conv.conversation_id for conv in selected_convs}
        dataset.conversations = selected_convs
        dataset.qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if qa.metadata.get("conversation_id") in selected_conv_ids
        ]
        console.print(
            f"[cyan]📌 LoCoMo Sample Filter Applied: "
            f"{len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} questions[/cyan]"
        )

    if args.from_conv > 0 or args.to_conv is not None:
        total_convs = len(dataset.conversations)
        end_idx = args.to_conv if args.to_conv is not None else total_convs
        selected_convs = dataset.conversations[args.from_conv:end_idx]
        selected_conv_ids = {conv.conversation_id for conv in selected_convs}
        dataset.conversations = selected_convs
        dataset.qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if qa.metadata.get("conversation_id") in selected_conv_ids
        ]
        console.print(
            f"[cyan]📌 Conversation Range Filter Applied: "
            f"[{args.from_conv}:{args.to_conv or 'end'}], "
            f"{len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} questions[/cyan]"
        )

    if args.smoke:
        dataset.conversations = dataset.conversations[:1]
        selected_conv_ids = {conv.conversation_id for conv in dataset.conversations}
        dataset.qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if qa.metadata.get("conversation_id") in selected_conv_ids
        ][: args.smoke_questions]
        console.print(
            f"[yellow]✂️  Smoke test applied: "
            f"{len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} questions[/yellow]"
        )

    if args.only_category:
        requested_categories = {str(category) for category in args.only_category}
        dataset.qa_pairs = [
            qa
            for qa in dataset.qa_pairs
            if str(qa.category).split("_", 1)[0] in requested_categories
        ]
        console.print(
            "[cyan]Only-category filter applied: "
            + ", ".join(sorted(requested_categories))
            + f" ({len(dataset.qa_pairs)} questions)[/cyan]"
        )

    return dataset


def apply_fast_path_category_filter(dataset, dataset_config: dict, console):
    """Apply dataset-configured category filters before aligning saved search results."""
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])
    if not filter_categories:
        return dataset

    filter_set = {str(category) for category in filter_categories}
    original_count = len(dataset.qa_pairs)
    dataset.qa_pairs = [
        qa for qa in dataset.qa_pairs if str(qa.category) not in filter_set
    ]
    filtered_count = original_count - len(dataset.qa_pairs)
    if filtered_count > 0:
        console.print(
            f"[dim]🔍 Filtered out {filtered_count} questions from categories: "
            f"{', '.join(sorted(filter_set))}[/dim]"
        )
        console.print(
            f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n"
        )

    return dataset


class AnswerOnlyAdapter:
    """Minimal adapter for saved-search answer reruns."""

    def __init__(self, config: dict):
        self.config = config
        llm_config = config.get("llm", {})

        from memory_layer.llm.llm_provider import LLMProvider

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0),
            max_tokens=llm_config.get("max_tokens", 16384),
        )

        evaluation_root = Path(__file__).parent
        prompts_path = evaluation_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))

    async def answer(self, query: str, context: str, **kwargs) -> str:
        prompt_key = kwargs.get("prompt_key")
        if prompt_key and prompt_key.startswith("answer_prompt_mab_"):
            messages = self._build_memoryagentbench_answer_messages(query, context)
        else:
            prompt = self._resolve_answer_prompt(prompt_key).format(
                context=context, question=query
            )
            messages = [{"role": "user", "content": prompt}]

        max_retries = int(self.config.get("answer", {}).get("max_retries", 3))
        for attempt in range(max_retries):
            try:
                result = await self.llm_provider.generate_messages(
                    messages=messages,
                    temperature=self.config.get("llm", {}).get("temperature", 0),
                )
                return self._extract_final_answer(result)
            except Exception as exc:
                print(
                    f"⚠️  Answer generation error "
                    f"(attempt {attempt + 1}/{max_retries}): {exc}"
                )
                if attempt == max_retries - 1:
                    raise
        return ""

    async def add(self, conversations, **kwargs):
        raise NotImplementedError("AnswerOnlyAdapter cannot run add stage")

    async def search(self, query: str, conversation_id: str, index, **kwargs):
        raise NotImplementedError("AnswerOnlyAdapter cannot run search stage")

    def get_system_info(self):
        return {"name": "AnswerOnlyAdapter", "config": self.config}

    def _build_memoryagentbench_answer_messages(self, query: str, context: str):
        import time

        system_prompt = (
            "You are a helpful AI. Answer the question based on query and memories.\n"
            f"{context}\n"
        )
        user_content = (
            f"{query}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _get_answer_prompt(self) -> str:
        prompt_key = self.config.get("answer", {}).get("prompt_key")
        if prompt_key:
            return self._resolve_answer_prompt(prompt_key)

        adapter_name = str(self.config.get("adapter", ""))
        default_prompts = self._prompts["online_api"]["default"]
        if adapter_name == "rag":
            return default_prompts["answer_prompt_cot"]
        if adapter_name in {"evermemos", "evermemos_api", "memmachine"}:
            from evaluation.src.adapters.evermemos.prompts.answer_prompts import (
                ANSWER_PROMPT,
            )

            return ANSWER_PROMPT
        if adapter_name == "mem0":
            return default_prompts["answer_prompt_mem0"]
        return default_prompts["answer_prompt_memos"]

    def _resolve_answer_prompt(self, prompt_key=None) -> str:
        if prompt_key:
            prompts = self._prompts["online_api"]["default"]
            if prompt_key not in prompts:
                raise KeyError(f"Unknown answer prompt key '{prompt_key}'")
            return prompts[prompt_key]
        return self._get_answer_prompt()

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        result = (text or "").strip()
        if "FINAL ANSWER:" in result:
            parts = result.split("FINAL ANSWER:")
            if len(parts) > 1:
                result = parts[-1].strip()
        return result


def eval_result_to_dict(eval_result) -> dict:
    """Convert EvaluationResult to JSON-serializable dict."""
    return {
        "total_questions": eval_result.total_questions,
        "correct": eval_result.correct,
        "accuracy": eval_result.accuracy,
        "detailed_results": eval_result.detailed_results,
        "metadata": eval_result.metadata,
    }


def load_saved_answer_results(output_dir: Path):
    """Load answer_results.json for evaluate-only mode."""
    answer_path = output_dir / "answer_results.json"
    if not answer_path.exists():
        raise FileNotFoundError(
            f"Answer results not found: {answer_path}. "
            "Please run 'answer' stage first."
        )

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    return [dict_to_answer_result(item) for item in answer_data], answer_data


def apply_evaluate_only_filters(answer_results, dataset_config: dict, args, console):
    """Apply filters that do not require loading the source dataset."""
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])
    if filter_categories:
        filter_set = {str(category) for category in filter_categories}
        original_count = len(answer_results)
        answer_results = [
            result for result in answer_results if str(result.category) not in filter_set
        ]
        filtered_count = original_count - len(answer_results)
        if filtered_count > 0:
            console.print(
                f"[dim]🔍 Filtered out {filtered_count} questions from categories: "
                f"{', '.join(sorted(filter_set))}[/dim]"
            )

    if (
        args.smoke
        or args.from_conv != 0
        or args.to_conv is not None
        or args.locomo_sample
    ):
        console.print(
            "[yellow]⚠️  Evaluate-only mode skips dataset loading; "
            "--smoke/--from-conv/--to-conv/--locomo-sample are not applied.[/yellow]"
        )

    return answer_results


def save_evaluate_only_report(output_dir: Path, system_name: str, eval_result, elapsed_time: float, console):
    """Write the standard report.txt without constructing an adapter or pipeline."""
    report_lines = [
        "=" * 60,
        "📊 Evaluation Report",
        "=" * 60,
        "",
        f"System: {system_name}",
        f"Time Elapsed: {elapsed_time:.2f}s",
        "",
        f"Total Questions: {eval_result.total_questions}",
        f"Correct: {eval_result.correct}",
        f"Accuracy: {eval_result.accuracy:.2%}",
        "",
    ]
    task_scores = (eval_result.metadata or {}).get("task_scores", {})
    if task_scores:
        report_lines[9] = f"Accuracy (MEME Overall): {eval_result.accuracy:.2%}"
        report_lines.extend(["Task Accuracy (official MEME scoring):"])
        for task in ("ER", "Agg", "Tr", "Del", "Cas", "Abs"):
            score = task_scores.get(task, {})
            report_lines.append(
                f"  {task}: {score.get('pass', 0):.2f}/{score.get('total', 0)} "
                f"({score.get('accuracy', 0):.2%})"
            )
        report_lines.append("")
    report_lines.append("=" * 60)
    report_text = "\n".join(report_lines)

    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    console.print("\n" + report_text, style="bold green")


async def run_evaluate_only(args, dataset_config: dict, system_config: dict, output_dir: Path, console):
    """Run Evaluate directly from answer_results.json without loading dataset/adapter/LLM provider."""
    import time

    start_time = time.time()
    console.print("\n[bold cyan]Evaluate-only mode[/bold cyan]")
    console.print("  Loading only answer_results.json and evaluator configuration")

    answer_results, _answer_data = load_saved_answer_results(output_dir)
    answer_results = apply_evaluate_only_filters(answer_results, dataset_config, args, console)
    guard_artifacts(
        output_dir, "evaluation",
        {"answers": answer_results, "config": public_config(dataset_config["evaluation"]),
         "prompts": prompt_fingerprint()},
        EVALUATION_ARTIFACTS,
    )
    console.print(f"  ✅ Loaded {len(answer_results)} answers from: {output_dir / 'answer_results.json'}")

    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    if hasattr(evaluator, "set_checkpoint_path"):
        evaluator.set_checkpoint_path(output_dir / "judge_progress.json")
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")

    eval_result = await evaluator.evaluate(answer_results)
    eval_data = eval_result_to_dict(eval_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)

    save_evaluate_only_report(
        output_dir=output_dir,
        system_name=system_config.get("name", args.system),
        eval_result=eval_result,
        elapsed_time=time.time() - start_time,
        console=console,
    )


async def run_answer_fast_path(
    args,
    dataset,
    dataset_config: dict,
    system_config: dict,
    output_dir: Path,
    console,
):
    """Run Answer, and optionally Evaluate, without constructing the system adapter."""
    import time
    from evaluation.src.core.stages.answer_stage import run_answer_stage
    from evaluation.src.utils.checkpoint import CheckpointManager
    from evaluation.src.utils.logger import setup_logger

    start_time = time.time()
    console.print("\n[bold cyan]Answer fast path[/bold cyan]")
    console.print("  Loading saved search_results.json; skipping system adapter initialization")

    dataset = apply_fast_path_dataset_scope(dataset, args, console)
    dataset = apply_fast_path_category_filter(dataset, dataset_config, console)
    search_results, search_data = load_saved_search_results(output_dir)
    search_results = align_search_results_to_qa_pairs(search_results, dataset.qa_pairs)
    guard_artifacts(
        output_dir, "answer",
        {"questions": dataset.qa_pairs, "search_results": search_results,
         "config": public_config(system_config), "prompts": prompt_fingerprint()},
        ANSWER_ARTIFACTS,
    )
    console.print(
        f"  ✅ Loaded {len(search_results)} aligned search results from: "
        f"{output_dir / 'search_results.json'}"
    )

    if system_config.get("adapter") == "full_context":
        from evaluation.src.adapters.full_context_adapter import FullContextAdapter
        adapter = FullContextAdapter(system_config)
    else:
        adapter = AnswerOnlyAdapter(system_config)
    checkpoint = CheckpointManager(output_dir, run_name="default")
    logger = setup_logger(output_dir / "evaluation.log")

    try:
        answer_results = await run_answer_stage(
            adapter=adapter,
            qa_pairs=dataset.qa_pairs,
            search_results=search_results,
            checkpoint_manager=checkpoint,
            logger=logger,
        )
    finally:
        if system_config.get("adapter") == "full_context":
            await adapter.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    answer_data = [answer_result_to_dict(result) for result in answer_results]
    with open(output_dir / "answer_results.json", "w", encoding="utf-8") as f:
        json.dump(answer_data, f, indent=2, ensure_ascii=False, default=str)

    completed_stages = {"answer"}
    checkpoint.save_checkpoint(
        completed_stages,
        search_results=search_data,
        answer_results=answer_data,
    )
    console.print(f"  ✅ Saved answers to: {output_dir / 'answer_results.json'}")

    if "evaluate" not in args.stages:
        return

    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    if hasattr(evaluator, "set_checkpoint_path"):
        evaluator.set_checkpoint_path(output_dir / "judge_progress.json")
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")
    eval_result = await evaluator.evaluate(answer_results)
    eval_data = eval_result_to_dict(eval_result)

    with open(output_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)

    completed_stages.add("evaluate")
    checkpoint.save_checkpoint(
        completed_stages,
        search_results=search_data,
        answer_results=answer_data,
        eval_results=eval_data,
    )
    save_evaluate_only_report(
        output_dir=output_dir,
        system_name=system_config.get("name", args.system),
        eval_result=eval_result,
        elapsed_time=time.time() - start_time,
        console=console,
    )


def apply_longmemeval_judge_mode(dataset_config: dict, mode: str) -> None:
    """Apply explicit CLI selection after dataset/system overrides."""
    config = dataset_config.setdefault("evaluation", {})
    if mode == "official":
        config["type"] = "longmemeval_official_judge"
        config["num_runs"] = 1
        config.setdefault("llm", {})["model"] = (
            os.environ.get("EVALUATE_MODEL") or "gpt-4o-2024-08-06"
        )
    elif mode == "evermemos":
        config["type"] = "llm_judge"
        config["prompt_key"] = "llm_judge"
        config["num_runs"] = 3
    else:
        raise ValueError(f"Unknown LongMemEval judge mode: {mode}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Memory System Evaluation Framework")
    parser.add_argument(
        "--lme-judge",
        choices=["evermemos", "official"],
        default=None,
        help="LongMemEval judge mode. Default: use the dataset configuration (evermemos). "
             "Official uses one judgment and defaults to gpt-4o-2024-08-06; "
             "EVALUATE_MODEL can override the model.",
    )

    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., locomo)"
    )
    parser.add_argument(
        "--system", type=str, required=True, help="System name (e.g., evermemos)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (add, search, answer, evaluate). Default: all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode (process small dataset for quick validation)",
    )
    parser.add_argument(
        "--smoke-messages",
        type=int,
        default=10,
        help="Smoke test: number of messages to process (use 0 for all). Default: 10",
    )
    parser.add_argument(
        "--smoke-questions",
        type=int,
        default=3,
        help="Smoke test: number of questions to test (use 0 for all). Default: 3",
    )
    parser.add_argument(
        "--from-conv",
        type=int,
        default=0,
        help="Starting conversation index to process (inclusive, 0-based). Default: 0",
    )
    parser.add_argument(
        "--to-conv",
        type=int,
        default=None,
        help="Ending conversation index to process (exclusive). Default: None (process all remaining)",
    )
    parser.add_argument(
        "--conv-range",
        type=str,
        default=None,
        help=(
            "Conversation index range to process as START:END (0-based, END exclusive). "
            "Either side may be omitted, e.g. 10:20, 10:, or :20."
        ),
    )
    parser.add_argument(
        "--locomo-sample",
        nargs="+",
        type=int,
        default=None,
        help="Run only the specified LoCoMo sample indices, e.g. --locomo-sample 0 3 5. Default: all samples",
    )
    parser.add_argument(
        "--sample-workers",
        type=int,
        default=None,
        help="Override sample/conversation-level concurrency for systems that support it.",
    )
    parser.add_argument(
        "--search-workers",
        type=int,
        default=None,
        help="Override question-level Search concurrency.",
    )
    parser.add_argument(
        "--post-add-wait-seconds",
        type=float,
        default=None,
        help="Override the fixed wait between Add and downstream stages.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name/version for distinguishing multiple runs (e.g., 'v1', 'baseline', '20241104')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/{dataset}-{system}[-{run_name}]",
    )
    parser.add_argument(
        "--only-category",
        nargs="+",
        default=None,
        help=(
            "Fast-path only: answer only categories by their prefix, e.g. "
            "--only-category Abs matches Abs_hop1."
        ),
    )
    parser.add_argument(
        "--clean-groups",
        action="store_true",
        help="Before Add stage, clear database data for the groups (group_id=conversation_id) involved in this run. "
             "Useful for debugging to avoid polluted data.",
    )
    parser.add_argument(
        "--mab-split",
        type=str,
        default=None,
        help=(
            "MemoryAgentBench top-level HF split. "
            "Examples: Accurate_Retrieval, Test_Time_Learning, "
            "Long_Range_Understanding, Conflict_Resolution."
        ),
    )
    parser.add_argument(
        "--mab-source",
        nargs="+",
        action="append",
        default=None,
        help=(
            "MemoryAgentBench source/task name(s) inside the selected split. "
            "Accepts multiple values and may be repeated, for example "
            "--mab-source eventqa_full 'longmemeval_s*'."
        ),
    )
    parser.add_argument(
        "--mab-disable-official-chunking",
        action="store_true",
        help=(
            "Disable official MemoryAgentBench context chunking before ingestion. "
            "By default, long contexts are split with the benchmark's "
            "sentence-preserving token-budget chunker before being added."
        ),
    )
    parser.add_argument(
        "--mab-chunk-strategy",
        type=str,
        default="source_aware",
        choices=["source_aware", "official", "token", "paragraph", "fact", "turn", "none"],
        help=(
            "MemoryAgentBench context chunking strategy. Default source_aware uses "
            "the per-source best setting from Memos runs comparing official/4096 "
            "against the newer source-aware converter."
        ),
    )
    parser.add_argument(
        "--mab-chunk-size",
        type=int,
        default=None,
        help="Override MemoryAgentBench chunk token budget. Default depends on --mab-chunk-strategy/source.",
    )
    parser.add_argument(
        "--mab-chunk-overlap",
        type=int,
        default=None,
        help="Override MemoryAgentBench chunk token overlap. Default depends on --mab-chunk-strategy/source.",
    )
    parser.add_argument(
        "--mab-report-accuracy",
        action="store_true",
        help=(
            "Print all MemoryAgentBench accuracies for one run name and exit. "
            "Use with --dataset memoryagentbench --system <system> --run-name <name>."
        ),
    )
    parser.add_argument(
        "--mab-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--context-window", type=int, help="Full context total token window, including prompt and reserved output")
    args = parser.parse_args()
    if args.dataset == "lme":
        args.dataset = "longmemeval"
    if args.lme_judge is not None and args.dataset != "longmemeval":
        parser.error("--lme-judge requires --dataset longmemeval")
    if args.context_window is not None and args.context_window <= 0:
        parser.error("--context-window must be positive")
    if args.context_window is not None and args.system != "full_context":
        parser.error("--context-window currently requires --system full_context")
    if args.system == "full_context":
        args.mab_disable_official_chunking = True

    if args.stages:
        invalid_stages = [stage for stage in args.stages if stage not in VALID_STAGES]
        if invalid_stages:
            parser.error(
                "--stages contains unknown value(s): "
                + ", ".join(invalid_stages)
                + f". Valid stages: {', '.join(sorted(VALID_STAGES))}"
            )

    evaluate_only = is_evaluate_only(args.stages)
    answer_fast_path = is_answer_fast_path(args.stages)

    if args.conv_range is not None:
        if args.from_conv != 0 or args.to_conv is not None:
            parser.error("--conv-range cannot be combined with --from-conv or --to-conv")
        try:
            range_start, range_end = args.conv_range.split(":", 1)
            args.from_conv = int(range_start) if range_start else 0
            args.to_conv = int(range_end) if range_end else None
        except ValueError:
            parser.error("--conv-range must use START:END format, e.g. 10:20")

    if args.locomo_sample and (args.from_conv != 0 or args.to_conv is not None):
        parser.error(
            "--locomo-sample cannot be combined with --from-conv, --to-conv, or --conv-range"
        )

    console = get_console()
    if args.mab_report_accuracy:
        sys.exit(report_memoryagentbench_accuracies(args, console))

    memoryagentbench_plan = resolve_memoryagentbench_plan(args, parser)
    if (
        memoryagentbench_plan
        and not args.mab_worker
        and (len(memoryagentbench_plan) > 1 or args.system == "memora")
    ):
        exit_code = run_memoryagentbench_batch(args, memoryagentbench_plan, console)
        if exit_code:
            sys.exit(exit_code)
        return
    if len(memoryagentbench_plan) == 1:
        args.mab_split, args.mab_source = memoryagentbench_plan[0]
        # Resolve defaults only after selecting one source. A batch parent must
        # not propagate a raw-context default to fact-consolidation workers.
        if args.system == "rag" and args.mab_chunk_strategy == "source_aware":
            if args.mab_source in {
                "factconsolidation_sh_262k", "factconsolidation_mh_262k",
            }:
                args.mab_chunk_strategy = "fact"
            else:
                args.mab_disable_official_chunking = True

    # Load configurations
    setup_environment(load_env_file_name=".env", check_env_var="LLM_API_KEY")
    console.print("\n[bold cyan]Loading configurations...[/bold cyan]")

    evaluation_root = Path(__file__).parent

    # Load dataset configuration
    dataset_config_path = (
        evaluation_root / "config" / "datasets" / f"{args.dataset}.yaml"
    )
    if not dataset_config_path.exists():
        console.print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return

    dataset_config = load_yaml(str(dataset_config_path))
    console.print(f"  ✅ Loaded dataset config: {args.dataset}")

    memoryagentbench_selection = None
    if args.dataset == "memoryagentbench":
        if not args.mab_split:
            console.print(
                "[red]❌ --mab-split is required when --dataset memoryagentbench[/red]"
            )
            return
        if evaluate_only:
            memoryagentbench_selection = memoryagentbench_selection_from_args(args)
            dataset_config["evaluation"]["selected_split"] = args.mab_split
            dataset_config["evaluation"]["selected_source"] = args.mab_source
            console.print(
                f"  📦 MemoryAgentBench evaluate-only target: "
                f"{args.mab_split} / {args.mab_source}"
            )
        else:
            mab_root = evaluation_root / "data" / "memoryagentbench"
            prepared_dir = prepare_memoryagentbench_dataset(
                data_root=mab_root,
                split_name=args.mab_split,
                source_name=args.mab_source,
                official_chunking=not args.mab_disable_official_chunking,
                chunk_strategy=args.mab_chunk_strategy,
                chunk_size=args.mab_chunk_size,
                chunk_overlap=args.mab_chunk_overlap,
            )
            manifest = None
            selected_source = args.mab_source or prepared_dir.name
            if (prepared_dir / "manifest.json").exists():
                import json
                with open(prepared_dir / "manifest.json", "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                selected_source = manifest["source"]
            memoryagentbench_selection = {
                "split": args.mab_split,
                "source": selected_source,
                "data_dir": prepared_dir,
                "profile": source_profile(selected_source),
            }
            dataset_config["data"]["path"] = str(prepared_dir)
            dataset_config["evaluation"]["selected_split"] = args.mab_split
            dataset_config["evaluation"]["selected_source"] = selected_source
            console.print(
                f"  📦 Prepared MemoryAgentBench split/source: "
                f"{args.mab_split} / {selected_source}"
            )
            console.print(
                "  ✂️  MemoryAgentBench chunking: "
                f"{manifest.get('chunk_strategy', args.mab_chunk_strategy) if manifest else args.mab_chunk_strategy} "
                f"size={manifest.get('chunk_size') if manifest else args.mab_chunk_size} "
                f"overlap={manifest.get('chunk_overlap') if manifest else args.mab_chunk_overlap}"
            )

    # Override MEMORY_LANGUAGE from dataset config if specified
    if "memory_language" in dataset_config:
        os.environ["MEMORY_LANGUAGE"] = dataset_config["memory_language"]
        console.print(
            f"  🌐 Memory language: {dataset_config['memory_language']} (from dataset config)"
        )

    # Load system configuration
    system_config_path = evaluation_root / "config" / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        console.print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return

    system_config = load_yaml(str(system_config_path))
    console.print(f"  ✅ Loaded system config: {args.system}")

    # Apply dataset-specific configuration overrides
    if (
        "dataset_overrides" in system_config
        and args.dataset in system_config["dataset_overrides"]
    ):
        overrides = system_config["dataset_overrides"][args.dataset]
        # Deep merge override configurations (supports nested field overrides)
        system_config = deep_merge_config(system_config, overrides)
        console.print(
            f"  🔧 Applied dataset overrides for {args.dataset}: {list(overrides.keys())}"
        )

    if args.context_window is not None:
        system_config.setdefault("answer", {})["context_window"] = args.context_window
    if args.system == "full_context":
        dataset_config.setdefault("data", {})["max_content_length"] = None

    # Apply optional source-specific overrides after the selected
    # MemoryAgentBench source is known.
    if args.dataset == "memoryagentbench" and memoryagentbench_selection:
        selected_source = memoryagentbench_selection.get("source")
        source_overrides = (
            system_config.get("dataset_overrides", {})
            .get("memoryagentbench", {})
            .get("sources", {})
        )
        if selected_source in source_overrides:
            source_override = source_overrides[selected_source]
            system_config = deep_merge_config(system_config, source_override)
            console.print(
                f"Applied MemoryAgentBench source override for {selected_source}: "
                f"{list(source_override.keys())}"
            )

    # Allow system dataset_overrides to replace dataset-specific evaluation
    # settings without changing other datasets/benchmarks.
    evaluation_override = system_config.get("evaluation")
    if isinstance(evaluation_override, dict):
        dataset_config["evaluation"] = deep_merge_config(
            dataset_config.get("evaluation", {}),
            evaluation_override,
        )
        console.print(
            f"  ⚖️  Applied evaluation override for {args.dataset}: "
            f"{list(evaluation_override.keys())}"
        )

    if not apply_env_evaluation_overrides(args.dataset, dataset_config, console):
        return

    if args.lme_judge is not None:
        apply_longmemeval_judge_mode(dataset_config, args.lme_judge)
        console.print(f"  LongMemEval judge mode: {args.lme_judge}")

    output_dir = resolve_output_dir(
        args,
        evaluation_root,
        memoryagentbench_selection=memoryagentbench_selection,
    )

    if evaluate_only:
        try:
            await run_evaluate_only(
                args=args,
                dataset_config=dataset_config,
                system_config=system_config,
                output_dir=output_dir,
                console=console,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]❌ {exc}[/red]")
            raise

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")
        return

    # Load dataset
    console.print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")

    from evaluation.src.core.loaders import load_dataset

    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        # Priority: load from evaluation/data/, fall back to project root
        eval_data_path = evaluation_root / "data" / data_path
        root_data_path = evaluation_root.parent / data_path

        if eval_data_path.exists():
            data_path = eval_data_path
            console.print(f"  📂 Using evaluation/data/{data_path}")
        elif root_data_path.exists():
            data_path = root_data_path
            console.print(f"  📂 Using project root data/{data_path}")
        else:
            console.print(
                f"[red]❌ Data not found in evaluation/data/ or project root data/[/red]"
            )
            return

    # Get max_content_length from dataset config (if specified)
    max_content_length = dataset_config.get("data", {}).get("max_content_length", None)
    if max_content_length:
        console.print(f"  ⚠️  Max content length: {max_content_length} characters")

    # Smart load with auto conversion
    dataset = load_dataset(
        args.dataset,
        str(data_path),
        max_content_length=max_content_length,
        dataset_format=dataset_config.get("data", {}).get("format"),
    )

    console.print(
        f"  ✅ Loaded {len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} QA pairs"
    )

    # Create components
    console.print(f"\n[bold cyan]Initializing components...[/bold cyan]")

    # Add dataset_name to system_config for adapter initialization
    # (Used to determine num_workers based on adapter + dataset combination)
    system_config["dataset_name"] = args.dataset
    system_config["run_name"] = args.run_name or ""
    # Pass CLI switch down to adapter via config (adapters can opt-in)
    system_config["clean_groups"] = bool(args.clean_groups)
    if system_config.get("adapter") in {"memos", "memoryos"}:
        system_config["memos_user_id_prefix"] = args.run_name or ""
    if memoryagentbench_selection:
        system_config["memoryagentbench"] = memoryagentbench_selection
    if args.sample_workers is not None:
        if args.sample_workers < 1:
            console.print("[red]❌ --sample-workers must be >= 1[/red]")
            return
        system_config["sample_workers"] = args.sample_workers
        # Online adapters use num_workers for their conversation-level Add
        # semaphore. Apply the CLI override consistently across adapter types.
        system_config["num_workers"] = args.sample_workers
    if args.search_workers is not None:
        if args.search_workers < 1:
            console.print("[red]--search-workers must be >= 1[/red]")
            return
        system_config.setdefault("search", {})["num_workers"] = args.search_workers
    if args.post_add_wait_seconds is not None:
        if args.post_add_wait_seconds < 0:
            console.print("[red]--post-add-wait-seconds must be >= 0[/red]")
            return
        system_config["post_add_wait_seconds"] = args.post_add_wait_seconds

    if answer_fast_path and system_config.get("runner") != "emem":
        try:
            await run_answer_fast_path(
                args=args,
                dataset=dataset,
                dataset_config=dataset_config,
                system_config=system_config,
                output_dir=output_dir,
                console=console,
            )
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]❌ {exc}[/red]")
            raise

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")
        return

    if system_config.get("runner") == "emem":
        evaluator = create_evaluator(
            dataset_config["evaluation"]["type"], dataset_config["evaluation"]
        )
        console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")

        from evaluation.src.runners.emem_runner import run_emem_evaluation

        await run_emem_evaluation(
            dataset=dataset,
            dataset_config=dataset_config,
            system_config=system_config,
            evaluator=evaluator,
            output_dir=output_dir,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
            from_conv=args.from_conv,
            to_conv=args.to_conv,
            locomo_samples=args.locomo_sample,
            filter_categories=dataset_config.get("evaluation", {}).get("filter_category", []),
            stages=args.stages,
        )

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")
        return

    from evaluation.src.adapters.registry import create_adapter
    from evaluation.src.core.pipeline import Pipeline
    from memory_layer.llm.llm_provider import LLMProvider

    # Create adapter (pass output_dir for persistence)
    adapter = create_adapter(
        system_config["adapter"], system_config, output_dir=output_dir
    )
    console.print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")

    # Create evaluator
    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"], dataset_config["evaluation"]
    )
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")

    # Create LLM Provider for answer generation
    llm_config = system_config.get("llm", {})
    llm_provider = LLMProvider(
        provider_type=llm_config.get("provider", "openai"),
        model=llm_config.get("model"),
        api_key=llm_config.get("api_key"),
        base_url=llm_config.get("base_url"),
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=llm_config.get("max_tokens", 16384),
    )
    console.print(f"  Created LLM provider: {llm_config.get('model')}")

    # Create pipeline
    # Read filter categories from dataset configuration
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])

    pipeline = Pipeline(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        filter_categories=filter_categories,
    )

    console.print(f"  ✅ Created pipeline, output: {output_dir}")
    if filter_categories:
        console.print(f"  📋 Filter categories: {filter_categories}")

    # Run pipeline
    try:
        results = await pipeline.run(
            dataset=dataset,
            stages=args.stages,
            smoke_test=args.smoke,
            smoke_messages=args.smoke_messages,
            smoke_questions=args.smoke_questions,
            from_conv=args.from_conv,
            to_conv=args.to_conv,
            locomo_samples=args.locomo_sample,
        )

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")

    finally:
        # Cleanup resources
        # Clean up adapter session (e.g., aiohttp.ClientSession)
        if hasattr(adapter, 'close') and callable(getattr(adapter, 'close')):
            try:
                await adapter.close()
                console.print("[dim]🧹 Cleaned up adapter resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup adapter resources: {e}[/dim]")

        # Only systems using rerank need cleanup
        systems_need_rerank = ["evermemos"]
        if args.system in systems_need_rerank:
            try:
                from agentic_layer import rerank_service

                reranker = rerank_service.get_rerank_service()
                if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
                    await reranker.close()
                    console.print("[dim]🧹 Cleaned up rerank service resources[/dim]")
            except Exception as e:
                # Cleanup failure doesn't affect main process
                console.print(f"[dim]⚠️  Failed to cleanup rerank resources: {e}[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:
        if getattr(error, "process_recycle_requested", False):
            print(f"Memora controlled worker recycle: {error}", flush=True)
            sys.exit(getattr(error, "exit_code", MEMORA_RECYCLE_EXIT_CODE))
        raise
