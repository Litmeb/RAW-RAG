"""
MemoryAgentBench dataset preparation helpers.

This module mirrors the benchmark's official Hugging Face loading pattern:
- top-level HF split: Accurate_Retrieval / Test_Time_Learning /
  Long_Range_Understanding / Conflict_Resolution
- per-task selection is done via metadata.source

Prepared data is cached under evaluation/data/memoryagentbench/{split}/{source}/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


HF_DATASET_NAME = "ai-hyz/MemoryAgentBench"
HF_REVISION = "main"

MAB_SYSTEM_MESSAGE = (
    "You are a helpful assistant that can read the context and memorize it for "
    "future retrieval."
)

MAB_TASK_TEMPLATES = {
    "ruler_qa": {
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
        "query": {
            "rag_agent": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
            "agentic_memory_agent": "Search Archival Memory and answer my question. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
        },
    },
    "longmemeval": {
        "memorize": "Dialogue between User and Assistant \n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
        "query": {
            "rag_agent": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
            "agentic_memory_agent": "Search Archival Memory and answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
        },
    },
    "eventqa": {
        "memorize": "Dialogue between User and Assistant {time_stamp}\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
        "query": {
            "rag_agent": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
            "agentic_memory_agent": "Search Archival Memory, complete the task below:\n\n{question}\n\n The event that happens next is:",
        },
    },
    "in_context_learning": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
        "query": {
            "rag_agent": 'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\nQuestion:{question} \n\n label:',
            "agentic_memory_agent": 'Search Archival Memory and use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\n{question} \n\n label:',
        },
    },
    "recsys_redial": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
        "query": {
            "rag_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
            "agentic_memory_agent": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Search Archival Memory, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
        },
    },
    "infbench_sum": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "rag_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
            "agentic_memory_agent": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
        },
    },
    "detective_qa": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
        "query": {
            "rag_agent": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
            "agentic_memory_agent": "Search Archival Memory and answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
        },
    },
    "factconsolidation": {
        "memorize": "Dialogue between User and Assistant {time_stamp} \n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
        "query": {
            "rag_agent": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
            "agentic_memory_agent": "Pretend you are a knowledge management system. Each fact in the  Archival Memory is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the Archival Memory by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Archival Memory] \n\n Question: Based on the Archival Memory, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the  Archival Memory, {question} \nAnswer:",
        },
    },
}

SUPPORTED_SPLITS = {
    "Accurate_Retrieval": [
        "eventqa_full",
        "eventqa_65536",
        "eventqa_131072",
        "longmemeval_s*",
        "ruler_qa1_197K",
        "ruler_qa2_421K",
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
        "factconsolidation_mh_6k",
        "factconsolidation_mh_32k",
        "factconsolidation_mh_64k",
        "factconsolidation_mh_262k",
        "factconsolidation_sh_6k",
        "factconsolidation_sh_32k",
        "factconsolidation_sh_64k",
        "factconsolidation_sh_262k",
    ],
}


def sanitize_path_component(value: str) -> str:
    """Convert split/source labels into filesystem-safe directory names."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "default"


def available_sources(split_name: str) -> List[str]:
    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported MemoryAgentBench split '{split_name}'. "
            f"Available: {sorted(SUPPORTED_SPLITS)}"
        )
    return list(SUPPORTED_SPLITS[split_name])


def source_profile(source: str) -> Dict[str, str]:
    """Return prompt/evaluation routing for a MemoryAgentBench source."""
    if source.startswith("longmemeval"):
        return {
            "prompt_key": "answer_prompt_mab_longmemeval",
            "evaluation_family": "longmemeval",
        }
    if source.startswith("eventqa"):
        return {
            "prompt_key": "answer_prompt_mab_eventqa",
            "evaluation_family": "substring_exact_match",
        }
    if source.startswith("ruler_qa"):
        return {
            "prompt_key": "answer_prompt_mab_ruler_qa",
            "evaluation_family": "substring_exact_match",
        }
    if source.startswith("factconsolidation"):
        return {
            "prompt_key": "answer_prompt_mab_factconsolidation",
            "evaluation_family": "substring_exact_match",
        }
    if source.startswith("icl_"):
        return {
            "prompt_key": "answer_prompt_mab_in_context_learning",
            "evaluation_family": "icl_exact_match",
        }
    if source.startswith("recsys_"):
        return {
            "prompt_key": "answer_prompt_mab_recsys_redial",
            "evaluation_family": "recsys_recall",
        }
    if source == "detective_qa":
        return {
            "prompt_key": "answer_prompt_mab_detective_qa",
            "evaluation_family": "detective_qa",
        }
    if source.startswith("infbench_sum"):
        return {
            "prompt_key": "answer_prompt_mab_infbench_sum",
            "evaluation_family": "infbench_sum",
        }
    raise ValueError(f"Unsupported MemoryAgentBench source '{source}'")


def source_task_name(source: str) -> str:
    """Return the official MemoryAgentBench template family for a source."""
    if source.startswith("longmemeval"):
        return "longmemeval"
    if source.startswith("eventqa"):
        return "eventqa"
    if source.startswith("ruler_qa"):
        return "ruler_qa"
    if source.startswith("factconsolidation"):
        return "factconsolidation"
    if source.startswith("icl_"):
        return "in_context_learning"
    if source.startswith("recsys_"):
        return "recsys_redial"
    if source == "detective_qa":
        return "detective_qa"
    if source.startswith("infbench_sum"):
        return "infbench_sum"
    raise ValueError(f"Unsupported MemoryAgentBench source '{source}'")


MAB_SOURCE_AWARE_CHUNKING = {
    "eventqa_full": ("paragraph", 1024, 128),
    "longmemeval_s*": ("turn", 1024, 0),
    "ruler_qa1_197K": ("token", 512, 64),
    "ruler_qa2_421K": ("token", 512, 64),
    "recsys_redial_full": ("official", 1024, 0),
    "icl_banking77_5900shot_balance": ("official", 1024, 0),
    "icl_trec_coarse_6600shot_balance": ("official", 1024, 0),
    "factconsolidation_mh_262k": ("official", 1024, 0),
    "icl_trec_fine_6400shot_balance": ("official", 1024, 0),
    "icl_clinic150_7050shot_balance": ("official", 1024, 0),
    "icl_nlu_8296shot_balance": ("official", 1024, 0),
    "factconsolidation_sh_262k": ("official", 1024, 0),
}

MAB_OFFICIAL_4096_CHUNKING = {
    "detective_qa",
    "infbench_sum_eng_shots2"
}


def default_chunking_for_source(source: str) -> Tuple[str, int, int]:
    """Return Memos run best-of-two MemoryAgentBench chunking defaults."""
    if source in MAB_SOURCE_AWARE_CHUNKING:
        return MAB_SOURCE_AWARE_CHUNKING[source]
    if source in MAB_OFFICIAL_4096_CHUNKING:
        return "official", 4096, 0

    if source.startswith("eventqa"):
        return "paragraph", 1024, 128
    if source.startswith("factconsolidation"):
        return "fact", 0, 0
    if source.startswith("longmemeval"):
        return "turn", 1024, 0
    if source.startswith("icl_") or source == "detective_qa" or source.startswith("infbench_sum"):
        return "official", 4096, 0
    return "official", 1024, 0


def resolve_chunking_options(
    source: str,
    *,
    chunk_strategy: str = "source_aware",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    official_chunking: bool = True,
) -> Tuple[str, int, int]:
    """Resolve CLI chunking overrides into concrete converter settings."""
    if not official_chunking:
        return "none", 0, 0

    if chunk_strategy == "source_aware":
        resolved_strategy, default_size, default_overlap = default_chunking_for_source(source)
    else:
        resolved_strategy = chunk_strategy
        default_size, default_overlap = {
            "official": (1024, 0),
            "token": (1024, 0),
            "paragraph": (1024, 128),
            "fact": (0, 0),
            "turn": (1024, 0),
            "none": (0, 0),
        }.get(chunk_strategy, (4096, 0))

    resolved_size = default_size if chunk_size is None else int(chunk_size)
    resolved_overlap = default_overlap if chunk_overlap is None else int(chunk_overlap)
    return resolved_strategy, resolved_size, resolved_overlap


def format_memoryagentbench_memorize_context(
    source: str, context: str, *, time_stamp: str = ""
) -> str:
    """Format context with the official MemoryAgentBench memorize template."""
    task_name = source_task_name(source)
    template = MAB_TASK_TEMPLATES[task_name]["memorize"]
    return template.format(context=context, time_stamp=time_stamp)


def format_memoryagentbench_query(
    source: str, question: str, *, agent_kind: str = "rag_agent"
) -> str:
    """Format a question with the official MemoryAgentBench query template."""
    task_name = source_task_name(source)
    query_templates = MAB_TASK_TEMPLATES[task_name]["query"]
    template = query_templates.get(agent_kind) or query_templates["rag_agent"]
    # Official templates contain a literal "{label}" placeholder in instructions.
    return template.replace("{label}", "{{label}}").format(question=question)


def prepare_memoryagentbench_dataset(
    *,
    data_root: Path,
    split_name: str,
    source_name: Optional[str],
    official_chunking: bool = True,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    chunk_strategy: str = "source_aware",
) -> Path:
    """
    Download and cache a MemoryAgentBench split/source locally.

    Returns:
        Path to the source-specific cache directory.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    valid_sources = available_sources(split_name)
    if source_name is None:
        if len(valid_sources) != 1:
            raise ValueError(
                f"MemoryAgentBench split '{split_name}' contains multiple heterogeneous sources: "
                f"{valid_sources}. Please pass --mab-source to select one source."
            )
        source_name = valid_sources[0]

    if source_name not in valid_sources:
        raise ValueError(
            f"Source '{source_name}' is not valid for split '{split_name}'. "
            f"Available sources: {valid_sources}"
        )

    source_dir = (
        data_root
        / sanitize_path_component(split_name)
        / sanitize_path_component(source_name)
    )
    source_dir.mkdir(parents=True, exist_ok=True)

    raw_path = source_dir / "memoryagentbench_raw.json"
    manifest_path = source_dir / "manifest.json"

    if not raw_path.exists():
        dataset = load_dataset(HF_DATASET_NAME, split=split_name, revision=HF_REVISION)
        filtered = dataset.filter(
            lambda sample: sample.get("metadata", {}).get("source", "") == source_name
        )
        if len(filtered) == 0:
            raise ValueError(
                f"No samples found for MemoryAgentBench split='{split_name}', source='{source_name}'."
            )
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(filtered.to_list(), f, ensure_ascii=False, indent=2)

    if source_name.startswith("recsys_"):
        entity2id_path = source_dir / "entity2id.json"
        if not entity2id_path.exists():
            downloaded = hf_hub_download(
                repo_id=HF_DATASET_NAME,
                repo_type="dataset",
                filename="entity2id.json",
                revision=HF_REVISION,
            )
            entity2id_path.write_text(
                Path(downloaded).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    profile = source_profile(source_name)
    resolved_strategy, resolved_chunk_size, resolved_chunk_overlap = resolve_chunking_options(
        source_name,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        official_chunking=official_chunking,
    )
    manifest = {
        "hf_dataset": HF_DATASET_NAME,
        "hf_revision": HF_REVISION,
        "split": split_name,
        "source": source_name,
        "official_chunking": bool(official_chunking),
        "chunk_strategy": resolved_strategy,
        "chunk_size": int(resolved_chunk_size),
        "chunk_overlap": int(resolved_chunk_overlap),
        "profile": profile,
        "raw_file": raw_path.name,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return source_dir
