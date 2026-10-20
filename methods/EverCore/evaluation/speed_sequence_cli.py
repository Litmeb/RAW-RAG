"""Measure sequential memory-ingestion latency for BEAM-shaped datasets.

Each BEAM message is submitted in its own incremental-ingest call.  The backend
is created once and the conversation id is kept stable, so memory systems see an
increasingly long conversation while every new item has an independent latency
measurement.  Backends without a specialized incremental primitive fall back to
a single-message ``adapter.add`` call.

Example:
    python -m evaluation.speed_sequence_cli --system mem0_local \
        --run-name beam10m-sequence-20260902 --conv-range 0:1
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


@dataclass(frozen=True)
class BeamItem:
    """One message in the deterministic order used by a BEAM conversation."""

    message: Any
    plan: str
    batch_number: Any
    turn_index: int
    message_index: int
    source_message_id: Any
    role: str


@dataclass(frozen=True)
class MemMachineAsyncWaitConfig:
    """Docker PostgreSQL barrier for MemMachine's semantic ingestion worker."""

    container: str = "memmachine-postgres"
    database: str = "memmachine"
    user: str = "memmachine"
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 3600.0


CSV_FIELDS = [
    "conversation_file",
    "conversation_id",
    "sequence_index",
    "global_sequence_index",
    "plan",
    "batch_number",
    "turn_index",
    "message_index",
    "source_message_id",
    "role",
    "content_sha256",
    "content_chars",
    "content_tokens",
    "cumulative_messages",
    "cumulative_chars",
    "cumulative_tokens",
    "add_seconds",
    "cumulative_add_seconds",
    "global_cumulative_add_seconds",
    "wall_elapsed_seconds",
    "status",
    "error",
    "ingest_metadata",
]


def memmachine_async_wait_config(system_config: dict) -> MemMachineAsyncWaitConfig:
    raw = system_config.get("speed_sequence_async_wait") or {}
    config = MemMachineAsyncWaitConfig(
        container=str(raw.get("docker_container", "memmachine-postgres")),
        database=str(raw.get("database", "memmachine")),
        user=str(raw.get("user", "memmachine")),
        poll_interval_seconds=float(raw.get("poll_interval_seconds", 1.0)),
        timeout_seconds=float(raw.get("timeout_seconds", 3600.0)),
    )
    if config.poll_interval_seconds <= 0:
        raise ValueError("speed_sequence_async_wait.poll_interval_seconds must be > 0")
    if config.timeout_seconds <= 0:
        raise ValueError("speed_sequence_async_wait.timeout_seconds must be > 0")
    return config


def memmachine_episode_ids(add_result: Any) -> list[str]:
    """Extract the server-generated episode ids exposed by MemMachineAdapter.add."""
    if not isinstance(add_result, dict):
        return []
    episode_ids: list[str] = []
    for result in add_result.get("add_results") or []:
        if not isinstance(result, dict):
            continue
        episode_ids.extend(str(value) for value in result.get("episode_ids") or [])
    return list(dict.fromkeys(episode_ids))


async def memmachine_pending_ingestion_count(
    episode_ids: list[str], config: MemMachineAsyncWaitConfig
) -> int:
    """Count unfinished semantic-ingestion rows from inside PostgreSQL's container."""
    quoted_ids = ", ".join(
        "'" + episode_id.replace("'", "''") + "'" for episode_id in episode_ids
    )
    sql = (
        "SELECT count(*) FROM set_ingested_history "
        f"WHERE history_id IN ({quoted_ids}) AND ingested IS NOT TRUE;"
    )
    process = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        config.container,
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        config.user,
        "-d",
        config.database,
        "-Atc",
        sql,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Cannot inspect MemMachine async ingestion in Docker: {detail}"
        )
    try:
        return int(stdout.decode("utf-8").strip())
    except ValueError as exc:
        raise RuntimeError(
            "MemMachine async-ingestion query returned a non-integer result: "
            f"{stdout.decode('utf-8', errors='replace').strip()!r}"
        ) from exc


async def wait_for_memmachine_async_ingestion(
    add_result: Any, config: MemMachineAsyncWaitConfig
) -> dict[str, Any]:
    """Wait until every semantic row created by one Add has been ingested."""
    episode_ids = memmachine_episode_ids(add_result)
    if not episode_ids:
        raise RuntimeError(
            "MemMachine Add did not return episode ids; cannot establish the Docker "
            "async-ingestion completion barrier"
        )

    started = time.perf_counter()
    polls = 0
    while True:
        polls += 1
        pending = await memmachine_pending_ingestion_count(episode_ids, config)
        elapsed = time.perf_counter() - started
        if pending == 0:
            return {
                "barrier": "set_ingested_history",
                "episode_ids": episode_ids,
                "pending": 0,
                "polls": polls,
                "wait_seconds": elapsed,
            }
        if elapsed >= config.timeout_seconds:
            raise TimeoutError(
                "MemMachine semantic ingestion did not finish within "
                f"{config.timeout_seconds:g}s ({pending} row(s) still pending)"
            )
        await asyncio.sleep(
            min(config.poll_interval_seconds, config.timeout_seconds - elapsed)
        )

def deep_merge_config(base: dict, override: dict) -> dict:
    """Return a recursive configuration merge without modifying either input."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(result[key], value)
        else:
            result[key] = value
    return result


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse the date formats found in BEAM, returning None when unavailable."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%B-%d-%Y",
        "%B %d, %Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        with suppress(ValueError):
            return datetime.strptime(text, fmt)
    with suppress(ValueError):
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    return None


def iter_beam_items(document: dict) -> Iterator[BeamItem]:
    """Flatten BEAM's chat -> plan -> batch -> turn -> message hierarchy."""
    from evaluation.src.core.data_models import Message

    for chat_entry in document.get("chat", []):
        if not isinstance(chat_entry, dict):
            raise ValueError("BEAM chat entries must be objects")
        for plan_name, batches in chat_entry.items():
            if batches is None:
                continue
            if not isinstance(batches, list):
                raise ValueError(f"BEAM {plan_name} batches must be a list")
            for batch in batches:
                batch_number = batch.get("batch_number")
                batch_time = batch.get("time_anchor")
                for turn_index, turn in enumerate(batch.get("turns", []), start=1):
                    if not isinstance(turn, list):
                        raise ValueError(f"BEAM {plan_name} turn must be a list")
                    for message_index, raw in enumerate(turn, start=1):
                        content = str(raw.get("content") or "")
                        role = str(raw.get("role") or "user").lower()
                        if role not in {"user", "assistant"}:
                            role = "user"
                        raw_time = raw.get("time_anchor") or batch_time
                        metadata = {
                            "beam_plan": plan_name,
                            "beam_batch_number": batch_number,
                            "beam_turn_index": turn_index,
                            "beam_message_index": message_index,
                            "beam_message_id": raw.get("id"),
                            "beam_index": raw.get("index"),
                            "beam_question_type": raw.get("question_type"),
                            "beam_time_anchor": raw_time,
                        }
                        yield BeamItem(
                            message=Message(
                                sender_id=role,
                                sender_name=role,
                                content=content,
                                timestamp=parse_timestamp(raw_time),
                                metadata=metadata,
                            ),
                            plan=plan_name,
                            batch_number=batch_number,
                            turn_index=turn_index,
                            message_index=message_index,
                            source_message_id=raw.get("id"),
                            role=role,
                        )


def numeric_json_files(data_dir: Path) -> list[Path]:
    """Find a contiguous, one-based set of numbered conversation files."""
    files = [path for path in data_dir.glob("*.json") if path.stem.isdigit()]
    files.sort(key=lambda path: int(path.stem))
    if not files:
        raise ValueError(f"No numeric conversation JSON files found in {data_dir}")
    found = [int(path.stem) for path in files]
    expected = list(range(1, found[-1] + 1))
    if found != expected:
        raise ValueError(
            f"Numbered conversation files in {data_dir} must be contiguous from "
            f"1.json through {found[-1]}.json; found {found}"
        )
    return files


def infer_dataset_label(data_dir: Path) -> str:
    """Return a human-readable label for manifests, output paths, and plots."""
    if data_dir == (REPOSITORY_ROOT / "beam").resolve():
        return "BEAM-10M"
    return data_dir.name.replace("_", "-").upper()


def dataset_slug(label: str) -> str:
    """Convert a display label into a filesystem-safe result prefix."""
    slug = "".join(character.lower() if character.isalnum() else "_" for character in label)
    return "_".join(part for part in slug.split("_") if part) or "sequence"


def parse_conv_range(raw: str, total: int) -> tuple[int, int]:
    """Parse a Python-style START:END conversation slice."""
    try:
        start_text, end_text = raw.split(":", 1)
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else total
    except ValueError as exc:
        raise ValueError("--conv-range must use START:END, for example 0:1") from exc
    if start < 0 or end < 0 or start >= end or end > total:
        raise ValueError(f"--conv-range must satisfy 0 <= START < END <= {total}")
    return start, end


def create_token_counter(model: str):
    """Return a token-count callable, or None when tiktoken is unavailable."""
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except Exception:
        return None


class DurableMeasurementWriter:
    """Append and durably flush both machine-friendly measurement formats."""

    def __init__(self, output_dir: Path):
        self.json_file = (output_dir / "speed_sequence.jsonl").open(
            "x", encoding="utf-8", newline=""
        )
        self.csv_file = (output_dir / "speed_sequence.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()
        self._sync(self.csv_file)

    @staticmethod
    def _sync(file_object) -> None:
        file_object.flush()
        os.fsync(file_object.fileno())

    def write(self, record: dict) -> None:
        self.json_file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        csv_record = {field: record.get(field) for field in CSV_FIELDS}
        if csv_record.get("ingest_metadata") is not None:
            csv_record["ingest_metadata"] = json.dumps(
                csv_record["ingest_metadata"], ensure_ascii=False, default=str
            )
        self.csv_writer.writerow(csv_record)
        self._sync(self.json_file)
        self._sync(self.csv_file)

    def close(self) -> None:
        self.json_file.close()
        self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def build_record(
    *,
    item: BeamItem,
    conversation_file: Path,
    conversation_id: str,
    sequence_index: int,
    global_sequence_index: int,
    content_tokens: Optional[int],
    cumulative_messages: int,
    cumulative_chars: int,
    cumulative_tokens: Optional[int],
    add_seconds: float,
    cumulative_add_seconds: float,
    global_cumulative_add_seconds: float,
    wall_elapsed_seconds: float,
    status: str,
    error: str = "",
    ingest_metadata: Any = None,
) -> dict:
    content = item.message.content
    return {
        "conversation_file": conversation_file.name,
        "conversation_id": conversation_id,
        "sequence_index": sequence_index,
        "global_sequence_index": global_sequence_index,
        "plan": item.plan,
        "batch_number": item.batch_number,
        "turn_index": item.turn_index,
        "message_index": item.message_index,
        "source_message_id": item.source_message_id,
        "role": item.role,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "content_tokens": content_tokens,
        "cumulative_messages": cumulative_messages,
        "cumulative_chars": cumulative_chars,
        "cumulative_tokens": cumulative_tokens,
        "add_seconds": add_seconds,
        "cumulative_add_seconds": cumulative_add_seconds,
        "global_cumulative_add_seconds": global_cumulative_add_seconds,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "status": status,
        "error": error,
        "ingest_metadata": ingest_metadata,
    }


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Stream completed measurements without retaining the full long run."""
    with path.open(encoding="utf-8") as file_object:
        for line in file_object:
            if line.strip():
                yield json.loads(line)


def create_plot(
    output_dir: Path, records: Iterable[dict], dataset_label: str = "BEAM-10M"
) -> bool:
    """Write a compact per-conversation latency curve, if matplotlib exists."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    grouped: dict[str, dict[str, Any]] = {}
    for row in records:
        if row.get("status") == "ok":
            group = grouped.setdefault(
                str(row["conversation_id"]), {"seen": 0, "stride": 1, "rows": []}
            )
            group["seen"] += 1
            if group["seen"] % group["stride"] == 0:
                group["rows"].append(row)
            if len(group["rows"]) > 4000:
                group["rows"] = group["rows"][::2]
                group["stride"] *= 2
    if not grouped:
        return False

    fig, axis = plt.subplots(figsize=(11, 6))
    for conversation_id, group in grouped.items():
        sampled = group["rows"]
        use_tokens = all(row.get("cumulative_tokens") is not None for row in sampled)
        x_key = "cumulative_tokens" if use_tokens else "cumulative_chars"
        axis.plot(
            [row[x_key] for row in sampled],
            [row["add_seconds"] for row in sampled],
            linewidth=0.8,
            alpha=0.8,
            label=conversation_id,
        )
    axis.set_xlabel("Cumulative tokens (characters when tokenizer is unavailable)")
    axis.set_ylabel("Single-item ingest time (seconds)")
    axis.set_title(f"{dataset_label} sequential memory-ingestion latency")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "speed_sequence.png", dpi=180)
    plt.close(fig)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequential per-message Add timing for BEAM-shaped datasets"
    )
    parser.add_argument("--system", required=True, help="System config name")
    parser.add_argument("--run-name", required=True, help="Unique run name")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPOSITORY_ROOT / "beam",
        help="Directory containing contiguous 1.json ... N.json conversations",
    )
    parser.add_argument(
        "--conv-range", default=":", help="0-based file slice, END exclusive (default: all)"
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Optional label used in manifests, result paths, and plot titles",
    )
    parser.add_argument(
        "--max-items-per-conversation",
        type=int,
        default=None,
        help="Optional smoke-test limit; omitted means every message",
    )
    parser.add_argument(
        "--tokenizer-model", default="gpt-4", help="tiktoken model used for x-axis lengths"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print one progress line per N ingested messages (default: 100)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not create the final PNG")
    return parser


async def run(args: argparse.Namespace) -> Path:
    """Run one sequential timing experiment and return its output directory."""
    from common_utils.load_env import setup_environment

    setup_environment(load_env_file_name=".env", check_env_var="LLM_API_KEY")

    from evaluation.src.adapters.registry import create_adapter
    from evaluation.src.core.data_models import Conversation
    from evaluation.src.utils.config import load_yaml
    from evaluation.src.utils.logger import get_console

    console = get_console()
    data_dir = args.data_dir.resolve()
    all_files = numeric_json_files(data_dir)
    dataset_label = args.dataset_label or infer_dataset_label(data_dir)
    range_start, range_end = parse_conv_range(args.conv_range, len(all_files))
    selected_files = all_files[range_start:range_end]
    if args.max_items_per_conversation is not None and args.max_items_per_conversation < 1:
        raise ValueError("--max-items-per-conversation must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")

    system_config_path = PROJECT_ROOT / "evaluation" / "config" / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        raise FileNotFoundError(f"System config not found: {system_config_path}")
    system_config = load_yaml(str(system_config_path))
    beam_override = system_config.get("dataset_overrides", {}).get("beam_10m", {})
    if beam_override:
        system_config = deep_merge_config(system_config, beam_override)
    adapter_name = system_config.get("adapter")
    runner_name = system_config.get("runner")
    if not adapter_name and runner_name != "emem":
        raise ValueError(
            f"System {args.system!r} uses a dedicated runner rather than an Add adapter; "
            "it cannot perform incremental single-message ingestion."
        )

    # This benchmark is intentionally sequential. Mem0's normal checkpoint is
    # batch-oriented and would otherwise treat later one-message calls as done.
    system_config["dataset_name"] = "beam_10m"
    system_config["run_name"] = args.run_name
    system_config["sample_workers"] = 1
    system_config["num_workers"] = 1
    system_config["clean_groups"] = False
    if adapter_name in {"memos", "memoryos"}:
        system_config["memos_user_id_prefix"] = args.run_name
    if isinstance(system_config.get("add"), dict):
        system_config["add"]["max_concurrent_requests"] = 1
        system_config["add"]["atomic_fact_max_concurrency"] = 1
    if isinstance(system_config.get("add_resume"), dict):
        system_config["add_resume"]["enabled"] = False
    memmachine_wait = (
        memmachine_async_wait_config(system_config)
        if adapter_name == "memmachine"
        else None
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "evaluation"
        / "results"
        / f"{dataset_slug(dataset_label)}-{args.system}-{args.run_name}"
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use a new --run-name."
        )
    output_dir.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "dataset": dataset_label,
        "system": args.system,
        "adapter": adapter_name,
        "runner": runner_name,
        "run_name": args.run_name,
        "data_dir": str(data_dir),
        "conversation_files": [path.name for path in selected_files],
        "conv_range": args.conv_range,
        "max_items_per_conversation": args.max_items_per_conversation,
        "tokenizer_model": args.tokenizer_model,
        "concurrency": 1,
        "add_error_policy": "record failed item and continue with the next item",
        "timing_boundary": (
            "await ingestor.add_incremental_message(single_message_conversation)"
            if adapter_name in {"rag", "evermemos", "memora"} or runner_name == "emem"
            else (
                "await adapter.add(single_message_conversation), then wait for all "
                "returned episode ids to have zero pending set_ingested_history rows"
                if adapter_name == "memmachine"
                else "await adapter.add(single_message_conversation)"
            )
        ),
        "memmachine_async_wait": (
            {
                "barrier": "set_ingested_history",
                "scope": "episode ids returned by each Add",
                "docker_container": memmachine_wait.container,
                "database": memmachine_wait.database,
                "user": memmachine_wait.user,
                "poll_interval_seconds": memmachine_wait.poll_interval_seconds,
                "timeout_seconds": memmachine_wait.timeout_seconds,
            }
            if memmachine_wait is not None
            else None
        ),
        "conversation_namespace": "beam10m_<sha256(run-name)[:10]>_<source-id>",
        "incremental_semantics": {
            "rag": "append token units; re-embed only the mutable overlap tail",
            "evermemos": "append one closed MemCell per message; append embeddings; refresh BM25 IDF",
            "memora": "segment one message and append with stable sequence-addressed store keys",
            "emem": "append one unique EMem session and update the persistent graph",
            "default": "one single-message adapter.add call",
        }.get(str(adapter_name or runner_name), "one single-message adapter.add call"),
        "started_at": datetime.now().astimezone().isoformat(),
        "system_config_path": str(system_config_path),
        "system_config_sha256": hashlib.sha256(system_config_path.read_bytes()).hexdigest(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    token_counter = create_token_counter(args.tokenizer_model)
    if token_counter is None:
        console.print("[yellow]tiktoken is unavailable; cumulative_chars will be used.[/yellow]")

    if runner_name == "emem":
        from evaluation.src.runners.emem_runner import EMemIncrementalIngestor

        adapter = EMemIncrementalIngestor(system_config, output_dir)
    else:
        adapter = create_adapter(adapter_name, system_config, output_dir=output_dir)
    console.print(f"Sequential {dataset_label} Add: {len(selected_files)} conversation(s)")
    console.print(f"Output: {output_dir}")

    global_sequence_index = 0
    global_cumulative_add_seconds = 0.0
    successful_items = 0
    failed_items = 0
    successful_add_seconds = 0.0
    benchmark_start = time.perf_counter()
    try:
        with DurableMeasurementWriter(output_dir) as writer:
            for file_number, conversation_file in enumerate(selected_files, start=1):
                console.print(
                    f"Loading {conversation_file.name} ({file_number}/{len(selected_files)})"
                )
                document = json.loads(conversation_file.read_text(encoding="utf-8"))
                source_conversation_id = str(
                    document.get("conversation_id") or conversation_file.stem
                )
                run_hash = hashlib.sha256(args.run_name.encode("utf-8")).hexdigest()[:10]
                conversation_id = f"beam10m_{run_hash}_{source_conversation_id}"
                if hasattr(adapter, "start_incremental_conversation"):
                    await adapter.start_incremental_conversation(conversation_id)
                cumulative_chars = 0
                cumulative_tokens: Optional[int] = 0 if token_counter else None
                cumulative_add_seconds = 0.0
                sequence_index = 0
                for item in iter_beam_items(document):
                    sequence_index += 1
                    if (
                        args.max_items_per_conversation is not None
                        and sequence_index > args.max_items_per_conversation
                    ):
                        break
                    global_sequence_index += 1
                    content = item.message.content
                    content_tokens = token_counter(content) if token_counter else None
                    cumulative_chars += len(content)
                    if cumulative_tokens is not None and content_tokens is not None:
                        cumulative_tokens += content_tokens

                    single_item_conversation = Conversation(
                        conversation_id=conversation_id,
                        messages=[item.message],
                        metadata={
                            "speaker_a": "user",
                            "speaker_b": "assistant",
                            "beam_source_file": conversation_file.name,
                        },
                    )
                    error_text = ""
                    status = "ok"
                    ingest_metadata = None
                    memmachine_wait_started = False
                    caught_error: Optional[Exception] = None
                    add_start = time.perf_counter()
                    try:
                        if hasattr(adapter, "add_incremental_message"):
                            ingest_metadata = await adapter.add_incremental_message(
                                single_item_conversation,
                                sequence_index=sequence_index,
                                output_dir=output_dir,
                            )
                        else:
                            ingest_metadata = await adapter.add(
                                conversations=[single_item_conversation],
                                output_dir=output_dir,
                                checkpoint_manager=None,
                            )
                            if memmachine_wait is not None:
                                memmachine_wait_started = True
                                wait_metadata = await wait_for_memmachine_async_ingestion(
                                    ingest_metadata, memmachine_wait
                                )
                                ingest_metadata = dict(ingest_metadata or {})
                                ingest_metadata["async_wait"] = wait_metadata
                    except Exception as error:
                        status = "error"
                        error_text = f"{type(error).__name__}: {error}"
                        caught_error = error
                    add_seconds = time.perf_counter() - add_start
                    cumulative_add_seconds += add_seconds
                    global_cumulative_add_seconds += add_seconds
                    record = build_record(
                        item=item,
                        conversation_file=conversation_file,
                        conversation_id=conversation_id,
                        sequence_index=sequence_index,
                        global_sequence_index=global_sequence_index,
                        content_tokens=content_tokens,
                        cumulative_messages=sequence_index,
                        cumulative_chars=cumulative_chars,
                        cumulative_tokens=cumulative_tokens,
                        add_seconds=add_seconds,
                        cumulative_add_seconds=cumulative_add_seconds,
                        global_cumulative_add_seconds=global_cumulative_add_seconds,
                        wall_elapsed_seconds=time.perf_counter() - benchmark_start,
                        status=status,
                        error=error_text,
                        ingest_metadata=ingest_metadata,
                    )
                    writer.write(record)
                    if status == "ok":
                        successful_items += 1
                        successful_add_seconds += add_seconds
                    else:
                        failed_items += 1
                        if memmachine_wait_started:
                            console.print(
                                "MemMachine async-ingestion barrier failed at "
                                f"{conversation_file.stem}:{sequence_index}; stopping: "
                                f"{error_text}"
                            )
                        else:
                            console.print(
                                f"Add failed at {conversation_file.stem}:{sequence_index}; "
                                f"skipping item and continuing: {error_text}"
                            )
                    if memmachine_wait_started and caught_error is not None:
                        raise RuntimeError(
                            "Stopping MemMachine sequence because the async-ingestion "
                            "completion barrier failed; no later Add was submitted"
                        ) from caught_error
                    if sequence_index == 1 or sequence_index % args.progress_every == 0:
                        console.print(
                            f"{conversation_file.stem}:{sequence_index} "
                            f"length={cumulative_tokens if cumulative_tokens is not None else cumulative_chars} "
                            f"add={add_seconds:.3f}s {status}"
                        )
                if hasattr(adapter, "finalize_incremental_conversation"):
                    await adapter.finalize_incremental_conversation(conversation_id)
                del document
    finally:
        if hasattr(adapter, "close") and callable(adapter.close):
            with suppress(Exception):
                await adapter.close()

    completed_at = datetime.now().astimezone().isoformat()
    summary = {
        "schema_version": 1,
        "dataset": dataset_label,
        "system": args.system,
        "run_name": args.run_name,
        "completed_at": completed_at,
        "total_items": successful_items + failed_items,
        "successful_items": successful_items,
        "failed_items": failed_items,
        "total_add_seconds": global_cumulative_add_seconds,
        "mean_add_seconds": (
            successful_add_seconds / successful_items
            if successful_items
            else None
        ),
        "wall_seconds": time.perf_counter() - benchmark_start,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.no_plot:
        summary["plot_created"] = create_plot(
            output_dir,
            iter_jsonl_records(output_dir / "speed_sequence.jsonl"),
            dataset_label,
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output_dir


async def main() -> None:
    args = build_parser().parse_args()
    output_dir = await run(args)
    print(f"Completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
