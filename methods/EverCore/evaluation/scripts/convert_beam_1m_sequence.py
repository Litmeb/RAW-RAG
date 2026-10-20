"""Normalize raw BEAM-1M files for ``evaluation.speed_sequence_cli``.

Raw BEAM-1M stores ten chronological batches as ten flat message lists under
``chat``.  The speed CLI uses the newer BEAM hierarchy, so this converter wraps
them as ``chat -> plan -> batch -> turn -> message`` without changing messages.

Example (run from ``methods/EverCore``):
    python -m evaluation.scripts.convert_beam_1m_sequence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = REPOSITORY_ROOT / "beam" / "1M"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "beam" / "BEAM-1M"


def group_messages_into_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a BEAM batch whenever a new main question begins."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("question_type") == "main_question" and current:
            turns.append(current)
            current = []
        current.append(message.copy())
    if current:
        turns.append(current)
    return turns


def flatten_raw_chat(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return raw BEAM-1M messages in their stored order."""
    return [message for batch in document.get("chat", []) for message in batch]


def flatten_normalized_chat(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized messages in speed CLI traversal order."""
    return [
        message
        for chat_entry in document.get("chat", [])
        for batches in chat_entry.values()
        for batch in batches
        for turn in batch.get("turns", [])
        for message in turn
    ]


def convert_document(document: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw BEAM-1M conversation to the newer BEAM hierarchy."""
    chat = document.get("chat")
    if not isinstance(chat, list) or not chat:
        raise ValueError("BEAM-1M chat must be a non-empty list")
    if not all(isinstance(batch, list) for batch in chat):
        raise ValueError("Expected raw BEAM-1M chat batches as flat message lists")

    batches: list[dict[str, Any]] = []
    for batch_number, raw_messages in enumerate(chat, start=1):
        for message_index, message in enumerate(raw_messages, start=1):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Batch {batch_number} message {message_index} must be an object"
                )
        time_anchor = next(
            (message.get("time_anchor") for message in raw_messages if message.get("time_anchor")),
            None,
        )
        batches.append(
            {
                "batch_number": batch_number,
                "time_anchor": time_anchor,
                "turns": group_messages_into_turns(raw_messages),
            }
        )

    converted = {
        **document,
        "schema_version": 1,
        "dataset": "BEAM-1M",
        "source_format": "beam_1m_flat_batches",
        "chat": [{"plan-1": batches}],
    }
    if flatten_normalized_chat(converted) != flatten_raw_chat(document):
        raise ValueError(
            f"Message order/content changed in conversation {document.get('conversation_id')}"
        )
    return converted


def numeric_source_files(input_dir: Path) -> list[Path]:
    """Find a non-empty contiguous 1.json ... N.json source set."""
    files = sorted(
        (path for path in input_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not files:
        raise ValueError(f"No numeric JSON files found in {input_dir}")
    numbers = [int(path.stem) for path in files]
    expected = list(range(1, numbers[-1] + 1))
    if numbers != expected:
        raise ValueError(f"Source files must be contiguous from 1; found {numbers}")
    return files


def convert_directory(
    input_dir: Path, output_dir: Path, *, force: bool = False
) -> dict[str, Any]:
    """Normalize every numbered BEAM-1M conversation and return its manifest."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    source_files = numeric_source_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / path.name for path in source_files]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing[:5])
        suffix = " ..." if len(existing) > 5 else ""
        raise FileExistsError(f"Refusing to overwrite existing files: {names}{suffix}")

    records: list[dict[str, Any]] = []
    for source_path, target_path in zip(source_files, targets, strict=True):
        with source_path.open(encoding="utf-8") as source_file:
            source = json.load(source_file)
        converted = convert_document(source)
        temporary_path = target_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(converted, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_path.replace(target_path)
        records.append(
            {
                "file": target_path.name,
                "conversation_id": converted.get("conversation_id"),
                "batches": len(converted["chat"][0]["plan-1"]),
                "messages": len(flatten_normalized_chat(converted)),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "BEAM-1M",
        "source": str(input_dir),
        "conversion": "flat chat batch -> plan-1 batch; main_question boundary -> turn",
        "conversation_count": len(records),
        "conversations": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize BEAM-1M for evaluation.speed_sequence_cli"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = convert_directory(args.input_dir, args.output_dir, force=args.force)
    total_messages = sum(item["messages"] for item in manifest["conversations"])
    print(
        f"Converted {manifest['conversation_count']} BEAM-1M conversations "
        f"({total_messages} messages) to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
