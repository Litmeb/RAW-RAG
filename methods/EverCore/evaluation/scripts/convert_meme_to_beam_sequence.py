"""Convert the first MEME episodes to the BEAM hierarchy used by the speed CLI.

The output directory contains numeric ``1.json`` ... ``N.json`` files.  Each
MEME episode becomes one conversation, each session becomes one BEAM batch,
and messages retain their original chronological order.

Example (run from ``methods/EverCore``):
    python -m evaluation.scripts.convert_meme_to_beam_sequence
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = EVALUATION_ROOT / "data" / "meme" / "meme_filler32k.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "beam" / "MEME-10"


def normalize_timestamp(value: Any) -> str | None:
    """Convert MEME's timestamp to an ISO value understood by the speed CLI."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError as exc:
        raise ValueError(f"Unsupported MEME timestamp: {text!r}") from exc


def group_messages_into_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group a session at user-message boundaries without changing message order."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def convert_episode(episode: dict[str, Any], source_index: int) -> dict[str, Any]:
    """Return one MEME episode in the subset of BEAM consumed by the speed CLI."""
    episode_id = str(episode.get("episode_id") or f"meme_{source_index:03d}")
    batches: list[dict[str, Any]] = []
    next_message_id = 0

    for session_index, session in enumerate(episode.get("sessions", []), start=1):
        source_timestamp = session.get("timestamp")
        time_anchor = normalize_timestamp(source_timestamp)
        converted_messages: list[dict[str, Any]] = []
        for message_index, raw_message in enumerate(
            session.get("conversation", []), start=1
        ):
            if not isinstance(raw_message, dict):
                raise ValueError(
                    f"{episode_id} session {session_index} message {message_index} "
                    "must be an object"
                )
            role = str(raw_message.get("role") or "user").lower()
            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"{episode_id} session {session_index} has unsupported role {role!r}"
                )
            converted_messages.append(
                {
                    "content": str(raw_message.get("content") or ""),
                    "id": next_message_id,
                    "index": f"{session_index},{message_index}",
                    "question_type": session.get("type"),
                    "role": role,
                    "time_anchor": time_anchor if message_index == 1 else None,
                    "source_session_id": session.get("session_id"),
                    "source_message_index": message_index - 1,
                }
            )
            next_message_id += 1

        batches.append(
            {
                "batch_number": session_index,
                "time_anchor": time_anchor,
                "turns": group_messages_into_turns(converted_messages),
                "source_session_id": session.get("session_id"),
                "source_session_type": session.get("type"),
                "source_evidence_type": session.get("evidence_type"),
                "source_timestamp": source_timestamp,
            }
        )

    return {
        "schema_version": 1,
        "dataset": "MEME",
        "source_condition": "meme_filler32k",
        "source_episode_index": source_index,
        "conversation_id": episode_id,
        "domain": episode.get("domain"),
        "chat": [{"plan-1": batches}],
    }


def validate_conversion(source: dict[str, Any], converted: dict[str, Any]) -> None:
    """Verify that conversion preserves every source role/content in order."""
    source_messages = [
        (message.get("role"), str(message.get("content") or ""))
        for session in source.get("sessions", [])
        for message in session.get("conversation", [])
    ]
    batches = converted["chat"][0]["plan-1"]
    output_messages = [
        (message["role"], message["content"])
        for batch in batches
        for turn in batch["turns"]
        for message in turn
    ]
    if output_messages != source_messages:
        raise ValueError(
            f"Message order/content changed while converting {converted['conversation_id']}"
        )
    if len(batches) != len(source.get("sessions", [])):
        raise ValueError(
            f"Session count changed while converting {converted['conversation_id']}"
        )


def convert_file(
    input_path: Path, output_dir: Path, *, limit: int = 10, force: bool = False
) -> dict[str, Any]:
    """Convert the first ``limit`` MEME episodes and return a small manifest."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    with input_path.open(encoding="utf-8") as source_file:
        episodes = json.load(source_file)
    if not isinstance(episodes, list):
        raise ValueError("MEME input must be a JSON array")
    if len(episodes) < limit:
        raise ValueError(f"Requested {limit} episodes, but input contains {len(episodes)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / f"{index}.json" for index in range(1, limit + 1)]
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {names}")

    records: list[dict[str, Any]] = []
    for output_index, episode in enumerate(episodes[:limit], start=1):
        converted = convert_episode(episode, output_index - 1)
        validate_conversion(episode, converted)
        target = targets[output_index - 1]
        target.write_text(
            json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        message_count = sum(
            len(turn)
            for batch in converted["chat"][0]["plan-1"]
            for turn in batch["turns"]
        )
        records.append(
            {
                "file": target.name,
                "conversation_id": converted["conversation_id"],
                "sessions": len(converted["chat"][0]["plan-1"]),
                "messages": message_count,
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "MEME",
        "source": str(input_path.resolve()),
        "conversion": "episode -> conversation; session -> batch; user boundary -> turn",
        "conversation_count": limit,
        "conversations": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MEME episodes to speed_sequence_cli-compatible BEAM files"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = convert_file(
        args.input.resolve(),
        args.output_dir.resolve(),
        limit=args.limit,
        force=args.force,
    )
    print(
        f"Converted {manifest['conversation_count']} MEME conversations to "
        f"{args.output_dir.resolve()}"
    )
    for record in manifest["conversations"]:
        print(
            f"  {record['file']}: {record['conversation_id']} - "
            f"{record['sessions']} sessions, {record['messages']} messages"
        )


if __name__ == "__main__":
    main()
