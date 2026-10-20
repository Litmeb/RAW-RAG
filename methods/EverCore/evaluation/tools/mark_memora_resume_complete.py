"""Mark safely identified conversations complete in a legacy Memora checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mark_complete(checkpoint: Path, conversation_ids: list[str]) -> int:
    checkpoint = checkpoint.resolve()
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    conversations = state.get("conversations", {})
    missing = [item for item in conversation_ids if item not in conversations]
    if missing:
        raise ValueError(f"Checkpoint is missing conversations: {missing}")
    for conversation_id in conversation_ids:
        conversations[conversation_id]["complete"] = True

    temp_path = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(checkpoint)
    return len(conversation_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Conversation ID prefix")
    parser.add_argument(
        "--range",
        dest="index_range",
        required=True,
        help="Half-open numeric range, for example 0:29",
    )
    args = parser.parse_args()
    start_text, end_text = args.index_range.split(":", 1)
    start, end = int(start_text), int(end_text)
    if start < 0 or end <= start:
        raise ValueError("--range must be a non-empty half-open range")
    conversation_ids = [f"{args.prefix}{index}" for index in range(start, end)]
    count = mark_complete(args.checkpoint, conversation_ids)
    print(f"Marked {count} conversations complete in {args.checkpoint.resolve()}")


if __name__ == "__main__":
    main()
