"""Create a MemMachine Add checkpoint from a captured progress report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.src.adapters.memmachine_adapter import MemMachineAdapter
from evaluation.src.core.loaders import load_dataset


def parse_partial(value: str) -> dict[int, int]:
    partial: dict[int, int] = {}
    if not value.strip():
        return partial
    for item in value.split(","):
        index_text, completed_text = item.split(":", 1)
        partial[int(index_text)] = int(completed_text)
    return partial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--completed-through", type=int, required=True)
    parser.add_argument("--partial", default="")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--org-id", default="ragforlocomo")
    parser.add_argument("--project-prefix", default="eval")
    parser.add_argument(
        "--memory-types", nargs="+", default=["episodic", "semantic"]
    )
    args = parser.parse_args()

    dataset = load_dataset(
        "longmemeval",
        str(args.dataset_dir),
        max_content_length=8000,
        dataset_format="longmemeval",
    )
    partial = parse_partial(args.partial)
    conversations: dict[str, dict[str, object]] = {}

    for index, conversation in enumerate(dataset.conversations):
        if index <= args.completed_through:
            completed = len(conversation.messages)
        else:
            completed = partial.get(index, 0)
        if not 0 <= completed <= len(conversation.messages):
            raise ValueError(
                f"Invalid progress for conversation {index}: "
                f"{completed}/{len(conversation.messages)}"
            )
        if completed != len(conversation.messages) and completed % args.batch_size:
            raise ValueError(
                f"Conversation {index} stops inside a batch: {completed}"
            )
        conversations[conversation.conversation_id] = {
            "message_count": len(conversation.messages),
            "message_sha256": MemMachineAdapter._conversation_digest(conversation),
            "completed_messages": completed,
        }

    state = {
        "version": 1,
        "adapter": "memmachine",
        "scope": {
            "org_id": MemMachineAdapter._safe_id(args.org_id),
            "project_scope": [
                MemMachineAdapter._safe_id(args.project_prefix),
                MemMachineAdapter._safe_id(args.run_name),
            ],
            "memory_types": args.memory_types,
            "batch_size": args.batch_size,
        },
        "conversations": conversations,
    }
    target = args.run_dir / "memmachine_add_resume.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(target)
    print(f"Wrote {target}")
    print(
        f"Completed conversations: {args.completed_through + 1}; "
        f"partial conversations: {len(partial)}"
    )


if __name__ == "__main__":
    main()
