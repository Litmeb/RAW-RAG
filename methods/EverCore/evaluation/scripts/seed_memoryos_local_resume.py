"""Seed a MemoryOS local Add checkpoint from a pre-resume progress report."""

import argparse
import json
from pathlib import Path

from evaluation.src.adapters.memoryos_adapter import MemoryOSAdapter
from evaluation.src.core.loaders import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--completed", type=int, nargs="+", required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        "memoryagentbench", str(args.data_dir), dataset_format="memoryagentbench"
    )
    if len(dataset.conversations) != len(args.completed):
        raise ValueError(
            f"expected {len(dataset.conversations)} completion counts, "
            f"got {len(args.completed)}"
        )

    resume_adapter = object.__new__(MemoryOSAdapter)
    resume_adapter.config = {"dataset_name": "memoryagentbench"}
    conversations = {}
    for conversation, completed_count in zip(dataset.conversations, args.completed):
        messages = MemoryOSAdapter._conversation_to_messages(
            resume_adapter,
            conversation,
            format_type="basic",
            perspective=None,
        )
        if not 0 <= completed_count <= len(messages):
            raise ValueError(
                f"invalid completed count {completed_count} for "
                f"{conversation.conversation_id} (total={len(messages)})"
            )
        conversations[f"{conversation.conversation_id}::speaker_a"] = {
            "speaker": "speaker_a",
            "fingerprint": MemoryOSAdapter._messages_fingerprint(messages),
            "completed_units": list(range(completed_count)),
            "completed": completed_count == len(messages),
        }

    state = {
        "version": 1,
        "source": "manual_progress_seed",
        "conversations": conversations,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    target = args.run_dir / "memoryos_local_add_resume.json"
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    print(f"Wrote {target}")
    for conversation, completed_count in zip(dataset.conversations, args.completed):
        print(f"{conversation.conversation_id}: {completed_count}/{len(conversation.messages)}")


if __name__ == "__main__":
    main()
