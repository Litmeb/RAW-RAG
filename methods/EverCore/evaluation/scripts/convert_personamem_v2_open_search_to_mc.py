"""Convert PersonaMem-v2 open-ended RAG search artifacts for MC reruns."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


OPEN_PREFIX = "personamem_v2_open_ended_"
MC_PREFIX = "personamem_v2_"


def convert_question_id(question_id: str) -> str:
    return re.sub(rf"^{re.escape(OPEN_PREFIX)}", MC_PREFIX, question_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    src_search_path = src_dir / "search_results.json"

    if not src_search_path.exists():
        raise FileNotFoundError(f"Missing source search_results.json: {src_search_path}")
    if dst_dir.exists() and not args.force:
        raise FileExistsError(f"Destination already exists: {dst_dir}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    with open(src_search_path, "r", encoding="utf-8") as f:
        search_results = json.load(f)

    top_level_changed = 0
    metadata_changed = 0
    unchanged_ids = []

    for record in search_results:
        old_question_id = record.get("question_id", "")
        new_question_id = convert_question_id(old_question_id)
        if new_question_id != old_question_id:
            record["question_id"] = new_question_id
            top_level_changed += 1
        else:
            unchanged_ids.append(old_question_id)

        retrieval_metadata = record.get("retrieval_metadata")
        if isinstance(retrieval_metadata, dict):
            old_meta_question_id = retrieval_metadata.get("question_id", "")
            new_meta_question_id = convert_question_id(old_meta_question_id)
            if new_meta_question_id != old_meta_question_id:
                retrieval_metadata["question_id"] = new_meta_question_id
                metadata_changed += 1

    with open(dst_dir / "search_results.json", "w", encoding="utf-8") as f:
        json.dump(search_results, f, indent=2, ensure_ascii=False)

    checkpoint = {
        "run_name": "default",
        "last_updated": datetime.now().astimezone().isoformat(),
        "completed_stages": ["search"],
        "metadata": {
            "converted_from": str(src_dir),
            "conversion": (
                "Rewrote PersonaMem-v2 open-ended question_id prefixes to "
                "PersonaMem-v2 multiple-choice prefixes; retrieval content unchanged."
            ),
        },
    }
    with open(dst_dir / "checkpoint_default.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    note = (
        f"Converted from: {src_dir}\n"
        "Conversion: question_id and retrieval_metadata.question_id prefixes were "
        f"rewritten from {OPEN_PREFIX}* to {MC_PREFIX}*.\n"
        "Search retrieval content is unchanged.\n"
        "Next command:\n"
        "python -m evaluation.cli --dataset personamem_v2 --system rag "
        f"--run-name {dst_dir.name.removeprefix('personamem_v2-rag-')} "
        "--stages answer evaluate\n"
    )
    (dst_dir / "CONVERSION_NOTE.txt").write_text(note, encoding="utf-8")

    print(
        json.dumps(
            {
                "records": len(search_results),
                "top_level_ids_changed": top_level_changed,
                "metadata_ids_changed": metadata_changed,
                "unchanged_id_count": len(unchanged_ids),
                "destination": str(dst_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
