import json

import pytest

from evaluation.scripts.convert_beam_1m_sequence import (
    convert_directory,
    convert_document,
    flatten_normalized_chat,
    flatten_raw_chat,
)
from evaluation.speed_sequence_cli import iter_beam_items, numeric_json_files


def sample_document(conversation_id=1):
    return {
        "conversation_id": str(conversation_id),
        "conversation_seed": {"seed": conversation_id},
        "chat": [
            [
                {
                    "id": 0,
                    "role": "user",
                    "content": "main",
                    "question_type": "main_question",
                    "time_anchor": "March-01-2024",
                },
                {"id": 1, "role": "assistant", "content": "answer"},
                {
                    "id": 2,
                    "role": "user",
                    "content": "follow up",
                    "question_type": "followup_question",
                },
                {"id": 3, "role": "assistant", "content": "follow answer"},
                {
                    "id": 4,
                    "role": "user",
                    "content": "next main",
                    "question_type": "main_question",
                },
                {"id": 5, "role": "assistant", "content": "next answer"},
            ],
            [
                {
                    "id": 6,
                    "role": "user",
                    "content": "batch two",
                    "question_type": "main_question",
                    "time_anchor": "March-12-2024",
                },
                {"id": 7, "role": "assistant", "content": "last"},
            ],
        ],
    }


def test_convert_document_preserves_messages_and_builds_beam_hierarchy():
    source = sample_document()
    converted = convert_document(source)

    batches = converted["chat"][0]["plan-1"]
    assert [len(batch["turns"]) for batch in batches] == [2, 1]
    assert [len(turn) for turn in batches[0]["turns"]] == [4, 2]
    assert flatten_normalized_chat(converted) == flatten_raw_chat(source)
    assert [item.message.content for item in iter_beam_items(converted)] == [
        "main",
        "answer",
        "follow up",
        "follow answer",
        "next main",
        "next answer",
        "batch two",
        "last",
    ]


def test_convert_directory_supports_35_conversations(tmp_path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "normalized"
    input_dir.mkdir()
    for number in range(1, 36):
        (input_dir / f"{number}.json").write_text(
            json.dumps(sample_document(number)), encoding="utf-8"
        )

    manifest = convert_directory(input_dir, output_dir)

    assert manifest["conversation_count"] == 35
    assert len(numeric_json_files(output_dir)) == 35
    assert json.loads((output_dir / "35.json").read_text(encoding="utf-8"))[
        "conversation_id"
    ] == "35"

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_directory(input_dir, output_dir)
