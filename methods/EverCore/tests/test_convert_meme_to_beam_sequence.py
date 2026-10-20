import json

import pytest

from evaluation.scripts.convert_meme_to_beam_sequence import (
    convert_episode,
    convert_file,
    normalize_timestamp,
    validate_conversion,
)
from evaluation.speed_sequence_cli import iter_beam_items, numeric_json_files


def sample_episode(number=1):
    return {
        "episode_id": f"episode_{number}",
        "domain": "test",
        "sessions": [
            {
                "session_id": "filler_1",
                "type": "filler",
                "timestamp": "2023/03/01 (Wed) 09:00",
                "conversation": [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                    {"role": "assistant", "content": "four"},
                ],
            },
            {
                "session_id": "evidence_1",
                "type": "evidence",
                "evidence_type": "Fact Introduction",
                "timestamp": "2023/03/02 (Thu) 10:30",
                "conversation": [{"role": "user", "content": "five"}],
            },
        ],
    }


def test_convert_episode_is_consumable_by_speed_sequence_cli():
    source = sample_episode()
    converted = convert_episode(source, 0)
    validate_conversion(source, converted)

    batches = converted["chat"][0]["plan-1"]
    assert converted["conversation_id"] == "episode_1"
    assert len(batches) == 2
    assert [len(turn) for turn in batches[0]["turns"]] == [2, 2]
    assert batches[0]["time_anchor"] == "2023-03-01T09:00:00"

    items = list(iter_beam_items(converted))
    assert [item.message.content for item in items] == [
        "one",
        "two",
        "three",
        "four",
        "five",
    ]
    assert items[0].message.timestamp.isoformat() == "2023-03-01T09:00:00"
    assert items[-1].batch_number == 2


def test_convert_file_writes_numeric_set_and_manifest(tmp_path):
    source_path = tmp_path / "meme.json"
    source_path.write_text(
        json.dumps([sample_episode(index) for index in range(1, 11)]),
        encoding="utf-8",
    )
    output_dir = tmp_path / "converted"

    manifest = convert_file(source_path, output_dir)

    assert manifest["conversation_count"] == 10
    assert [path.name for path in numeric_json_files(output_dir)] == [
        f"{index}.json" for index in range(1, 11)
    ]
    assert json.loads((output_dir / "10.json").read_text(encoding="utf-8"))[
        "conversation_id"
    ] == "episode_10"
    assert (output_dir / "manifest.json").exists()


def test_normalize_timestamp_rejects_unknown_format():
    assert normalize_timestamp("2023/03/01 (Wed) 09:00") == "2023-03-01T09:00:00"
    with pytest.raises(ValueError, match="Unsupported MEME timestamp"):
        normalize_timestamp("March-ish, probably")
