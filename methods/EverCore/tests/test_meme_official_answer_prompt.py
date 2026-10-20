import json

import pytest

from evaluation.src.core.loaders import (
    MEME_OFFICIAL_ANSWER_PROMPT_KEY,
    load_locomo_dataset,
)
from evaluation.src.utils.config import load_yaml


@pytest.fixture
def locomo_data_path(tmp_path):
    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "conversation": {
                        "speaker_a": "User",
                        "speaker_b": "Assistant",
                        "session_0_date_time": "9:00 am on 1 March, 2023",
                        "session_0": [
                            {
                                "speaker": "User",
                                "text": "My favorite color is blue.",
                                "dia_id": "D0:0",
                            }
                        ],
                    },
                    "qa": [
                        {
                            "question_id": "q1",
                            "question": "What is my favorite color?",
                            "answer": "blue",
                            "category": "ER",
                            "evidence": [],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return data_path


@pytest.mark.parametrize("dataset_name", ["meme", "meme_filler128k", "meme_nofiller"])
def test_all_meme_variants_use_official_answer_prompt(dataset_name, locomo_data_path):
    dataset = load_locomo_dataset(str(locomo_data_path), dataset_name=dataset_name)

    assert dataset.qa_pairs[0].metadata["prompt_key"] == (
        MEME_OFFICIAL_ANSWER_PROMPT_KEY
    )


def test_non_meme_dataset_does_not_receive_meme_prompt(locomo_data_path):
    dataset = load_locomo_dataset(str(locomo_data_path), dataset_name="locomo")

    assert "prompt_key" not in dataset.qa_pairs[0].metadata


def test_official_meme_prompt_matches_reference_text():
    prompts = load_yaml("evaluation/config/prompts.yaml")

    assert prompts["online_api"]["default"][MEME_OFFICIAL_ANSWER_PROMPT_KEY] == (
        "Answer the user's question based ONLY on the context provided below.\n"
        "\n"
        "If the information is not in the context, say you don't have that information.\n"
        "\n"
        "Answer with ONLY the value. Do not explain or add context.\n"
        "\n"
        "Context:\n"
        "\n"
        "{context}\n"
        "\n"
        "Question: {question}\n"
    )
