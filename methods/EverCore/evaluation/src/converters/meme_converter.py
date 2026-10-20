"""
MEME converter - convert MEME benchmark episodes to Locomo format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


HF_BASE_URL = "https://huggingface.co/datasets/meme-benchmark/MEME/resolve/main"


def _download_file(url: str, dest: Path) -> None:
    """Download a file if it does not already exist."""
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"   ⬇️  Downloading {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _convert_time_format(input_str: str) -> str:
    """
    Convert MEME timestamp from "YYYY/MM/DD (Day) HH:MM"
    to LoCoMo-compatible "H:MM am/pm on D Month, YYYY".
    """
    dt_object = datetime.strptime(input_str, "%Y/%m/%d (%a) %H:%M")
    hour_12 = dt_object.hour % 12 or 12
    minute = dt_object.strftime("%M")
    am_pm = dt_object.strftime("%p").lower()
    month = dt_object.strftime("%B")
    return f"{hour_12}:{minute} {am_pm} on {dt_object.day} {month}, {dt_object.year}"


def _normalize_speaker(role: str) -> str:
    role_norm = str(role).strip().lower()
    if role_norm == "user":
        return "User"
    if role_norm == "assistant":
        return "Assistant"
    return str(role).strip() or "Unknown"


def _get_gold_answer(question: dict) -> str:
    if "expected_answer" in question:
        return str(question["expected_answer"])
    if "gold_answer" in question:
        return str(question["gold_answer"])
    return ""


def _get_category(question: dict) -> str:
    task_type = str(question.get("task_type", ""))
    hop = question.get("hop")
    if hop is None:
        return task_type
    return f"{task_type}_hop{hop}"


def _entity_key(entity_values: Dict[str, object]) -> str:
    if not entity_values:
        return ""
    return list(entity_values.keys())[0]


def _build_qa_items(
    episode: dict,
    phase: str,
    questions_block: dict,
) -> List[dict]:
    episode_id = str(episode["episode_id"])
    question_date = _convert_time_format(questions_block["timestamp"])
    qa_items = []

    for idx, question in enumerate(questions_block.get("questions", [])):
        entity_values = question.get("entity_values", {}) or {}
        qa_items.append(
            {
                "question_id": f"meme_{episode_id}_{phase}_qa{idx}",
                "question": question.get("question", ""),
                "answer": _get_gold_answer(question),
                "category": _get_category(question),
                "evidence": [],
                "question_date": question_date,
                "phase": phase,
                "task_type": question.get("task_type"),
                "entity": question.get("entity"),
                "entity_values": entity_values,
                "entity_key": _entity_key(entity_values),
                "hop": question.get("hop"),
                "position_after_session": questions_block.get("position_after_session"),
                "timestamp_raw": questions_block.get("timestamp"),
                "episode_id": episode_id,
                "domain": episode.get("domain"),
                "root": episode.get("root"),
                "root_change": episode.get("root_change"),
            }
        )

    return qa_items


def convert_episode_to_locomo_style(episode: dict) -> dict:
    conversation = {
        "speaker_a": "User",
        "speaker_b": "Assistant",
    }

    for session_idx, session in enumerate(episode.get("sessions", [])):
        conversation[f"session_{session_idx}_date_time"] = _convert_time_format(
            session["timestamp"]
        )
        conversation[f"session_{session_idx}"] = []
        for msg_idx, msg in enumerate(session.get("conversation", [])):
            conversation[f"session_{session_idx}"].append(
                {
                    "speaker": _normalize_speaker(msg.get("role", "")),
                    "text": msg.get("content", ""),
                    "dia_id": f"D{session_idx}:{msg_idx}",
                }
            )

    qa_items = []
    qa_items.extend(_build_qa_items(episode, "before", episode["before_questions"]))
    qa_items.extend(_build_qa_items(episode, "after", episode["after_questions"]))

    return {
        "qa": qa_items,
        "conversation": conversation,
        "metadata": {
            "episode_id": episode.get("episode_id"),
            "domain": episode.get("domain"),
            "root": episode.get("root"),
            "root_change": episode.get("root_change"),
            "total_sessions": episode.get("total_sessions"),
            "evidence_session_indices": episode.get("evidence_session_indices", []),
        },
    }


class _BaseMEMEConverter(BaseConverter):
    raw_filename = "meme_filler32k.json"
    output_filename = "meme_filler32k_locomo_style.json"

    def get_input_files(self) -> Dict[str, str]:
        return {"raw": self.raw_filename}

    def get_output_filename(self) -> str:
        return self.output_filename

    def _ensure_assets(self, raw_path: Path) -> Path:
        _download_file(f"{HF_BASE_URL}/{self.raw_filename}?download=true", raw_path)
        return raw_path

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        print(f"🔄 Converting MEME to Locomo format ({self.raw_filename})...")

        raw_path = self._ensure_assets(Path(input_paths["raw"]))
        with open(raw_path, "r", encoding="utf-8") as f:
            episodes = json.load(f)

        print(f"   Loaded {len(episodes)} episodes")
        locomo_style_data = [
            convert_episode_to_locomo_style(episode) for episode in episodes
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_style_data, f, indent=2, ensure_ascii=False)

        total_questions = sum(len(item["qa"]) for item in locomo_style_data)
        print(f"   ✅ Saved {len(locomo_style_data)} episodes to {output_path}")
        print(f"   Total questions: {total_questions}")


@register_converter("meme")
class MEMEConverter(_BaseMEMEConverter):
    raw_filename = "meme_filler32k.json"
    output_filename = "meme_filler32k_locomo_style.json"


@register_converter("meme_filler128k")
class MEMEFiller128kConverter(_BaseMEMEConverter):
    raw_filename = "meme_filler128k.json"
    output_filename = "meme_filler128k_locomo_style.json"


@register_converter("meme_nofiller")
class MEMENoFillerConverter(_BaseMEMEConverter):
    raw_filename = "meme_nofiller.json"
    output_filename = "meme_nofiller_locomo_style.json"
