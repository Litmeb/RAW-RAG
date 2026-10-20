"""
PersonaMem v2 converter - convert PersonaMem-v2 benchmark data to Locomo format.
"""
import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter


HF_BASE_URL = "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/main"
BENCHMARK_REL_PATH = "benchmark/text/benchmark.csv"


def _download_file(url: str, dest: Path) -> None:
    """Download a file if it does not already exist."""
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"   ⬇️  Downloading {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _load_json_text(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_persona_name(system_content: str) -> str:
    """Extract persona name from the PersonaMem-v2 system message."""
    marker = "You are an AI assistant helping a user with the following persona:\n\n"
    if marker in system_content:
        payload = system_content.split(marker, 1)[1].strip()
        try:
            persona = json.loads(payload)
            if isinstance(persona, dict):
                name = persona.get("name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
        except Exception:
            pass

    return "User"


def _clean_message_prefix(text: str) -> str:
    """Normalize the PersonaMem chat history message content."""
    return text.strip()


def _extract_user_query_content(user_query_value: str) -> str:
    """Extract the raw PersonaMem-v2 user query content."""
    try:
        user_query = ast.literal_eval(user_query_value)
        if isinstance(user_query, dict):
            content = user_query.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    except Exception:
        pass

    return str(user_query_value).strip().strip('"').strip("'")


def _normalize_user_query(user_query_value: str) -> str:
    """Extract PersonaMem-v2 user query content in the official MCQ style."""
    query = _extract_user_query_content(user_query_value)
    return (
        query
        + " Please recall my related preferences from our conversation history to give personalized responses."
    )


def _normalize_answer_pool(correct_answer: str, incorrect_answers_value: str) -> List[str]:
    """Build an answer pool from the benchmark row."""
    try:
        incorrect_answers = ast.literal_eval(incorrect_answers_value)
        if not isinstance(incorrect_answers, list):
            incorrect_answers = [str(incorrect_answers)]
    except Exception:
        incorrect_answers = [incorrect_answers_value]

    pool = [correct_answer] + [str(answer) for answer in incorrect_answers]
    # Remove duplicates while preserving order.
    deduped = []
    seen = set()
    for answer in pool:
        if answer not in seen:
            deduped.append(answer)
            seen.add(answer)

    return deduped[:4]


def _build_option_map(seed_text: str, correct_answer: str, incorrect_answers_value: str) -> Tuple[Dict[str, str], str]:
    """
    Create the official PersonaMem-v2 multiple-choice mapping.

    The official inference script seeds shuffling with Python's hash of
    f"{persona_id}_{user_query_content}".
    """
    answer_pool = _normalize_answer_pool(correct_answer, incorrect_answers_value)
    labels = ["A", "B", "C", "D"][: len(answer_pool)]
    order = list(range(len(answer_pool)))

    seed = hash(seed_text) % 2**32
    rng = random.Random(seed)
    rng.shuffle(order)

    shuffled_answers = [answer_pool[idx] for idx in order]
    option_map = {label: answer for label, answer in zip(labels, shuffled_answers)}
    correct_label = next(label for label, text in option_map.items() if text == correct_answer)
    return option_map, correct_label


class _BasePersonaMemV2Converter(BaseConverter):
    """Shared PersonaMem-v2 converter for multiple-choice and open-ended modes."""

    question_mode = "multiple_choice"
    output_filename = "personamem_v2_32k_locomo_style.json"

    def get_input_files(self) -> Dict[str, str]:
        return {
            "benchmark": BENCHMARK_REL_PATH,
            "histories": "data/chat_history_32k",
        }

    def get_output_filename(self) -> str:
        return self.output_filename

    def _ensure_benchmark_assets(self, data_dir: Path) -> Path:
        """Download benchmark CSV and referenced chat histories if missing."""
        benchmark_path = data_dir / BENCHMARK_REL_PATH
        _download_file(f"{HF_BASE_URL}/{BENCHMARK_REL_PATH}?download=true", benchmark_path)

        with open(benchmark_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        chat_links = sorted({row["chat_history_32k_link"] for row in rows})
        for rel_link in chat_links:
            dest = data_dir / rel_link
            _download_file(f"{HF_BASE_URL}/{rel_link}?download=true", dest)

        return benchmark_path

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        print(f"🔄 Converting PersonaMem-v2 to Locomo format ({self.question_mode})...")

        data_dir = Path(input_paths["benchmark"]).resolve().parents[2]
        benchmark_path = self._ensure_benchmark_assets(data_dir)

        print("   Loading benchmark CSV...")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"   Loaded {len(rows)} benchmark rows")

        grouped_rows = defaultdict(list)
        for row_idx, row in enumerate(rows):
            grouped_rows[row["chat_history_32k_link"]].append((row_idx, row))

        print(f"   Grouped into {len(grouped_rows)} chat histories")
        print("   Converting chat histories and QA rows...")

        locomo_data = []

        for link, row_group in grouped_rows.items():
            chat_history_path = data_dir / link
            chat_obj = _load_json_text(chat_history_path)
            chat_history = chat_obj.get("chat_history", [])

            persona_name = "User"
            if chat_history and chat_history[0].get("role") == "system":
                persona_name = _extract_persona_name(chat_history[0].get("content", ""))

            conversation = {
                "speaker_a": persona_name,
                "speaker_b": "Assistant",
                "session_0_date_time": "Unknown",
                "session_0": [],
            }

            dialogue_idx = 0
            for msg in chat_history:
                role = msg.get("role")
                if role == "system":
                    continue

                speaker = persona_name if role == "user" else "Assistant"
                conversation["session_0"].append(
                    {
                        "speaker": speaker,
                        "text": _clean_message_prefix(msg.get("content", "")),
                        "dia_id": f"D0:{dialogue_idx}",
                    }
                )
                dialogue_idx += 1

            qa_items = []
            for local_idx, (_, row) in enumerate(row_group):
                question_id = self._build_question_id(row["persona_id"], local_idx)
                qa_item = self._build_qa_item(question_id, row)
                qa_items.append(qa_item)

            locomo_data.append(
                {
                    "qa": qa_items,
                    "conversation": conversation,
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(locomo_data, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Saved {len(locomo_data)} entries to {output_path}")
        print(f"   Total questions: {sum(len(entry['qa']) for entry in locomo_data)}")

    def _base_qa_item(self, question_id: str, row: dict) -> dict:
        return {
            "question_id": question_id,
            "question": _normalize_user_query(row["user_query"]),
            "evidence": [],
            "category": row["conversation_scenario"],
            "topic": row["topic_query"],
            "persona_id": row["persona_id"],
            "conversation_scenario": row["conversation_scenario"],
            "pref_type": row["pref_type"],
            "who": row["who"],
            "updated": row["updated"],
            "preference": row["preference"],
            "topic_preference": row["topic_preference"],
            "related_conversation_snippet": row["related_conversation_snippet"],
            "prev_pref": row["prev_pref"],
            "sensitive_info": row["sensitive_info"],
            "raw_persona_file": row["raw_persona_file"],
            "chat_history_32k_link": row["chat_history_32k_link"],
            "chat_history_128k_link": row["chat_history_128k_link"],
            "question_mode": self.question_mode,
            "answer_text": row["correct_answer"],
            "raw_user_query": row["user_query"],
        }

    def _build_qa_item(self, question_id: str, row: dict) -> dict:
        raise NotImplementedError

    def _build_question_id(self, persona_id: str, local_idx: int) -> str:
        return f"personamem_v2_{persona_id}_qa{local_idx}"


@register_converter("personamem_v2")
class PersonaMemV2Converter(_BasePersonaMemV2Converter):
    """PersonaMem-v2 multiple-choice benchmark converter."""

    question_mode = "multiple_choice"
    output_filename = "personamem_v2_32k_locomo_style.json"

    def _build_qa_item(self, question_id: str, row: dict) -> dict:
        query_content = _extract_user_query_content(row["user_query"])
        option_map, correct_label = _build_option_map(
            f"{row['persona_id']}_{query_content}",
            row["correct_answer"],
            row["incorrect_answers"],
        )
        qa_item = self._base_qa_item(question_id, row)
        qa_item.update(
            {
                "answer": correct_label,
                "all_options": option_map,
            }
        )
        return qa_item


@register_converter("personamem_v2_open_ended")
class PersonaMemV2OpenEndedConverter(_BasePersonaMemV2Converter):
    """PersonaMem-v2 open-ended benchmark converter."""

    question_mode = "open_ended"
    output_filename = "personamem_v2_32k_open_ended_locomo_style.json"

    def _build_question_id(self, persona_id: str, local_idx: int) -> str:
        return f"personamem_v2_open_ended_{persona_id}_qa{local_idx}"

    def _build_qa_item(self, question_id: str, row: dict) -> dict:
        qa_item = self._base_qa_item(question_id, row)
        qa_item["answer"] = row["correct_answer"]
        return qa_item
