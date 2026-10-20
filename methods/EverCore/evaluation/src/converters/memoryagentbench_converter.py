"""
MemoryAgentBench converter.

Converts a cached MemoryAgentBench split/source into LoCoMo-style JSON so the
existing evaluation pipeline can ingest it naturally.
"""

from __future__ import annotations

import json
import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from evaluation.src.converters.base import BaseConverter
from evaluation.src.converters.registry import register_converter
from evaluation.src.datasets.memoryagentbench import (
    format_memoryagentbench_memorize_context,
    source_profile,
)


def _get_token_counter() -> Tuple[Callable[[str], int], Optional[Any]]:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

        def count_tokens(value: str) -> int:
            return len(encoding.encode(value, disallowed_special=()))

        return count_tokens, encoding
    except Exception:

        def count_tokens(value: str) -> int:
            return max(1, len(value.split()))

        return count_tokens, None


def _split_token_window(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    if chunk_size <= 0:
        return [text]

    _, encoding = _get_token_counter()
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    if encoding is not None:
        token_ids = encoding.encode(text, disallowed_special=())
        if not token_ids:
            return [text]
        step = chunk_size - chunk_overlap
        return [
            encoding.decode(token_ids[start : start + chunk_size])
            for start in range(0, len(token_ids), step)
        ]

    words = text.split()
    if not words:
        return [text]
    step = chunk_size - chunk_overlap
    return [" ".join(words[start : start + chunk_size]) for start in range(0, len(words), step)]


def _tail_token_overlap(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""

    _, encoding = _get_token_counter()
    if encoding is not None:
        token_ids = encoding.encode(text, disallowed_special=())
        return encoding.decode(token_ids[-token_count:]) if token_ids else ""

    words = text.split()
    return " ".join(words[-token_count:])


def _chunk_text_into_sentences(text: str, chunk_size: int = 4096) -> List[str]:
    """
    Replicate the benchmark's chunking strategy as closely as possible.

    Official MemoryAgentBench uses NLTK sentence tokenization plus tiktoken-based
    chunk budgets. We mirror that here so incremental "inject once, query many"
    behavior matches the benchmark's evaluation flow better than treating each
    sample as one monolithic message.
    """
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: rough sentence split if NLTK is unavailable.
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

    count_tokens, _ = _get_token_counter()

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0
            chunks.extend(_split_token_window(sentence, chunk_size, 0))
            continue
        if current_sentences and current_tokens + sentence_tokens > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks or [text]


def _chunk_text_by_paragraph(
    text: str, chunk_size: int = 1024, chunk_overlap: int = 128
) -> List[str]:
    if chunk_size <= 0:
        return [text]

    count_tokens, _ = _get_token_counter()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        return _split_token_window(text, chunk_size, chunk_overlap)

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)
        if paragraph_tokens > chunk_size:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.extend(_split_token_window(paragraph, chunk_size, chunk_overlap))
            continue

        if current_parts and current_tokens + paragraph_tokens > chunk_size:
            chunks.append("\n\n".join(current_parts))
            current_parts = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current_parts.append(paragraph)
            current_tokens += paragraph_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks or [text]

    overlapped = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        overlap_text = _tail_token_overlap(previous, chunk_overlap).strip()
        overlapped.append(f"{overlap_text}\n\n{chunk}" if overlap_text else chunk)
    return overlapped


def _chunk_text_by_fact(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return [text]

    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*((?:\[\d+\]|\(\d+\)|\d+[\).:]|Fact\s+\d+[:.]).*?)(?=\n\s*(?:\[\d+\]|\(\d+\)|\d+[\).:]|Fact\s+\d+[:.])|\Z)"
    )
    chunks = [match.group(1).strip() for match in pattern.finditer(stripped)]
    if chunks:
        return chunks

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines or [text]


def _format_longmemeval_session_chunk(marker: str, turns: Any, turn_slice: List[Any]) -> str:
    if isinstance(turns, list):
        return repr([marker, turn_slice])
    return repr([marker, turn_slice])


def _chunk_longmemeval_turns(marker: str, turns: Any, chunk_size: int = 1024) -> List[str]:
    if not isinstance(turns, list) or not turns:
        return [repr([marker, turns])]
    if chunk_size <= 0:
        return [repr([marker, turns])]

    count_tokens, _ = _get_token_counter()
    chunks: List[str] = []
    current_turns: List[Any] = []
    current_tokens = count_tokens(marker)

    for turn in turns:
        turn_tokens = count_tokens(repr(turn))
        if current_turns and current_tokens + turn_tokens > chunk_size:
            chunks.append(_format_longmemeval_session_chunk(marker, turns, current_turns))
            current_turns = [turn]
            current_tokens = count_tokens(marker) + turn_tokens
        else:
            current_turns.append(turn)
            current_tokens += turn_tokens

    if current_turns:
        chunks.append(_format_longmemeval_session_chunk(marker, turns, current_turns))

    return chunks or [repr([marker, turns])]


def _build_context_chunks(
    context: str,
    *,
    chunk_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    if chunk_strategy == "none":
        return [context]
    if chunk_strategy == "token":
        return _split_token_window(context, chunk_size, chunk_overlap)
    if chunk_strategy == "paragraph":
        return _chunk_text_by_paragraph(context, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if chunk_strategy == "fact":
        return _chunk_text_by_fact(context)
    if chunk_strategy == "turn":
        return _chunk_text_into_sentences(context, chunk_size=chunk_size or 1024)
    return _chunk_text_into_sentences(context, chunk_size=chunk_size)


def _parse_longmemeval_chat_time(value: str) -> Optional[datetime]:
    """Parse MemoryAgentBench LongMemEval chat time markers."""
    match = re.search(
        r"Chat Time:\s*(\d{4}/\d{2}/\d{2})\s*\([^)]*\)\s*(\d{2}:\d{2})",
        value,
    )
    if not match:
        return None
    try:
        return datetime.strptime(
            f"{match.group(1)} {match.group(2)}", "%Y/%m/%d %H:%M"
        )
    except ValueError:
        return None


def _format_locomo_session_time(value: datetime) -> str:
    return (
        f"{value.hour % 12 or 12}:{value.strftime('%M')} "
        f"{value.strftime('%p').lower()} on {value.day} "
        f"{value.strftime('%B')}, {value.year}"
    )


def _extract_longmemeval_context_sessions(
    context: str,
    *,
    chunk_strategy: str,
    chunk_size: int,
) -> List[Tuple[datetime, str]]:
    """
    Split MemoryAgentBench LongMemEval context into dated chat sessions.

    The HF field is a string repr of an alternating list:
    ["Chat Time: ...", [{role/content turns}], "Chat Time: ...", ...].
    Keeping these dates is important because query-time temporal filtering
    otherwise treats the synthetic converted session dates as real memory dates.
    """
    try:
        parsed = ast.literal_eval(context)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    sessions: List[Tuple[datetime, str]] = []
    idx = 0
    while idx < len(parsed):
        marker = parsed[idx]
        if isinstance(marker, str) and marker.startswith("Chat Time:"):
            chat_time = _parse_longmemeval_chat_time(marker)
            turns = parsed[idx + 1] if idx + 1 < len(parsed) else []
            if chat_time is not None:
                if chunk_strategy == "turn":
                    for chunk in _chunk_longmemeval_turns(marker, turns, chunk_size=chunk_size):
                        sessions.append((chat_time, chunk))
                else:
                    sessions.append((chat_time, repr([marker, turns])))
            idx += 2
        else:
            idx += 1

    return sessions


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _build_qa_entry(
    sample: Dict[str, Any],
    question_idx: int,
    *,
    split_name: str,
) -> Dict[str, Any]:
    metadata = sample.get("metadata", {}) or {}
    source = str(metadata.get("source", ""))
    profile = source_profile(source)

    questions = _ensure_list(sample.get("questions"))
    answers = _ensure_list(sample.get("answers"))

    question = questions[question_idx] if question_idx < len(questions) else ""
    answer_value = answers[question_idx] if question_idx < len(answers) else ""

    gold_answers = _ensure_list(answer_value)
    primary_answer = gold_answers[0] if gold_answers else ""

    question_types = _ensure_list(metadata.get("question_types"))
    question_ids = _ensure_list(metadata.get("question_ids"))
    question_dates = _ensure_list(metadata.get("question_dates"))
    previous_events = _ensure_list(metadata.get("previous_events"))
    qa_pair_ids = _ensure_list(metadata.get("qa_pair_ids"))

    qa_entry: Dict[str, Any] = {
        "question_id": (
            str(question_ids[question_idx])
            if question_idx < len(question_ids) and question_ids[question_idx] is not None
            else ""
        ),
        "question": question,
        "answer": str(primary_answer),
        "category": (
            str(question_types[question_idx])
            if question_idx < len(question_types) and question_types[question_idx] is not None
            else source
        ),
        "source": source,
        "mab_split": split_name,
        "prompt_key": profile["prompt_key"],
        "evaluation_family": profile["evaluation_family"],
        "gold_answers": [str(item) for item in gold_answers],
    }

    if question_idx < len(question_dates) and question_dates[question_idx]:
        qa_entry["question_date"] = str(question_dates[question_idx])
    if question_idx < len(previous_events) and previous_events[question_idx]:
        qa_entry["previous_event"] = previous_events[question_idx]
    if question_idx < len(qa_pair_ids) and qa_pair_ids[question_idx]:
        qa_entry["qa_pair_id"] = str(qa_pair_ids[question_idx])

    keypoints = metadata.get("keypoints")
    if keypoints:
        qa_entry["keypoints"] = keypoints

    return qa_entry


def convert_memoryagentbench_to_locomo_style(
    samples: List[Dict[str, Any]],
    *,
    split_name: str,
    official_chunking: bool = True,
    chunk_size: int = 4096,
    chunk_overlap: int = 0,
    chunk_strategy: str = "official",
) -> List[Dict[str, Any]]:
    locomo_style: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(samples):
        source = str((sample.get("metadata") or {}).get("source", ""))
        context = str(sample.get("context", ""))

        conversation: Dict[str, Any] = {
            "speaker_a": "MemoryAgentBenchUser",
            "speaker_b": "MemoryAgentBenchSystem",
        }

        dated_context_chunks: List[Tuple[datetime, str]] = []
        resolved_strategy = chunk_strategy if official_chunking else "none"
        if source.startswith("longmemeval"):
            for session_time, session_text in _extract_longmemeval_context_sessions(
                context,
                chunk_strategy=resolved_strategy,
                chunk_size=chunk_size,
            ):
                session_chunks = _build_context_chunks(
                    session_text,
                    chunk_strategy="none" if resolved_strategy == "turn" else resolved_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                offset = len(dated_context_chunks)
                dated_context_chunks.extend(
                    (session_time + timedelta(seconds=offset + chunk_idx), chunk)
                    for chunk_idx, chunk in enumerate(session_chunks)
                )

        if not dated_context_chunks:
            base_time = datetime(2024, 1, 1, 9, 0, 0)
            context_chunks = _build_context_chunks(
                context,
                chunk_strategy=resolved_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            dated_context_chunks = [
                (base_time + timedelta(hours=chunk_idx), chunk)
                for chunk_idx, chunk in enumerate(context_chunks)
            ]

        for chunk_idx, (session_time, chunk) in enumerate(dated_context_chunks):
            session_key = f"session_{chunk_idx}"
            conversation[f"{session_key}_date_time"] = _format_locomo_session_time(session_time)
            content = format_memoryagentbench_memorize_context(
                source,
                chunk,
                time_stamp=session_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            conversation[session_key] = [
                {
                    "speaker": "MemoryAgentBenchUser",
                    "text": content,
                    "dia_id": f"MB{sample_idx}:{chunk_idx}",
                }
            ]

        questions = _ensure_list(sample.get("questions"))
        qa_entries = [
            _build_qa_entry(sample, question_idx, split_name=split_name)
            for question_idx in range(len(questions))
        ]

        locomo_style.append(
            {
                "conversation": conversation,
                "qa": qa_entries,
                "source": source,
            }
        )

    return locomo_style


@register_converter("memoryagentbench")
class MemoryAgentBenchConverter(BaseConverter):
    """MemoryAgentBench dataset converter."""

    META_FILENAME = "memoryagentbench_locomo_style.meta.json"
    CONVERTER_VERSION = 5

    def get_input_files(self) -> Dict[str, str]:
        return {"raw": "memoryagentbench_raw.json"}

    def get_output_filename(self) -> str:
        return "memoryagentbench_locomo_style.json"

    def _get_meta_path(self, data_dir: Path) -> Path:
        return data_dir / self.META_FILENAME

    def needs_conversion(self, data_dir: Path) -> bool:
        output_file = self.get_converted_path(data_dir)
        if not output_file.exists():
            return True

        manifest_path = data_dir / "manifest.json"
        meta_path = self._get_meta_path(data_dir)
        if not manifest_path.exists() or not meta_path.exists():
            return True

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return True

        expected_official_chunking = bool(manifest.get("official_chunking", True))
        expected_chunk_strategy = str(manifest.get("chunk_strategy", "official"))
        expected_chunk_size = int(manifest.get("chunk_size", 4096))
        expected_chunk_overlap = int(manifest.get("chunk_overlap", 0))
        return (
            int(meta.get("converter_version", 1)) != self.CONVERTER_VERSION
            or bool(meta.get("official_chunking", True)) != expected_official_chunking
            or str(meta.get("chunk_strategy", "official")) != expected_chunk_strategy
            or int(meta.get("chunk_size", 4096)) != expected_chunk_size
            or int(meta.get("chunk_overlap", 0)) != expected_chunk_overlap
        )

    def convert(self, input_paths: Dict[str, str], output_path: str) -> None:
        input_dir = Path(input_paths["raw"]).parent
        manifest_path = input_dir / "manifest.json"
        split_name = "unknown"
        official_chunking = True
        chunk_strategy = "official"
        chunk_size = 4096
        chunk_overlap = 0
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            split_name = manifest.get("split", "unknown")
            official_chunking = bool(manifest.get("official_chunking", True))
            chunk_strategy = str(manifest.get("chunk_strategy", "official"))
            chunk_size = int(manifest.get("chunk_size", 4096))
            chunk_overlap = int(manifest.get("chunk_overlap", 0))

        with open(input_paths["raw"], "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        converted = convert_memoryagentbench_to_locomo_style(
            raw_samples,
            split_name=split_name,
            official_chunking=official_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        with open(self._get_meta_path(output_file.parent), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "converter_version": self.CONVERTER_VERSION,
                    "official_chunking": official_chunking,
                    "chunk_strategy": chunk_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(
            f"✅ Converted MemoryAgentBench samples: {len(raw_samples)} raw sample(s) "
            f"-> {len(converted)} LoCoMo-style conversation(s); "
            f"official_chunking={official_chunking}, chunk_strategy={chunk_strategy}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )
