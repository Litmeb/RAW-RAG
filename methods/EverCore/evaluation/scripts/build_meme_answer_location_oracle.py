"""Build an untruncated raw-evidence oracle for MEME.

The dataset's ``evidence_session_indices`` are treated as oracle answer
locations.  For each question, this builder exposes every indexed session that
is visible at that question's before/after cutoff, in chronological order.
It deliberately does *not* use the question, gold answer, entity values, a
retrieval score, or a token cap to select among those sessions.

The resulting ``search_results.json`` can be answered and judged through the
normal CLI fast path, e.g.::

    python -m evaluation.cli --dataset meme --system rag \
      --output-dir evaluation/results/meme-rag-answer-location-oracle-<tag> \
      --stages answer evaluate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA = EVALUATION_DIR / "data/meme/meme_filler32k.json"
DEFAULT_OUTPUT_ROOT = EVALUATION_DIR / "results"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def session_text(session: dict[str, Any], index: int) -> str:
    """Format one complete source session without changing message content."""
    messages = [
        f"{str(message.get('role') or '').capitalize()}: "
        f"{str(message.get('content') or '').strip()}"
        for message in session.get("conversation") or []
    ]
    return (
        f"<ORACLE_EVIDENCE_SESSION index={index} "
        f"time={session.get('timestamp', '')}>\n"
        + "\n".join(messages)
        + "\n</ORACLE_EVIDENCE_SESSION>"
    )


def build_rows(
    episodes: list[dict[str, Any]], start_conv: int, end_conv: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build saved-search rows and an audit record for every selected question."""
    if not 0 <= start_conv < end_conv <= len(episodes):
        raise ValueError("Invalid conversation range")

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for episode_index in range(start_conv, end_conv):
        episode = episodes[episode_index]
        episode_id = str(episode["episode_id"])
        sessions = episode.get("sessions") or []
        evidence_indices = [int(index) for index in episode.get("evidence_session_indices") or []]
        if evidence_indices != sorted(set(evidence_indices)):
            raise ValueError(
                f"{episode_id}: evidence_session_indices must be sorted and unique"
            )
        invalid = [index for index in evidence_indices if index < 0 or index >= len(sessions)]
        if invalid:
            raise ValueError(f"{episode_id}: invalid evidence session indices: {invalid}")

        for phase in ("before", "after"):
            question_block = episode[f"{phase}_questions"]
            cutoff = int(question_block["position_after_session"])
            selected_indices = [index for index in evidence_indices if index <= cutoff]
            formatted_sessions = [session_text(sessions[index], index) for index in selected_indices]
            context = "\n\n".join(formatted_sessions)
            results = [
                {
                    "content": session_text(sessions[index], index),
                    "score": 1.0,
                    "metadata": {
                        "source": "meme_dataset_evidence_session_indices",
                        "session_index": index,
                        "timestamp": sessions[index].get("timestamp"),
                    },
                }
                for index in selected_indices
            ]
            for question_index, question in enumerate(question_block.get("questions") or []):
                question_id = f"meme_{episode_id}_{phase}_qa{question_index}"
                rows.append(
                    {
                        "question_id": question_id,
                        "query": str(question.get("question") or ""),
                        "conversation_id": f"meme_{episode_index}",
                        "results": results,
                        "retrieval_metadata": {
                            "system": "meme_answer_location_raw_context_oracle",
                            "formatted_context": context,
                            "oracle_source": "dataset.evidence_session_indices",
                            "oracle_selection": (
                                "all evidence-session locations visible at the question cutoff"
                            ),
                            "oracle_uses_question_or_answer_for_selection": False,
                            "context_is_untruncated": True,
                            "question_phase": phase,
                            "question_cutoff_session": cutoff,
                            "dataset_evidence_session_indices": evidence_indices,
                            "selected_evidence_session_indices": selected_indices,
                        },
                    }
                )
                audit.append(
                    {
                        "question_id": question_id,
                        "phase": phase,
                        "cutoff_session": cutoff,
                        "dataset_evidence_session_indices": evidence_indices,
                        "selected_evidence_session_indices": selected_indices,
                        "selected_session_count": len(selected_indices),
                        "context_characters": len(context),
                    }
                )
    return rows, audit


def create_output(output_dir: Path, rows: list[dict[str, Any]], audit: list[dict[str, Any]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse result directory: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "search_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "answer_location_oracle_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-conv", type=int, default=0)
    parser.add_argument("--end-conv", type=int, default=100)
    args = parser.parse_args()

    episodes = load_json(args.data.resolve())
    rows, audit = build_rows(episodes, args.start_conv, args.end_conv)
    output_dir = args.output_dir.resolve()
    create_output(output_dir, rows, audit)
    print(f"Built answer-location oracle: {output_dir} ({len(rows)} questions)")


if __name__ == "__main__":
    main()
