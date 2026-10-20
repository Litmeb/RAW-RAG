import json
from pathlib import Path

from evaluation.tools.summarize_mem0_add_audit import summarize_audit


def test_summary_separates_effective_duplicate_and_invalid_updates(
    tmp_path: Path,
):
    audit_path = tmp_path / "mem0_add_audit.jsonl"
    record = {
        "status": "success",
        "context": {
            "conversation_id": "conversation-1",
            "speaker": "speaker_a",
            "batch_index": 0,
        },
        "llm_calls": [
            {
                "stage": "memory_action_decision",
                "proposed_actions": [
                    {"event": "UPDATE"},
                    {"event": "UPDATE"},
                    {"event": "UPDATE"},
                    {"event": "DELETE"},
                    {"event": "NONE"},
                ],
            }
        ],
        "vector_searches": [],
        "persisted_history_rows": [
            {"event": "UPDATE", "old_memory": "same", "new_memory": "same"},
            {"event": "UPDATE", "old_memory": "old", "new_memory": "new"},
            {"event": "DELETE"},
        ],
    }
    audit_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    events = summarize_audit(audit_path)["events"]

    assert events["decision_total"] == 5
    assert events["proposed_update_delete"] == {
        "count": 4,
        "share_percent": 80.0,
    }
    assert events["update_validity"] == {
        "proposed": 3,
        "persisted": 2,
        "unpersisted_or_invalid_references": 1,
        "exact_old_new_duplicates": 1,
        "persisted_missing_old_or_new": 0,
        "effective_text_changes": 1,
        "effective_share_of_proposed_percent": 100.0 / 3.0,
        "effective_share_of_all_decisions_percent": 20.0,
    }
    assert events["effective_update_delete"] == {
        "count": 2,
        "share_percent": 40.0,
    }
