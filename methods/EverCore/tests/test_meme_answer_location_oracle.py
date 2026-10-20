from evaluation.scripts.build_meme_answer_location_oracle import build_rows


def _episode():
    return {
        "episode_id": "fixture",
        "evidence_session_indices": [0, 2],
        "sessions": [
            {"timestamp": "t0", "conversation": [{"role": "user", "content": "old fact"}]},
            {"timestamp": "t1", "conversation": [{"role": "user", "content": "filler"}]},
            {"timestamp": "t2", "conversation": [{"role": "assistant", "content": "new fact"}]},
        ],
        "before_questions": {
            "position_after_session": 1,
            "questions": [{"question": "What was the fact?"}],
        },
        "after_questions": {
            "position_after_session": 2,
            "questions": [{"question": "What is the fact now?"}],
        },
    }


def test_answer_location_oracle_uses_all_and_only_phase_visible_evidence_sessions():
    rows, audit = build_rows([_episode()], 0, 1)

    before, after = rows
    assert before["question_id"] == "meme_fixture_before_qa0"
    assert before["retrieval_metadata"]["selected_evidence_session_indices"] == [0]
    assert "old fact" in before["retrieval_metadata"]["formatted_context"]
    assert "filler" not in before["retrieval_metadata"]["formatted_context"]
    assert "new fact" not in before["retrieval_metadata"]["formatted_context"]

    assert after["retrieval_metadata"]["selected_evidence_session_indices"] == [0, 2]
    assert "old fact" in after["retrieval_metadata"]["formatted_context"]
    assert "new fact" in after["retrieval_metadata"]["formatted_context"]
    assert after["retrieval_metadata"]["context_is_untruncated"] is True
    assert after["retrieval_metadata"]["oracle_uses_question_or_answer_for_selection"] is False
    assert audit[1]["selected_session_count"] == 2
