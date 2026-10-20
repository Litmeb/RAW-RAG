from evaluation.scripts.build_meme_abs_dependency_oracle import (
    DEPENDENCY_VALIDITY_RULE,
    build_rows,
)


def test_abs_dependency_rule_is_value_free_and_applied_only_to_selected_rows():
    source = [
        {
            "question_id": "abs-question",
            "retrieval_metadata": {"formatted_context": "old evidence"},
        },
        {
            "question_id": "other-question",
            "retrieval_metadata": {"formatted_context": "other evidence"},
        },
    ]

    rows, audit = build_rows(source, {"abs-question"})

    assert len(rows) == len(audit) == 1
    assert rows[0]["question_id"] == "abs-question"
    assert rows[0]["retrieval_metadata"]["formatted_context"] == (
        DEPENDENCY_VALIDITY_RULE + "\nold evidence"
    )
    assert rows[0]["retrieval_metadata"]["abs_rule_contains_gold_values"] is False
