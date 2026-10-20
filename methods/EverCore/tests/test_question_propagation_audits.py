from types import SimpleNamespace

from evaluation.tools.meme_question_propagation_common import (
    evaluator_correct,
    index_rows,
    infer_first_signal_gap,
    official_credit,
    question_signal_spec,
    scan_signals,
    summarize,
)


SIGNALS = [
    {
        "kind": "question_value",
        "entity": "medication",
        "value": "Quelmithin",
        "normalized": "quelmithin",
    }
]


def coverage(text: str):
    return scan_signals(SIGNALS, [(text, {"id": 1})], max_examples=2)


def test_signal_scan_reports_all_matches_before_capping_examples():
    result = scan_signals(
        SIGNALS,
        [("Quelmithin", {"id": index}) for index in range(4)],
        max_examples=2,
    )

    signal = result["signals"][0]
    assert result["status"] == "all"
    assert signal["match_count"] == 4
    assert len(signal["match_examples"]) == 2
    assert signal["examples_capped"] is True


def test_correct_answer_does_not_erase_persisted_store_signal_gap():
    assert infer_first_signal_gap(
        coverage("Quelmithin"),
        coverage("unrelated"),
        coverage("unrelated"),
        coverage("unrelated"),
        True,
    ) == "persisted_store_signal_gap"


def test_context_sufficient_wrong_answer_is_noncausal_answer_mismatch():
    complete = coverage("Quelmithin")
    assert (
        infer_first_signal_gap(complete, complete, complete, complete, False)
        == "answer_mismatch_with_all_value_signals"
    )


def test_meme_u_pass_is_used_for_final_correctness():
    assert evaluator_correct({"u_pass": True, "filtered_u_pass": 0.0}) is True
    assert evaluator_correct({"u_pass": False, "u_score": 1.0}) is False


def test_official_credit_excludes_before_and_uses_filtered_after_score():
    assert official_credit({"u_pass": True, "filtered_u_pass": 0.0}, "before") is None
    assert official_credit({"u_pass": True, "filtered_u_pass": 0.0}, "after") == 0.0
    assert official_credit({"u_pass": False, "filtered_u_pass": 1.0}, "after") == 1.0


def test_abstract_dataset_states_are_not_treated_as_literal_text_signals():
    episode = {
        "tasks": [{"type": "Del", "target_entities": ["medication"]}],
        "entities": {"medication": {"before": "none", "after": "deleted"}},
        "dependency_edges_used": [],
    }
    question = {
        "task_type": "Del",
        "entity": ["medication"],
        "entity_values": {"medication": "none"},
    }
    spec = SimpleNamespace(task_type="Del", phase="after")

    result = question_signal_spec(episode, spec, question)

    assert result["required_value_signals"] == []
    assert result["semantic_expectations"]["requires_deletion_semantics"] is True


def test_summary_reports_official_six_task_macro_separately_from_raw_accuracy():
    complete = coverage("Quelmithin")
    rows = []
    for task in ("ER", "Agg", "Tr", "Del", "Cas", "Abs"):
        rows.append(
            {
                "task_type": task,
                "phase": "after",
                "final_correct": True,
                "official_meme_credit": 0.0 if task == "Del" else 1.0,
                "first_signal_gap": "no_signal_gap",
                "official_source": {"coverage": complete},
                "persisted_store_snapshot": {"coverage": complete},
                "native_retrieval": {"coverage": complete},
                "reader_context": {"coverage": complete},
            }
        )

    result = summarize(rows)

    assert result["raw_question_accuracy"] == 1.0
    assert result["official_meme"]["overall_accuracy"] == 5 / 6


def test_nested_meme_evaluation_rows_are_indexed():
    nested = {
        "detailed_results": {
            "Cas": [{"question_id": "q1", "u_pass": True}],
            "Abs": [{"question_id": "q2", "u_pass": False}],
        }
    }

    assert set(index_rows(nested)) == {"q1", "q2"}
