from evaluation.scripts.analyze_meme_answer_only_evidence import (
    aggregate,
    answer_values,
    verify_answer_only,
)


def question(task_type="Cas", value="multivitamin"):
    return {"task_type": task_type, "entity_values": {"target": value}}


def test_answer_values_are_flattened_normalized_and_deduplicated():
    q = {
        "entity_values": {
            "first": ["Alpha-One", "Beta"],
            "second": {"again": "alpha one"},
        }
    }
    assert answer_values(q) == ("alpha one", "beta")


def test_answer_only_requires_every_value_but_no_path():
    q = {"task_type": "Agg", "entity_values": {"hobby": "pottery", "sport": "climbing"}}
    assert verify_answer_only("Pottery and climbing.", q).sufficient
    partial = verify_answer_only("Only pottery.", q)
    assert not partial.sufficient
    assert partial.missing_values == ("climbing",)


def test_cascade_does_not_require_rule_or_trigger():
    assert verify_answer_only("The value multivitamin appears alone.", question(),).sufficient


def test_tracking_does_not_require_chronology():
    q = question("Tr", ["Alpha", "Beta", "Gamma"])
    assert verify_answer_only("Gamma, Alpha, Beta.", q).sufficient


def test_deletion_after_is_uniform_value_visibility_not_deletion_support():
    q = question("Del", "James")
    assert verify_answer_only("My partner is James.", q).sufficient
    assert not verify_answer_only("Please forget my partner's name.", q).sufficient


def test_aggregate_computes_coverage_and_conditional_utilization():
    rows = [
        {"system": "demo", "answer_only_sufficient": 1, "correct": 1, "status_answer_limitation": 0},
        {"system": "demo", "answer_only_sufficient": 1, "correct": 0, "status_answer_limitation": 1},
        {"system": "demo", "answer_only_sufficient": 0, "correct": 1, "status_answer_limitation": 0},
    ]
    result = aggregate(rows, ("system",))[0]
    assert result["answer_only_coverage"] == 2 / 3
    assert result["reader_utilization"] == 0.5
    assert result["status_answer_limitation_n"] == 1
