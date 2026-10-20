from evaluation.scripts.analyze_meme_task_specific_evidence import (
    parse_decomposition_spec,
    representation_decomposition,
    verify,
)


def episode(task_type, target, before, after=None, source=None, source_before=None, source_after=None):
    entities = {target: {"before": before, "after": after}}
    task = {"type": task_type, "target_entities": [target], "entity_values": {target: after or before}}
    if source:
        task["cascade_source"] = source
        entities[source] = {"before": source_before, "after": source_after}
    return {"entities": entities, "tasks": [task]}


def question(task_type, target, value):
    return {"task_type": task_type, "entity": [target], "entity_values": {target: value}}


def test_deletion_is_phase_aware_and_reports_non_disclosure():
    data = episode("Del", "partner", "James", "deleted")
    q = question("Del", "partner", "James")
    assert verify("My partner is James.", data, q, "before").sufficient
    after = verify("Please remove my partner's name from memory.", data, q, "after")
    assert after.sufficient
    assert after.deletion_event_support
    assert after.non_disclosure


def test_tracking_requires_chronology_not_just_all_values():
    data = episode("Tr", "vehicle", "Alpha", "Gamma")
    data["entities"]["vehicle"]["history"] = ["Alpha", "Beta", "Gamma"]
    data["tasks"][0]["entity_values"] = {"vehicle": ["Alpha", "Beta", "Gamma"]}
    q = question("Tr", "vehicle", ["Alpha", "Beta", "Gamma"])
    assert not verify("Beta. Alpha. Gamma.", data, q, "after").sufficient
    result = verify(
        "2023-03-01: Alpha. 2023-03-02: Beta. 2023-03-03: Gamma.", data, q, "after"
    )
    assert result.sufficient
    assert result.order_support
    iso_result = verify(
        "2023-03-01T10:00:00Z Alpha. 2023-03-02T10:00:00Z Beta. "
        "2023-03-03T10:00:00Z Gamma.",
        data,
        q,
        "after",
    )
    assert iso_result.sufficient


def test_cascade_requires_rule_trigger_and_replacement():
    data = episode(
        "Cas", "medication", "Quelmithin", "multivitamin",
        source="health_condition", source_before="lactose intolerance", source_after="high blood pressure",
    )
    q = question("Cas", "medication", "multivitamin")
    incomplete = verify("I take multivitamin. I have high blood pressure.", data, q, "after")
    assert not incomplete.sufficient
    complete = verify(
        "If my health condition changes, my medication will change to multivitamin. "
        "My health condition changed to high blood pressure.",
        data, q, "after",
    )
    assert complete.sufficient


def test_absence_requires_dependency_and_invalidating_trigger():
    data = episode(
        "Abs", "appointment", "weekly tutoring", None,
        source="health_condition", source_before="lactose intolerance", source_after="high blood pressure",
    )
    q = question("Abs", "appointment", "weekly tutoring")
    complete = verify(
        "Weekly tutoring depends on my health condition. My health condition changed to high blood pressure.",
        data, q, "after",
    )
    assert complete.sufficient
    assert complete.dependency_rule_support
    assert complete.trigger_support


def test_er_and_aggregation_require_complete_values():
    er = episode("ER", "motto", "Exact words")
    assert verify("The motto is Exact words.", er, question("ER", "motto", "Exact words"), "before").sufficient
    agg = {
        "entities": {"hobby": {"before": "pottery"}, "sport": {"before": "climbing"}},
        "tasks": [{"type": "Agg", "target_entities": ["hobby", "sport"], "entity_values": {"hobby": "pottery", "sport": "climbing"}}],
    }
    q = {"task_type": "Agg", "entity": ["hobby", "sport"], "entity_values": {"hobby": "pottery", "sport": "climbing"}}
    assert verify("Pottery and climbing.", agg, q, "after").sufficient
    assert not verify("Only pottery.", agg, q, "after").sufficient


def test_representation_retrieval_readout_decomposition():
    records = []
    conditions = {
        "raw": [(1, 1), (1, 1), (1, 0), (1, 1)],
        "store": [(1, 1), (1, 0), (1, 0), (0, 0)],
        "native": [(1, 1), (1, 0), (0, 0), (0, 0)],
    }
    for label, values in conditions.items():
        for index, (sufficient, correct) in enumerate(values):
            records.append(
                {
                    "system": label,
                    "question_id": f"q{index}",
                    "evidence_sufficient": sufficient,
                    "correct": correct,
                }
            )

    spec = parse_decomposition_spec("demo=raw,store,native")
    row = representation_decomposition(records, [spec])[0]

    assert row["raw_coverage"] == 1.0
    assert row["store_coverage"] == 0.75
    assert row["native_coverage"] == 0.5
    assert row["representation_loss_raw_minus_store"] == 0.25
    assert row["retrieval_loss_store_minus_native"] == 0.25
    assert row["native_reader_utilization"] == 0.5
    assert row["native_readout_loss_sufficient_but_wrong"] == 0.25
