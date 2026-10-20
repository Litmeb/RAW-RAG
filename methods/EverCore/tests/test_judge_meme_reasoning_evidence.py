from __future__ import annotations

from evaluation.scripts.judge_meme_reasoning_evidence import (
    REASONING_TASKS,
    build_prompt,
    evidence_contract,
    parse_judgment,
)


def episode(task_type: str) -> tuple[dict, dict]:
    question = {
        "task_type": task_type,
        "entity": ["target"],
        "entity_values": {"target": ["old", "new"] if task_type == "Tr" else "new"},
        "question": "What is the state?",
    }
    task = {
        "type": task_type,
        "target_entities": ["target"],
        "entity_values": question["entity_values"],
    }
    if task_type in {"Cas", "Abs"}:
        task["cascade_source"] = "source"
    return {
        "entities": {
            "target": {"before": "old", "after": "new"},
            "source": {"before": "off", "after": "on"},
        },
        "tasks": [task],
    }, question


def test_only_reasoning_chain_conditions_are_rejudged() -> None:
    assert ("Tr", "before") in REASONING_TASKS
    assert ("Tr", "after") in REASONING_TASKS
    assert ("Del", "after") in REASONING_TASKS
    assert ("Cas", "after") in REASONING_TASKS
    assert ("Abs", "after") in REASONING_TASKS
    assert ("ER", "after") not in REASONING_TASKS
    assert ("Agg", "after") not in REASONING_TASKS
    assert ("Del", "before") not in REASONING_TASKS
    assert ("Cas", "before") not in REASONING_TASKS
    assert ("Abs", "before") not in REASONING_TASKS


def test_cascade_contract_contains_all_three_required_parts() -> None:
    data, question = episode("Cas")
    contract = evidence_contract(data, question, "after")
    assert contract["new_target_values"] == ["new"]
    assert contract["upstream_entity"] == "source"
    assert contract["upstream_after_values"] == ["on"]
    assert "dependency rule" in contract["sufficiency_rule"]


def test_invalidation_contract_uses_old_target_and_trigger() -> None:
    data, question = episode("Abs")
    contract = evidence_contract(data, question, "after")
    assert contract["old_target_values"] == ["old"]
    assert contract["upstream_after_values"] == ["on"]
    assert "invalidates" in contract["sufficiency_rule"]


def test_prompt_separates_contract_from_retrieved_evidence() -> None:
    data, question = episode("Del")
    prompt = build_prompt("What is the state?", evidence_contract(data, question, "after"), "delete old")
    assert "contract tells you what to look for, but it is NOT evidence" in prompt
    assert "absence of an old value is never proof of deletion" in prompt
    assert prompt.endswith("delete old\n")


def test_parser_requires_new_sufficient_schema() -> None:
    parsed = parse_judgment(
        '{"sufficient": true, "value_complete": true, "temporal_chain": true, '
        '"deletion_targeted": false, "dependency_rule": false, '
        '"upstream_trigger_state": false, "linked": false, "notes": "dated"}'
    )
    assert parsed["valid"] is True
    assert parsed["sufficient"] is True
    assert parse_judgment('{"answerable": true}')["valid"] is False
