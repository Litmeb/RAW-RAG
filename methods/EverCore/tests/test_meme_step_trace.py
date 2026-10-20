from evaluation.scripts.build_meme_continuity_oracle_runs import MemoryObject
from evaluation.tools.analyze_meme_step_trace import (
    add_transitions,
    add_system_transitions,
    build_evermemos_question_observation,
    build_mem0_question_observation,
    build_memos_question_observation,
    summarize,
    system_scaffold,
)


def test_transitions_keep_multiple_failures_and_recovery() -> None:
    steps = [
        {"step_id": "source", "status": "pass"},
        {"step_id": "hidden", "status": "not_observable"},
        {"step_id": "store", "status": "fail"},
        {"step_id": "retrieval", "status": "fail"},
        {"step_id": "context", "status": "pass"},
        {"step_id": "answer", "status": "fail"},
    ]

    assert add_transitions(steps) == [
        {"from": "source", "to": "store", "kind": "new_error", "unobserved_steps_between": 1},
        {"from": "store", "to": "retrieval", "kind": "propagated", "unobserved_steps_between": 0},
        {"from": "retrieval", "to": "context", "kind": "recovered", "unobserved_steps_between": 0},
        {"from": "context", "to": "answer", "kind": "new_error", "unobserved_steps_between": 0},
    ]


def test_summary_excludes_unobservable_from_error_rate() -> None:
    rows = [
        {
            "final_correct": False,
            "steps": [
                {"step_id": "store", "status": "fail"},
                {"step_id": "internal", "status": "not_observable"},
                {"step_id": "replay", "status": "replayable"},
                {"step_id": "answer", "status": "fail"},
            ],
            "transitions": [{"kind": "propagated"}],
        },
        {
            "final_correct": True,
            "steps": [
                {"step_id": "store", "status": "pass"},
                {"step_id": "internal", "status": "not_observable"},
                {"step_id": "replay", "status": "replayable"},
                {"step_id": "answer", "status": "pass"},
            ],
            "transitions": [{"kind": "preserved"}],
        },
    ]

    result = summarize(rows)

    assert result["step_error_rates"]["store"] == 0.5
    assert result["step_error_rates"]["internal"] is None
    assert result["step_error_rates"]["replay"] is None
    assert result["failed_observable_steps_per_question"] == {0: 1, 2: 1}
    assert result["transition_counts"] == {"propagated": 1, "preserved": 1}


def _mem0_memory(memory_id: str, content: str, memory_hash: str) -> MemoryObject:
    return MemoryObject(
        content=content,
        score=0.0,
        metadata={
            "memory_id": memory_id,
            "source_session_indices": [0],
            "store_payload": {
                "hash": memory_hash,
                "user_id": "normal__meme_0_User",
                "created_at": "2023-03-01T09:00:00",
            },
        },
        first_seen=0,
    )


def test_mem0_observation_exposes_history_duplicates_entities_and_cutoff() -> None:
    first = _mem0_memory("memory-1", "User drives the Blue Car.", "same-hash")
    duplicate = _mem0_memory("memory-2", "User drives the Blue Car.", "same-hash")
    artifact = {
        "history_add_records": {
            "memory-1": {"memory_id": "memory-1"},
            "memory-2": {"memory_id": "memory-2"},
        },
        "entity_by_memory": {
            "memory-1": [
                {"entity_id": "entity-1", "text": "Blue Car", "entity_type": "PROPER"}
            ]
        },
        "broken_entity_links": [],
        "residual_duplicate_groups": {
            ("normal__meme_0_User", "same-hash"): ["memory-1", "memory-2"]
        },
    }
    row = {
        "question": "What do I drive?",
        "cutoff_session": 0,
        "meme_metadata": {
            "required_value_signals": [{"normalized": "blue car"}]
        },
        "official_source": {"evidence_session_indices": [0]},
        "native_retrieval": {
            "results": [
                {
                    "content": first.content,
                    "metadata": {
                        "raw": {
                            "id": "memory-1",
                            "created_at": "2023-03-02T09:00:00",
                        }
                    },
                }
            ]
        },
    }

    result = build_mem0_question_observation(
        row,
        [first, duplicate],
        [first, duplicate],
        artifact,
        [("2023-03-01T09:00:00", 0), ("2023-03-02T09:00:00", 1)],
    )
    steps = {step["step_id"]: step for step in result["deterministic_steps"]}

    assert steps["post_dedup_history_lineage"]["status"] == "pass"
    assert steps["task_relevant_residual_exact_duplicates"]["status"] == "fail"
    assert steps["entity_link_integrity"]["status"] == "pass"
    assert steps["entity_link_integrity"]["details"]["memories_without_entity_links"] == 1
    assert steps["temporal_filter_integrity"]["status"] == "fail"
    semantic = result["semantic_artifact"]["task_relevant_memories_with_entities"]
    assert semantic[0]["entities"][0]["text"] == "Blue Car"


def test_mem0_scaffold_distinguishes_not_observable_replayable_and_inactive() -> None:
    scaffold = system_scaffold(
        "mem0_local",
        {},
        mem0_observation={
            "deterministic_steps": [],
            "semantic_artifact": {"entity_boost_executed": False},
        },
    )
    statuses = {step["step_id"]: step["status"] for step in scaffold}

    assert statuses["write_existing_candidates_original"] == "not_observable"
    assert statuses["write_existing_candidates_replay"] == "replayable"
    assert statuses["raw_llm_extraction_response"] == "not_observable"
    assert statuses["entity_link_task_support"] == "not_applicable"
    assert statuses["retrieval_components_replay"] == "replayable"


def _memos_memory(memory_id: str, sources: list[dict], **metadata_overrides) -> MemoryObject:
    store_metadata = {
        "sources": sources,
        "background": "Vehicle discussion",
        "key": "Vehicle",
        "tags": ["vehicle"],
        "memory_type": "UserMemory",
        "status": "activated",
        "history": [],
        "evolve_to": [],
        **metadata_overrides,
    }
    return MemoryObject(
        content="The user drives the Blue Car.",
        score=0.0,
        metadata={
            "memory_id": memory_id,
            "source_session_indices": [0],
            "store_metadata": store_metadata,
        },
        first_seen=0,
    )


def test_memos_docker_observation_exposes_windows_vectors_and_cutoff_leak() -> None:
    evidence_sources = [
        {
            "role": "user",
            "content": "I drive the Blue Car.",
            "chat_time": "2023-03-01T09:00:00+00:00",
        },
        {
            "role": "assistant",
            "content": "Noted.",
            "chat_time": "2023-03-01T09:00:30+00:00",
        },
    ]
    memory = _memos_memory("memory-1", evidence_sources)
    row = {
        "cutoff_session": 0,
        "task_type": "Del",
        "official_source": {
            "sessions": [
                {
                    "session_index": 0,
                    "timestamp": "2023/03/01 (Wed) 09:00",
                    "conversation": [
                        {"role": "user", "content": "I drive the Blue Car."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                }
            ]
        },
        "native_retrieval": {
            "results": [
                {
                    "content": memory.content,
                    "metadata": {
                        "memory_id": "memory-1",
                        "raw": {
                            "metadata": {
                                "sources": [
                                    *evidence_sources,
                                    {
                                        "role": "user",
                                        "content": "Future update.",
                                        "chat_time": "2023-03-02T09:00:00+00:00",
                                    },
                                ]
                            }
                        },
                    },
                }
            ]
        },
    }

    observation = build_memos_question_observation(
        row,
        [memory],
        [memory],
        {"memory-1"},
        [("2023-03-01T09:00:00", 0), ("2023-03-02T09:00:00", 1)],
        max_store_chars=100_000,
    )
    steps = {step["step_id"]: step for step in observation["deterministic_steps"]}

    assert steps["multimodal_window_provenance"]["status"] == "pass"
    assert steps["graph_vector_persistence"]["status"] == "pass"
    assert steps["state_evolution_mechanism"]["status"] == "not_applicable"
    assert steps["temporal_filter_integrity"]["status"] == "fail"
    assert steps["temporal_filter_integrity"]["details"]["cross_cutoff_results"] == 1
    assert observation["semantic_artifact"]["state_change_task"] is True


def test_memos_vector_mismatch_is_a_deterministic_failure() -> None:
    sources = [
        {
            "role": "user",
            "content": "I drive the Blue Car.",
            "chat_time": "2023-03-01T09:00:00+00:00",
        }
    ]
    memory = _memos_memory("missing-vector", sources)
    row = {
        "cutoff_session": 0,
        "task_type": "ER",
        "official_source": {
            "sessions": [
                {
                    "session_index": 0,
                    "timestamp": "2023/03/01 (Wed) 09:00",
                    "conversation": [{"role": "user", "content": "I drive the Blue Car."}],
                }
            ]
        },
        "native_retrieval": {"results": []},
    }

    observation = build_memos_question_observation(
        row,
        [memory],
        [memory],
        set(),
        [("2023-03-01T09:00:00", 0)],
        max_store_chars=100_000,
    )
    steps = {step["step_id"]: step for step in observation["deterministic_steps"]}

    assert steps["graph_vector_persistence"]["status"] == "fail"
    assert steps["graph_vector_persistence"]["details"]["missing_vector_ids"] == [
        "missing-vector"
    ]


def _evermemos_row() -> dict:
    return {
        "cutoff_session": 0,
        "official_source": {
            "sessions": [
                {
                    "session_index": 0,
                    "timestamp": "2023/03/01 (Wed) 09:00",
                    "conversation": [
                        {"role": "user", "content": "I drive the Blue Car."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                }
            ]
        },
    }


def _evermemos_artifact(source_sessions: list[int] | None = None) -> dict:
    cell = {
        "event_id": "event-1",
        "source_session_indices": source_sessions or [0],
        "messages": [
            {
                "sender_name": "User",
                "content": "I drive the Blue Car.",
                "timestamp": "2023-03-01T09:00:00+00:00",
            },
            {
                "sender_name": "Assistant",
                "content": "Noted.",
                "timestamp": "2023-03-01T09:00:30+00:00",
            },
        ],
        "subject": "Vehicle",
        "summary": "The user drives the Blue Car.",
        "episode": "The user drives the Blue Car.",
        "atomic_facts": ["The user drives the Blue Car."],
        "timestamp": "2023-03-01T09:00:30+00:00",
        "atomic_metadata": {
            "timestamp": "2023-03-01T09:00:30+00:00",
            "time": "March 01, 2023 at 09:00 AM",
        },
    }
    indexed = {
        "event_id": "event-1",
        "subject": cell["subject"],
        "summary": cell["summary"],
        "episode": cell["episode"],
        "atomic_fact": {
            "atomic_fact": cell["atomic_facts"],
            **cell["atomic_metadata"],
        },
    }
    return {
        "memcells": [cell],
        "vector_docs": {"event-1": indexed},
        "bm25_docs": {"event-1": indexed},
        "paths": {"memcells": "m", "vector": "v", "bm25": "b"},
    }


def test_evermemos_saved_memcell_steps_are_observable() -> None:
    result = build_evermemos_question_observation(
        _evermemos_row(), _evermemos_artifact()
    )
    steps = {step["step_id"]: step for step in result["deterministic_steps"]}

    assert steps["memcell_message_integrity"]["status"] == "pass"
    assert steps["memcell_cutoff_visibility"]["status"] == "pass"
    assert steps["episode_index_integrity"]["status"] == "pass"
    assert steps["atomic_fact_index_integrity"]["status"] == "pass"
    assert len(result["semantic_artifact"]["relevant_memcells"]) == 1


def test_evermemos_cross_cutoff_memcell_is_visibility_failure() -> None:
    result = build_evermemos_question_observation(
        _evermemos_row(), _evermemos_artifact([0, 1])
    )
    steps = {step["step_id"]: step for step in result["deterministic_steps"]}

    assert steps["memcell_message_integrity"]["status"] == "pass"
    assert steps["memcell_cutoff_visibility"]["status"] == "fail"
    assert steps["memcell_cutoff_visibility"]["details"]["invisible_message_count"] == 2


def test_evermemos_transitions_do_not_cross_parallel_representation_branches() -> None:
    statuses = {
        "official_source": "pass",
        "raw_boundary_decision_trace": "not_observable",
        "memcell_message_integrity": "pass",
        "memcell_cutoff_visibility": "pass",
        "memcell_partition_quality": "pass",
        "episode_fidelity": "pass",
        "episode_task_support": "pass",
        "episode_index_integrity": "pass",
        "atomic_fact_fidelity": "pass",
        "atomic_fact_task_support": "fail",
        "atomic_fact_index_integrity": "pass",
        "round1_hybrid_candidates": "not_observable",
        "round1_rerank": "not_observable",
        "sufficiency_decision": "not_observable",
        "query_refinement": "pass",
        "round2_hybrid_rrf": "not_observable",
        "final_merge_rerank": "not_observable",
        "native_retrieval": "pass",
        "reader_context": "pass",
        "answer": "pass",
    }
    transitions = add_system_transitions(
        "evermemos",
        [{"step_id": step_id, "status": status} for step_id, status in statuses.items()],
    )
    edges = {(item["from"], item["to"]) for item in transitions}

    assert ("episode_task_support", "atomic_fact_fidelity") not in edges
    assert ("atomic_fact_task_support", "episode_index_integrity") not in edges
    assert ("memcell_partition_quality", "episode_fidelity") in edges
    assert ("memcell_partition_quality", "atomic_fact_fidelity") in edges
