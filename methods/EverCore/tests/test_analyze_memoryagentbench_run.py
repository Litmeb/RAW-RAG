import json

from evaluation.tools.analyze_memoryagentbench_run import (
    ChunkRecord,
    MemCellRecord,
    build_boundary_candidates,
    judge_prompt,
    label_is_issue,
    resolve_memcell_run_dir,
    summarize_judgments,
)


def make_memcell(index, chunk_indices, subject):
    return MemCellRecord(
        conv=0,
        index=index,
        event_id=f"event-{index}",
        timestamp="2026-01-01T00:00:00+00:00",
        subject=subject,
        summary="",
        episode="",
        original_text="",
        original_timestamps=(),
        original_chunk_indices=tuple(chunk_indices),
        atomic_facts=(),
        cluster_id=None,
    )


def test_boundary_candidates_follow_consecutive_source_chunks():
    chunks = [
        ChunkRecord(0, index, f"2026-01-0{index + 1}T00:00:00+00:00", f"chunk {index}")
        for index in range(4)
    ]
    first = make_memcell(10, [0, 1], "first")
    second = make_memcell(20, [2, 3], "second")

    candidates = build_boundary_candidates([chunks], [[second, first]])

    assert [(row["left_chunk"], row["right_chunk"]) for row in candidates] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert [row["observed_decision"] for row in candidates] == [
        "merged",
        "split",
        "merged",
    ]


def test_boundary_prompt_uses_decision_compatible_labels():
    base = {
        "category": "boundary",
        "cross_day": False,
        "left_text": "left",
        "right_text": "right",
    }

    split_prompt, split_labels = judge_prompt({**base, "observed_decision": "split"})
    merge_prompt, merge_labels = judge_prompt({**base, "observed_decision": "merged"})

    assert split_labels == {"correct_split", "false_split"}
    assert '"correct_split|false_split"' in split_prompt
    assert "missed_split" not in split_prompt
    assert merge_labels == {"correct_merge", "missed_split"}
    assert '"correct_merge|missed_split"' in merge_prompt
    assert label_is_issue("boundary", "correct_merge") is False
    assert label_is_issue("boundary", "missed_split") is True


def test_atomic_prompt_treats_source_as_authoritative():
    long_source = "start " + ("middle " * 1000) + "end"
    prompt, _ = judge_prompt(
        {
            "category": "atomic",
            "original_text": long_source,
            "episode": "Lossy summary.",
            "atomic_facts": ["Source-supported fact."],
        }
    )

    assert "SOURCE TEXT is\nthe authoritative evidence" in prompt
    assert "supported by\nSOURCE TEXT is accurate" in prompt
    assert "fact-level count must be zero" in prompt
    assert long_source in prompt
    assert "[truncated]" not in prompt


def test_resolve_memcell_run_dir_from_enriched_manifest(tmp_path):
    source = tmp_path / "source"
    (source / "memcells").mkdir(parents=True)
    enriched = tmp_path / "enriched"
    enriched.mkdir()
    (enriched / "enriched_context_manifest.json").write_text(
        json.dumps({"source_run_dir": str(source)}), encoding="utf-8"
    )

    assert resolve_memcell_run_dir(enriched) == source


def test_boundary_summary_is_split_by_observed_decision():
    results = [
        {
            "category": "boundary",
            "observed_decision": "split",
            "ok": True,
            "judgment": {"label": "correct_split"},
        },
        {
            "category": "boundary",
            "observed_decision": "merged",
            "ok": True,
            "judgment": {"label": "missed_split"},
        },
    ]

    boundary = summarize_judgments(results)["boundary"]

    assert boundary["issues"] == 1
    assert boundary["by_observed_decision"]["split"]["issues"] == 0
    assert boundary["by_observed_decision"]["merged"]["issues"] == 1
