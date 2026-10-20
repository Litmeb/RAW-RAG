"""Budget removal must preserve legacy selection and retain every admitted object."""

from pathlib import Path
import sys

import tiktoken

SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parents[1] / "src"))
import build_meme_continuity_oracle_runs as base  # noqa: E402
import build_meme_complete_store_oracle_runs as complete  # noqa: E402


def encoding():
    return tiktoken.get_encoding("o200k_base")


def candidates():
    return [
        (base.MemoryObject("alpha " * 200, 0, {
            "source_session_indices": [2], "start_token": 200,
            "end_token": 400, "chunk_index": 1,
        }, 0), 2.0, ["alpha"]),
        (base.MemoryObject("beta", 0, {
            "source_session_indices": [0], "start_token": 0,
            "end_token": 200, "chunk_index": 0,
        }, 1), 1.0, ["beta"]),
    ]


def spec():
    return base.QuestionSpec(
        "q", "meme_0", 0, "after", "alpha", "ER", ("alpha",),
        {"alpha": "alpha"}, "alpha", "", 2, ("alpha",),
    )


def test_uncapped_memory_keeps_oversize_and_tail():
    ranked = candidates()
    capped, context = base.pack_memory_context(ranked, encoding(), 50)
    full, whole = base.pack_memory_context(ranked, encoding(), None)
    assert len(capped) == 1
    assert len(encoding().encode(context)) == 50
    assert len(full) == 2
    assert whole == "".join(f"<MEMORY>\n{m.content}\n</MEMORY>\n\n" for m, _, _ in ranked)


def test_uncapped_emem_keeps_chronology():
    selected, context = complete.pack_emem_context(candidates(), encoding(), None)
    assert len(selected) == 2
    assert selected[0]["content"] == "beta"
    assert context.index("beta") < context.index("alpha")


def test_uncapped_rag_keeps_every_whole_chunk():
    selected, context, skipped = complete.pack_rag_chunk_context(candidates(), encoding(), None)
    assert len(selected) == 2
    assert skipped == 0
    assert context.index("beta") < context.index("alpha")
    assert "alpha " * 200 in context
    capped, _, skipped = complete.pack_rag_chunk_context(candidates(), encoding(), 80)
    assert len(capped) == 1 and capped[0]["content"] == "beta"
    assert skipped == 1


def test_uncapped_rag_source_ties_match_legacy_greedy_order():
    ranked = candidates()
    ranked[0][0].metadata["start_token"] = 0
    ranked[1][0].metadata["start_token"] = 0
    no_cap = complete.pack_rag_chunk_context(ranked, encoding(), None)
    large_cap = complete.pack_rag_chunk_context(ranked, encoding(), 100000)
    assert no_cap == large_cap


def test_uncapped_raw_preserves_gold_selection_not_all_sessions():
    episode = {"evidence_session_indices": [0, 1], "sessions": [
        {"conversation": [{"role": "user", "content": "alpha " * 200}]},
        {"conversation": [{"role": "user", "content": "irrelevant"}]},
    ]}
    selected, context, visible = base.pack_raw_context(episode, spec(), encoding(), None)
    assert len(selected) == 1
    assert visible == [0, 1]
    assert "alpha " * 199 in context
    assert "irrelevant" not in context


def test_none_budget_survives_policy_reservation():
    assert base.evidence_budget(encoding(), None, spec(), True) is None
    assert 0 < base.evidence_budget(encoding(), 5000, spec(), True) < 5000


def test_paired_bootstrap_preserves_episode_and_task_weighting():
    from analyze_meme_complete_store_uncapped import paired

    rows = [
        {"question_id": f"{c}-{task}", "conversation_id": c,
         "phase": "after", "task_type": task, "score": 1}
        for c in ("meme_0", "meme_1") for task in ("ER", "Cas")
    ]
    result = paired(rows, [{**r, "score": 0} for r in rows], samples=100)
    assert result["difference"] == 1
    assert result["ci95_low"] == result["ci95_high"] == 1
    assert result["tasks"] == ["ER", "Cas"]


def test_saved_judgment_before_gate_must_be_recomputed():
    import pytest
    from analyze_meme_complete_store_uncapped import scores

    rows = [
        {"question_id": "before", "conversation_id": "meme_0", "entity_key": "x",
         "phase": "before", "task_type": "Cas", "u_pass": False},
        {"question_id": "after", "conversation_id": "meme_0", "entity_key": "x",
         "phase": "after", "task_type": "Cas", "u_pass": True, "filtered_u_pass": 0},
    ]
    assert scores({"detailed_results": rows})[1]["score"] == 0
    rows[1]["filtered_u_pass"] = 1
    with pytest.raises(ValueError, match="before gate"):
        scores({"detailed_results": rows})


def test_reader_message_matches_shared_answer_stage():
    from run_meme_complete_store_uncapped import reader_messages
    from evaluation.src.core.data_models import QAPair
    from evaluation.src.core.stages.answer_stage import _build_answer_query

    answer = {"question": "Which value?", "metadata": {"question_date": "March 19, 2023"}}
    qa = QAPair("q", answer["question"], "x", metadata=answer["metadata"])
    expected = _build_answer_query(qa, None)
    messages = reader_messages(answer, "context", "{context}\n{question}")
    assert messages == [{"role": "user", "content": f"context\n{expected}"}]


def test_filtered_response_is_not_a_fabricated_judge_call():
    from run_meme_complete_store_uncapped import refusal_judgment

    original = {"question_id": "q", "u_pass": True, "u_score": 1,
                "judge_runs": [{"u_pass": True}], "filtered_u_pass": 1,
                "generated_answer": "old", "u_pass_count": 2}
    result = refusal_judgment(original)
    assert result["u_pass"] is False and result["generated_answer"] == ""
    assert result["judge_runs"] == []
    assert "u_pass_count" not in result and "filtered_u_pass" not in result
    assert original["u_pass"] is True


def test_memmachine_unknown_citations_are_not_empty_citations(tmp_path):
    from run_meme_complete_store_uncapped import memmachine_store, save

    export = {"meme_0": {
        "session_key": "org/project", "episodic_memory": [],
        "semantic_memory": [{"metadata": {"id": "7"}, "value": "value"}],
        "working_memory_summary": {},
    }}
    save(tmp_path / "memmachine_complete_store_export.json", export)
    save(tmp_path / "search_results.json", [])
    save(tmp_path / "complete_store_audit.json", {"total_objects": 1})
    maps = {"meme_0": [("2023-01-01T00:00:00", 0), ("2023-01-02T00:00:00", 1)]}
    store, audit = memmachine_store(tmp_path, ["meme_0"], maps)
    assert len(audit["unresolved_archived_semantic_provenance"]) == 1
    assert store["meme_0"][0].metadata["source_session_indices"] == []
    citations = tmp_path / "citations.json"
    save(citations, {"citations": {"7": []}})
    store, audit = memmachine_store(tmp_path, ["meme_0"], maps, citations)
    assert audit["unresolved_archived_semantic_provenance"] == []
    assert store["meme_0"][0].metadata["source_session_indices"] == [0, 1]
    save(citations, {"citations": {"7": [{"metadata": {"session": "session_0"}}]}})
    store, audit = memmachine_store(tmp_path, ["meme_0"], maps, citations)
    assert store["meme_0"][0].metadata["source_session_indices"] == [0]


def test_report_rejects_substituted_reader_or_judge(tmp_path):
    import pytest
    from analyze_meme_complete_store_uncapped import verify_requests
    from run_meme_complete_store_uncapped import READER_RESPONSE_MODEL, digest, save

    save(tmp_path / "manifest.json", {"judge_model": "gpt-4o-2024-11-20"})
    save(tmp_path / "answer_results.json", [{"question_id": "q", "formatted_context": "text"}])
    audit = [{"question_id": "q", "context_sha256": digest("text"), "context_changed": True,
              "task_type": "Cas", "prompt_tokens_estimate": 10, "request_sha256": "request"}]
    entry = {
        "status": "complete", "reuse": "none",
        "judge_requests": [{"response_model": "gpt-4o-2024-11-20"}],
        "request": {
            "response_model": READER_RESPONSE_MODEL, "request_sha256": "request",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    }
    save(tmp_path / "qa_checkpoint.json", {"q": entry})
    assert verify_requests(tmp_path, audit)["server_prompt_tokens_exact_match"]
    entry["request"]["response_model"] = "gpt-4.1-mini-2025-04-14"
    save(tmp_path / "qa_checkpoint.json", {"q": entry})
    with pytest.raises(ValueError, match="substituted reader"):
        verify_requests(tmp_path, audit)
    entry["request"]["response_model"] = READER_RESPONSE_MODEL
    entry["judge_requests"][0]["response_model"] = "gpt-4o-mini"
    save(tmp_path / "qa_checkpoint.json", {"q": entry})
    with pytest.raises(ValueError, match="substituted judge"):
        verify_requests(tmp_path, audit)


def test_cli_preserves_non_ascii_provider_errors(tmp_path, monkeypatch):
    import io
    import pytest
    import run_meme_complete_store_uncapped as runner

    message = "\u7528\u6237\u989d\u5ea6\u4e0d\u8db3"
    buffer = io.BytesIO()
    output = io.TextIOWrapper(buffer, encoding="cp1252")

    def fail_build(*args):
        raise ValueError(message)

    monkeypatch.setattr(runner, "EVAL", tmp_path)
    monkeypatch.setattr(runner, "build", fail_build)
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "argv", [
        "runner", "build", "--system", "raw", "--source-manifest", "sources.json",
    ])
    with pytest.raises(SystemExit) as error:
        runner.main()
    output.flush()
    assert error.value.code == 1
    assert message in buffer.getvalue().decode("utf-8")
    failures = runner.load(tmp_path / "analysis" / runner.TAG / "build_failures.json")
    assert failures == [{"system": "raw", "error": f"ValueError: {message}"}]
