"""Diagnostic controls can be prepared from new runs, without historical reports."""

import json
import sys
from types import SimpleNamespace

import pytest

from evaluation.scripts import build_rag_5k_intervention_runs as cap
from evaluation.scripts import build_meme_complete_store_oracle_runs as complete
from evaluation.scripts import run_meme_complete_store_uncapped as uncapped


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_5k_manifest_builds_from_new_run_with_nonhistorical_name(tmp_path, monkeypatch):
    source = tmp_path / "new-reader-run"
    row = {
        "question_id": "q1", "query": "question", "conversation_id": "conv",
        "results": [{"content": "raw evidence", "metadata": {
            "chunk_index": 0, "similarity_score": 0.9, "rerank_score": 0.8,
        }}],
        "retrieval_metadata": {"formatted_context": "expanded " * 5000},
    }
    write(source / "search_results.json", [row])
    write(source / "answer_results.json", [{
        "question_id": "q1", "formatted_context": row["retrieval_metadata"]["formatted_context"],
    }])
    write(source / "eval_results.json", {"metadata": {"model": "judge", "num_runs": 3}})
    manifest = tmp_path / "source-map.json"
    write(manifest, {"locomo": {
        "source": "new-reader-run", "questions": 1, "answer_model": "reader",
    }})
    output = tmp_path / "fresh-results"
    monkeypatch.setattr(sys, "argv", [
        "builder", "--source-manifest", str(manifest), "--results-root", str(output),
        "--run-name", "fresh", "--budget", "100",
    ])
    cap.main()
    summaries = cap.load_json(output / "rag_context_cap_fresh.json")
    assert len(summaries) == 1
    assert summaries[0]["source_run"] == str(source.resolve())
    assert summaries[0]["answer_model"] == "reader"
    assert summaries[0]["questions"] == 1
    assert summaries[0]["max_tokens_after"] <= 100
    assert not (source / "context_cap_summary.json").exists()


@pytest.mark.parametrize("spec", [
    {}, {"source": "run", "questions": 0, "answer_model": "reader"},
    {"source": "run", "questions": 1},
])
def test_5k_manifest_requires_explicit_counts_and_reader(tmp_path, spec):
    manifest = tmp_path / "sources.json"
    write(manifest, {"locomo": spec})
    with pytest.raises(ValueError):
        cap.load_source_manifest(manifest)


def test_uncapped_replays_fresh_complete_store_run(tmp_path, monkeypatch):
    evaluation = tmp_path / "evaluation"
    runs = evaluation / "results"
    native = runs / "new-native"
    data = evaluation / "data.json"
    question = {
        "question": "What is the city?", "task_type": "ER", "entity": ["city"],
        "entity_values": {"city": "Paris"}, "expected_answer": "Paris",
    }
    write(data, [{
        "episode_id": "0",
        "sessions": [{"timestamp": "2024-01-01T00:00:00",
                      "conversation": [{"role": "user", "content": "city is Paris"}]}],
        "evidence_session_indices": [0],
        "tasks": [{"type": "ER", "target_entities": ["city"]}],
        "entities": {"city": {"before": "Paris", "after": "Paris"}},
        "before_questions": {"position_after_session": 0, "questions": [question]},
        "after_questions": {"position_after_session": 0, "questions": [question]},
    }])
    write(native / "rag_index.json", {
        "metadata": {"chunk_size": 256, "chunk_overlap": 32},
        "conversations": {"meme_0": {
            "chunks": [{"chunk_id": "c0", "chunk_index": 0, "content": "city is Paris",
                        "start_turn": 0, "end_turn": 1, "start_token": 0, "end_token": 5}],
            "turn_context": {"turn_timestamps": ["2024-01-01T00:00:00"]},
        }},
    })
    monkeypatch.setattr(complete, "RESULTS_DIR", runs)
    monkeypatch.setattr(uncapped, "EVAL", evaluation)
    monkeypatch.setattr(uncapped, "prompt_template", lambda: "{context}\n{question}")
    complete.build_rag(SimpleNamespace(
        data=data, start_conv=0, end_conv=1, rag_run=native,
        token_budget=5000, task_aware_readout=True, tag="fresh-capped",
    ))
    capped = runs / "meme-rag-fresh-capped"
    rows = uncapped.load(capped / "search_results.json")
    write(capped / "answer_results.json", [{
        "question_id": row["question_id"], "question": row["query"],
        "formatted_context": row["retrieval_metadata"]["formatted_context"],
        "metadata": {},
    } for row in rows])
    write(capped / "eval_results.json", {"metadata": {"model": "judge", "num_runs": 1}})
    manifest = evaluation / "source-map.json"
    write(manifest, {"data": "data.json", "systems": {"rag": {
        "source_run": "results/meme-rag-fresh-capped", "store_run": "results/new-native",
    }}})
    result = uncapped.build("rag", "fresh-uncapped", source_manifest=manifest)
    assert result["questions"] == 2
    assert result["source"] == str(capped)
    assert result["inputs"]["store"] == str(native)
    assert result["changed_questions"] == 0
    audit = uncapped.load(runs / "meme-rag-fresh-uncapped/reconstruction_audit.json")
    assert audit["mismatches"] == []
    assert not (evaluation / "analysis").exists()
    from evaluation.scripts import analyze_meme_complete_store_uncapped as analyzer
    monkeypatch.setattr(analyzer, "EVAL", evaluation)
    monkeypatch.setattr(sys, "argv", [
        "analyze", "--source-manifest", str(manifest), "--tag", "fresh-uncapped",
    ])
    analyzer.main()
    report = uncapped.load(evaluation / "analysis/fresh-uncapped/report.json")
    assert set(report["systems"]) == {"rag"}
    assert report["systems"]["rag"]["status"] == "qa_incomplete"
    assert report["systems"]["rag"]["packing"]["questions"] == 2


def test_uncapped_requires_manifest_instead_of_missing_history():
    with pytest.raises(ValueError, match="source-manifest"):
        uncapped.sources(None)


def test_uncapped_requires_native_store_mapping(tmp_path):
    write(tmp_path / "data.json", [])
    manifest = tmp_path / "sources.json"
    write(manifest, {"data": "data.json", "systems": {"rag": {"source_run": "fresh"}}})
    with pytest.raises(ValueError, match="store_run"):
        uncapped.sources(manifest)
