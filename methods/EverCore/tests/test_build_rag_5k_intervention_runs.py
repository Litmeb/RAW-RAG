from copy import deepcopy

import pytest
import tiktoken

from evaluation.scripts.build_rag_5k_intervention_runs import (
    build_run,
    load_json,
    pack_prefix,
    raw_entry,
    rewrite_record,
    save_json,
    token_count,
)


@pytest.fixture
def encoding():
    return tiktoken.get_encoding("o200k_base")


def record(contents, context=" expanded" * 1000):
    return {
        "question_id": "q1",
        "query": "question",
        "conversation_id": "conv",
        "results": [
            {
                "content": text,
                "score": 0.5,
                "metadata": {
                    "chunk_index": 10 - i,
                    "similarity_score": 0.12345678,
                    "rerank_score": 0.5,
                    "rerank_rank": 3 - i,
                },
            }
            for i, text in enumerate(contents)
        ],
        "retrieval_metadata": {
            "formatted_context": context,
            "question_date": "2026-01-01",
            "all_rerank_results": [{"chunk_index": 3}],
        },
    }


def test_in_budget_context_stays_verbatim_including_boundary(encoding):
    source = record(["raw"], "expanded evidence\n\n")
    updated, stats = rewrite_record(source, encoding, token_count("expanded evidence\n\n", encoding))
    assert updated["retrieval_metadata"]["formatted_context"] == "expanded evidence\n\n"
    assert not stats["turn_expansion_disabled"]
    assert not stats["context_changed"]


def test_unexpand_all_before_clipping_preserves_selection_and_order(encoding):
    source = record(["first raw", "second raw"])
    before = deepcopy(source)
    updated, stats = rewrite_record(source, encoding, 100)
    expected = "\n\n".join(raw_entry(i, r) for i, r in enumerate(source["results"], 1))
    assert updated["retrieval_metadata"]["formatted_context"] == expected
    assert stats["turn_expansion_disabled"]
    assert not stats["prefix_clipped"]
    assert source == before
    assert updated["results"] == source["results"]
    assert updated["retrieval_metadata"]["question_date"] == "2026-01-01"
    assert updated["retrieval_metadata"]["all_rerank_results"] == [{"chunk_index": 3}]


def test_prefix_stops_at_first_nonfitting_entry_without_partial_or_skip(encoding):
    source = record(["first", " long" * 200, "last"])
    first = raw_entry(1, source["results"][0])
    updated, stats = rewrite_record(source, encoding, token_count(first, encoding) + 40)
    assert updated["retrieval_metadata"]["formatted_context"] == first
    assert stats["entries_after"] == 1
    assert stats["prefix_clipped"]
    assert not stats["first_entry_clipped"]
    assert updated["results"] == source["results"]


def test_oversized_first_entry_is_token_prefix(encoding):
    source = record([" raw" * 100])
    updated, stats = rewrite_record(source, encoding, 40)
    context = updated["retrieval_metadata"]["formatted_context"]
    assert raw_entry(1, source["results"][0]).startswith(context)
    assert token_count(context, encoding) <= 40
    assert stats["first_entry_clipped"]
    assert stats["entries_after"] == 1


def test_unicode_and_special_literals_remain_valid(encoding):
    text = "\U0001f642\u6c49\u5b57<|endoftext|>" * 30
    for budget in range(1, 30):
        context, count, clipped = pack_prefix([text], encoding, budget)
        assert text.startswith(context)
        assert "\ufffd" not in context
        assert token_count(context, encoding) <= budget
        assert count == 1 and clipped


def test_bad_inputs_fail_instead_of_silently_regenerating(encoding):
    with pytest.raises(ValueError, match="positive"):
        rewrite_record(record(["raw"]), encoding, 0)
    with pytest.raises(ValueError, match="recover raw"):
        rewrite_record(record([]), encoding, 20)
    with pytest.raises(ValueError, match="nonempty"):
        rewrite_record(record(["raw"], ""), encoding)


def test_build_run_writes_search_only_and_refuses_overwrite(tmp_path, monkeypatch, encoding):
    from evaluation.scripts import build_rag_5k_intervention_runs as module

    monkeypatch.setitem(module.RUNS, "locomo", {
        "source": "source", "questions": 1, "answer_model": "gpt-4o-mini",
    })
    source = tmp_path / "source"
    source.mkdir()
    row = record(["raw"])
    save_json(source / "search_results.json", [row])
    save_json(source / "answer_results.json", [{
        "question_id": "q1",
        "formatted_context": row["retrieval_metadata"]["formatted_context"],
    }])
    save_json(source / "eval_results.json", {"metadata": {"model": "judge", "num_runs": 3}})
    original = (source / "search_results.json").read_bytes()
    build_run(tmp_path, "locomo", "test", 100, encoding, dry_run=True)
    destination = tmp_path / "locomo-rag-test"
    assert not destination.exists()
    manifest = build_run(tmp_path, "locomo", "test", 100, encoding)
    assert {p.name for p in destination.iterdir()} == {
        "search_results.json", "context_cap_summary.json",
    }
    assert manifest["judge_num_runs"] == 3
    assert manifest["max_tokens_after"] <= 100
    assert load_json(destination / "search_results.json")[0]["results"] == row["results"]
    assert (source / "search_results.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        build_run(tmp_path, "locomo", "test", 100, encoding)
