from __future__ import annotations

from evaluation.scripts.build_mab_embedding_only_runs import rebuild_record_embedding_only


def test_rebuild_record_restores_embedding_order_and_removes_rerank() -> None:
    record = {
        "conversation_id": "conv",
        "query": "query",
        "results": [],
        "retrieval_metadata": {
            "rerank_top_k": 2,
            "all_rerank_results": [
                {"chunk_id": "c2", "chunk_index": 2, "embedding_rank": 2, "rerank_rank": 1, "similarity_score": 0.8, "rerank_score": 0.99},
                {"chunk_id": "c1", "chunk_index": 1, "embedding_rank": 1, "rerank_rank": 2, "similarity_score": 0.9, "rerank_score": 0.5},
                {"chunk_id": "c3", "chunk_index": 3, "embedding_rank": 3, "rerank_rank": 3, "similarity_score": 0.7, "rerank_score": 0.1},
            ],
        },
    }
    chunks = {
        "c1": {"chunk_id": "c1", "chunk_index": 1, "content": "first"},
        "c2": {"chunk_id": "c2", "chunk_index": 2, "content": "second"},
        "c3": {"chunk_id": "c3", "chunk_index": 3, "content": "third"},
    }

    rebuilt = rebuild_record_embedding_only(record, {"turn_context": None}, chunks)

    assert [result["content"] for result in rebuilt["results"]] == ["first", "second"]
    assert [result["score"] for result in rebuilt["results"]] == [0.9, 0.8]
    assert all(result["metadata"]["rerank_score"] is None for result in rebuilt["results"])
    metadata = rebuilt["retrieval_metadata"]
    assert metadata["retrieval_mode"] == "embedding_only"
    assert metadata["reranker_enabled"] is False
    assert "all_rerank_results" not in metadata
    assert [item["chunk_id"] for item in metadata["all_embedding_results"]] == ["c1", "c2", "c3"]
