import asyncio

from evaluation.src.adapters.rag_memory_ablation_adapter import RAGMemoryAblationAdapter
from evaluation.src.core.data_models import Conversation, SearchResult


def _chunk(index: int) -> dict:
    return {
        "chunk_id": f"conv_chunk_{index}",
        "chunk_index": index,
        "source_chunk_id": f"conv_chunk_{index}",
        "source_chunk_index": index,
        "memory_type": "raw",
        "content": f"chunk {index}",
    }


def test_add_graph_considers_only_nearest_prior_chunks_and_keeps_llm_links():
    adapter = RAGMemoryAblationAdapter.__new__(RAGMemoryAblationAdapter)
    adapter.graph_link_embedding_top_k = 2
    chunks = [_chunk(index) for index in range(4)]
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    judged_candidates = {}

    async def judge_chunk_links(*, conv, new_chunk, candidates):
        new_idx = new_chunk["chunk_index"]
        judged_candidates[new_idx] = [candidate["index"] for candidate in candidates]
        return {
            "related_chunks": [
                {"chunk_index": candidates[0]["index"], "reason": "confirmed"},
                {"chunk_index": new_idx, "reason": "not an existing candidate"},
            ]
        }

    adapter._judge_chunk_links = judge_chunk_links
    edges = asyncio.run(
        adapter._build_graph_expanded_raw_edges(
            Conversation(conversation_id="conv", messages=[]),
            chunks,
            embeddings,
        )
    )

    assert judged_candidates == {1: [0], 2: [1, 0], 3: [2, 1]}
    assert [(edge["source_chunk_index"], edge["target_chunk_index"]) for edge in edges] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert all(edge["reason"] == "confirmed" for edge in edges)


def test_search_expands_embedding_seeds_with_undirected_graph_neighbors():
    adapter = RAGMemoryAblationAdapter.__new__(RAGMemoryAblationAdapter)
    adapter.system_name = "rag_graph_expanded_raw"
    adapter.embedding_top_k = 2
    adapter.graph_link_embedding_top_k = 10
    adapter.graph_max_raw_candidates = 0
    chunks = [_chunk(index) for index in range(4)]
    captured = {}

    async def finish_candidate_search(**kwargs):
        captured.update(kwargs)
        return SearchResult(
            query=kwargs["query"],
            conversation_id=kwargs["conversation_id"],
            results=[],
        )

    adapter._finish_candidate_search = finish_candidate_search
    conv_index = {
        "chunks": chunks,
        "raw_chunks": chunks,
        "embeddings": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        "raw_embeddings": [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        # The edge is stored older -> newer, but expansion must be undirected.
        "graph_edges": [{"source_chunk_index": 3, "target_chunk_index": 0}],
    }

    asyncio.run(
        adapter._search_graph_expanded_raw(
            query="query",
            query_embedding=[1.0, 0.0],
            conversation_id="conv",
            conv_index=conv_index,
            question_id="q1",
        )
    )

    candidates_by_index = {candidate["index"]: candidate for candidate in captured["candidates"]}
    assert set(candidates_by_index) == {0, 1, 3}
    assert candidates_by_index[0]["selection_sources"] == ["embedding_seed"]
    assert candidates_by_index[3]["selection_sources"] == ["graph_neighbor"]
    assert captured["retrieval_metadata_extra"]["embedding_seed_count"] == 2
    assert captured["retrieval_metadata_extra"]["graph_neighbor_count"] == 1
