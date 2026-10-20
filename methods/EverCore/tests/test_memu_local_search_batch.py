import asyncio
from types import SimpleNamespace

from evaluation.src.adapters.memu_local_adapter import (
    MemuLocalAdapter,
    _BoundedItemPool,
    _gpu_cosine_topk,
)


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

    async def embed(self, inputs):
        self.calls.append(list(inputs))
        return [[float(index)] for index, _ in enumerate(inputs)]


class FakeMemUService:
    def __init__(self):
        self.embedding = FakeEmbeddingClient()

    def _get_llm_base_client(self, profile):
        assert profile == "embedding"
        return self.embedding

    async def retrieve(self, *, queries, where):
        query = queries[-1]["content"]
        await self.embedding.embed([query])
        return {
            "items": [{"summary": f"memory for {query}", "score": 1.0}],
            "categories": [],
        }


def make_adapter(service):
    adapter = object.__new__(MemuLocalAdapter)
    adapter.config = {
        "run_name": "batch-test",
        "search": {"top_k": 2, "embedding_batch_size": 128},
        "memu": {"retrieve": {"method": "rag"}},
    }
    adapter.service = service
    adapter._service_lock = asyncio.Lock()
    return adapter


def test_memu_local_search_batch_coalesces_query_embeddings():
    service = FakeMemUService()
    adapter = make_adapter(service)

    results = asyncio.run(
        adapter.search_batch(
            [
                {"query": "first", "conversation_id": "conv-1", "question_id": "q1"},
                {"query": "second", "conversation_id": "conv-2", "question_id": "q2"},
            ]
        )
    )

    assert service.embedding.calls == [["first", "second"]]
    assert [result.question_id for result in results] == ["q1", "q2"]
    assert [result.results[0]["content"] for result in results] == [
        "memory for first",
        "memory for second",
    ]


class FakeVectorRepo:
    def __init__(self, records):
        self.records = records
        self.native_search_calls = []

    def list_items(self, where):
        raise AssertionError("native vector search must not load the full item corpus")

    def vector_search_items(
        self,
        query_vec,
        top_k,
        where=None,
        *,
        ranking="similarity",
        recency_decay_days=30.0,
    ):
        self.native_search_calls.append((query_vec, top_k, where, ranking))
        return [("item-native", 0.99)]


class FakeCategoryEmbeddingClient:
    async def embed(self, inputs):
        assert inputs == ["horizontal", "vertical", "diagonal"]
        return [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_memu_local_keeps_native_item_search_and_uses_cuda_for_auxiliary_similarity():
    import memu.app.retrieve as retrieve_module
    import torch

    assert torch.cuda.is_available()

    items = {
        "item-horizontal": SimpleNamespace(embedding=[1.0, 0.0]),
        "item-vertical": SimpleNamespace(embedding=[0.0, 1.0]),
        "item-diagonal": SimpleNamespace(embedding=[1.0, 1.0]),
    }
    item_repo = FakeVectorRepo(items)
    adapter = object.__new__(MemuLocalAdapter)
    adapter._gpu_similarity_enabled = True
    adapter.service = SimpleNamespace(
        database=SimpleNamespace(memory_item_repo=item_repo)
    )

    original_cosine_topk = retrieve_module.cosine_topk
    try:
        adapter._configure_gpu_similarity()
        assert retrieve_module.cosine_topk is _gpu_cosine_topk

        item_hits = item_repo.vector_search_items(
            [1.0, 0.0],
            2,
            where={"user_id": "smoke-user"},
        )
        assert item_hits == [("item-native", 0.99)]
        assert item_repo.native_search_calls == [
            ([1.0, 0.0], 2, {"user_id": "smoke-user"}, "similarity")
        ]

        retriever = object.__new__(retrieve_module.RetrieveMixin)
        categories = {
            "category-horizontal": SimpleNamespace(summary="horizontal"),
            "category-vertical": SimpleNamespace(summary="vertical"),
            "category-diagonal": SimpleNamespace(summary="diagonal"),
        }
        category_hits, summaries = asyncio.run(
            retriever._rank_categories_by_summary(
                [1.0, 0.0],
                2,
                SimpleNamespace(),
                SimpleNamespace(),
                embed_client=FakeCategoryEmbeddingClient(),
                categories=categories,
            )
        )
        assert [category_id for category_id, _ in category_hits] == [
            "category-horizontal",
            "category-diagonal",
        ]
        assert summaries["category-horizontal"] == "horizontal"

        resources = {
            "resource-horizontal": SimpleNamespace(embedding=[1.0, 0.0]),
            "resource-vertical": SimpleNamespace(embedding=[0.0, 1.0]),
            "resource-diagonal": SimpleNamespace(embedding=[1.0, 1.0]),
        }
        retriever.retrieve_config = SimpleNamespace(
            resource=SimpleNamespace(top_k=2)
        )
        resource_state = {
            "needs_retrieval": True,
            "retrieve_resource": True,
            "proceed_to_resources": True,
            "store": SimpleNamespace(
                resource_repo=SimpleNamespace(
                    list_resources=lambda where: resources
                )
            ),
            "where": {"user_id": "smoke-user"},
            "active_query": "horizontal",
            "query_vector": [1.0, 0.0],
        }
        resource_result = asyncio.run(
            retriever._rag_recall_resources(resource_state, None)
        )
        assert [resource_id for resource_id, _ in resource_result["resource_hits"]] == [
            "resource-horizontal",
            "resource-diagonal",
        ]
    finally:
        retrieve_module.cosine_topk = original_cosine_topk


def test_bounded_item_pool_is_truthy_when_empty():
    pool = _BoundedItemPool()

    assert pool == {}
    assert pool


def test_bounded_item_recall_replaces_registered_workflow_and_fetches_only_hits():
    item = SimpleNamespace(id="item-1")

    class Repo:
        def list_items(self, where):
            raise AssertionError("full item corpus must not be loaded")

        def vector_search_items(self, query, top_k, **kwargs):
            assert query == [1.0, 0.0]
            assert top_k == 2
            return [("item-1", 0.9)]

        def get_item(self, item_id):
            assert item_id == "item-1"
            return item

    class Pipelines:
        def replace_step(self, pipeline, step_id, replacement):
            self.call = (pipeline, step_id, replacement)

    service = SimpleNamespace(
        retrieve_config=SimpleNamespace(
            item=SimpleNamespace(
                top_k=2,
                ranking="similarity",
                recency_decay_days=30.0,
            )
        ),
        _pipelines=Pipelines(),
    )
    service._build_rag_retrieve_workflow = lambda: [
        SimpleNamespace(
            step_id="recall_items",
            handler=service._rag_recall_items,
        )
    ]
    adapter = object.__new__(MemuLocalAdapter)
    adapter.service = service

    adapter._install_bounded_item_recall()

    pipeline, step_id, replacement = service._pipelines.call
    assert (pipeline, step_id) == ("retrieve_rag", "recall_items")
    assert replacement.handler == service._rag_recall_items

    state = {
        "retrieve_item": True,
        "needs_retrieval": True,
        "proceed_to_items": True,
        "store": SimpleNamespace(memory_item_repo=Repo()),
        "where": {"user_id": "user-1"},
        "query_vector": [1.0, 0.0],
    }
    result = asyncio.run(service._rag_recall_items(state, None))

    assert result["item_hits"] == [("item-1", 0.9)]
    assert result["item_pool"] == {"item-1": item}


def test_memu_local_disables_registered_category_summary_step():
    class Pipelines:
        def replace_step(self, pipeline, step_id, replacement):
            self.call = (pipeline, step_id, replacement)

    service = SimpleNamespace(
        memorize_config=SimpleNamespace(enable_item_references=False),
        _pipelines=Pipelines(),
    )
    service._build_memorize_workflow = lambda: [
        SimpleNamespace(
            step_id="persist_index",
            handler=service._memorize_persist_and_index,
        )
    ]
    adapter = object.__new__(MemuLocalAdapter)
    adapter.service = service
    adapter._category_summary_updates_enabled = False

    adapter._configure_category_summary_updates()

    pipeline, step_id, replacement = service._pipelines.call
    assert (pipeline, step_id) == ("memorize", "persist_index")
    assert replacement.handler == service._memorize_persist_and_index
    state = {"category_updates": {"category-1": [("item-1", "summary")]}}
    result = asyncio.run(service._memorize_persist_and_index(state, None))
    assert result["category_updates"] == state["category_updates"]
    assert result["categories"] == []
