import asyncio
import csv
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from evaluation.speed_sequence_cli import (
    DurableMeasurementWriter,
    MemMachineAsyncWaitConfig,
    memmachine_episode_ids,
    iter_beam_items,
    numeric_json_files,
    parse_conv_range,
    build_parser,
    run,
    wait_for_memmachine_async_ingestion,
)
from evaluation.src.adapters.rag_adapter import RAGAdapter
from evaluation.src.adapters.evermemos_adapter import EverCoreAdapter
from evaluation.src.adapters.evermemos import stage1_memcells_extraction
from evaluation.src.adapters.evermemos import stage2_index_building
from evaluation.src.core.data_models import Conversation, Message
from evaluation.src.runners import emem_runner
from evaluation.src.runners.emem_runner import EMemIncrementalIngestor


def test_iter_beam_items_preserves_plan_batch_turn_message_order():
    document = {
        "chat": [
            {
                "plan-1": [
                    {
                        "batch_number": 7,
                        "time_anchor": "July-01-2024",
                        "turns": [
                            [
                                {"id": 10, "role": "user", "content": "first"},
                                {
                                    "id": 11,
                                    "role": "assistant",
                                    "content": "second",
                                },
                            ],
                            [{"id": 12, "role": "user", "content": "third"}],
                        ],
                    }
                ],
                "plan-2": None,
            }
        ]
    }

    items = list(iter_beam_items(document))

    assert [item.message.content for item in items] == ["first", "second", "third"]
    assert [item.source_message_id for item in items] == [10, 11, 12]
    assert [(item.turn_index, item.message_index) for item in items] == [
        (1, 1),
        (1, 2),
        (2, 1),
    ]
    assert items[0].message.timestamp.isoformat() == "2024-07-01T00:00:00"


def test_numeric_json_files_accepts_any_contiguous_numbered_set(tmp_path):
    for number in range(1, 36):
        (tmp_path / f"{number}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in numeric_json_files(tmp_path)] == [
        f"{number}.json" for number in range(1, 36)
    ]

    (tmp_path / "10.json").unlink()
    with pytest.raises(ValueError, match="must be contiguous"):
        numeric_json_files(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0:10", (0, 10)), (":1", (0, 1)), ("9:", (9, 10)), (":", (0, 10))],
)
def test_parse_conv_range(raw, expected):
    assert parse_conv_range(raw, 10) == expected


def test_durable_writer_keeps_jsonl_and_csv_in_sync(tmp_path):
    record = {
        "conversation_file": "1.json",
        "conversation_id": "1",
        "sequence_index": 1,
        "global_sequence_index": 1,
        "content_chars": 5,
        "add_seconds": 0.25,
        "status": "ok",
    }
    with DurableMeasurementWriter(tmp_path) as writer:
        writer.write(record)

    json_row = json.loads(
        (tmp_path / "speed_sequence.jsonl").read_text(encoding="utf-8")
    )
    with (tmp_path / "speed_sequence.csv").open(encoding="utf-8", newline="") as f:
        csv_row = next(csv.DictReader(f))

    assert json_row["add_seconds"] == 0.25
    assert csv_row["add_seconds"] == "0.25"
    assert csv_row["conversation_file"] == "1.json"


def test_memmachine_waits_until_current_add_has_no_pending_rows(monkeypatch):
    observed_ids = []
    pending_counts = iter([2, 0])

    async def fake_pending_count(episode_ids, _config):
        observed_ids.append(episode_ids)
        return next(pending_counts)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.memmachine_pending_ingestion_count",
        fake_pending_count,
    )
    monkeypatch.setattr("evaluation.speed_sequence_cli.asyncio.sleep", no_sleep)
    add_result = {
        "add_results": [
            {
                "project_id": "project",
                "episode_ids": ["episode-1", "episode-2", "episode-1"],
            }
        ]
    }

    result = asyncio.run(
        wait_for_memmachine_async_ingestion(
            add_result,
            MemMachineAsyncWaitConfig(poll_interval_seconds=0.01),
        )
    )

    assert memmachine_episode_ids(add_result) == ["episode-1", "episode-2"]
    assert observed_ids == [
        ["episode-1", "episode-2"],
        ["episode-1", "episode-2"],
    ]
    assert result["pending"] == 0
    assert result["polls"] == 2


def test_run_waits_after_each_memmachine_add_before_submitting_next(tmp_path, monkeypatch):
    data_dir = tmp_path / "beam"
    data_dir.mkdir()
    document = {
        "conversation_id": "1",
        "chat": [
            {
                "plan-1": [
                    {
                        "batch_number": 1,
                        "turns": [
                            [
                                {"id": 1, "role": "user", "content": "one"},
                                {"id": 2, "role": "assistant", "content": "two"},
                            ]
                        ],
                    }
                ]
            }
        ],
    }
    (data_dir / "1.json").write_text(json.dumps(document), encoding="utf-8")
    events = []

    class FakeAdapter:
        async def add(self, conversations, **_kwargs):
            content = conversations[0].messages[0].content
            events.append(("add", content))
            return {"add_results": [{"episode_ids": [f"episode-{content}"]}]}

        async def close(self):
            return None

    async def fake_wait(add_result, _config):
        episode_id = add_result["add_results"][0]["episode_ids"][0]
        events.append(("wait", episode_id))
        return {"pending": 0}

    monkeypatch.setattr("common_utils.load_env.setup_environment", lambda **_kwargs: None)
    monkeypatch.setattr(
        "evaluation.src.adapters.registry.create_adapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )
    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.wait_for_memmachine_async_ingestion",
        fake_wait,
    )
    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.create_token_counter", lambda _model: len
    )
    args = build_parser().parse_args(
        [
            "--system",
            "memmachine",
            "--run-name",
            "unit-test-memmachine-wait",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "results"),
            "--no-plot",
        ]
    )

    asyncio.run(run(args))

    assert events == [
        ("add", "one"),
        ("wait", "episode-one"),
        ("add", "two"),
        ("wait", "episode-two"),
    ]


def test_run_stops_memmachine_sequence_when_completion_cannot_be_verified(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "beam"
    data_dir.mkdir()
    (data_dir / "1.json").write_text(
        json.dumps(
            {
                "conversation_id": "1",
                "chat": [
                    {
                        "plan": [
                            {
                                "batch_number": 1,
                                "turns": [
                                    [
                                        {"role": "user", "content": "one"},
                                        {"role": "user", "content": "two"},
                                    ]
                                ],
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    add_calls = []

    class FakeAdapter:
        async def add(self, conversations, **_kwargs):
            add_calls.append(conversations[0].messages[0].content)
            return {"add_results": [{"episode_ids": ["episode"]}]}

        async def close(self):
            return None

    async def fail_wait(_add_result, _config):
        raise TimeoutError("still pending")

    monkeypatch.setattr("common_utils.load_env.setup_environment", lambda **_kwargs: None)
    monkeypatch.setattr(
        "evaluation.src.adapters.registry.create_adapter",
        lambda *_args, **_kwargs: FakeAdapter(),
    )
    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.wait_for_memmachine_async_ingestion", fail_wait
    )
    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.create_token_counter", lambda _model: len
    )
    args = build_parser().parse_args(
        [
            "--system",
            "memmachine",
            "--run-name",
            "unit-test-memmachine-stop",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(tmp_path / "results"),
            "--no-plot",
        ]
    )

    with pytest.raises(RuntimeError, match="no later Add was submitted"):
        asyncio.run(run(args))

    assert add_calls == ["one"]


def test_run_skips_failed_add_and_keeps_storage_context_stable(
    tmp_path, monkeypatch
):
    data_dir = tmp_path / "beam"
    data_dir.mkdir()
    for number in range(1, 11):
        messages = (
            [
                {"id": 1, "role": "user", "content": "one"},
                {"id": 2, "role": "assistant", "content": "two"},
            ]
            if number == 1
            else []
        )
        document = {
            "conversation_id": str(number),
            "chat": [
                {
                    "plan-1": [
                        {"batch_number": 1, "turns": [messages] if messages else []}
                    ]
                }
            ],
        }
        (data_dir / f"{number}.json").write_text(
            json.dumps(document), encoding="utf-8"
        )

    class FakeAdapter:
        def __init__(self):
            self.calls = []
            self.closed = False

        async def add(self, conversations, **kwargs):
            self.calls.append((conversations, kwargs))
            if conversations[0].messages[0].content == "one":
                raise ValueError("deliberate add failure")

        async def close(self):
            self.closed = True

    adapter = FakeAdapter()
    monkeypatch.setattr(
        "common_utils.load_env.setup_environment", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        "evaluation.src.adapters.registry.create_adapter",
        lambda *_args, **_kwargs: adapter,
    )
    monkeypatch.setattr(
        "evaluation.speed_sequence_cli.create_token_counter", lambda _model: len
    )
    output_dir = tmp_path / "results"
    args = build_parser().parse_args(
        [
            "--system",
            "memos_local",
            "--run-name",
            "unit-test",
            "--data-dir",
            str(data_dir),
            "--conv-range",
            "0:1",
            "--output-dir",
            str(output_dir),
            "--no-plot",
        ]
    )

    asyncio.run(run(args))

    assert len(adapter.calls) == 2
    contexts = [call[0][0].conversation_id for call in adapter.calls]
    assert contexts[0] == contexts[1]
    assert contexts[0].startswith("beam10m_")
    assert contexts[0].endswith("_1")
    assert [call[0][0].messages[0].content for call in adapter.calls] == ["one", "two"]
    assert adapter.closed is True
    rows = [
        json.loads(line)
        for line in (output_dir / "speed_sequence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["cumulative_messages"] for row in rows] == [1, 2]
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["error"] == "ValueError: deliberate add failure"
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["successful_items"] == 1
    assert summary["failed_items"] == 1


def test_rag_incremental_add_reembeds_only_mutable_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(RAGAdapter, "_load_tokenizer", lambda _self: None)
    adapter = RAGAdapter(
        {
            "chunking": {
                "strategy": "token",
                "chunk_size": 8,
                "chunk_overlap": 2,
                "expand_turns_on_display": True,
            },
            "embedding": {"api_key": "test", "base_url": "http://localhost"},
            "llm": {"api_key": "test", "base_url": "http://localhost"},
            "rerank": {"enabled": False},
            "storage": {"backend": "local"},
        },
        output_dir=tmp_path,
    )
    embedded_batches = []

    async def fake_embed(texts, **_kwargs):
        embedded_batches.append(list(texts))
        return [[float(len(text))] for text in texts]

    adapter._embed_texts = fake_embed

    def one_message(content):
        return Conversation(
            conversation_id="beam10m_test_1",
            messages=[Message("user", "user", content)],
            metadata={"speaker_a": "user", "speaker_b": "assistant"},
        )

    first = asyncio.run(
        adapter.add_incremental_message(
            one_message("one two three four five six seven eight"), sequence_index=1
        )
    )
    second = asyncio.run(
        adapter.add_incremental_message(
            one_message("nine ten eleven twelve"), sequence_index=2
        )
    )
    asyncio.run(adapter.finalize_incremental_conversation("beam10m_test_1"))

    assert first["total_chunks"] >= 2
    assert second["changed_chunks"] < second["total_chunks"]
    assert len(embedded_batches[-1]) == second["changed_chunks"]
    saved = json.loads((tmp_path / "rag_index.json").read_text(encoding="utf-8"))
    conv_index = saved["conversations"]["beam10m_test_1"]
    assert len(conv_index["chunks"]) == len(conv_index["embeddings"])
    assert len(conv_index["turn_context"]["turn_lines"]) == 2
    assert len(
        (tmp_path / "rag_incremental_updates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2


def test_evermemos_incremental_add_appends_memcells_and_embeddings(
    tmp_path, monkeypatch
):
    adapter = object.__new__(EverCoreAdapter)
    adapter.output_dir = tmp_path
    adapter.llm_provider = object()
    adapter.atomic_fact_extractor = object()
    adapter._temporal_search_cache = {}
    adapter._convert_config_to_experiment_config = lambda: SimpleNamespace()
    seen_conv_ids = []

    class FakeMemCell:
        atomic_fact = None

        def __init__(self, content):
            self.content = content

        def to_dict(self):
            return {
                "subject": self.content,
                "summary": self.content,
                "episode": self.content,
            }

    async def fake_process(conv_id, conversation, **_kwargs):
        seen_conv_ids.append(conv_id)
        return conv_id, [FakeMemCell(conversation[0]["content"])]

    class FakeVectorizer:
        async def get_embeddings(self, texts):
            return [[float(index)] for index, _text in enumerate(texts)]

    monkeypatch.setattr(
        stage1_memcells_extraction, "process_single_conversation", fake_process
    )
    monkeypatch.setattr(
        stage2_index_building,
        "tokenize",
        lambda text, _stemmer, _stop_words: text.lower().split(),
    )
    monkeypatch.setattr(
        stage2_index_building.stopwords, "words", lambda _language: []
    )
    monkeypatch.setattr(
        stage2_index_building,
        "get_vectorize_service",
        lambda: FakeVectorizer(),
    )

    def one_message(content):
        return Conversation(
            conversation_id="beam10m_hash_1",
            messages=[
                Message(
                    "user",
                    "user",
                    content,
                    timestamp=datetime(2024, 1, 1),
                )
            ],
            metadata={"speaker_a": "user", "speaker_b": "assistant"},
        )

    asyncio.run(
        adapter.add_incremental_message(one_message("first"), sequence_index=1)
    )
    result = asyncio.run(
        adapter.add_incremental_message(one_message("second"), sequence_index=2)
    )
    asyncio.run(adapter.finalize_incremental_conversation("beam10m_hash_1"))

    assert seen_conv_ids == ["1", "1"]
    assert result["total_memcells"] == 2
    docs = json.loads(
        (tmp_path / "memcells" / "memcell_list_conv_1.json").read_text(
            encoding="utf-8"
        )
    )
    assert [doc["episode"] for doc in docs] == ["first", "second"]
    assert (tmp_path / "bm25_index" / "bm25_index_conv_1.pkl").exists()
    assert (tmp_path / "vectors" / "embedding_index_conv_1.pkl").exists()
    assert len(
        (tmp_path / "evermemos_incremental_updates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 2


def test_emem_incremental_add_maps_each_message_to_unique_session(monkeypatch):
    class FakeGraph:
        def vcount(self):
            return 3

        def ecount(self):
            return 2

    class FakeModel:
        def __init__(self):
            self.graph = FakeGraph()
            self.samples = []

        def index_conversation(self, sample):
            self.samples.append(sample)

    ingestor = object.__new__(EMemIncrementalIngestor)
    ingestor.model = FakeModel()
    ingestor.conversation_id = "beam10m_hash_1"
    ingestor.symbols = {}
    ingestor.date_format_type = "locomo"
    monkeypatch.setattr(
        emem_runner,
        "_build_emem_sample",
        lambda conversation, *_args, **_kwargs: conversation,
    )
    conversation = Conversation(
        conversation_id="beam10m_hash_1",
        messages=[Message("user", "user", "hello", metadata={})],
        metadata={"speaker_a": "user", "speaker_b": "assistant"},
    )

    result = asyncio.run(
        ingestor.add_incremental_message(conversation, sequence_index=17)
    )

    assert conversation.messages[0].metadata["session"] == "session_17"
    assert conversation.messages[0].metadata["dia_id"] == "D17:0"
    assert ingestor.model.samples == [conversation]
    assert result == {"new_sessions": 1, "graph_nodes": 3, "graph_edges": 2}
