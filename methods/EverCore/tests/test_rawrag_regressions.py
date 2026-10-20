"""Offline regression coverage for the nine supplementary-package audit findings."""

import asyncio
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rich.console import Console

from evaluation import cli
from evaluation.src.adapters.rag_adapter import RAGAdapter
from evaluation.src.core.data_models import AnswerResult, Conversation, Dataset, Message, QAPair
from evaluation.src.core.pipeline import Pipeline
from evaluation.src.core.stages.search_stage import run_search_stage
from evaluation.src.datasets.memoryagentbench import resolve_chunking_options
from evaluation.src.evaluators.llm_judge import LLMJudge
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.fingerprint import fingerprint, public_config
from evaluation.src.utils.run_identity import guard_artifacts, RUN_ARTIFACTS


class StopBeforeDataAccess(Exception):
    pass


def response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def adapter(tmp_path):
    config = {
        "adapter": "rag",
        "chunking": {"strategy": "turn", "chunk_size": 1, "chunk_overlap": 0},
        "embedding": {"model": "offline-embedding", "api_key": "offline",
                      "base_url": "http://127.0.0.1:1/v1"},
        "llm": {"model": "offline-reader", "api_key": "offline",
                "base_url": "http://127.0.0.1:1/v1"},
        "rerank": {"enabled": True, "max_retries": 1},
        "search": {"embedding_top_k": 128, "rerank_top_k": 15, "max_retries": 1},
    }
    instance = RAGAdapter(config, tmp_path)
    instance._embed_texts = AsyncMock(side_effect=lambda texts, **_: [[1.0, 0.0] for _ in texts])
    instance._post_json = lambda **kwargs: {
        "results": [{"index": i, "relevance_score": 1.0 / (i + 1)}
                    for i in range(len(kwargs["payload"]["documents"]))]
    }
    instance.answer = AsyncMock(return_value="answer")
    yield instance
    asyncio.run(instance.close())


def conversation(content="old", timestamp=None):
    return Conversation("meme_0", [
        Message("user", "User", content, timestamp or datetime(2024, 1, 1)),
        Message("user", "User", "second message", datetime(2024, 2, 1)),
    ])


def judge(num_runs=1):
    instance = LLMJudge({
        "llm": {"api_key": "offline", "base_url": "http://127.0.0.1:1/v1"},
        "num_runs": num_runs, "max_attempts": 1, "retry_delay_seconds": 0,
    })
    instance.client.chat.completions.create = AsyncMock(return_value=response('{"label":"CORRECT"}'))
    return instance


@pytest.mark.parametrize("dataset", ["locomo", "meme", "personamem_v2", "longmemeval"])
def test_real_cli_defaults_reach_config_loading(dataset, monkeypatch):
    monkeypatch.setattr(cli, "setup_environment", lambda **_: True)
    paths = []

    def stop(path):
        paths.append(path)
        raise StopBeforeDataAccess

    monkeypatch.setattr(cli, "load_yaml", stop)
    monkeypatch.setattr(sys, "argv", ["cli", "--dataset", dataset, "--system", "rag"])
    with pytest.raises(StopBeforeDataAccess):
        asyncio.run(cli.main())
    assert Path(paths[0]).name == f"{dataset}.yaml"


def test_cli_explicit_lme_judge_rejects_other_dataset(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "cli", "--dataset", "locomo", "--system", "rag", "--lme-judge", "official",
    ])
    with pytest.raises(SystemExit) as error:
        asyncio.run(cli.main())
    assert error.value.code == 2


def test_lme_alias_is_normalized_before_judge_validation(monkeypatch):
    monkeypatch.setattr(cli, "setup_environment", lambda **_: True)
    monkeypatch.setattr(cli, "load_yaml", lambda _: (_ for _ in ()).throw(StopBeforeDataAccess()))
    monkeypatch.setattr(sys, "argv", [
        "cli", "--dataset", "lme", "--system", "rag", "--lme-judge", "official",
    ])
    with pytest.raises(StopBeforeDataAccess):
        asyncio.run(cli.main())


@pytest.mark.parametrize("extra,expected", [
    ([], "fact"),
    (["--mab-disable-official-chunking"], "none"),
    (["--mab-chunk-strategy", "token", "--mab-chunk-size", "128"], "token"),
])
def test_mab_batch_resolves_each_source_independently(monkeypatch, extra, expected):
    children = []
    captured = []
    monkeypatch.setattr(cli, "setup_environment", lambda **_: True)

    def batch(args, plan, console):
        children.extend(cli.build_memoryagentbench_child_command(args, split, source, None)
                        for split, source in plan)
        return 0

    monkeypatch.setattr(cli, "run_memoryagentbench_batch", batch)
    monkeypatch.setattr(sys, "argv", ["cli", "--dataset", "memoryagentbench", "--system", "rag", *extra])
    asyncio.run(cli.main())
    assert len(children) == sum(len(sources) for sources in cli.MAB_DEFAULT_BENCHES.values())

    def prepare(**kwargs):
        captured.append(kwargs)
        raise StopBeforeDataAccess

    monkeypatch.setattr(cli, "prepare_memoryagentbench_dataset", prepare)
    for command in children:
        monkeypatch.setattr(sys, "argv", ["cli", *command[3:]])
        with pytest.raises(StopBeforeDataAccess):
            asyncio.run(cli.main())
    for entry in captured:
        resolved = resolve_chunking_options(
            entry["source_name"], official_chunking=entry["official_chunking"],
            chunk_strategy=entry["chunk_strategy"], chunk_size=entry["chunk_size"],
            chunk_overlap=entry["chunk_overlap"],
        )[0]
        if "factconsolidation" in entry["source_name"]:
            assert resolved == expected
        elif not extra:
            assert resolved == "none"


def test_environment_accepts_exported_credentials_without_dotenv(tmp_path, monkeypatch):
    from common_utils import load_env
    monkeypatch.setattr(load_env, "PROJECT_DIR", tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "offline")
    assert load_env.load_env_file(check_env_var="LLM_API_KEY")


def test_loader_reconverts_when_original_input_changes(tmp_path, monkeypatch):
    from evaluation.src.core import loaders
    from evaluation.src.converters.base import BaseConverter

    class Converter(BaseConverter):
        calls = 0

        def get_input_files(self):
            return {"raw": "raw.json"}

        def convert(self, input_paths, output_path):
            self.calls += 1
            Path(output_path).write_text(Path(input_paths["raw"]).read_text(encoding="utf-8"), encoding="utf-8")

    converter = Converter()
    monkeypatch.setattr(loaders, "get_converter", lambda _: converter)
    raw = [{"conversation": {
        "speaker_a": "User", "speaker_b": "Assistant",
        "session_1_date_time": "2024-01-01T00:00:00",
        "session_1": [{"speaker": "User", "text": "old", "dia_id": "D1:1"}],
    }, "qa": []}]
    source = tmp_path / "raw.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    assert loaders.load_dataset("fixture", str(tmp_path)).conversations[0].messages[0].content == "old"
    loaders.load_dataset("fixture", str(tmp_path))
    assert converter.calls == 1
    raw[0]["conversation"]["session_1"][0]["text"] = "new"
    source.write_text(json.dumps(raw), encoding="utf-8")
    assert loaders.load_dataset("fixture", str(tmp_path)).conversations[0].messages[0].content == "new"
    assert converter.calls == 2


def test_longmemeval_default_preserves_long_message_tail(tmp_path):
    from evaluation.src.core.loaders import load_locomo_dataset
    path = Path(cli.__file__).parent / "config/datasets/longmemeval.yaml"
    config = load_yaml(str(path))
    assert config["data"]["max_content_length"] is None
    content = "a" * 9000 + "ESSENTIAL_TAIL"
    source = tmp_path / "data.json"
    source.write_text(json.dumps([{
        "conversation": {"speaker_a": "User", "speaker_b": "Assistant",
                         "session_1_date_time": "2024-01-01T00:00:00",
                         "session_1": [{"speaker": "User", "text": content, "dia_id": "D1:1"}]},
        "qa": [],
    }]), encoding="utf-8")
    dataset = load_locomo_dataset(str(source), "longmemeval",
                                  max_content_length=config["data"]["max_content_length"])
    assert dataset.conversations[0].messages[0].content == content


@pytest.mark.parametrize("payload", [
    {}, {"results": []}, [], {"results": "invalid"},
    {"results": [{"index": 0, "score": 1}]},
    {"results": [{"index": 0, "score": 1}, {"index": 0, "score": 0}]},
    {"results": [{"index": -1, "score": 1}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0, "score": 1}, {"index": 2, "score": 0}]},
    {"results": [{"index": True, "score": 1}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0.1, "score": 1}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0, "score": float("nan")}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0, "score": float("inf")}, {"index": 1, "score": 0}]},
    {"results": [{"index": 0, "score": "0.5"}, {"index": 1, "score": 0}]},
])
def test_reranker_rejects_invalid_response(adapter, payload):
    with pytest.raises(ValueError):
        adapter._parse_rerank_results(payload, [{"id": 0}, {"id": 1}], 2)


def test_reranker_orders_valid_numeric_scores(adapter):
    results = adapter._parse_rerank_results({"data": {"results": [
        {"index": 0, "score": -1}, {"index": 1, "relevanceScore": 0.75},
    ]}}, [{"id": 0}, {"id": 1}], 2)
    assert [row["id"] for row in results] == [1, 0]


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_reranker_failure_propagates_through_search_stage(adapter, failure):
    conv = conversation()
    index = asyncio.run(adapter.add([conv]))
    if failure == "empty":
        adapter._post_json = lambda **_: {}
    else:
        adapter._rerank_candidates = AsyncMock(side_effect=RuntimeError("reranker unavailable"))
    qa = QAPair("q", "query", "gold", metadata={"conversation_id": conv.conversation_id})
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(run_search_stage(adapter, [qa], index, [conv], None, logging.getLogger("test")))
    adapter.answer.assert_not_called()


@pytest.mark.parametrize("change", ["text", "timestamp", "metadata", "length"])
def test_add_rebuilds_changed_sources_and_lazy_load_rejects_them(adapter, change):
    conv = conversation()
    original = asyncio.run(adapter.add([conv]))
    old_fp = original["conversations"][conv.conversation_id]["source_fingerprint"]
    adapter._embed_texts.reset_mock()
    asyncio.run(adapter.add([conv]))
    adapter._embed_texts.assert_not_called()
    changed = copy.deepcopy(conv)
    if change == "text":
        changed.messages[0].content = "NEW_CONTENT"
    elif change == "timestamp":
        changed.messages[0].timestamp = datetime(2024, 3, 1)
    elif change == "metadata":
        changed.messages[0].metadata["session_date"] = "2024-04-01"
    else:
        changed.messages.append(Message("u", "User", "extra message"))
    with pytest.raises(ValueError, match="source changed"):
        adapter.build_lazy_index([changed], adapter.output_dir)
    rebuilt = asyncio.run(adapter.add([changed]))
    assert rebuilt["conversations"][conv.conversation_id]["source_fingerprint"] != old_fp
    assert adapter._embed_texts.call_count > 0
    assert adapter.build_lazy_index([changed], adapter.output_dir)


def test_index_config_change_rebuilds_embeddings(adapter):
    conv = conversation()
    asyncio.run(adapter.add([conv]))
    adapter._embed_texts.reset_mock()
    adapter.embedding_model = "new-embedding-model"
    asyncio.run(adapter.add([conv]))
    assert adapter._embed_texts.call_count > 0


def test_pgvector_filters_before_topk_and_matches_local(adapter):
    chunks = [{"chunk_id": str(i), "chunk_index": i, "content": str(i),
               "start_turn": i, "end_turn": i + 1} for i in range(4)]
    conv_index = {"chunks": chunks, "embeddings": [[0.8, 0.2], [0.7, 0.3], [1, 0], [1, 0]],
                  "turn_context": {"turn_session_dates": [
                      "2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02",
                  ]}}
    index = {"conversations": {"meme_0": conv_index}}
    adapter.embedding_top_k = 2
    adapter.rerank_enabled = False
    local = asyncio.run(adapter.search("q", "meme_0", index, question_date="2024-01-15"))

    async def search(**kwargs):
        assert kwargs["allowed_chunk_ids"] == ["0", "1"]
        return [{"chunk_id": cid, "chunk_index": int(cid), "similarity_score": 0.9}
                for cid in kwargs["allowed_chunk_ids"][:kwargs["top_k"]]]

    store = SimpleNamespace(collection_name="test", table_name="vectors",
                            search=AsyncMock(side_effect=search), close=AsyncMock())
    adapter.pgvector_store = store
    adapter._ensure_pgvector_initialized = AsyncMock()
    pg = asyncio.run(adapter.search("q", "meme_0", index, question_date="2024-01-15"))
    assert [row["content"] for row in local.results] == [row["content"] for row in pg.results] == ["0", "1"]
    assert pg.retrieval_metadata["future_filtered_chunk_count"] == 2
    store.search.reset_mock()
    empty = asyncio.run(adapter.search("q", "meme_0", index, question_date="2023-01-01"))
    assert empty.results == []
    store.search.assert_not_called()


@pytest.mark.parametrize("bad", [None, "", "not json", "{}", '{"label":"UNKNOWN"}',
                                 '{"label":true}', RuntimeError("quota exceeded")])
def test_judge_errors_are_not_false(bad):
    evaluator = judge()
    create = evaluator.client.chat.completions.create
    if isinstance(bad, Exception):
        create.side_effect = bad
    else:
        create.return_value = response(bad)
    with pytest.raises(RuntimeError):
        asyncio.run(evaluator._judge_answer("Q", "A", "A"))


def test_judge_persists_partial_runs_and_retries_only_missing(tmp_path):
    evaluator = judge(num_runs=2)
    evaluator.set_checkpoint_path(tmp_path / "judge_progress.json")
    create = evaluator.client.chat.completions.create
    create.side_effect = [response('{"label":"CORRECT"}'), RuntimeError("quota exceeded")]
    answers = [AnswerResult("q", "Q", "A", "A", category="1")]
    with pytest.raises(RuntimeError, match="No accuracy"):
        asyncio.run(evaluator.evaluate(answers))
    saved = json.loads(evaluator.checkpoint_path.read_text(encoding="utf-8"))
    assert saved["answers"]["q"]["llm_judgments"] == {"judgment_1": True}
    assert saved["answers"]["q"]["status"] == "judge_error"
    assert "judgment_2" in saved["answers"]["q"]["judge_errors"]
    create.reset_mock()
    create.side_effect = [response('{"label":"WRONG"}')]
    result = asyncio.run(evaluator.evaluate(answers))
    assert create.call_count == 1
    assert result.accuracy == 0.5
    create.reset_mock()
    asyncio.run(evaluator.evaluate(answers))
    create.assert_not_called()
    answers[0].answer = "new answer"
    create.side_effect = [response('{"label":"CORRECT"}')] * 2
    asyncio.run(evaluator.evaluate(answers))
    assert create.call_count == 2


def test_run_identity_archives_stale_outputs_and_preserves_index(tmp_path):
    assert guard_artifacts(tmp_path, "run", {"source": "old"}, RUN_ARTIFACTS)
    (tmp_path / "answer_results.json").write_text("old answers", encoding="utf-8")
    (tmp_path / "rag_index.json").write_text("index", encoding="utf-8")
    assert not guard_artifacts(tmp_path, "run", {"source": "old"}, RUN_ARTIFACTS)
    with pytest.raises(ValueError, match="Run inputs"):
        guard_artifacts(tmp_path, "run", {"source": "new"}, RUN_ARTIFACTS, allow_reset=False)
    assert (tmp_path / "answer_results.json").exists()
    guard_artifacts(tmp_path, "run", {"source": "new"}, RUN_ARTIFACTS)
    assert not (tmp_path / "answer_results.json").exists()
    assert (tmp_path / "rag_index.json").read_text() == "index"
    archived = list((tmp_path / "stale_artifacts").glob("*/answer_results.json"))
    assert archived[0].read_text() == "old answers"


@pytest.mark.parametrize("change", ["source", "reader", "judge", "prompt", "question"])
def test_pipeline_invalidates_completed_stages_when_inputs_change(adapter, monkeypatch, change):
    evaluator = judge()
    dataset = Dataset("meme", [conversation()], [
        QAPair("q", "query", "answer", metadata={"conversation_id": "meme_0"}),
    ])
    pipeline = Pipeline(adapter, evaluator, None, adapter.output_dir)
    asyncio.run(pipeline.run(dataset))
    adapter.answer.reset_mock()
    evaluator.client.chat.completions.create.reset_mock()
    asyncio.run(pipeline.run(dataset))
    adapter.answer.assert_not_called()
    evaluator.client.chat.completions.create.assert_not_called()
    assert "Accuracy: 100.00%" in (adapter.output_dir / "report.txt").read_text(encoding="utf-8")
    if change == "source":
        dataset.conversations[0].messages[0].content = "NEW_CONTENT"
    elif change == "reader":
        adapter.config["llm"]["model"] = "new-reader"
        adapter.answer_model = "new-reader"
    elif change == "judge":
        evaluator.config["llm"]["model"] = "new-judge"
    elif change == "prompt":
        monkeypatch.setattr("evaluation.src.core.pipeline.prompt_fingerprint", lambda: "new-prompt")
    else:
        dataset.qa_pairs[0].question = "new question with same ID"
    asyncio.run(pipeline.run(dataset))
    assert adapter.answer.call_count == 1
    assert evaluator.client.chat.completions.create.call_count == 1


def test_config_fingerprints_do_not_depend_on_api_key_rotation():
    one = {"llm": {"model": "model", "api_key": "secret-one"}}
    two = {"llm": {"model": "model", "api_key": "secret-two"}}
    assert fingerprint(public_config(one)) == fingerprint(public_config(two))


def test_pipeline_judge_failure_leaves_evaluation_incomplete(adapter):
    evaluator = judge()
    evaluator.client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    dataset = Dataset("meme", [conversation()], [
        QAPair("q", "query", "answer", metadata={"conversation_id": "meme_0"}),
    ])
    pipeline = Pipeline(adapter, evaluator, None, adapter.output_dir)
    with pytest.raises(RuntimeError, match="No accuracy"):
        asyncio.run(pipeline.run(dataset))
    assert not (adapter.output_dir / "eval_results.json").exists()
    assert not (adapter.output_dir / "report.txt").exists()
    checkpoint = json.loads((adapter.output_dir / "checkpoint_default.json").read_text(encoding="utf-8"))
    assert "evaluate" not in checkpoint["completed_stages"]
    assert (adapter.output_dir / "judge_progress.json").exists()


def test_judge_retries_a_transient_failure():
    evaluator = judge()
    evaluator.max_attempts = 2
    evaluator.client.chat.completions.create.side_effect = [
        RuntimeError("temporary"), response('{"label":"CORRECT"}'),
    ]
    assert asyncio.run(evaluator._judge_answer("Q", "A", "A")) is True
    assert evaluator.client.chat.completions.create.call_count == 2


def test_pgvector_sql_applies_visibility_before_ranking(monkeypatch):
    from evaluation.src.adapters.rag_pgvector_store import RAGPgVectorStore
    store = RAGPgVectorStore("postgresql://offline", "vectors", "collection", "embed")
    calls = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, query, params):
            calls.append((query.as_string(), params))

        def fetchall(self):
            return [("past", 0, 0.9)]

    monkeypatch.setattr(store, "_connect", lambda: SimpleNamespace(cursor=Cursor))
    monkeypatch.setattr(store, "_ensure_index_sync", lambda _: None)
    result = asyncio.run(store.search("conv", [1.0, 0.0], 2, ["past"]))
    sql, params = calls[0]
    assert sql.index("chunk_id = ANY") < sql.index("ORDER BY") < sql.index("LIMIT")
    assert "MATERIALIZED" in sql
    assert params[3] == params[4] == ["past"]
    assert result[0]["chunk_id"] == "past"


def test_answer_fast_path_invalidates_reader_checkpoints(tmp_path, monkeypatch):
    from evaluation.src.core.data_models import SearchResult
    search = SearchResult("query", "meme_0", [], {"formatted_context": "evidence"}, "q")
    dataset = Dataset("meme", [conversation()], [
        QAPair("q", "query", "answer", metadata={"conversation_id": "meme_0"}),
    ])
    (tmp_path / "search_results.json").write_text(json.dumps([{
        "question_id": "q", "query": search.query, "conversation_id": search.conversation_id,
        "results": search.results, "retrieval_metadata": search.retrieval_metadata,
    }]), encoding="utf-8")
    args = SimpleNamespace(stages=["answer"], smoke=False, from_conv=0, to_conv=None,
                           locomo_sample=None, only_category=None)
    answer = AsyncMock(return_value="answer")
    monkeypatch.setattr(cli, "AnswerOnlyAdapter", lambda config: SimpleNamespace(config=config, answer=answer))
    config = {"llm": {"model": "reader-one"}}
    asyncio.run(cli.run_answer_fast_path(args, dataset, {"evaluation": {}}, config, tmp_path, Console()))
    (tmp_path / "responses_checkpoint_1.json").write_text(json.dumps({
        "q": {"question_id": "q", "question": "query", "answer": "STALE",
              "golden_answer": "answer", "formatted_context": "evidence"},
    }), encoding="utf-8")
    config["llm"]["model"] = "reader-two"
    answer.reset_mock()
    asyncio.run(cli.run_answer_fast_path(args, dataset, {"evaluation": {}}, config, tmp_path, Console()))
    assert answer.call_count == 1
    rows = json.loads((tmp_path / "answer_results.json").read_text(encoding="utf-8"))
    assert rows[0]["answer"] == "answer"
    assert list((tmp_path / "stale_artifacts").glob("*/responses_checkpoint_1.json"))
