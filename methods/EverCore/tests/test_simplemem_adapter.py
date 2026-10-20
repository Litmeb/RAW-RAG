"""Exercise native SimpleMem/LanceDB with deterministic network boundaries."""

import asyncio
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from evaluation.src.adapters.registry import create_adapter
from evaluation.src.adapters.simplemem_adapter import APIEmbedding, SimpleMemAdapter
from evaluation.src.core.data_models import Conversation, Message


@pytest.fixture
def setup(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[3] / 'external/SimpleMem'
    if not root.exists():
        pytest.skip('Clone external/SimpleMem to run integration tests')
    pytest.importorskip('lancedb')
    monkeypatch.syspath_prepend(str(root))
    from simplemem.core.utils.llm_client import LLMClient

    calls = []

    def completion(self, messages, **kwargs):
        calls.append(messages)
        return json.dumps([{'lossless_restatement': 'Alice keeps a telescope.',
                            'keywords': ['telescope'], 'persons': ['Alice'],
                            'timestamp': '2023-05-08T13:00:00'}])

    class Embedding:
        dimension = 3

        def __init__(self, *args):
            pass

        def encode_documents(self, texts):
            return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

        def encode_single(self, text, is_query=False):
            return self.encode_documents([text])[0]

    monkeypatch.setattr(LLMClient, 'chat_completion', completion)
    monkeypatch.setattr('evaluation.src.adapters.simplemem_adapter.APIEmbedding', Embedding)
    config = {
        'dataset_name': 'locomo', 'num_workers': 1,
        'simplemem': {'root': str(root), 'window_size': 3, 'overlap_size': 1,
                      'embedding': {'model': 'test', 'api_key': 'secret',
                                    'base_url': 'https://example.invalid/v1'},
                      'retrieval': {'enable_planning': False}},
        'llm': {'model': 'test', 'api_key': 'secret',
                'base_url': 'https://example.invalid/v1',
                'temperature': 0, 'max_tokens': 64},
        'search': {'top_k': 1},
    }
    conv = Conversation('locomo_0', [Message('alice', 'Alice', 'I keep a telescope.',
                                             datetime(2023, 5, 8, 13))])
    return config, conv, calls


def test_native_add_search_and_reopen(setup, tmp_path):
    config, conv, calls = setup
    adapter = create_adapter('simplemem', config, tmp_path)
    index = asyncio.run(adapter.add([conv]))
    assert calls and '2023-05-08T13:00:00' in calls[0][1]['content']
    assert 'Alice: I keep a telescope.' in calls[0][1]['content']
    assert adapter._store(conv.conversation_id).keyword_search(['telescope'])

    reopened = SimpleMemAdapter(config, tmp_path)
    asyncio.run(reopened.add([conv]))
    assert len(calls) == 1  # Completed add is reused, including across processes.
    result = asyncio.run(reopened.search('What does Alice keep?', conv.conversation_id,
                                         index, conversation=conv, question_id='q1'))
    assert result.question_id == 'q1'
    assert result.results[0]['content'] == '[2023-05-08T13:00:00] Alice keeps a telescope.'
    assert result.results[0]['metadata']['persons'] == ['Alice']
    marker = reopened._paths(conv.conversation_id)[1]
    assert 'secret' not in marker.read_text()


def test_isolation_and_missing_add(setup, tmp_path):
    config, conv, _ = setup
    adapter = SimpleMemAdapter(config, tmp_path)
    asyncio.run(adapter.add([conv]))
    other = Conversation('../other', [])
    asyncio.run(adapter.add([other]))
    assert adapter._store(other.conversation_id).backend.count() == 0
    assert adapter._store(conv.conversation_id).backend.count() == 1
    assert adapter._paths(other.conversation_id)[0].parent == adapter.store_dir
    with pytest.raises(FileNotFoundError, match='add first'):
        asyncio.run(adapter.search('query', 'missing', None))


def test_resume_rejects_changed_input_and_embedding(setup, tmp_path):
    config, conv, _ = setup
    adapter = SimpleMemAdapter(config, tmp_path)
    asyncio.run(adapter.add([conv]))
    changed = copy.deepcopy(conv)
    changed.messages[0].content = 'Different input'
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(adapter.add([changed]))
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(adapter.search('query', conv.conversation_id, None, conversation=changed))
    with pytest.raises(ValueError, match='changed'):
        adapter.build_lazy_index([changed], tmp_path)
    config = copy.deepcopy(config)
    config['simplemem']['embedding']['model'] = 'different'
    with pytest.raises(ValueError, match='changed'):
        asyncio.run(SimpleMemAdapter(config, tmp_path).add([conv]))


def test_failed_window_rebuilds_partial_store(setup, tmp_path, monkeypatch):
    config, conv, _ = setup
    from simplemem.core.memory_builder import MemoryBuilder

    original = MemoryBuilder._generate_memory_entries
    count = 0

    def fail_second(builder, dialogues):
        nonlocal count
        count += 1
        return [] if count == 2 else original(builder, dialogues)

    monkeypatch.setattr(MemoryBuilder, '_generate_memory_entries', fail_second)
    conv.messages *= 4
    adapter = SimpleMemAdapter(config, tmp_path)
    with pytest.raises(RuntimeError, match='no memory'):
        asyncio.run(adapter.add([conv]))
    assert not adapter._paths(conv.conversation_id)[1].exists()
    assert adapter._store(conv.conversation_id).backend.count() == 1
    monkeypatch.setattr(MemoryBuilder, '_generate_memory_entries', original)
    asyncio.run(adapter.add([conv]))
    assert adapter._store(conv.conversation_id).backend.count() == 2


def test_non_locomo_rejected():
    with pytest.raises(ValueError, match='only'):
        SimpleMemAdapter({'dataset_name': 'longmemeval'})


def test_lexical_index_includes_last_window(setup, tmp_path, monkeypatch):
    config, conv, _ = setup
    from simplemem.core.memory_builder import MemoryBuilder
    from simplemem.core.models.memory_entry import MemoryEntry

    memories = iter(['Alice likes astronomy.', 'Alice bought a zeppelin.'])
    monkeypatch.setattr(MemoryBuilder, '_generate_memory_entries',
                        lambda builder, dialogues: [MemoryEntry(lossless_restatement=next(memories))])
    conv.messages *= 4
    adapter = SimpleMemAdapter(config, tmp_path)
    asyncio.run(adapter.add([conv]))
    found = adapter._store(conv.conversation_id).keyword_search(['zeppelin'])
    assert [entry.lossless_restatement for entry in found] == ['Alice bought a zeppelin.']


def test_embedding_orders_normalizes_and_checks_batch():
    from types import SimpleNamespace

    embedding = object.__new__(APIEmbedding)
    embedding.model = 'test'
    rows = [SimpleNamespace(index=1, embedding=[0, 2]),
            SimpleNamespace(index=0, embedding=[3, 0])]
    embedding.client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(data=rows)))
    np.testing.assert_allclose(embedding.encode_documents(['a', 'b']), [[1, 0], [0, 1]])
    with pytest.raises(RuntimeError, match='incomplete'):
        embedding.encode_documents(['one'])
