"""SimpleMem's text memory pipeline, evaluated with the shared LoCoMo reader."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from threading import Lock

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import SearchResult


class APIEmbedding:
    """Implement SimpleMem's embedding interface with the baseline's API model."""

    def __init__(self, config, timeout):
        from openai import OpenAI

        self.model = config['model']
        self.client = OpenAI(api_key=config['api_key'], base_url=config['base_url'],
                             timeout=timeout, max_retries=2)
        self.dimension = len(self.encode_single('embedding dimension probe'))

    def encode_documents(self, texts):
        import numpy as np

        vectors = []
        for offset in range(0, len(texts), 64):
            batch = texts[offset:offset + 64]
            response = self.client.embeddings.create(model=self.model, input=batch)
            rows = sorted(response.data, key=lambda item: item.index)
            if len(rows) != len(batch):
                raise RuntimeError('Embedding API returned an incomplete batch')
            vectors.extend(row.embedding for row in rows)
        values = np.asarray(vectors, dtype=np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError('Invalid embedding vectors')
        if hasattr(self, 'dimension') and values.shape[1] != self.dimension:
            raise ValueError('Embedding dimension changed')
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError('Embedding API returned a zero vector')
        return values / norms

    def encode_single(self, text, is_query=False):
        return self.encode_documents([text])[0]


@register_adapter('simplemem')
class SimpleMemAdapter(BaseAdapter):
    def __init__(self, config, output_dir=None):
        super().__init__(config)
        if config.get('dataset_name', 'locomo') != 'locomo':
            raise ValueError('SimpleMem currently supports only --dataset locomo')
        self.output_dir = Path(output_dir or '.')
        self.store_dir = self.output_dir / 'simplemem_store'
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.options = config.get('simplemem', {})
        repo = Path(__file__).resolve().parents[5]
        self.root = (repo / self.options.get('root', 'external/SimpleMem')).resolve()
        if not (self.root / 'simplemem/core/memory_builder.py').is_file():
            raise FileNotFoundError(f'Clone aiming-lab/SimpleMem into {self.root}')
        self.revision = subprocess.check_output(
            ['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True).strip()
        expected = self.options.get('revision')
        if expected and expected != self.revision:
            raise ValueError(f'SimpleMem revision {self.revision} differs from {expected}')
        self.window_size = int(self.options.get('window_size', 40))
        self.overlap_size = int(self.options.get('overlap_size', 2))
        if not 0 <= self.overlap_size < self.window_size:
            raise ValueError('Require 0 <= overlap_size < window_size')
        self._components = None
        self._load_lock = Lock()
        self._stores = {}
        self._store_lock = Lock()
        self._answer_client = None

    def _load(self):
        with self._load_lock:
            if self._components is not None:
                return self._components
            sys.path.insert(0, str(self.root))
            from simplemem.core.database.vector_store import VectorStore
            from simplemem.core.hybrid_retriever import HybridRetriever
            from simplemem.core.memory_builder import MemoryBuilder
            from simplemem.core.models.memory_entry import Dialogue
            from simplemem.core.utils.llm_client import LLMClient

            cfg = self.config['llm']
            timeout = float(self.options.get('timeout_seconds', 120))
            llm = LLMClient(api_key=cfg['api_key'], model=cfg['model'],
                            base_url=cfg['base_url'], use_streaming=False,
                            enable_thinking=False)
            llm.client = llm.client.with_options(timeout=timeout, max_retries=0)
            embedding = APIEmbedding(self.options['embedding'], timeout)
            self._components = VectorStore, HybridRetriever, MemoryBuilder, Dialogue, llm, embedding
            return self._components

    def _paths(self, conversation_id):
        key = hashlib.sha256(conversation_id.encode()).hexdigest()[:24]
        directory = self.store_dir / key
        return directory, directory / 'complete.json'

    def _store(self, conversation_id):
        VectorStore, _, _, _, _, embedding = self._load()
        with self._store_lock:
            if conversation_id not in self._stores:
                directory, _ = self._paths(conversation_id)
                self._stores[conversation_id] = VectorStore(
                    db_path=str(directory / 'lance'), table_name='memories',
                    embedding_model=embedding)
            return self._stores[conversation_id]

    def _fingerprint(self, conversation):
        # Credentials do not affect the index and must never be persisted.
        payload = {
            'revision': self.revision,
            'window': self.window_size, 'overlap': self.overlap_size,
            'llm': {k: self.config['llm'].get(k) for k in ('model', 'base_url')},
            'embedding': {k: self.options.get('embedding', {}).get(k)
                          for k in ('model', 'base_url')},
            'messages': [(m.sender_name, m.content,
                          m.timestamp.isoformat() if m.timestamp else None)
                         for m in conversation.messages],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _add_conversation(self, conversation):
        directory, marker = self._paths(conversation.conversation_id)
        fingerprint = self._fingerprint(conversation)
        if marker.exists():
            saved = json.loads(marker.read_text(encoding='utf-8'))
            if saved['fingerprint'] != fingerprint:
                raise ValueError('SimpleMem input/config changed; use a new --run-name')
            if not (directory / 'lance/memories.lance').is_dir():
                raise FileNotFoundError(f'SimpleMem store missing: {directory}')
            return
        _, _, MemoryBuilder, Dialogue, llm, _ = self._load()
        store = self._store(conversation.conversation_id)
        # A failed conversation has no completion marker. Rebuild its partial
        # table so retry cannot duplicate already-written windows.
        store.clear()
        # Build Tantivy once after the final window. Besides indexing all rows,
        # this avoids replacing freshly opened index files on Windows.
        store.backend._fts_initialized = True
        builder = MemoryBuilder(llm, store, window_size=self.window_size,
                                enable_parallel_processing=False)
        builder.overlap_size = self.overlap_size
        builder.step_size = self.window_size - self.overlap_size
        generate = builder._generate_memory_entries

        def checked_generate(dialogues):
            entries = generate(dialogues)
            # Upstream returns [] after exhausted API/parse failures. Do not
            # checkpoint such a window as successfully ingested.
            if not entries:
                raise RuntimeError('SimpleMem produced no memory for a window; '
                                   'inspect extraction logs and rerun')
            return entries

        builder._generate_memory_entries = checked_generate
        dialogues = [Dialogue(dialogue_id=i + 1, speaker=m.sender_name,
                              content=m.content,
                              timestamp=m.timestamp.isoformat() if m.timestamp else None)
                     for i, m in enumerate(conversation.messages)]
        builder.add_dialogues(dialogues)
        builder.process_remaining()
        # Refresh lexical index after all writes, before concurrent retrieval.
        store.backend._fts_initialized = False
        store.backend._init_fts_index()
        if not store.backend._fts_initialized:
            raise RuntimeError('SimpleMem lexical index creation failed')
        store.backend.optimize()
        state = {'conversation_id': conversation.conversation_id,
                 'fingerprint': fingerprint, 'revision': self.revision,
                 'memory_count': store.backend.count()}
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(marker)

    async def add(self, conversations, **kwargs):
        semaphore = asyncio.Semaphore(max(1, int(self.config.get('num_workers', 2))))
        failures = []

        async def ingest(conversation):
            async with semaphore:
                if failures:
                    return
                try:
                    await asyncio.to_thread(self._add_conversation, conversation)
                except Exception as error:
                    failures.append(error)

        await asyncio.gather(*(ingest(conv) for conv in conversations))
        if failures:
            raise failures[0]
        return self.build_lazy_index(conversations, self.output_dir)

    def build_lazy_index(self, conversations, output_dir):
        for conversation in conversations:
            self._validate_completed(conversation.conversation_id, conversation)
        return {'conversation_ids': [c.conversation_id for c in conversations]}

    def _validate_completed(self, conversation_id, conversation=None):
        directory, marker = self._paths(conversation_id)
        if not marker.is_file() or not (directory / 'lance/memories.lance').is_dir():
            raise FileNotFoundError(f'Run SimpleMem add first for {conversation_id}')
        if conversation is not None:
            saved = json.loads(marker.read_text(encoding='utf-8'))
            if saved['fingerprint'] != self._fingerprint(conversation):
                raise ValueError('SimpleMem input/config changed; use a new --run-name')

    def _search(self, query, conversation_id, top_k, conversation):
        self._validate_completed(conversation_id, conversation)
        _, HybridRetriever, _, _, llm, _ = self._load()
        retriever = HybridRetriever(llm, self._store(conversation_id),
                                    **self.options.get('retrieval', {}))
        entries = retriever.retrieve(query)
        results = []
        for entry in entries[:top_k]:
            content = entry.lossless_restatement
            if entry.timestamp:
                content = f'[{entry.timestamp}] {content}'
            results.append({'content': content, 'score': None,
                            'metadata': entry.model_dump()})
        return SearchResult(query, conversation_id, results,
                            retrieval_metadata={'system': 'simplemem',
                                                'native_result_count': len(entries),
                                                'top_k': top_k})

    async def search(self, query, conversation_id, index, **kwargs):
        top_k = int(kwargs.get('top_k', self.config.get('search', {}).get('top_k', 20)))
        if top_k < 1:
            raise ValueError('SimpleMem search.top_k must be positive')
        result = await asyncio.to_thread(self._search, query, conversation_id,
                                        top_k, kwargs.get('conversation'))
        result.question_id = kwargs.get('question_id', '')
        return result

    async def answer(self, query, context, **kwargs):
        # Reuse the framework's prompt selection, retries and output parsing.
        from evaluation.src.adapters.online_base import OnlineAPIAdapter

        if self._answer_client is None:
            from types import SimpleNamespace
            from memory_layer.llm.llm_provider import LLMProvider
            from evaluation.src.utils.config import load_yaml

            prompts = load_yaml(str(Path(__file__).resolve().parents[2] / 'config/prompts.yaml'))
            cfg = self.config['llm']
            provider = LLMProvider(provider_type=cfg.get('provider', 'openai'),
                                   **{k: cfg[k] for k in ('model', 'api_key', 'base_url',
                                                         'temperature', 'max_tokens')})
            self._answer_client = SimpleNamespace(config=self.config, llm_provider=provider,
                _resolve_answer_prompt=lambda key: prompts['online_api']['default'][
                    key or self.config.get('answer', {}).get('prompt_key', 'answer_prompt_cot')])
        return await OnlineAPIAdapter.answer(self._answer_client, query, context, **kwargs)
