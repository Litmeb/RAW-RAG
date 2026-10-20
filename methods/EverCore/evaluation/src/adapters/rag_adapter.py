"""
Local RAG adapter for the evaluation framework.

Builds a per-conversation chunk index, retrieves with embedding similarity,
reranks candidates with an HTTP rerank API, and answers with the configured LLM.
"""

import asyncio
import gc
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from bisect import bisect_left, bisect_right
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.rag_pgvector_store import RAGPgVectorStore
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, Message, SearchResult
from evaluation.src.utils.config import load_yaml
from evaluation.src.utils.fingerprint import fingerprint


@register_adapter("rag")
class RAGAdapter(BaseAdapter):
    """OpenAI-compatible embedding + HTTP rerank local RAG adapter."""

    INDEX_FILENAME = "rag_index.json"
    fail_on_search_error = True

    @staticmethod
    def _resolve_config_or_env(
        configured_value: Optional[str],
        *env_names: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve a config value, treating empty/unexpanded placeholders as missing."""
        value = str(configured_value).strip() if configured_value is not None else ""
        if value and not (value.startswith("${") and value.endswith("}")):
            return value

        for env_name in env_names:
            env_value = os.getenv(env_name)
            if env_value and env_value.strip():
                return env_value.strip()
        return default

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()

        chunk_cfg = config.get("chunking", {})
        self.chunk_size = int(chunk_cfg.get("chunk_size", 500))
        self.chunk_overlap = int(chunk_cfg.get("chunk_overlap", 0))
        self.chunk_strategy = chunk_cfg.get("strategy", "token")
        self.expand_turns_on_display = bool(chunk_cfg.get("expand_turns_on_display", False))
        self.dedup_turns_on_display = bool(chunk_cfg.get("dedup_turns_on_display", False))
        self.tokenizer_model = chunk_cfg.get("tokenizer_model", "gpt-4")
        self._validate_chunking()
        self._encoding = self._load_tokenizer()

        self.sample_workers = int(config.get("sample_workers", config.get("num_workers", 4)))

        embedding_cfg = config.get("embedding", {})
        self.embedding_model = embedding_cfg.get("model", "text-embedding-3-small")
        self.embedding_batch_size = int(embedding_cfg.get("batch_size", 64))
        self.embedding_max_retries = int(embedding_cfg.get("max_retries", 3))
        embedding_api_key = (
            embedding_cfg.get("api_key")
            or os.getenv("VECTORIZE_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        embedding_base_url = (
            embedding_cfg.get("base_url")
            or os.getenv("VECTORIZE_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.embedding_client = AsyncOpenAI(
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        )

        storage_cfg = config.get("storage", {})
        pgvector_cfg = storage_cfg.get("pgvector", {})
        storage_backend = storage_cfg.get("backend", "local")
        if bool(pgvector_cfg.get("enabled", False)):
            storage_backend = "pgvector"
        if storage_backend not in {"local", "pgvector"}:
            raise ValueError("storage.backend must be 'local' or 'pgvector'.")
        self.storage_backend = storage_backend
        self.index_save_every_conversations = max(
            1, int(storage_cfg.get("save_every_conversations", 1))
        )
        self.pgvector_store: Optional[RAGPgVectorStore] = None
        self._pgvector_initialized = False
        if self.storage_backend == "pgvector":
            collection_name = pgvector_cfg.get("collection_name") or self._default_collection_name()
            table_name = pgvector_cfg.get("table_name", "rag_vectors")
            database_url = pgvector_cfg.get("database_url") or os.getenv(
                "RAG_PGVECTOR_DATABASE_URL", ""
            )
            self.pgvector_store = RAGPgVectorStore(
                database_url=database_url,
                table_name=table_name,
                collection_name=collection_name,
                embedding_model=self.embedding_model,
            )

        rerank_cfg = config.get("rerank", {})
        self.rerank_enabled = bool(rerank_cfg.get("enabled", True))
        self.rerank_model = rerank_cfg.get("model", "jina-reranker-v2-base-multilingual")
        self.rerank_base_url = rerank_cfg.get("base_url", "https://api.jina.ai").rstrip("/")
        self.rerank_endpoint = rerank_cfg.get("endpoint", "/v1/rerank")
        self.rerank_api_key = rerank_cfg.get("api_key") or os.getenv("RERANK_API_KEY")
        self.rerank_timeout_seconds = float(rerank_cfg.get("timeout_seconds", 120))
        self.rerank_max_retries = max(1, int(rerank_cfg.get("max_retries", 3)))

        search_cfg = config.get("search", {})
        self.embedding_top_k = int(search_cfg.get("embedding_top_k", search_cfg.get("top_k", 30)))
        self.rerank_top_k = int(search_cfg.get("rerank_top_k", 10))
        self.sort_selected_chunks_by_time = bool(
            search_cfg.get("sort_selected_chunks_by_time", False)
        )
        dynamic_top_k_cfg = search_cfg.get("dynamic_top_k", {})
        self.dynamic_top_k_enabled = bool(dynamic_top_k_cfg.get("enabled", False))
        self.dynamic_top_k_score_threshold = float(
            dynamic_top_k_cfg.get("score_threshold", 0.0)
        )
        self.dynamic_top_k_fallback_top_k = int(
            dynamic_top_k_cfg.get("fallback_top_k", self.rerank_top_k)
        )
        if self.dynamic_top_k_fallback_top_k <= 0:
            raise ValueError("search.dynamic_top_k.fallback_top_k must be positive.")

        llm_cfg = config.get("llm", {})
        self.answer_model = llm_cfg.get("model", "openai/gpt-4.1-mini")
        self.answer_temperature = float(llm_cfg.get("temperature", 0))
        self.answer_max_tokens = int(llm_cfg.get("max_tokens", 16384))
        answer_api_key = self._resolve_config_or_env(
            llm_cfg.get("api_key"),
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        )
        answer_base_url = self._resolve_config_or_env(
            llm_cfg.get("base_url"),
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default="https://openrouter.ai/api/v1",
        )
        if not answer_api_key:
            raise ValueError(
                "RAG answer LLM API key is missing. Set llm.api_key in "
                "evaluation/config/systems/rag.yaml or define LLM_API_KEY in "
                "methods/EverCore/.env."
            )
        self.answer_client = AsyncOpenAI(api_key=answer_api_key, base_url=answer_base_url)

        evaluation_root = Path(__file__).parent.parent.parent
        prompts_path = evaluation_root / "config" / "prompts.yaml"
        self._prompts = load_yaml(str(prompts_path))

        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.request_log_path = self.logs_dir / "rag_requests.jsonl"
        self._log_lock = asyncio.Lock()
        self._embedding_hydration_locks: Dict[str, asyncio.Lock] = {}

        print("RAGAdapter initialized")
        print(f"   Output Dir: {self.output_dir}")
        expand_label = "+turn-expand" if self.expand_turns_on_display else ""
        dedup_label = "+turn-dedup" if self.dedup_turns_on_display else ""
        print(
            f"   Chunk strategy/size/overlap: {self.chunk_strategy}{expand_label}{dedup_label}/"
            f"{self.chunk_size}/{self.chunk_overlap}"
        )
        print(f"   Embedding model: {self.embedding_model}")
        print(f"   Storage backend: {self.storage_backend}")
        print(
            f"   Index save interval: every {self.index_save_every_conversations} conversation(s)"
        )
        if self.pgvector_store:
            print(
                f"   pgvector: table={self.pgvector_store.table_name}, "
                f"collection={self.pgvector_store.collection_name}"
            )
        print(f"   Rerank model: {self.rerank_model}")
        print(f"   Answer model: {self.answer_model}")

    def _default_collection_name(self) -> str:
        return self.output_dir.name or "rag_default"

    async def _ensure_pgvector_initialized(self) -> None:
        if not self.pgvector_store or self._pgvector_initialized:
            return
        try:
            await self.pgvector_store.initialize()
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize RAG pgvector storage. Check "
                "RAG_PGVECTOR_DATABASE_URL, PostgreSQL reachability, and pgvector "
                f"availability. Original error: {exc}"
            ) from exc
        self._pgvector_initialized = True

    def _validate_chunking(self) -> None:
        if self.chunk_strategy not in {"turn", "token"}:
            raise ValueError("chunking.strategy must be 'turn' or 'token'.")
        if self.expand_turns_on_display and self.chunk_strategy != "token":
            raise ValueError("chunking.expand_turns_on_display requires chunking.strategy='token'.")
        if self.dedup_turns_on_display and not self.expand_turns_on_display:
            raise ValueError(
                "chunking.dedup_turns_on_display requires chunking.expand_turns_on_display=true."
            )
        if self.chunk_size == -1:
            return
        if self.chunk_size <= 0:
            raise ValueError("chunking.chunk_size must be positive or -1 for full context.")
        if self.chunk_overlap < 0:
            raise ValueError("chunking.chunk_overlap must be >= 0.")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunking.chunk_overlap must be smaller than chunking.chunk_size.")

    def _load_tokenizer(self):
        try:
            import tiktoken

            return tiktoken.encoding_for_model(self.tokenizer_model)
        except Exception as exc:
            print(f"Warning: tiktoken unavailable ({exc}); falling back to word chunks.")
            return None

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """Build and persist a local chunk + embedding index."""
        self.console.print("\n" + "=" * 60, style="bold cyan")
        self.console.print("Stage 1: Add (RAG local index)", style="bold cyan")
        self.console.print("=" * 60, style="bold cyan")

        longmemeval_only = self._all_longmemeval_conversations(conversations)

        if self.pgvector_store and not longmemeval_only:
            await self._ensure_pgvector_initialized()

        semaphore = asyncio.Semaphore(max(1, self.sample_workers))
        persist_embeddings = (
            False if self.pgvector_store else self._should_persist_embeddings(conversations)
        )
        index = self._load_resumeable_index()
        indexed_conversations: Dict[str, Any] = dict(index.get("conversations", {}))

        pending_conversations = []
        hydration_queue = []
        reused_count = 0
        for conv in conversations:
            cached = indexed_conversations.get(conv.conversation_id)
            if cached and cached.get("source_fingerprint") == fingerprint(conv) and self._is_conversation_index_reusable(
                cached,
                allow_missing_embeddings=not persist_embeddings,
            ):
                reused_count += 1
                if (
                    not self.pgvector_store
                    and not self._is_longmemeval_conversation_id(conv.conversation_id)
                    and not self._has_complete_embeddings(cached)
                ):
                    hydration_queue.append(conv.conversation_id)
                continue
            pending_conversations.append(conv)

        if reused_count:
            self.console.print(
                f"Resuming Add stage from cached rag_index: reused {reused_count}/"
                f"{len(conversations)} conversations",
                style="yellow",
            )
        if not pending_conversations:
            if hydration_queue and not self.pgvector_store:
                self.console.print(
                    f"Hydrating {len(hydration_queue)} cached conversations with in-memory embeddings",
                    style="yellow",
                )
                for conv_id in tqdm(hydration_queue, desc="RAG hydrate", unit="conv"):
                    conv_index = indexed_conversations[conv_id]
                    await self._hydrate_conversation_embeddings(
                        conv_index,
                        conv_id,
                        stage="add",
                    )
            self.console.print(
                "All conversations already have reusable cached RAG chunks/index data",
                style="green",
            )
            return index

        async def build_single(conv: Conversation) -> tuple[str, Dict[str, Any]]:
            async with semaphore:
                built = await self._build_conversation_index(conv)
                built["source_fingerprint"] = fingerprint(conv)
                return conv.conversation_id, built

        tasks = [build_single(conv) for conv in pending_conversations]
        processed_since_save = 0
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="RAG index",
            unit="conv",
        ):
            conv_id, conv_index = await coro
            indexed_conversations[conv_id] = conv_index
            index["conversations"] = indexed_conversations
            processed_since_save += 1
            if processed_since_save >= self.index_save_every_conversations:
                self._save_index(index, persist_embeddings=persist_embeddings)
                processed_since_save = 0

        if processed_since_save:
            self._save_index(index, persist_embeddings=persist_embeddings)

        return index

    async def add_incremental_message(
        self,
        conversation: Conversation,
        *,
        sequence_index: int,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Append one message while embedding only the mutable tail chunks.

        The regular ``add`` method owns whole-conversation checkpoint semantics.
        Sequential latency experiments instead keep a token stream in memory:
        chunks that already reached ``chunk_size`` remain immutable, while the
        last partial chunk and any newly opened chunks are regenerated and
        embedded.  This makes work at step N depend on the new content and the
        bounded overlap tail, not on re-embedding the entire prefix.
        """
        if self.pgvector_store:
            raise ValueError(
                "RAG sequential ingestion currently requires storage.backend=local"
            )
        if len(conversation.messages) != 1:
            raise ValueError("RAG incremental ingestion requires exactly one message")
        if self.chunk_strategy != "token":
            raise ValueError(
                "RAG sequential ingestion currently requires chunking.strategy=token"
            )

        if not hasattr(self, "_incremental_add_states"):
            self._incremental_add_states: Dict[str, Dict[str, Any]] = {}
            self._incremental_add_index = self._load_resumeable_index()

        conversation_id = conversation.conversation_id
        state = self._incremental_add_states.get(conversation_id)
        if state is None:
            header = self._format_conversation_header(conversation)
            if self._encoding is not None:
                units: List[Any] = self._encode_text_units(header + "\n\n")
            else:
                units = (header + "\n\n").split()
            state = {
                "units": units,
                "chunks": [],
                "embeddings": [],
                "header": header,
                "turn_lines": [],
                "turn_spans": [],
                "turn_timestamps": [],
                "turn_session_dates": [],
                "speaker_a": conversation.metadata.get("speaker_a", ""),
                "speaker_b": conversation.metadata.get("speaker_b", ""),
            }
            self._incremental_add_states[conversation_id] = state

        message = conversation.messages[0]
        line = self._format_message_line(message)
        separator = "" if not state["turn_lines"] else "\n"
        start_unit = len(state["units"])
        if self._encoding is not None:
            state["units"].extend(self._encode_text_units(separator + line))
        else:
            state["units"].extend((separator + line).split())
        end_unit = len(state["units"])
        state["turn_lines"].append(line)
        state["turn_spans"].append([start_unit, end_unit])
        state["turn_timestamps"].append(
            message.timestamp.isoformat() if message.timestamp else None
        )
        state["turn_session_dates"].append(
            message.metadata.get("session_date")
            if message.metadata
            else None
        )

        units = state["units"]
        old_chunks = state["chunks"]
        if self.chunk_size == -1:
            first_changed = 0
        elif not old_chunks:
            first_changed = 0
        else:
            last = old_chunks[-1]
            last_size = int(last["end_token"]) - int(last["start_token"])
            first_changed = len(old_chunks) if last_size >= self.chunk_size else len(old_chunks) - 1

        stable_chunks = old_chunks[:first_changed]
        stable_embeddings = state["embeddings"][:first_changed]
        if self.chunk_size == -1:
            starts = [0]
        else:
            step = self.chunk_size - self.chunk_overlap
            starts = []
            start = first_changed * step
            while start < len(units):
                starts.append(start)
                if start + self.chunk_size >= len(units):
                    break
                start += step

        tail_chunks = []
        for offset, start in enumerate(starts):
            end = len(units) if self.chunk_size == -1 else min(start + self.chunk_size, len(units))
            chunk_units = units[start:end]
            content = (
                self._encoding.decode(chunk_units)
                if self._encoding is not None
                else " ".join(chunk_units)
            )
            tail_chunks.append(
                {
                    "chunk_id": f"{conversation_id}_chunk_{first_changed + offset}",
                    "chunk_index": first_changed + offset,
                    "content": content,
                    "start_token": start,
                    "end_token": end,
                }
            )
        self._annotate_token_chunks_with_turns(tail_chunks, state["turn_spans"])
        tail_embeddings = await self._embed_texts(
            [chunk["content"] for chunk in tail_chunks],
            stage="add_incremental",
            conversation_id=conversation_id,
            metadata={
                "sequence_index": sequence_index,
                "first_changed_chunk": first_changed,
                "changed_chunk_count": len(tail_chunks),
            },
        )
        state["chunks"] = stable_chunks + tail_chunks
        state["embeddings"] = stable_embeddings + tail_embeddings

        turn_context = {
            "header": state["header"],
            "turn_lines": state["turn_lines"],
            "turn_spans": state["turn_spans"],
            "turn_timestamps": state["turn_timestamps"],
            "turn_session_dates": state["turn_session_dates"],
        }
        self._incremental_add_index["conversations"][conversation_id] = {
            "conversation_id": conversation_id,
            "speaker_a": state["speaker_a"],
            "speaker_b": state["speaker_b"],
            "chunks": state["chunks"],
            "embeddings": state["embeddings"],
            "turn_context": turn_context,
        }
        update_path = self.output_dir / "rag_incremental_updates.jsonl"
        with update_path.open("a", encoding="utf-8") as file_object:
            file_object.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "sequence_index": sequence_index,
                        "first_changed_chunk": first_changed,
                        "chunks": tail_chunks,
                        "embeddings": tail_embeddings,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            file_object.flush()
            os.fsync(file_object.fileno())
        return {
            "changed_chunks": len(tail_chunks),
            "total_chunks": len(state["chunks"]),
        }

    async def finalize_incremental_conversation(
        self, conversation_id: str, **_kwargs
    ) -> None:
        """Persist the accumulated incremental index outside per-item timing."""
        if hasattr(self, "_incremental_add_index"):
            self._save_index(self._incremental_add_index, persist_embeddings=True)

    async def _build_conversation_index(self, conv: Conversation) -> Dict[str, Any]:
        if self.chunk_strategy == "turn":
            chunks = self._create_turn_chunks(conv, conv.conversation_id)
            turn_context = None
        else:
            document = self._format_conversation(conv)
            chunks = self._create_token_chunks(document, conv.conversation_id)
            turn_context = self._build_turn_context(
                conv,
                include_render_text=self.expand_turns_on_display,
            )
            self._annotate_token_chunks_with_turns(chunks, turn_context["turn_spans"])

        conv_index: Dict[str, Any] = {
            "conversation_id": conv.conversation_id,
            "speaker_a": conv.metadata.get("speaker_a", ""),
            "speaker_b": conv.metadata.get("speaker_b", ""),
            "chunks": chunks,
        }
        if turn_context is not None:
            conv_index["turn_context"] = turn_context

        if self._is_longmemeval_conversation_id(conv.conversation_id):
            return conv_index

        embeddings: List[List[float]] = []

        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            batch_texts = [chunk["content"] for chunk in batch]
            batch_embeddings = await self._embed_texts(
                batch_texts,
                stage="add",
                conversation_id=conv.conversation_id,
                metadata={"chunk_start": start, "chunk_count": len(batch)},
            )
            embeddings.extend(batch_embeddings)

        if self.pgvector_store:
            await self._ensure_pgvector_initialized()
            await self.pgvector_store.upsert_conversation(
                conversation_id=conv.conversation_id,
                chunks=chunks,
                embeddings=embeddings,
            )
        else:
            conv_index["embeddings"] = embeddings
        return conv_index

    def build_lazy_index(self, conversations: List[Conversation], output_dir: Any) -> Any:
        """Load a previously built local RAG index."""
        index_path = Path(output_dir) / self.INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"RAG index not found: {index_path}. Run the add stage first."
            )

        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)

        self._validate_index_metadata(index)
        for conv in conversations:
            cached = index.get("conversations", {}).get(conv.conversation_id, {})
            if cached.get("source_fingerprint") != fingerprint(conv):
                raise ValueError(
                    f"RAG source changed or lacks a fingerprint: {conv.conversation_id}. "
                    "Run the add stage before using this index."
                )

        if self.expand_turns_on_display:
            for conv_id, conv_index in index.get("conversations", {}).items():
                if not conv_index.get("turn_context"):
                    raise ValueError(
                        f"RAG index for {conv_id} lacks turn_context metadata. "
                        "Re-run the add stage with expand_turns_on_display enabled."
                    )

        return index

    async def search(
        self,
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs,
    ) -> SearchResult:
        """Retrieve chunks using embedding similarity and rerank candidates."""
        question_id = kwargs.get("question_id")
        question_date = kwargs.get("question_date")
        conv_index = index.get("conversations", {}).get(conversation_id)
        if not conv_index:
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"system": "rag", "error": "conversation not found"},
            )

        chunks = conv_index.get("chunks", [])
        if not chunks:
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"system": "rag", "error": "empty index"},
            )

        query_embedding = (
            await self._embed_texts(
                [query],
                stage="search",
                conversation_id=conversation_id,
                question_id=question_id,
            )
        )[0]

        use_transient_embeddings = self._is_longmemeval_conversation_id(conversation_id)

        temporal_filter_metadata: Dict[str, Any] = {}
        if self.pgvector_store and not use_transient_embeddings:
            filtered_chunks, _, temporal_filter_metadata = self._filter_chunks_for_question_date(
                chunks, [[] for _ in chunks], conv_index, conversation_id, question_date,
            )
            await self._ensure_pgvector_initialized()
            candidates = await self._get_pgvector_candidates(
                conversation_id=conversation_id,
                chunks=filtered_chunks,
                query_embedding=query_embedding,
            )
            embedding_candidate_source = "pgvector"
        else:
            embeddings = await self._ensure_conversation_embeddings(
                conv_index,
                conversation_id,
                stage="search",
            )
            if not embeddings:
                return SearchResult(
                    query=query,
                    conversation_id=conversation_id,
                    results=[],
                    retrieval_metadata={"system": "rag", "error": "empty index"},
                )
            filtered_chunks, filtered_embeddings, temporal_filter_metadata = (
                self._filter_chunks_for_question_date(
                    chunks=chunks,
                    embeddings=embeddings,
                    conv_index=conv_index,
                    conversation_id=conversation_id,
                    question_date=question_date,
                )
            )
            if not filtered_chunks:
                return SearchResult(
                    query=query,
                    conversation_id=conversation_id,
                    results=[],
                    retrieval_metadata={
                        "system": "rag",
                        "error": "no visible memories before question_date",
                        "question_id": question_id,
                        **temporal_filter_metadata,
                    },
                    question_id=question_id or "",
                )
            candidates = self._get_local_embedding_candidates(
                filtered_chunks,
                filtered_embeddings,
                query_embedding,
            )
            embedding_candidate_source = "local_transient" if use_transient_embeddings else "local"

        all_reranked = candidates
        if self.rerank_enabled and candidates:
            all_reranked = await self._rerank_candidates(
                query=query,
                candidates=candidates,
                top_n=len(candidates),
                conversation_id=conversation_id,
                question_id=question_id,
            )

        reranked, selection_metadata = self._select_final_rerank_results(all_reranked)
        ranked_candidates = list(enumerate(reranked, 1))
        if self.sort_selected_chunks_by_time:
            ranked_candidates.sort(key=self._selected_chunk_time_sort_key)

        results = []
        for rerank_rank, candidate in ranked_candidates:
            chunk = candidate["chunk"]
            results.append(
                {
                    "content": candidate["content"],
                    "score": float(candidate.get("rerank_score", candidate["similarity_score"])),
                    "metadata": {
                        "chunk_id": chunk.get("chunk_id"),
                        "chunk_index": chunk.get("chunk_index"),
                        "start_turn": chunk.get("start_turn"),
                        "end_turn": chunk.get("end_turn"),
                        "start_token": chunk.get("start_token"),
                        "end_token": chunk.get("end_token"),
                        "embedding_rank": candidate["embedding_rank"],
                        "rerank_rank": rerank_rank,
                        "similarity_score": candidate["similarity_score"],
                        "rerank_score": candidate.get("rerank_score"),
                    },
                }
            )

        all_rerank_results = self._build_all_rerank_results(all_reranked)
        formatted_context = self._format_context(results, conv_index)
        retrieval_metadata = {
            "system": "rag",
            "formatted_context": formatted_context,
            "top_k": len(results),
            "embedding_top_k": self.embedding_top_k,
            "rerank_top_k": self.rerank_top_k,
            "candidate_count": len(candidates),
            "all_rerank_results": all_rerank_results,
            "question_id": question_id,
            "dynamic_top_k": selection_metadata,
            "sort_selected_chunks_by_time": self.sort_selected_chunks_by_time,
            "storage_backend": self.storage_backend,
            "embedding_candidate_source": embedding_candidate_source,
            "transient_embeddings": use_transient_embeddings,
            "rerank_status": "completed" if self.rerank_enabled and candidates else (
                "no_candidates" if self.rerank_enabled else "disabled"
            ),
        }
        retrieval_metadata.update(temporal_filter_metadata)
        if self.pgvector_store:
            retrieval_metadata["pgvector_collection"] = self.pgvector_store.collection_name
            retrieval_metadata["pgvector_table"] = self.pgvector_store.table_name
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=retrieval_metadata,
            question_id=question_id or "",
        )

    @staticmethod
    def _selected_chunk_time_sort_key(
        ranked_candidate: Tuple[int, Dict[str, Any]],
    ) -> Tuple[int, int, int, int]:
        """Order selected chunks by original position, using rerank rank as fallback."""
        rerank_rank, candidate = ranked_candidate
        chunk = candidate.get("chunk", {})
        positions = (
            chunk.get("start_turn"),
            chunk.get("start_token"),
            chunk.get("chunk_index"),
        )
        for priority, value in enumerate(positions):
            if value is not None:
                chunk_index = chunk.get("chunk_index")
                return priority, int(value), int(chunk_index or 0), rerank_rank
        return len(positions), rerank_rank, 0, rerank_rank

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """Generate an answer with the configured prompt and return FINAL ANSWER only."""
        prompt_key = kwargs.get("prompt_key")
        if prompt_key and prompt_key.startswith("answer_prompt_mab_"):
            request_payload = {
                "model": self.answer_model,
                "messages": self._build_memoryagentbench_answer_messages(query, context),
                "temperature": self.answer_temperature,
                "max_tokens": self.answer_max_tokens,
            }
        else:
            prompt = self._resolve_answer_prompt(prompt_key).format(
                context=context, question=query
            )
            request_payload = {
                "model": self.answer_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.answer_temperature,
                "max_tokens": self.answer_max_tokens,
            }
        max_retries = int(self.config.get("answer", {}).get("max_retries", 3))

        for attempt in range(max_retries):
            started = time.perf_counter()
            try:
                response = await self.answer_client.chat.completions.create(
                    **request_payload
                )
                raw_response = self._model_dump(response)
                await self._write_request_log(
                    call_type="answer",
                    stage="answer",
                    request=request_payload,
                    response=raw_response,
                    started=started,
                    conversation_id=kwargs.get("conversation_id"),
                    question_id=kwargs.get("question_id"),
                    attempt=attempt + 1,
                )
                content = response.choices[0].message.content or ""
                final_answer = self._extract_final_answer(content)
                if final_answer:
                    return final_answer
            except Exception as exc:
                await self._write_request_log(
                    call_type="answer",
                    stage="answer",
                    request=request_payload,
                    response=None,
                    started=started,
                    conversation_id=kwargs.get("conversation_id"),
                    question_id=kwargs.get("question_id"),
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        return ""

    def _build_memoryagentbench_answer_messages(
        self, query: str, context: str
    ) -> List[Dict[str, str]]:
        """Mirror the official MemoryAgentBench answer call shape."""
        system_prompt = (
            "You are a helpful AI. Answer the question based on query and memories.\n"
            f"{context}\n"
        )
        user_content = (
            f"{query}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def _embed_texts(
        self,
        texts: List[str],
        stage: str,
        conversation_id: Optional[str] = None,
        question_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        request_payload = {"model": self.embedding_model, "input": texts}
        if metadata:
            request_payload["metadata"] = metadata

        for attempt in range(self.embedding_max_retries):
            started = time.perf_counter()
            try:
                api_payload = {
                    "model": self.embedding_model,
                    "input": texts,
                }
                response = await self.embedding_client.embeddings.create(**api_payload)
                raw_response = self._model_dump(response)
                await self._write_request_log(
                    call_type="embedding",
                    stage=stage,
                    request=request_payload,
                    response=raw_response,
                    started=started,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    attempt=attempt + 1,
                )
                return [item.embedding for item in response.data]
            except Exception as exc:
                await self._write_request_log(
                    call_type="embedding",
                    stage=stage,
                    request=request_payload,
                    response=None,
                    started=started,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == self.embedding_max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        return []

    async def _ensure_conversation_embeddings(
        self,
        conv_index: Dict[str, Any],
        conversation_id: str,
        stage: str,
    ) -> List[List[float]]:
        embeddings = conv_index.get("embeddings", [])
        chunks = conv_index.get("chunks", [])
        if self._has_complete_embeddings(conv_index):
            return embeddings
        if not chunks:
            return []

        lock = self._embedding_hydration_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            if self._has_complete_embeddings(conv_index):
                return conv_index.get("embeddings", [])
            await self._hydrate_conversation_embeddings(conv_index, conversation_id, stage=stage)
        return conv_index.get("embeddings", [])

    def release_conversation_search_cache(self, conversation_id: str, index: Any) -> None:
        """Drop LongMemEval transient embeddings after a conversation's search completes."""
        if not self._is_longmemeval_conversation_id(conversation_id):
            return
        if not isinstance(index, dict):
            return
        conv_index = index.get("conversations", {}).get(conversation_id)
        if isinstance(conv_index, dict):
            conv_index.pop("embeddings", None)
        self._embedding_hydration_locks.pop(conversation_id, None)
        gc.collect()

    async def _hydrate_conversation_embeddings(
        self,
        conv_index: Dict[str, Any],
        conversation_id: str,
        stage: str,
    ) -> None:
        chunks = conv_index.get("chunks", [])
        if not chunks:
            conv_index["embeddings"] = []
            return

        embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            batch_texts = [chunk["content"] for chunk in batch]
            batch_embeddings = await self._embed_texts(
                batch_texts,
                stage=stage,
                conversation_id=conversation_id,
                metadata={
                    "chunk_start": start,
                    "chunk_count": len(batch),
                    "hydrated_from_chunks": True,
                },
            )
            embeddings.extend(batch_embeddings)
        conv_index["embeddings"] = embeddings

    def _get_local_embedding_candidates(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        query_embedding: List[float],
    ) -> List[Dict[str, Any]]:
        scored = []
        for idx, embedding in enumerate(embeddings):
            scored.append(
                {
                    "index": idx,
                    "similarity_score": self._cosine_similarity(query_embedding, embedding),
                }
            )
        scored.sort(key=lambda item: item["similarity_score"], reverse=True)

        embedding_top_k = min(self.embedding_top_k, len(scored))
        candidates = []
        for rank, item in enumerate(scored[:embedding_top_k], 1):
            chunk = chunks[item["index"]]
            candidates.append(
                {
                    "index": item["index"],
                    "content": chunk["content"],
                    "chunk": chunk,
                    "embedding_rank": rank,
                    "similarity_score": item["similarity_score"],
                }
            )
        return candidates

    async def _get_pgvector_candidates(
        self,
        conversation_id: str,
        chunks: List[Dict[str, Any]],
        query_embedding: List[float],
    ) -> List[Dict[str, Any]]:
        if not self.pgvector_store or not chunks:
            return []
        chunk_by_id = {str(chunk.get("chunk_id")): idx for idx, chunk in enumerate(chunks)}
        chunk_by_index = {int(chunk.get("chunk_index", idx)): idx for idx, chunk in enumerate(chunks)}
        rows = await self.pgvector_store.search(
            conversation_id=conversation_id,
            query_embedding=query_embedding,
            top_k=self.embedding_top_k,
            allowed_chunk_ids=list(chunk_by_id),
        )
        candidates = []
        for rank, row in enumerate(rows, 1):
            idx = chunk_by_id.get(str(row.get("chunk_id")))
            if idx is None and row.get("chunk_index") is not None:
                idx = chunk_by_index.get(int(row["chunk_index"]))
            if idx is None:
                continue
            chunk = chunks[idx]
            candidates.append(
                {
                    "index": idx,
                    "content": chunk["content"],
                    "chunk": chunk,
                    "embedding_rank": rank,
                    "similarity_score": row["similarity_score"],
                }
            )
        return candidates

    def _filter_chunks_for_question_date(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        conv_index: Dict[str, Any],
        conversation_id: str,
        question_date: Optional[str],
    ) -> tuple[List[Dict[str, Any]], List[List[float]], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        if not question_date:
            return chunks, embeddings, metadata

        question_dt = self._parse_question_date(question_date)
        metadata["question_date"] = question_date
        metadata["temporal_filter_enabled"] = question_dt is not None
        if question_dt is None:
            metadata["temporal_filter_error"] = "failed to parse question_date"
            return chunks, embeddings, metadata

        turn_context = conv_index.get("turn_context", {})
        turn_filter_dates = self._get_turn_filter_dates(turn_context)
        if not turn_filter_dates:
            metadata["temporal_filter_error"] = "missing turn timestamps"
            return chunks, embeddings, metadata

        visible_chunks: List[Dict[str, Any]] = []
        visible_embeddings: List[List[float]] = []
        filtered_count = 0
        for chunk, embedding in zip(chunks, embeddings):
            if self._chunk_is_visible_at_question_date(
                chunk=chunk,
                conv_index=conv_index,
                question_dt=question_dt,
                turn_filter_dates=turn_filter_dates,
            ):
                visible_chunks.append(chunk)
                visible_embeddings.append(embedding)
            else:
                filtered_count += 1

        metadata.update(
            {
                "temporal_filter_granularity": "date",
                "temporal_filter_date_source": "session_date",
                "visible_chunk_count": len(visible_chunks),
                "future_filtered_chunk_count": filtered_count,
                "original_chunk_count": len(chunks),
            }
        )
        return visible_chunks, visible_embeddings, metadata

    def _chunk_is_visible_at_question_date(
        self,
        chunk: Dict[str, Any],
        conv_index: Dict[str, Any],
        question_dt: datetime,
        turn_filter_dates: Optional[List[Any]] = None,
    ) -> bool:
        turn_context = conv_index.get("turn_context", {})
        if turn_filter_dates is None:
            turn_filter_dates = self._get_turn_filter_dates(turn_context)
        if not turn_filter_dates:
            return True

        turn_range = self._get_chunk_turn_range(chunk, turn_context)
        if turn_range is None:
            return True

        question_date = question_dt.date()
        first_turn, last_turn = turn_range
        for turn_idx in range(first_turn, last_turn + 1):
            if turn_idx >= len(turn_filter_dates):
                continue
            turn_date = turn_filter_dates[turn_idx]
            if turn_date is not None and turn_date > question_date:
                return False
        return True

    def _get_turn_filter_dates(self, turn_context: Dict[str, Any]) -> List[Any]:
        turn_session_dates = turn_context.get("turn_session_dates", [])
        if turn_session_dates:
            return [
                parsed.date() if parsed is not None else None
                for parsed in (self._parse_question_date(value) for value in turn_session_dates)
            ]

        # Backward compatibility for indexes built before turn_session_dates existed.
        turn_timestamps = self._get_turn_timestamps(turn_context)
        if turn_timestamps:
            return [
                parsed.date() if parsed is not None else None
                for parsed in (self._parse_question_date(value) for value in turn_timestamps)
            ]
        return []

    def _get_turn_timestamps(self, turn_context: Dict[str, Any]) -> List[Optional[str]]:
        turn_timestamps = turn_context.get("turn_timestamps", [])
        if turn_timestamps:
            return turn_timestamps

        # Backward compatibility for indexes built before turn_timestamps existed.
        timestamps: List[Optional[str]] = []
        for line in turn_context.get("turn_lines", []):
            match = re.match(r"^\[(?P<date>\d{4}-\d{2}-\d{2})\]", str(line))
            timestamps.append(match.group("date") if match else None)
        return timestamps

    def _get_chunk_turn_range(
        self,
        chunk: Dict[str, Any],
        turn_context: Dict[str, Any],
    ) -> Optional[tuple[int, int]]:
        if chunk.get("start_turn") is not None and chunk.get("end_turn") is not None:
            start_turn = int(chunk["start_turn"])
            end_turn = int(chunk["end_turn"]) - 1
            return start_turn, max(start_turn, end_turn)
        return self._get_overlapping_turn_range(chunk, turn_context)

    @staticmethod
    def _parse_question_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value

        text = str(value).strip()
        for candidate in (text, text.upper(), text.title()):
            for fmt in (
                "%I:%M %p on %d %B, %Y",
                "%Y/%m/%d (%a) %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return datetime.strptime(candidate, fmt)
                except ValueError:
                    continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    async def _rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int,
        conversation_id: Optional[str],
        question_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        documents = [candidate["content"] for candidate in candidates]
        rerank_query, rerank_documents = self._format_rerank_inputs(query, documents)
        payload = {
            "model": self.rerank_model,
            "query": rerank_query,
            "documents": rerank_documents,
            "top_n": top_n,
        }
        url = f"{self.rerank_base_url}{self.rerank_endpoint}"

        for attempt in range(self.rerank_max_retries):
            started = time.perf_counter()
            try:
                raw_response = await asyncio.to_thread(
                    self._post_json,
                    url=url,
                    payload=payload,
                    timeout_seconds=self.rerank_timeout_seconds,
                )
                await self._write_request_log(
                    call_type="rerank",
                    stage="search",
                    request={"url": url, **payload},
                    response=raw_response,
                    started=started,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    attempt=attempt + 1,
                )
                return self._parse_rerank_results(raw_response, candidates, top_n)
            except Exception as exc:
                await self._write_request_log(
                    call_type="rerank",
                    stage="search",
                    request={"url": url, **payload},
                    response=None,
                    started=started,
                    conversation_id=conversation_id,
                    question_id=question_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == self.rerank_max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)

        raise RuntimeError("Reranker exhausted retries without a valid response")

    def _format_rerank_inputs(self, query: str, documents: List[str]) -> Tuple[str, List[str]]:
        """Apply model-specific rerank prompt formatting when required."""
        if "qwen3-reranker" not in str(self.rerank_model).lower():
            return query, documents

        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        instruction = (
            "Given a question and a passage, determine if the passage contains "
            "information relevant to answering the question."
        )

        formatted_query = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        formatted_documents = [f"<Document>: {document}{suffix}" for document in documents]
        return formatted_query, formatted_documents

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.rerank_api_key:
            headers["Authorization"] = f"Bearer {self.rerank_api_key}"

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def _parse_rerank_results(
        self,
        raw_response: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        if not isinstance(raw_response, dict):
            raise ValueError("Reranker response must be a JSON object")
        raw_results = raw_response.get("results")
        if raw_results is None and isinstance(raw_response.get("data"), dict):
            raw_results = raw_response["data"].get("results")
        expected = min(top_n, len(candidates))
        if not isinstance(raw_results, list) or len(raw_results) != expected:
            raise ValueError(f"Reranker must return exactly {expected} scored results")

        reranked = []
        seen = set()
        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError("Reranker result must be a JSON object")
            idx = item.get("index")
            if type(idx) is not int or not 0 <= idx < len(candidates) or idx in seen:
                raise ValueError("Reranker indices must be unique integers within the candidate range")
            seen.add(idx)
            candidate = dict(candidates[idx])
            score = item.get(
                "relevance_score",
                item.get("score", item.get("relevanceScore")),
            )
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Reranker score must be a finite number")
            candidate["rerank_score"] = float(score)
            reranked.append(candidate)

        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)

    def _build_all_rerank_results(
        self,
        reranked_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        all_results = []
        for rerank_rank, candidate in enumerate(reranked_candidates, 1):
            chunk = candidate["chunk"]
            all_results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_index": chunk.get("chunk_index"),
                    "start_turn": chunk.get("start_turn"),
                    "end_turn": chunk.get("end_turn"),
                    "start_token": chunk.get("start_token"),
                    "end_token": chunk.get("end_token"),
                    "embedding_rank": candidate["embedding_rank"],
                    "rerank_rank": rerank_rank,
                    "similarity_score": candidate["similarity_score"],
                    "rerank_score": candidate.get("rerank_score"),
                }
            )
        return all_results

    def _select_final_rerank_results(
        self,
        all_reranked: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.dynamic_top_k_enabled:
            selected = all_reranked[: self.rerank_top_k]
            return selected, {
                "enabled": False,
                "mode": "fixed_top_k",
                "score_threshold": None,
                "fallback_top_k": self.rerank_top_k,
                "selected_count": len(selected),
            }

        threshold_selected = [
            candidate
            for candidate in all_reranked
            if candidate.get("rerank_score", candidate["similarity_score"])
            >= self.dynamic_top_k_score_threshold
        ]
        if len(threshold_selected) >= self.dynamic_top_k_fallback_top_k:
            return threshold_selected, {
                "enabled": True,
                "mode": "score_threshold",
                "score_threshold": self.dynamic_top_k_score_threshold,
                "fallback_top_k": self.dynamic_top_k_fallback_top_k,
                "selected_count": len(threshold_selected),
            }

        selected = all_reranked[: self.dynamic_top_k_fallback_top_k]
        return selected, {
            "enabled": True,
            "mode": "fallback_top_k",
            "score_threshold": self.dynamic_top_k_score_threshold,
            "fallback_top_k": self.dynamic_top_k_fallback_top_k,
            "selected_count": len(selected),
            "threshold_selected_count": len(threshold_selected),
        }

    def _format_conversation_header(self, conversation: Conversation) -> str:
        return "\n".join(
            [
                f"Conversation ID: {conversation.conversation_id}",
                f"Speaker A: {conversation.metadata.get('speaker_a', '')}",
                f"Speaker B: {conversation.metadata.get('speaker_b', '')}",
            ]
        )

    def _format_message_line(self, msg: Message) -> str:
        if msg.timestamp:
            prefix = f"[{msg.timestamp.date().isoformat()}]"
        else:
            prefix = "[unknown date]"
        return f"{prefix} {msg.sender_name}: {msg.content}"

    def _format_conversation(self, conversation: Conversation) -> str:
        lines = [self._format_conversation_header(conversation), ""]
        lines.extend(self._format_message_line(msg) for msg in conversation.messages)
        return "\n".join(lines)

    def _text_unit_count(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encode_text_units(text))
        return len(text.split())

    def _encode_text_units(self, text: str) -> List[int]:
        """Encode literal dataset text without treating special-token strings specially."""
        return self._encoding.encode(text, disallowed_special=())

    def _join_turn_lines(self, parts: List[str]) -> str:
        return "\n".join(parts)

    def _create_turn_chunks(
        self, conversation: Conversation, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Pack complete message turns into chunks without splitting turn metadata."""
        if self.chunk_size == -1:
            return [
                {
                    "chunk_id": f"{conversation_id}_chunk_0",
                    "chunk_index": 0,
                    "content": self._format_conversation(conversation),
                    "start_turn": 0,
                    "end_turn": len(conversation.messages),
                    "start_token": 0,
                    "end_token": None,
                }
            ]

        header = self._format_conversation_header(conversation)
        header_units = self._text_unit_count(header)
        turns = [self._format_message_line(msg) for msg in conversation.messages]
        turn_units = [self._text_unit_count(turn) for turn in turns]

        if not turns:
            return [
                {
                    "chunk_id": f"{conversation_id}_chunk_0",
                    "chunk_index": 0,
                    "content": header,
                    "start_turn": 0,
                    "end_turn": 0,
                    "start_token": 0,
                    "end_token": 0,
                }
            ]

        chunks: List[Dict[str, Any]] = []
        turn_start = 0
        chunk_index = 0

        while turn_start < len(turns):
            parts: List[str] = []
            units = 0
            if chunk_index == 0:
                parts.append(header)
                units = header_units

            turn_end = turn_start
            while turn_end < len(turns):
                next_units = turn_units[turn_end]
                if parts and units + next_units > self.chunk_size:
                    break
                parts.append(turns[turn_end])
                units += next_units
                turn_end += 1

            chunks.append(
                {
                    "chunk_id": f"{conversation_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "content": self._join_turn_lines(parts),
                    "start_turn": turn_start,
                    "end_turn": turn_end,
                    "start_token": turn_start,
                    "end_token": turn_end,
                }
            )

            if turn_end >= len(turns):
                break

            if self.chunk_overlap <= 0:
                next_turn_start = turn_end
            else:
                overlap_units = 0
                overlap_turns = 0
                for idx in range(turn_end - 1, turn_start - 1, -1):
                    candidate_units = turn_units[idx]
                    if overlap_turns > 0 and overlap_units + candidate_units > self.chunk_overlap:
                        break
                    overlap_units += candidate_units
                    overlap_turns += 1

                next_turn_start = turn_end - overlap_turns
                if next_turn_start <= turn_start:
                    next_turn_start = turn_start + 1

            turn_start = next_turn_start
            chunk_index += 1

        return chunks

    def _create_token_chunks(self, document: str, conversation_id: str) -> List[Dict[str, Any]]:
        if self.chunk_size == -1:
            return [
                {
                    "chunk_id": f"{conversation_id}_chunk_0",
                    "chunk_index": 0,
                    "content": document,
                    "start_token": 0,
                    "end_token": None,
                }
            ]

        if self._encoding is None:
            return self._create_word_chunks(document, conversation_id)

        token_ids = self._encode_text_units(document)
        if not token_ids:
            token_ids = self._encode_text_units(" ")

        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for chunk_index, start in enumerate(range(0, len(token_ids), step)):
            end = min(start + self.chunk_size, len(token_ids))
            chunk_tokens = token_ids[start:end]
            chunks.append(
                {
                    "chunk_id": f"{conversation_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "content": self._encoding.decode(chunk_tokens),
                    "start_token": start,
                    "end_token": end,
                }
            )
            if end >= len(token_ids):
                break

        return chunks

    def _create_word_chunks(self, document: str, conversation_id: str) -> List[Dict[str, Any]]:
        words = document.split()
        if not words:
            words = [""]
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        for chunk_index, start in enumerate(range(0, len(words), step)):
            end = min(start + self.chunk_size, len(words))
            chunks.append(
                {
                    "chunk_id": f"{conversation_id}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "content": " ".join(words[start:end]),
                    "start_token": start,
                    "end_token": end,
                }
            )
            if end >= len(words):
                break
        return chunks

    def _build_turn_context(
        self,
        conversation: Conversation,
        include_render_text: bool = True,
    ) -> Dict[str, Any]:
        header = self._format_conversation_header(conversation)
        turn_lines = [self._format_message_line(msg) for msg in conversation.messages]
        turn_timestamps = [
            msg.timestamp.isoformat() if msg.timestamp else None
            for msg in conversation.messages
        ]
        session_dates_by_id: Dict[str, Optional[str]] = {}
        for msg in conversation.messages:
            session_id = msg.metadata.get("session") if msg.metadata else None
            if not session_id or session_id in session_dates_by_id:
                continue
            session_dates_by_id[session_id] = (
                msg.metadata.get("session_date")
                if msg.metadata and msg.metadata.get("session_date")
                else (msg.timestamp.date().isoformat() if msg.timestamp else None)
            )
        turn_session_dates = [
            session_dates_by_id.get(msg.metadata.get("session"))
            if msg.metadata and msg.metadata.get("session")
            else (msg.timestamp.date().isoformat() if msg.timestamp else None)
            for msg in conversation.messages
        ]
        turn_spans: List[List[int]] = []

        line_start = len(header) + 2
        if self._encoding is None:
            for line in turn_lines:
                line_end = line_start + len(line)
                turn_spans.append([line_start, line_end])
                line_start = line_end + 1
            turn_context = {
                "turn_spans": turn_spans,
                "turn_timestamps": turn_timestamps,
                "turn_session_dates": turn_session_dates,
            }
            if include_render_text:
                turn_context.update({"header": header, "turn_lines": turn_lines})
            return turn_context

        document = "\n".join([header, "", *turn_lines])
        token_ids = self._encode_text_units(document)
        if token_ids:
            _, token_start_offsets = self._encoding.decode_with_offsets(token_ids)
            token_end_offsets = token_start_offsets[1:] + [len(document)]
        else:
            token_start_offsets = []
            token_end_offsets = []

        for line in turn_lines:
            line_end = line_start + len(line)
            token_start = bisect_right(token_end_offsets, line_start)
            token_end = bisect_left(token_start_offsets, line_end)
            turn_spans.append([token_start, max(token_start, token_end)])
            line_start = line_end + 1

        turn_context = {
            "turn_spans": turn_spans,
            "turn_timestamps": turn_timestamps,
            "turn_session_dates": turn_session_dates,
        }
        if include_render_text:
            turn_context.update({"header": header, "turn_lines": turn_lines})
        return turn_context

    def _turns_overlapping_range(
        self, start: int, end: int, turn_spans: List[List[int]]
    ) -> List[int]:
        return [
            idx
            for idx, (turn_start, turn_end) in enumerate(turn_spans)
            if turn_start < end and turn_end > start
        ]

    def _annotate_token_chunks_with_turns(
        self, chunks: List[Dict[str, Any]], turn_spans: List[List[int]]
    ) -> None:
        if not turn_spans:
            return
        turn_starts = [int(span[0]) for span in turn_spans]
        turn_ends = [int(span[1]) for span in turn_spans]
        for chunk in chunks:
            start = int(chunk.get("start_token", 0))
            end = chunk.get("end_token")
            if end is None:
                continue
            chunk_end = int(end)
            first_turn = bisect_right(turn_ends, start)
            end_turn = bisect_left(turn_starts, chunk_end)
            if first_turn < end_turn:
                chunk["start_turn"] = first_turn
                chunk["end_turn"] = end_turn

    def _expand_chunk_to_turns(
        self,
        chunk: Dict[str, Any],
        turn_context: Dict[str, Any],
    ) -> str:
        turn_range = self._get_overlapping_turn_range(chunk, turn_context)
        if turn_range is None:
            return chunk.get("content", "")
        first_turn, last_turn = turn_range
        return self._render_turn_range(
            first_turn=first_turn,
            last_turn=last_turn,
            turn_context=turn_context,
            include_header=(chunk.get("chunk_index", 0) == 0 or first_turn == 0),
        )

    def _get_overlapping_turn_range(
        self,
        chunk: Dict[str, Any],
        turn_context: Dict[str, Any],
    ) -> Optional[tuple[int, int]]:
        turn_spans = turn_context.get("turn_spans", [])
        if not turn_spans:
            return None

        start = int(chunk.get("start_token", 0))
        end = chunk.get("end_token")
        if end is None:
            return None

        overlapping = self._turns_overlapping_range(start, int(end), turn_spans)
        if not overlapping:
            return None
        return overlapping[0], overlapping[-1]

    def _render_turn_range(
        self,
        first_turn: int,
        last_turn: int,
        turn_context: Dict[str, Any],
        include_header: bool,
        skip_turns: Optional[set[int]] = None,
    ) -> str:
        turn_lines = turn_context.get("turn_lines", [])
        header = turn_context.get("header", "")
        if not turn_lines:
            return ""
        parts: List[str] = []
        if include_header:
            parts.extend([header, ""])
        turns = range(first_turn, last_turn + 1)
        if skip_turns:
            turns = [turn_idx for turn_idx in turns if turn_idx not in skip_turns]
        parts.extend(turn_lines[turn_idx] for turn_idx in turns)
        return self._join_turn_lines(parts)

    def _format_context(
        self, results: List[Dict[str, Any]], conv_index: Optional[Dict[str, Any]] = None
    ) -> str:
        if not results:
            return "(No relevant memories found)"

        turn_context = None
        if self.expand_turns_on_display and conv_index:
            turn_context = conv_index.get("turn_context")
        seen_turns: set[int] = set()

        parts = []
        for idx, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            chunk = {
                "content": result.get("content", ""),
                "chunk_index": metadata.get("chunk_index"),
                "start_token": metadata.get("start_token"),
                "end_token": metadata.get("end_token"),
            }
            display_content = result.get("content", "")
            if turn_context:
                turn_range = self._get_overlapping_turn_range(chunk, turn_context)
                if turn_range is not None:
                    first_turn, last_turn = turn_range
                    if self.dedup_turns_on_display:
                        display_content = self._render_turn_range(
                            first_turn=first_turn,
                            last_turn=last_turn,
                            turn_context=turn_context,
                            include_header=(chunk.get("chunk_index", 0) == 0 or first_turn == 0),
                            skip_turns=seen_turns,
                        )
                        seen_turns.update(range(first_turn, last_turn + 1))
                        if not display_content.strip():
                            continue
                    else:
                        display_content = self._render_turn_range(
                            first_turn=first_turn,
                            last_turn=last_turn,
                            turn_context=turn_context,
                            include_header=(chunk.get("chunk_index", 0) == 0 or first_turn == 0),
                        )
                else:
                    display_content = result.get("content", "")
            parts.append(
                f"Memory {idx} "
                f"(chunk={metadata.get('chunk_index')}, "
                f"similarity={metadata.get('similarity_score'):.6f}, "
                f"rerank={metadata.get('rerank_score')}):\n"
                f"{display_content}"
            )
        return "\n\n".join(parts)

    def _get_answer_prompt(self) -> str:
        prompt_key = self.config.get("answer", {}).get("prompt_key", "answer_prompt_cot")
        prompts = self._prompts["online_api"]["default"]
        if prompt_key not in prompts:
            raise KeyError(
                f"RAG answer prompt key '{prompt_key}' not found in prompts.yaml"
            )
        return prompts[prompt_key]

    def _resolve_answer_prompt(self, prompt_key: Optional[str] = None) -> str:
        if prompt_key:
            prompts = self._prompts["online_api"]["default"]
            if prompt_key not in prompts:
                raise KeyError(
                    f"RAG answer prompt key '{prompt_key}' not found in prompts.yaml"
                )
            return prompts[prompt_key]
        return self._get_answer_prompt()

    def _extract_final_answer(self, text: str) -> str:
        pattern = r"(?:^|\n)\s*(?:#+\s*)?FINAL ANSWER\s*:\s*"
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if not matches:
            return text.strip()
        return text[matches[-1].end() :].strip()

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for a, b in zip(left, right):
            dot += a * b
            left_norm += a * a
            right_norm += b * b
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

    async def _write_request_log(
        self,
        call_type: str,
        stage: str,
        request: Dict[str, Any],
        response: Any,
        started: float,
        conversation_id: Optional[str] = None,
        question_id: Optional[str] = None,
        attempt: int = 1,
        error: Optional[str] = None,
    ) -> None:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "request_id": str(uuid.uuid4()),
            "call_type": call_type,
            "stage": stage,
            "conversation_id": conversation_id,
            "question_id": question_id,
            "attempt": attempt,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "request": request,
            "response": response,
            "error": error,
        }
        async with self._log_lock:
            with open(self.request_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _save_index(self, index: Dict[str, Any], persist_embeddings: bool = True) -> None:
        index_path = self.output_dir / self.INDEX_FILENAME
        serialized = self._build_serializable_index(
            index,
            persist_embeddings=persist_embeddings,
        )
        temporary = index_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        temporary.replace(index_path)
        self.console.print(f"Saved RAG index: {index_path}", style="green")

    def _build_serializable_index(
        self,
        index: Dict[str, Any],
        persist_embeddings: bool,
    ) -> Dict[str, Any]:
        serialized = dict(index)
        metadata = dict(serialized.get("metadata", {}))
        metadata["persist_embeddings"] = persist_embeddings
        serialized["metadata"] = metadata

        if persist_embeddings:
            return serialized

        serialized_conversations: Dict[str, Any] = {}
        for conv_id, conv_index in index.get("conversations", {}).items():
            conv_copy = dict(conv_index)
            conv_copy.pop("embeddings", None)
            serialized_conversations[conv_id] = conv_copy
        serialized["conversations"] = serialized_conversations
        return serialized

    def _build_index_metadata(self) -> Dict[str, Any]:
        metadata = {
            "chunk_strategy": self.chunk_strategy,
            "expand_turns_on_display": self.expand_turns_on_display,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "storage_backend": self.storage_backend,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_encoding": getattr(self._encoding, "name", None),
            "embedding_base_url": str(self.embedding_client.base_url),
        }
        if self.chunk_strategy == "token":
            metadata["turn_context_for_temporal_filter"] = True
        if self.pgvector_store:
            metadata.update(self.pgvector_store.metadata())
            metadata["pgvector_database_fingerprint"] = fingerprint(self.pgvector_store.database_url)
        return metadata

    def _validate_index_metadata(self, index: Dict[str, Any]) -> None:
        metadata = index.get("metadata", {})
        expected = self._build_index_metadata()
        for key, value in expected.items():
            index_value = metadata.get(key)
            if key == "chunk_strategy" and index_value is None:
                index_value = "token"
            if key == "expand_turns_on_display" and index_value is None:
                index_value = False
            if key == "storage_backend" and index_value is None:
                index_value = "local"
            if index_value != value:
                raise ValueError(
                    f"RAG index config mismatch for {key}: "
                    f"index={index_value!r}, current={value!r}. "
                    "Re-run the add stage or use a different output directory."
                )

    def _load_resumeable_index(self) -> Dict[str, Any]:
        index = {
            "type": "rag_local",
            "system": "rag",
            "metadata": self._build_index_metadata(),
            "conversations": {},
        }
        index_path = self.output_dir / self.INDEX_FILENAME
        if not index_path.exists():
            return index

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                cached_index = json.load(f)
            self._validate_index_metadata(cached_index)
            cached_convs = cached_index.get("conversations", {})
            if not isinstance(cached_convs, dict):
                raise ValueError("conversations must be a dict")
            index["conversations"] = cached_convs
            self.console.print(
                f"Loaded cached RAG index: {index_path} ({len(cached_convs)} conversations)",
                style="yellow",
            )
            return index
        except Exception as exc:
            self.console.print(
                f"Failed to reuse cached RAG index at {index_path}: {exc}. Rebuilding add stage from scratch.",
                style="yellow",
            )
            return index

    def _has_complete_embeddings(self, conv_index: Optional[Dict[str, Any]]) -> bool:
        if not conv_index:
            return False
        chunks = conv_index.get("chunks")
        embeddings = conv_index.get("embeddings")
        return isinstance(chunks, list) and bool(chunks) and isinstance(embeddings, list) and len(
            embeddings
        ) == len(chunks)

    def _is_conversation_index_reusable(
        self,
        conv_index: Optional[Dict[str, Any]],
        allow_missing_embeddings: bool = False,
    ) -> bool:
        if not conv_index:
            return False
        chunks = conv_index.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return False
        if not allow_missing_embeddings and not self._has_complete_embeddings(conv_index):
            return False
        if self.chunk_strategy == "token" and not conv_index.get("turn_context"):
            return False
        return True

    def _should_persist_embeddings(self, conversations: List[Conversation]) -> bool:
        return not self._all_longmemeval_conversations(conversations)

    @classmethod
    def _all_longmemeval_conversations(cls, conversations: List[Conversation]) -> bool:
        conversation_ids = [conv.conversation_id for conv in conversations if conv.conversation_id]
        if not conversation_ids:
            return False
        return all(cls._is_longmemeval_conversation_id(conv_id) for conv_id in conversation_ids)

    @staticmethod
    def _is_longmemeval_conversation_id(conversation_id: Optional[str]) -> bool:
        return bool(conversation_id and conversation_id.startswith("longmemeval_"))

    def _model_dump(self, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json")
        if hasattr(response, "dict"):
            return response.dict()
        return response

    async def close(self) -> None:
        for client in (self.embedding_client, self.answer_client):
            close = getattr(client, "close", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        if self.pgvector_store:
            await self.pgvector_store.close()

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "RAG",
            "type": "local_rag",
            "description": "Local chunk RAG with embedding similarity and rerank",
            "adapter": "RAGAdapter",
        }
