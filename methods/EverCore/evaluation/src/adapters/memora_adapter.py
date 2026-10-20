"""Adapter for Microsoft's local Memora implementation.

The upstream repository is loaded from ``external/Memora`` instead of being
vendored into EverCore.  Memora owns memory construction and retrieval; this
adapter only supplies dataset-neutral conversation/session boundaries and
converts results to the common evaluation schema.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


class MemoraProcessRecycleRequested(RuntimeError):
    """Ask the CLI supervisor to restart this resumable Memora worker."""

    exit_code = 75
    process_recycle_requested = True


@register_adapter("memora")
class MemoraAdapter(OnlineAPIAdapter):
    """Run Memora locally while retaining the shared answer/evaluator stages."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)
        memora = dict(config.get("memora", {}) or {})
        self.memora_root = self._resolve_memora_root(memora.get("root"))
        self.expected_revision = str(memora.get("revision", "")).strip()
        self.batch_size = max(1, int(memora.get("batch_size", 2)))
        self.segment_workers = max(1, int(memora.get("segment_workers", 2)))
        self.enable_segmentation = bool(memora.get("enable_segmentation", True))
        # MemoryAgentBench's official chunks can still be large enough for
        # Memora's structured factual extraction to exhaust the model's output
        # budget.  Bound each individual store request without changing the
        # benchmark conversation or query representation.
        self.store_chunk_chars = max(0, int(memora.get("store_chunk_chars", 10000)))
        self.recycle_every_sessions = max(
            0, int(memora.get("recycle_every_sessions", 0))
        )
        self.min_store_chunk_chars = max(
            1, int(memora.get("min_store_chunk_chars", 1000))
        )
        self.retrieval_strategy = str(
            (config.get("retrieval", {}) or {}).get("strategy", "prompt")
        ).lower()
        if self.retrieval_strategy not in {"semantic", "prompt"}:
            raise ValueError(
                "Memora retrieval.strategy must be 'semantic' or 'prompt' "
                "for benchmark evaluation"
            )

        self.persist_path = self.output_dir / str(
            memora.get("persist_dir", "memora_store")
        )
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.resume_path = self.output_dir / str(
            memora.get("resume_file", "memora_add_resume.json")
        )
        self._resume_lock = asyncio.Lock()
        # Segment construction is an LLM request per session.  Its concurrency
        # must not be coupled to the evaluation's search-worker count: a large
        # add batch otherwise overloads OpenAI-compatible gateways and leaves
        # their in-flight worker threads alive after a failed run.
        self._segment_semaphore = asyncio.Semaphore(self.segment_workers)
        self._client_lock = Lock()
        self._clients: Dict[str, Any] = {}
        self._client_class = None
        self._segmenter_class = None
        self._omega_conf = None
        self._validate_checkout()

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[5]

    def _resolve_memora_root(self, configured: Optional[str]) -> Path:
        root = Path(configured) if configured else self._repo_root() / "external" / "Memora"
        if not root.is_absolute():
            root = self._repo_root() / root
        return root.resolve()

    def _validate_checkout(self) -> None:
        marker = self.memora_root / "src" / "memora" / "memora_client.py"
        if not marker.exists():
            raise FileNotFoundError(
                f"Memora checkout not found at {self.memora_root}. Run "
                "scripts/memora/fetch_memora.ps1 first."
            )
        if self.expected_revision:
            completed = subprocess.run(
                ["git", "-C", str(self.memora_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            head = completed.stdout.strip()
            if head != self.expected_revision:
                raise RuntimeError(
                    f"Memora checkout is {head}, expected {self.expected_revision}. "
                    "Run scripts/memora/fetch_memora.ps1."
                )

    def _load_upstream(self) -> None:
        if self._client_class is not None:
            return
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        source_root = self.memora_root / "src"
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

        from omegaconf import OmegaConf
        from openai import OpenAI
        from memora.memora_client import MemoraClient
        import memora.utils.embedding as embedding_utils
        import memora.utils.llm as llm_utils

        # Upstream currently accepts separate Azure endpoints but relies on
        # OPENAI_BASE_URL for non-Azure clients.  Patch its factories in-memory
        # so the benchmark's independently configured LLM and embedding
        # endpoints match mem0_local without modifying the checkout.
        def build_llm_client(cfg):
            return OpenAI(
                api_key=cfg.openai.llm_api_key,
                base_url=cfg.openai.llm_api_base,
                timeout=float(cfg.openai.get("timeout_seconds", 90)),
                max_retries=int(cfg.openai.get("max_retries", 0)),
            )

        def build_embedding_client(cfg):
            return OpenAI(
                api_key=cfg.openai.embedding_api_key,
                base_url=cfg.openai.embedding_api_base,
                timeout=float(cfg.openai.get("timeout_seconds", 90)),
                max_retries=int(cfg.openai.get("max_retries", 0)),
            )

        llm_utils.get_openai_chat_completion_client = build_llm_client
        embedding_utils.get_openai_embedding_client = build_embedding_client
        if not getattr(llm_utils.ChatCompletionModel, "_evercore_temperature_patch", False):
            original_invoke = llm_utils.ChatCompletionModel._invoke_azure

            def deterministic_invoke(model_self, messages, response_format, source, **kwargs):
                kwargs.setdefault(
                    "temperature", float(model_self.cfg.llm.get("temperature", 0))
                )
                return original_invoke(
                    model_self, messages, response_format, source, **kwargs
                )

            llm_utils.ChatCompletionModel._invoke_azure = deterministic_invoke
            llm_utils.ChatCompletionModel._evercore_temperature_patch = True
        self._client_class = MemoraClient
        self._omega_conf = OmegaConf

        segmentation_path = (
            self.memora_root / "app" / "locomo" / "providers" / "memora" / "segmentation.py"
        )
        spec = importlib.util.spec_from_file_location(
            "evercore_memora_segmentation", segmentation_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Memora segmentation module: {segmentation_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._segmenter_class = module.ConversationSegmenter

    def _build_upstream_config(self, persist_path: Optional[Path] = None):
        self._load_upstream()
        memora = dict(self.config.get("memora", {}) or {})
        memory = dict(memora.get("memory", {}) or {})
        retrieval = dict(self.config.get("retrieval", {}) or {})
        llm = dict(memora.get("llm", {}) or {})
        embedding = dict(memora.get("embedding", {}) or {})
        return self._omega_conf.create(
            {
                "llm": {
                    "model": llm.get("model", self.config.get("llm", {}).get("model")),
                    "seed": int(llm.get("seed", 42)),
                    "temperature": float(llm.get("temperature", 0)),
                },
                "openai": {
                    "api_type": "openai",
                    "api_key": llm.get("api_key", ""),
                    "llm_api_key": llm.get("api_key", ""),
                    "llm_api_base": llm.get("base_url", "https://api.openai.com/v1"),
                    "embedding_api_key": embedding.get("api_key", ""),
                    "embedding_api_base": embedding.get(
                        "base_url", "https://api.openai.com/v1"
                    ),
                    "embedding_model": embedding.get(
                        "model", "text-embedding-3-small"
                    ),
                    "embedding_deployment_name": embedding.get(
                        "model", "text-embedding-3-small"
                    ),
                    "timeout_seconds": float(memora.get("timeout_seconds", 90)),
                    "max_retries": int(memora.get("sdk_max_retries", 0)),
                    "managed_identity": "",
                    "llm_api_version": "",
                    "embedding_api_version": "",
                },
                "memory": {
                    "memory_store": "evercore_memora",
                    "persist_path": str(persist_path or self.persist_path),
                    "collection_name": str(memory.get("collection_name", "memora")),
                    "db_type": "chromadb",
                    "distance": str(memory.get("distance", "cosine")),
                    "query_score_threshold": float(
                        memory.get("query_score_threshold", 0.4)
                    ),
                    "update_score_threshold": float(
                        memory.get("update_score_threshold", 0.8)
                    ),
                    "return_history": bool(memory.get("return_history", True)),
                    "enhance_query": bool(memory.get("enhance_query", False)),
                    "multimodal_support": False,
                    "top_k": int(self.config.get("search", {}).get("top_k", 20)),
                    "cue_top_k": int(memory.get("cue_top_k", 20)),
                    "enable_cue_index": bool(memory.get("enable_cue_index", True)),
                    "enable_hybrid_search": bool(
                        memory.get("enable_hybrid_search", True)
                    ),
                    "hybrid_search_method": str(
                        memory.get("hybrid_search_method", "bm25")
                    ),
                    "hybrid_top_k": int(memory.get("hybrid_top_k", 10)),
                    "bm25_score_threshold": float(
                        memory.get("bm25_score_threshold", 0.4)
                    ),
                    "enable_segmentation": self.enable_segmentation,
                    "enable_episodic_memory": bool(
                        memory.get("enable_episodic_memory", True)
                    ),
                    "use_segments_as_episodic": bool(
                        memory.get("use_segments_as_episodic", True)
                    ),
                },
                "retrieval": {
                    "strategy": self.retrieval_strategy,
                    "enable_llm_filter": bool(retrieval.get("enable_llm_filter", False)),
                    "enable_relaxed_frontier": bool(
                        retrieval.get("enable_relaxed_frontier", False)
                    ),
                    "prompted_policy": {
                        "max_steps": int(
                            (retrieval.get("prompted_policy", {}) or {}).get("max_steps", 4)
                        )
                    },
                },
                "eval": {"max_workers": max(1, self.num_workers)},
            }
        )

    @staticmethod
    def _user_id(conversation_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", conversation_id).strip("_")
        digest = hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:10]
        return f"{safe[:40] or 'conversation'}_{digest}"

    def _get_client(self, conversation_id: str):
        with self._client_lock:
            client = self._clients.get(conversation_id)
            if client is None:
                user_id = self._user_id(conversation_id)
                # A separate Chroma persistence directory per conversation is
                # required for safe parallel HNSW mutation.  User-specific
                # collections alone are insufficient because Chroma still
                # shares SQLite and HNSW state inside one PersistentClient.
                shard_path = self.persist_path / user_id
                shard_path.mkdir(parents=True, exist_ok=True)
                cfg = self._build_upstream_config(persist_path=shard_path)
                client = self._client_class(cfg=cfg, user_id=user_id)
                self._clients[conversation_id] = client
            return client

    def _load_resume_state(self) -> Dict[str, Any]:
        if not self.resume_path.exists():
            return {"version": 1, "conversations": {}}
        try:
            data = json.loads(self.resume_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "conversations": {}}
        if not isinstance(data.get("conversations"), dict):
            return {"version": 1, "conversations": {}}
        return data

    async def _save_resume_state(self, state: Dict[str, Any]) -> None:
        async with self._resume_lock:
            temp_path = self.resume_path.with_suffix(self.resume_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temp_path.replace(self.resume_path)

    @staticmethod
    def _conversation_signature(conversation: Conversation) -> str:
        payload = [
            [msg.sender_name, msg.content, msg.timestamp.isoformat() if msg.timestamp else ""]
            for msg in conversation.messages
        ]
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _session_groups(self, conversation: Conversation) -> List[List[Any]]:
        groups: Dict[str, List[Any]] = defaultdict(list)
        order: List[str] = []
        for index, message in enumerate(conversation.messages):
            session = str(message.metadata.get("session") or f"message_{index}")
            if session not in groups:
                order.append(session)
            groups[session].append(message)
        return [groups[key] for key in order]

    @staticmethod
    def _format_messages(messages: List[Any]) -> List[Dict[str, str]]:
        return [
            {"role": "user", "content": f"{message.sender_name}: {message.content}"}
            for message in messages
            if str(message.content).strip()
        ]

    @staticmethod
    def _split_text(content: str, limit: int) -> List[str]:
        """Split long store input near natural boundaries without losing text."""
        if limit <= 0 or len(content) <= limit:
            return [content]

        chunks = []
        remaining = content
        minimum_boundary = max(1, limit // 2)
        while len(remaining) > limit:
            window = remaining[:limit]
            split_at = -1
            for separator in ("\n\n", "\n", ". ", "! ", "? ", " "):
                candidate = window.rfind(separator, minimum_boundary, limit)
                if candidate >= 0:
                    split_at = candidate + len(separator)
                    break
            if split_at <= 0:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:]
        if remaining:
            chunks.append(remaining)
        return chunks

    def _store_message_chunks(
        self, messages: List[Dict[str, str]], limit: Optional[int] = None
    ) -> List[List[Dict[str, str]]]:
        """Create bounded Memora requests while preserving message order/content."""
        chunk_limit = self.store_chunk_chars if limit is None else max(1, limit)
        if chunk_limit <= 0:
            return [messages]

        requests = []
        current = []
        current_chars = 0
        for message in messages:
            content = str(message.get("content", ""))
            parts = self._split_text(content, chunk_limit)
            for part in parts:
                part_message = {**message, "content": part}
                if current and current_chars + len(part) > chunk_limit:
                    requests.append(current)
                    current = []
                    current_chars = 0
                current.append(part_message)
                current_chars += len(part)
        if current:
            requests.append(current)
        return requests or [[]]

    @staticmethod
    def _is_length_limit_error(error: BaseException) -> bool:
        """Recognize OpenAI structured-output truncation without hard coupling tests."""
        try:
            from openai import LengthFinishReasonError

            if isinstance(error, LengthFinishReasonError):
                return True
        except ImportError:
            pass
        return error.__class__.__name__ == "LengthFinishReasonError"

    @staticmethod
    def _episodic_key(metadata: Dict[str, Any]) -> str:
        return (
            f"[EPISODIC] {metadata.get('segment_topic', 'conversation')} "
            f"(segment {metadata.get('segment_index', 0)})"
        )

    def _remove_failed_episodic(self, client: Any, metadata: Dict[str, Any]) -> None:
        """Roll back the episodic entry Memora writes before factual extraction."""
        key = self._episodic_key(metadata)
        if client.get(key) is not None:
            client.delete(key)

    async def _add_store_with_fallback(
        self,
        client: Any,
        messages: List[Dict[str, str]],
        metadata: Dict[str, Any],
        store_key: str,
        completed: set,
        conv_state: Dict[str, Any],
        state: Dict[str, Any],
    ) -> int:
        """Add one request, recursively bisecting structured-output overflows."""
        if store_key in completed:
            return 0

        split_marker = f"{store_key}:split"
        should_split = split_marker in completed
        if not should_split:
            try:
                entries = await asyncio.to_thread(
                    self._add_to_store, client, messages, metadata
                )
            except Exception as error:
                if not self._is_length_limit_error(error):
                    raise
                await asyncio.to_thread(self._remove_failed_episodic, client, metadata)
                completed.add(split_marker)
                conv_state["completed"] = sorted(completed)
                await self._save_resume_state(state)
                should_split = True
            else:
                completed.add(store_key)
                conv_state["completed"] = sorted(completed)
                await self._save_resume_state(state)
                return len(entries or [])

        total_chars = sum(len(str(message.get("content", ""))) for message in messages)
        if total_chars <= self.min_store_chunk_chars:
            raise RuntimeError(
                "Memora structured output still reached the completion limit for "
                f"a {total_chars}-character input; lower memora.min_store_chunk_chars "
                "or use a model with a larger output limit."
            )
        retry_limit = max(self.min_store_chunk_chars, total_chars // 2)
        children = self._store_message_chunks(messages, limit=retry_limit)
        if len(children) < 2:
            raise RuntimeError(
                f"Could not split Memora store request {store_key} below {total_chars} chars"
            )

        added_entries = 0
        for child_index, child_messages in enumerate(children):
            child_metadata = {
                **metadata,
                "segment_index": f"{metadata.get('segment_index', 0)}.r{child_index}",
                "retry_chunk_index": child_index,
                "retry_chunk_count": len(children),
            }
            added_entries += await self._add_store_with_fallback(
                client,
                child_messages,
                child_metadata,
                f"{store_key}.r{child_index}",
                completed,
                conv_state,
                state,
            )
        return added_entries

    async def _segments_for_session(self, messages: List[Any]) -> List[Dict[str, Any]]:
        formatted = self._format_messages(messages)
        if not formatted:
            return []
        cfg = self._build_upstream_config()
        segmenter = self._segmenter_class(cfg)
        segmenter.batch_size = self.batch_size
        async with self._segment_semaphore:
            return await asyncio.to_thread(segmenter.segment_conversation, formatted)

    async def _add_conversation(
        self,
        conversation: Conversation,
        state: Dict[str, Any],
        stop_event: Optional[asyncio.Event] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        client = self._get_client(conversation.conversation_id)
        signature = self._conversation_signature(conversation)
        conv_state = state["conversations"].get(conversation.conversation_id)
        if (
            conv_state
            and conv_state.get("signature") == signature
            and conv_state.get("complete") is True
        ):
            self._advance_add_progress(kwargs, len(conversation.messages))
            return {
                "conversation_id": conversation.conversation_id,
                "entries": 0,
                "resumed_complete": True,
            }
        if not conv_state or conv_state.get("signature") != signature:
            await asyncio.to_thread(self._clear_client, client)
            conv_state = {"signature": signature, "completed": [], "complete": False}
            state["conversations"][conversation.conversation_id] = conv_state
            await self._save_resume_state(state)

        completed = set(conv_state.get("completed", []))
        added_entries = 0
        processed_sessions = 0
        session_groups = self._session_groups(conversation)
        for session_index, session_messages in enumerate(session_groups):
            session_changed = False
            if stop_event is not None and stop_event.is_set():
                return {
                    "conversation_id": conversation.conversation_id,
                    "entries": added_entries,
                    "stopped_after_peer_error": True,
                }
            formatted = self._format_messages(session_messages)
            segments = await self._segments_for_session(session_messages)
            for segment_index, segment in enumerate(segments):
                segment_key = f"{session_index}:{segment_index}"
                segment_indices = [int(i) for i in segment.get("indices", [])]
                segment_messages = [
                    formatted[i] for i in segment_indices if 0 <= i < len(formatted)
                ]
                if segment_key in completed or not segment_messages:
                    self._advance_add_progress(kwargs, len(segment_messages))
                    continue
                source = session_messages[min(segment_indices or [0])]
                timestamp = source.timestamp.isoformat() if source.timestamp else ""
                metadata = {
                    "timestamp": timestamp,
                    "conversation_id": conversation.conversation_id,
                    "session": source.metadata.get("session", f"session_{session_index}"),
                    "segment_topic": str(segment.get("topic", "conversation")),
                    "segment_index": segment_index,
                }
                store_requests = self._store_message_chunks(segment_messages)
                for store_index, store_messages in enumerate(store_requests):
                    if stop_event is not None and stop_event.is_set():
                        return {
                            "conversation_id": conversation.conversation_id,
                            "entries": added_entries,
                            "stopped_after_peer_error": True,
                        }
                    store_key = (
                        segment_key
                        if len(store_requests) == 1
                        else f"{segment_key}:{store_index}"
                    )
                    if store_key in completed:
                        continue
                    store_metadata = {
                        **metadata,
                        # Memora derives the episodic-memory key from
                        # segment_index. A unique suffix prevents later store
                        # chunks from overwriting earlier episodic text.
                        "segment_index": (
                            segment_index
                            if len(store_requests) == 1
                            else f"{segment_index}.{store_index}"
                        ),
                        "store_chunk_index": store_index,
                        "store_chunk_count": len(store_requests),
                    }
                    added_entries += await self._add_store_with_fallback(
                        client,
                        store_messages,
                        store_metadata,
                        store_key,
                        completed,
                        conv_state,
                        state,
                    )
                    session_changed = True
                self._advance_add_progress(kwargs, len(segment_messages))
            if session_changed:
                processed_sessions += 1
                if (
                    self.recycle_every_sessions
                    and processed_sessions >= self.recycle_every_sessions
                    and session_index + 1 < len(session_groups)
                ):
                    raise MemoraProcessRecycleRequested(
                        "Memora completed "
                        f"{processed_sessions} new session(s); restarting the worker "
                        "to release resident native memory."
                    )
        conv_state["complete"] = True
        await self._save_resume_state(state)
        if self.recycle_every_sessions and processed_sessions:
            raise MemoraProcessRecycleRequested(
                "Memora Add completed its final "
                f"{processed_sessions} new session(s); restarting the worker before "
                "the next pipeline stage to release resident native memory."
            )
        return {"conversation_id": conversation.conversation_id, "entries": added_entries}

    def _clear_client(self, client: Any) -> None:
        """Clear one conversation's independently persisted store."""
        client.clear()

    def _add_to_store(
        self, client: Any, messages: List[Dict[str, str]], metadata: Dict[str, Any]
    ) -> Any:
        """Build and store memory in one conversation-isolated Chroma shard."""
        return client.add(messages, type="chat", metadata=metadata)

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        state = self._load_resume_state()
        semaphore = asyncio.Semaphore(self.num_workers)
        stop_event = asyncio.Event()

        async def add_one(conversation):
            async with semaphore:
                try:
                    return await self._add_conversation(
                        conversation, state, stop_event=stop_event, **kwargs
                    )
                except Exception as error:
                    stop_event.set()
                    return error

        results = await asyncio.gather(*(add_one(conv) for conv in conversations))
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]
        return {
            "type": "local",
            "system": "memora",
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "add_results": results,
            "persist_path": str(self.persist_path),
            "persist_layout": "conversation_shards_v1",
        }

    async def add_incremental_message(
        self,
        conversation: Conversation,
        *,
        sequence_index: int,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Segment and append one message without whole-conversation clearing.

        The normal resumable Add compares a signature for the complete input and
        clears a shard when that signature changes.  A growing conversation
        necessarily changes on every step, so sequential benchmarks use stable,
        sequence-addressed store keys instead.
        """
        if len(conversation.messages) != 1:
            raise ValueError("Memora incremental ingestion requires one message")
        if not hasattr(self, "_incremental_add_state"):
            self._incremental_add_state = self._load_resume_state()

        state = self._incremental_add_state
        conv_state = state["conversations"].setdefault(
            conversation.conversation_id,
            {
                "mode": "incremental_v1",
                "completed": [],
                "complete": False,
            },
        )
        if conv_state.get("mode") != "incremental_v1":
            raise RuntimeError(
                "Memora output contains non-incremental resume state for this context"
            )
        completed = set(conv_state.get("completed", []))
        message = conversation.messages[0]
        formatted = self._format_messages([message])
        if not formatted:
            return {"entries": 0, "segments": 0}

        segments = await self._segments_for_session([message])
        if not segments:
            segments = [{"indices": [0], "topic": "conversation"}]
        client = self._get_client(conversation.conversation_id)
        added_entries = 0
        store_calls = 0
        for segment_index, segment in enumerate(segments):
            indices = [int(index) for index in segment.get("indices", [])]
            segment_messages = [
                formatted[index] for index in indices if 0 <= index < len(formatted)
            ]
            if not segment_messages:
                continue
            requests = self._store_message_chunks(segment_messages)
            for store_index, store_messages in enumerate(requests):
                store_key = (
                    f"incremental:{sequence_index}:{segment_index}:{store_index}"
                )
                metadata = {
                    "timestamp": message.timestamp.isoformat() if message.timestamp else "",
                    "conversation_id": conversation.conversation_id,
                    "session": message.metadata.get(
                        "session", f"message_{sequence_index}"
                    ),
                    "segment_topic": str(segment.get("topic", "conversation")),
                    "segment_index": f"inc.{sequence_index}.{segment_index}.{store_index}",
                    "incremental_sequence_index": sequence_index,
                    "store_chunk_index": store_index,
                    "store_chunk_count": len(requests),
                }
                added_entries += await self._add_store_with_fallback(
                    client,
                    store_messages,
                    metadata,
                    store_key,
                    completed,
                    conv_state,
                    state,
                )
                store_calls += 1
        conv_state["completed"] = sorted(completed)
        await self._save_resume_state(state)
        return {
            "entries": added_entries,
            "segments": len(segments),
            "store_calls": store_calls,
        }

    async def finalize_incremental_conversation(
        self, conversation_id: str, **_kwargs
    ) -> None:
        if not hasattr(self, "_incremental_add_state"):
            return
        conv_state = self._incremental_add_state["conversations"].get(conversation_id)
        if conv_state is not None:
            conv_state["complete"] = True
            await self._save_resume_state(self._incremental_add_state)

    def build_lazy_index(self, conversations: List[Conversation], output_dir: Any) -> Any:
        return {"persist_path": str(self.persist_path)}

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        top_k = int(kwargs.get("top_k", self.config.get("search", {}).get("top_k", 20)))
        question_date = kwargs.get("question_date")
        fetch_top_k = self._temporal_fetch_top_k(top_k, question_date)
        client = self._get_client(conversation_id)
        memories = await asyncio.to_thread(
            client.advance_query,
            query,
            top_k=fetch_top_k,
            query_type=self.retrieval_strategy,
        )
        results = []
        for memory in memories or []:
            timestamp = str(getattr(memory, "timestamp", "") or "")
            value = memory.get_memory_value(
                return_history=bool(
                    (self.config.get("memora", {}).get("memory", {}) or {}).get(
                        "return_history", True
                    )
                )
            )
            episode_blocks = []
            for episode_id in dict.fromkeys(
                getattr(memory, "episodic_memory_ids", []) or []
            ):
                episode = await asyncio.to_thread(client.get, episode_id)
                if episode is None:
                    continue
                episode_timestamp = str(getattr(episode, "timestamp", "") or "")
                episode_value = episode.get_memory_value(return_history=False)
                episode_blocks.append(
                    f"Conversation on {episode_timestamp or 'unknown date'}:\n{episode_value}"
                )
            factual = f"{timestamp}: {value}" if timestamp else str(value)
            content = "\n".join([*episode_blocks, f"Details: {factual}"])
            results.append(
                {
                    "content": content,
                    "score": float(getattr(memory, "score", 0.0) or 0.0),
                    "user_id": self._user_id(conversation_id),
                    "metadata": {
                        "index": getattr(memory, "index", ""),
                        "timestamp": timestamp,
                        "memory_type": getattr(memory, "memory_type", ""),
                        "episodic_memory_ids": list(
                            getattr(memory, "episodic_memory_ids", []) or []
                        ),
                    },
                }
            )
        results, temporal = self._filter_results_for_question_date(
            results,
            question_date,
            requested_top_k=top_k,
            fetch_top_k=fetch_top_k,
        )
        formatted_context = "\n".join(
            f"{number}. {item['content']}"
            for number, item in enumerate(results, start=1)
        )
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            question_id=str(kwargs.get("question_id", "")),
            retrieval_metadata={
                "system": "memora",
                "top_k": top_k,
                "retrieval_strategy": self.retrieval_strategy,
                "formatted_context": formatted_context,
                "combined_conversation_memory": True,
                **temporal,
            },
        )

    # These hooks remain abstract in OnlineAPIAdapter, although this adapter
    # overrides the complete add/search templates above.
    async def _add_user_messages(self, conv, messages, speaker, **kwargs):
        raise NotImplementedError

    async def _search_single_user(self, query, conversation_id, user_id, top_k, **kwargs):
        raise NotImplementedError

    def _build_single_search_result(self, *args, **kwargs):
        raise NotImplementedError

    def _build_dual_search_result(self, *args, **kwargs):
        raise NotImplementedError

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "Memora",
            "type": "local",
            "description": "Microsoft Memora local SDK",
            "adapter": "MemoraAdapter",
            "upstream_revision": self.expected_revision,
        }
