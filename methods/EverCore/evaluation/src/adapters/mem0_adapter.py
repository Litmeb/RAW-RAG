"""
Mem0 Adapter - adapt Mem0 online API for evaluation framework.
Reference: https://mem0.ai/

Key features:
- Dual-perspective handling: separate storage and retrieval for speaker_a and speaker_b
- Supports custom instructions
"""

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from rich.console import Console

from evaluation.src.adapters.mem0_add_audit import (
    Mem0AddAuditWriteError,
    append_jsonl_fsynced,
    run_audited_add,
)
from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.datasets.memoryagentbench import MAB_SYSTEM_MESSAGE
from evaluation.src.utils.config import load_yaml


class _AsyncLocalMemoryClient:
    """Small async facade over mem0's synchronous local Memory SDK."""

    def __init__(self, memory: Any):
        self._memory = memory
        # QdrantLocal mutates several parallel numpy arrays during writes. A
        # concurrent search can observe one array after it has grown but another
        # before it has, which surfaces as an out-of-bounds payload-filter access.
        # Keep each synchronous Memory operation atomic while still moving the
        # blocking SDK work off the event loop.
        self._operation_lock = asyncio.Lock()

    @staticmethod
    async def _run_thread(operation: Callable[..., Any], *args, **kwargs) -> Any:
        """Run blocking SDK work without abandoning its thread on cancellation.

        Cancelling ``asyncio.to_thread`` only cancels the asyncio waiter; Python
        cannot stop the already-running worker thread. Wait for that worker
        before propagating cancellation so adapter cleanup cannot close Qdrant
        while the SDK is still using it.
        """
        worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise

    async def _run_serialized(self, operation: Callable[..., Any], **kwargs) -> Any:
        async with self._operation_lock:
            return await self._run_thread(operation, **kwargs)

    async def add(self, **kwargs) -> Any:
        audit_context = kwargs.pop("_audit_context", None)
        audit_callback = kwargs.pop("_audit_callback", None)
        if audit_context is None or not callable(audit_callback):
            return await self._run_serialized(self._memory.add, **kwargs)

        record: Dict[str, Any] = {}
        async with self._operation_lock:
            try:
                result = await self._run_thread(
                    run_audited_add,
                    self._memory,
                    add_kwargs=kwargs,
                    context=audit_context,
                    record=record,
                )
            except BaseException:
                await audit_callback(record)
                raise
            else:
                await audit_callback(record)
                return result

    async def search(self, **kwargs) -> Any:
        return await self._run_serialized(self._memory.search, **kwargs)

    async def delete_all(self, **kwargs) -> Any:
        return await self._run_serialized(self._memory.delete_all, **kwargs)

    async def update_project(self, **kwargs) -> Any:
        return None

    async def close(self) -> None:
        store_clients = []
        for store_name in ("vector_store", "entity_store"):
            store = getattr(self._memory, store_name, None)
            client = getattr(store, "client", None)
            if client is not None and all(
                id(client) != id(existing) for existing in store_clients
            ):
                store_clients.append(client)

        for client in store_clients:
            close = getattr(client, "close", None)
            if callable(close):
                await self._run_serialized(close)

        history_db = getattr(self._memory, "db", None)
        close_history = getattr(history_db, "close", None)
        if callable(close_history):
            await self._run_serialized(close_history)


@register_adapter("mem0")
class Mem0Adapter(OnlineAPIAdapter):
    """
    Mem0 online API adapter.

    Supports:
    - Standard memory storage and retrieval

    Config example:
    ```yaml
    adapter: "mem0"
    api_key: "${MEM0_API_KEY}"
    batch_size: 2
    display_timezone_offset: 8  # Optional: convert timestamps to UTC+8 for display
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Set custom instructions (loaded from prompts.yaml)
        # Prioritize config settings (backward compatible), otherwise load from prompts
        custom_instructions = config.get("custom_instructions", None)
        if not custom_instructions:
            # Load from prompts.yaml
            custom_instructions = (
                self._prompts.get("add_stage", {})
                .get("mem0", {})
                .get("custom_instructions", None)
            )
            print("   ✅ Custom instructions set (from prompts.yaml)")

        # Store custom_instructions for async initialization
        self._custom_instructions = custom_instructions
        self.client_mode = str(config.get("client_mode", "cloud")).lower()
        local_config = config.get("local", {}) or {}
        self._local_shard_by_conversation = bool(
            local_config.get("shard_by_conversation", False)
        )
        self._local_http_timeout_seconds = float(
            local_config.get("http_timeout_seconds", 90.0)
        )
        self._local_http_max_retries = max(
            0, int(local_config.get("http_max_retries", 0))
        )
        self._local_memory_action_policy = str(
            local_config.get("memory_action_policy", "default")
        ).lower()
        if self._local_memory_action_policy not in {"default", "add_only"}:
            raise ValueError(
                "Unsupported local.memory_action_policy "
                f"{self._local_memory_action_policy!r}. Expected 'default' or "
                "'add_only'."
            )
        self._local_shard_clients: Dict[str, _AsyncLocalMemoryClient] = {}
        self._local_shard_client_tasks: Dict[str, asyncio.Task] = {}
        self.client = self._create_mem0_client(config)
        self.batch_size = config.get("batch_size", 2)
        self.max_retries = config.get("max_retries", 3)
        self.max_content_length = config.get("max_content_length", 12000)
        self.add_interval = config.get("add_interval", 0.0)
        self.add_timeout_split_enabled = bool(
            config.get("add_timeout_split_on_timeout", self.client_mode == "local")
        )
        self.add_timeout_split_max_attempts = max(
            1, int(config.get("add_timeout_split_max_attempts", 1))
        )
        self.search_interval = config.get("search", {}).get("search_interval", 0.0)
        self.search_operation_max_attempts = max(
            1, int(config.get("search", {}).get("sdk_operation_max_attempts", 1))
        )
        self.debug_search_results = bool(
            config.get("search", {}).get("debug_raw_results", False)
        )
        self.mem0_search_rerank = bool(
            config.get("search", {}).get("mem0_rerank", self.client_mode == "local")
        )
        resume_config = config.get("add_resume", {})
        self.add_resume_enabled = resume_config.get("enabled", True)
        self.add_resume_state_file = resume_config.get(
            "state_file", "mem0_add_resume.json"
        )
        self.add_resume_flush_every_batches = max(
            1, int(resume_config.get("flush_every_batches", 20))
        )
        self.add_resume_partial = bool(resume_config.get("resume_partial", False))
        self._add_resume_state_path = None
        self._add_resume_state = None
        self._add_resume_state_lock = asyncio.Lock()
        self._add_resume_dirty_batches = 0
        audit_config = dict(config.get("add_audit", {}) or {})
        self.add_audit_enabled = bool(
            audit_config.get("enabled", self.client_mode == "local")
        )
        self.add_audit_file = str(
            audit_config.get("file", "mem0_add_audit.jsonl")
        )
        self.add_audit_fsync = bool(audit_config.get("fsync_each_record", True))
        self._add_audit_path = self.output_dir / self.add_audit_file
        self._add_audit_lock = asyncio.Lock()
        self._add_audit_attempts: Dict[str, int] = {}
        self.console = Console()

        print(f"   Mem0 Client Mode: {self.client_mode}")
        print(f"   Batch Size: {self.batch_size}")
        print(f"   Max Content Length: {self.max_content_length}")
        if self.add_interval > 0:
            print(f"   Add Interval: {self.add_interval}s (rate limiting)")
        if self.search_interval > 0:
            print(f"   Search Interval: {self.search_interval}s (rate limiting)")
        if self.add_resume_enabled:
            print(
                f"   Add Resume: enabled ({self.add_resume_state_file}, "
                f"flush every {self.add_resume_flush_every_batches} batches)"
            )
            if self.add_resume_partial:
                print("   Partial Add Resume: enabled")
        if self.add_audit_enabled and self.client_mode == "local":
            print(
                f"   Add Audit: enabled ({self.add_audit_file}, "
                f"fsync={self.add_audit_fsync})"
            )
        if self._uses_local_conversation_shards():
            print("   Local Storage: one QdrantLocal/SQLite shard per conversation")
        if self.client_mode == "local":
            print(
                "   Local SDK HTTP: "
                f"timeout={self._local_http_timeout_seconds:g}s, "
                f"retries={self._local_http_max_retries}"
            )
            if self._local_memory_action_policy == "add_only":
                print("   Local Memory Actions: ADD/NONE only (UPDATE→ADD, DELETE→NONE)")

    def _create_mem0_client(self, config: dict) -> Any:
        if self.client_mode == "cloud":
            try:
                from mem0 import AsyncMemoryClient
            except ImportError:
                raise ImportError(
                    "Mem0 client not installed. Please install: pip install mem0ai"
                )

            api_key = config.get("api_key", "")
            if not api_key:
                raise ValueError("Mem0 API key is required. Set 'api_key' in config.")
            return AsyncMemoryClient(api_key=api_key)

        if self.client_mode == "local":
            # Mem0 reads this flag while importing mem0.memory.telemetry. Set it
            # before importing the SDK so blocked PostHog requests cannot stall
            # local evaluation runs or flood their logs.
            local_config = dict(config.get("local", {}) or {})
            telemetry_enabled = bool(local_config.get("telemetry_enabled", False))
            os.environ["MEM0_TELEMETRY"] = str(telemetry_enabled).lower()

            if self._uses_local_conversation_shards():
                return None
            return self._create_local_mem0_client(config)

        raise ValueError(
            f"Unsupported Mem0 client_mode '{self.client_mode}'. "
            "Expected 'cloud' or 'local'."
        )

    def _load_model_config(self, path_value: Optional[str]) -> Dict[str, Any]:
        if not path_value:
            return {}

        path = Path(path_value)
        if not path.is_absolute():
            evaluation_root = Path(__file__).parent.parent.parent
            path = evaluation_root / path
        return load_yaml(str(path))

    def _uses_local_conversation_shards(self) -> bool:
        return self.client_mode == "local" and self._local_shard_by_conversation

    @staticmethod
    def _conversation_shard_key(conversation_id: str) -> str:
        raw_key = str(conversation_id or "unknown")
        safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_key).strip("._-")
        safe_key = safe_key or "conversation"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:10]
        return f"{safe_key[:80]}-{digest}"

    def _local_shard_dir(self, conversation_id: str) -> Path:
        local_config = dict(self.config.get("local", {}) or {})
        shard_root_value = local_config.get("shard_root")
        if shard_root_value:
            shard_root = Path(shard_root_value)
            if not shard_root.is_absolute():
                shard_root = self.output_dir / shard_root
        else:
            shard_root = self.output_dir / "mem0_shards"
        return shard_root / self._conversation_shard_key(conversation_id)

    def _create_local_mem0_client(
        self, config: dict, conversation_id: Optional[str] = None
    ) -> _AsyncLocalMemoryClient:
        try:
            from mem0 import Memory
        except ImportError:
            raise ImportError(
                "Mem0 local SDK not installed. Please install: pip install mem0ai"
            )

        memory_config = self._build_local_memory_config(config, conversation_id)
        memory = Memory.from_config(memory_config)
        self._configure_local_http_clients(memory)
        self._apply_local_memory_action_policy(memory)
        return _AsyncLocalMemoryClient(memory)

    @staticmethod
    def _rewrite_mem0_actions_add_only(response: Any) -> Any:
        """Convert Mem0 v1 UPDATE/DELETE decisions into ADD/NONE decisions.

        Mem0 v1 makes its mutations from the JSON returned by the second LLM
        call in ``Memory._add_to_vector_store``.  Rewriting that response is
        deliberately done before Mem0 consumes it, rather than merely asking
        the LLM to comply in a prompt: this guarantees the benchmark cannot
        invoke ``_update_memory`` or ``_delete_memory`` even on a malformed or
        non-compliant model response.  An update's replacement text remains as
        a new fact; a delete is a no-op.
        """
        if isinstance(response, dict):
            payload = response
            serialize = False
        elif isinstance(response, str):
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = re.sub(
                    r"^```(?:json)?\s*", "", response_text, flags=re.IGNORECASE
                )
                response_text = re.sub(r"\s*```$", "", response_text)
            try:
                payload = json.loads(response_text)
            except (TypeError, ValueError):
                return response
            serialize = True
        else:
            return response

        actions = payload.get("memory") if isinstance(payload, dict) else None
        if not isinstance(actions, list):
            return response

        changed = False
        rewritten_actions = []
        for action in actions:
            if not isinstance(action, dict):
                rewritten_actions.append(action)
                continue
            rewritten = dict(action)
            event = str(rewritten.get("event", "")).upper()
            if event == "UPDATE":
                rewritten["event"] = "ADD"
                # An ADD has no target memory.  Removing these fields makes
                # the persisted decision unambiguous in the add audit log.
                rewritten.pop("id", None)
                rewritten.pop("old_memory", None)
                changed = True
            elif event == "DELETE":
                rewritten["event"] = "NONE"
                # Mem0's NONE branch can still call vector_store.update to
                # attach a run/agent id when an id is present.  This variant
                # promises no update operation at all, so make it a true noop.
                rewritten.pop("id", None)
                rewritten.pop("old_memory", None)
                changed = True
            elif event == "NONE" and rewritten.get("id") is not None:
                # See the DELETE case above: remove incidental metadata-only
                # vector updates from otherwise no-op decisions as well.
                rewritten.pop("id", None)
                rewritten.pop("old_memory", None)
                changed = True
            rewritten_actions.append(rewritten)

        if not changed:
            return response
        rewritten_payload = dict(payload)
        rewritten_payload["memory"] = rewritten_actions
        return json.dumps(rewritten_payload) if serialize else rewritten_payload

    def _apply_local_memory_action_policy(self, memory: Any) -> None:
        """Install the configured mutation policy on a local Mem0 v1 instance."""
        if self._local_memory_action_policy != "add_only":
            return
        llm = getattr(memory, "llm", None)
        original_generate = getattr(llm, "generate_response", None)
        if not callable(original_generate):
            raise RuntimeError(
                "Mem0 add-only policy requires memory.llm.generate_response"
            )

        def add_only_generate(*args: Any, **kwargs: Any) -> Any:
            return self._rewrite_mem0_actions_add_only(
                original_generate(*args, **kwargs)
            )

        llm.generate_response = add_only_generate

    def _configure_local_http_clients(self, memory: Any) -> None:
        """Bound network calls made by the synchronous local Mem0 SDK.

        Mem0 creates OpenAI-compatible clients without an explicit timeout. Its
        synchronous search is run in a worker thread, and cancelling the
        awaiting coroutine cannot terminate that thread. A short per-request
        timeout lets the normal sequence of query/entity embedding calls return
        before the longer outer Search-stage retry boundary.
        """
        for component_name in ("embedding_model", "llm", "reranker"):
            component = getattr(memory, component_name, None)
            api_client = getattr(component, "client", None)
            with_options = getattr(api_client, "with_options", None)
            if not callable(with_options):
                continue
            component.client = with_options(
                timeout=self._local_http_timeout_seconds,
                max_retries=self._local_http_max_retries,
            )

    async def _client_for_conversation(self, conversation_id: str) -> Any:
        if not self._uses_local_conversation_shards():
            return self.client

        shard_key = self._conversation_shard_key(conversation_id)
        existing = self._local_shard_clients.get(shard_key)
        if existing is not None:
            return existing

        task = self._local_shard_client_tasks.get(shard_key)
        if task is None:
            task = asyncio.create_task(
                asyncio.to_thread(
                    self._create_local_mem0_client,
                    self.config,
                    conversation_id,
                )
            )
            self._local_shard_client_tasks[shard_key] = task

        try:
            client = await task
        except BaseException:
            self._local_shard_client_tasks.pop(shard_key, None)
            raise

        self._local_shard_clients[shard_key] = client
        self._local_shard_client_tasks.pop(shard_key, None)
        return client

    def _build_local_memory_config(
        self, config: dict, conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        local_config = dict(config.get("local", {}) or {})
        model_files = local_config.get("model_config_files", {}) or {}

        llm_config = self._load_model_config(model_files.get("llm"))
        embedding_config = self._load_model_config(model_files.get("embedding"))
        reranker_config = self._load_model_config(model_files.get("reranker"))

        memory_config = dict(local_config.get("memory", {}) or {})
        if llm_config:
            memory_config["llm"] = llm_config.get("mem0", llm_config)
        if embedding_config:
            embedder_config = dict(embedding_config.get("mem0", embedding_config))
            embedder_provider_config = dict(embedder_config.get("config", {}) or {})
            embedding_dims = embedder_provider_config.get("embedding_dims")
            if embedding_dims is not None:
                try:
                    embedding_dims = int(embedding_dims)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        "Mem0 local embedding_dims must be a positive integer; "
                        f"got {embedding_dims!r}. Check VECTORIZE_DIMS."
                    ) from e
                if embedding_dims <= 0:
                    raise ValueError(
                        "Mem0 local embedding_dims must be a positive integer; "
                        f"got {embedding_dims!r}. Check VECTORIZE_DIMS."
                    )
                embedder_provider_config["embedding_dims"] = embedding_dims
            embedder_config["config"] = embedder_provider_config
            memory_config["embedder"] = embedder_config
        if reranker_config:
            memory_config["reranker"] = reranker_config.get("mem0", reranker_config)

        vector_store = dict(memory_config.get("vector_store", {}) or {})
        vector_config = dict(vector_store.get("config", {}) or {})
        if self._uses_local_conversation_shards():
            if conversation_id is None:
                raise ValueError(
                    "conversation_id is required for sharded Mem0 local storage"
                )
            shard_dir = self._local_shard_dir(conversation_id)
            vector_config["path"] = str(shard_dir / "qdrant")
            memory_config["history_db_path"] = str(shard_dir / "history.db")
        elif not vector_config.get("path"):
            vector_config["path"] = str(self.output_dir / "mem0_qdrant")
        vector_store["config"] = vector_config
        memory_config["vector_store"] = vector_store or {
            "provider": "qdrant",
            "config": vector_config,
        }

        if not memory_config.get("history_db_path"):
            memory_config["history_db_path"] = str(self.output_dir / "mem0_history.db")

        if self._custom_instructions:
            memory_config["custom_instructions"] = self._custom_instructions

        return memory_config

    async def _add_messages_to_client(
        self,
        *,
        messages: List[Dict[str, Any]],
        source_timestamp: Optional[datetime],
        user_id: str,
        run_id: str,
        metadata: Dict[str, Any],
    ) -> Any:
        """Add messages using the temporal fields supported by each Mem0 mode."""
        if self.client_mode == "local":
            client = await self._client_for_conversation(run_id)
            local_metadata = dict(metadata)
            if source_timestamp is not None:
                # OSS Mem0 rejects the platform-only top-level `timestamp`
                # argument, but honors `created_at` supplied as metadata when it
                # creates the vector-store record. Search then returns it as the
                # memory's top-level created_at field.
                local_metadata["created_at"] = source_timestamp.isoformat()
            add_kwargs: Dict[str, Any] = {
                "messages": messages,
                "user_id": user_id,
                "run_id": run_id,
                "metadata": local_metadata,
            }
            if getattr(self, "add_audit_enabled", False):
                audit_key = "|".join(
                    [
                        str(run_id),
                        str(metadata.get("speaker", "")),
                        str(metadata.get("batch_index", "")),
                        str(metadata.get("timeout_split_path", "")),
                    ]
                )
                attempt_index = self._add_audit_attempts.get(audit_key, 0) + 1
                self._add_audit_attempts[audit_key] = attempt_index
                add_kwargs["_audit_context"] = {
                    "dataset_name": self.config.get("dataset_name"),
                    "run_name": self.config.get("run_name"),
                    "conversation_id": run_id,
                    "speaker": metadata.get("speaker"),
                    "batch_index": metadata.get("batch_index"),
                    "total_batches": metadata.get("total_batches"),
                    "attempt_index": attempt_index,
                    "memory_action_policy": self._local_memory_action_policy,
                    "timeout_split_path": metadata.get("timeout_split_path"),
                    "timeout_split_size": metadata.get("timeout_split_size"),
                    "source_timestamp": (
                        source_timestamp.isoformat()
                        if source_timestamp is not None
                        else None
                    ),
                }
                add_kwargs["_audit_callback"] = self._append_add_audit_record
            return await client.add(
                **add_kwargs,
            )

        timestamp = (
            int(source_timestamp.timestamp()) if source_timestamp is not None else None
        )
        try:
            return await self.client.add(
                messages=messages,
                timestamp=timestamp,
                user_id=user_id,
                run_id=run_id,
                metadata=metadata,
            )
        except TypeError:
            return await self.client.add(
                messages=messages,
                timestamp=timestamp,
                user_id=user_id,
            )

    async def _append_add_audit_record(self, record: Dict[str, Any]) -> None:
        """Persist one completed/failed local Mem0 add attempt."""
        if not getattr(self, "add_audit_enabled", False):
            return
        async with self._add_audit_lock:
            record["audit_written_at"] = datetime.now(timezone.utc).isoformat()
            try:
                await asyncio.to_thread(
                    append_jsonl_fsynced,
                    self._add_audit_path,
                    record,
                    fsync=self.add_audit_fsync,
                )
            except Exception as error:
                raise Mem0AddAuditWriteError(
                    f"Failed to persist Mem0 add audit record to "
                    f"{self._add_audit_path}: {error}"
                ) from error


    async def _call_with_retry(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[Any]],
        max_attempts: Optional[int] = None,
    ) -> Any:
        """
        Retry one Mem0 SDK operation before surfacing the error.

        The retry happens at the failed operation boundary, so a failed batch add
        only repeats that batch, and a failed search only repeats that search call.
        """
        max_attempts = max(
            1,
            int(self.max_retries if max_attempts is None else max_attempts),
        )

        for attempt in range(1, max_attempts + 1):
            try:
                return await operation()
            except Exception as e:
                error_detail = self._format_exception_detail(e)
                if isinstance(e, Mem0AddAuditWriteError):
                    self.console.print(
                        f"   ❌ {error_detail}; aborting without retry to avoid "
                        "replaying an already-committed Mem0 mutation",
                        style="red",
                    )
                    raise
                if (
                    operation_name.lower().startswith("add")
                    and self._is_content_filter_error(e, error_detail)
                ):
                    # Azure/OpenAI-compatible gateways can reject an otherwise
                    # valid conversation with a non-retryable content filter.
                    # The audited call has already persisted the failure record;
                    # returning normally lets the caller checkpoint this batch
                    # as intentionally skipped and continue the benchmark.
                    self.console.print(
                        f"   ⚠️  Mem0 {operation_name} skipped because the "
                        f"upstream provider returned content_filter: {error_detail}",
                        style="yellow",
                    )
                    return None
                if attempt >= max_attempts:
                    self.console.print(
                        f"   ❌ Mem0 {operation_name} failed after {max_attempts} attempts: {error_detail}",
                        style="red",
                    )
                    raise

                wait_seconds = min(2 ** (attempt - 1), 30)
                self.console.print(
                    f"   ⚠️  Mem0 {operation_name} failed "
                    f"(attempt {attempt}/{max_attempts}); retrying in {wait_seconds}s: {error_detail}",
                    style="yellow",
                )
                await asyncio.sleep(wait_seconds)

    @staticmethod
    def _is_content_filter_error(error: Exception, detail: str) -> bool:
        """Return whether an OpenAI-compatible provider rejected a prompt.

        ``content_filter`` is a permanent request rejection, so retrying the
        identical Mem0 add batch only repeats the same failure. Keep this
        deliberately narrow: only the provider error code is recognized.
        """
        return "content_filter" in f"{type(error).__name__}: {detail}".lower()

    async def close(self) -> None:
        clients = list(self._local_shard_clients.values())
        if self.client is not None:
            clients.append(self.client)
        if self._local_shard_client_tasks:
            pending_clients = await asyncio.gather(
                *self._local_shard_client_tasks.values(), return_exceptions=True
            )
            clients.extend(
                client for client in pending_clients if not isinstance(client, BaseException)
            )

        seen = set()
        for client in clients:
            if id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result

    def _format_exception_detail(self, error: Exception) -> str:
        detail = str(error)
        response = getattr(error, "response", None)
        if response is None:
            return detail

        body = getattr(response, "text", "")
        if body:
            detail = f"{detail} | response body: {body[:1000]}"
        return detail

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """
        Ingest conversations with local batch-state tracking for automatic resume.

        Mem0 memory counts are not reliable for resume because extracted memories
        may be capped or deduplicated. The local state file records completed add
        batches, so resume decisions are independent of remote memory limits.
        """
        if not self.add_resume_enabled:
            return await super().add(conversations, **kwargs)

        conversations_to_add = []
        skipped_conversation_ids = []

        self._init_add_resume_state(kwargs.get("output_dir"))

        self.console.print("\n🔎 Checking local Mem0 add resume state...", style="bold cyan")

        for conv in conversations:
            status = await self._get_conversation_add_status(conv)
            complete = status["expected_batches"] > 0 and status["complete"]
            if complete:
                skipped_conversation_ids.append(conv.conversation_id)
                self.console.print(
                    f"   ⏭️  {conv.conversation_id}: complete "
                    f"({status['completed_batches']}/{status['expected_batches']} batches); skipping",
                    style="green",
                )
                continue

            if status["completed_batches"]:
                if self.add_resume_partial:
                    self._validate_partial_resume_state(conv)
                    self.console.print(
                        f"   ↪️  {conv.conversation_id}: partial "
                        f"({status['completed_batches']}/{status['expected_batches']} batches); "
                        "continuing from checkpoint",
                        style="yellow",
                    )
                    conversations_to_add.append(conv)
                    continue

                self.console.print(
                    f"   ♻️  {conv.conversation_id}: partial "
                    f"({status['completed_batches']}/{status['expected_batches']} batches); "
                    "clearing and re-adding",
                    style="yellow",
                )
            else:
                self.console.print(
                    f"   ➕ {conv.conversation_id}: no completed add found; adding",
                    style="dim",
                )

            await self._clear_conversation_add_data(conv)
            await self._reset_conversation_add_state(conv.conversation_id)
            conversations_to_add.append(conv)

        if not conversations_to_add:
            self.console.print(
                "   ✅ All selected conversations are already fully added",
                style="bold green",
            )
            return self._build_add_result(
                skipped_conversation_ids,
                [{"skipped": skipped_conversation_ids}],
                **kwargs,
            )

        try:
            result = await super().add(conversations_to_add, **kwargs)
        finally:
            await self._save_add_resume_state()
        result["conversation_ids"] = skipped_conversation_ids + result.get(
            "conversation_ids", []
        )
        result["skipped_conversation_ids"] = skipped_conversation_ids
        return result

    def _get_user_id_prefix(self) -> str:
        """
        Scope Mem0 storage by run name when available.

        Mem0 search is currently filtered by ``user_id`` only, so encoding the
        run namespace into ``user_id`` keeps add/search results isolated across
        evaluation runs that share the same conversation and speaker IDs.
        """
        return str(self.config.get("run_name") or "")

    def _is_memoryagentbench_mode(self) -> bool:
        return str(self.config.get("dataset_name", "")) == "memoryagentbench"

    def _memoryagentbench_source(self) -> str:
        mab_cfg = self.config.get("memoryagentbench", {}) or {}
        return str(mab_cfg.get("source") or "")

    def _is_memoryagentbench_icl_mode(self) -> bool:
        return self._is_memoryagentbench_mode() and self._memoryagentbench_source().startswith("icl_")

    def _get_memoryagentbench_user_id(self, conversation_id: str) -> str:
        """
        Use the same single-user MemoryAgentBench storage shape as Memos.

        MAB contexts are stored under context_{context_id}_{source}; they are not
        split into speaker-specific user IDs even when speaker names are custom.
        """
        parts = str(conversation_id).rsplit("_", 1)
        context_id = parts[1] if len(parts) == 2 and parts[1].isdigit() else conversation_id
        source = self._sanitize_user_id_component(
            self._memoryagentbench_source() or "memoryagentbench"
        )
        return self._apply_user_id_prefix(f"context_{context_id}_{source}")

    @staticmethod
    def _has_icl_label_mapping(text: str) -> bool:
        return (
            "examples I have learned" in text
            and len(re.findall(r"\blabel\s*:\s*[0-9A-Za-z_-]+\b", text, re.IGNORECASE)) >= 5
        )

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        ignored = {
            "label",
            "search",
            "archival",
            "memory",
            "provided",
            "mapping",
            "context",
            "numerical",
            "assign",
            "only",
            "output",
            "question",
        }
        return {
            token
            for token in re.findall(r"[A-Za-z0-9]+", text.lower())
            if len(token) > 2 and token not in ignored
        }

    def _extract_icl_label_chunks(self, conversation: Optional[Conversation]) -> List[str]:
        if conversation is None:
            return []

        chunks = []
        for msg in conversation.messages:
            content = str(msg.content or "")
            if self._has_icl_label_mapping(content):
                chunks.append(content)
        return chunks

    def _select_icl_label_chunks(
        self, query: str, conversation: Optional[Conversation], limit: int = 5
    ) -> List[str]:
        query_tokens = self._content_tokens(query)
        scored_chunks = []
        for chunk in self._extract_icl_label_chunks(conversation):
            chunk_tokens = self._content_tokens(chunk)
            scored_chunks.append((len(query_tokens & chunk_tokens), chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _score, chunk in scored_chunks[:limit]]

    def _build_icl_raw_mapping_context(
        self, query: str, conversation: Optional[Conversation]
    ) -> str:
        chunks = self._select_icl_label_chunks(query, conversation)
        if not chunks:
            return ""

        formatted_chunks = [
            "[EXACT TRAINING EXAMPLES WITH GOLD LABELS "
            f"{idx}]\nUse these verbatim utterance-to-label mappings to answer "
            f"the query. Output only the matched numeric label.\n{chunk}"
            for idx, chunk in enumerate(chunks, start=1)
        ]
        return "\n\n".join(formatted_chunks)

    def _extract_conversation_info(
        self,
        conversation: Optional[Conversation] = None,
        conversation_id: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if self._is_memoryagentbench_mode():
            if conversation is None:
                conversation = kwargs.get("conversation")
            if conversation is not None:
                conversation_id = conversation.conversation_id
                speaker_a = conversation.metadata.get("speaker_a", "user")
                speaker_b = conversation.metadata.get("speaker_b", "assistant")
            else:
                conversation_id = conversation_id or "unknown"
                speaker_a = "user"
                speaker_b = "assistant"

            user_id = self._get_memoryagentbench_user_id(conversation_id)
            return {
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "speaker_a_user_id": user_id,
                "speaker_b_user_id": user_id,
                "need_dual_perspective": False,
            }

        return super()._extract_conversation_info(
            conversation=conversation,
            conversation_id=conversation_id,
            **kwargs,
        )

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._is_memoryagentbench_mode():
            return super()._conversation_to_messages(
                conversation,
                format_type=format_type,
                perspective=perspective,
            )

        messages: List[Dict[str, Any]] = []
        for msg in conversation.messages:
            message = {"role": "user", "content": msg.content}
            if msg.timestamp:
                from common_utils.datetime_utils import to_iso_format

                message["chat_time"] = to_iso_format(msg.timestamp)
            messages.append(message)
        return messages

    def _build_memoryagentbench_messages(
        self, source_message: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        assistant_ack = "I'll make sure to add the content into the memory."
        system_message = MAB_SYSTEM_MESSAGE
        if self._is_memoryagentbench_icl_mode():
            assistant_ack = (
                "I'll preserve each example and its exact numerical label mapping "
                "verbatim in memory."
            )
            system_message += (
                "\nFor in-context learning examples, preserve every utterance and "
                "its exact `label: N` value verbatim."
            )

        triple = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": source_message.get("content", "")},
            {"role": "assistant", "content": assistant_ack},
        ]
        chat_time = source_message.get("chat_time")
        if chat_time:
            for item in triple:
                item["chat_time"] = chat_time
        return triple

    def _get_batch_count(self, messages: List[Dict[str, Any]]) -> int:
        # `_add_user_messages` drops blank content before batching. Resume must
        # use the identical population or a trailing blank message turns a
        # complete N/N checkpoint into a phantom extra batch on every restart.
        addable_count = sum(
            1 for message in messages if str(message.get("content", "")).strip()
        )
        if not addable_count:
            return 0
        if self._is_memoryagentbench_mode():
            return addable_count
        return math.ceil(addable_count / self.batch_size)

    @staticmethod
    def _is_timeout_error(error: BaseException) -> bool:
        """Recognize OpenAI/httpx timeout errors through wrapped causes."""
        current: Optional[BaseException] = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if type(current).__name__ in {
                "APITimeoutError",
                "ReadTimeout",
                "TimeoutException",
            }:
                return True
            current = current.__cause__ or current.__context__
        return "timed out" in str(error).lower()

    async def _add_with_timeout_split(
        self,
        operation_name: str,
        batch_items: List[Any],
        operation: Callable[[List[Any], str], Awaitable[Any]],
        split_path: str = "",
        completed_split_paths: Optional[set[str]] = None,
        record_completed_split: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Retry an add batch, recursively splitting deterministic timeouts."""
        completed_split_paths = completed_split_paths or set()
        if split_path and split_path in completed_split_paths:
            return

        descendant_prefixes = (
            (f"{split_path}L", f"{split_path}R")
            if split_path
            else ("L", "R")
        )
        has_completed_descendant = any(
            any(path.startswith(prefix) for prefix in descendant_prefixes)
            for path in completed_split_paths
        )
        suffix = f" split={split_path}" if split_path else ""
        if not has_completed_descendant:
            try:
                await self._call_with_retry(
                    f"{operation_name}{suffix}",
                    lambda: operation(batch_items, split_path),
                    max_attempts=(
                        getattr(self, "add_timeout_split_max_attempts", 1)
                        if getattr(self, "add_timeout_split_enabled", False)
                        and len(batch_items) > 1
                        else None
                    ),
                )
                if split_path and record_completed_split is not None:
                    await record_completed_split(split_path)
                return
            except Exception as error:
                if (
                    not getattr(self, "add_timeout_split_enabled", False)
                    or len(batch_items) <= 1
                    or not self._is_timeout_error(error)
                ):
                    raise

        midpoint = len(batch_items) // 2
        self.console.print(
            f"   ⚠️  Mem0 {operation_name}{suffix} timed out repeatedly; "
            f"splitting {len(batch_items)} items into "
            f"{midpoint}+{len(batch_items) - midpoint}",
            style="yellow",
        )
        await self._add_with_timeout_split(
            operation_name,
            batch_items[:midpoint],
            operation,
            f"{split_path}L" if split_path else "L",
            completed_split_paths,
            record_completed_split,
        )
        await self._add_with_timeout_split(
            operation_name,
            batch_items[midpoint:],
            operation,
            f"{split_path}R" if split_path else "R",
            completed_split_paths,
            record_completed_split,
        )

    def _init_add_resume_state(self, output_dir: Path = None) -> None:
        output_path = Path(output_dir) if output_dir else self.output_dir
        self._add_resume_state_path = output_path / self.add_resume_state_file
        self._add_resume_state_path.parent.mkdir(parents=True, exist_ok=True)

        if self._add_resume_state_path.exists():
            try:
                self._add_resume_state = json.loads(
                    self._add_resume_state_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid Mem0 add resume state file: {self._add_resume_state_path}"
                ) from e
            expected_layout = self._add_resume_storage_layout()
            existing_layout = self._add_resume_state.get("storage_layout", "shared")
            if existing_layout != expected_layout:
                raise ValueError(
                    "Mem0 add resume state storage layout mismatch: "
                    f"checkpoint uses {existing_layout!r}, current config uses "
                    f"{expected_layout!r}. Start a new run with a unique run-name."
                )
        else:
            self._add_resume_state = {
                "version": 2,
                "system": "mem0",
                "run_name": self.config.get("run_name", ""),
                "batch_size": self.batch_size,
                "storage_layout": self._add_resume_storage_layout(),
                "conversations": {},
            }
        self._add_resume_dirty_batches = 0

    def _add_resume_storage_layout(self) -> str:
        if self._uses_local_conversation_shards():
            return "conversation_shards"
        return "shared"

    def _get_speaker_add_state(self, conv_id: str, speaker: str) -> Dict[str, Any]:
        conversations = self._add_resume_state.setdefault("conversations", {})
        conv_state = conversations.setdefault(conv_id, {"speakers": {}})
        return conv_state.setdefault(
            "speakers",
            {},
        ).setdefault(
            speaker,
            {
                "completed_batches": [],
                "total_batches": None,
                "user_id": None,
                "updated_at": None,
            },
        )

    def _get_completed_batch_indices(self, conv_id: str, speaker: str) -> set:
        if not self._add_resume_state:
            return set()
        conv_state = self._add_resume_state.get("conversations", {}).get(conv_id, {})
        speaker_state = conv_state.get("speakers", {}).get(speaker, {})
        completed = speaker_state.get("completed_batches", [])
        result = set()
        for batch_index in completed:
            try:
                result.add(int(batch_index))
            except (TypeError, ValueError):
                continue
        return result

    def _get_completed_timeout_split_paths(
        self, conv_id: str, speaker: str, batch_index: int
    ) -> set[str]:
        if not self._add_resume_state:
            return set()
        conv_state = self._add_resume_state.get("conversations", {}).get(conv_id, {})
        speaker_state = conv_state.get("speakers", {}).get(speaker, {})
        split_state = speaker_state.get("completed_timeout_splits", {})
        paths = split_state.get(str(batch_index), [])
        return {str(path) for path in paths if path}

    async def _save_add_resume_state(self) -> None:
        if (
            not getattr(self, "add_resume_enabled", False)
            or not getattr(self, "_add_resume_state_path", None)
            or getattr(self, "_add_resume_state", None) is None
        ):
            return
        async with self._add_resume_state_lock:
            if self._add_resume_dirty_batches:
                await self._write_add_resume_state_locked()
                self._add_resume_dirty_batches = 0

    async def _write_add_resume_state_locked(self) -> None:
        if not self._add_resume_state_path or self._add_resume_state is None:
            return

        text = json.dumps(self._add_resume_state, ensure_ascii=False, indent=2)
        task_id = id(asyncio.current_task())
        tmp_path = self._add_resume_state_path.with_name(
            f"{self._add_resume_state_path.name}.{os.getpid()}.{task_id}.tmp"
        )

        try:
            tmp_path.write_text(text, encoding="utf-8")
            for attempt in range(5):
                try:
                    tmp_path.replace(self._add_resume_state_path)
                    return
                except PermissionError:
                    if attempt >= 4:
                        raise
                    await asyncio.sleep(0.2 * (attempt + 1))
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    async def _reset_conversation_add_state(self, conv_id: str) -> None:
        if self._add_resume_state is None:
            return
        async with self._add_resume_state_lock:
            self._add_resume_state.setdefault("conversations", {}).pop(conv_id, None)
            await self._write_add_resume_state_locked()
            self._add_resume_dirty_batches = 0

    async def _record_completed_batch(
        self,
        conv: Conversation,
        user_id: str,
        speaker: str,
        batch_index: int,
        total_batches: int,
        batch_start: int,
        batch_size: int,
    ) -> None:
        if not self.add_resume_enabled or self._add_resume_state is None:
            return
        async with self._add_resume_state_lock:
            speaker_state = self._get_speaker_add_state(conv.conversation_id, speaker)
            completed = self._get_completed_batch_indices(conv.conversation_id, speaker)
            completed.add(batch_index)
            speaker_state.update(
                {
                    "completed_batches": sorted(completed),
                    "total_batches": total_batches,
                    "user_id": user_id,
                    "batch_start_last": batch_start,
                    "batch_size_last": batch_size,
                    "message_count": len(conv.messages),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            speaker_state.setdefault("completed_timeout_splits", {}).pop(
                str(batch_index), None
            )
            self._add_resume_dirty_batches += 1
            if (
                self._add_resume_dirty_batches
                >= self.add_resume_flush_every_batches
            ):
                await self._write_add_resume_state_locked()
                self._add_resume_dirty_batches = 0

    async def _record_completed_timeout_split(
        self,
        conv_id: str,
        speaker: str,
        batch_index: int,
        split_path: str,
    ) -> None:
        """Durably checkpoint one successful timeout-fallback sub-batch."""
        if not self.add_resume_enabled or self._add_resume_state is None:
            return
        async with self._add_resume_state_lock:
            speaker_state = self._get_speaker_add_state(conv_id, speaker)
            split_state = speaker_state.setdefault("completed_timeout_splits", {})
            completed = {str(path) for path in split_state.get(str(batch_index), [])}
            completed.add(split_path)
            split_state[str(batch_index)] = sorted(completed)
            speaker_state["updated_at"] = datetime.now(timezone.utc).isoformat()
            # A successful sub-batch has already mutated Mem0. Persist it
            # immediately so a crash cannot replay that mutation on restart.
            await self._write_add_resume_state_locked()
            self._add_resume_dirty_batches = 0

    async def _get_conversation_add_status(self, conv: Conversation) -> Dict[str, Any]:
        speakers = self._get_conversation_speakers(conv)
        expected_batches = 0
        completed_batches = 0
        complete = True

        for speaker in speakers:
            messages = self._conversation_to_messages(
                conv,
                format_type=self._get_format_type(),
                perspective=speaker if len(speakers) > 1 else None,
            )
            expected = self._get_batch_count(messages)
            expected_batches += expected

            completed = self._get_completed_batch_indices(conv.conversation_id, speaker)
            matching_completed = {idx for idx in completed if 0 <= idx < expected}
            completed_batches += len(matching_completed)
            complete = complete and len(matching_completed) >= expected

        return {
            "complete": complete,
            "expected_batches": expected_batches,
            "completed_batches": completed_batches,
        }

    def _validate_partial_resume_state(self, conv: Conversation) -> None:
        speakers = self._get_conversation_speakers(conv)
        for speaker in speakers:
            messages = self._conversation_to_messages(
                conv,
                format_type=self._get_format_type(),
                perspective=speaker if len(speakers) > 1 else None,
            )
            completed = self._get_completed_batch_indices(
                conv.conversation_id, speaker
            )
            if completed != set(range(len(completed))):
                raise ValueError(
                    "Cannot resume Mem0 partial conversation with non-contiguous "
                    f"completed batches: {conv.conversation_id}/{speaker} has "
                    f"{sorted(completed)}"
                )

            expected_batches = self._get_batch_count(messages)
            if len(completed) > expected_batches:
                raise ValueError(
                    "Cannot resume Mem0 partial conversation because checkpoint "
                    f"contains {len(completed)} batches but current input has "
                    f"{expected_batches}: {conv.conversation_id}/{speaker}"
                )

        if self._uses_local_conversation_shards():
            shard_dir = self._local_shard_dir(conv.conversation_id)
            if not (shard_dir / "qdrant").exists() or not (
                shard_dir / "history.db"
            ).exists():
                raise ValueError(
                    "Cannot resume Mem0 partial conversation because its local "
                    f"shard is incomplete: {shard_dir}"
                )

    def _get_conversation_speakers(self, conv: Conversation) -> List[str]:
        if self._is_memoryagentbench_mode():
            return ["speaker_a"]
        speaker_a = conv.metadata.get("speaker_a", "")
        speaker_b = conv.metadata.get("speaker_b", "")
        if self._need_dual_perspective(speaker_a, speaker_b):
            return ["speaker_a", "speaker_b"]
        return ["speaker_a"]

    async def _clear_conversation_add_data(self, conv: Conversation) -> None:
        client = await self._client_for_conversation(conv.conversation_id)
        for speaker in self._get_conversation_speakers(conv):
            if self._is_memoryagentbench_mode():
                user_id = self._get_memoryagentbench_user_id(conv.conversation_id)
            else:
                user_id = self._extract_user_id(conv, speaker=speaker)
            try:
                await self._call_with_retry(
                    f"delete_all user_id={user_id}",
                    lambda user_id=user_id: client.delete_all(user_id=user_id),
                )
            except Exception as e:
                self.console.print(
                    f"   ⚠️  Failed to clear {user_id}: {e}",
                    style="yellow",
                )
                raise

    def _convert_timestamp_to_display_timezone(self, timestamp_str: str) -> str:
        """
        Convert mem0's timestamp to display timezone.

        Default behavior (if display_timezone_offset not set):
        - Convert to system local timezone (symmetric with add stage where naive datetime
          is treated as local timezone by Python's .timestamp() method)

        Optional behavior (if display_timezone_offset is set):
        - Convert to specified timezone (e.g., UTC for explicit UTC handling)

        Args:
            timestamp_str: ISO format timestamp string with timezone (e.g., "2023-05-07T22:56:00-07:00")

        Returns:
            Formatted timestamp string in display timezone or original if conversion fails
        """
        if not timestamp_str:
            return timestamp_str

        try:
            # Parse ISO format timestamp (with timezone)
            dt = datetime.fromisoformat(timestamp_str)

            dt_display = dt.astimezone(None)

            # Format as readable string (YYYY-MM-DD HH:MM:SS)
            return dt_display.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            # If conversion fails, return original string
            self.console.print(
                f"⚠️  Failed to convert timestamp '{timestamp_str}': {e}", style="yellow"
            )
            return timestamp_str

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        Preparation stage: update project configuration and clean existing data.

        Args:
            conversations: Standard format conversation list
            **kwargs: Extra parameters
        """
        # Update project with custom instructions (if set)
        if self._custom_instructions and self.client_mode == "cloud":
            try:
                await self._call_with_retry(
                    "update_project",
                    lambda: self.client.update_project(
                        custom_instructions=self._custom_instructions
                    ),
                )
                self.console.print("   ✅ Custom instructions set", style="green")
            except Exception as e:
                self.console.print(
                    f"   ⚠️  Failed to set custom instructions: {e}", style="yellow"
                )

        # Check if need to clean existing data
        clean_before_add = self.config.get("clean_before_add", False)

        if not clean_before_add:
            self.console.print(
                "   ⏭️  Skipping data cleanup (clean_before_add=false)", style="dim"
            )
            return

        self.console.print(f"\n{'='*60}", style="bold yellow")
        self.console.print("Preparation: Cleaning existing data", style="bold yellow")
        self.console.print(f"{'='*60}", style="bold yellow")

        # Collect all user_ids to clean
        user_ids_to_clean = set()

        for conv in conversations:
            for speaker in self._get_conversation_speakers(conv):
                if self._is_memoryagentbench_mode():
                    user_ids_to_clean.add(
                        (
                            conv.conversation_id,
                            self._get_memoryagentbench_user_id(conv.conversation_id),
                        )
                    )
                else:
                    user_ids_to_clean.add(
                        (
                            conv.conversation_id,
                            self._extract_user_id(conv, speaker=speaker),
                        )
                    )

        # Clean all user data
        self.console.print(
            f"\n🗑️  Cleaning data for {len(user_ids_to_clean)} user(s)...",
            style="yellow",
        )

        cleaned_count = 0
        failed_count = 0

        for conversation_id, user_id in user_ids_to_clean:
            try:
                # Use async client for delete operation
                client = await self._client_for_conversation(conversation_id)
                await self._call_with_retry(
                    f"delete_all user_id={user_id}",
                    lambda user_id=user_id, client=client: client.delete_all(
                        user_id=user_id
                    ),
                )
                cleaned_count += 1
                self.console.print(f"   ✅ Cleaned: {user_id}", style="green")
            except Exception as e:
                failed_count += 1
                self.console.print(
                    f"   ⚠️  Failed to clean {user_id}: {e}", style="yellow"
                )

        self.console.print(
            f"\n✅ Cleanup completed: {cleaned_count} succeeded, {failed_count} failed",
            style="bold green",
        )

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Mem0.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id
        if self._is_memoryagentbench_mode():
            user_id = self._get_memoryagentbench_user_id(conv.conversation_id)
        else:
            user_id = self._extract_user_id(conv, speaker=speaker)

        message_pairs = [
            (source_msg, msg)
            for source_msg, msg in zip(conv.messages, messages)
            if str(msg.get("content", "")).strip()
        ]
        skipped_empty_count = len(messages) - len(message_pairs)
        messages = [msg for _, msg in message_pairs]

        # Handle content truncation (Mem0 specific)
        truncated_count = 0
        for msg in messages:
            if len(msg["content"]) > self.max_content_length:
                msg["content"] = msg["content"][: self.max_content_length]
                truncated_count += 1

        # Log info
        sender_name = (
            "MemoryAgentBench"
            if self._is_memoryagentbench_mode()
            else conv.metadata.get(speaker, speaker)
        )
        is_fake_timestamp = (
            conv.messages[0].metadata.get("is_fake_timestamp", False)
            if conv.messages
            else False
        )

        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )
        if skipped_empty_count > 0:
            self.console.print(
                f"   ⚠️  Skipped {skipped_empty_count} empty messages",
                style="yellow",
            )
        if is_fake_timestamp:
            self.console.print("   ⚠️  Using fake timestamp", style="yellow")
        if truncated_count > 0:
            self.console.print(
                f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)",
                style="yellow",
            )

        # Add messages in batches with retry.
        # In MAB mode, each original context chunk is wrapped as a dedicated
        # system/user/assistant triple to mirror Memos and the benchmark shape.
        total_batches = self._get_batch_count(messages)
        completed_batches = self._get_completed_batch_indices(
            conv.conversation_id, speaker
        )
        if self._is_memoryagentbench_mode():
            for batch_index, (source_msg, msg) in enumerate(message_pairs):
                if batch_index in completed_batches:
                    self._advance_add_progress(kwargs, 1)
                    continue
                mab_messages = self._build_memoryagentbench_messages(msg)

                async def add_mab_message() -> Any:
                    metadata = {
                        "conversation_id": conv.conversation_id,
                        "speaker": "memoryagentbench",
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "memoryagentbench_source": self._memoryagentbench_source(),
                        "memoryagentbench_icl": self._is_memoryagentbench_icl_mode(),
                    }
                    return await self._add_messages_to_client(
                        messages=mab_messages,
                        source_timestamp=source_msg.timestamp,
                        user_id=user_id,
                        run_id=conv.conversation_id,
                        metadata=metadata,
                    )

                await self._call_with_retry(
                    (
                        f"add MAB chunk {batch_index + 1}/{total_batches} "
                        f"user_id={user_id}"
                    ),
                    add_mab_message,
                )
                await self._record_completed_batch(
                    conv=conv,
                    user_id=user_id,
                    speaker=speaker,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    batch_start=batch_index,
                    batch_size=1,
                )
                self._advance_add_progress(kwargs, 1)
                if self.add_interval > 0:
                    await asyncio.sleep(self.add_interval)
            await self._save_add_resume_state()
            return None

        for i in range(0, len(messages), self.batch_size):
            batch_pairs = message_pairs[i : i + self.batch_size]
            batch_messages = [msg for _, msg in batch_pairs]
            batch_index = i // self.batch_size
            if batch_index in completed_batches:
                self._advance_add_progress(kwargs, len(batch_messages))
                continue

            async def add_batch_items(
                pairs: List[Any], split_path: str
            ) -> Any:
                messages_to_add = [msg for _, msg in pairs]
                source_timestamp = pairs[0][0].timestamp if pairs else None
                metadata = {
                    "conversation_id": conv.conversation_id,
                    "speaker": speaker,
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                }
                if split_path:
                    metadata.update(
                        {
                            "timeout_split_path": split_path,
                            "timeout_split_size": len(messages_to_add),
                        }
                    )
                return await self._add_messages_to_client(
                    messages=messages_to_add,
                    source_timestamp=source_timestamp,
                    user_id=user_id,
                    run_id=conv.conversation_id,
                    metadata=metadata,
                )

            await self._add_with_timeout_split(
                f"add batch {batch_index + 1}/{total_batches} "
                f"for {sender_name} user_id={user_id}",
                batch_pairs,
                add_batch_items,
                completed_split_paths=self._get_completed_timeout_split_paths(
                    conv.conversation_id, speaker, batch_index
                ),
                record_completed_split=lambda split_path: (
                    self._record_completed_timeout_split(
                        conv.conversation_id,
                        speaker,
                        batch_index,
                        split_path,
                    )
                ),
            )
            await self._record_completed_batch(
                conv=conv,
                user_id=user_id,
                speaker=speaker,
                batch_index=batch_index,
                total_batches=total_batches,
                batch_start=i,
                batch_size=len(batch_messages),
            )
            self._advance_add_progress(kwargs, len(batch_messages))
            # Wait between add requests to avoid rate limits
            if self.add_interval > 0:
                await asyncio.sleep(self.add_interval)

        await self._save_add_resume_state()
        return None

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Mem0-specific with timezone conversion).

        Calls Mem0 search API and converts results to standard format,
        applying timezone conversion to timestamps.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Mem0)
            user_id: User ID to search for
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with timezone-converted timestamps
        """
        # Add interval before search to avoid rate limiting (429 errors)
        if self.search_interval > 0:
            await asyncio.sleep(self.search_interval)

        question_date = kwargs.get("question_date")
        fetch_top_k = self._temporal_fetch_top_k(top_k, question_date)

        async def search_user() -> Any:
            client = await self._client_for_conversation(conversation_id)
            search_kwargs = dict(
                query=query,
                # Mem0 OSS 1.x exposes this as ``limit`` (not ``top_k``).
                # ``limit`` is also accepted by later local SDK releases, so
                # keep the adapter on the stable local-SDK interface.
                limit=fetch_top_k,
                # Mem0 1.x validates the identity from a top-level argument;
                # putting it only in ``filters`` fails before the vector search.
                # This is also the documented local-SDK interface in later
                # releases.
                user_id=user_id,
            )
            if self.mem0_search_rerank:
                search_kwargs["rerank"] = True
            return await client.search(**search_kwargs)

        raw_results = await self._call_with_retry(
            f"search user_id={user_id}",
            search_user,
            max_attempts=self.search_operation_max_attempts,
        )

        # Full raw results can be very large and may block a detached VS Code
        # terminal. Keep them opt-in; the normal progress bar is sufficient.
        if self.debug_search_results:
            self.console.print("\n[DEBUG] Mem0 Search Results:", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  User ID: {user_id}", style="dim")
            self.console.print(
                f"  Results: {json.dumps(raw_results, indent=2, ensure_ascii=False)}",
                style="dim",
            )

        # Convert to standard format with timezone conversion
        results = []
        for memory in raw_results.get("results", []):
            # Apply timezone conversion to timestamp
            created_at_original = memory.get("created_at", "")
            created_at_display = self._convert_timestamp_to_display_timezone(
                created_at_original
            )

            results.append(
                {
                    "content": f"{created_at_display}: {memory['memory']}",  # Add timestamp prefix
                    "score": memory.get("score", 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "id": memory.get("id", ""),
                        "created_at": created_at_original,
                        "created_at_display": created_at_display,
                        "updated_at": memory.get("updated_at", ""),
                        "metadata": memory.get("metadata", {}),
                        "memory": memory.get("memory", ""),
                        "user_id": memory.get("user_id", ""),
                        "raw": memory,
                    },
                }
            )

        results, temporal_filter_metadata = self._filter_results_for_question_date(
            results,
            question_date,
            requested_top_k=top_k,
            fetch_top_k=fetch_top_k,
        )
        self._remember_temporal_filter_metadata(
            conversation_id, user_id, query, temporal_filter_metadata
        )
        return results

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for single perspective (Mem0: simple metadata).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult (no formatted_context, uses fallback)
        """
        formatted_context = ""
        if self._is_memoryagentbench_icl_mode():
            raw_mapping_context = self._build_icl_raw_mapping_context(
                query, kwargs.get("conversation")
            )
            retrieved_context = "\n\n".join(
                f"{idx}. {result['content']}"
                for idx, result in enumerate(results[:top_k], start=1)
                if result.get("content")
            )
            context_parts = [part for part in [raw_mapping_context, retrieved_context] if part]
            formatted_context = "\n\n".join(context_parts)
        temporal_filter_metadata = self._get_temporal_filter_metadata(
            conversation_id, user_id, query
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [user_id],
                "formatted_context": formatted_context,
                "icl_raw_mapping_chunks": bool(formatted_context),
                **temporal_filter_metadata,
            },
        )

    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        """
        Build SearchResult for dual perspective (Mem0: use template).

        Formats memories using the default template for dual-speaker scenarios.

        Args:
            query: Query text
            conversation_id: Conversation ID
            all_results: Merged results (for fallback)
            results_a: Speaker A's search results
            results_b: Speaker B's search results
            speaker_a: Speaker A name
            speaker_b: Speaker B name
            speaker_a_user_id: Speaker A user ID
            speaker_b_user_id: Speaker B user ID
            top_k: Number of results per user
            **kwargs: Additional parameters

        Returns:
            SearchResult with formatted_context
        """
        # Extract content from results (already includes timezone-converted timestamps)
        speaker_a_memories_text = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories_text = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )
        temporal_filter_metadata = self._merge_temporal_filter_metadata(
            self._get_temporal_filter_metadata(conversation_id, speaker_a_user_id, query),
            self._get_temporal_filter_metadata(conversation_id, speaker_b_user_id, query),
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": formatted_context,
                "speaker_a_memories_count": len(results_a),
                "speaker_b_memories_count": len(results_b),
                **temporal_filter_metadata,
            },
        )

    def _get_answer_prompt(self) -> str:
        """Load the configured answer prompt (RAG CoT by default)."""
        prompt_key = self.config.get("answer", {}).get(
            "prompt_key", "answer_prompt_cot"
        )
        prompts = self._prompts["online_api"]["default"]
        if prompt_key not in prompts:
            raise KeyError(
                f"Mem0 answer prompt key '{prompt_key}' not found in prompts.yaml"
            )
        return prompts[prompt_key]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        display_name = "Mem0 Local" if self.client_mode == "local" else "Mem0"
        description = (
            "Mem0 - Local Python SDK"
            if self.client_mode == "local"
            else "Mem0 - Personalized AI Memory Layer"
        )
        return {
            "name": display_name,
            "type": "online_api",
            "description": description,
            "adapter": "Mem0Adapter",
        }
