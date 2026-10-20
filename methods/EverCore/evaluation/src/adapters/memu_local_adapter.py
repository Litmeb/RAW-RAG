"""Local memU v1.5.1 adapter for the evaluation framework.

memU's self-hosted Python SDK exposes an async ``MemUService`` rather than the
HTTP API used by :mod:`memu_adapter`.  This adapter keeps one SQLite-backed
service per evaluation run, scopes every conversation with ``user_id``, and
converts memU's structured retrieval response to EverCore's SearchResult.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import types
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.memu_gpu_index import GpuItemIndexCache
from evaluation.src.adapters.memu_xml_recovery import parse_memory_type_response_xml
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


def _gpu_cosine_topk(query_vec, corpus, k: int):
    """Compute cosine scores on CUDA and return MemU's ``(id, score)`` pairs."""
    import torch

    entries = [(item_id, vector) for item_id, vector in corpus if vector is not None]
    if not entries:
        return []
    query = torch.as_tensor(query_vec, dtype=torch.float32, device="cuda")
    matrix = torch.as_tensor(
        [vector for _, vector in entries], dtype=torch.float32, device="cuda"
    )
    scores = (matrix @ query) / (
        torch.linalg.vector_norm(matrix, dim=1)
        * torch.linalg.vector_norm(query)
        + 1e-9
    )
    actual_k = min(max(0, int(k)), len(entries))
    if actual_k == 0:
        return []
    values, indices = torch.topk(scores, k=actual_k)
    return [
        (entries[int(index)][0], float(value))
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    ]


class _BoundedItemPool(dict):
    """A truthy empty mapping prevents memU from falling back to list_items()."""

    def __bool__(self) -> bool:
        return True


class _EmbeddingBatcher:
    """Coalesce concurrent singleton embedding calls into one request.

    memU's retrieve workflow asks for one query vector at a time.  The
    workflow itself is still run independently for every query; this helper
    only combines concurrent ``embed([query])`` calls at the client boundary.
    Non-singleton calls (for example indexing several memory items) are passed
    through unchanged.
    """

    def __init__(self, embed, max_batch_size: int):
        self._embed = embed
        self._max_batch_size = max(1, int(max_batch_size))
        self._pending: list[tuple[str, asyncio.Future]] = []
        self._flush_task: asyncio.Task | None = None

    async def embed(self, inputs: list[str]):
        if len(inputs) != 1 or self._max_batch_size == 1:
            return await self._embed(inputs)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append((inputs[0], future))
        if len(self._pending) >= self._max_batch_size:
            if self._flush_task and not self._flush_task.done():
                self._flush_task.cancel()
            self._flush_task = asyncio.create_task(self._flush())
        elif self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_soon())
        return await future

    async def _flush_soon(self) -> None:
        await asyncio.sleep(0)
        await self._flush()

    async def _flush(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        try:
            for start in range(0, len(pending), self._max_batch_size):
                chunk = pending[start : start + self._max_batch_size]
                response = await self._embed([text for text, _ in chunk])
                if isinstance(response, tuple) and len(response) == 2:
                    vectors, raw_response = response

                    def response_factory(values):
                        return values, raw_response

                else:
                    vectors = response

                    def response_factory(values):
                        return values

                if len(vectors) != len(chunk):
                    raise RuntimeError(
                        "Embedding client returned a different number of vectors "
                        f"({len(vectors)}) than inputs ({len(chunk)})"
                    )
                for vector, (_, future) in zip(vectors, chunk):
                    if not future.done():
                        future.set_result(response_factory([vector]))
        except Exception as exc:
            for _, future in pending:
                if not future.done():
                    future.set_exception(exc)


@register_adapter("memu_local")
class MemuLocalAdapter(OnlineAPIAdapter):
    """Use memU's local SQLite database and OpenAI-compatible LLM clients."""

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert memU SDK payloads to values that can be stored in JSON.

        memU retrieval items may contain ``datetime`` values (for example in
        their raw metadata).  Search results are checkpointed as JSON, so keep
        those timestamps as ISO-8601 strings at the adapter boundary.
        """
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {
                str(key): MemuLocalAdapter._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [MemuLocalAdapter._json_safe(item) for item in value]
        return value

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        try:
            from memu import MemUService
        except ImportError as exc:
            raise ImportError(
                "memu-py==1.5.1 is required for memu_local. "
                "Install it in the Python 3.13 memu-py313 environment."
            ) from exc

        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.console = Console()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resource_dir = self.output_dir / "memu_resources"
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        self.database_path = self.output_dir / "memu.sqlite"
        self._service_lock = asyncio.Lock()
        self._add_resume_lock = asyncio.Lock()
        self._resource_paths: Dict[str, Path] = {}
        self._gpu_similarity_enabled = bool(
            (config.get("search", {}) or {}).get("gpu_similarity", True)
        )
        self._gpu_item_indexes: GpuItemIndexCache | None = None

        llm = config.get("memu", {}).get("llm", config.get("llm", {}))
        database = config.get("memu", {}).get("database", {})
        database_provider = str(database.get("provider", "inmemory")).lower()
        dsn = database.get("dsn") or f"sqlite:///{self.database_path.as_posix()}"

        resume_config = config.get("add_resume", {}) or {}
        self.add_resume_enabled = bool(resume_config.get("enabled", True))
        self.add_resume_state_path = self.output_dir / str(
            resume_config.get("state_file", "memu_local_add_resume.json")
        )
        self._add_resume_state: dict[str, Any] = self._load_add_resume_state()

        if self.add_resume_enabled and database_provider not in {"sqlite", "postgres"}:
            raise ValueError(
                "memu_local Add resume requires a persistent SQLite or PostgreSQL database. "
                "Set memu.database.provider to 'sqlite' or 'postgres', or disable "
                "add_resume.enabled. The default in-memory backend cannot be "
                "reconstructed after a process restart."
            )

        memorize_config = dict(config.get("memu", {}).get("memorize", {}) or {})
        self._category_summary_updates_enabled = bool(
            memorize_config.pop("update_category_summaries", False)
        )

        self.service = MemUService(
            llm_profiles={"default": llm},
            database_config={
                "metadata_store": {
                    "provider": database_provider,
                    "dsn": dsn,
                    "ddl_mode": database.get("ddl_mode", "create"),
                },
                **(
                    {"vector_index": database["vector_index"]}
                    if database.get("vector_index")
                    else {}
                ),
            },
            blob_config={
                "provider": "local",
                "resources_dir": str(self.resource_dir / "blob"),
            },
            memorize_config=memorize_config,
            retrieve_config=config.get("memu", {}).get("retrieve", {}),
        )
        self._install_xml_parser_recovery()
        self._configure_category_summary_updates()
        self._configure_gpu_similarity()
        self._configure_item_similarity(database_provider, dsn)
        self._install_bounded_item_recall()
        self.num_workers = max(1, int(config.get("num_workers", 1)))
        print("   memU local SDK: memu-py==1.5.1")
        print(f"   Database provider: {database_provider}")
        if database_provider == "sqlite":
            print(f"   SQLite database: {self.database_path}")

    def _install_xml_parser_recovery(self) -> None:
        """Replace memU 1.5.1's all-or-nothing strict XML parser."""

        def parse(_service, raw: str) -> list[dict[str, Any]]:
            return parse_memory_type_response_xml(raw)

        self.service._parse_memory_type_response_xml = types.MethodType(
            parse, self.service
        )

    def _configure_category_summary_updates(self) -> None:
        """Optionally remove category-summary LLM calls from Add."""
        if self._category_summary_updates_enabled:
            print("   Add category summary updates: enabled", flush=True)
            return
        if self.service.memorize_config.enable_item_references:
            raise ValueError(
                "memu.memorize.enable_item_references requires "
                "update_category_summaries=true"
            )

        async def skip_summary_updates(service, state, step_context):
            del service, step_context
            state["categories"] = []
            return state

        self.service._memorize_persist_and_index = types.MethodType(
            skip_summary_updates,
            self.service,
        )
        replacement = next(
            step
            for step in self.service._build_memorize_workflow()
            if step.step_id == "persist_index"
        )
        self.service._pipelines.replace_step(
            "memorize",
            "persist_index",
            replacement,
        )
        print("   Add category summary updates: disabled", flush=True)

    def _configure_gpu_similarity(self) -> None:
        if not self._gpu_similarity_enabled:
            print("   Auxiliary similarity backend: CPU (configured)")
            print("   Item similarity backend: database-native vector search")
            return
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            import memu.app.retrieve as retrieve_module

            # Category and resource retrieval call this module-level helper.
            # Item retrieval deliberately remains on the repository's native
            # vector_search_items implementation. In particular, PostgreSQL's
            # pgvector path performs ORDER BY cosine_distance LIMIT top_k in
            # the database and must not hydrate the entire embedding corpus.
            retrieve_module.cosine_topk = _gpu_cosine_topk
            database = self.service.database
            repo = getattr(database, "memory_item_repo", None)
            if repo is None or not hasattr(repo, "vector_search_items"):
                raise RuntimeError("MemU memory-item repository is unavailable")
            print("   Auxiliary similarity backend: CUDA PyTorch", flush=True)
            print("   Item similarity backend: database-native vector search", flush=True)
        except Exception as exc:
            raise RuntimeError(
                "memu_local gpu_similarity=true but CUDA initialization failed"
            ) from exc

    def _configure_item_similarity(self, database_provider: str, dsn: str) -> None:
        search_config = self.config.get("search", {}) or {}
        backend = str(
            search_config.get("item_similarity_backend", "pgvector")
        ).lower()
        if backend == "pgvector":
            print("   Item similarity backend: PostgreSQL pgvector", flush=True)
            return
        if backend not in {"torch_gpu", "faiss_gpu"}:
            raise ValueError(
                "search.item_similarity_backend must be one of "
                "'pgvector', 'torch_gpu', or 'faiss_gpu'"
            )
        if database_provider != "postgres":
            raise ValueError(f"{backend} item similarity requires PostgreSQL storage")

        cache = GpuItemIndexCache(
            dsn=dsn,
            backend=backend,
            device=int(search_config.get("gpu_device", 0)),
            fetch_size=int(search_config.get("gpu_index_fetch_size", 512)),
        )
        repo = self.service.database.memory_item_repo
        original_search = repo.vector_search_items

        def gpu_item_search(
            query_vec,
            top_k,
            where=None,
            *,
            ranking="similarity",
            recency_decay_days=30.0,
        ):
            if ranking == "similarity":
                hits = cache.search(query_vec, top_k, where)
                if hits is not None:
                    return hits
            return original_search(
                query_vec,
                top_k,
                where=where,
                ranking=ranking,
                recency_decay_days=recency_decay_days,
            )

        repo.vector_search_items = gpu_item_search
        self._gpu_item_indexes = cache
        print(f"   Item similarity backend: {backend} exact cosine", flush=True)

    def _install_bounded_item_recall(self) -> None:
        """Avoid memU 1.5.1's full item-pool load before vector top-k."""

        async def recall(service, state, step_context):
            if (
                not state.get("retrieve_item")
                or not state.get("needs_retrieval")
                or not state.get("proceed_to_items")
            ):
                state["item_hits"] = []
                state["item_pool"] = _BoundedItemPool()
                return state

            store = state["store"]
            where_filters = state.get("where") or {}
            query_vector = state.get("query_vector")
            if query_vector is None:
                embed_client = service._get_step_embedding_client(step_context)
                query_vector = (
                    await embed_client.embed([state["active_query"]])
                )[0]
                state["query_vector"] = query_vector
            hits = store.memory_item_repo.vector_search_items(
                query_vector,
                service.retrieve_config.item.top_k,
                where=where_filters,
                ranking=service.retrieve_config.item.ranking,
                recency_decay_days=service.retrieve_config.item.recency_decay_days,
            )
            item_pool = _BoundedItemPool()
            for item_id, _score in hits:
                item = store.memory_item_repo.get_item(item_id)
                if item is not None:
                    item_pool[item_id] = item
            state["item_hits"] = hits
            state["item_pool"] = item_pool
            return state

        self.service._rag_recall_items = types.MethodType(recall, self.service)
        replacement = next(
            step
            for step in self.service._build_rag_retrieve_workflow()
            if step.step_id == "recall_items"
        )
        self.service._pipelines.replace_step(
            "retrieve_rag",
            "recall_items",
            replacement,
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
        return safe[:100] or "conversation"

    def _user_id(self, conversation_id: str) -> str:
        scope_parts = [str(self.config.get("run_name") or "").strip()]
        if self.config.get("dataset_name") == "memoryagentbench":
            selection = self.config.get("memoryagentbench", {}) or {}
            scope_parts.extend(
                [
                    str(selection.get("split") or "").strip(),
                    str(selection.get("source") or "").strip(),
                ]
            )
        scope = ":".join(part for part in scope_parts if part)
        return f"{scope}:{conversation_id}" if scope else str(conversation_id)

    def _conversation_payload(self, conversation: Conversation) -> list[dict[str, Any]]:
        speaker_a = conversation.metadata.get("speaker_a", "")
        speaker_b = conversation.metadata.get("speaker_b", "")
        messages = []
        for message in conversation.messages:
            sender = str(message.sender_name or "user")
            if sender == speaker_a:
                role = "user"
            elif sender == speaker_b:
                role = "assistant"
            else:
                role = "user"
            item: dict[str, Any] = {
                "role": role,
                "name": sender,
                "content": str(message.content),
            }
            if message.timestamp:
                item["created_at"] = message.timestamp.isoformat()
            messages.append(item)
        return messages

    def _write_resource(
        self,
        conversation: Conversation,
        messages: list[dict[str, Any]],
        suffix: str,
    ) -> Path:
        path = self.resource_dir / (
            f"{self._safe_name(conversation.conversation_id)}-{suffix}.json"
        )
        path.write_text(
            json.dumps(messages, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _conversation_fingerprint(
        conversation: Conversation, messages: list[dict[str, Any]]
    ) -> str:
        payload = {
            "conversation_id": conversation.conversation_id,
            "messages": messages,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_add_resume_state(self) -> dict[str, Any]:
        empty = {"version": 1, "conversations": {}}
        if not self.add_resume_enabled or not self.add_resume_state_path.exists():
            return empty
        try:
            state = json.loads(self.add_resume_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load memU Add resume state: {self.add_resume_state_path}"
            ) from exc
        if state.get("version") != 1 or not isinstance(
            state.get("conversations"), dict
        ):
            raise RuntimeError(
                f"Unsupported memU Add resume state: {self.add_resume_state_path}"
            )
        return state

    async def _save_add_resume_state(self) -> None:
        if not self.add_resume_enabled:
            return
        async with self._add_resume_lock:
            self.add_resume_state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.add_resume_state_path.with_name(
                f"{self.add_resume_state_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temp_path.write_text(
                    json.dumps(self._add_resume_state, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                for attempt in range(8):
                    try:
                        temp_path.replace(self.add_resume_state_path)
                        break
                    except PermissionError:
                        if attempt == 7:
                            raise
                        # Windows antivirus/indexing can briefly hold the target.
                        await asyncio.sleep(0.1 * (attempt + 1))
            finally:
                temp_path.unlink(missing_ok=True)

    def _resume_entry(
        self,
        conversation: Conversation,
        messages: list[dict[str, Any]],
        total_parts: int,
    ) -> dict[str, Any]:
        conv_id = str(conversation.conversation_id)
        fingerprint = self._conversation_fingerprint(conversation, messages)
        entry = self._add_resume_state["conversations"].get(conv_id)
        if entry is None:
            entry = {
                "fingerprint": fingerprint,
                "total_parts": total_parts,
                "completed_parts": [],
            }
            self._add_resume_state["conversations"][conv_id] = entry
        elif (
            entry.get("fingerprint") != fingerprint
            or int(entry.get("total_parts", -1)) != total_parts
        ):
            raise RuntimeError(
                f"memU Add resume state does not match conversation {conv_id}; "
                "use a new run-name/output directory for changed input."
            )
        return entry

    def _split_resource_messages(
        self, conversation: Conversation
    ) -> list[list[dict[str, Any]]]:
        """Map each official MemoryAgentBench chunk to one memU resource."""
        messages = self._conversation_payload(conversation)
        return [[message] for message in messages] or [[]]

    async def _add_serialized_legacy(
        self, conversations: List[Conversation], **kwargs
    ) -> Dict[str, Any]:
        self.console.print("\n" + "=" * 60, style="bold cyan")
        self.console.print("Stage 1: Add (memU local)", style="bold cyan")
        self.console.print("=" * 60, style="bold cyan")

        added = []
        # A single service/database is intentionally serialized.  The SDK's
        # SQLite repository and category context are process-local caches.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=1,
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]memU conversations",
                total=len(conversations),
            )

            async with self._service_lock:
                for conversation in conversations:
                    chunks = self._split_resource_messages(conversation)
                    messages_for_fingerprint = self._conversation_payload(conversation)
                    resume_entry = self._resume_entry(
                        conversation, messages_for_fingerprint, len(chunks)
                    )
                    completed_parts = {
                        int(index) for index in resume_entry.get("completed_parts", [])
                    }
                    if completed_parts:
                        self.console.print(
                            f"[yellow]↻ Resuming {conversation.conversation_id}: "
                            f"skipping {len(completed_parts)}/{len(chunks)} chunks[/yellow]"
                        )
                    chunk_task = progress.add_task(
                        f"[yellow]{conversation.conversation_id} chunks",
                        total=len(chunks),
                    )
                    chunk_items = 0
                    try:
                        for chunk_index, messages in enumerate(chunks):
                            if chunk_index in completed_parts:
                                progress.advance(chunk_task)
                                continue
                            resource = self._write_resource(
                                conversation, messages, f"part-{chunk_index:04d}"
                            )
                            self._resource_paths[conversation.conversation_id] = resource
                            result = await self.service.memorize(
                                resource_url=str(resource),
                                modality="conversation",
                                user={"user_id": self._user_id(conversation.conversation_id)},
                            )
                            chunk_items += len(result.get("items", []))
                            completed_parts.add(chunk_index)
                            resume_entry["completed_parts"] = sorted(completed_parts)
                            await self._save_add_resume_state()
                            progress.advance(chunk_task)
                    except Exception:
                        progress.update(chunk_task, description=f"[red]{conversation.conversation_id} chunks failed")
                        raise
                    else:
                        progress.update(chunk_task, description=f"[green]{conversation.conversation_id} chunks complete")
                        resume_entry["completed"] = True
                        await self._save_add_resume_state()
                        progress.advance(overall_task)
                        added.append(
                            {
                                "conversation_id": conversation.conversation_id,
                                "parts": len(chunks),
                                "items": chunk_items,
                            }
                        )
            progress.update(overall_task, description="[bold green]memU conversations complete")
        return {"system": "memu_local", "conversations": added}

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """Add conversations concurrently, preserving chunk order per conversation."""
        del kwargs
        semaphore = asyncio.Semaphore(max(1, int(self.num_workers)))
        self.console.print(
            f"Add Concurrency: {max(1, int(self.num_workers))} conversation worker(s)",
            style="bold cyan",
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
            refresh_per_second=1,
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]memU conversations",
                total=len(conversations),
            )

            async def process(conversation: Conversation):
                async with semaphore:
                    return await self._add_one_conversation(
                        conversation, progress, overall_task
                    )

            added = await asyncio.gather(
                *(process(conversation) for conversation in conversations)
            )
            progress.update(
                overall_task,
                description="[bold green]memU conversations complete",
            )
        return {"system": "memu_local", "conversations": added}

    async def _add_one_conversation(
        self, conversation: Conversation, progress: Progress, overall_task: int
    ) -> dict[str, Any]:
        """Process one conversation sequentially while peers run concurrently."""
        chunks = self._split_resource_messages(conversation)
        messages = self._conversation_payload(conversation)
        resume_entry = self._resume_entry(conversation, messages, len(chunks))
        completed_parts = {
            int(index) for index in resume_entry.get("completed_parts", [])
        }
        chunk_task = progress.add_task(
            f"[yellow]{conversation.conversation_id} chunks",
            total=len(chunks),
        )
        chunk_items = 0
        try:
            for chunk_index, chunk_messages in enumerate(chunks):
                if chunk_index in completed_parts:
                    progress.advance(chunk_task)
                    continue
                resource = self._write_resource(
                    conversation, chunk_messages, f"part-{chunk_index:04d}"
                )
                self._resource_paths[conversation.conversation_id] = resource
                result = await self.service.memorize(
                    resource_url=str(resource),
                    modality="conversation",
                    user={"user_id": self._user_id(conversation.conversation_id)},
                )
                chunk_items += len(result.get("items", []))
                completed_parts.add(chunk_index)
                resume_entry["completed_parts"] = sorted(completed_parts)
                await self._save_add_resume_state()
                progress.advance(chunk_task)
        except Exception:
            progress.update(
                chunk_task,
                description=f"[red]{conversation.conversation_id} chunks failed",
            )
            raise

        progress.update(
            chunk_task,
            description=f"[green]{conversation.conversation_id} chunks complete",
        )
        resume_entry["completed"] = True
        await self._save_add_resume_state()
        progress.advance(overall_task)
        return {
            "conversation_id": conversation.conversation_id,
            "parts": len(chunks),
            "items": chunk_items,
        }

    @staticmethod
    def _result_content(item: dict[str, Any]) -> str:
        summary = item.get("summary") or item.get("content") or item.get("caption")
        return str(summary or "").strip()

    def _format_context(self, results: list[dict[str, Any]]) -> str:
        lines = [
            f"- {result['content']}"
            for result in results
            if str(result.get("content", "")).strip()
        ]
        return "<MEMU MEMORY>\n" + "\n".join(lines) + "\n</MEMU MEMORY>" if lines else ""

    def _build_search_result(
        self,
        query: str,
        conversation_id: str,
        response: dict[str, Any],
        top_k: int,
        question_id: str | None = None,
    ) -> SearchResult:
        raw_items = list(response.get("items", []))
        raw_categories = list(response.get("categories", []))
        results = []
        for item in raw_items[:top_k]:
            content = self._result_content(item)
            if content:
                results.append(
                    {
                        "content": content,
                        "score": float(item.get("score", 0.0) or 0.0),
                        "user_id": self._user_id(conversation_id),
                        "metadata": {
                            "memory_type": item.get("memory_type"),
                            "raw": self._json_safe(item),
                        },
                    }
                )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "memu_local",
                "top_k": top_k,
                "memu_method": self.config.get("memu", {})
                .get("retrieve", {})
                .get("method", "rag"),
                "category_count": len(raw_categories),
                "formatted_context": self._format_context(results),
            },
            question_id=question_id or "",
        )

    async def _retrieve_one(
        self,
        query: str,
        conversation_id: str,
        top_k: int,
        question_id: str | None = None,
    ) -> SearchResult:
        response = await self.service.retrieve(
            queries=[{"role": "user", "content": query}],
            where={"user_id": self._user_id(conversation_id)},
        )
        return self._build_search_result(
            query, conversation_id, response, top_k, question_id
        )

    def _install_embedding_batcher(self, batch_size: int):
        """Install a temporary batcher on memU's cached embedding client."""
        get_client = getattr(self.service, "_get_llm_base_client", None)
        if not callable(get_client):
            return None
        try:
            client = get_client("embedding")
        except Exception:
            try:
                client = get_client("default")
            except Exception:
                return None
        original_embed = getattr(client, "embed", None)
        if not callable(original_embed):
            return None
        batcher = _EmbeddingBatcher(original_embed, batch_size)
        client.embed = batcher.embed

        def restore() -> None:
            client.embed = original_embed

        return restore

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        del index
        top_k = int(kwargs.get("top_k", self.config.get("search", {}).get("top_k", 20)))
        async with self._service_lock:
            return await self._retrieve_one(
                query,
                conversation_id,
                top_k,
                kwargs.get("question_id"),
            )

    async def search_batch(self, requests: list[dict[str, Any]], index: Any = None):
        """Search independent queries while batching their embedding calls."""
        del index
        if not requests:
            return []
        search_cfg = self.config.get("search", {}) or {}
        batch_size = max(
            1,
            int(search_cfg.get("embedding_batch_size", 8)),
        )
        top_k_default = int(search_cfg.get("top_k", 20))
        restore = None
        async with self._service_lock:
            try:
                restore = self._install_embedding_batcher(batch_size)
                return await asyncio.gather(
                    *(
                        self._retrieve_one(
                            str(request["query"]),
                            str(request["conversation_id"]),
                            int(request.get("top_k", top_k_default)),
                            request.get("question_id"),
                        )
                        for request in requests
                    )
                )
            finally:
                if restore:
                    restore()

    def release_conversation_search_cache(
        self, conversation_id: str, index: Any = None
    ) -> None:
        del index
        if self._gpu_item_indexes is not None:
            self._gpu_item_indexes.release(self._user_id(conversation_id))

    async def close(self) -> None:
        if self._gpu_item_indexes is not None:
            self._gpu_item_indexes.clear()
        close_service = getattr(self.service, "close", None)
        if callable(close_service):
            result = close_service()
            if asyncio.iscoroutine(result):
                await result

    # OnlineAPIAdapter keeps these hooks abstract for its HTTP subclasses.  We
    # implement the complete local flow above, so the template hooks are only
    # present to satisfy the shared adapter contract.
    async def _add_user_messages(self, conv, messages, speaker, **kwargs):
        raise NotImplementedError("memu_local uses its serialized add() flow")

    async def _search_single_user(self, query, conversation_id, user_id, top_k, **kwargs):
        raise NotImplementedError("memu_local implements search() directly")

    def _build_single_search_result(self, query, conversation_id, results, user_id, top_k, **kwargs):
        raise NotImplementedError("memu_local implements search() directly")

    def _build_dual_search_result(
        self, query, conversation_id, all_results, results_a, results_b,
        speaker_a, speaker_b, speaker_a_user_id, speaker_b_user_id, top_k, **kwargs
    ):
        raise NotImplementedError("memu_local implements search() directly")

    def get_system_info(self) -> Dict[str, Any]:
        info = super().get_system_info()
        info["sdk"] = "memu-py==1.5.1"
        info["database"] = self.config.get("memu", {}).get("database", {}).get("provider", "inmemory")
        return info
