"""LLM-derived memory reverse-ablation variants over the strong local RAG base.

The adapter deliberately keeps the baseline RAG adapter untouched. It reuses the
same chunking, embedding, reranking, and answer stack, but replaces or augments
the indexed evidence with LLM-extracted summaries, atomic facts, or graph-like
structured memories.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openai import AsyncOpenAI
from rich.console import Console
from tqdm import tqdm

from evaluation.src.adapters.rag_adapter import RAGAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.utils.fingerprint import fingerprint

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional robustness dependency
    repair_json = None


@register_adapter("rag_memory_ablation")
class RAGMemoryAblationAdapter(RAGAdapter):
    """RAG variants that index LLM-derived memory representations."""

    INDEX_FILENAME = "rag_memory_ablation_index.json"
    PROMPT_VERSION = 2
    CHUNK_GRAPH_VERSION = 1

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir=output_dir)
        if self.storage_backend != "local":
            raise ValueError("rag_memory_ablation currently supports storage.backend='local' only.")

        self.console = Console()
        self.system_name = str(config.get("name") or "rag_memory_ablation")
        storage_cfg = config.get("storage", {}) or {}
        self.index_save_every_chunks = max(1, int(storage_cfg.get("save_every_chunks", 25)))

        memory_cfg = config.get("memory", {}) or {}
        self.memory_variant = str(memory_cfg.get("variant", "atomic_facts_only"))
        allowed_variants = {
            "atomic_facts_only",
            "raw_summary",
            "graph_only",
            "graph_expanded_raw",
        }
        if self.memory_variant not in allowed_variants:
            raise ValueError(
                f"memory.variant must be one of {sorted(allowed_variants)}, "
                f"got {self.memory_variant!r}."
            )

        self.fail_on_extraction_error = bool(memory_cfg.get("fail_on_extraction_error", True))
        self.max_extract_concurrency = max(1, int(memory_cfg.get("max_extract_concurrency", 4)))
        self._extract_semaphore = asyncio.Semaphore(self.max_extract_concurrency)

        atomic_cfg = memory_cfg.get("atomic_facts", {}) or {}
        self.atomic_max_facts_per_chunk = max(1, int(atomic_cfg.get("max_facts_per_chunk", 6)))
        self.atomic_max_fact_words = max(1, int(atomic_cfg.get("max_fact_words", 28)))
        self.atomic_max_quote_words = max(1, int(atomic_cfg.get("max_quote_words", 12)))
        self.atomic_max_entities = max(0, int(atomic_cfg.get("max_entities", 5)))

        memory_llm_cfg = memory_cfg.get("llm", {}) or {}
        self.memory_model = memory_llm_cfg.get("model") or self.answer_model
        self.memory_temperature = float(memory_llm_cfg.get("temperature", 0))
        self.memory_max_tokens = int(memory_llm_cfg.get("max_tokens", 4096))
        self.memory_max_retries = max(1, int(memory_llm_cfg.get("max_retries", 3)))
        self.memory_retry_interval_seconds = max(
            0.0,
            float(memory_llm_cfg.get("retry_interval_seconds", 1.0)),
        )
        memory_retry_max_interval = memory_llm_cfg.get("retry_max_interval_seconds")
        self.memory_retry_max_interval_seconds = (
            None
            if memory_retry_max_interval is None
            else max(
                self.memory_retry_interval_seconds,
                float(memory_retry_max_interval),
            )
        )
        self.memory_json_response = bool(memory_llm_cfg.get("json_response", False))
        memory_api_key = self._resolve_config_or_env(
            memory_llm_cfg.get("api_key"),
            "MEMORY_EXTRACT_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        )
        memory_base_url = self._resolve_config_or_env(
            memory_llm_cfg.get("base_url"),
            "MEMORY_EXTRACT_BASE_URL",
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            default="https://openrouter.ai/api/v1",
        )
        if (
            self.memory_model == self.answer_model
            and memory_api_key == self.answer_client.api_key
            and str(memory_base_url).rstrip("/") == str(self.answer_client.base_url).rstrip("/")
        ):
            self.memory_client = self.answer_client
            self._memory_client_is_answer_client = True
        else:
            self.memory_client = AsyncOpenAI(api_key=memory_api_key, base_url=memory_base_url)
            self._memory_client_is_answer_client = False

        graph_cfg = memory_cfg.get("graph", {}) or {}
        self.graph_link_embedding_top_k = max(
            1,
            int(graph_cfg.get("link_embedding_top_k", 10)),
        )
        self.graph_max_raw_candidates = int(graph_cfg.get("max_raw_candidates", self.embedding_top_k))
        self.include_graph_cues_in_context = bool(graph_cfg.get("include_graph_cues_in_context", False))

        print("RAGMemoryAblationAdapter initialized")
        print(f"   Variant: {self.memory_variant}")
        print(f"   Memory extraction model: {self.memory_model}")
        print(f"   Memory extraction concurrency: {self.max_extract_concurrency}")
        if self.memory_variant == "atomic_facts_only":
            print(
                "   Atomic fact caps: "
                f"max_facts={self.atomic_max_facts_per_chunk}, "
                f"max_fact_words={self.atomic_max_fact_words}, "
                f"max_quote_words={self.atomic_max_quote_words}, "
                f"max_entities={self.atomic_max_entities}"
            )
            print(
                "   Atomic extraction checkpoint interval: "
                f"every {self.index_save_every_chunks} raw chunk(s)"
            )
        print(
            f"   Memory extraction retries: {self.memory_max_retries} "
            f"(base wait {self.memory_retry_interval_seconds}s, "
            f"max wait {self.memory_retry_max_interval_seconds or 'uncapped'}s)"
        )

    async def add(self, conversations: List[Conversation], **kwargs) -> Dict[str, Any]:
        """Build and persist an LLM-derived memory index."""
        self.console.print("\n" + "=" * 60, style="bold cyan")
        self.console.print(
            f"Stage 1: Add ({self.system_name}, variant={self.memory_variant})",
            style="bold cyan",
        )
        self.console.print("=" * 60, style="bold cyan")

        persist_embeddings = self._should_persist_embeddings(conversations)
        index = self._load_resumeable_index()
        indexed_conversations: Dict[str, Any] = dict(index.get("conversations", {}))

        pending_conversations = []
        reused_count = 0
        for conv in conversations:
            cached = indexed_conversations.get(conv.conversation_id)
            if cached and cached.get("source_fingerprint") == fingerprint(conv) and self._is_conversation_index_reusable(
                cached,
                allow_missing_embeddings=not persist_embeddings,
            ):
                reused_count += 1
                continue
            pending_conversations.append(conv)

        if reused_count:
            self.console.print(
                f"Resuming Add from cached memory index: reused {reused_count}/"
                f"{len(conversations)} conversations",
                style="yellow",
            )
        if not pending_conversations:
            self.console.print("All conversations already have reusable memory index data", style="green")
            return index

        semaphore = asyncio.Semaphore(max(1, self.sample_workers))
        save_lock = asyncio.Lock()
        chunks_since_save = 0

        async def save_conversation_progress(
            conv_id: str,
            conv_index: Dict[str, Any],
            *,
            extracted_chunk_delta: int = 0,
            force: bool = False,
        ) -> None:
            nonlocal chunks_since_save
            async with save_lock:
                indexed_conversations[conv_id] = conv_index
                index["conversations"] = indexed_conversations
                chunks_since_save += extracted_chunk_delta
                if force or chunks_since_save >= self.index_save_every_chunks:
                    self._save_index(index, persist_embeddings=persist_embeddings)
                    chunks_since_save = 0

        async def build_single(conv: Conversation) -> tuple[str, Dict[str, Any]]:
            cached_conv_index = indexed_conversations.get(conv.conversation_id)
            if cached_conv_index and cached_conv_index.get("source_fingerprint") != fingerprint(conv):
                cached_conv_index = None

            async def save_progress(
                conv_index: Dict[str, Any],
                extracted_chunk_delta: int = 0,
                force: bool = False,
            ) -> None:
                conv_index["source_fingerprint"] = fingerprint(conv)
                await save_conversation_progress(
                    conv.conversation_id,
                    conv_index,
                    extracted_chunk_delta=extracted_chunk_delta,
                    force=force,
                )

            async with semaphore:
                conv_index = await self._build_conversation_index(
                    conv,
                    cached_conv_index=cached_conv_index,
                    save_progress=save_progress if self.memory_variant == "atomic_facts_only" else None,
                )
                conv_index["source_fingerprint"] = fingerprint(conv)
                return conv.conversation_id, conv_index

        tasks = [build_single(conv) for conv in pending_conversations]
        processed_since_save = 0
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{self.memory_variant} index",
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

    async def _build_conversation_index(
        self,
        conv: Conversation,
        *,
        cached_conv_index: Optional[Dict[str, Any]] = None,
        save_progress: Optional[
            Callable[[Dict[str, Any], int, bool], Awaitable[None]]
        ] = None,
    ) -> Dict[str, Any]:
        raw_chunks, turn_context = self._build_raw_chunks(conv)
        conv_index: Dict[str, Any] = {
            "conversation_id": conv.conversation_id,
            "speaker_a": conv.metadata.get("speaker_a", ""),
            "speaker_b": conv.metadata.get("speaker_b", ""),
            "memory_variant": self.memory_variant,
            "memory_build_status": "extracting",
            "raw_chunks": raw_chunks,
            "chunks": [],
        }
        if turn_context is not None:
            conv_index["turn_context"] = turn_context

        if self.memory_variant == "atomic_facts_only":
            memory_chunks = await self._build_atomic_fact_chunks(
                conv,
                raw_chunks,
                cached_conv_index=cached_conv_index,
                conv_index=conv_index,
                save_progress=save_progress,
            )
        elif self.memory_variant == "raw_summary":
            memory_chunks = await self._build_raw_summary_chunks(conv, raw_chunks)
        elif self.memory_variant == "graph_only":
            memory_chunks = await self._build_graph_chunks(conv, raw_chunks)
        elif self.memory_variant == "graph_expanded_raw":
            # In this variant the graph nodes are the source chunks themselves.
            # Keeping chunks/raw_chunks aligned also lets the normal persistence
            # and embedding recovery paths continue to work.
            memory_chunks = raw_chunks
        else:
            raise ValueError(f"Unsupported memory variant: {self.memory_variant}")

        if not memory_chunks:
            raise RuntimeError(
                f"No memory chunks produced for conversation {conv.conversation_id} "
                f"with variant {self.memory_variant}."
            )

        conv_index["chunks"] = memory_chunks
        conv_index["memory_build_status"] = "embedding"
        if save_progress is not None:
            await save_progress(conv_index, 0, True)

        conv_index["embeddings"] = await self._embed_chunks(
            memory_chunks,
            stage="add",
            conversation_id=conv.conversation_id,
            metadata_key="memory_chunk_start",
        )

        if self.memory_variant == "graph_expanded_raw":
            conv_index["raw_embeddings"] = conv_index["embeddings"]
            conv_index["graph_edges"] = await self._build_graph_expanded_raw_edges(
                conv,
                raw_chunks,
                conv_index["raw_embeddings"],
            )

        conv_index["memory_build_status"] = "complete"
        if self.memory_variant == "atomic_facts_only":
            conv_index["atomic_fact_progress"] = {
                "extracted_chunks": len(raw_chunks),
                "total_chunks": len(raw_chunks),
                "complete": True,
            }
        if save_progress is not None:
            await save_progress(conv_index, 0, True)

        return conv_index

    def _build_raw_chunks(
        self,
        conv: Conversation,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.chunk_strategy == "turn":
            chunks = self._create_turn_chunks(conv, conv.conversation_id)
            turn_context = None
        else:
            document = self._format_conversation(conv)
            chunks = self._create_token_chunks(document, conv.conversation_id)
            turn_context = None
            if self.expand_turns_on_display:
                turn_context = self._build_turn_context(conv)
                self._annotate_token_chunks_with_turns(chunks, turn_context["turn_spans"])

        for chunk in chunks:
            chunk["memory_type"] = "raw"
            chunk["source_chunk_id"] = chunk.get("chunk_id")
            chunk["source_chunk_index"] = chunk.get("chunk_index")
        return chunks, turn_context

    async def _embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        *,
        stage: str,
        conversation_id: str,
        metadata_key: str,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            batch_embeddings = await self._embed_texts(
                [chunk["content"] for chunk in batch],
                stage=stage,
                conversation_id=conversation_id,
                metadata={metadata_key: start, "chunk_count": len(batch)},
            )
            embeddings.extend(batch_embeddings)
        return embeddings

    async def _build_atomic_fact_chunks(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
        *,
        cached_conv_index: Optional[Dict[str, Any]] = None,
        conv_index: Optional[Dict[str, Any]] = None,
        save_progress: Optional[
            Callable[[Dict[str, Any], int, bool], Awaitable[None]]
        ] = None,
    ) -> List[Dict[str, Any]]:
        extracted_by_source_index = self._load_cached_atomic_fact_extractions(
            cached_conv_index,
            raw_chunks,
        )
        if extracted_by_source_index:
            self.console.print(
                f"Resuming atomic facts for {conv.conversation_id}: "
                f"reused {len(extracted_by_source_index)}/{len(raw_chunks)} raw chunks",
                style="yellow",
            )

        self._update_atomic_fact_progress(
            conv,
            raw_chunks,
            extracted_by_source_index,
            conv_index,
        )

        async def extract(source_idx: int, chunk: Dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
            payload = await self._extract_atomic_facts(conv, chunk)
            facts = payload.get("facts", [])
            return source_idx, [fact for fact in facts if isinstance(fact, dict)]

        pending_items = [
            (source_idx, chunk)
            for source_idx, chunk in enumerate(raw_chunks)
            if source_idx not in extracted_by_source_index
        ]
        tasks = [asyncio.create_task(extract(source_idx, chunk)) for source_idx, chunk in pending_items]
        try:
            for coro in asyncio.as_completed(tasks):
                source_idx, facts = await coro
                extracted_by_source_index[source_idx] = facts
                self._update_atomic_fact_progress(
                    conv,
                    raw_chunks,
                    extracted_by_source_index,
                    conv_index,
                )
                if save_progress is not None and conv_index is not None:
                    await save_progress(conv_index, 1, False)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        self._update_atomic_fact_progress(
            conv,
            raw_chunks,
            extracted_by_source_index,
            conv_index,
        )
        return self._atomic_extractions_to_memory_chunks(
            conv,
            raw_chunks,
            extracted_by_source_index,
        )

    def _load_cached_atomic_fact_extractions(
        self,
        cached_conv_index: Optional[Dict[str, Any]],
        raw_chunks: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not isinstance(cached_conv_index, dict):
            return {}
        if cached_conv_index.get("memory_variant") != self.memory_variant:
            return {}
        if not self._raw_chunks_match(cached_conv_index.get("raw_chunks"), raw_chunks):
            return {}

        records = cached_conv_index.get("atomic_fact_extractions")
        if not isinstance(records, list):
            return {}

        extracted_by_source_index: Dict[int, List[Dict[str, Any]]] = {}
        for record in records:
            if not isinstance(record, dict):
                continue
            try:
                source_idx = int(record.get("source_chunk_index"))
            except (TypeError, ValueError):
                continue
            if source_idx < 0 or source_idx >= len(raw_chunks):
                continue
            expected_chunk_id = raw_chunks[source_idx].get("chunk_id")
            if record.get("source_chunk_id") != expected_chunk_id:
                continue
            facts = record.get("facts", [])
            if not isinstance(facts, list):
                facts = []
            extracted_by_source_index[source_idx] = [
                fact for fact in facts if isinstance(fact, dict)
            ]
        return extracted_by_source_index

    def _raw_chunks_match(
        self,
        cached_raw_chunks: Any,
        raw_chunks: List[Dict[str, Any]],
    ) -> bool:
        if not isinstance(cached_raw_chunks, list) or len(cached_raw_chunks) != len(raw_chunks):
            return False
        for cached_chunk, current_chunk in zip(cached_raw_chunks, raw_chunks):
            if not isinstance(cached_chunk, dict):
                return False
            for key in ("chunk_id", "chunk_index", "start_token", "end_token", "content"):
                if cached_chunk.get(key) != current_chunk.get(key):
                    return False
        return True

    def _update_atomic_fact_progress(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
        extracted_by_source_index: Dict[int, List[Dict[str, Any]]],
        conv_index: Optional[Dict[str, Any]],
    ) -> None:
        if conv_index is None:
            return
        conv_index["atomic_fact_extractions"] = self._serialize_atomic_fact_extractions(
            raw_chunks,
            extracted_by_source_index,
        )
        conv_index["chunks"] = self._atomic_extractions_to_memory_chunks(
            conv,
            raw_chunks,
            extracted_by_source_index,
        )
        conv_index["atomic_fact_progress"] = {
            "extracted_chunks": len(extracted_by_source_index),
            "total_chunks": len(raw_chunks),
            "complete": len(extracted_by_source_index) == len(raw_chunks),
        }

    def _serialize_atomic_fact_extractions(
        self,
        raw_chunks: List[Dict[str, Any]],
        extracted_by_source_index: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for source_idx, source_chunk in enumerate(raw_chunks):
            if source_idx not in extracted_by_source_index:
                continue
            records.append(
                {
                    "source_chunk_id": source_chunk.get("chunk_id"),
                    "source_chunk_index": source_chunk.get("chunk_index", source_idx),
                    "facts": extracted_by_source_index[source_idx],
                }
            )
        return records

    def _atomic_extractions_to_memory_chunks(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
        extracted_by_source_index: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        memory_chunks: List[Dict[str, Any]] = []
        for source_idx, source_chunk in enumerate(raw_chunks):
            facts = extracted_by_source_index.get(source_idx, [])
            for fact_idx, fact in enumerate(facts):
                fact_no = len(memory_chunks)
                memory_chunks.append(
                    {
                        "chunk_id": f"{conv.conversation_id}_fact_{fact_no}",
                        "chunk_index": fact_no,
                        "memory_type": "atomic_fact",
                        "source_chunk_id": source_chunk.get("chunk_id"),
                        "source_chunk_index": source_chunk.get("chunk_index"),
                        "start_token": source_chunk.get("start_token"),
                        "end_token": source_chunk.get("end_token"),
                        "content": self._render_atomic_fact(
                            fact=fact,
                            source_chunk=source_chunk,
                            fact_idx=fact_idx,
                        ),
                        "extracted": fact,
                    }
                )
        return memory_chunks

    async def _build_raw_summary_chunks(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        async def extract(chunk: Dict[str, Any]) -> Dict[str, Any]:
            return await self._extract_summary(conv, chunk)

        summaries = await asyncio.gather(*(extract(chunk) for chunk in raw_chunks))
        memory_chunks: List[Dict[str, Any]] = []
        for chunk, summary in zip(raw_chunks, summaries):
            idx = len(memory_chunks)
            memory_chunks.append(
                {
                    "chunk_id": f"{conv.conversation_id}_raw_summary_{idx}",
                    "chunk_index": idx,
                    "memory_type": "raw_summary",
                    "source_chunk_id": chunk.get("chunk_id"),
                    "source_chunk_index": chunk.get("chunk_index"),
                    "start_token": chunk.get("start_token"),
                    "end_token": chunk.get("end_token"),
                    "content": self._render_raw_summary(chunk, summary),
                    "summary": summary,
                }
            )
        return memory_chunks

    async def _build_graph_chunks(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        async def extract(chunk: Dict[str, Any]) -> Dict[str, Any]:
            return await self._extract_graph(conv, chunk)

        payloads = await asyncio.gather(*(extract(chunk) for chunk in raw_chunks))
        memory_chunks: List[Dict[str, Any]] = []
        for chunk, payload in zip(raw_chunks, payloads):
            memory_chunks.extend(self._graph_payload_to_chunks(conv, chunk, payload, len(memory_chunks)))
        return memory_chunks

    async def _build_graph_expanded_raw_edges(
        self,
        conv: Conversation,
        raw_chunks: List[Dict[str, Any]],
        raw_embeddings: List[List[float]],
    ) -> List[Dict[str, Any]]:
        """Link each chunk to LLM-confirmed neighbors among prior chunks."""

        async def judge_new_chunk(new_idx: int) -> List[Dict[str, Any]]:
            embedding_candidates = self._get_embedding_candidates(
                raw_chunks[:new_idx],
                raw_embeddings[:new_idx],
                raw_embeddings[new_idx],
                top_k=self.graph_link_embedding_top_k,
            )
            if not embedding_candidates:
                return []

            payload = await self._judge_chunk_links(
                conv=conv,
                new_chunk=raw_chunks[new_idx],
                candidates=embedding_candidates,
            )
            candidates_by_index = {
                int(candidate["index"]): candidate for candidate in embedding_candidates
            }
            edges = []
            seen_existing_indices = set()
            for related in payload.get("related_chunks") or []:
                if not isinstance(related, dict):
                    continue
                try:
                    existing_idx = int(related.get("chunk_index"))
                except (TypeError, ValueError):
                    continue
                candidate = candidates_by_index.get(existing_idx)
                if candidate is None or existing_idx in seen_existing_indices:
                    continue
                seen_existing_indices.add(existing_idx)
                edges.append(
                    {
                        "source_chunk_index": existing_idx,
                        "target_chunk_index": new_idx,
                        "embedding_similarity": candidate["similarity_score"],
                        "reason": str(related.get("reason") or ""),
                    }
                )
            return edges

        edge_groups = await asyncio.gather(
            *(judge_new_chunk(new_idx) for new_idx in range(1, len(raw_chunks)))
        )
        return [edge for group in edge_groups for edge in group]

    async def _judge_chunk_links(
        self,
        *,
        conv: Conversation,
        new_chunk: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidate_text = "\n\n".join(
            "EXISTING CHUNK "
            f"{candidate['index']} (embedding_similarity={candidate['similarity_score']:.6f}):\n"
            f"{candidate['content']}"
            for candidate in candidates
        )
        prompt = f"""
Judge which existing chunks are genuinely related to the new chunk and should
be connected by an undirected graph edge.

Link chunks only when their connection would be useful for multi-hop retrieval
or joint question answering: for example, they describe the same entity,
event, relationship, evolving state, cause/effect chain, or a concrete follow-up.
Topical or stylistic similarity alone is not enough. Use only the supplied text.
It is valid to return no related chunks. Return JSON only.

Required JSON schema:
{{
  "related_chunks": [
    {{"chunk_index": 0, "reason": "brief source-supported connection"}}
  ]
}}

Conversation ID: {conv.conversation_id}
New chunk index: {new_chunk.get('chunk_index')}

NEW CHUNK:
{new_chunk.get('content', '')}

CANDIDATE EXISTING CHUNKS (embedding Top-{self.graph_link_embedding_top_k}):
{candidate_text}
"""
        return await self._complete_memory_json(
            prompt=prompt,
            call_type="chunk_graph_link_judge",
            conversation_id=conv.conversation_id,
            metadata={"source_chunk_id": new_chunk.get("chunk_id")},
        )

    async def _extract_atomic_facts(
        self,
        conv: Conversation,
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""
Extract explicit atomic facts from the source memory chunk.

Rules:
- Use only facts directly supported by the text. Do not infer hidden causes.
- Preserve names, titles, locations, dates, event order, speaker labels, and serial numbers exactly.
- For conflict-resolution facts, keep the original serial number. Larger serial numbers mean newer facts; do not drop this ordering clue.
- Return at most {self.atomic_max_facts_per_chunk} high-salience facts for retrieval and question answering.
- Prefer concrete events, state changes, decisions, locations, dates, names, and conflict/newness cues.
- Omit generic commentary, style descriptions, duplicate details, and low-value micro-actions when the chunk has many candidates.
- Keep each fact under {self.atomic_max_fact_words} words.
- Keep each source_quote under {self.atomic_max_quote_words} words.
- Keep entities to at most {self.atomic_max_entities} names.
- Ignore wrapper instructions such as "I have memorized"; extract from the actual remembered content.
- Return JSON only.

Required JSON schema:
{{
  "facts": [
    {{
      "fact": "one concise standalone fact",
      "speaker": "speaker if present, else empty string",
      "time": "time/date if present, else empty string",
      "order": 0,
      "serial_number": "exact serial/newness marker if present, else empty string",
      "entities": ["up to {self.atomic_max_entities} entity names"],
      "source_quote": "short exact quote supporting the fact"
    }}
  ]
}}

Conversation ID: {conv.conversation_id}
Source chunk id: {chunk.get('chunk_id')}
Source chunk index: {chunk.get('chunk_index')}

SOURCE MEMORY CHUNK:
{chunk.get('content', '')}
"""
        return await self._complete_memory_json(
            prompt=prompt,
            call_type="atomic_fact_extract",
            conversation_id=conv.conversation_id,
            metadata={"source_chunk_id": chunk.get("chunk_id")},
        )

    async def _extract_summary(
        self,
        conv: Conversation,
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""
Summarize this source memory chunk for retrieval and question answering.

Rules:
- Use only source-supported information.
- Preserve event order, entities, dates/times, speaker names, and serial numbers.
- For conflict-resolution facts, explicitly mention serial numbers and newer/older ordering.
- Do not replace the raw text; produce a compact structured companion summary.
- Return JSON only.

Required JSON schema:
{{
  "summary": "dense factual summary",
  "key_events": ["ordered event or state update"],
  "entities": ["important entities"],
  "time_order": ["date/time/order/serial cues"],
  "conflicts_or_updates": ["conflicts, revisions, newest facts, if any"]
}}

Conversation ID: {conv.conversation_id}
Source chunk id: {chunk.get('chunk_id')}
Source chunk index: {chunk.get('chunk_index')}

SOURCE MEMORY CHUNK:
{chunk.get('content', '')}
"""
        return await self._complete_memory_json(
            prompt=prompt,
            call_type="summary_extract",
            conversation_id=conv.conversation_id,
            metadata={"source_chunk_id": chunk.get("chunk_id")},
        )

    async def _extract_graph(
        self,
        conv: Conversation,
        chunk: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt = f"""
Convert this source memory chunk into a compact structured memory graph.

Rules:
- Use only source-supported information.
- Preserve exact entity names, event order, dates/times, and serial numbers.
- For conflict-resolution facts, serial_number must copy the source marker and newer facts must remain distinguishable.
- Prefer entity-event-relation and entity-attribute-time statements.
- Ignore wrapper instructions such as "I have memorized"; structure the actual remembered content.
- Return JSON only.

Required JSON schema:
{{
  "nodes": [
    {{"id": "stable local id", "label": "entity/event label", "type": "person|place|object|event|attribute|other", "aliases": []}}
  ],
  "events": [
    {{
      "event_id": "local event id",
      "description": "source-supported event/state update",
      "participants": ["entity labels"],
      "time": "time/date if present",
      "order": 0,
      "serial_number": "exact serial/newness marker if present",
      "source_quote": "short supporting quote"
    }}
  ],
  "relations": [
    {{
      "subject": "entity label",
      "relation": "relation or attribute name",
      "object": "entity/value/event label",
      "time": "time/date if present",
      "order": 0,
      "serial_number": "exact serial/newness marker if present",
      "source_quote": "short supporting quote"
    }}
  ]
}}

Conversation ID: {conv.conversation_id}
Source chunk id: {chunk.get('chunk_id')}
Source chunk index: {chunk.get('chunk_index')}

SOURCE MEMORY CHUNK:
{chunk.get('content', '')}
"""
        return await self._complete_memory_json(
            prompt=prompt,
            call_type="graph_extract",
            conversation_id=conv.conversation_id,
            metadata={"source_chunk_id": chunk.get("chunk_id")},
        )

    async def _complete_memory_json(
        self,
        *,
        prompt: str,
        call_type: str,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "model": self.memory_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a careful memory extraction engine. "
                        "Return one valid JSON object and no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.memory_temperature,
            "max_tokens": self.memory_max_tokens,
        }
        if self.memory_json_response:
            request_payload["response_format"] = {"type": "json_object"}
        if metadata:
            request_payload["metadata"] = metadata

        async with self._extract_semaphore:
            last_error = ""
            for attempt in range(self.memory_max_retries):
                started = time.perf_counter()
                try:
                    api_payload = dict(request_payload)
                    api_payload.pop("metadata", None)
                    response = await self.memory_client.chat.completions.create(**api_payload)
                    raw_response = self._model_dump(response)
                    await self._write_request_log(
                        call_type=call_type,
                        stage="add",
                        request=request_payload,
                        response=raw_response,
                        started=started,
                        conversation_id=conversation_id,
                        attempt=attempt + 1,
                    )
                    text = response.choices[0].message.content or ""
                    parsed = self._extract_json_object(text)
                    if parsed:
                        return parsed
                    last_error = "empty JSON object"
                except Exception as exc:
                    last_error = str(exc)
                    await self._write_request_log(
                        call_type=call_type,
                        stage="add",
                        request=request_payload,
                        response=None,
                        started=started,
                        conversation_id=conversation_id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )
                if attempt < self.memory_max_retries - 1:
                    await asyncio.sleep(self._memory_retry_wait_seconds(attempt))

        if self.fail_on_extraction_error:
            raise RuntimeError(f"{call_type} failed for {conversation_id}: {last_error}")
        return {}

    def _memory_retry_wait_seconds(self, attempt: int) -> float:
        wait_seconds = self.memory_retry_interval_seconds * (2**attempt)
        if self.memory_retry_max_interval_seconds is not None:
            wait_seconds = min(wait_seconds, self.memory_retry_max_interval_seconds)
        return wait_seconds

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        parsed = self._parse_memory_json_object(text)
        if parsed is not None:
            return parsed

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            parsed = self._parse_memory_json_object(text[start : end + 1])
            if parsed is not None:
                return parsed
        return {}

    def _parse_memory_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._repair_memory_json_object(text)
        return parsed if isinstance(parsed, dict) else None

    def _repair_memory_json_object(self, text: str) -> Optional[Any]:
        if repair_json is None:
            return None
        try:
            return repair_json(text, return_objects=True)
        except Exception:
            return None

    def _render_atomic_fact(
        self,
        *,
        fact: Dict[str, Any],
        source_chunk: Dict[str, Any],
        fact_idx: int,
    ) -> str:
        entities = fact.get("entities") or []
        if isinstance(entities, list):
            entity_text = ", ".join(str(entity) for entity in entities)
        else:
            entity_text = str(entities)
        return "\n".join(
            [
                "ATOMIC FACT MEMORY",
                f"source_id: {fact.get('source_id') or source_chunk.get('chunk_id')}",
                f"source_chunk_index: {source_chunk.get('chunk_index')}",
                f"fact_order_within_source: {fact.get('order', fact_idx)}",
                f"speaker: {fact.get('speaker', '')}",
                f"time: {fact.get('time', '')}",
                f"serial_number: {fact.get('serial_number', '')}",
                f"entities: {entity_text}",
                f"fact: {fact.get('fact', '')}",
                f"source_quote: {fact.get('source_quote', '')}",
            ]
        )

    def _render_raw_summary(self, source_chunk: Dict[str, Any], summary: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "RAW MEMORY WITH LLM SUMMARY",
                f"source_id: {source_chunk.get('chunk_id')}",
                f"source_chunk_index: {source_chunk.get('chunk_index')}",
                "",
                "RAW MEMORY:",
                str(source_chunk.get("content", "")),
                "",
                "LLM SUMMARY:",
                str(summary.get("summary", "")),
                f"key_events: {self._compact_list(summary.get('key_events'))}",
                f"entities: {self._compact_list(summary.get('entities'))}",
                f"time_order: {self._compact_list(summary.get('time_order'))}",
                f"conflicts_or_updates: {self._compact_list(summary.get('conflicts_or_updates'))}",
            ]
        )

    def _graph_payload_to_chunks(
        self,
        conv: Conversation,
        source_chunk: Dict[str, Any],
        payload: Dict[str, Any],
        start_index: int,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for event in payload.get("events") or []:
            if not isinstance(event, dict):
                continue
            idx = start_index + len(chunks)
            chunks.append(
                {
                    "chunk_id": f"{conv.conversation_id}_graph_{idx}",
                    "chunk_index": idx,
                    "memory_type": "graph_event",
                    "source_chunk_id": source_chunk.get("chunk_id"),
                    "source_chunk_index": source_chunk.get("chunk_index"),
                    "content": self._render_graph_event(event, source_chunk),
                    "graph_item": event,
                }
            )

        for relation in payload.get("relations") or []:
            if not isinstance(relation, dict):
                continue
            idx = start_index + len(chunks)
            chunks.append(
                {
                    "chunk_id": f"{conv.conversation_id}_graph_{idx}",
                    "chunk_index": idx,
                    "memory_type": "graph_relation",
                    "source_chunk_id": source_chunk.get("chunk_id"),
                    "source_chunk_index": source_chunk.get("chunk_index"),
                    "content": self._render_graph_relation(relation, source_chunk),
                    "graph_item": relation,
                }
            )

        if not chunks and payload.get("nodes"):
            idx = start_index
            chunks.append(
                {
                    "chunk_id": f"{conv.conversation_id}_graph_{idx}",
                    "chunk_index": idx,
                    "memory_type": "graph_nodes",
                    "source_chunk_id": source_chunk.get("chunk_id"),
                    "source_chunk_index": source_chunk.get("chunk_index"),
                    "content": "\n".join(
                        [
                            "GRAPH NODE MEMORY",
                            f"source_id: {source_chunk.get('chunk_id')}",
                            f"source_chunk_index: {source_chunk.get('chunk_index')}",
                            f"nodes: {self._compact_list(payload.get('nodes'))}",
                        ]
                    ),
                    "graph_item": {"nodes": payload.get("nodes")},
                }
            )

        return chunks

    def _render_graph_event(self, event: Dict[str, Any], source_chunk: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "GRAPH EVENT MEMORY",
                f"source_id: {source_chunk.get('chunk_id')}",
                f"source_chunk_index: {source_chunk.get('chunk_index')}",
                f"event_id: {event.get('event_id', '')}",
                f"participants: {self._compact_list(event.get('participants'))}",
                f"time: {event.get('time', '')}",
                f"order: {event.get('order', '')}",
                f"serial_number: {event.get('serial_number', '')}",
                f"description: {event.get('description', '')}",
                f"source_quote: {event.get('source_quote', '')}",
            ]
        )

    def _render_graph_relation(self, relation: Dict[str, Any], source_chunk: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "GRAPH RELATION MEMORY",
                f"source_id: {source_chunk.get('chunk_id')}",
                f"source_chunk_index: {source_chunk.get('chunk_index')}",
                f"subject: {relation.get('subject', '')}",
                f"relation: {relation.get('relation', '')}",
                f"object: {relation.get('object', '')}",
                f"time: {relation.get('time', '')}",
                f"order: {relation.get('order', '')}",
                f"serial_number: {relation.get('serial_number', '')}",
                f"source_quote: {relation.get('source_quote', '')}",
            ]
        )

    def _compact_list(self, value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, list):
            return "; ".join(json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item) for item in value)
        return str(value)

    async def search(
        self,
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs,
    ) -> SearchResult:
        question_id = kwargs.get("question_id")
        conv_index = index.get("conversations", {}).get(conversation_id)
        if not conv_index:
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"system": self.system_name, "error": "conversation not found"},
                question_id=question_id or "",
            )

        query_embedding = (
            await self._embed_texts(
                [query],
                stage="search",
                conversation_id=conversation_id,
                question_id=question_id,
            )
        )[0]

        if self.memory_variant == "graph_expanded_raw":
            return await self._search_graph_expanded_raw(
                query=query,
                query_embedding=query_embedding,
                conversation_id=conversation_id,
                conv_index=conv_index,
                question_id=question_id,
            )

        chunks = conv_index.get("chunks", [])
        embeddings = await self._ensure_conversation_embeddings(conv_index, conversation_id, stage="search")
        if not chunks or not embeddings:
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"system": self.system_name, "error": "empty index"},
                question_id=question_id or "",
            )

        candidates = self._get_embedding_candidates(
            chunks,
            embeddings,
            query_embedding,
            top_k=self.embedding_top_k,
        )
        return await self._finish_candidate_search(
            query=query,
            conversation_id=conversation_id,
            conv_index=conv_index,
            candidates=candidates,
            question_id=question_id,
            retrieval_metadata_extra={
                "embedding_candidate_source": "local_memory",
                "memory_chunk_count": len(chunks),
                "raw_chunk_count": len(conv_index.get("raw_chunks", [])),
            },
        )

    async def _search_graph_expanded_raw(
        self,
        *,
        query: str,
        query_embedding: List[float],
        conversation_id: str,
        conv_index: Dict[str, Any],
        question_id: Optional[str],
    ) -> SearchResult:
        raw_chunks = conv_index.get("raw_chunks", [])
        raw_embeddings = conv_index.get("raw_embeddings", [])
        if not raw_embeddings:
            raw_embeddings = await self._ensure_conversation_embeddings(
                conv_index,
                conversation_id,
                stage="search",
            )
            conv_index["raw_embeddings"] = raw_embeddings
        if not raw_chunks or not raw_embeddings:
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"system": self.system_name, "error": "empty raw index"},
                question_id=question_id or "",
            )

        embedding_seeds = self._get_embedding_candidates(
            raw_chunks,
            raw_embeddings,
            query_embedding,
            top_k=self.embedding_top_k,
        )
        raw_candidate_map: Dict[int, Dict[str, Any]] = {}

        def add_raw_candidate(raw_idx: int, source: str) -> None:
            if raw_idx < 0 or raw_idx >= len(raw_chunks):
                return
            raw_chunk = raw_chunks[raw_idx]
            similarity_score = self._cosine_similarity(query_embedding, raw_embeddings[raw_idx])
            existing = raw_candidate_map.get(raw_idx)
            if existing is None:
                existing = {
                    "index": raw_idx,
                    "content": raw_chunk.get("content", ""),
                    "chunk": raw_chunk,
                    "embedding_rank": len(raw_candidate_map) + 1,
                    "similarity_score": similarity_score,
                    "selection_sources": [],
                }
                raw_candidate_map[raw_idx] = existing
            if source not in existing["selection_sources"]:
                existing["selection_sources"].append(source)
            existing["similarity_score"] = max(existing["similarity_score"], similarity_score)

        seed_indices = set()
        for candidate in embedding_seeds:
            seed_idx = int(candidate["index"])
            seed_indices.add(seed_idx)
            add_raw_candidate(seed_idx, source="embedding_seed")

        adjacency: Dict[int, set[int]] = {}
        for edge in conv_index.get("graph_edges", []):
            if not isinstance(edge, dict):
                continue
            try:
                source_idx = int(edge.get("source_chunk_index"))
                target_idx = int(edge.get("target_chunk_index"))
            except (TypeError, ValueError):
                continue
            if source_idx == target_idx:
                continue
            adjacency.setdefault(source_idx, set()).add(target_idx)
            adjacency.setdefault(target_idx, set()).add(source_idx)

        expanded_indices = set()
        for seed_idx in seed_indices:
            for neighbor_idx in sorted(adjacency.get(seed_idx, set())):
                expanded_indices.add(neighbor_idx)
                add_raw_candidate(neighbor_idx, source="graph_neighbor")

        raw_candidates = sorted(
            raw_candidate_map.values(),
            key=lambda item: (item["similarity_score"], len(item["selection_sources"])),
            reverse=True,
        )
        if self.graph_max_raw_candidates > 0:
            raw_candidates = raw_candidates[: self.graph_max_raw_candidates]
        for embedding_rank, candidate in enumerate(raw_candidates, 1):
            candidate["embedding_rank"] = embedding_rank

        return await self._finish_candidate_search(
            query=query,
            conversation_id=conversation_id,
            conv_index=conv_index,
            candidates=raw_candidates,
            question_id=question_id,
            retrieval_metadata_extra={
                "embedding_candidate_source": "graph_expanded_raw",
                "embedding_seed_count": len(embedding_seeds),
                "graph_neighbor_count": len(expanded_indices - seed_indices),
                "graph_edge_count": len(conv_index.get("graph_edges", [])),
                "raw_candidate_count": len(raw_candidates),
                "raw_chunk_count": len(raw_chunks),
                "graph_search_embedding_top_k": self.embedding_top_k,
                "graph_link_embedding_top_k": self.graph_link_embedding_top_k,
            },
        )

    async def _finish_candidate_search(
        self,
        *,
        query: str,
        conversation_id: str,
        conv_index: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        question_id: Optional[str],
        retrieval_metadata_extra: Dict[str, Any],
    ) -> SearchResult:
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

        results = []
        for rerank_rank, candidate in enumerate(reranked, 1):
            chunk = candidate["chunk"]
            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "memory_type": chunk.get("memory_type"),
                "source_chunk_id": chunk.get("source_chunk_id"),
                "source_chunk_index": chunk.get("source_chunk_index"),
                "start_turn": chunk.get("start_turn"),
                "end_turn": chunk.get("end_turn"),
                "start_token": chunk.get("start_token"),
                "end_token": chunk.get("end_token"),
                "embedding_rank": candidate.get("embedding_rank"),
                "rerank_rank": rerank_rank,
                "similarity_score": candidate.get("similarity_score", 0.0),
                "rerank_score": candidate.get("rerank_score"),
                "selection_sources": candidate.get("selection_sources", []),
                "graph_cues": candidate.get("graph_cues", []),
            }
            results.append(
                {
                    "content": candidate.get("content", ""),
                    "score": float(candidate.get("rerank_score", candidate.get("similarity_score", 0.0))),
                    "metadata": metadata,
                }
            )

        formatted_context = self._format_memory_context(results, conv_index)
        retrieval_metadata = {
            "system": self.system_name,
            "memory_variant": self.memory_variant,
            "formatted_context": formatted_context,
            "top_k": len(results),
            "embedding_top_k": self.embedding_top_k,
            "rerank_top_k": self.rerank_top_k,
            "candidate_count": len(candidates),
            "all_rerank_results": self._build_all_memory_rerank_results(all_reranked),
            "question_id": question_id,
            "dynamic_top_k": selection_metadata,
            "storage_backend": self.storage_backend,
            "rerank_status": "completed" if self.rerank_enabled and candidates else (
                "no_candidates" if self.rerank_enabled else "disabled"
            ),
        }
        retrieval_metadata.update(retrieval_metadata_extra)
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=retrieval_metadata,
            question_id=question_id or "",
        )

    def _get_embedding_candidates(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        query_embedding: List[float],
        *,
        top_k: int,
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

        candidates = []
        for rank, item in enumerate(scored[: min(top_k, len(scored))], 1):
            chunk = chunks[item["index"]]
            candidates.append(
                {
                    "index": item["index"],
                    "content": chunk.get("content", ""),
                    "chunk": chunk,
                    "embedding_rank": rank,
                    "similarity_score": item["similarity_score"],
                }
            )
        return candidates

    def _format_memory_context(
        self,
        results: List[Dict[str, Any]],
        conv_index: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not results:
            return "(No relevant memories found)"

        parts = []
        for idx, result in enumerate(results, 1):
            metadata = result.get("metadata", {})
            display_content = result.get("content", "")
            if (
                self.memory_variant == "graph_expanded_raw"
                and self.include_graph_cues_in_context
                and metadata.get("graph_cues")
            ):
                cues = "\n".join(
                    f"- {cue.get('graph_memory_type')}: {cue.get('graph_content')}"
                    for cue in metadata.get("graph_cues", [])[:3]
                )
                display_content = f"GRAPH CUES:\n{cues}\n\nRAW SOURCE MEMORY:\n{display_content}"
            parts.append(
                f"Memory {idx} "
                f"(type={metadata.get('memory_type')}, "
                f"chunk={metadata.get('chunk_index')}, "
                f"source_chunk={metadata.get('source_chunk_index')}, "
                f"similarity={float(metadata.get('similarity_score') or 0.0):.6f}, "
                f"rerank={metadata.get('rerank_score')}):\n"
                f"{display_content}"
            )
        return "\n\n".join(parts)

    def _build_all_memory_rerank_results(
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
                    "memory_type": chunk.get("memory_type"),
                    "source_chunk_id": chunk.get("source_chunk_id"),
                    "source_chunk_index": chunk.get("source_chunk_index"),
                    "embedding_rank": candidate.get("embedding_rank"),
                    "rerank_rank": rerank_rank,
                    "similarity_score": candidate.get("similarity_score"),
                    "rerank_score": candidate.get("rerank_score"),
                    "selection_sources": candidate.get("selection_sources", []),
                }
            )
        return all_results

    def _build_index_metadata(self) -> Dict[str, Any]:
        metadata = super()._build_index_metadata()
        metadata.update(
            {
                "adapter": "rag_memory_ablation",
                "prompt_version": self.PROMPT_VERSION,
                "memory_variant": self.memory_variant,
                "memory_model": self.memory_model,
                "memory_json_response": self.memory_json_response,
                "atomic_max_facts_per_chunk": self.atomic_max_facts_per_chunk,
                "atomic_max_fact_words": self.atomic_max_fact_words,
                "atomic_max_quote_words": self.atomic_max_quote_words,
                "atomic_max_entities": self.atomic_max_entities,
            }
        )
        if self.memory_variant == "graph_expanded_raw":
            metadata.update(
                {
                    "chunk_graph_version": self.CHUNK_GRAPH_VERSION,
                    "graph_link_embedding_top_k": self.graph_link_embedding_top_k,
                    "graph_search_embedding_top_k": self.embedding_top_k,
                    "graph_search_rerank_top_k": self.rerank_top_k,
                    "graph_max_raw_candidates": self.graph_max_raw_candidates,
                }
            )
        return metadata

    def _load_resumeable_index(self) -> Dict[str, Any]:
        index = {
            "type": "rag_memory_ablation",
            "system": self.system_name,
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
                f"Loaded cached memory index: {index_path} ({len(cached_convs)} conversations)",
                style="yellow",
            )
            return index
        except Exception as exc:
            self.console.print(
                f"Failed to reuse cached memory index at {index_path}: {exc}. Rebuilding.",
                style="yellow",
            )
            return index

    def _build_serializable_index(
        self,
        index: Dict[str, Any],
        persist_embeddings: bool,
    ) -> Dict[str, Any]:
        serialized = super()._build_serializable_index(index, persist_embeddings=persist_embeddings)
        if persist_embeddings:
            return serialized
        for conv_index in serialized.get("conversations", {}).values():
            if isinstance(conv_index, dict):
                conv_index.pop("raw_embeddings", None)
        return serialized

    def _is_conversation_index_reusable(
        self,
        conv_index: Optional[Dict[str, Any]],
        allow_missing_embeddings: bool = False,
    ) -> bool:
        if conv_index and conv_index.get("memory_build_status") not in (None, "complete"):
            return False
        if not super()._is_conversation_index_reusable(conv_index, allow_missing_embeddings):
            return False
        if not conv_index:
            return False
        if conv_index.get("memory_variant") != self.memory_variant:
            return False
        raw_chunks = conv_index.get("raw_chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            return False
        if self.memory_variant == "graph_expanded_raw":
            if not isinstance(conv_index.get("graph_edges"), list):
                return False
            raw_embeddings = conv_index.get("raw_embeddings")
            if not allow_missing_embeddings and (
                not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(raw_chunks)
            ):
                return False
        return True

    async def close(self) -> None:
        if not self._memory_client_is_answer_client:
            close = getattr(self.memory_client, "close", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        await super().close()

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": self.system_name,
            "type": "local_rag_memory_ablation",
            "description": f"Local RAG reverse ablation using {self.memory_variant}",
            "adapter": "RAGMemoryAblationAdapter",
            "memory_variant": self.memory_variant,
        }
