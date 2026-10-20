"""MemMachine REST API v2 adapter for the evaluation pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from common_utils.datetime_utils import to_iso_format
from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("memmachine")
class MemMachineAdapter(OnlineAPIAdapter):
    """Adapter for a local MemMachine server using `/api/v2`."""

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        self.base_url = self._normalize_api_v2_url(str(config.get("base_url", "")))
        self.org_id = self._safe_id(str(config.get("org_id", "ragforlocomo")))
        self.project_prefix = self._safe_id(str(config.get("project_prefix", "eval")))
        self.run_name = self._safe_id(str(config.get("run_name", "")))
        self.dataset_name = str(config.get("dataset_name", "")).strip().lower()
        memoryagentbench = config.get("memoryagentbench") or {}
        self.memoryagentbench_split = self._optional_safe_id(
            memoryagentbench.get("split")
        )
        self.memoryagentbench_source = self._optional_safe_id(
            memoryagentbench.get("source")
        )
        if self.dataset_name == "memoryagentbench" and not self.memoryagentbench_source:
            raise ValueError(
                "MemMachine MemoryAgentBench runs require a selected source so "
                "each source receives an isolated project namespace."
            )
        self.memory_types = self._normalize_types(config.get("memory_types", ["episodic"]))
        self.batch_size = int(config.get("batch_size", 10))
        self.max_retries = int(config.get("max_retries", 3))
        self.timeout_seconds = float(config.get("timeout_seconds", 120))
        self.request_interval = float(config.get("request_interval", 0.0))
        add_resume = config.get("add_resume") or {}
        self.add_resume_enabled = bool(add_resume.get("enabled", True))
        state_file = Path(
            str(add_resume.get("state_file", "memmachine_add_resume.json"))
        )
        self.add_resume_state_path = (
            state_file if state_file.is_absolute() else self.output_dir / state_file
        )
        self._add_resume_lock = asyncio.Lock()
        self._add_resume_state = self._load_add_resume_state()
        self.expand_context = int((config.get("search", {}) or {}).get("expand_context", 0))
        self.agent_mode = bool((config.get("search", {}) or {}).get("agent_mode", False))
        self.score_threshold = (config.get("search", {}) or {}).get("score_threshold")

        self._session: Optional[aiohttp.ClientSession] = None
        self._add_url = f"{self.base_url}/memories"
        self._search_url = f"{self.base_url}/memories/search"
        self._health_url = f"{self.base_url}/health"

        print(f"   MemMachine API: {self.base_url}")
        print(f"   Org ID: {self.org_id}")
        print(f"   Project prefix: {self._project_prefix_for_log()}")
        print(f"   Memory types: {self.memory_types}")
        if self.add_resume_enabled:
            print(f"   Add Resume: {self.add_resume_state_path}")

    @staticmethod
    def _normalize_api_v2_url(base_url: str) -> str:
        url = (base_url or "http://localhost:8080").rstrip("/")
        if url.endswith("/api/v2"):
            return url
        return f"{url}/api/v2"

    @staticmethod
    def _safe_id(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_:-]+", "_", str(value).strip())
        return cleaned.strip("_:-") or "default"

    @classmethod
    def _optional_safe_id(cls, value: Any) -> str:
        if value is None or not str(value).strip():
            return ""
        return cls._safe_id(str(value))

    @staticmethod
    def _normalize_types(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            values = [value]
        else:
            values = list(value)
        out = []
        for item in values:
            normalized = str(item).strip().lower()
            if normalized and normalized not in out:
                out.append(normalized)
        return out

    def _project_prefix_for_log(self) -> str:
        return "__".join(self._project_scope_parts())

    def _project_scope_parts(self) -> List[str]:
        parts = [self.project_prefix]
        if self.run_name:
            parts.append(self.run_name)
        if self.dataset_name == "memoryagentbench":
            parts.append("mab")
            if self.memoryagentbench_split:
                parts.append(self.memoryagentbench_split)
            parts.append(self.memoryagentbench_source)
        return parts

    def _project_id(self, conversation_id: str) -> str:
        parts = self._project_scope_parts()
        parts.append(self._safe_id(conversation_id))
        return "__".join(parts)

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _add_resume_scope(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "project_scope": self._project_scope_parts(),
            "memory_types": self.memory_types,
            "batch_size": self.batch_size,
        }

    def _empty_add_resume_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "adapter": "memmachine",
            "scope": self._add_resume_scope(),
            "conversations": {},
        }

    def _load_add_resume_state(self) -> Dict[str, Any]:
        if not self.add_resume_enabled:
            return self._empty_add_resume_state()
        pending_path = self.add_resume_state_path.with_suffix(
            self.add_resume_state_path.suffix + ".tmp"
        )
        if not self.add_resume_state_path.exists():
            if not pending_path.exists():
                return self._empty_add_resume_state()
            pending = self._read_add_resume_state(pending_path)
            self._replace_file_with_retry(pending_path, self.add_resume_state_path)
            return pending

        state = self._read_add_resume_state(self.add_resume_state_path)
        if pending_path.exists():
            pending = self._read_add_resume_state(pending_path)
            if self._checkpoint_is_at_least(pending, state):
                self._replace_file_with_retry(
                    pending_path, self.add_resume_state_path
                )
                return pending
        return state

    def _read_add_resume_state(self, path: Path) -> Dict[str, Any]:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load MemMachine Add checkpoint {path}: {exc}"
            ) from exc

        if (
            state.get("version") != 1
            or state.get("adapter") != "memmachine"
            or not isinstance(state.get("conversations"), dict)
        ):
            raise RuntimeError(
                f"Unsupported MemMachine Add checkpoint: {path}"
            )
        if state.get("scope") != self._add_resume_scope():
            raise RuntimeError(
                "MemMachine Add checkpoint scope does not match the current "
                f"configuration: {path}"
            )
        return state

    @staticmethod
    def _checkpoint_is_at_least(
        candidate: Dict[str, Any], current: Dict[str, Any]
    ) -> bool:
        candidate_conversations = candidate["conversations"]
        for conv_id, entry in current["conversations"].items():
            candidate_entry = candidate_conversations.get(conv_id)
            if candidate_entry is None:
                return False
            if candidate_entry.get("completed_messages", -1) < entry.get(
                "completed_messages", -1
            ):
                return False
        return True

    @staticmethod
    def _replace_file_with_retry(source: Path, target: Path) -> None:
        for attempt in range(10):
            try:
                os.replace(source, target)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(min(0.05 * (2**attempt), 1.0))

    @staticmethod
    def _conversation_digest(conv: Conversation) -> str:
        payload = {
            "conversation_id": conv.conversation_id,
            "messages": [
                {
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name,
                    "content": message.content,
                    "timestamp": (
                        to_iso_format(message.timestamp) if message.timestamp else None
                    ),
                    "metadata": message.metadata,
                }
                for message in conv.messages
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _add_resume_entry(
        self, conv: Conversation, message_count: int
    ) -> Dict[str, Any]:
        conversations = self._add_resume_state["conversations"]
        digest = self._conversation_digest(conv)
        entry = conversations.get(conv.conversation_id)
        if entry is None:
            entry = {
                "message_count": message_count,
                "message_sha256": digest,
                "completed_messages": 0,
            }
            conversations[conv.conversation_id] = entry
        elif (
            entry.get("message_count") != message_count
            or entry.get("message_sha256") != digest
        ):
            raise RuntimeError(
                "MemMachine Add checkpoint does not match conversation "
                f"{conv.conversation_id}"
            )

        completed = entry.get("completed_messages")
        if not isinstance(completed, int) or not 0 <= completed <= message_count:
            raise RuntimeError(
                "Invalid completed_messages for conversation "
                f"{conv.conversation_id} in {self.add_resume_state_path}"
            )
        if completed != message_count and completed % self.batch_size:
            raise RuntimeError(
                "MemMachine Add checkpoint stops inside a request batch for "
                f"conversation {conv.conversation_id}"
            )
        return entry

    async def _save_add_resume_state(self) -> None:
        if not self.add_resume_enabled:
            return
        async with self._add_resume_lock:
            self.add_resume_state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.add_resume_state_path.with_suffix(
                self.add_resume_state_path.suffix + ".tmp"
            )
            text = json.dumps(
                self._add_resume_state, ensure_ascii=False, indent=2
            )
            await asyncio.to_thread(temp_path.write_text, text, encoding="utf-8")
            await asyncio.to_thread(
                self._replace_file_with_retry,
                temp_path,
                self.add_resume_state_path,
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def prepare(self, conversations: List[Conversation], **kwargs: Any) -> None:
        del conversations, kwargs
        try:
            await self._request_json_with_retry("get", self._health_url)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"MemMachine health check failed at {self._health_url}: {exc}"
            ) from exc

    async def _request_json_with_retry(
        self, method: str, url: str, **kwargs: Any
    ) -> Dict[str, Any]:
        session = await self._get_session()
        request = getattr(session, method)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs.setdefault("headers", self._headers())
                async with request(url, **kwargs) as resp:
                    text = await resp.text(encoding="utf-8")
                    if resp.status >= 400:
                        raise RuntimeError(
                            f"{method.upper()} {url} -> {resp.status}: {text[:800]}"
                        )
                    if not text:
                        return {}
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return await resp.json(content_type=None)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(2**attempt, 8))
                    continue
                raise
        raise last_exc or RuntimeError("request failed")

    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        del speaker_a, speaker_b
        return False

    def _conversation_to_messages(
        self,
        conversation: Conversation,
        format_type: str = "basic",
        perspective: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        del format_type, perspective

        messages: List[Dict[str, Any]] = []
        for idx, msg in enumerate(conversation.messages):
            metadata = {
                "conversation_id": conversation.conversation_id,
                "message_index": str(idx),
                "source_speaker": msg.sender_name,
            }
            if msg.timestamp:
                metadata["source_timestamp"] = to_iso_format(msg.timestamp)
            for key, value in (msg.metadata or {}).items():
                if value is not None and key not in metadata:
                    metadata[str(key)] = str(value)

            payload: Dict[str, Any] = {
                "producer": msg.sender_name or msg.sender_id or "user",
                "timestamp": (
                    to_iso_format(msg.timestamp)
                    if msg.timestamp
                    else to_iso_format(datetime.now(timezone.utc))
                ),
                "content": msg.content,
                "metadata": metadata,
            }
            if msg.sender_id:
                payload["produced_for"] = msg.sender_id
            messages.append(payload)
        return messages

    async def _add_user_messages(
        self,
        conv: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        **kwargs: Any,
    ) -> Any:
        del speaker
        progress = kwargs.get("progress")
        task_id = kwargs.get("task_id")
        project_id = self._project_id(conv.conversation_id)
        resume_entry = (
            self._add_resume_entry(conv, len(messages))
            if self.add_resume_enabled
            else None
        )
        completed_messages = int(
            resume_entry.get("completed_messages", 0) if resume_entry else 0
        )
        episode_ids: List[str] = []
        if progress is not None and task_id is not None and completed_messages:
            progress.update(task_id, advance=completed_messages)

        for start in range(completed_messages, len(messages), self.batch_size):
            batch = messages[start : start + self.batch_size]
            payload: Dict[str, Any] = {
                "org_id": self.org_id,
                "project_id": project_id,
                "messages": batch,
            }
            if self.memory_types:
                payload["types"] = self.memory_types

            response = await self._request_json_with_retry(
                "post", self._add_url, json=payload
            )
            for result in response.get("results") or []:
                episode_id = (result or {}).get("uid")
                if episode_id:
                    episode_ids.append(str(episode_id))
            if resume_entry is not None:
                resume_entry["completed_messages"] = start + len(batch)
                await self._save_add_resume_state()

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)

        return {
            "org_id": self.org_id,
            "project_id": project_id,
            "episode_ids": episode_ids,
        }

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        question_date = kwargs.get("question_date")
        del user_id, kwargs
        fetch_top_k = self._temporal_fetch_top_k(top_k, question_date)
        payload: Dict[str, Any] = {
            "org_id": self.org_id,
            "project_id": self._project_id(conversation_id),
            "query": query,
            "top_k": int(fetch_top_k),
            "expand_context": self.expand_context,
            "agent_mode": self.agent_mode,
        }
        if self.memory_types:
            payload["types"] = self.memory_types
        if self.score_threshold is not None:
            payload["score_threshold"] = float(self.score_threshold)

        data = await self._request_json_with_retry("post", self._search_url, json=payload)
        parsed_results = self._parse_search_results(data)
        # The API returns ranked episodic and semantic lists separately.  Semantic
        # features have no comparable score (we use 0.0 locally), so sorting the
        # combined list and applying one top-k silently drops every semantic item.
        # Apply the episodic budget after the temporal cutoff, keep an independent
        # semantic budget, then concatenate the two result types for the answerer.
        episodic_results = [
            result
            for result in parsed_results
            if (result.get("metadata") or {}).get("memory_type") != "semantic"
        ]
        semantic_results = [
            result
            for result in parsed_results
            if (result.get("metadata") or {}).get("memory_type") == "semantic"
        ]
        episodic_results, temporal_filter_metadata = self._filter_results_for_question_date(
            episodic_results,
            question_date,
            requested_top_k=top_k,
            fetch_top_k=fetch_top_k,
            keep_unknown_timestamps=False,
        )
        semantic_results = semantic_results[: int(top_k)]
        results = episodic_results + semantic_results
        temporal_filter_metadata.update(
            {
                "episodic_result_count": len(episodic_results),
                "semantic_result_count": len(semantic_results),
                "combined_result_count": len(results),
                "semantic_temporal_filter_bypassed": True,
            }
        )
        self._remember_temporal_filter_metadata(
            conversation_id, self._project_id(conversation_id), query, temporal_filter_metadata
        )
        return results

    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = (data or {}).get("content") or {}
        results: List[Dict[str, Any]] = []

        episodic = content.get("episodic_memory") or {}
        long_term = (episodic.get("long_term_memory") or {}).get("episodes") or []
        short_term = (episodic.get("short_term_memory") or {}).get("episodes") or []
        summaries = (episodic.get("short_term_memory") or {}).get("episode_summary") or []

        for episode in list(long_term) + list(short_term):
            item = self._episode_to_result(episode)
            if item:
                results.append(item)

        for idx, summary in enumerate(summaries):
            if not summary:
                continue
            results.append(
                {
                    "content": str(summary),
                    "score": 0.0,
                    "user_id": "",
                    "metadata": {
                        "memory_type": "episodic_summary",
                        "rank": idx,
                        "raw": summary,
                    },
                }
            )

        for feature in content.get("semantic_memory") or []:
            item = self._semantic_to_result(feature)
            if item:
                results.append(item)

        return results

    def _episode_to_result(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = (episode or {}).get("content")
        if not content:
            return None
        metadata = (episode or {}).get("metadata") or {}
        timestamp = (
            metadata.get("source_timestamp")
            or (episode or {}).get("created_at")
            or ""
        )
        speaker = metadata.get("source_speaker") or (episode or {}).get("producer_id") or ""
        prefix_parts = []
        if timestamp:
            prefix_parts.append(f"[{timestamp}]")
        if speaker:
            prefix_parts.append(f"{speaker}:")
        prefix = " ".join(prefix_parts)
        formatted = f"{prefix} {content}".strip()
        score = (episode or {}).get("score")
        return {
            "content": formatted,
            "score": float(score) if score is not None else 0.0,
            "user_id": str((episode or {}).get("producer_id") or ""),
            "metadata": {"memory_type": "episodic", "raw": episode},
        }

    def _semantic_to_result(self, feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        value = (feature or {}).get("value")
        if not value:
            return None
        name = (feature or {}).get("feature_name") or (feature or {}).get("tag") or "fact"
        category = (feature or {}).get("category") or "semantic"
        content = f"{category} / {name}: {value}"
        return {
            "content": content,
            "score": 0.0,
            "user_id": str((feature or {}).get("set_id") or ""),
            "metadata": {"memory_type": "semantic", "raw": feature},
        }

    def _build_formatted_context(self, results: List[Dict[str, Any]], top_k: int) -> str:
        episodic_lines = []
        summary_lines = []
        semantic_lines = []

        for result in results[: int(top_k)]:
            content = (result.get("content") or "").strip()
            if not content:
                continue
            memory_type = (result.get("metadata") or {}).get("memory_type")
            if memory_type == "semantic":
                semantic_lines.append(f"- {content}")
            elif memory_type == "episodic_summary":
                summary_lines.append(content)
            else:
                episodic_lines.append(content)

        sections = []
        if episodic_lines:
            sections.append(
                "<EPISODIC MEMORY>\n" + "\n".join(episodic_lines) + "\n</EPISODIC MEMORY>"
            )
        if summary_lines:
            sections.append(
                "<WORKING MEMORY SUMMARY>\n"
                + "\n".join(summary_lines)
                + "\n</WORKING MEMORY SUMMARY>"
            )
        if semantic_lines:
            sections.append(
                "<SEMANTIC MEMORY>\n" + "\n".join(semantic_lines) + "\n</SEMANTIC MEMORY>"
            )
        return "\n\n".join(sections)

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        del user_id, kwargs
        context_top_k = len(results)
        retrieval_metadata = {
            "system": "memmachine",
            "org_id": self.org_id,
            "project_id": self._project_id(conversation_id),
            "top_k": context_top_k,
            "requested_top_k": int(top_k),
            "memory_types": self.memory_types,
            "expand_context": self.expand_context,
            "agent_mode": self.agent_mode,
            "formatted_context": self._build_formatted_context(results, context_top_k),
        }
        retrieval_metadata.update(
            self._get_temporal_filter_metadata(
                conversation_id, self._project_id(conversation_id), query
            )
        )
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=retrieval_metadata,
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
        **kwargs: Any,
    ) -> SearchResult:
        del results_a, results_b, speaker_a, speaker_b, speaker_b_user_id
        return self._build_single_search_result(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            user_id=speaker_a_user_id,
            top_k=top_k,
            **kwargs,
        )

    def _get_answer_prompt(self) -> str:
        from evaluation.src.adapters.evermemos.prompts.answer_prompts import ANSWER_PROMPT

        return ANSWER_PROMPT
