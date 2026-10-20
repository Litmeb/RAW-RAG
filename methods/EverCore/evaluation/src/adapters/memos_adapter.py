"""
Memos Adapter - adapt Memos online API for evaluation framework.
Reference: https://www.memos.so/
"""

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.datasets.memoryagentbench import MAB_SYSTEM_MESSAGE


@register_adapter("memos")
class MemosAdapter(OnlineAPIAdapter):
    """
    Memos online API adapter.

    Supports:
    - Memory ingestion (supports conversation context)
    - Memory retrieval

    Official API supported parameters:
    - user_id (required) - Format: {conv_id}_{speaker}, already contains session info
    - query (required)
    - memory_limit_number (optional, default 6)

    Note: Does not use conversation_id parameter, as user_id already contains session info

    Config example:
    ```yaml
    adapter: "memos"
    api_url: "${MEMOS_URL}"
    api_key: "${MEMOS_KEY}"
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # Get API configuration
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos API URL is required. Set 'api_url' in config.")
        self.api_url = self.api_url.rstrip("/")
        self.api_flavor = str(config.get("api_flavor", "openmem")).lower()
        if self.api_flavor not in {"openmem", "product"}:
            raise ValueError(
                "Memos api_flavor must be either 'openmem' or 'product', "
                f"got: {self.api_flavor}"
            )

        api_key = config.get("api_key", "")
        require_api_key = bool(config.get("require_api_key", True))
        if require_api_key and not api_key:
            raise ValueError("Memos API key is required. Set 'api_key' in config.")

        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = api_key

        # Retrieval configuration (only keep batch_size and max_retries, other params not supported by official API)
        self.batch_size = config.get("batch_size", 9999)  # Memos supports large batches
        self.max_retries = max(1, int(config.get("max_retries", 5)))
        self.http_timeout_seconds = float(config.get("http_timeout_seconds", 60))
        self.retry_backoff_seconds = max(0.0, float(config.get("retry_backoff_seconds", 2.0)))
        self.retry_backoff_max_seconds = max(
            self.retry_backoff_seconds,
            float(config.get("retry_backoff_max_seconds", 60.0)),
        )
        self.retry_jitter_seconds = max(0.0, float(config.get("retry_jitter_seconds", 1.0)))

        # Rate limiting configuration (default: 10 requests/second)
        requests_per_second = config.get("requests_per_second", 10)
        self.rate_limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)

        # Create aiohttp session (will be initialized on first use)
        self._session: Optional[aiohttp.ClientSession] = None

        self.console = Console()

        print(f"   API URL: {self.api_url}")
        print(f"   API Flavor: {self.api_flavor}")
        print(f"   Rate Limit: {requests_per_second} requests/second (async)")

    def _add_url(self) -> str:
        if self.api_flavor == "product":
            return f"{self.api_url}/product/add"
        return f"{self.api_url}/add/message"

    def _search_url(self) -> str:
        if self.api_flavor == "product":
            return f"{self.api_url}/product/search"
        return f"{self.api_url}/search/memory"

    def _success_messages(self) -> set[str]:
        if self.api_flavor == "product":
            return {"Memory added successfully", "Search completed successfully"}
        return {"ok"}

    def _get_user_id_prefix(self) -> str:
        """
        Scope Memos storage by run name when available.

        This keeps memories from different evaluation runs isolated even when
        they share the same conversation_id and speaker names.
        """
        return str(self.config.get("memos_user_id_prefix") or self.config.get("run_name") or "")

    def _is_memoryagentbench_mode(self) -> bool:
        return str(self.config.get("dataset_name", "")) == "memoryagentbench"

    def _memoryagentbench_source(self) -> str:
        mab_cfg = self.config.get("memoryagentbench", {}) or {}
        return str(mab_cfg.get("source") or "")

    def _is_memoryagentbench_icl_mode(self) -> bool:
        return self._is_memoryagentbench_mode() and self._memoryagentbench_source().startswith("icl_")

    def _get_memoryagentbench_user_id(self, conversation_id: str) -> str:
        """
        Match MemoryAgentBench's official single-user storage style.

        Official mem0 uses one user_id per context/source, not one per speaker.
        We keep the run-name prefix from _apply_user_id_prefix for cloud isolation.
        """
        parts = str(conversation_id).rsplit("_", 1)
        context_id = parts[1] if len(parts) == 2 and parts[1].isdigit() else conversation_id
        mab_cfg = self.config.get("memoryagentbench", {}) or {}
        source = self._sanitize_user_id_component(mab_cfg.get("source") or "memoryagentbench")
        return self._apply_user_id_prefix(f"context_{context_id}_{source}")

    @staticmethod
    def _has_icl_label_mapping(text: str) -> bool:
        return (
            "examples I have learned" in text
            and len(re.findall(r"\blabel\s*:\s*[0-9A-Za-z_-]+\b", text, re.IGNORECASE)) >= 5
        )

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9]+", text.lower())
            if len(token) > 2
            and token
            not in {
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
            overlap = len(query_tokens & chunk_tokens)
            scored_chunks.append((overlap, chunk))

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
        self, conv: Conversation, source_messages: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        triples: List[List[Dict[str, Any]]] = []
        assistant_ack = "I'll make sure to add the content into the memory."
        if self._is_memoryagentbench_icl_mode():
            assistant_ack = (
                "I'll preserve each example and its exact numerical label mapping "
                "verbatim in memory."
            )

        for message in source_messages:
            chat_time = message.get("chat_time")
            triple = [
                {
                    "role": "system",
                    "content": (
                        MAB_SYSTEM_MESSAGE
                        + (
                            "\nFor in-context learning examples, preserve every "
                            "utterance and its exact `label: N` value verbatim."
                            if self._is_memoryagentbench_icl_mode()
                            else ""
                        )
                    ),
                },
                {"role": "user", "content": message.get("content", "")},
                {"role": "assistant", "content": assistant_ack},
            ]
            if chat_time:
                for item in triple:
                    item["chat_time"] = chat_time
            triples.append(triple)

        return triples

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session (lazy initialization).

        Returns:
            aiohttp.ClientSession instance
        """
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.http_timeout_seconds,
                connect=min(30.0, self.http_timeout_seconds),
            )
            self._session = aiohttp.ClientSession(headers=self.headers, timeout=timeout)
        return self._session

    async def close(self):
        """
        Close aiohttp session.

        Should be called when adapter is no longer needed.
        """
        if self._session and not self._session.closed:
            await self._session.close()

    async def _add_user_messages(
        self, conv: Conversation, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """
        Add messages for a single user to Memos.

        Args:
            conv: Original conversation object
            messages: Formatted message list
            speaker: "speaker_a" or "speaker_b"
            **kwargs: Extra parameters

        Returns:
            None
        """
        # Extract user_id and conv_id
        if self._is_memoryagentbench_mode():
            user_id = self._get_memoryagentbench_user_id(conv.conversation_id)
        else:
            user_id = self._extract_user_id(conv, speaker=speaker)
        conv_id = conv.conversation_id

        # Log info
        sender_name = "MemoryAgentBench" if self._is_memoryagentbench_mode() else conv.metadata.get(speaker, speaker)
        self.console.print(
            f"   📤 Adding for {sender_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )

        # Get session
        session = await self._get_session()

        # Send messages in batches with retry
        url = self._add_url()

        if self._is_memoryagentbench_mode():
            for message_triple in self._build_memoryagentbench_messages(conv, messages):
                await self._send_message_batch(
                    url=url,
                    batch_messages=message_triple,
                    user_id=user_id,
                    conv_id=conv_id,
                    sender_name=sender_name,
                    session=session,
                )
                self._advance_add_progress(kwargs, 1)
            return None

        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]

            # Try to send the batch with automatic batch size reduction on token limit error
            await self._send_message_batch(
                url=url,
                batch_messages=batch_messages,
                user_id=user_id,
                conv_id=conv_id,
                sender_name=sender_name,
                session=session,
            )
            self._advance_add_progress(kwargs, len(batch_messages))

        return None

    async def _send_message_batch(
        self,
        url: str,
        batch_messages: List[Dict[str, Any]],
        user_id: str,
        conv_id: str,
        sender_name: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """
        Send a batch of messages to Memos API.

        Handles token limit exceeded errors by automatically splitting batches and
        oversized single messages.

        Args:
            url: API endpoint URL
            batch_messages: Messages to send in this batch
            user_id: User ID
            conv_id: Conversation ID
            sender_name: Sender name (for logging)
            session: aiohttp session
        """
        payload_dict = self._build_add_payload(batch_messages, user_id, conv_id)

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            error_payload = self._parse_error_payload(text)
                            if self._is_token_limit_error(
                                response.status, error_payload, text
                            ):
                                await self._handle_token_limit_error(
                                    url=url,
                                    batch_messages=batch_messages,
                                    user_id=user_id,
                                    conv_id=conv_id,
                                    sender_name=sender_name,
                                    session=session,
                                    error_detail=f"HTTP {response.status}: {text}",
                                )
                                return
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()

                        # Check for token limit exceeded error
                        if self._is_token_limit_error(response.status, result):
                            await self._handle_token_limit_error(
                                url=url,
                                batch_messages=batch_messages,
                                user_id=user_id,
                                conv_id=conv_id,
                                sender_name=sender_name,
                                session=session,
                                error_detail=f"API error: {result}",
                            )
                            return

                        if result.get("message") not in self._success_messages():
                            raise Exception(f"API error: {result}")

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.retry_backoff_max_seconds,
                        self.retry_backoff_seconds * (2**attempt),
                    )
                    if self.retry_jitter_seconds:
                        delay += random.uniform(0, self.retry_jitter_seconds)
                    self.console.print(
                        f"   ⚠️  [{sender_name}] Retry {attempt + 1}/{self.max_retries}: {e}",
                        style="yellow",
                    )
                    await asyncio.sleep(delay)
                else:
                    self.console.print(
                        f"   ❌ [{sender_name}] Failed after {self.max_retries} retries: {e}",
                        style="red",
                    )
                    raise e

    def _build_add_payload(
        self, batch_messages: List[Dict[str, Any]], user_id: str, conv_id: str
    ) -> Dict[str, Any]:
        payload_dict = {
            "messages": batch_messages,
            "user_id": user_id,
            "conversation_id": conv_id,
        }
        if self.api_flavor == "product":
            payload_dict.update(
                {
                    "mem_cube_id": user_id,
                    "writable_cube_ids": [user_id],
                    "session_id": conv_id,
                    "async_mode": self.config.get("add_async_mode", "sync"),
                }
            )
            add_mode = self.config.get("add_mode")
            if add_mode:
                payload_dict["mode"] = add_mode
        return payload_dict

    def _parse_error_payload(self, text: str) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _is_token_limit_error(
        self,
        http_status: int,
        payload: Optional[Dict[str, Any]],
        raw_text: str = "",
    ) -> bool:
        payload = payload or {}
        message = str(payload.get("message", "") or raw_text).lower()
        code = payload.get("code")

        if "input token limit exceeded" in message:
            return True
        if http_status == 429 and "token" in message and "limit" in message:
            return True
        return code in {40302, 40305}

    async def _handle_token_limit_error(
        self,
        url: str,
        batch_messages: List[Dict[str, Any]],
        user_id: str,
        conv_id: str,
        sender_name: str,
        session: aiohttp.ClientSession,
        error_detail: str,
    ) -> None:
        if len(batch_messages) > 1:
            new_batch_size = max(1, len(batch_messages) // 2)
            self.console.print(
                f"   ⚠️  [{sender_name}] Token limit exceeded, splitting batch of {len(batch_messages)} into smaller batches (size={new_batch_size})",
                style="yellow",
            )
            for j in range(0, len(batch_messages), new_batch_size):
                sub_batch = batch_messages[j : j + new_batch_size]
                await self._send_message_batch(
                    url=url,
                    batch_messages=sub_batch,
                    user_id=user_id,
                    conv_id=conv_id,
                    sender_name=sender_name,
                    session=session,
                )
            return

        message = batch_messages[0]
        split_messages = self._split_oversized_message(message)
        if split_messages is None:
            raise Exception(
                f"API error (token limit, single message too large, content length={len(message.get('content', ''))}): {error_detail}"
            )

        self.console.print(
            f"   ⚠️  [{sender_name}] Single message token limit exceeded, splitting one message into {len(split_messages)} smaller messages",
            style="yellow",
        )
        for split_message in split_messages:
            await self._send_message_batch(
                url=url,
                batch_messages=[split_message],
                user_id=user_id,
                conv_id=conv_id,
                sender_name=sender_name,
                session=session,
            )

    def _split_oversized_message(
        self, message: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        content = str(message.get("content", ""))
        split_idx = self._find_split_index(content)
        if split_idx is None:
            return None

        left_content = content[:split_idx].rstrip()
        right_content = content[split_idx:].lstrip()
        if not left_content or not right_content:
            return None

        left_message = message.copy()
        right_message = message.copy()
        left_message["content"] = left_content
        right_message["content"] = right_content
        return [left_message, right_message]

    def _find_split_index(self, content: str) -> Optional[int]:
        if len(content) < 2:
            return None

        midpoint = len(content) // 2
        search_radius = max(100, len(content) // 4)
        start = max(1, midpoint - search_radius)
        end = min(len(content) - 1, midpoint + search_radius)

        preferred_breaks = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
        for delimiter in preferred_breaks:
            idx = content.rfind(delimiter, start, end)
            if idx != -1:
                return idx + len(delimiter)

        return midpoint

    async def _search_single_user(
        self, query: str, conversation_id: str, user_id: str, top_k: int, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memories for a single user (Memos-specific with preference extraction).

        Calls Memos HTTP API and extracts preference information.

        Args:
            query: Query text
            conversation_id: Conversation ID (not used by Memos, user_id contains this info)
            user_id: User ID to search for (format: {conv_id}_{speaker})
            top_k: Number of results to retrieve
            **kwargs: Additional parameters

        Returns:
            List of search results with preference information in metadata

        Note:
            user_id already contains session info (format: {conv_id}_{speaker}).
            Example: user_id="locomo_0_Caroline" uniquely identifies the locomo_0 conversation.
        """
        # Get session
        session = await self._get_session()

        # Prepare HTTP request
        url = self._search_url()
        question_date = kwargs.get("question_date")
        fetch_top_k = self._temporal_fetch_top_k(top_k, question_date)
        payload_dict = self._build_search_payload(
            query=query,
            user_id=user_id,
            conversation_id=conversation_id,
            top_k=fetch_top_k,
        )

        # Call API with retry mechanism
        text_mem_res = []
        pref_string = ""

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                async with self.rate_limiter:
                    async with session.post(url, json=payload_dict) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise Exception(f"HTTP {response.status}: {text}")

                        result = await response.json()
                        if result.get("message") not in self._success_messages():
                            raise Exception(f"API error: {result}")

                        data = result.get("data", {})
                        text_mem_res, pref_mem_res, preference_note = (
                            self._extract_search_payload_data(data)
                        )

                        # Standardize field names: rename memory_value to memory
                        for i in text_mem_res:
                            i.update(
                                {"memory": i.pop("memory_value", i.get("memory", ""))}
                            )

                        # Format preference string
                        explicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "explicit_preference"
                        ]
                        implicit_prefs = [
                            p["preference"]
                            for p in pref_mem_res
                            if p.get("preference_type", "") == "implicit_preference"
                        ]

                        pref_parts = []
                        if explicit_prefs:
                            pref_parts.append(
                                "Explicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(explicit_prefs)
                                )
                            )
                        if implicit_prefs:
                            pref_parts.append(
                                "Implicit Preference:\n"
                                + "\n".join(
                                    f"{i + 1}. {p}"
                                    for i, p in enumerate(implicit_prefs)
                                )
                            )

                        pref_string = "\n".join(pref_parts) + preference_note

                # Success - break retry loop
                break

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    self.console.print(f"❌ Memos search error: {e}", style="red")
                    return []

        # Convert to standard format
        results = []
        for item in text_mem_res:
            created_at = item.get("memory_time") or item.get("create_time", "")
            results.append(
                {
                    "content": item.get("memory", ""),
                    "score": item.get("relativity", item.get("score", 0.0)),
                    "user_id": user_id,
                    "metadata": {
                        "memory_id": item.get("id", ""),
                        "created_at": str(created_at) if created_at else "",
                        "memory_time": item.get("memory_time", ""),
                        "create_time": item.get("create_time", ""),
                        "updated_at": item.get("update_time", item.get("updated_at", "")),
                        "memory_type": item.get("memory_type", ""),
                        "confidence": item.get("confidence", 0.0),
                        "tags": item.get("tags", []),
                        "pref_string": pref_string,  # Store preference for this user
                        "raw": item,
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

    def _build_search_payload(
        self, query: str, user_id: str, conversation_id: str, top_k: int
    ) -> Dict[str, Any]:
        if self.api_flavor == "product":
            return {
                "query": query,
                "user_id": user_id,
                "mem_cube_id": user_id,
                "readable_cube_ids": [user_id],
                "session_id": conversation_id,
                "conversation_id": conversation_id,
                "top_k": top_k,
                "mode": self.config.get("search_mode", "fast"),
                "include_preference": True,
                "pref_top_k": self.config.get("pref_top_k", 6),
                "relativity": self.config.get("relativity", 0),
                "dedup": self.config.get("dedup", "mmr"),
                "rerank": self.config.get("memos_rerank", True),
            }
        return {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
        }

    def _extract_search_payload_data(
        self, data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        if self.api_flavor != "product":
            return (
                data.get("memory_detail_list", []),
                data.get("preference_detail_list", []),
                data.get("preference_note", ""),
            )

        text_mem_res = self._flatten_product_memory_buckets(data.get("text_mem", []))
        pref_mem_res = self._flatten_product_memory_buckets(data.get("pref_mem", []))
        preference_note = data.get("preference_note", "")
        return text_mem_res, pref_mem_res, preference_note

    @staticmethod
    def _product_source_chat_times(metadata: Dict[str, Any]) -> List[str]:
        sources = metadata.get("sources")
        if not isinstance(sources, list):
            return []

        chat_times: List[str] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            chat_time = source.get("chat_time")
            if chat_time:
                chat_times.append(str(chat_time))
        return chat_times

    @staticmethod
    def _flatten_product_memory_buckets(
        buckets: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(buckets, list):
            return []

        flattened: List[Dict[str, Any]] = []
        for bucket in buckets:
            if isinstance(bucket, dict) and isinstance(bucket.get("memories"), list):
                for memory in bucket["memories"]:
                    if not isinstance(memory, dict):
                        continue
                    item = memory.copy()
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    source_chat_times = MemosAdapter._product_source_chat_times(metadata)
                    source_timestamp = source_chat_times[0] if source_chat_times else ""
                    item.setdefault("memory", item.get("memory_value", ""))
                    item.setdefault("id", metadata.get("id", ""))
                    item.setdefault("relativity", metadata.get("relativity", metadata.get("score", 0.0)))
                    item.setdefault("source_timestamp", source_timestamp)
                    item.setdefault("memory_time", metadata.get("memory_time") or source_timestamp)
                    item.setdefault("create_time", metadata.get("created_at", ""))
                    item.setdefault("update_time", metadata.get("updated_at", ""))
                    item.setdefault("memory_type", metadata.get("memory_type", metadata.get("type", "")))
                    item.setdefault("confidence", metadata.get("confidence", 0.0))
                    item.setdefault("tags", metadata.get("tags", []))
                    flattened.append(item)
        return flattened

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
        Build SearchResult for single perspective (Memos: include preference).

        Args:
            query: Query text
            conversation_id: Conversation ID
            results: Search results from _search_single_user
            user_id: User ID
            top_k: Number of results requested
            **kwargs: Additional parameters

        Returns:
            SearchResult with preference metadata (no formatted_context, uses fallback)
        """
        # Extract pref_string from first result's metadata (all results share same pref_string)
        pref_string = results[0]["metadata"]["pref_string"] if results else ""
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
                "system": "memos",
                "preferences": {"pref_string": pref_string},
                "top_k": top_k,
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
        Build SearchResult for dual perspective (Memos: use template + preference).

        Formats memories using the default template, including preference information
        for both speakers.

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
            SearchResult with formatted_context and preferences
        """
        # Extract preferences from results' metadata
        pref_a = results_a[0]["metadata"]["pref_string"] if results_a else ""
        pref_b = results_b[0]["metadata"]["pref_string"] if results_b else ""

        # Build context for each speaker (memories + preferences)
        speaker_a_memories = (
            "\n".join([r["content"] for r in results_a])
            if results_a
            else "(No memories found)"
        )
        speaker_b_memories = (
            "\n".join([r["content"] for r in results_b])
            if results_b
            else "(No memories found)"
        )

        speaker_a_context = speaker_a_memories + (f"\n{pref_a}" if pref_a else "")
        speaker_b_context = speaker_b_memories + (f"\n{pref_b}" if pref_b else "")

        # Use default template
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_context,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_context,
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
                "system": "memos",
                "dual_perspective": True,
                "formatted_context": formatted_context,
                "top_k": top_k,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "preferences": {"speaker_a_pref": pref_a, "speaker_b_pref": pref_b},
                **temporal_filter_metadata,
            },
        )

    def _get_answer_prompt(self) -> str:
        """
        Get answer prompt.

        Subclasses can override this method to return their own prompt.
        Defaults to generic default prompt.
        """
        return self._prompts["online_api"]["default"]["answer_prompt_memos"]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "Memos",
            "type": "online_api",
            "description": "Memos - Memory System with Preference Support",
            "adapter": "MemosAdapter",
        }
