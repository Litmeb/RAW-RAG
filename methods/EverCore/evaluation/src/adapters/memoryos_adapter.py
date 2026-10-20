"""Adapter for the BAI-LAB MemoryOS HTTP bridge."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.src.adapters.memos_adapter import MemosAdapter
from evaluation.src.adapters.registry import register_adapter


@register_adapter("memoryos")
class MemoryOSAdapter(MemosAdapter):
    """Reuse the benchmark plumbing of Memos with MemoryOS-shaped payloads.

    The official MemoryOS Docker image exposes the Python package, rather than
    a REST API. ``scripts/memoryos/memoryos_http_server.py`` supplies the small
    HTTP bridge consumed here.
    """

    def __init__(self, config: dict, output_dir: Path = None):
        # Let the shared Memos implementation initialize its HTTP/session
        # machinery; the endpoint and payload methods below are overridden.
        shared_config = dict(config)
        shared_config["api_flavor"] = "product"
        super().__init__(shared_config, output_dir)
        self.api_flavor = "memoryos"

        resume_config = config.get("add_resume", {}) or {}
        self.add_resume_enabled = bool(resume_config.get("enabled", True))
        state_file = (
            resume_config.get("state_file")
            or resume_config.get("ckpt_file")
            or "memoryos_local_add_resume.json"
        )
        self.add_resume_state_path = Path(state_file)
        if not self.add_resume_state_path.is_absolute():
            self.add_resume_state_path = self.output_dir / self.add_resume_state_path
        self._add_resume_lock = asyncio.Lock()
        self._add_resume_state = self._load_add_resume_state()

    def _load_add_resume_state(self) -> Dict[str, Any]:
        if not self.add_resume_enabled or not self.add_resume_state_path.exists():
            return {"version": 1, "conversations": {}}
        try:
            with self.add_resume_state_path.open("r", encoding="utf-8") as stream:
                state = json.load(stream)
            if not isinstance(state, dict) or not isinstance(
                state.get("conversations", {}), dict
            ):
                raise ValueError("invalid conversations map")
            return state
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load MemoryOS Add checkpoint {self.add_resume_state_path}: {exc}"
            ) from exc

    @staticmethod
    def _messages_fingerprint(messages: List[Dict[str, Any]]) -> str:
        payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _save_add_resume_state(self) -> None:
        if not self.add_resume_enabled:
            return
        async with self._add_resume_lock:
            self.add_resume_state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.add_resume_state_path.with_suffix(
                self.add_resume_state_path.suffix + ".tmp"
            )
            temp_path.write_text(
                json.dumps(self._add_resume_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.add_resume_state_path)

    async def _mark_add_unit_completed(
        self,
        conv_id: str,
        speaker: str,
        fingerprint: str,
        unit_index: int,
        total_units: int,
    ) -> None:
        if not self.add_resume_enabled:
            return
        async with self._add_resume_lock:
            conversations = self._add_resume_state.setdefault("conversations", {})
            state_key = f"{conv_id}::{speaker}"
            entry = conversations.setdefault(
                state_key,
                {
                    "speaker": speaker,
                    "fingerprint": fingerprint,
                    "completed_units": [],
                },
            )
            if entry.get("speaker") != speaker or entry.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"MemoryOS Add checkpoint does not match conversation {conv_id}"
                )
            completed = entry.setdefault("completed_units", [])
            if unit_index not in completed:
                completed.append(unit_index)
                completed.sort()
            entry["completed"] = len(completed) >= total_units
            self.add_resume_state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.add_resume_state_path.with_suffix(
                self.add_resume_state_path.suffix + ".tmp"
            )
            temp_path.write_text(
                json.dumps(self._add_resume_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(self.add_resume_state_path)

    async def _add_user_messages(
        self, conv, messages: List[Dict[str, Any]], speaker: str, **kwargs
    ) -> Any:
        """Add messages with a durable per-request checkpoint.

        MemoryAgentBench sends one request per source message. For other datasets,
        a unit is one configured batch, so an interrupted request is retried but a
        successfully committed earlier batch is skipped.
        """
        if self._is_memoryagentbench_mode():
            units = [
                triple
                for triple in self._build_memoryagentbench_messages(conv, messages)
            ]
        else:
            units = [
                messages[i : i + self.batch_size]
                for i in range(0, len(messages), self.batch_size)
            ]

        fingerprint = self._messages_fingerprint(messages)
        state_key = f"{conv.conversation_id}::{speaker}"
        entry = self._add_resume_state.get("conversations", {}).get(state_key)
        completed_units = set()
        if entry is not None:
            if entry.get("speaker") != speaker or entry.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"MemoryOS Add checkpoint does not match conversation {conv.conversation_id}"
                )
            completed_units = set(entry.get("completed_units", []))

        user_id = (
            self._get_memoryagentbench_user_id(conv.conversation_id)
            if self._is_memoryagentbench_mode()
            else self._extract_user_id(conv, speaker=speaker)
        )
        sender_name = (
            "MemoryAgentBench"
            if self._is_memoryagentbench_mode()
            else conv.metadata.get(speaker, speaker)
        )
        session = await self._get_session()
        for unit_index, unit in enumerate(units):
            if unit_index in completed_units:
                self._advance_add_progress(
                    kwargs, len(unit) if not self._is_memoryagentbench_mode() else 1
                )
                continue
            await self._send_message_batch(
                url=self._add_url(),
                batch_messages=unit,
                user_id=user_id,
                conv_id=conv.conversation_id,
                sender_name=sender_name,
                session=session,
            )
            await self._mark_add_unit_completed(
                conv.conversation_id,
                speaker,
                fingerprint,
                unit_index,
                len(units),
            )
            self._advance_add_progress(kwargs, len(unit) if not self._is_memoryagentbench_mode() else 1)
        return None

    def _add_url(self) -> str:
        return f"{self.api_url}/memory/add"

    def _search_url(self) -> str:
        return f"{self.api_url}/memory/search"

    def _success_messages(self) -> set[str]:
        return {"Memory added successfully", "Search completed successfully"}

    def _build_add_payload(
        self, batch_messages: List[Dict[str, Any]], user_id: str, conv_id: str
    ) -> Dict[str, Any]:
        return {
            "messages": batch_messages,
            "user_id": user_id,
            "conversation_id": conv_id,
        }

    def _build_search_payload(
        self, query: str, user_id: str, conversation_id: str, top_k: int
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "top_k": top_k,
        }

    def _extract_search_payload_data(
        self, data: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        return data.get("memories", []), [], ""

    def get_system_name(self) -> str:
        return "memoryos"
