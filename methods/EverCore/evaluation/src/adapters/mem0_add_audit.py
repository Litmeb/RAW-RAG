"""Lossless per-call auditing for the local Mem0 ``Memory.add`` pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import time
import traceback
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


AUDIT_SCHEMA_VERSION = 1
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:api_?key|authorization|password|secret|token)(?:$|_)",
    re.IGNORECASE,
)


class Mem0AddAuditWriteError(RuntimeError):
    """Raised when an add result cannot be durably written to its audit log."""


def json_safe(value: Any, *, key: Optional[str] = None) -> Any:
    """Convert SDK values to JSON-compatible data while redacting credentials."""
    if key and _SENSITIVE_KEY.search(str(key)):
        return "<redacted>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(item_key): json_safe(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass
    return repr(value)


def parse_json_response(response: Any) -> Optional[Any]:
    """Best-effort JSON parsing without changing the raw response."""
    if isinstance(response, (dict, list)):
        return json_safe(response)
    if not isinstance(response, str):
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def append_jsonl_fsynced(path: Path, record: Dict[str, Any], *, fsync: bool) -> None:
    """Append one complete UTF-8 JSON line and optionally force it to disk."""
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(record), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.write("\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def _call_payload(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    request_args = list(args)
    request_kwargs = dict(kwargs)
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
        request_args = request_args[1:]
    else:
        request_kwargs.pop("messages", None)
    return {
        "args": json_safe(request_args),
        "kwargs": json_safe(request_kwargs),
        "messages": json_safe(messages),
    }


def _vector_descriptor(value: Any) -> Dict[str, Any]:
    safe_value = json_safe(value)
    encoded = json.dumps(
        safe_value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    dimension = len(value) if isinstance(value, (list, tuple)) else None
    return {
        "dimension": dimension,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_values_omitted": True,
    }


def _search_payload(
    args: tuple[Any, ...], kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    request_args = list(args)
    request_kwargs = dict(kwargs)
    vectors = request_kwargs.pop("vectors", None)
    if vectors is None and len(request_args) > 1:
        vectors = request_args[1]
        request_args[1] = "<omitted-vector>"
    return {
        "args": json_safe(request_args),
        "kwargs": json_safe(request_kwargs),
        "vector": _vector_descriptor(vectors),
    }


def _serialize_search_result(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return json_safe(item)
    return {
        "id": json_safe(getattr(item, "id", None)),
        "score": json_safe(getattr(item, "score", None)),
        "payload": json_safe(getattr(item, "payload", None)),
    }


def _history_connection(memory: Any) -> Any:
    db = getattr(memory, "db", None)
    return getattr(db, "connection", None)


def _history_max_rowid(memory: Any) -> Optional[int]:
    connection = _history_connection(memory)
    if connection is None:
        return None
    try:
        row = connection.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM history"
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


def _history_since(memory: Any, rowid: Optional[int]) -> list[Dict[str, Any]]:
    connection = _history_connection(memory)
    if connection is None or rowid is None:
        return []
    rows = connection.execute(
        """
        SELECT rowid, id, memory_id, old_memory, new_memory, event,
               created_at, updated_at, is_deleted, actor_id, role
        FROM history
        WHERE rowid > ?
        ORDER BY rowid ASC
        """,
        (rowid,),
    ).fetchall()
    return [
        {
            "rowid": row[0],
            "id": row[1],
            "memory_id": row[2],
            "old_memory": row[3],
            "new_memory": row[4],
            "event": row[5],
            "created_at": row[6],
            "updated_at": row[7],
            "is_deleted": bool(row[8]),
            "actor_id": row[9],
            "role": row[10],
        }
        for row in rows
    ]


def _finish_llm_call(call: Dict[str, Any], response: Any) -> None:
    call["raw_response"] = json_safe(response)
    parsed = parse_json_response(response)
    call["parsed_response"] = parsed
    if not isinstance(parsed, dict):
        return
    if call["stage"] == "fact_extraction":
        call["extracted_facts"] = json_safe(parsed.get("facts", []))
    elif call["stage"] == "memory_action_decision":
        call["proposed_actions"] = json_safe(parsed.get("memory", []))


def run_audited_add(
    memory: Any,
    *,
    add_kwargs: Dict[str, Any],
    context: Dict[str, Any],
    record: Dict[str, Any],
) -> Any:
    """Run one synchronous ``Memory.add`` while capturing its internal calls."""
    started = time.perf_counter()
    record.update(
        {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "operation_id": str(uuid.uuid4()),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "context": json_safe(context),
            "add_input": json_safe(add_kwargs),
            "llm_calls": [],
            "vector_searches": [],
        }
    )
    history_rowid = _history_max_rowid(memory)
    record["history_rowid_before"] = history_rowid

    llm = getattr(memory, "llm", None)
    vector_store = getattr(memory, "vector_store", None)
    original_generate: Optional[Callable[..., Any]] = getattr(
        llm, "generate_response", None
    )
    original_search: Optional[Callable[..., Any]] = getattr(
        vector_store, "search", None
    )

    def audited_generate(*args: Any, **kwargs: Any) -> Any:
        call_index = len(record["llm_calls"])
        stage = {
            0: "fact_extraction",
            1: "memory_action_decision",
        }.get(call_index, f"llm_call_{call_index + 1}")
        call = {
            "index": call_index,
            "stage": stage,
            "request": _call_payload(args, kwargs),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record["llm_calls"].append(call)
        call_started = time.perf_counter()
        try:
            response = original_generate(*args, **kwargs)
        except BaseException as error:
            call["status"] = "error"
            call["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            raise
        else:
            call["status"] = "success"
            _finish_llm_call(call, response)
            return response
        finally:
            call["finished_at"] = datetime.now(timezone.utc).isoformat()
            call["duration_seconds"] = time.perf_counter() - call_started

    def audited_search(*args: Any, **kwargs: Any) -> Any:
        search = {
            "index": len(record["vector_searches"]),
            "request": _search_payload(args, kwargs),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        record["vector_searches"].append(search)
        search_started = time.perf_counter()
        try:
            results = original_search(*args, **kwargs)
        except BaseException as error:
            search["status"] = "error"
            search["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            raise
        else:
            search["status"] = "success"
            search["results"] = [
                _serialize_search_result(result) for result in (results or [])
            ]
            return results
        finally:
            search["finished_at"] = datetime.now(timezone.utc).isoformat()
            search["duration_seconds"] = time.perf_counter() - search_started

    if callable(original_generate):
        llm.generate_response = audited_generate
    if callable(original_search):
        vector_store.search = audited_search

    try:
        result = memory.add(**deepcopy(add_kwargs))
    except BaseException as error:
        record["status"] = "error"
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    else:
        record["status"] = "success"
        record["add_result"] = json_safe(result)
        return result
    finally:
        if callable(original_generate):
            llm.generate_response = original_generate
        if callable(original_search):
            vector_store.search = original_search
        try:
            record["persisted_history_rows"] = _history_since(
                memory, history_rowid
            )
        except Exception as history_error:
            record["history_capture_error"] = {
                "type": type(history_error).__name__,
                "message": str(history_error),
                "traceback": traceback.format_exc(),
            }
        record["normalized_facts_searched"] = [
            search.get("request", {}).get("kwargs", {}).get("query")
            or (
                search.get("request", {}).get("args", [None])[0]
                if search.get("request", {}).get("args")
                else None
            )
            for search in record["vector_searches"]
        ]
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_seconds"] = time.perf_counter() - started
