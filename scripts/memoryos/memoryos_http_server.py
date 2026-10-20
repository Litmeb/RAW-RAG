"""Minimal REST bridge for the BAI-LAB MemoryOS Docker image.

Endpoints:
  GET  /health
  POST /memory/add     {messages, user_id, conversation_id}
  POST /memory/search  {query, user_id, top_k}
"""

import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from openai import OpenAI


PACKAGE_ROOT = os.environ.get(
    "MEMORYOS_PACKAGE_ROOT", "/app/MemoryOS-main/memoryos-pypi"
)
sys.path.insert(0, PACKAGE_ROOT)
from memoryos import Memoryos  # noqa: E402
import long_term as memoryos_long_term  # noqa: E402
import mid_term as memoryos_mid_term  # noqa: E402
import updater as memoryos_updater  # noqa: E402
import utils as memoryos_utils  # noqa: E402


DATA_ROOT = Path(os.environ.get("MEMORYOS_DATA_DIR", "/app/memoryos_data"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)
INSTANCES = {}
INSTANCE_LOCK = threading.RLock()
USER_LOCKS = {}
_VECTORIZE_MODEL = os.environ.get("VECTORIZE_MODEL", "").strip()
_VECTORIZE_API_KEY = os.environ.get("VECTORIZE_API_KEY", "").strip()
_VECTORIZE_BASE_URL = os.environ.get("VECTORIZE_BASE_URL", "").strip()
_VECTORIZE_CLIENT = None
_VECTORIZE_CACHE = {}
_FAISS_GPU_RESOURCES = None
_FAISS_GPU_ENABLED = False


def _get_vectorize_embedding(text, model_name=None, use_cache=True, **kwargs):
    """Create embeddings with the configured VECTORIZE_* OpenAI endpoint."""
    model = (model_name or _VECTORIZE_MODEL).strip()
    cache_key = (model, text)
    if use_cache and cache_key in _VECTORIZE_CACHE:
        return _VECTORIZE_CACHE[cache_key]

    global _VECTORIZE_CLIENT
    if _VECTORIZE_CLIENT is None:
        _VECTORIZE_CLIENT = OpenAI(
            api_key=_VECTORIZE_API_KEY,
            base_url=_VECTORIZE_BASE_URL or None,
        )
    response = _VECTORIZE_CLIENT.embeddings.create(model=model, input=[text])
    embedding = np.asarray(response.data[0].embedding, dtype=np.float32)
    if use_cache:
        _VECTORIZE_CACHE[cache_key] = embedding
    return embedding


def _configure_embedding_backend():
    """Make MemoryOS' add/update/retrieve paths use VECTORIZE_* when configured."""
    if not (_VECTORIZE_MODEL and _VECTORIZE_API_KEY):
        return

    # MemoryOS imports get_embedding into these modules at import time, so patch
    # each bound reference as well as the utility module reference.
    memoryos_utils.get_embedding = _get_vectorize_embedding
    memoryos_long_term.get_embedding = _get_vectorize_embedding
    memoryos_mid_term.get_embedding = _get_vectorize_embedding
    memoryos_updater.get_embedding = _get_vectorize_embedding


_configure_embedding_backend()


def _configure_faiss_gpu():
    """Make MemoryOS' ``IndexFlatIP`` instances execute on the GPU."""
    global _FAISS_GPU_RESOURCES, _FAISS_GPU_ENABLED
    if os.environ.get("MEMORYOS_USE_GPU", "true").strip().lower() not in {
        "1", "true", "yes", "on"
    }:
        return
    try:
        import faiss

        if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
            raise RuntimeError("FAISS GPU support is unavailable or no GPU is visible")
        _FAISS_GPU_RESOURCES = faiss.StandardGpuResources()
        cpu_index_flat_ip = faiss.IndexFlatIP

        def gpu_index_flat_ip(dimension):
            cpu_index = cpu_index_flat_ip(dimension)
            return faiss.index_cpu_to_gpu(_FAISS_GPU_RESOURCES, 0, cpu_index)

        faiss.IndexFlatIP = gpu_index_flat_ip
        _FAISS_GPU_ENABLED = True
        print("MemoryOS similarity backend: FAISS GPU", flush=True)
    except Exception as exc:
        raise RuntimeError(
            "MEMORYOS_USE_GPU=true but FAISS GPU initialization failed"
        ) from exc


_configure_faiss_gpu()


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:160] or "default"


def _instance(user_id: str):
    with INSTANCE_LOCK:
        if user_id not in INSTANCES:
            kwargs = {
                "user_id": user_id,
                "openai_api_key": os.environ.get("LLM_API_KEY", ""),
                "openai_base_url": os.environ.get("LLM_BASE_URL") or None,
                "data_storage_path": str(DATA_ROOT / _safe_component(user_id)),
                "assistant_id": os.environ.get("MEMORYOS_ASSISTANT_ID", "memoryagentbench"),
                "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                "embedding_model_name": _VECTORIZE_MODEL or "all-MiniLM-L6-v2",
            }
            INSTANCES[user_id] = Memoryos(**kwargs)
        return INSTANCES[user_id]


def _user_lock(user_id: str):
    """Serialize operations touching one MemoryOS instance only."""
    with INSTANCE_LOCK:
        return USER_LOCKS.setdefault(user_id, threading.RLock())


def _memory_text(item):
    if not isinstance(item, dict):
        return str(item)
    user_input = str(item.get("user_input", "") or "")
    agent_response = str(item.get("agent_response", "") or "")
    return "\n".join(part for part in (user_input, agent_response) if part)


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "system": "memoryos"})
        else:
            self._json(404, {"message": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            user_id = str(payload.get("user_id") or "default")
            memo = _instance(user_id)
            # MemoryOS instances mutate local state during add and may share
            # files/indexes between add and search. Keep one user's operations
            # ordered, while allowing up to the HTTP server's worker capacity
            # across different users.
            with _user_lock(user_id):
                if self.path == "/memory/add":
                    messages = payload.get("messages") or []
                    user_input = "\n".join(
                        str(m.get("content", ""))
                        for m in messages
                        if m.get("role") in {"system", "user"}
                    ).strip()
                    agent_response = "\n".join(
                        str(m.get("content", ""))
                        for m in messages
                        if m.get("role") == "assistant"
                    ).strip() or "Memory added to MemoryOS."
                    if not user_input:
                        raise ValueError("messages must contain user/system content")
                    memo.add_memory(
                        user_input=user_input,
                        agent_response=agent_response,
                        meta_data={"conversation_id": payload.get("conversation_id")},
                    )
                    self._json(200, {"message": "Memory added successfully"})
                    return
                if self.path == "/memory/search":
                    query = str(payload.get("query") or "").strip()
                    if not query:
                        raise ValueError("query is required")
                    top_k = max(1, int(payload.get("top_k", 20)))
                    context = memo.retriever.retrieve_context(
                        user_query=query, user_id=memo.user_id
                    )
                    memories = []
                    for page in (context.get("retrieved_pages") or [])[:top_k]:
                        memories.append(
                            {
                                "memory_value": _memory_text(page),
                                "score": page.get("score", 0.0),
                                "memory_time": page.get("timestamp", ""),
                                "raw": page,
                            }
                        )
                    for item in (context.get("retrieved_user_knowledge") or [])[:top_k]:
                        memories.append(
                            {"memory_value": _memory_text(item), "score": 0.0, "raw": item}
                        )
                    self._json(200, {"message": "Search completed successfully", "data": {"memories": memories[:top_k]}})
                    return
                self._json(404, {"message": "not found"})
        except Exception as exc:
            self._json(500, {"message": f"MemoryOS bridge error: {exc}"})

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    port = int(os.environ.get("MEMORYOS_PORT", "8010"))
    print(f"MemoryOS HTTP bridge listening on 0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
