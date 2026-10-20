"""Exact GPU item search for memU without materializing Python float lists."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any


def _psycopg_dsn(dsn: str) -> str:
    """Convert SQLAlchemy's psycopg DSN to a DSN accepted by psycopg."""
    return str(dsn).replace("postgresql+psycopg://", "postgresql://", 1)


def load_pgvector_matrix(
    dsn: str,
    user_id: str,
    fetch_size: int = 512,
):
    """Stream one user scope into one compact float32 matrix."""
    import numpy as np
    import psycopg

    connection = psycopg.connect(_psycopg_dsn(dsn))
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT count(*), vector_dims(embedding)
                FROM memory_items
                WHERE user_id = %s AND embedding IS NOT NULL
                GROUP BY vector_dims(embedding)
                """,
                (user_id,),
            )
            shapes = cursor.fetchall()
        if not shapes:
            return [], np.empty((0, 0), dtype=np.float32)
        if len(shapes) != 1:
            raise RuntimeError(
                f"Expected one embedding dimension for {user_id}, got {shapes}"
            )

        count, dimensions = (int(shapes[0][0]), int(shapes[0][1]))
        matrix = np.empty((count, dimensions), dtype=np.float32)
        item_ids: list[str] = []
        offset = 0
        with connection.cursor(name="memu_gpu_item_stream") as cursor:
            cursor.itersize = max(1, int(fetch_size))
            cursor.execute(
                """
                SELECT id, vector_send(embedding)
                FROM memory_items
                WHERE user_id = %s AND embedding IS NOT NULL
                ORDER BY id
                """,
                (user_id,),
            )
            while rows := cursor.fetchmany(max(1, int(fetch_size))):
                for item_id, embedding_binary in rows:
                    # pgvector binary layout: int16 dimensions, int16 unused,
                    # then big-endian float32 values. Reading the payload
                    # directly avoids 145k Vector/Python-list conversions.
                    vector = np.frombuffer(
                        embedding_binary,
                        dtype=">f4",
                        count=dimensions,
                        offset=4,
                    )
                    matrix[offset] = vector
                    item_ids.append(str(item_id))
                    offset += 1
        return item_ids, matrix[:offset]
    finally:
        connection.close()


def _normalize_rows_in_place(matrix, chunk_size: int = 8192) -> None:
    import numpy as np

    for start in range(0, len(matrix), chunk_size):
        block = matrix[start : start + chunk_size]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        block /= norms


class ExactGpuItemIndex:
    """A source-scoped exact inner-product index on one CUDA device."""

    def __init__(
        self,
        item_ids: list[str],
        matrix,
        *,
        backend: str,
        device: int,
    ) -> None:
        import numpy as np

        if not item_ids or matrix.size == 0:
            raise ValueError("Cannot build a GPU index from an empty item corpus")
        if len(item_ids) != len(matrix):
            raise ValueError("Item ID count does not match embedding row count")

        self.item_ids = item_ids
        self.backend = str(backend)
        self.device = int(device)
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        _normalize_rows_in_place(matrix)

        if self.backend == "torch_gpu":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch_gpu requires an available CUDA device")
            self._torch = torch
            self._matrix = torch.from_numpy(matrix).to(
                device=f"cuda:{self.device}",
                dtype=torch.float32,
            )
            self._faiss_index = None
            self._faiss_resources = None
        elif self.backend == "faiss_gpu":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError(
                    "faiss_gpu requested but GPU Faiss is not installed. "
                    "Official prebuilt faiss-gpu packages are Linux-only; "
                    "use torch_gpu on native Windows."
                ) from exc
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError(
                    "Installed Faiss has no GPU support; install a faiss-gpu build"
                )
            resources = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(resources, int(matrix.shape[1]))
            index.add(matrix)
            self._torch = None
            self._matrix = None
            self._faiss_index = index
            self._faiss_resources = resources
        else:
            raise ValueError(
                f"Unknown GPU item backend {self.backend!r}; "
                "expected 'torch_gpu' or 'faiss_gpu'"
            )

    def search(self, query_vec, top_k: int) -> list[tuple[str, float]]:
        import numpy as np

        actual_k = min(max(0, int(top_k)), len(self.item_ids))
        if actual_k == 0:
            return []
        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        norm = float(np.linalg.norm(query))
        if norm > 1e-12:
            query /= norm

        if self.backend == "torch_gpu":
            torch = self._torch
            gpu_query = torch.from_numpy(query).to(
                device=f"cuda:{self.device}",
                dtype=torch.float32,
            )
            scores = self._matrix @ gpu_query[0]
            values, indices = torch.topk(scores, k=actual_k)
            score_values = values.cpu().tolist()
            index_values = indices.cpu().tolist()
        else:
            values, indices = self._faiss_index.search(query, actual_k)
            score_values = values[0].tolist()
            index_values = indices[0].tolist()

        return [
            (self.item_ids[int(index)], float(score))
            for score, index in zip(score_values, index_values)
            if int(index) >= 0
        ]


class GpuItemIndexCache:
    """Lazily build and retain one exact GPU index per memU user scope."""

    def __init__(
        self,
        *,
        dsn: str,
        backend: str,
        device: int = 0,
        fetch_size: int = 512,
        loader: Callable[[str, str, int], tuple[list[str], Any]] = load_pgvector_matrix,
    ) -> None:
        self.dsn = dsn
        self.backend = backend
        self.device = int(device)
        self.fetch_size = max(1, int(fetch_size))
        self.loader = loader
        self._indexes: dict[str, ExactGpuItemIndex] = {}
        self._lock = threading.RLock()

    @staticmethod
    def user_id_from_where(where: Mapping[str, Any] | None) -> str | None:
        if not where or set(where) != {"user_id"}:
            return None
        value = where.get("user_id")
        return str(value) if value else None

    def search(
        self,
        query_vec,
        top_k: int,
        where: Mapping[str, Any] | None,
    ) -> list[tuple[str, float]] | None:
        user_id = self.user_id_from_where(where)
        if user_id is None:
            return None
        with self._lock:
            index = self._indexes.get(user_id)
            if index is None:
                item_ids, matrix = self.loader(self.dsn, user_id, self.fetch_size)
                if not item_ids:
                    return []
                index = ExactGpuItemIndex(
                    item_ids,
                    matrix,
                    backend=self.backend,
                    device=self.device,
                )
                self._indexes[user_id] = index
            return index.search(query_vec, top_k)

    def release(self, user_id: str) -> None:
        with self._lock:
            self._indexes.pop(str(user_id), None)

    def clear(self) -> None:
        with self._lock:
            self._indexes.clear()
        if self.backend == "torch_gpu":
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
