import numpy as np
import pytest

from evaluation.src.adapters.memu_gpu_index import (
    ExactGpuItemIndex,
    GpuItemIndexCache,
)


def test_torch_gpu_exact_cosine_search():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    index = ExactGpuItemIndex(
        ["horizontal", "vertical", "diagonal"],
        np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        backend="torch_gpu",
        device=0,
    )

    hits = index.search([1.0, 0.0], 2)

    assert [item_id for item_id, _score in hits] == ["horizontal", "diagonal"]
    assert hits[0][1] == pytest.approx(1.0)


def test_gpu_cache_loads_each_user_scope_once():
    calls = []

    def loader(dsn, user_id, fetch_size):
        calls.append((dsn, user_id, fetch_size))
        return ["only"], np.asarray([[1.0, 0.0]], dtype=np.float32)

    cache = GpuItemIndexCache(
        dsn="postgresql://test",
        backend="torch_gpu",
        fetch_size=64,
        loader=loader,
    )

    assert cache.search([1.0, 0.0], 1, {"user_id": "user-1"})
    assert cache.search([0.0, 1.0], 1, {"user_id": "user-1"})
    assert calls == [("postgresql://test", "user-1", 64)]
    assert cache.search([1.0, 0.0], 1, {"other": "unsafe"}) is None


def test_faiss_gpu_has_actionable_error_when_unavailable():
    with pytest.raises(RuntimeError, match="faiss_gpu requested"):
        ExactGpuItemIndex(
            ["only"],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            backend="faiss_gpu",
            device=0,
        )
