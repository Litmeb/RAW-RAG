from __future__ import annotations

import numpy as np

from evaluation.scripts.analyze_mab_utilization_significance import (
    holm_adjust,
    run_bootstrap,
)


def test_holm_adjust_is_monotone_in_sorted_pvalues() -> None:
    raw = [0.03, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    ordered = sorted(zip(raw, adjusted), key=lambda item: item[0])
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)
    assert adjusted[1] == 0.003


def test_paired_bootstrap_is_reproducible_and_preserves_identical_systems() -> None:
    sufficient = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.int8)
    correct = np.array([[1, 1], [0, 0], [1, 1]], dtype=np.int8)
    blocks = [("split/source", sufficient, correct)]
    counts_a, correct_a, draws_a = run_bootstrap(blocks, 1000, 17, batch_size=100)
    counts_b, correct_b, draws_b = run_bootstrap(blocks, 1000, 17, batch_size=100)
    np.testing.assert_array_equal(counts_a, [3, 3])
    np.testing.assert_array_equal(correct_a, [2, 2])
    np.testing.assert_array_equal(draws_a, draws_b)
    np.testing.assert_array_equal(draws_a[:, 0], draws_a[:, 1])
