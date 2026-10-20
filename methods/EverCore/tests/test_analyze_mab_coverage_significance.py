from __future__ import annotations

import numpy as np

from evaluation.scripts.analyze_mab_coverage_significance import (
    bootstrap_coverage,
    cochran_q,
)


def test_cochran_q_is_zero_for_equal_column_totals() -> None:
    matrix = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.int8)
    statistic, degrees_of_freedom, pvalue = cochran_q(matrix)
    assert statistic == 0.0
    assert degrees_of_freedom == 1
    assert pvalue == 1.0


def test_bootstrap_coverage_preserves_identical_systems_and_seed() -> None:
    matrix = np.array([[1, 1], [0, 0], [1, 1]], dtype=np.int8)
    blocks = [("split/source", matrix)]
    first = bootstrap_coverage(blocks, 1000, 23, batch_size=100)
    second = bootstrap_coverage(blocks, 1000, 23, batch_size=100)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 0], first[:, 1])
