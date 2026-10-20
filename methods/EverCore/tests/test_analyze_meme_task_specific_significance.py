from __future__ import annotations

import numpy as np

from evaluation.scripts.analyze_meme_task_specific_significance import (
    bootstrap_metrics,
    omnibus_max_range,
)


def test_episode_bootstrap_preserves_identical_systems() -> None:
    questions = np.array([2, 3])
    sufficient = np.array([[1, 1], [2, 2]])
    sufficient_correct = np.array([[1, 1], [1, 1]])
    coverage, utilization = bootstrap_metrics(
        questions, sufficient, sufficient_correct, 1000, 31, batch_size=100
    )
    np.testing.assert_array_equal(coverage[:, 0], coverage[:, 1])
    np.testing.assert_array_equal(utilization[:, 0], utilization[:, 1])


def test_omnibus_range_is_one_for_identical_draws() -> None:
    estimates = np.array([0.5, 0.5])
    draws = np.tile(np.array([[0.4, 0.4], [0.6, 0.6]]), (500, 1))
    result = omnibus_max_range(estimates, draws)
    assert result["observed_range"] == 0.0
    assert result["p_value"] == 1.0
