from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.scripts.analyze_meme_relation_retention import (
    add_relation_complete,
    bootstrap_retention,
)


def test_relation_complete_uses_task_specific_structure() -> None:
    frame = pd.DataFrame(
        {
            "task_type": ["Tr", "Cas", "Cas", "Abs"],
            "order_support": [1, 0, 0, 0],
            "dependency_rule_support": [0, 1, 1, 1],
            "trigger_support": [0, 1, 0, 1],
        }
    )
    result = add_relation_complete(frame)
    assert result["relation_complete"].tolist() == [True, True, False, True]


def test_episode_bootstrap_preserves_identical_systems() -> None:
    eligible = np.array([[2, 2], [3, 3], [1, 1]])
    retained = np.array([[1, 1], [2, 2], [0, 0]])
    draws = bootstrap_retention(eligible, retained, 1000, 17, batch_size=100)
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
