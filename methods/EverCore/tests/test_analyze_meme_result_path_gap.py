import numpy as np
import pandas as pd

from evaluation.scripts.analyze_meme_result_path_gap import (
    aligned_counts,
    bootstrap_metrics,
    estimates,
)


def sample_frame():
    rows = []
    for system in ("a", "b"):
        for episode, values in {
            "e1": [(True, True), (True, False)],
            "e2": [(False, False), (True, True)],
        }.items():
            for index, (value, path) in enumerate(values):
                rows.append(
                    {
                        "system": system,
                        "conversation_id": episode,
                        "question_id": f"{episode}_q{index}",
                        "value_complete": value,
                        "path_sufficient": path,
                    }
                )
    return pd.DataFrame(rows)


def test_counts_and_estimates_preserve_nested_gap():
    counts = aligned_counts(sample_frame(), ["a", "b"], ["e1", "e2"])
    point = estimates(counts)
    np.testing.assert_allclose(point["value_coverage"], [0.75, 0.75])
    np.testing.assert_allclose(point["path_coverage"], [0.5, 0.5])
    np.testing.assert_allclose(point["gap"], [0.25, 0.25])
    np.testing.assert_allclose(point["retention"], [2 / 3, 2 / 3])


def test_shared_bootstrap_keeps_identical_systems_identical():
    counts = aligned_counts(sample_frame(), ["a", "b"], ["e1", "e2"])
    draws = bootstrap_metrics({"deterministic": counts, "llm": counts}, 1000, 9)
    np.testing.assert_array_equal(
        draws["deterministic"]["gap"][:, 0], draws["deterministic"]["gap"][:, 1]
    )
    np.testing.assert_array_equal(
        draws["deterministic"]["retention"], draws["llm"]["retention"]
    )
