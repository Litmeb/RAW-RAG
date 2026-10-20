import numpy as np
import pandas as pd

from evaluation.scripts.analyze_meme_four_audits import (
    cohen_kappa,
    criterion_validity_rows,
    holm_adjust,
    paired_stats,
)


def test_cohen_kappa_perfect_and_chance_agreement():
    values = np.array([True, False, True, False])
    assert cohen_kappa(values, values) == 1.0
    left = np.array([True, True, False, False])
    right = np.array([True, False, True, False])
    assert cohen_kappa(left, right) == 0.0


def test_paired_stats_preserves_direction_and_counts():
    frame = pd.DataFrame(
        {
            "conversation_id": ["e1", "e1", "e2", "e2"],
            "left": [True, True, False, False],
            "right": [True, False, True, True],
        }
    )
    result = paired_stats(frame, "left", "right", samples=200, seed=7)
    assert result["valid_n"] == 4
    assert result["both_positive"] == 1
    assert result["left_only"] == 1
    assert result["right_only"] == 2
    assert result["both_negative"] == 0
    assert result["right_minus_left"] == 0.25


def test_paired_stats_drops_missing_pairs():
    frame = pd.DataFrame(
        {
            "conversation_id": ["e1", "e2", "e3"],
            "left": pd.Series([True, False, True], dtype="boolean"),
            "right": pd.Series([True, pd.NA, False], dtype="boolean"),
        }
    )
    result = paired_stats(frame, "left", "right", samples=100, seed=3)
    assert result["valid_n"] == 2
    assert result["agreement"] == 0.5


def test_holm_adjust_is_monotone_in_sorted_order():
    raw = [0.04, 0.001, 0.02]
    adjusted = holm_adjust(raw)
    ordered = sorted(zip(raw, adjusted))
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)
    assert all(adjusted[index] >= raw[index] for index in range(len(raw)))


def test_criterion_validity_reports_positive_accuracy_lift():
    rows = []
    for system in ("a", "b"):
        rows.extend(
            [
                {
                    "system": system,
                    "conversation_id": "e1",
                    "correct": 1,
                    "det_result_full": True,
                    "llm_result_excerpt": True,
                    "det_path_full": True,
                    "llm_path_combined": True,
                },
                {
                    "system": system,
                    "conversation_id": "e2",
                    "correct": 0,
                    "det_result_full": False,
                    "llm_result_excerpt": False,
                    "det_path_full": False,
                    "llm_path_combined": False,
                },
            ]
        )
    result = criterion_validity_rows(pd.DataFrame(rows), samples=100, seed=4)
    assert all(row["accuracy_lift"] == 1.0 for row in result)
