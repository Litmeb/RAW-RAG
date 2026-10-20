from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "scripts"
    / "analyze_mab_rag_ablation_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("mab_rag_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scope_weights_sum_to_one() -> None:
    for weights in MODULE.SCOPE_WEIGHTS.values():
        assert np.isclose(sum(weights.values()), 1.0)


def test_total_reproduces_published_baseline_macro() -> None:
    task_scores = {
        "Accurate_Retrieval": 0.6785,
        "Test_Time_Learning": 0.4769,
        "Long_Range_Understanding": 0.3174,
        "Conflict_Resolution": 0.46,
    }
    assert np.isclose(sum(task_scores.values()) / 4, 0.4832)


def test_paired_bootstrap_is_deterministic_and_preserves_constant_delta() -> None:
    differences = np.asarray([0.25, 0.25, 0.25])
    first = MODULE.bootstrap_source_mean_differences(
        differences, 2_000, np.random.default_rng(7)
    )
    second = MODULE.bootstrap_source_mean_differences(
        differences, 2_000, np.random.default_rng(7)
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, 0.25)
    assert MODULE.bootstrap_sign_pvalue(first) < 0.01


def test_holm_adjustment_is_monotone_within_scope() -> None:
    rows = [
        {"scope": "Total", "p_bootstrap_two_sided": value}
        for value in (0.01, 0.04, 0.03)
    ]
    for scope in MODULE.SCOPE_ORDER[1:]:
        rows.extend(
            {"scope": scope, "p_bootstrap_two_sided": value}
            for value in (0.2, 0.3, 0.4)
        )
    MODULE.holm_adjust(rows)
    ordered = sorted(
        (row for row in rows if row["scope"] == "Total"),
        key=lambda row: row["p_bootstrap_two_sided"],
    )
    adjusted = [row["p_holm_within_scope"] for row in ordered]
    assert adjusted == sorted(adjusted)


def test_combined_table_contains_baseline_and_all_conditions() -> None:
    rows = []
    for condition in MODULE.CONDITION_SUFFIXES:
        for scope in MODULE.SCOPE_ORDER:
            rows.append(
                {
                    "condition": condition,
                    "scope": scope,
                    "baseline_score": 0.5,
                    "variant_score": 0.4,
                    "delta": -0.1,
                    "ci_low": -0.2,
                    "ci_high": -0.01,
                    "p_holm_within_scope": 0.01,
                    "significant_holm_0_05": True,
                }
            )
    table = MODULE.render_combined_table(rows, 20_000)
    assert "Baseline：chunk=256" in table
    assert "display turn dedup" in table
    assert table.count("\n|") == len(MODULE.CONDITION_SUFFIXES) + 3
