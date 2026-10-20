from pathlib import Path

import pytest

from evaluation.src.adapters.memmachine_adapter import MemMachineAdapter


def _config(**overrides):
    config = {
        "base_url": "http://localhost:8080/api/v2",
        "org_id": "ragforlocomo",
        "project_prefix": "eval",
        "run_name": "normal",
        "memory_types": ["episodic", "semantic"],
        "llm": {
            "provider": "openai",
            "model": "test-model",
            "api_key": "test-api-key",
        },
    }
    config.update(overrides)
    return config


def test_memoryagentbench_project_id_is_scoped_by_split_and_source(tmp_path: Path):
    banking = MemMachineAdapter(
        _config(
            dataset_name="memoryagentbench",
            memoryagentbench={
                "split": "Test_Time_Learning",
                "source": "icl_banking77_5900shot_balance",
            },
        ),
        output_dir=tmp_path,
    )
    clinic = MemMachineAdapter(
        _config(
            dataset_name="memoryagentbench",
            memoryagentbench={
                "split": "Test_Time_Learning",
                "source": "icl_clinic150_7050shot_balance",
            },
        ),
        output_dir=tmp_path,
    )

    banking_id = banking._project_id("memoryagentbench_0")
    clinic_id = clinic._project_id("memoryagentbench_0")

    assert banking_id == (
        "eval__normal__mab__Test_Time_Learning__"
        "icl_banking77_5900shot_balance__memoryagentbench_0"
    )
    assert clinic_id == (
        "eval__normal__mab__Test_Time_Learning__"
        "icl_clinic150_7050shot_balance__memoryagentbench_0"
    )
    assert banking_id != clinic_id


def test_memoryagentbench_source_is_sanitized_for_project_id(tmp_path: Path):
    adapter = MemMachineAdapter(
        _config(
            dataset_name="memoryagentbench",
            memoryagentbench={
                "split": "Accurate Retrieval",
                "source": "longmemeval_s*",
            },
        ),
        output_dir=tmp_path,
    )

    assert adapter._project_id("memoryagentbench/0") == (
        "eval__normal__mab__Accurate_Retrieval__"
        "longmemeval_s__memoryagentbench_0"
    )


def test_memoryagentbench_requires_source_for_safe_isolation(tmp_path: Path):
    with pytest.raises(ValueError, match="require a selected source"):
        MemMachineAdapter(
            _config(
                dataset_name="memoryagentbench",
                memoryagentbench={"split": "Accurate_Retrieval"},
            ),
            output_dir=tmp_path,
        )


def test_non_memoryagentbench_project_id_remains_backward_compatible(tmp_path: Path):
    adapter = MemMachineAdapter(
        _config(dataset_name="meme"),
        output_dir=tmp_path,
    )

    assert adapter._project_id("meme_0") == "eval__normal__meme_0"
    assert adapter._project_prefix_for_log() == "eval__normal"
