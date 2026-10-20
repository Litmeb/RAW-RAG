from types import SimpleNamespace

from rich.console import Console

from evaluation.cli import (
    MEMORA_RECYCLE_EXIT_CODE,
    apply_fast_path_category_filter,
    run_memoryagentbench_batch,
)
from evaluation.src.core.data_models import QAPair


def test_fast_path_applies_dataset_category_filter():
    dataset = SimpleNamespace(
        qa_pairs=[
            QAPair("q1", "question 1", "answer 1", category="1"),
            QAPair("q5", "question 5", "answer 5", category="5"),
        ]
    )
    console = Console(record=True)

    apply_fast_path_category_filter(
        dataset,
        {"evaluation": {"filter_category": [5]}},
        console,
    )

    assert [qa.question_id for qa in dataset.qa_pairs] == ["q1"]
    assert "Filtered out 1 questions" in console.export_text()


def test_memoryagentbench_batch_restarts_controlled_memora_worker(monkeypatch):
    return_codes = iter([MEMORA_RECYCLE_EXIT_CODE, MEMORA_RECYCLE_EXIT_CODE, 0])
    launches = []

    monkeypatch.setattr(
        "evaluation.cli.build_memoryagentbench_child_command",
        lambda **_kwargs: ["memora-worker"],
    )

    def fake_run(command, cwd):
        launches.append((command, cwd))
        return SimpleNamespace(returncode=next(return_codes))

    monkeypatch.setattr("evaluation.cli.subprocess.run", fake_run)
    console = Console(record=True)

    exit_code = run_memoryagentbench_batch(
        SimpleNamespace(output_dir=None, system="memora"),
        [("Test_Time_Learning", "recsys_redial_full")],
        console,
    )

    assert exit_code == 0
    assert len(launches) == 3
    assert "recycle 2" in console.export_text()
