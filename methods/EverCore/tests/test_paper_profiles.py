"""Paper configuration contracts and offline PowerShell orchestration."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from evaluation.src.core.data_models import QAPair
from evaluation.src.core.stages.search_stage import _build_search_query
from evaluation.src.utils.config import load_yaml


EVERCORE = Path(__file__).resolve().parents[1]
PAPER = EVERCORE.parents[1] / "scripts" / "paper"
MANIFEST = json.loads((PAPER / "conditions.json").read_text(encoding="utf-8"))
CONDITIONS = MANIFEST["conditions"]
PWSH = shutil.which("pwsh")


def profile(kind, name):
    return yaml.safe_load(
        (PAPER / "configs" / kind / f"{name}.yaml").read_text(encoding="utf-8")
    )


def test_condition_matrix_and_dependency_order():
    assert len(CONDITIONS) == 15
    assert sum(c["group"] == "Main" for c in CONDITIONS) == 7
    assert sum(c["mode"] == "noabs" for c in CONDITIONS) == 2
    assert len(MANIFEST["mab_sources"]) == 14
    seen = set()
    for condition in CONDITIONS:
        assert condition["id"] not in seen
        if condition["source_condition"]:
            assert condition["source_condition"] in seen
        seen.add(condition["id"])
        assert condition["paper_labels"]
        assert profile("datasets", condition["dataset_config"])["name"] == condition["dataset"]
        system = profile("systems", condition["system_config"])
        assert system["adapter"] == "rag"
        assert system["chunking"]["chunk_size"] == 256
        assert system["chunking"]["chunk_overlap"] == 32
        assert system["search"]["embedding_top_k"] == 128
        assert system["search"]["rerank_top_k"] == 15
        assert system["rerank"]["enabled"] is True
        assert system["storage"]["backend"] == "local"
        assert not system["search"]["dynamic_top_k"]["enabled"]
        assert not system["chunking"]["dedup_turns_on_display"]


@pytest.mark.parametrize("condition", CONDITIONS, ids=lambda c: c["id"])
def test_model_settings_are_not_environment_selected(condition, monkeypatch):
    for name in ("LLM_MODEL", "VECTORIZE_MODEL", "RERANK_MODEL", "EVALUATE_MODEL",
                 "MEME_EVALUATE_MODEL", "MEMORYAGENTBENCH_EVALUATE_MODEL"):
        monkeypatch.setenv(name, "incorrect-model")
    system_path = PAPER / "configs/systems" / f"{condition['system_config']}.yaml"
    dataset_path = PAPER / "configs/datasets" / f"{condition['dataset_config']}.yaml"
    system = load_yaml(str(system_path))
    dataset = load_yaml(str(dataset_path))
    for section in ("embedding", "rerank", "llm"):
        assert system[section]["model"] != "incorrect-model"
        assert "${" not in system[section]["model"]
    if "llm" in dataset["evaluation"]:
        assert dataset["evaluation"]["llm"]["model"] != "incorrect-model"
    assert system["llm"]["temperature"] == 0
    assert system["llm"]["max_tokens"] == 16384


def test_judges_and_specialized_profiles():
    for dataset in ("locomo", "longmemeval"):
        evaluation = profile("datasets", dataset)["evaluation"]
        assert evaluation["num_runs"] == 3
        assert evaluation["llm"]["model"] == "gpt-4o-mini-2024-07-18"
    for name, count in (("meme-gpt4omini", 3), ("meme-flash", 1)):
        evaluation = profile("datasets", name)["evaluation"]
        assert evaluation["num_runs"] == count
        assert evaluation["llm"]["model"] == "gpt-4o-2024-11-20"
    assert profile("datasets", "mab-gpt4omini")["evaluation"]["llm"]["model"] == "gpt-4o"
    assert profile("datasets", "mab-flash")["evaluation"]["llm"]["model"] == "gpt-4o-2024-11-20"
    assert profile("datasets", "longmemeval")["data"]["max_content_length"] is None
    assert profile("datasets", "locomo")["evaluation"]["filter_category"] == [5]
    assert profile("datasets", "personamem_v2")["evaluation"]["type"] == "exact_match"
    for name in ("textemb3-gpt4omini", "textemb3-flash"):
        system = profile("systems", name)
        overrides = system["dataset_overrides"]
        assert overrides["meme"]["search"]["sort_selected_chunks_by_time"] is True
        assert overrides["meme"]["answer"]["prompt_key"] == "answer_prompt_meme_official"
        assert not overrides["memoryagentbench"]["chunking"]["expand_turns_on_display"]
        for source in ("factconsolidation_sh_262k", "factconsolidation_mh_262k"):
            assert overrides["memoryagentbench"]["sources"][source]["chunking"] == {
                "strategy": "turn", "chunk_size": 1, "chunk_overlap": 0,
            }
    flash = profile("systems", "textemb3-flash")
    assert flash["llm"]["api_key"] == "${DEEPSEEK_API_KEY}"
    qwen = profile("systems", "qwen3-gpt41mini")
    assert qwen["embedding"]["model"] == "Qwen/Qwen3-Embedding-4B"
    assert qwen["rerank"]["model"] == "Qwen/Qwen3-Reranker-4B"
    assert qwen["dataset_overrides"]["personamem_v2"]["search"]["personamem_raw_query"]


def test_minimal_changes_only_reader_prompt():
    for reader in ("gpt4omini", "gpt41mini"):
        base = profile("systems", f"textemb3-{reader}")
        minimal = profile("systems", f"textemb3-{reader}-minimal")
        # Only the baseline shared with MEME/MAB has these irrelevant overrides.
        base.pop("dataset_overrides", None)
        base["answer"]["prompt_key"] = "answer_prompt_cot_minimal"
        assert base == minimal


def test_raw_query_is_opt_in_and_keeps_reader_question():
    question = "What should I cook? Please recall my related preferences."
    qa = QAPair("q1", question, "A", metadata={
        "raw_user_query": "{'role': 'user', 'content': 'What should I cook?'}",
    })
    assert _build_search_query(qa, SimpleNamespace(config={})) == question
    adapter = SimpleNamespace(config={"search": {"personamem_raw_query": True}})
    assert _build_search_query(qa, adapter) == "What should I cook?"
    assert qa.question == question
    qa.metadata.clear()
    with pytest.raises(ValueError, match="raw_user_query"):
        _build_search_query(qa, adapter)


FAKE_CLI = r'''
import json
import os
from pathlib import Path
import sys
import yaml

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

dataset = value("--dataset")
system = yaml.safe_load(Path("evaluation/config/systems/rag.yaml").read_text())
data = yaml.safe_load(Path(f"evaluation/config/datasets/{dataset}.yaml").read_text())
with Path("calls.jsonl").open("a") as f:
    f.write(json.dumps({
        "args": args, "system": system, "data": data,
        "meme_runs": os.environ.get("MEME_JUDGE_NUM_RUNS"),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }) + "\n")
if os.environ.get("PAPER_FAKE_FAIL") == "1":
    sys.exit(9)
if "--mab-report-accuracy" in args:
    print("offline MAB aggregate")
    sys.exit(0)
output = Path(value("--output-dir"))
output.mkdir(parents=True, exist_ok=True)
if "add" in args:
    (output / "search_results.json").write_text(json.dumps({"source": output.name}))
else:
    assert (output / "search_results.json").is_file()
(output / "eval_results.json").write_text(json.dumps({"detailed_results": [
    {"question_id": "question1", "category": "temporal",
     "llm_judgments": {"judgment_1": True, "judgment_2": True, "judgment_3": False}},
    {"question_id": "question2_abs", "category": "temporal",
     "llm_judgments": {"judgment_1": False, "judgment_2": False, "judgment_3": False}}
]}))
if dataset == "locomo":
    answers = [
        {"question_id": f"locomo_0_qa{i}", "category": str(i % 4 + 1),
         "answer": "Blue.", "golden_answer": "blue"}
        for i in range(1540)
    ]
    (output / "answer_results.json").write_text(json.dumps(answers))
    (output / "eval_results.json").write_text(json.dumps({"detailed_results": {
        "locomo_exp_user_0": [
            {"question_id": row["question_id"], "category": row["category"],
             "generated_answer": row["answer"], "golden_answer": row["golden_answer"],
             "llm_judgments": {"judgment_1": True, "judgment_2": True, "judgment_3": False}}
            for row in answers
        ]
    }}))
'''


@pytest.fixture
def sandbox(tmp_path):
    if not PWSH:
        pytest.skip("PowerShell 7 is needed for runner integration tests")
    root = tmp_path / "package with spaces"
    shutil.copytree(PAPER, root / "scripts/paper")
    evercore = root / "methods/EverCore"
    for folder in ("evaluation/config/systems", "evaluation/config/datasets",
                   "evaluation/scripts"):
        (evercore / folder).mkdir(parents=True)
    (evercore / "evaluation/__init__.py").write_text("")
    (evercore / "evaluation/cli.py").write_text(FAKE_CLI)
    shutil.copy(
        EVERCORE / "evaluation/scripts/analyze_longmemeval_no_abs.py",
        evercore / "evaluation/scripts/analyze_longmemeval_no_abs.py",
    )
    shutil.copy(
        EVERCORE / "evaluation/scripts/score_locomo_lexical.py",
        evercore / "evaluation/scripts/score_locomo_lexical.py",
    )
    return root, evercore


def run_script(sandbox, *args, fail=False):
    root, _ = sandbox
    environment = os.environ.copy()
    environment["MEME_JUDGE_NUM_RUNS"] = "99"
    environment["PAPER_FAKE_FAIL"] = "1" if fail else "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [PWSH, "-NoProfile", "-File", str(root / "scripts/paper/run_rawrag.ps1"),
         "-Python", sys.executable, *args],
        cwd=root, env=environment, capture_output=True, encoding="utf-8",
        errors="replace", timeout=120,
    )


def calls(evercore):
    return [json.loads(line) for line in (evercore / "calls.jsonl").read_text().splitlines()]


def test_runner_all_copy_order_dependencies_and_reports(sandbox):
    result = run_script(sandbox, "-Group", "All", "-Tag", "offline")
    assert result.returncode == 0, result.stdout + result.stderr
    root, evercore = sandbox
    recorded = calls(evercore)
    # 13 execution conditions, two MAB conditions expand to 14 sources,
    # and each of those also invokes the existing aggregate reporter.
    assert len(recorded) == 41
    assert all(c["hash_seed"] == "0" for c in recorded)
    meme = [c for c in recorded if c["data"]["name"] == "meme"]
    assert [c["meme_runs"] for c in meme] == ["3", "1"]
    assert [c["data"]["evaluation"]["num_runs"] for c in meme] == [3, 1]
    assert "add" not in meme[1]["args"]
    for spec in CONDITIONS:
        snapshot = evercore / "evaluation/results/paper-configs" / f"offline-{spec['id']}"
        saved = json.loads((snapshot / "condition.json").read_text(encoding="utf-8-sig"))
        assert saved["condition"]["id"] == spec["id"]
        assert saved["system_sha256"]
        assert (snapshot / "system.yaml").read_bytes() == (
            root / f"scripts/paper/configs/systems/{spec['system_config']}.yaml"
        ).read_bytes()
        if spec["mode"] == "noabs":
            report = json.loads((snapshot / "non_abstention.json").read_text(encoding="utf-8-sig"))
            assert report["kept_questions"] == 1
            assert report["removed_questions"] == 1
        elif spec["mode"] == "answer_only" and spec["dataset"] != "memoryagentbench":
            prefix = f"{spec['dataset']}-rag-offline-"
            results = evercore / "evaluation/results"
            assert (results / f"{prefix}{spec['id']}/search_results.json").read_bytes() == (
                results / f"{prefix}{spec['source_condition']}/search_results.json"
            ).read_bytes()
            if spec["id"].endswith("-minimal"):
                metrics = json.loads(
                    (results / f"{prefix}{spec['id']}/lexical_metrics.json").read_text()
                )
                assert metrics["overall"]["questions"] == 1540
                assert metrics["overall"]["f1"] == 100
                assert metrics["overall"]["bleu1"] == 50
                assert metrics["overall"]["j"] == pytest.approx(200 / 3)
    # No restoration: the final profile remains installed.
    assert (evercore / "evaluation/config/systems/rag.yaml").read_bytes() == (
        root / "scripts/paper/configs/systems/qwen3-gpt41mini.yaml"
    ).read_bytes()


@pytest.mark.parametrize("option", ["-DryRun", "-List"])
def test_preview_does_not_write_or_call_python(sandbox, option):
    result = run_script(sandbox, "-Group", "All", option)
    assert result.returncode == 0, result.stderr
    _, evercore = sandbox
    assert not (evercore / "calls.jsonl").exists()
    assert not (evercore / "evaluation/config/systems/rag.yaml").exists()
    assert not (evercore / "evaluation/results").exists()


def test_selection_order_and_invalid_condition(sandbox):
    result = run_script(
        sandbox, "-Condition",
        "locomo-textemb3-gpt4omini-minimal,locomo-textemb3-gpt4omini",
    )
    assert result.returncode == 0, result.stderr
    _, evercore = sandbox
    recorded = calls(evercore)
    assert len(recorded) == 2
    assert "add" in recorded[0]["args"]
    assert "add" not in recorded[1]["args"]
    result = run_script(sandbox, "-Condition", "not-a-paper-condition")
    assert result.returncode != 0
    assert len(calls(evercore)) == 2


def test_failure_stops_without_restoring(sandbox):
    result = run_script(sandbox, "-Group", "Main", fail=True)
    assert result.returncode != 0
    _, evercore = sandbox
    assert len(calls(evercore)) == 1
    system = yaml.safe_load((evercore / "evaluation/config/systems/rag.yaml").read_text())
    assert system["llm"]["model"] == "gpt-4o-mini"
