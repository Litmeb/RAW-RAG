from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "evaluation" / "scripts" / "analyze_reverse_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_reverse_ablation_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stats(calls: int, errors: int, prompt: int, completion: int, latency_ms: float):
    return {
        "calls": calls,
        "errors": errors,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "latency_ms_sum": latency_ms,
    }


def test_resource_kind_separates_embedding_reranker_and_llm():
    module = load_module()

    assert module.resource_kind("embedding") == "embedding"
    assert module.resource_kind("dense_embedding") == "embedding"
    assert module.resource_kind("rerank") == "reranker"
    assert module.resource_kind("cross_encoder_reranker") == "reranker"
    assert module.resource_kind("answer") == "llm"
    assert module.resource_kind("chunk_graph_link_judge") == "llm"


def test_summarize_resource_usage_reports_separate_resources(tmp_path):
    module = load_module()
    run = {
        "label": "variant",
        "split": "split",
        "source": "source",
        "run_dir": tmp_path,
        "report_elapsed_seconds": 12.5,
        "request_stats": {
            "add:embedding": stats(2, 0, 120, 0, 1000),
            "search:rerank": stats(3, 1, 999, 999, 2000),
            "add:summary_extract": stats(4, 0, 300, 40, 3000),
            "answer:answer": stats(5, 2, 500, 50, 4000),
        },
    }

    summary = module.summarize_resource_usage(run)

    assert summary["embedding_calls"] == 2
    assert summary["embedding_tokens"] == 120
    assert summary["reranker_calls"] == 3
    assert summary["reranker_errors"] == 1
    assert summary["llm_calls"] == 9
    assert summary["llm_errors"] == 2
    assert summary["llm_prompt_tokens"] == 800
    assert summary["llm_completion_tokens"] == 90
    assert summary["llm_total_tokens"] == 890
    assert summary["embedding_latency_seconds"] == 1.0
    assert summary["reranker_latency_seconds"] == 2.0
    assert summary["llm_latency_seconds"] == 7.0
