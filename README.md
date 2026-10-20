# RAGforLoCoMo Reproduction

This directory contains the code and configuration needed to reproduce the
experiments in the RAGforLoCoMo paper.

## 1. Install

From this directory:

```powershell
cd methods\EverCore
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-eval.txt
```

`requirements-eval.txt` covers RAW-RAG, its benchmark preparation, reader and
judges. Analysis and offline tests have separate dependency sets:

```powershell
python -m pip install -r requirements-analysis.txt
python -m pip install -r requirements-test.txt
```

Install an additional requirements file only when reproducing the
corresponding local baseline:

```powershell
python -m pip install -r requirements-simplemem.txt
python -m pip install -r requirements-memora-local.txt
```

## 2. Match the Paper Configuration

Before every reproduction run, first update:

1. `methods/EverCore/.env`
2. The relevant benchmark YAML file under
   `methods/EverCore/evaluation/config/datasets`
3. The relevant baseline/system YAML file under
   `methods/EverCore/evaluation/config/systems`

Set the API credentials, endpoints, model identifiers, prompts, chunking,
retrieval, context-budget, and evaluator settings to the values used in the
paper. **The configuration files currently included in this package are not
guaranteed to represent the final configurations used for the paper.**
Reproduction is valid only after these files have been checked and, where
necessary, updated.

Create `.env` from the template if needed:

```powershell
Copy-Item .env.example .env
```

At minimum, configure the variables required by the selected system:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
VECTORIZE_API_KEY=...
VECTORIZE_BASE_URL=...
VECTORIZE_MODEL=...
RERANK_API_KEY=...
RERANK_BASE_URL=...
RERANK_MODEL=...
RERANK_ENDPOINT=...
```

Do not commit `.env` or API keys.

## 3. Prepare Data

Place the benchmark files under `methods/EverCore/evaluation/data`:

| Benchmark | File or directory |
| --- | --- |
| LoCoMo | `locomo/locomo10.json` |
| LongMemEval-S | `longmemeval/longmemeval_s_cleaned.json` |
| MEME | `meme/meme_filler32k.json` |
| PersonaMem-v2 | `personamem_v2/benchmark/text/benchmark.csv` and `personamem_v2/data/chat_history_32k/*.json` |
| MemoryAgentBench | Source files required by the selected dataset configuration |
| BEAM-1M | Ordered conversations required by the timing experiment |

The dataset YAML files define conversions and benchmark-specific settings.

## 4. Reproduce RAW-RAG

Run from `methods/EverCore`:

```powershell
python -m evaluation.cli --dataset locomo --system rag
python -m evaluation.cli --dataset longmemeval --system rag
python -m evaluation.cli --dataset meme --system rag
python -m evaluation.cli --dataset personamem_v2 --system rag
python -m evaluation.cli --dataset memoryagentbench --system rag
```

The MemoryAgentBench command follows the source selection and chunking defined
in its YAML configuration. RAW-RAG resolves converter defaults independently
for each source: the two fact-consolidation sources use one fact per message,
while other sources retain raw context for token chunking in the adapter.
Explicit converter overrides remain supported. Use the same configuration for the RAW-RAG
ablation systems:

```powershell
python -m evaluation.cli --dataset memoryagentbench --system rag_atomic_facts
python -m evaluation.cli --dataset memoryagentbench --system rag_raw_summary
python -m evaluation.cli --dataset memoryagentbench --system rag_graph_expanded_raw
```

## 5. Reproduce Full-Context Controls

Configure `evaluation/config/systems/full_context.yaml` to match the paper,
then run:

```powershell
python -m evaluation.cli --dataset locomo --system full_context --context-window 128000
python -m evaluation.cli --dataset meme --system full_context --context-window 128000
python -m evaluation.cli --dataset longmemeval --system full_context --context-window 128000
python -m evaluation.cli --dataset memoryagentbench --system full_context --context-window 128000
```

## 6. External Baselines

The baseline YAML files are included, but some upstream implementations and
services must be installed separately. Read and update the matching YAML file
before running each baseline.

```powershell
# SimpleMem
cd ..\..
.\scripts\simplemem\fetch_simplemem.ps1
cd methods\EverCore
python -m pip install -r requirements-simplemem.txt
python -m evaluation.cli --dataset locomo --system simplemem

# Memora
cd ..\..
.\scripts\memora\fetch_memora.ps1
cd methods\EverCore
python -m pip install -r requirements-memora-local.txt
python -m evaluation.cli --dataset locomo --system memora
```

For local MemOS or MemoryOS, start the corresponding service from the package
root, verify that it is ready, and then run the matching system configuration:

```powershell
python -m evaluation.cli --dataset locomo --system memos_local
python -m evaluation.cli --dataset memoryagentbench --system memoryos_local
```

Other systems are selected with their corresponding names in
`evaluation/config/systems`, including `mem0`, `mem0_local`, `emem`, `emem_g`,
`memmachine`, `memu`, `memu_local`, `memos`, `evermemos`, and
`evermemos_local_api`.

## 7. Timing Experiment

For the sequential BEAM-1M ingestion experiment:

```powershell
python -m evaluation.scripts.convert_beam_1m_sequence
python -m evaluation.speed_sequence_cli `
  --system rag `
  --data-dir ..\..\beam\BEAM-1M
```

Use the paper's dataset slice and system YAML configuration.

## 8. Inspect Results

Results are written under `methods/EverCore/evaluation/results`. Inspect:

```powershell
Get-Content evaluation\results\<result-directory>\report.txt
Get-Content evaluation\results\<result-directory>\eval_results.json
```

Before comparing with the paper, verify that the dataset, question count,
reader model, embedding model, reranker, prompts, judge, context budget, and
chunking settings match the paper. Also check that API or answer-generation
failures were not incorrectly counted as correct answers.

LongMemEval reproduction uses the EverMemOS-style judge:

```powershell
python -m evaluation.cli --dataset longmemeval --system rag
```

LongMemEval now preserves complete messages before indexing; the previous
8,000-character loader cap is disabled. Rebuild affected runs before comparing
results. This correction does not retroactively validate or update paper scores.

Reranking is mandatory when enabled. Invalid/missing scores, duplicate indices,
empty responses, and exhausted service retries abort the run instead of
silently falling back to dense retrieval. An intentional embedding-only
ablation must explicitly set `rerank.enabled: false`.

Run identities cover source data, questions, model/configuration settings and
prompts. Changed full-pipeline inputs archive stale downstream artifacts under
`stale_artifacts/` before rebuilding. Per-conversation source fingerprints also
protect the index. Legacy indexes must be rebuilt. The loader
also refreshes converted dataset caches when their raw inputs or converter
implementation change; legacy conversion caches require their original inputs
for this one-time rebuild. A partial-stage invocation
cannot silently adopt incompatible full-pipeline checkpoints. Answer-only
reruns have separate identities tied to the saved evidence and reader settings.

Generic judge failures are recorded in `judge_progress.json` and prevent an
accuracy report. Retrying the same evaluation reuses successful judgments and
retries missing ones. Changing answers, judge settings or prompts prevents
reuse of incompatible judgments. API failures are not incorrect answers.

## Diagnostic Experiments

See [DIAGNOSTICS.md](DIAGNOSTICS.md) for the dependency chain and explicit source
manifests for 5k evidence controls and complete-store uncapped experiments.
No unpublished historical analysis reports are required by those workflows.

## Tests

Run the audit regression suite from `methods/EverCore`, without credentials,
benchmark downloads or model calls:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/test_rawrag_regressions.py tests/test_diagnostic_source_manifests.py tests/test_cli_fast_path_filter.py tests/test_longmemeval_official_judge.py tests/test_build_rag_5k_intervention_runs.py tests/test_search_stage_checkpoint.py tests/test_rag_graph_expanded_raw.py tests/test_meme_complete_store_uncapped.py -q
```

The full suite also includes external-baseline integration tests. Install the
corresponding upstream checkouts, GPU and service dependencies before running:

```powershell
python -m pytest tests -q
```
