# RAW-RAG: Naive RAG for Long-Term LLM Memory

Implementation and experimental code for **"Is Complex Agentic Memory Always
Needed? Naive RAG Is Stronger Than We Thought"**.

**[Introduction](#introduction) | [Framework](#framework) |
[Results](#evaluation-results) | [Analysis](#what-explains-the-results) |
[Reproduction](#reproduction) |
[Citation](#citation)**

## Introduction

Do long-term conversation tasks always require complex agentic memory?
RAW-RAG revisits a simple alternative: **preserve raw history, retrieve relevant
evidence, and let the reader reason over it**.

> **Abstract.** LLM memory is widely used in long-term conversation understanding.
> Agentic memory systems often use mutation, structural organization, learned
> retrieval strategies, or other complex operations. Is this complexity always
> necessary? We study RAW-RAG, a naive RAG baseline using dense retrieval and
> reranking over raw history to supply evidence to a reader LLM. Across LoCoMo,
> LongMemEval, MEME, and MemoryAgentBench, RAW-RAG ranks first among the compared
> memory systems in six of seven benchmark-model settings, outperforming the
> strongest compared agentic memory system by 0.13-5.53 points. We investigate
> why raw-history retrieval remains competitive. Evidence analysis shows that
> its higher coverage of necessary evidence can offset lower utilization.
> Useful evidence includes both answer values and the relations and dependency
> paths needed to derive them. Even complete access to transformed memory stores
> leaves an accuracy gap relative to raw evidence, highlighting the importance
> of preserving raw information. These findings challenge the necessity of
> complex agentic memory for the evaluated conversation tasks and establish
> naive RAG as an essential baseline.

### Key Contributions

1. **Revisit explicit memory mutation.** Capable readers can resolve dynamic
   states from sufficient raw evidence. Disabling UPDATE/DELETE improves
   accuracy in the tested Mem0 configuration.
2. **Establish a strong raw-history baseline.** Dense retrieval plus reranking
   ranks first among the compared memory systems in six of seven settings
   across four benchmarks, without LLM-based memory extraction or organization.
3. **Diagnose evidence loss.** Answer values alone are not enough: temporal
   relations and dependency paths matter. Complete access to a transformed
   memory store does not necessarily recover information preserved in raw history.

## Framework

RAW-RAG uses a fixed **build, retrieve, rerank, read** pipeline:

1. **Build:** serialize conversation history with timestamps and available
   image captions; embed overlapping raw-text windows in a flat index.
   Use a window size of $L$ tokens with an overlap of $O$ tokens.
2. **Retrieve:** embed the question and select the top $K$ windows by cosine
   similarity.
3. **Rerank:** score question-window pairs with a cross-encoder and select the
   top $k$ evidence items, where $k \leq K$.
4. **Read:** supply the selected raw evidence and question to the reader LLM,
   which resolves relevant temporal order and state changes.

Ingestion preserves raw text without LLM-based extraction or summarization.
The retrieval parameters are configurable; $k$ counts evidence items, not
tokens. Benchmark-specific settings are listed in the
[paper profiles](scripts/paper/README.md).

## Evaluation Results

Manuscript results. Higher is better; scores and differences are in percentage
points. "Best compared memory system" excludes RAW-RAG and Full-context.

| Benchmark | Reader | RAW-RAG | Best compared memory system | Difference |
| --- | --- | ---: | --- | ---: |
| LoCoMo | GPT-4o-mini | **89.07** | MemMachine (88.12) | +0.95 |
| LoCoMo | GPT-4.1-mini | **93.70** | EverMemOS (93.05) | +0.65 |
| LongMemEval | GPT-4.1-mini | **87.53** | Memora (87.40) | +0.13 |
| MEME | GPT-4o-mini | **39.30** | MemMachine (37.06) | +2.24 |
| MEME | DeepSeek-V4-Flash | 43.19 | **MemMachine (44.71)** | -1.52 |
| MemoryAgentBench | GPT-4o-mini | **48.32** | Memora (43.84) | +4.48 |
| MemoryAgentBench | DeepSeek-V4-Flash | **53.41** | Memora (47.88) | +5.53 |

LoCoMo uses 1,540 non-adversarial questions from 10 conversations; LongMemEval
uses 500 examples. MEME reports the six-task macro-average over 1,188
phase-specific questions from 100 episodes. MemoryAgentBench reports the mean
of four capability macro-averages over 2,071 questions from 14 sources.
PersonaMem-v2 is evaluated in the appendix.

<p align="center">
  <img width="100%" src="assets/teaser_results.png" alt="Paper overview: accuracy in seven benchmark-model settings and sequential BEAM-1M ingestion time">
</p>

*The overview figure shows selected comparators and uses sector-wise truncated
scales for accuracy. The table above includes the strongest compared memory
system in each setting.*

**Efficiency.** On the paper's 10-episode, 110-question MEME timing experiment,
RAW-RAG takes 50.32 seconds for add-to-ready and 255.59 seconds for search,
totaling 305.91 seconds. Answer generation and evaluation are excluded.
The figure also reports cumulative sequential ingestion on BEAM-1M.

## What Explains the Results?

- **Retrieval quality matters.** Removing reranking reduces MemoryAgentBench
  Total from 48.32 to 44.40 with GPT-4o-mini. Returning only one item reduces it
  by 11.73 points; simply increasing retrieval depth is not always better.
- **More memory processing is not automatically better.** In the same
  MemoryAgentBench ablation, atomic-fact extraction, summary augmentation,
  and graph expansion score 28.70, 42.51, and 42.54, respectively, versus
  48.32 for RAW-RAG.
- **Evidence coverage and utilization are distinct.** A reader may use
  retrieved evidence imperfectly, yet benefit from access to more of the
  information necessary to answer the question.
- **Relations are part of the evidence.** Temporal tracking, deletion,
  cascades, and invalidation require linked events and dependencies, not
  merely isolated mentions of answer values.

## Repository Guide

| Location | Contents |
| --- | --- |
| [`methods/EverCore/evaluation`](methods/EverCore/evaluation) | Evaluation CLI, retrieval adapters, readers, evaluators, and analysis scripts |
| [`methods/EverCore/evaluation/config`](methods/EverCore/evaluation/config) | Dataset, model, prompt, system, and diagnostic configurations |
| [`methods/EverCore/tests`](methods/EverCore/tests) | Regression and integration tests |
| [`scripts`](scripts) | Setup helpers for selected external baselines |
| [`assets`](assets) | Paper figures used in this README |

Benchmark data, credentials, generated indexes, and experiment outputs are
not bundled. Some baselines require separate upstream checkouts or services.

## Reproduction

RAW-RAG uses the system name `rag`. Complete the setup and data preparation
below before running the seven main paper conditions. Commands in steps 1-5
use the repository root as the working directory.

### 1. Install

Use Python 3.12 and PowerShell 7 (`pwsh`):

```powershell
python -m pip install -r methods/EverCore/requirements-eval.txt
python -m nltk.downloader punkt_tab
```

`requirements-eval.txt` covers RAW-RAG, its benchmark preparation, reader and
judges. Analysis and offline tests have separate dependency sets:

```powershell
python -m pip install -r methods/EverCore/requirements-analysis.txt
python -m pip install -r methods/EverCore/requirements-test.txt
```

### 2. Configure Services

Create `methods/EverCore/.env` from the
[template](methods/EverCore/.env.example) if it does not already exist:

```powershell
if (-not (Test-Path methods/EverCore/.env)) {
  Copy-Item methods/EverCore/.env.example methods/EverCore/.env
}
```

Fill in credentials and set the endpoints for the main conditions.
Replace the template's blank endpoint values and its `/rerank` path with
the matching values below, or with those of your compatible services:

```dotenv
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
VECTORIZE_API_KEY=...
VECTORIZE_BASE_URL=https://api.openai.com/v1
RERANK_API_KEY=...
RERANK_BASE_URL=https://api.cohere.com
RERANK_ENDPOINT=/v2/rerank
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

The [paper profiles](scripts/paper/README.md) select the reader, embedding,
reranker, prompts, and judges automatically; model environment variables such
as `LLM_MODEL` do not override these frozen profiles. Flash readers use
`DEEPSEEK_*`, while their judges continue to use `LLM_*`.

### 3. Prepare Data

Place the main benchmark files under `methods/EverCore/evaluation/data`.
MemoryAgentBench is downloaded and converted automatically on first use,
so its initial run requires access to Hugging Face:

| Benchmark | File or preparation | Expected questions |
| --- | --- | ---: |
| LoCoMo | `locomo/locomo10.json` | 1,540 non-adversarial |
| LongMemEval-S | `longmemeval/longmemeval_s_cleaned.json` | 500 |
| MEME | `meme/meme_filler32k.json` | 1,188 from 100 episodes |
| MemoryAgentBench | Automatically cached under `memoryagentbench/` for 14 sources | 2,071 |

The dataset YAML files define conversions and benchmark-specific settings.
RAW-RAG resolves MemoryAgentBench converter defaults independently for each
source: the two fact-consolidation sources use one fact per message, while
other sources retain raw context for token chunking in the adapter. The
10,000,000-character message limit covers the complete main benchmark contexts,
including ReDial's roughly 5.7-million-character serialized message.

### 4. Run the Main Experiments

Preview the seven main conditions, then run them:

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main -List
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main -Tag paper
```

The runner selects each condition's configuration and executes it in order.
The Flash conditions reuse retrieval from their matching GPT-4o-mini runs.
Runs in one checkout must be serial because the runner overwrites active
system, dataset, and prompt YAML files.

Use a new `-Tag`, such as `paper-v2`, when regenerating results after data or
configuration changes. In particular, rerun both MAB main conditions to replace
retrieval previously built with the old 2,000,000-character limit.

### 5. Inspect Results

Results are written under `methods/EverCore/evaluation/results`.
Each run contains `report.txt`, `eval_results.json`, `answer_results.json`,
and `search_results.json`. Paper-profile configuration snapshots are saved
under `evaluation/results/paper-configs`. MAB runs produce separate results
for all 14 sources and a `mab_accuracy.txt` aggregate in the corresponding
configuration snapshot directory.

### Optional: Appendix Conditions

`-Group All` includes Main and the appendix conditions, including LoCoMo
J/F1/BLEU-1 reports. It is an alternative to Main, not a required next step.
Before running it, configure the Qwen services described in the
[paper profiles](scripts/paper/README.md) and add the PersonaMem-v2 files under
`evaluation/data`: `personamem_v2/benchmark/text/benchmark.csv` and
`personamem_v2/data/chat_history_32k/*.json`.

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All -Tag paper-all
```

## Custom Runs

The generic CLI commands below are alternatives for manually configured runs,
not additional steps after Main. First configure the active dataset, system,
and prompt YAML files under `methods/EverCore/evaluation/config`.
After Main, the active `rag.yaml` remains the Flash profile; running these
commands unchanged would retain that reader even for LoCoMo or LongMemEval.

Run from `methods/EverCore`:

```powershell
python -m evaluation.cli --dataset locomo --system rag
python -m evaluation.cli --dataset longmemeval --system rag
python -m evaluation.cli --dataset meme --system rag
python -m evaluation.cli --dataset personamem_v2 --system rag
python -m evaluation.cli --dataset memoryagentbench --system rag
```

The RAW-RAG ablation systems use the same dataset configuration:

```powershell
python -m evaluation.cli --dataset memoryagentbench --system rag_atomic_facts
python -m evaluation.cli --dataset memoryagentbench --system rag_raw_summary
python -m evaluation.cli --dataset memoryagentbench --system rag_graph_expanded_raw
```

## Timing Experiment

The sequential BEAM-1M ingestion experiment is separate from the main accuracy
runs. Prepare its ordered conversations and select the intended RAW-RAG
configuration first. Run from `methods/EverCore`:

```powershell
python -m evaluation.scripts.convert_beam_1m_sequence
python -m evaluation.speed_sequence_cli `
  --system rag `
  --run-name paper-beam-1m `
  --data-dir ..\..\beam\BEAM-1M
```

## Diagnostic Experiments

The diagnostic entry points include
[`build_rag_5k_intervention_runs.py`](methods/EverCore/evaluation/scripts/build_rag_5k_intervention_runs.py)
for 5k evidence controls and
[`run_meme_complete_store_uncapped.py`](methods/EverCore/evaluation/scripts/run_meme_complete_store_uncapped.py)
for complete-store uncapped experiments.

## Tests

From `methods/EverCore`:

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/test_rawrag_regressions.py tests/test_diagnostic_source_manifests.py tests/test_cli_fast_path_filter.py tests/test_longmemeval_official_judge.py tests/test_build_rag_5k_intervention_runs.py tests/test_search_stage_checkpoint.py tests/test_rag_graph_expanded_raw.py tests/test_meme_complete_store_uncapped.py -q
```

The full suite includes external-baseline integration tests:

```powershell
python -m pytest tests -q
```

## Citation

This repository accompanies **"Is Complex Agentic Memory Always Needed?
Naive RAG Is Stronger Than We Thought"**.
