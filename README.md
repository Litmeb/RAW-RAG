# RAW-RAG: Naive RAG for Long-Term LLM Memory

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://arxiv.org/abs/2610.30755"><img src="https://img.shields.io/badge/arXiv-2610.30755-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/papers/2610.30755"><img src="https://img.shields.io/badge/Hugging%20Face-Papers-ffd21e.svg" alt="Hugging Face Papers"></a>
</p>

Implementation and experimental code for **["Is Complex Agentic Memory Always
Needed? Naive RAG Is Stronger Than We Thought"](https://arxiv.org/abs/2610.30755)**.

**[Introduction](#introduction) | [Framework](#framework) |
[Results](#evaluation-results) | [Analysis](#what-explains-the-results) |
[Reproduction](#reproduction) |
[Community Reproduction](#community-reproduction) |
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
Benchmark-specific settings are listed in the
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

*The overview shows selected comparators, with sector-wise truncated accuracy
scales.*

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

## Reproduction

Commands in steps 1-5 run from the repository root.

### 1. Install

Use Python 3.12 and PowerShell 7 (`pwsh`):

```powershell
python -m pip install -r methods/EverCore/requirements-eval.txt
python -m nltk.downloader punkt_tab
```

For analysis and tests:

```powershell
python -m pip install -r methods/EverCore/requirements-analysis.txt
python -m pip install -r methods/EverCore/requirements-test.txt
```

### 2. Configure Services

Create `methods/EverCore/.env` from the
[template](methods/EverCore/.env.example):

```powershell
if (-not (Test-Path methods/EverCore/.env)) {
  Copy-Item methods/EverCore/.env.example methods/EverCore/.env
}
```

Set credentials and service endpoints:

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

The [paper profiles](scripts/paper/README.md) specify models and prompts.
DeepSeek readers use `DEEPSEEK_*`; GPT readers and judges use `LLM_*`.

### 3. Prepare Data

Place benchmark files under `methods/EverCore/evaluation/data`.
MemoryAgentBench is downloaded automatically:

| Benchmark | File or preparation | Expected questions |
| --- | --- | ---: |
| LoCoMo | `locomo/locomo10.json` | 1,540 non-adversarial |
| LongMemEval-S | `longmemeval/longmemeval_s_cleaned.json` | 500 |
| MEME | `meme/meme_filler32k.json` | 1,188 from 100 episodes |
| MemoryAgentBench | Automatically cached under `memoryagentbench/` for 14 sources | 2,071 |

### 4. Run the Main Experiments

Preview the seven main conditions, then run them:

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main -List
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main -Tag paper
```

`-Tag` names the experiment. Run experiments sequentially within one checkout.

### 5. Inspect Results

Results are written under `methods/EverCore/evaluation/results`.
Each run contains `report.txt`, `eval_results.json`, `answer_results.json`,
and `search_results.json`. MemoryAgentBench aggregate scores are saved as
`mab_accuracy.txt` under `evaluation/results/paper-configs/<Tag>-<condition>/`.

### Optional: Appendix Conditions

`-Group All` runs the main and appendix conditions, including LoCoMo J/F1/BLEU-1
reports. Configure the Qwen services in the [paper profiles](scripts/paper/README.md)
and add these PersonaMem-v2 files under `methods/EverCore/evaluation/data`:

- `personamem_v2/benchmark/text/benchmark.csv`
- `personamem_v2/data/chat_history_32k/*.json`

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All -Tag paper-all
```

## Community Reproduction

We thank the GoodMem team for sharing an independent, GoodMem-based reproduction
of our LoCoMo evaluation, including code and per-question results, in
their [goodmem-memory-benchmarks repository](https://github.com/PAIR-Systems-Inc/goodmem-memory-benchmarks).
Using the embedding model, reranker, and retrieval depths specified in Table 10
of our paper, they report the following scores:

| Reader | RAW-RAG, Table 10 | GoodMem reproduction |
| --- | ---: | ---: |
| GPT-4o-mini | 89.07 | 88.68 |
| GPT-4.1-mini | 93.70 | 93.48 |

## Custom Runs

For custom runs, set the dataset, models, and prompts under
`methods/EverCore/evaluation/config`, then run from `methods/EverCore`.
RAW-RAG uses the system name `rag`:

```powershell
python -m evaluation.cli --dataset locomo --system rag
python -m evaluation.cli --dataset longmemeval --system rag
python -m evaluation.cli --dataset meme --system rag
python -m evaluation.cli --dataset personamem_v2 --system rag
python -m evaluation.cli --dataset memoryagentbench --system rag
```

MemoryAgentBench ablations:

```powershell
python -m evaluation.cli --dataset memoryagentbench --system rag_atomic_facts
python -m evaluation.cli --dataset memoryagentbench --system rag_raw_summary
python -m evaluation.cli --dataset memoryagentbench --system rag_graph_expanded_raw
```

## Timing Experiment

For sequential BEAM-1M ingestion timing, prepare the conversations, configure
RAW-RAG, and run from `methods/EverCore`:

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

## License

The code in this repository is released under the [MIT License](LICENSE).
Third-party dependencies and datasets retain their respective licenses.

## Citation

If you find this work helpful for your research, please consider citing:

```bibtex
@misc{bao2026complexagenticmemory,
  title         = {Is Complex Agentic Memory Always Needed? Naive {RAG} Is Stronger Than We Thought},
  author        = {Bao, Zhengye and Zhang, Shengtao and Zhang, Haokai and Lou, Shiqi and Wen, Muning and Luo, Minnan},
  year          = {2026},
  note          = {arXiv preprint},
  eprint        = {2610.30755},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2610.30755}
}
```
