# Evaluation Framework

Shared `add -> search -> answer -> evaluate` pipeline for RAW-RAG and memory
baselines. Paper settings are in the
[RAW-RAG profiles](../../../scripts/paper/README.md).
Commands below run from `methods/EverCore` unless stated otherwise.

## Configuration

| Location | Contents |
| --- | --- |
| `config/datasets/` | Dataset paths, conversion, and evaluation settings |
| `config/systems/` | Memory systems, retrieval, and reader settings |
| `config/prompts.yaml` | Prompt templates |
| `src/adapters/` | System adapters |
| `src/converters/` | Dataset converters |
| `src/evaluators/` | Benchmark scoring |

Install `requirements-eval.txt` and configure `.env` using
[`.env.example`](../.env.example). Readers and judges use `LLM_API_KEY` and
`LLM_BASE_URL`; retrieval uses `VECTORIZE_*` and `RERANK_*`.
External systems have their own dependencies and credentials.

## Data

Paths are relative to `evaluation/data`.

| Dataset | Input |
| --- | --- |
| LoCoMo | `locomo/locomo10.json` |
| LongMemEval | `longmemeval/longmemeval_s_cleaned.json` |
| MEME | `meme/meme_filler32k.json` |
| PersonaMem v1 | `personamem/questions_32k.csv`, `personamem/shared_contexts_32k.jsonl` |
| PersonaMem v2 | `personamem_v2/benchmark/text/benchmark.csv`, `personamem_v2/data/chat_history_32k/*.json` |
| MemoryAgentBench | Source files selected in `config/datasets/memoryagentbench.yaml` |

Non-LoCoMo datasets are converted on loading.
`personamem_v2` uses multiple-choice exact match;
`personamem_v2_open_ended` uses the original user query, a gold text answer,
and an LLM judge.

## Evaluation

```powershell
python -m evaluation.cli --dataset locomo --system rag
python -m evaluation.cli --dataset longmemeval --system rag
python -m evaluation.cli --dataset meme --system rag
python -m evaluation.cli --dataset memoryagentbench --system rag
python -m evaluation.cli --dataset personamem_v2 --system rag
```

`--system` selects a YAML file in `config/systems`.
`--run-name` identifies an experiment; rerunning resumes its checkpoint.
`--conv-range` selects a zero-based, end-exclusive conversation range.
`--stages` selects pipeline stages:

```powershell
python -m evaluation.cli --dataset locomo --system rag --run-name baseline --conv-range 0:1
python -m evaluation.cli --dataset locomo --system rag --run-name baseline --stages search answer evaluate
```

## LongMemEval Judges

`--lme-judge evermemos` selects the generic JSON-label judge with three
judgments. `--lme-judge official` selects the upstream LongMemEval protocol.
Without this flag, dataset and system configuration determine the judge.

```powershell
python -m evaluation.cli --dataset longmemeval --system rag --stages evaluate --output-dir evaluation/results/longmemeval-rag --lme-judge official
```

Evaluation reads `answer_results.json` and replaces `eval_results.json`
and `report.txt` in the output directory.

Official mode uses category-specific prompts and detects abstention questions
by `_abs` in the original question ID. Each question receives one judgment
with `n=1`, `temperature=0`, and `max_tokens=10`; correctness follows
`'yes' in response.lower()`.
The default model is `gpt-4o-2024-08-06`, overridable by `EVALUATE_MODEL`.
Credentials use `LLM_API_KEY` and `LLM_BASE_URL`.

## Full-Context Baseline

`full_context` supplies visible conversation history directly to the reader:

```powershell
python -m evaluation.cli --dataset locomo --system full_context --context-window 128000
python -m evaluation.cli --dataset meme --system full_context --context-window 128000
python -m evaluation.cli --dataset longmemeval --system full_context --context-window 128000
python -m evaluation.cli --dataset memoryagentbench --system full_context --context-window 128000
```

The window includes the rendered prompt, question, chat framing, reserved
output (default 4,096 tokens), and margin (default 256).
`config/systems/full_context.yaml` sets `answer.context_window`,
`answer.context_safety_margin`, `answer.tokenizer_model`, and
`answer.truncation_strategy`: `head_tail` (default), `head`, or `tail`.
Truncation applies to history. Known models use tiktoken; unknown models use
UTF-8 byte counting.

History retains message order, speakers, timestamps, and question-date
filtering. Sent context and budget details are recorded in
`answer_results.json`.

## Local Baselines

### SimpleMem

Uses the text pipeline from SimpleMem commit
`db80b6a7c591e0ea730a058e9f5fc4eb06572299`.
Fetch from the repository root:

```powershell
.\scripts\simplemem\fetch_simplemem.ps1
```

From `methods/EverCore`:

```powershell
python -m pip install -r requirements-simplemem.txt
python -m evaluation.cli --dataset locomo --system simplemem
```

The adapter uses upstream memory construction, planning/reflection retrieval,
and LanceDB/Tantivy storage with the shared reader and LoCoMo judge.
Embedding models and API endpoints are configured in `.env`.
Memory windows contain 40 messages with overlap 2; retrieval returns at most
20 entries. `--sample-workers` and `--search-workers` control concurrency.
Conversation databases are stored under `simplemem_store/`.

### Memora

Uses Microsoft Memora commit `dec3f8f2444eace7004fc084abe1be9f3d88270e`.
Fetch from the repository root:

```powershell
.\scripts\memora\fetch_memora.ps1
```

From `methods/EverCore`:

```powershell
python -m pip install -r requirements-memora-local.txt
python -m evaluation.cli --dataset locomo --system memora
python -m evaluation.cli --dataset meme --system memora
python -m evaluation.cli --dataset memoryagentbench --system memora --mab-split Accurate_Retrieval --mab-source eventqa_full
```

The adapter uses native construction and prompted retrieval with `top_k=20`,
isolated Chroma collections, shared readers, and benchmark evaluators.
MEME retrieval filters by question date.

### MemOS

Start the local service from the repository root:

```powershell
.\scripts\memos\start_memos_local.ps1 -Build
```

From `methods/EverCore`:

```powershell
python -m evaluation.cli --dataset locomo --system memos_local
```

`memos_local.yaml` uses `api_flavor: product` for the local
`/product/add` and `/product/search` endpoints.

## RAG Storage

RAG defaults to local `rag_index.json` storage. For pgvector, set
`storage.backend: pgvector` in `config/systems/rag.yaml` and configure:

```dotenv
RAG_PGVECTOR_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
RAG_PGVECTOR_COLLECTION=
RAG_PGVECTOR_TABLE=rag_vectors
```

An empty collection name defaults to the output directory/run name.

## Sequential Ingestion Timing

```powershell
python -m evaluation.scripts.convert_beam_1m_sequence
python -m evaluation.speed_sequence_cli `
  --system rag `
  --run-name beam1m-rag-seq `
  --data-dir ..\..\beam\BEAM-1M
```

The benchmark processes 35 BEAM-1M conversations sequentially, per message.
`--conv-range` selects conversations.

## Results

Outputs: `evaluation/results/{dataset}-{system}[-{run-name}]/`.

| File | Contents |
| --- | --- |
| `report.txt` | Aggregate metrics |
| `eval_results.json` | Per-question scores |
| `answer_results.json` | Answers and reader context |
| `search_results.json` | Retrieved evidence |
| `pipeline.log` | Execution log |

### Upstream Reference Results

EverCore reference evaluation with GPT-4.1-mini, separate from the RAW-RAG
paper results:

| LoCoMo | Single hop | Multi hop | Temporal | Open domain | Overall | Average tokens | Version |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full-context | 94.93 | 90.43 | 87.95 | 71.88 | 91.21 | 20281 | |
| Mem0 | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 | 1016 | web API/v1.0.0 (2025.11) |
| Zep | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 | 1411 | web API/v3 (2025.11) |
| MemOS | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 | 2498 | web API/v1 (2025.11) |
| MemU | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 | 3964 | web API/v1 (2025.11) |
| EverCore | 96.08 | 91.13 | 89.72 | 70.83 | 92.32 | 2298 | EverCore v1.0.0 companion |

| LongMemEval | Single-session user | Single-session assistant | Single-session preference | Multi-session | Knowledge update | Temporal reasoning | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EverCore | 100.00 | 78.57 | 96.67 | 78.45 | 87.18 | 71.18 | 82.00 |
