# EverCore Evaluation Framework

A unified, modular evaluation framework for benchmarking memory systems on standard datasets.

### LongMemEval judge modes

Use `--lme-judge evermemos` for the existing generic JSON-label judge (three
judgments), or `--lme-judge official` for the
[upstream LongMemEval judge](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py).
Omitting the flag preserves the existing dataset/system configuration.
The flag is only valid with `--dataset longmemeval` and applies to full runs,
answer/evaluate runs, and evaluate-only runs.

From `methods/EverCore`, re-evaluate saved answers with:

```powershell
python -m evaluation.cli --dataset longmemeval --system rag --stages evaluate --output-dir evaluation/results/longmemeval-rag --lme-judge official
```

This reads `answer_results.json` and replaces `eval_results.json` and `report.txt`
in that directory. To retain both judgments, copy `answer_results.json` into a
separate directory and use that directory as `--output-dir`.

Official mode uses the upstream prompts verbatim, routes by question category,
and detects abstention questions using `_abs` in the original question ID.
It sends one user message with `n=1`, `temperature=0`, `max_tokens=10`, performs
one judgment per question, and uses the upstream `'yes' in response.lower()` rule.
Overall and per-category accuracy use that single judgment. Raw judge responses
and the mode/model/request settings are recorded in the JSON results.
API failures abort evaluation rather than count as incorrect answers; transport
retries use the OpenAI SDK's bounded retry policy rather than upstream backoff.

The default official-mode model is `gpt-4o-2024-08-06`. `EVALUATE_MODEL` explicitly
overrides it (`gpt-4o` and `gpt-4o-mini` map to upstream snapshot names).
API credentials and endpoint continue to use `LLM_API_KEY` and `LLM_BASE_URL`.
When comparing scores, report the actual judge model as well as the judge mode.

## 📖 Overview

### Evaluation Scope

In addition to **EverCore**, this framework supports evaluation of several influential memory systems in the industry:
- **Mem0** 
- **MemOS** 
- **Zep** 
- **MemU** 

These systems were selected based on recent industry benchmarks and their prominence in global markets. Since many commercial systems have web-based optimizations not available in their open-source versions, we evaluate them through their **online API interfaces** to ensure fair comparison with production-grade capabilities.

### Implementation

Our adapter implementations are based on:
- **Official open-source repositories**: Mem0, MemOS, Zep on GitHub
- **Official documentation**: Mem0, MemOS, MemU, Zep quick start guide and API documentation
- **Consistent methodology**: All systems evaluated using the same pipeline, datasets, and metrics
- **Unified answer generation**: All systems use **GPT-4.1-mini** as the answer LLM to ensure fair comparison across different memory backends

During our evaluation, we identified several issues in existing open-source reference implementations for benchmarking these systems that could negatively impact their performance. We addressed these implementation gaps to ensure each system is evaluated at its best potential:

- **Mem0 timezone handling**: The latest version returns timestamps in PDT format in search results, requiring additional timezone conversion for accurate temporal reasoning.

- **MemU retrieval enhancement**: While some memories are visible in the backend dashboard, the `/memory/retrieve/related-memory-items` API likely relies on simple vector-based retrieval, which may miss relevant context. Following the official documentation examples, we included category summaries as additional context to improve recall.

- **Zep API migration**: Zep's official open-source evaluation implementation was based on the earlier v2 API. Since Zep has officially upgraded to v3 API, we migrated the evaluation code to v3 following the official documentation to benchmark the latest capabilities.

- **Zep timestamp semantics**: Unlike most memory systems that record conversation timestamps, Zep records event occurrence timestamps. For example, a conversation on March 2nd mentioning "Anna ate a burger yesterday" would be timestamped March 1st, with the memory content preserving the original phrasing. Using standard answer prompts leads to significant errors on temporal questions. Zep's team provides optimized prompts in their open-source evaluation code to handle this. This informed one of our evaluation principles: **each memory system uses its own official answer prompts** rather than a unified prompt template, ensuring fair evaluation of each system's intended usage.
 
 


### Evaluation Results
**Results on Locomo**

| Locomo    | single hop | multi hop | temporal | open domain | Overall | Average Tokens | Version                                         | Answer LLM |
|-----------|------------|-----------|----------|-------------|---------|----------------|----------------------------------------------|-----------------|
| Full-context | 94.93      | 90.43     | 87.95    | 71.88       | 91.21   | 20281          |                                              | gpt-4.1-mini    |
| Mem0      | 68.97      | 61.70     | 58.26    | 50.00       | 64.20   | 1016           | web API/v1.0.0 (2025.11)                   | gpt-4.1-mini    |
| Zep       | 90.84      | 81.91     | 77.26    | 75.00       | 85.22   | 1411           | web API/v3 (2025.11)                       | gpt-4.1-mini    |
| MemOS     | 85.37      | 79.43     | 75.08    | 64.58       | 80.76   | 2498           | web API/v1 (2025.11)                       | gpt-4.1-mini    |
| MemU      | 74.91      | 72.34     | 43.61    | 54.17       | 66.67   | 3964           | web API/v1 (2025.11)                      | gpt-4.1-mini    |
| EverCore | 96.08      | 91.13     | 89.72    | 70.83       | 92.32   | 2298           | open-source EverCore v1.0.0 companion | gpt-4.1-mini    |

*Full-context: using the whole conversation as context for answering questions.


**Results on Longmemeval**

| Longmemeval | Single-session-user  | Single-session-assistant  | Single-session-preference  | Multi-session  | Knowledge-update  | Temporal-reasoning  | Overall |
|-------------|----------------------|---------------------------|----------------------------|----------------|-------------------|---------------------|---------|
| EverCore   | 100.00               | 78.57                     | 96.67                      | 78.45          | 87.18             | 71.18               | 82.00   |

> **Note on Reproducibility**: To ensure the reproducibility of our evaluation, we provide full evaluation intermediate data for all methods. You can access the data at [EverMind-AI/EverMemOS_Eval_Results](https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results).


## 🌟 Key Features

### Unified & Modular Framework
- **One codebase for all**: No need to write separate code for each dataset or system
- **Plug-and-play systems**: Support multiple memory systems (EverCore, Mem0, MemOS, MemU, etc.)
- **Multiple benchmarks**: LoCoMo, LongMemEval, PersonaMem out of the box
- **Consistent evaluation**: All systems evaluated with the same pipeline and metrics

### Automatic Compatibility Detection
The framework automatically detects and adapts to:
- **Multi-user vs Single-user conversations**: Handles both conversation types seamlessly
- **Q&A vs Multiple-choice questions**: Adapts evaluation approach based on question format  
- **With/without timestamps**: Works with or without temporal information

### Robust Checkpoint System
- **Cross-stage checkpoints**: Resume from any pipeline stage (add → search → answer → evaluate)
- **Fine-grained resume**: Saves progress every conversation (search) and every 400 questions (answer)


## 🏗️ Architecture Overview

### Code Structure

```
evaluation/
├── src/
│   ├── core/           # Pipeline orchestration and data models
│   ├── adapters/       # System-specific implementations
│   ├── evaluators/     # Answer evaluation (LLM Judge, Exact Match)
│   ├── converters/     # Dataset format converters
│   └── utils/          # Configuration, logging, I/O
├── config/
│   ├── datasets/       # Dataset configurations (locomo.yaml, etc.)
│   ├── systems/        # System configurations (evermemos.yaml, etc.)
│   └── prompts.yaml    # Prompt templates
├── data/               # Benchmark datasets
└── results/            # Evaluation results and logs
```

### Pipeline Flow

The evaluation consists of 4 sequential stages:

1. **Add**: Ingest conversations and build indexes
2. **Search**: Retrieve relevant memories for each question
3. **Answer**: Generate answers using retrieved context
4. **Evaluate**: Assess answer quality with LLM Judge or Exact Match

Each stage saves its output and can be resumed independently.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- EverCore environment configured (see main project's `env.template`)

### Data Preparation

Place your dataset files in the `evaluation/data/` directory:

**LoCoMo** (native format, no conversion needed):
Get data from: https://github.com/snap-research/locomo/tree/main/data

```
evaluation/data/locomo/
└── locomo10.json
```

**LongMemEval** (auto-converts to LoCoMo format):
Get data from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

```
evaluation/data/longmemeval/
└── longmemeval_s_cleaned.json  # Original file
# → Will auto-generate: longmemeval_s_locomo_style.json
```

**PersonaMem v1** (legacy, auto-converts to LoCoMo format):
Get data from: https://huggingface.co/datasets/bowen-upenn/PersonaMem

```
evaluation/data/personamem/
├── questions_32k.csv           # Original file
└── shared_contexts_32k.jsonl   # Original file
# → Will auto-generate: personamem_32k_locomo_style.json
```

**PersonaMem v2** (official 5000-question benchmark, auto-converts to LoCoMo format):
Get data from: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2

```
evaluation/data/personamem_v2/
├── benchmark/text/benchmark.csv          # 5000 benchmark rows
└── data/chat_history_32k/*.json          # 200 unique chat histories referenced by benchmark.csv
# → Multiple-choice mode auto-generates: personamem_v2_32k_locomo_style.json
# → Open-ended mode auto-generates: personamem_v2_32k_open_ended_locomo_style.json
```

PersonaMem v2 has two supported evaluation modes:

- `personamem_v2`: multiple-choice mode. The converter builds stable `(a)`-`(d)` options from `correct_answer` and `incorrect_answers`; evaluation uses exact match on the chosen option label.
- `personamem_v2_open_ended`: open-ended mode. The converter uses the original `user_query` content directly, does not expose answer options, uses `correct_answer` as the gold text answer, and evaluates generated answers with an LLM judge.

The framework will automatically detect and convert non-LoCoMo formats on first run. You don't need to manually run any conversion scripts.

### Installation

Install evaluation-specific dependencies:

```bash
# For evaluating local systems (EverCore)
uv sync --group evaluation

# For evaluating online API systems (Mem0, MemOS, MemU, etc.)
uv sync --group evaluation-full
```

### Environment Configuration

The evaluation framework reuses most environment variables from the main EverCore `.env` file:
- `LLM_API_KEY`, `LLM_BASE_URL` (for answer generation with GPT-4.1-mini)
- `VECTORIZE_API_KEY` and  `RERANK_API_KEY` (for embeddings/reranker)

**⚠️ Important**: For OpenRouter API (used by gpt-4.1-mini), make sure `LLM_API_KEY` is set to your OpenRouter API key (format: `sk-or-v1-xxx`). The system will look for API keys in this order:
1. Explicit `api_key` parameter in config
2. `LLM_API_KEY` environment variable

For testing EverCore, please first configure the whole .env file.

**Optional RAG pgvector storage**: `config/systems/rag.yaml` defaults to local
`rag_index.json` embedding storage. To store RAG embeddings in a local
`pgvector/pgvector:pg17` container instead, set:

```bash
RAG_PGVECTOR_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
# Optional: defaults to the output directory/run name when empty.
RAG_PGVECTOR_COLLECTION=
RAG_PGVECTOR_TABLE=rag_vectors
```

Then set `storage.backend: "pgvector"` or `storage.pgvector.enabled: true` in
`config/systems/rag.yaml` and run with a fresh `--run-name`. The adapter creates
the pgvector extension, table, and vector index automatically.

**Additional variables for online API systems** (add to `.env` if testing these systems):

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# MemOS
MEMOS_KEY=your_memos_api_key

# MemOS local Docker (optional)
MEMOS_LOCAL_URL=http://localhost:8000
MEMOS_LOCAL_KEY=

# MemU
MEMU_API_KEY=your_memu_api_key
```

### Quick Test (Smoke Test)

Run a quick test with limited data to verify everything works:

```bash
# Navigate to project root
cd /path/to/memsys-opensource

# Default: first conversation, first 10 messages, first 3 questions 
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom: first conversation, 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5

# You can also evaluate specific conversations by 0-based conversation index (end exclusive):
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom --conv-range 0:1
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom --from-conv 0 --to-conv 1
```


### Full Evaluation

Run the complete benchmark:

```bash
# Evaluate EverCore on LoCoMo
uv run python -m evaluation.cli --dataset locomo --system evermemos

# Evaluate EverCore via local API (start server first)
uv run python src/run.py
# Use --clean-groups to clear existing data before Add stage
uv run python -m evaluation.cli --dataset locomo --system evermemos_local_api --clean-groups

# Evaluate other systems
uv run python -m evaluation.cli --dataset locomo --system memos
uv run python -m evaluation.cli --dataset locomo --system memos_local
uv run python -m evaluation.cli --dataset locomo --system memu
# Local memU v1.5.1 (use the dedicated Python 3.13 memu-py313 environment)
python -m evaluation.cli --dataset locomo --system memu_local --run-name memu-local-YYYYMMDD-HHMM
# For mem0, it's recommended to run add first, check the memory status on the web console to make sure it's finished and then following stages.
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages add
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages search answer evaluate

# Evaluate on other datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# Use --run-name to distinguish multiple runs (useful for A/B testing)
# Results will be saved to: results/{dataset}-{system}-{run-name}/
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name 20241107

# Resume from checkpoint if interrupted (automatic)
# Just re-run the same command - it will detect and resume from checkpoint
uv run python -m evaluation.cli --dataset locomo --system evermemos

```

### Local MemOS Docker

`memos_local` is for MemTensor/MemOS, the memory management system, not the
`usememos` note-taking app. The current open-source Docker server exposes the
product API (`/product/add`, `/product/search`), while the cloud API uses
OpenMem paths. The `memos_local.yaml` config sets `api_flavor: product` for
that local server shape.

From the repository root:

```powershell
.\scripts\memos\start_memos_local.ps1 -Build
.\scripts\memos\smoke_memos_local.ps1
```

Then run evaluation from `methods/EverCore` with a fresh run name:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd methods\EverCore
python -m evaluation.cli --dataset locomo --system memos_local --run-name memos-local-YYYYMMDD-HHMM
```

### Local Microsoft Memora

Memora is pinned to Microsoft commit
`dec3f8f2444eace7004fc084abe1be9f3d88270e`. From the repository root, fetch
the upstream checkout and install the adapter's runtime dependencies into the
Python 3.12 environment:

```powershell
.\scripts\memora\fetch_memora.ps1
python -m pip install -r methods\EverCore\requirements-memora-local.txt
```

Run each benchmark from `methods/EverCore`, always choosing a new run name:

```powershell
python -m evaluation.cli --dataset locomo --system memora --run-name memora-locomo-YYYYMMDD-HHMM
python -m evaluation.cli --dataset meme --system memora --run-name memora-meme-YYYYMMDD-HHMM
python -m evaluation.cli --dataset memoryagentbench --system memora --run-name memora-mab-YYYYMMDD-HHMM --mab-split Accurate_Retrieval --mab-source eventqa_full
```

The adapter keeps Memora's native construction and prompted retrieval, but
uses the same normalized inputs, `top_k=20`, answer model/prompt routing, and
benchmark evaluators as `mem0_local`. Each conversation has an isolated Chroma
collection. MEME question dates are applied after over-fetching so a before
question cannot read later sessions. MemoryAgentBench continues to use the
shared official source-aware converter and prompt/evaluation routing.

For MemoryAgentBench, the CLI supervises Memora in a child process. During
Add, the worker persists its normal per-session resume checkpoint and exits
after every 100 newly completed sessions; the supervisor immediately restarts
it in the same logical run. The final Add batch also recycles once so Search
starts without the native Chroma/HNSW memory retained by ingestion.

### View Results

Results are saved to `evaluation/results/{dataset}-{system}[-{run-name}]/`:

```bash
# View summary report
cat evaluation/results/locomo-evermemos/report.txt

# View detailed evaluation results
cat evaluation/results/locomo-evermemos/eval_results.json

# View pipeline execution log
cat evaluation/results/locomo-evermemos/pipeline.log
```

**Result files:**
- `report.txt` - Summary metrics (accuracy, total questions)
- `eval_results.json` - Per-question evaluation details
- `answer_results.json` - Generated answers and retrieved context
- `search_results.json` - Retrieved memories for each question
- `pipeline.log` - Detailed execution logs

## 📊 Understanding Results

### Metric

- **Accuracy**: Percentage of correct answers (QA judged by LLM, multiple choice questions judged by exact match)

### Detailed Results

Check `eval_results.json` for per-question breakdown:

**LoCoMo example (Q&A format, evaluated by LLM Judge):**

```json
{
  "total_questions": ...,
  "correct": ...,
  "accuracy": ...,
  "detailed_results": {
      "locomo_exp_user_0": [
         {
            "question_id": "locomo_0_qa0",
            "question": "What is my favorite food?",
            "golden_answer": "Pizza",
            "generated_answer": "Your favorite food is pizza.",
            "judgments": [
               true,
               true,
               true
            ],
            "category": "1"
         }
         ...
      ]
  }
}
```

**PersonaMem example (Multiple-choice format, evaluated by Exact Match):**

```json
{
  "overall_accuracy": ...,
  "total_questions": ...,
  "correct_count": ...,
  "detailed_results": [
    {
      "question_id": "acd74206-37dc-4756-94a8-b99a395d9a21",
      "question": "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.",
      "golden_answer": "(c)",
      "generated_answer": "(c)",
      "is_correct": true,
      "category": "recall_user_shared_facts"
    }
    ...
  ]
}
```

**PersonaMem v2 open-ended example (evaluated by LLM Judge):**

```bash
python -m evaluation.cli --dataset personamem_v2_open_ended --system rag --run-name <unique-run-name>
```

Use a fresh `--run-name` for each trial so existing result directories are not overwritten.

### Sequential BEAM-1M ingestion timing

Normalize the 35 raw BEAM-1M conversations once:

```powershell
python -m evaluation.scripts.convert_beam_1m_sequence
```

Then run all conversations through the sequential per-message Add benchmark:

```powershell
python -m evaluation.speed_sequence_cli `
  --system rag `
  --run-name beam1m-rag-seq-<unique-suffix> `
  --data-dir ..\..\beam\BEAM-1M
```

The default `--conv-range :` selects all numbered files. For a smoke test of
only the first conversation, add `--conv-range 0:1` and optionally
`--max-items-per-conversation 10`.

## SimpleMem (LoCoMo only)

`--system simplemem` uses the text components from
[aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem), pinned at
`db80b6a7c591e0ea730a058e9f5fc4eb06572299` under `external/SimpleMem`.
It runs upstream `MemoryBuilder`, `HybridRetriever` (planning + reflection),
and LanceDB/Tantivy storage, then uses this framework's shared answer prompt
and LoCoMo judge. It does not use SimpleMem's own answer generator or its
multimodal/evolution pipelines.

Fetch the pinned checkout from the repository root:

```powershell
.\scripts\simplemem\fetch_simplemem.ps1
```

Install the text evaluation dependencies into a Python 3.12 environment:

```powershell
cd methods\EverCore
python -m pip install -r requirements-simplemem.txt
```

Use a Python 3.12 virtual environment. From `methods/EverCore`:

```powershell
$env:PYTHONUTF8 = '1'
python -m evaluation.cli --dataset locomo --system simplemem --smoke --locomo-sample 0 --smoke-questions 1 --run-name simplemem-smoke
```

Run all LoCoMo conversations with a separate run name:

```powershell
python -m evaluation.cli --dataset locomo --system simplemem --run-name simplemem-full
```

Configuration: `evaluation/config/systems/simplemem.yaml`. Existing `.env`
values `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`, `VECTORIZE_MODEL`,
`VECTORIZE_API_KEY`, and `VECTORIZE_BASE_URL` are reused. Embeddings use the
configured OpenAI-compatible API and normalized vectors, **not upstream's
default local Qwen3 model**. This configuration is for comparison within the
existing evaluation protocol, not a reproduction of the paper's scores.
The upstream extraction prompt is retained; memory windows run sequentially
within each conversation (40 messages, overlap 2), with two conversations in
parallel by default. `--sample-workers` changes conversation concurrency.
Retrieval preserves upstream merge order and passes at most 20 memory entries
to the shared reader; `--search-workers` controls query concurrency.

Each run stores isolated conversation databases under `simplemem_store/` and
the usual `search_results.json`, `answer_results.json`, `eval_results.json`,
and `report.txt`. Repeat the same command to resume. A completed conversation
has a `complete.json` marker; a failed conversation is rebuilt without
duplicating partially stored memories. Exhausted extraction failures (including
empty extraction windows) fail Add instead of marking incomplete memory as
complete. The Tantivy index is built after all windows to cover every write
and avoid Windows index replacement locks. Do not run simultaneous CLI
processes against the same output directory. Use a fresh run name when changing
input limits, embedding models, or memory settings, and for a full run after a
smoke test. To reuse Add in a later process, keep the same input selection and
run name and pass `--stages search answer evaluate`.

Adapter integration tests use real upstream construction and LanceDB storage
with mocked network calls:

```powershell
python -m pip install pytest
python -m pytest tests/test_simplemem_adapter.py
```

## 🔧 Advanced Usage

### Run Specific Stages

Skip completed stages to iterate faster:

```bash
# Only run search stage (if add is already done)
uv run python -m evaluation.cli --dataset locomo --system evermemos --stages search

# Run search, answer, and evaluate (skip add)
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --stages search answer evaluate
```
If you have already done search, and you want to do it again, please remove the "search" (and following stages from the completed_stages in the checkpoint_default.json file):
```
  "completed_stages": [
    "answer",
    "search",
    "evaluate",
    "add"
  ]
```


### Custom Configuration

Modify system or dataset configurations:

```bash
# Copy and edit configuration
cp evaluation/config/systems/evermemos.yaml evaluation/config/systems/evermemos_custom.yaml
# Edit evermemos_custom.yaml with your changes

# Run with custom config
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom
```

## 📄 License

Same as the parent project.
# Full context baseline

`full_context` sends the complete visible conversation to the answer model,
without embeddings or retrieval. It supports LoCoMo, MEME (including nofiller
and filler128k), LongMemEval (`longmemeval`, alias `lme`), and MemoryAgentBench.
Run these commands from `methods/EverCore`, with `LLM_API_KEY`, `LLM_BASE_URL`
and optionally `LLM_MODEL` set as for the other systems:

```bash
python -m evaluation.cli --dataset locomo --system full_context --context-window 128000 --run-name cw128k
python -m evaluation.cli --dataset meme --system full_context --context-window 128000 --run-name cw128k
python -m evaluation.cli --dataset lme --system full_context --context-window 128000 --run-name cw128k
python -m evaluation.cli --dataset memoryagentbench --system full_context --context-window 128000 --run-name cw128k
```

The total window includes the fully rendered prompt, question, chat framing,
reserved output (`llm.max_tokens`, default 4096), and safety margin (default 256).
Configure `answer.context_window`, `answer.context_safety_margin`, and
`answer.truncation_strategy` in `config/systems/full_context.yaml`. Strategies
are `head_tail` (default, retain both ends), `head`, and `tail`. Truncation only
changes history; an oversized question/prompt fails before an API call.
Use the actual window supported by your chosen model; this option does not
change the provider's limit.

Known models use tiktoken; `answer.tokenizer_model` can specify a tiktoken model
or encoding name. Unknown models default to conservative UTF-8 byte counting.
An encoding from a different model is only an estimate: select a matching
encoding and increase the margin if the provider uses different chat framing.

The adapter preserves message order, speakers and timestamps, filters history
by question date when supplied, and uses the existing dataset answer prompts
and evaluators. It disables the loader's per-message character cap and
MemoryAgentBench ingestion chunking. Search-only runs reconstruct history from
the dataset; answer-only reruns retain the same window enforcement.
`answer_results.json` records the sent history in `formatted_context` and the
budget/truncation audit in `metadata.context_budget`. Use a new run name when
changing window settings so completed checkpoints are not reused.
