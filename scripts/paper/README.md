# RAW-RAG paper evaluation profiles

Profiles for the RAW-RAG main and appendix evaluations.
Requires PowerShell 7 and the [evaluation dependencies](../../README.md#1-install).
Configure [services](#services-and-credentials) and
[prepare the benchmark data](../../README.md#3-prepare-data) before running.
Runner commands use the repository root.

## Run

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Appendix
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 `
  -Condition "locomo-textemb3-gpt4omini,locomo-textemb3-gpt4omini-minimal" `
  -Tag paper-v1
```

`Main` is the default group; `All` includes main and appendix conditions.
`-Condition` selects individual conditions; `-Python` selects the interpreter;
`-Tag` names the experiment. Minimal, Flash, and non-abstention conditions reuse
matching runs with the same tag. Run experiments sequentially within one checkout.

## Conditions

`textemb3` means text-embedding-3-small + Cohere Rerank-v3.5;
`qwen3` means Qwen3-Embedding-4B + Qwen3-Reranker-4B.
Retrieval uses 256-token windows with 32-token overlap, 128 candidates, and
15 selected windows. MemoryAgentBench keeps fact-consolidation units atomic
and uses the selected windows without turn expansion. MEME presents evidence
chronologically.

| ID | Reader | Judge / scoring | Execution |
| --- | --- | --- | --- |
| locomo-textemb3-gpt4omini | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-textemb3-gpt41mini | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| longmemeval-textemb3-gpt41mini | GPT-4.1-mini | EverMemOS-style, GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| meme-textemb3-gpt4omini | GPT-4o-mini | GPT-4o-2024-11-20 x3 | Full pipeline |
| meme-textemb3-flash | DeepSeek-V4-Flash | GPT-4o-2024-11-20 x1 | Reuse main MEME retrieval; answer/evaluate |
| mab-textemb3-gpt4omini | GPT-4o-mini | Native metrics; GPT-4o where needed | 14 sources + aggregate |
| mab-textemb3-flash | DeepSeek-V4-Flash | Native metrics; GPT-4o-2024-11-20 where needed | Reuse main MAB retrieval; answer/evaluate |
| locomo-qwen3-gpt4omini | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-qwen3-gpt41mini | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| longmemeval-qwen3-gpt41mini | GPT-4.1-mini | EverMemOS-style, GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-textemb3-gpt4omini-minimal | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3; F1/BLEU-1 | Reuse matching LoCoMo retrieval; answer/evaluate; offline lexical scoring |
| locomo-textemb3-gpt41mini-minimal | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3; F1/BLEU-1 | Reuse matching LoCoMo retrieval; answer/evaluate; offline lexical scoring |
| longmemeval-textemb3-gpt41mini-noabs | Existing GPT-4.1-mini answers | Existing judgments, 470-question subset | Report only |
| longmemeval-qwen3-gpt41mini-noabs | Existing GPT-4.1-mini answers | Existing judgments, 470-question subset | Report only |
| personamem-v2-qwen3-gpt41mini | GPT-4.1-mini | Exact-match multiple choice | Full pipeline |

## Services and credentials

Credentials and endpoints are read from `methods/EverCore/.env` or the process
environment.

```dotenv
# OpenAI-compatible GPT readers and judges.
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1

# textemb3 / Cohere retrieval.
VECTORIZE_API_KEY=...
VECTORIZE_BASE_URL=https://api.openai.com/v1
RERANK_API_KEY=...
RERANK_BASE_URL=https://api.cohere.com
RERANK_ENDPOINT=/v2/rerank

# DeepSeek reader.
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# OpenAI-compatible Qwen embedding and HTTP rerank services.
QWEN_VECTORIZE_API_KEY=...
QWEN_VECTORIZE_BASE_URL=...
QWEN_RERANK_API_KEY=...
QWEN_RERANK_BASE_URL=...
QWEN_RERANK_ENDPOINT=/v1/rerank
```

## Outputs

Under `methods/EverCore/evaluation/results`, each condition has a result
directory; MemoryAgentBench has one per source. Configuration snapshots,
MemoryAgentBench aggregates (`mab_accuracy.txt`), and non-abstention reports
(`non_abstention.json`) are saved under `paper-configs/<Tag>-<condition>/`.

## Lexical Scoring

LoCoMo minimal conditions report J, F1, and BLEU-1 over 1,540 non-adversarial
questions on a 0-100 scale:

- **J:** mean of three correctness judgments per question.
- **F1:** set-based overlap of lowercased whitespace tokens after replacing
  `.`, `,`, `!`, and `?` with spaces.
- **BLEU-1:** mean sentence-level unigram BLEU on lowercased, tokenized text,
  with method-1 smoothing and a brevity penalty.

Reports are saved as `lexical_metrics.json` and `lexical_report.txt`.

To score saved answers, from `methods/EverCore`:

```powershell
python -m evaluation.scripts.score_locomo_lexical `
  --run-dir evaluation/results/locomo-rag-paper-locomo-textemb3-gpt4omini-minimal `
  --with-judge --expected-questions 1540 --expected-judgments 3
```

## Offline tests

From `methods/EverCore`:

```powershell
python -m pytest tests/test_paper_profiles.py tests/test_score_locomo_lexical.py -q -p no:cacheprovider
```
