# RAW-RAG paper evaluation profiles

Profiles for the RAW-RAG main and appendix evaluations.
Requires PowerShell 7 and the [evaluation dependencies](../../README.md#1-install).
Configure [services](#services-and-credentials) and
[prepare the benchmark data](../../README.md#3-prepare-data) before running.
Runner commands use the repository root. The commands below are alternatives;
`All` already includes `Main`.

## Run

```powershell
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Appendix
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 `
  -Condition "locomo-textemb3-gpt4omini,locomo-textemb3-gpt4omini-minimal" `
  -Tag paper-v1
```

`Main` is the default group. `-Condition` overrides `-Group`; conditions run in
manifest order. Minimal, Flash, and non-abstention conditions require their
`source_condition` outputs under the same `Tag`. `All` includes these dependencies.
`-Python` selects the interpreter; `-Tag` names the experiment.

The runner overwrites active YAML files, so runs in one checkout must be serial.

The script copies:

1. `configs/systems/<profile>.yaml` to `evaluation/config/systems/rag.yaml`.
2. `configs/datasets/<profile>.yaml` to `evaluation/config/datasets/<dataset>.yaml`.
3. The frozen `configs/prompts.yaml` to `evaluation/config/prompts.yaml`.

All destinations above are relative to `methods/EverCore`.
Recipes are defined in the profile files; each system YAML is self-contained.

## Conditions

`textemb3` means text-embedding-3-small + Cohere Rerank-v3.5;
`qwen3` means Qwen3-Embedding-4B + Qwen3-Reranker-4B.
The common retrieval settings are 256-token windows, overlap 32,
128 candidates, 15 selected windows, local storage, no dynamic top-k and no
display deduplication. MAB disables turn expansion and keeps fact-consolidation
units atomic. Its 10,000,000-character message limit preserves the complete
main benchmark contexts before adapter chunking. MEME displays selected evidence
chronologically.

| ID | Reader | Judge / scoring | Execution |
| --- | --- | --- | --- |
| locomo-textemb3-gpt4omini | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-textemb3-gpt41mini | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| longmemeval-textemb3-gpt41mini | GPT-4.1-mini | EverMemOS-style, GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| meme-textemb3-gpt4omini | GPT-4o-mini | GPT-4o-2024-11-20 x3 | Full pipeline |
| meme-textemb3-flash | DeepSeek-V4-Flash | GPT-4o-2024-11-20 x1 | Reuse main MEME retrieval; answer/evaluate |
| mab-textemb3-gpt4omini | GPT-4o-mini | Native metrics; GPT-4o where needed | 14 explicit sources + aggregate |
| mab-textemb3-flash | DeepSeek-V4-Flash | Native metrics; GPT-4o-2024-11-20 where needed | Reuse main MAB retrieval; answer/evaluate |
| locomo-qwen3-gpt4omini | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-qwen3-gpt41mini | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| longmemeval-qwen3-gpt41mini | GPT-4.1-mini | EverMemOS-style, GPT-4o-mini-2024-07-18 x3 | Full pipeline |
| locomo-textemb3-gpt4omini-minimal | GPT-4o-mini | GPT-4o-mini-2024-07-18 x3; F1/BLEU-1 | Reuse matching LoCoMo retrieval; answer/evaluate; offline lexical scoring |
| locomo-textemb3-gpt41mini-minimal | GPT-4.1-mini | GPT-4o-mini-2024-07-18 x3; F1/BLEU-1 | Reuse matching LoCoMo retrieval; answer/evaluate; offline lexical scoring |
| longmemeval-textemb3-gpt41mini-noabs | Existing GPT-4.1-mini answers | Existing judgments, 470-question subset | Report only |
| longmemeval-qwen3-gpt41mini-noabs | Existing GPT-4.1-mini answers | Existing judgments, 470-question subset | Report only |
| personamem-v2-qwen3-gpt41mini | GPT-4.1-mini | Exact-match multiple choice | Full pipeline |

`conditions.json` maps each condition to paper table labels, expected question
counts, and reference scores.

## Services and credentials

Credentials and endpoints are read from `methods/EverCore/.env` or the process
environment. Profile model identifiers are set in YAML, not model environment
variables.

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

# Independent Flash reader; judges continue to use LLM_*.
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# OpenAI-compatible Qwen embedding and HTTP rerank services.
QWEN_VECTORIZE_API_KEY=...
QWEN_VECTORIZE_BASE_URL=...
QWEN_RERANK_API_KEY=...
QWEN_RERANK_BASE_URL=...
QWEN_RERANK_ENDPOINT=/v1/rerank
```

Qwen profiles use the 4B embedding and reranker checkpoints.
MEME judge repetitions are set by `evaluation.num_runs` in the profile YAML.
For PersonaMem-v2, `search.personamem_raw_query` selects the raw user request
for retrieval; the reader receives the personalization reminder and options.
The runner sets `PYTHONHASHSEED=0` for repeatable PersonaMem-v2 option shuffling
on new conversions. Existing converted data retains its option order.

## Outputs

Results use `evaluation/results/<dataset>-rag-<Tag>-<condition>`.
MAB adds `<split>-<sanitized-source>` before `<Tag>`.
Flash/minimal runs reuse `search_results.json` and generate new answers.
Configuration snapshots and SHA-256 hashes are saved under
`evaluation/results/paper-configs/<Tag>-<condition>/`.
The same directory holds `mab_accuracy.txt` or `non_abstention.json` where
applicable.

## Lexical Scoring

### LoCoMo lexical reports

Both minimal conditions write `lexical_metrics.json` and `lexical_report.txt`
inside their result directories. JSON includes per-question scores, category
and question-weighted overall means, input SHA-256 hashes and the NLTK version.
All scores use 0-100. Category 5 is excluded.

F1 uses sets of lowercased whitespace tokens after replacing `.`, `,`, `!`,
and `?` with spaces; repeated tokens do not increase overlap. Numeric reference
answers are converted to text. Empty overlap
scores zero. BLEU-1 uses lowercased English `nltk.word_tokenize`, unigram
weights `(1,0,0,0)`, method-1 smoothing and the sentence-level brevity penalty;
it averages sentence scores, not corpus BLEU. J averages the three saved
boolean judgments per question. The saved `answer` is scored as-is, without
extracting another final answer.

To score saved answers, from `methods/EverCore`:

```powershell
python -m evaluation.scripts.score_locomo_lexical `
  --run-dir evaluation/results/locomo-rag-paper-locomo-textemb3-gpt4omini-minimal `
  --with-judge --expected-questions 1540 --expected-judgments 3
```

`--with-judge` includes J alongside F1/BLEU-1. Paper profiles score 1,540
questions with three judgments each.

## Offline tests

From `methods/EverCore`:

```powershell
python -m pytest tests/test_paper_profiles.py tests/test_score_locomo_lexical.py -q -p no:cacheprovider
```
