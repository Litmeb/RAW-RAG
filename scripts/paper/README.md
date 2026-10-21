# RAW-RAG paper evaluation profiles

Run these commands from the package root (`supplementary_rawrag_final`).
PowerShell 7 (`pwsh`) and the evaluation requirements are required.
Before running the minimal-answer conditions, install the evaluation requirements
and download the English NLTK tokenizer data once, from `methods/EverCore`:

```powershell
python -m pip install -r requirements-eval.txt
python -m nltk.downloader punkt_tab
```

Scoring never downloads resources automatically. Missing tokenizer data stops
the scorer with a setup command rather than silently changing tokenization.
These profiles cover the main comparisons and appendix evaluation protocols,
not ablations, oracle/coverage diagnostics, Full-context, other memory systems,
budget interventions, or timing experiments.

## Run

```powershell
# Inspect all 15 conditions, without changing files or calling models.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All -List
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All -DryRun

# Seven main conditions, in dependency order.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Main

# Main conditions followed by all appendix conditions.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group All

# Appendix only: requires the relevant Main outputs under the same Tag.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 -Group Appendix

# One independent condition, a chosen Python, and a new output namespace.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 `
  -Condition locomo-textemb3-gpt4omini `
  -Python "D:/Anaconda/envs/py312pt291cu128/python.exe" -Tag paper-v1

# Multiple conditions: comma-separated string also works with pwsh -File.
pwsh -NoProfile -File scripts/paper/run_rawrag.ps1 `
  -Condition "locomo-textemb3-gpt4omini,locomo-textemb3-gpt4omini-minimal" `
  -Tag paper-v1
```

No arguments selects `Main`. `-Condition` takes precedence over `-Group`.
Selection always follows manifest order, not command-line argument order.
Dependencies are NOT run implicitly: a standalone minimal/Flash/noabs condition
requires its `source_condition` output under the same `Tag`. `-Group All`
includes all producers before their consumers.

**Actual execution overwrites the active YAML files permanently.** There is no
backup, rollback, retry orchestration, data preflight, or automatic dependency
execution. A nonzero Python exit stops the script. Do not run two profile
scripts concurrently in the same checkout, or edit shared YAML during a run.
The underlying CLI retains its own checkpoint/cache behavior.
Use a new `Tag` for an independent experiment; using the same tag can reuse
compatible artifacts through the existing CLI.

The script copies:

1. `configs/systems/<profile>.yaml` to `evaluation/config/systems/rag.yaml`.
2. `configs/datasets/<profile>.yaml` to `evaluation/config/datasets/<dataset>.yaml`.
3. The frozen `configs/prompts.yaml` to `evaluation/config/prompts.yaml`.

All destinations above are relative to `methods/EverCore`.
Edit the **profile files**, not the active copies, to change a recipe.
Each system YAML is complete; there is no YAML inheritance or runtime merging
between profile files. Existing dataset/source overrides within a system YAML
are still handled by the unchanged CLI.

## Conditions

`textemb3` means text-embedding-3-small + Cohere Rerank-v3.5;
`qwen3` means Qwen3-Embedding-4B + Qwen3-Reranker-4B.
The common retrieval settings are 256-token windows, overlap 32,
128 candidates, 15 selected windows, local storage, no dynamic top-k and no
display deduplication. MAB disables turn expansion and keeps fact-consolidation
units atomic. MEME displays selected evidence chronologically.

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
counts, and reference scores. These are annotations, not values injected into
reports or runtime data checks. Main/appendix entries referring to the same
run are not duplicated.

## Services and credentials

Keep credentials and endpoint routing in `methods/EverCore/.env` or the process
environment. The existing CLI loads `.env`. Profile model identifiers are
literal YAML values: `LLM_MODEL`, `VECTORIZE_MODEL`, `RERANK_MODEL`,
`EVALUATE_MODEL`, `MEME_EVALUATE_MODEL`, and
`MEMORYAGENTBENCH_EVALUATE_MODEL` do not select these profile models.

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

Qwen services must serve the named **4B** checkpoints. The existing RAG adapter
applies the Qwen3 reranker instruction/template based on its model name.
No service installation or availability check is performed.
If a gateway requires prefixes such as `openai/text-embedding-3-small` or
`cohere/rerank-v3.5`, edit those literal model values in the profile YAML.
Record that change; do not silently replace the model with another checkpoint.
Endpoint defaults are routing examples, not proof of historical provider
identity or current service availability.

For MEME, the script reads `evaluation.num_runs` from the selected YAML and
sets `MEME_JUDGE_NUM_RUNS` for the child processes. This neutralizes any stale
value in the invoking environment or `.env`; the YAML remains authoritative.
For PersonaMem-v2, the optional `search.personamem_raw_query` flag uses the
converter's saved raw user request only for retrieval, preserving the
personalization reminder and multiple-choice options in the reader input.
Other configurations keep the original search behavior.

## Outputs and limits

Results use `evaluation/results/<dataset>-rag-<Tag>-<condition>`.
MAB adds `<split>-<sanitized-source>` before `<Tag>`.
Flash/minimal runs copy only `search_results.json`, never original answers,
judgments or checkpoints.
Configuration snapshots and SHA-256 hashes are saved under
`evaluation/results/paper-configs/<Tag>-<condition>/`.
The same directory holds `mab_accuracy.txt` or `non_abstention.json` where
applicable. Credentials and resolved `.env` contents are not copied.
The CLI's own public run identity records resolved execution settings.

These profiles reconstruct the stated protocols on the supplied implementation;
they do not guarantee equality with historical scores:

- MEME judge models/counts were checked against the local result metadata
  underlying 39.30 and 43.19. MAB judge identities were checked against the
  original GPT-4o-mini and Flash result metadata. They intentionally differ
  from the generic dataset defaults and earlier audit command examples.
- The two minimal conditions automatically run offline J/F1/BLEU-1 reporting
  after answer/evaluate; see the scoring command below.
- PersonaMem-v2 option shuffling uses Python's string hash. The runner fixes
  `PYTHONHASHSEED=0` for repeatable **new conversions**, recorded in the manifest
  and snapshots. The historical hash seed is not established; existing
  converted caches retain their option order. For identical option letters
  across machines, distribute the same converted data. The script does not
  remove or rewrite dataset caches.
- LongMemEval uses the corrected, uncapped source-message loader. Historical
  model aliases, provider changes, conversion inputs, and implementation
  corrections can change scores.
- Prompt YAML is frozen here; code-defined MEME/MAB judge and query templates
  still require a fixed code revision. This mechanism does not freeze services,
  raw datasets, or the whole Python environment.

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
extracting another final answer or calling any models.

To score existing outputs without regenerating answers, run from
`methods/EverCore` (replace `paper` if a different Tag was used):

```powershell
foreach ($reader in @("gpt4omini", "gpt41mini")) {
    python -m evaluation.scripts.score_locomo_lexical `
      --run-dir "evaluation/results/locomo-rag-paper-locomo-textemb3-$reader-minimal" `
      --with-judge --expected-questions 1540 --expected-judgments 3
    if ($LASTEXITCODE -ne 0) { throw "LoCoMo scoring failed: $reader" }
}
```

The same command accepts a standard-answer result directory. Without
`--with-judge`, only F1/BLEU-1 are computed and `eval_results.json` is not needed.
Without `--expected-questions`, subsets are allowed and their counts are reported.
Paper-runner scoring requires 1,540 questions and three complete judgments each.
Duplicate IDs, invalid categories, saved generation-error responses, or
judgments for different answers/references are rejected. Existing answer,
evaluation and `report.txt` files are not changed.

## Offline tests

From `methods/EverCore`, with evaluation dependencies and pytest installed:

```powershell
python -m pytest tests/test_paper_profiles.py tests/test_score_locomo_lexical.py -q -p no:cacheprovider
```

The tests execute the PowerShell runner against a temporary package with a
fake evaluation CLI. They do not replace the real active YAML, read benchmark
data, or call model services.
