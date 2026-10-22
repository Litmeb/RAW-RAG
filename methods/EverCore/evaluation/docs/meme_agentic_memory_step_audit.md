# MEME agentic-memory step-trace protocol

## Research question

The experiment measures error accumulation across agentic-memory pipelines. Each observable step is judged independently for every question, allowing multiple failures, propagation, and recovery.

Each step has one of six states:

- `pass`: the step output is semantically sufficient for the official MEME question at its cutoff.
- `fail`: the step output is not semantically sufficient.
- `not_observable`: the run did not persist the step input/output needed to decide.
- `replayable`: the original telemetry is absent, but surviving inputs, indexes, and code permit a separately labeled audit-time replay.
- `not_applicable`: the system did not execute that branch for this question.
- `judge_error`: the deterministic check or LLM judge did not return a valid decision.

For adjacent observable steps, the transition is `preserved` (pass→pass), `new_error` (pass→fail), `propagated` (fail→fail), or `recovered` (fail→pass). The main cumulative measures are the number of failed steps per question, the fraction of questions that have failed at least once by each step, and final accuracy conditioned on the number of failed steps.

## Common benchmark reference

The reference state uses MEME's official evidence sessions, phase cutoff, entity values, target states, dependency edges, and task metadata. The judge receives complete relevant sessions and evaluates before/after semantics at the official cutoff.

## Mem0 Local (`meme-mem0_local-normal`)

### Write path

1. The adapter converts the MEME conversation to user messages, removes empty messages, truncates an individual message above 8,000 characters, batches five messages, assigns the first message timestamp to the batch, and calls local `Memory.add`.
2. Mem0 loads the last ten SQLite messages as short-term context.
3. It embeds the incoming batch and retrieves ten existing memories as extraction context.
4. One LLM call performs additive fact extraction and may link a fact to an existing memory ID.
5. Extracted facts are embedded in a batch.
6. MD5 exact-text deduplication removes duplicates against retrieved memories and the current batch.
7. New facts are persisted to Qdrant and write history is persisted to SQLite.
8. spaCy extracts entities; entity embeddings and memory–entity links are written to the entity store.

### Query/read path

9. The query is lemmatized and entities are extracted.
10. Semantic vector search and keyword search independently generate candidates.
11. BM25 normalization, entity boosts, score fusion, thresholding, and top-k selection produce up to 100 adapter candidates.
12. The adapter filters memories newer than the MEME question timestamp and keeps the first 20.
13. The adapter serializes those results into the reader context.
14. The answer LLM generates the final answer.

Step observability:

1. `write_existing_candidates_original` remains `not_observable`; the function-local top-10 list and exact prompt were never saved. `write_existing_candidates_replay` is instead `replayable` from MEME batches, batch metadata, stored vectors, and the implementation, but is not original telemetry.
2. `raw_llm_extraction_response` and `dedup_rejected_candidates` remain `not_observable`: raw JSON, linked IDs, parse/embedding failures, and debug skip records were not persisted.
3. `post_dedup_history_lineage` deterministically checks that task-relevant Qdrant memories have SQLite `ADD` records. These records are created after extraction, embedding, and hash filtering.
4. `task_relevant_residual_exact_duplicates` checks same-user MD5 groups in the cutoff-visible store. The implementation only compares against the write-time top-10 and current batch, so this is not global deduplication.
5. `entity_link_integrity` reads the final entity collection and validates links; `entity_link_task_support` separately judges whether saved entity labels adequately represent the query entities used by entity boosting. If deterministic query entity extraction returns no entities, the boost branch did not execute and support is `not_applicable`, not a failure.
6. `retrieval_components_original` remains `not_observable` because final search artifacts omit query entities, branch candidates, and `score_details`. `retrieval_components_replay` is `replayable` through the surviving dense/sparse indexes, entity graph, saved queries, and SDK `explain` path.
7. `temporal_filter_integrity` deterministically maps every native result timestamp back to the MEME cutoff.

History records without corresponding Qdrant objects lack conversation/user metadata; these inconsistencies are reported at run level.

## MemOS Local fine mode (`meme-memos_local-fine`)

### Write path

1. The adapter sends batches of ten messages synchronously to `/product/add` in `fine` mode.
2. The running Docker server uses `MultiModalStruct`, not `SimpleStruct`. It performs a fast per-message parse and aggregates the conversation into 1,600-token windows with 200-token overlap.
3. Each aggregate window fans out into four fine-extraction branches (string, tool, skill, and preference), while source-message transfer is performed serially. The persisted `metadata.sources` and `background` identify the shared input window for the resulting outputs.
4. Extracted items receive key, value, tags, type, background, source-message provenance, timestamps, and embeddings.
5. Text memories are written to Neo4j and their vectors are synchronized to Qdrant.
6. The merge helper returns extraction output unchanged. All 13,013 inspected fine objects were active, with no history, evolution links, or graph relationships. Merge/archive is `not_applicable`.

### Query/read path

7. The adapter calls `/product/search` in `fast` mode and requests up to 100 candidates.
8. The fast handler expands the requested top-k by a factor of three and runs the configured retrieval paths. Each path applies its internal cosine reranking before returning candidates.
9. The handler applies MMR/deduplication to the combined text/preference candidates. The later `rerank_knowledge_mem` formatter is a no-op for conversation memories in this run.
10. The adapter applies its source-time filter and keeps the first 20.
11. The adapter serializes the results into reader context.
12. The answer LLM generates the final answer.

The run uses the `fine__meme_*_User` namespace. Step checks use its Docker artifacts:

1. `multimodal_window_provenance`: deterministic official-message identity against persisted `metadata.sources`, including overlap duplication.
2. `multimodal_extraction_fidelity`: LLM judgment of each complete source-window-to-output group.
3. `graph_vector_persistence`: deterministic equality of evidence-linked Neo4j memory IDs and Qdrant point IDs.
4. `state_evolution_mechanism`: observed merge/archive/history telemetry, which is `not_applicable` for this run when none exists.
5. `state_invalidation_support`: task-aware LLM judgment for Del/Cas/Abs over the active store, independent of extraction fidelity.
6. `temporal_filter_integrity`: deterministic check that every retrieved item's latest `metadata.sources[].chat_time` session is at or before the MEME cutoff.

Raw extraction responses and intermediate hybrid, cosine-reranked, and post-MMR lists were not persisted and are `not_observable`. New searches are classified as counterfactual replays.

## EverMemOS (`meme-evermemos-normal-3`)

### Write path

1. Messages are converted to timestamped `RawData` records.
2. `ConvMemCellExtractor` processes messages sequentially. An LLM decides memory-cell boundaries using the current history and new message.
3. For every completed memory cell, an LLM extracts an episode, subject, and summary.
4. A separate LLM extracts atomic facts from each memory cell.
5. Episodes and atomic facts are persisted. Episode fields and atomic facts are embedded; BM25-searchable text is built for both representations.
6. Memory clustering also runs, but profiles and foresight are disabled. Cluster output is not consumed by the retrieval/reader path in this run, so clustering errors are side-branch errors rather than propagated answer-path errors.

Saved memory cells support the following per-question checks:

1. `memcell_message_integrity`: every official evidence message occurs exactly once in saved `original_data` (deterministic content/role/timestamp comparison).
2. `memcell_cutoff_visibility`: evidence remains in a MemCell whose last source session is visible at the question cutoff; a Cell spanning into the future is counted as a visibility failure.
3. `memcell_partition_quality`: an LLM judge evaluates evidence fragmentation, harmful over/under-segmentation, and unrelated contamination from the complete evidence-linked Cells.
4. `episode_fidelity` and `episode_task_support`: transformation fidelity is separated from whether the resulting episodes jointly suffice for the question.
5. `atomic_fact_fidelity` and `atomic_fact_task_support`: faithful individual facts can still fail task support after losing time, update, or dependency relations.
6. `episode_index_integrity` and `atomic_fact_index_integrity`: each relevant `event_id` and its exact representation are compared against both the vector and BM25 documents.

Only the raw per-call boundary trace remains unobservable: the run did not persist each prompt, LLM response, `should_wait`, parse retry, rejected boundary, or whether a saved split was triggered by the LLM, token/message limit, or final flush.

### Query/read path

7. Documents newer than the MEME question time are excluded.
8. Round-one hybrid retrieval combines embedding search and BM25 with reciprocal-rank fusion and keeps 20.
9. A reranker reduces round one to ten results.
10. An LLM judges whether those results are sufficient.
11. If insufficient, an LLM generates three refined queries.
12. Each refined query retrieves 50 hybrid candidates; multi-query RRF retains 40.
13. Round-one and round-two candidates are merged and deduplicated by event ID, then reranked to 20.
14. The adapter exposes only the first ten episode results to the reader. Atomic facts affect ranking but are not directly included in the reader context.
15. The answer LLM generates the final answer.

The run persists the final retrieval and agent metadata (`is_sufficient`, reasoning, missing information, refined queries, query strategy, and counts), but not every intermediate document list. Refined-query quality can be judged from the saved queries and missing-information claim. Sufficiency-decision correctness cannot be judged reliably because the exact round-one reranked documents shown to that decision were not persisted. Exact round-one candidate loss and round-two recovery require an instrumented retrieval replay.

## Counting rules

1. Count one `fail` for every independently failing observable step; do not stop after the first failure.
2. Do not put `not_observable`, `replayable`, `not_applicable`, or `judge_error` in the denominator of a step error rate.
3. Report deterministic exact-signal coverage alongside semantic judgment, but do not substitute exact string matching for relation, deletion, cascade, or abstraction semantics.
4. Report system-specific steps separately. Only compare genuinely homologous steps (`persisted semantic representation`, `native retrieval`, `reader context`, and `answer`) across systems.
5. Report audit-time replays separately from original-run telemetry.
6. Preserve judge model, prompt version, temperature, response, and parse/error metadata for reproducibility.
7. Treat EverMemOS episode and atomic facts as parallel branches. Transition counts follow separate episode and atomic paths and never infer a causal transition from one representation branch to the other.
