"""
Answer stage - generate answers.
"""
import asyncio
import time
from typing import List, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult, AnswerResult
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.datasets.memoryagentbench import format_memoryagentbench_query
from evaluation.src.utils.checkpoint import CheckpointManager


def truncate_context_to_token_budget(
    context: str,
    max_context_tokens: int,
    model: str = "gpt-4o-mini",
):
    """Bound answer evidence while preserving highest-ranked context first."""
    if not context or max_context_tokens <= 0:
        return context, None

    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Most current OpenAI-compatible chat models use this vocabulary. The
        # explicit budget still leaves ample room for prompt and completion.
        encoding = tiktoken.get_encoding("o200k_base")

    # Tokenizing multi-megabyte Memora histories in full is expensive even
    # though only a prefix can be sent. Four characters per token closely
    # matches the benchmark's English data; the exact token pass below still
    # guarantees the hard limit when the text tokenizes more densely.
    candidate_chars = min(len(context), max_context_tokens * 4)
    while True:
        candidate = context[:candidate_chars]
        token_ids = encoding.encode(candidate, disallowed_special=())
        if len(token_ids) >= max_context_tokens or candidate_chars == len(context):
            break
        # Very repetitive text may encode below the usual four-char estimate.
        # Expand once or twice using its observed ratio without scanning the
        # full source history.
        estimated_chars = int(
            candidate_chars * max_context_tokens / max(1, len(token_ids)) * 1.05
        )
        candidate_chars = min(
            len(context), max(candidate_chars + 1, estimated_chars)
        )

    prefix_only = candidate_chars < len(context)
    original_tokens = None if prefix_only else len(token_ids)
    if len(token_ids) <= max_context_tokens:
        return candidate, (original_tokens, len(token_ids))

    truncated = encoding.decode(token_ids[:max_context_tokens]).rstrip()
    return truncated, (original_tokens, max_context_tokens)


def _has_temporal_cutoff(qa: QAPair) -> bool:
    return bool(qa.metadata.get("question_date"))


def _memoryagentbench_agent_kind(adapter: BaseAdapter) -> str:
    adapter_name = str(getattr(adapter, "config", {}).get("adapter", ""))
    rag_style_adapters = {"rag", "rag_memory_ablation", "full_context"}
    return "rag_agent" if adapter_name in rag_style_adapters else "agentic_memory_agent"


def _build_answer_query(qa: QAPair, adapter: BaseAdapter) -> str:
    """Expose the benchmark's query-time clock to the answering LLM."""
    source = qa.metadata.get("source")
    prompt_key = qa.metadata.get("prompt_key", "")
    if source and str(prompt_key).startswith("answer_prompt_mab_"):
        return format_memoryagentbench_query(
            str(source),
            qa.question,
            agent_kind=_memoryagentbench_agent_kind(adapter),
        )

    if not _has_temporal_cutoff(qa):
        return qa.question

    question_date = qa.metadata["question_date"]
    return (
        f"Question date: {question_date}\n"
        "Treat this as the current time when resolving relative phrases such as "
        "\"recently\", \"last month\", \"previous\", \"current\", or \"now\". "
        "Only use memories dated at or before this question date.\n"
        f"Question: {qa.question}"
    )


def build_context(search_result: SearchResult, qa: Optional[QAPair] = None) -> str:
    """
    Build context from search results.
    
    Prefer pre-formatted context (dual-speaker scenarios), else use simple numbering
    (single-speaker scenarios). MemoryAgentBench is special-cased to mirror the
    official benchmark's `- memory` formatting.
    
    Args:
        search_result: Search result
        
    Returns:
        Context string
    """
    prompt_key = (qa.metadata.get("prompt_key", "") if qa else "")

    # Adapter-provided contexts may contain task-specific raw evidence. For
    # MemoryAgentBench ICL, Memos injects exact label mappings here.
    formatted_context = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted_context:
        return formatted_context

    if prompt_key.startswith("answer_prompt_mab_"):
        top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
        memory_lines = []
        for result in search_result.results[:top_k]:
            content = (result.get("content", "") or "").strip()
            if content:
                memory_lines.append(f"- {content}")
        return "\n".join(memory_lines)

    # Single speaker scenario: simple formatting
    context_parts = []
    
    # Get top_k from retrieval_metadata, default to len(results) if not specified
    top_k = search_result.retrieval_metadata.get("top_k", len(search_result.results))
    
    # Add memory content (use top_k instead of hardcoded 10)
    for idx, result in enumerate(search_result.results[:top_k], 1):
        content = result.get("content", "")
        context_parts.append(f"{idx}. {content}")
    
    context = "\n\n".join(context_parts)
    
    # For systems supporting preferences (e.g., Memos), add formatted pref_string
    preferences = search_result.retrieval_metadata.get("preferences", {})
    pref_string = preferences.get("pref_string", "")
    
    if pref_string:
        context += "\n\n" + pref_string
    
    return context


async def run_answer_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    search_results: List[SearchResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[AnswerResult]:
    """
    Generate answers with fine-grained checkpointing.
    
    Save checkpoint every SAVE_INTERVAL questions.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        search_results: List of search results
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of answer results
    """
    print(f"\n{'='*60}")
    print(f"Stage 3/4: Answer")
    print(f"{'='*60}")
    
    SAVE_INTERVAL = 400  # Save every 400 tasks
    answer_cfg = adapter.config.get("answer", {})
    max_concurrent = int(answer_cfg.get("num_workers", 50))
    max_concurrent = max(1, max_concurrent)
    max_context_tokens = int(answer_cfg.get("max_context_tokens", 0) or 0)
    answer_model = str(adapter.config.get("llm", {}).get("model", "gpt-4o-mini"))
    
    # Load fine-grained checkpoint
    all_answer_results = {}
    if checkpoint_manager:
        loaded_results = checkpoint_manager.load_answer_progress()
        # Convert to {question_id: AnswerResult} format
        for result in loaded_results.values():
            all_answer_results[result["question_id"]] = result
    
    total_qa_count = len(qa_pairs)
    processed_count = len(all_answer_results)
    
    print(f"Total questions: {total_qa_count}")
    if processed_count > 0:
        print(f"Already processed: {processed_count} questions (from checkpoint)")
        print(f"Remaining: {total_qa_count - processed_count} questions")
    
    # Prepare pending tasks
    # qa_pairs and search_results should have matching order (both use numeric sort by conversation_id)
    pending_tasks = []
    for qa, sr in zip(qa_pairs, search_results):
        if qa.question_id not in all_answer_results:
            pending_tasks.append((qa, sr))
    
    if not pending_tasks:
        print(f"✅ All questions already processed!")
        # Convert to AnswerResult object list (original order)
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    formatted_context=result_dict.get("formatted_context", ""),  # Load formatted_context
                    # search_results not loaded to save space
                    metadata=result_dict.get("metadata", {}),
                ))
        return results
    
    semaphore = asyncio.Semaphore(max_concurrent)
    print(f"Answer concurrency: {max_concurrent} workers")
    if max_context_tokens:
        print(f"Answer context budget: {max_context_tokens} tokens")
    completed = processed_count
    failed = 0
    start_time = time.perf_counter()
    
    # Use tqdm progress bar
    pbar = tqdm(
        total=total_qa_count,
        initial=processed_count,
        desc="💬 Answer Progress",
        unit="qa"
    )
    
    async def answer_single_with_tracking(qa, search_result):
        nonlocal completed, failed
        
        async with semaphore:
            context = ""
            budget_metadata = {}
            try:
                # Build context
                context = build_context(search_result, qa)
                context, context_token_counts = truncate_context_to_token_budget(
                    context,
                    max_context_tokens,
                    answer_model,
                )
                if (
                    context_token_counts
                    and (
                        context_token_counts[0] is None
                        or context_token_counts[0] > context_token_counts[1]
                    )
                ):
                    original_count = (
                        f">{context_token_counts[1]}"
                        if context_token_counts[0] is None
                        else str(context_token_counts[0])
                    )
                    tqdm.write(
                        f"  Context truncated for {qa.question_id}: "
                        f"{original_count} -> "
                        f"{context_token_counts[1]} tokens"
                    )
                
                # Detect multiple-choice and enhance question if needed
                query = _build_answer_query(qa, adapter)
                if "all_options" in qa.metadata:
                    options = qa.metadata["all_options"]
                    options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
                    mcq_instruction = (
                        "Please choose the best answer from the following options:\n\n"
                        + options_text
                        + "\n\nThink step by step about which answer best fits the user's query and conversation context. "
                        "Provide your reasoning first, then give your final answer as 'Final Answer: [Letter]'"
                    )
                    
                    # PersonaMem-v2 official MCQ inference appends the MCQ
                    # instruction after the user query.
                    query = f"{qa.question}\n\n{mcq_instruction}"
                
                if hasattr(adapter, "prepare_answer_context"):
                    context, budget_metadata = adapter.prepare_answer_context(
                        query, context, qa.metadata.get("prompt_key")
                    )

                # Call adapter's answer method with timeout and retry
                max_retries = 3
                timeout_seconds = 300.0
                answer = None
                
                for attempt in range(max_retries):
                    try:
                        answer = await asyncio.wait_for(
                            adapter.answer(
                                query=query,
                                context=context,
                                conversation_id=search_result.conversation_id,
                                question_id=qa.question_id,
                                prompt_key=qa.metadata.get("prompt_key"),
                            ),
                            timeout=timeout_seconds
                        )
                        answer = answer.strip()
                        break  # Success, exit retry loop
                        
                    except asyncio.TimeoutError:
                        if attempt < max_retries - 1:
                            tqdm.write(f"  ⏱️  Timeout ({timeout_seconds}s) for {qa.question_id}, retry {attempt + 1}/{max_retries}...")
                            await asyncio.sleep(2)  # Short delay before retry
                        else:
                            tqdm.write(f"  ❌ Timeout after {max_retries} attempts for {qa.question_id}: {qa.question[:50]}...")
                            answer = "Error: Answer generation timeout after retries"
                            failed += 1
            
            except Exception as e:
                tqdm.write(f"  ⚠️ Answer generation failed for {qa.question_id}: {e}")
                answer = "Error: Failed to generate answer"
                failed += 1
            
            result = AnswerResult(
                question_id=qa.question_id,
                question=qa.question,
                answer=answer,
                golden_answer=qa.answer,
                category=qa.category,
                conversation_id=search_result.conversation_id,
                formatted_context=context,  # Save actual context used
                metadata={**qa.metadata, **({"context_budget": budget_metadata} if budget_metadata else {})},
            )
            
            # Save result
            all_answer_results[qa.question_id] = {
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "golden_answer": result.golden_answer,
                "category": result.category,
                "conversation_id": result.conversation_id,
                "formatted_context": result.formatted_context,  # Save formatted_context
                "metadata": result.metadata,  # Save metadata (contains all_options)
            }
            
            completed += 1
            pbar.update(1)  # Update progress bar
            
            # Save checkpoint periodically
            if checkpoint_manager and (completed % SAVE_INTERVAL == 0 or completed == total_qa_count):
                elapsed = time.perf_counter() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_qa_count - completed) / speed if speed > 0 else 0
                
                tqdm.write(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                          f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
                
                checkpoint_manager.save_answer_progress(all_answer_results, completed, total_qa_count)
            
            return result
    
    # Create all pending tasks
    tasks = [
        answer_single_with_tracking(qa, sr)
        for qa, sr in pending_tasks
    ]
    
    # Execute concurrently
    await asyncio.gather(*tasks)
    
    # Close progress bar
    pbar.close()
    
    # Statistics
    elapsed_time = max(time.perf_counter() - start_time, 1e-9)
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ All responses generated!")
    print(f"   - Total questions: {total_qa_count}")
    print(f"   - Successful: {completed - failed}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
    print(f"{'='*60}\n")
    
    # Delete fine-grained checkpoints after completion
    if checkpoint_manager:
        checkpoint_manager.delete_answer_checkpoints()
    
    # Convert to AnswerResult object list (original order)
    results = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            result_dict = all_answer_results[qa.question_id]
            results.append(AnswerResult(
                question_id=result_dict["question_id"],
                question=result_dict["question"],
                answer=result_dict["answer"],
                golden_answer=result_dict["golden_answer"],
                category=result_dict.get("category"),
                conversation_id=result_dict.get("conversation_id", ""),
                formatted_context=result_dict.get("formatted_context", ""),
                search_results=result_dict.get("search_results", []),
                metadata=result_dict.get("metadata", {}),  # Restore metadata
            ))
    
    return results

