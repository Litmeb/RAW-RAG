"""
Search stage - retrieve relevant memories.
"""
import asyncio
from typing import List, Any, Optional
from logging import Logger
from tqdm import tqdm

from evaluation.src.core.data_models import QAPair, SearchResult
from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.utils.checkpoint import CheckpointManager


def _build_search_query(qa: QAPair, adapter: BaseAdapter) -> str:
    """Use the original question text for retrieval.

    MemoryAgentBench's official query templates are answer prompts, not
    retrieval queries. Passing those long instructions into vector/BM25 search
    dilutes the entity-bearing question text, especially for short subsets such
    as FactConsolidation and RULER. The answer stage still applies the official
    prompt before calling the LLM.
    """
    search_config = getattr(adapter, "config", {}).get("search", {})
    if search_config.get("personamem_raw_query", False):
        from evaluation.src.converters.personamem_v2_converter import (
            _extract_user_query_content,
        )

        raw_query = qa.metadata.get("raw_user_query")
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError("personamem_raw_query requires raw_user_query metadata")
        return _extract_user_query_content(raw_query)
    return qa.question


async def run_search_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    index: Any,
    conversations: List,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[SearchResult]:
    """
    Execute concurrent search with fine-grained checkpointing.
    
    Process by conversation groups, save checkpoint after each conversation.
    
    Args:
        adapter: System adapter
        qa_pairs: List of QA pairs
        index: Index
        conversations: Conversation list (for online API cache rebuild)
        checkpoint_manager: Checkpoint manager for resume
        logger: Logger
        
    Returns:
        List of search results
    """
    print(f"\n{'='*60}")
    print("Stage 2/4: Search")
    print(f"{'='*60}")
    
    # Load fine-grained checkpoint
    all_search_results_dict = {}
    if checkpoint_manager:
        all_search_results_dict = checkpoint_manager.load_search_progress()
    
    # Group QA pairs by conversation
    conv_to_qa = {}
    for qa in qa_pairs:
        conv_id = qa.metadata.get("conversation_id", "unknown")
        if conv_id not in conv_to_qa:
            conv_to_qa[conv_id] = []
        conv_to_qa[conv_id].append(qa)
    
    total_convs = len(conv_to_qa)
    expected_question_ids = {
        conv_id: {qa.question_id for qa in qa_list}
        for conv_id, qa_list in conv_to_qa.items()
    }
    checkpointed_question_ids = {
        conv_id: {
            row.get("question_id")
            for row in all_search_results_dict.get(conv_id, [])
            if row.get("question_id") in expected_question_ids.get(conv_id, set())
        }
        for conv_id in conv_to_qa
    }
    processed_convs = {
        conv_id
        for conv_id, question_ids in expected_question_ids.items()
        if question_ids and question_ids <= checkpointed_question_ids[conv_id]
    }
    partial_convs = {
        conv_id
        for conv_id, question_ids in checkpointed_question_ids.items()
        if question_ids and conv_id not in processed_convs
    }
    
    print(f"Total conversations: {total_convs}")
    print(f"Total questions: {len(qa_pairs)}")
    if processed_convs:
        print(f"Already processed: {len(processed_convs)} conversations (from checkpoint)")
    if partial_convs:
        print(f"Partially processed: {len(partial_convs)} conversations (from checkpoint)")
    
    # Build conversation_id to conversation mapping (for online API cache rebuild)
    conv_id_to_conv = {conv.conversation_id: conv for conv in conversations}
    
    # Search-stage concurrency can be configured separately via system config:
    #   search.num_workers (fallback to adapter.num_workers, then 20)
    search_cfg = adapter.config.get("search", {})
    num_workers = int(search_cfg.get("num_workers", getattr(adapter, "num_workers", 20)))
    num_workers = max(1, num_workers)
    semaphore = asyncio.Semaphore(num_workers)
    sample_workers = int(adapter.config.get("sample_workers", 1))
    sample_workers = max(1, sample_workers)
    sample_semaphore = asyncio.Semaphore(sample_workers)
    checkpoint_lock = asyncio.Lock()
    print(f"Search concurrency: {num_workers} workers")
    print(f"Sample concurrency: {sample_workers} conversations")
    
    # Create fine-grained progress bar (track by questions)
    total_questions = len(qa_pairs)
    processed_questions = sum(
        len(question_ids) for question_ids in checkpointed_question_ids.values()
    )
    
    pbar = tqdm(
        total=total_questions,
        initial=processed_questions,
        desc="🔍 Search Progress",
        unit="qa"
    )
    
    async def search_single_with_tracking(qa):
        async with semaphore:
            conv_id = qa.metadata.get("conversation_id", "0")
            conversation = conv_id_to_conv.get(conv_id)
            query = _build_search_query(qa, adapter)
            
            # Search with timeout and retry (similar to answer_stage.py)
            max_retries = max(1, int(search_cfg.get("max_retries", 3)))
            timeout_seconds = float(search_cfg.get("timeout_seconds", 300.0))
            result = None
            
            for attempt in range(max_retries):
                try:
                    result = await asyncio.wait_for(
                        adapter.search(
                            query,
                            conv_id,
                            index,
                            conversation=conversation,
                            question_id=qa.question_id,
                            question_date=qa.metadata.get("question_date"),
                        ),
                        timeout=timeout_seconds
                    )
                    break  # Success, exit retry loop
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        tqdm.write(f"  ⏱️  Search timeout ({timeout_seconds}s) for question in {conv_id}, retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(2)  # Short delay before retry
                    else:
                        tqdm.write(f"  ❌ Search timeout after {max_retries} attempts for question in {conv_id}: {qa.question[:60]}...")
                        if getattr(adapter, "fail_on_search_error", False):
                            raise
                        # Return empty search result on timeout
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            query=query,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={"error": "Search timeout after retries"},
                            question_id=qa.question_id,
                        )
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        tqdm.write(f"  ⚠️  Search failed for question in {conv_id}: {str(e)}, retry {attempt + 1}/{max_retries}...")
                        await asyncio.sleep(2)
                    else:
                        tqdm.write(f"  ❌ Search failed after {max_retries} attempts for question in {conv_id}: {str(e)}")
                        if getattr(adapter, "fail_on_search_error", False):
                            raise
                        # Return empty search result on error
                        from evaluation.src.core.data_models import SearchResult
                        result = SearchResult(
                            query=query,
                            conversation_id=conv_id,
                            results=[],
                            retrieval_metadata={"error": f"Search error: {str(e)}"},
                            question_id=qa.question_id,
                        )
            
            pbar.update(1)  # Update progress bar after each question
            return result
    
    # Process by conversation (use numeric sort for conversation IDs like "longmemeval_10")
    def sort_key(item):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        conv_id = item[0]
        # Try to extract numeric suffix (e.g., "longmemeval_10" -> 10)
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    sorted_conv_items = sorted(conv_to_qa.items(), key=sort_key)

    async def process_conversation(idx, conv_id, qa_list):
        if conv_id in processed_convs:
            tqdm.write(f"⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            return

        async with sample_semaphore:
            try:
                existing_by_question = {
                    row.get("question_id"): row
                    for row in all_search_results_dict.get(conv_id, [])
                    if row.get("question_id") in expected_question_ids[conv_id]
                }
                pending_qa = [
                    qa for qa in qa_list if qa.question_id not in existing_by_question
                ]
                tqdm.write(
                    f"Processing Conversation ID: {conv_id} ({idx+1}/{total_convs}) - "
                    f"{len(pending_qa)}/{len(qa_list)} questions remaining"
                )

                async def search_and_checkpoint(qa):
                    result = await search_single_with_tracking(qa)
                    return await checkpoint_result(qa, result)

                async def checkpoint_result(qa, result):
                    row = {
                        "question_id": qa.question_id,
                        "query": result.query,
                        "conversation_id": conv_id,
                        "results": result.results,
                        "retrieval_metadata": result.retrieval_metadata,
                    }
                    async with checkpoint_lock:
                        existing_by_question[qa.question_id] = row
                        all_search_results_dict[conv_id] = [
                            existing_by_question[item.question_id]
                            for item in qa_list
                            if item.question_id in existing_by_question
                        ]
                        if checkpoint_manager:
                            checkpoint_manager.save_search_progress(
                                all_search_results_dict
                            )
                    return result

                # Persist each completed question instead of waiting for the
                # entire conversation. This makes forced-stop recovery useful
                # for conversations containing many questions.
                batch_search = getattr(adapter, "search_batch", None)
                batch_size = int(search_cfg.get("embedding_batch_size", 0))
                if callable(batch_search) and batch_size > 1:
                    # The adapter batches only the embedding calls. Each
                    # query still gets an independent retrieval response and
                    # checkpoint entry.
                    for start in range(0, len(pending_qa), batch_size):
                        qa_batch = pending_qa[start : start + batch_size]
                        requests = [
                            {
                                "query": _build_search_query(qa, adapter),
                                "conversation_id": conv_id,
                                "question_id": qa.question_id,
                                "question_date": qa.metadata.get("question_date"),
                            }
                            for qa in qa_batch
                        ]
                        try:
                            batch_results = await asyncio.wait_for(
                                batch_search(requests, index),
                                timeout=float(search_cfg.get("timeout_seconds", 300.0)),
                            )
                            if len(batch_results) != len(qa_batch):
                                raise RuntimeError(
                                    "search_batch returned an unexpected number of results"
                                )
                        except Exception as exc:
                            # Preserve the existing per-question retry and
                            # timeout behavior if the optional batch path
                            # cannot complete as a unit.
                            tqdm.write(
                                f"  Batch search failed for {conv_id}: {exc}; "
                                "falling back to per-question search"
                            )
                            await asyncio.gather(
                                *(search_and_checkpoint(qa) for qa in qa_batch)
                            )
                            continue

                        for qa, result in zip(qa_batch, batch_results):
                            pbar.update(1)
                            await checkpoint_result(qa, result)
                else:
                    await asyncio.gather(
                        *(search_and_checkpoint(qa) for qa in pending_qa)
                    )
            finally:
                release_cache = getattr(adapter, "release_conversation_search_cache", None)
                if callable(release_cache):
                    maybe_coro = release_cache(conv_id, index)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

    conversation_tasks = [
        process_conversation(idx, conv_id, qa_list)
        for idx, (conv_id, qa_list) in enumerate(sorted_conv_items)
    ]
    await asyncio.gather(*conversation_tasks)
    
    # Close progress bar
    pbar.close()
    
    # Delete fine-grained checkpoint after completion
    if checkpoint_manager:
        checkpoint_manager.delete_search_checkpoint()
    
    # Convert dict format to SearchResult object list (maintain original return format)
    # Use same numeric sort as above to ensure consistent ordering
    def sort_key_conv_id(conv_id):
        """Sort by numeric part of conversation_id if possible, else alphabetically."""
        parts = conv_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (conv_id, 0)
    
    all_results = []
    for conv_id in sorted(conv_to_qa.keys(), key=sort_key_conv_id):
        if conv_id in all_search_results_dict:
            for result_dict in all_search_results_dict[conv_id]:
                all_results.append(SearchResult(
                    query=result_dict["query"],
                    conversation_id=result_dict["conversation_id"],
                    results=result_dict["results"],
                    retrieval_metadata=result_dict.get("retrieval_metadata", {}),
                    question_id=result_dict.get("question_id", ""),
                ))
    
    print(f"\n{'='*60}")
    print("🎉 All conversations processed!")
    print(f"{'='*60}")
    print(f"✅ Search completed: {len(all_results)} results\n")
    return all_results

