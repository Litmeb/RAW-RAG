"""
EverCore Adapter - connects evaluation framework with EverCore implementation.
"""

import asyncio
import json
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from evaluation.src.adapters.base import BaseAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from common_utils.datetime_utils import to_iso_format

# Import EverCore implementation
from evaluation.src.adapters.evermemos import (
    stage1_memcells_extraction,
    stage2_index_building,
    stage3_memory_retrivel,
    stage4_response,
)

# Import Memory Layer components
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.atomic_fact_extractor import AtomicFactExtractor


@register_adapter("evermemos")
class EverCoreAdapter(BaseAdapter):
    """
    EverCore adapter.

    Responsibilities:
    1. Receive calls from evaluation framework
    2. Convert data formats (evaluation framework ↔ EverCore)
    3. Call stage*.py implementations
    4. Return results in evaluation framework format

    Implementation details:
    - MemCell extraction (stage1)
    - Index building (stage2)
    - Retrieval logic (stage3)
    - Answer generation (stage4)
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")

        # Initialize LLM Provider (shared across all stages)
        # Read from YAML llm configuration
        llm_config = config.get("llm", {})

        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 16384),
        )

        # Initialize Atomic Fact Extractor
        self.atomic_fact_extractor = AtomicFactExtractor(llm_provider=self.llm_provider)
        self._temporal_search_cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

        # Ensure NLTK data is available
        stage2_index_building.ensure_nltk_data()

        print(f"✅ EverCore Adapter initialized")
        print(f"   LLM Model: {llm_config.get('model')}")
        print(f"   Output Dir: {self.output_dir}")

    @staticmethod
    def _extract_conv_index(conversation_id: str) -> str:
        """
        Extract numeric index part from conversation_id.

        Examples:
        - "locomo_0" -> "0"
        - "personamem_42" -> "42"
        - "123" -> "123"
        - "test_abc_5" -> "5"

        Strategy: Take the part after the last underscore, or return original if no underscore
        """
        if "_" in conversation_id:
            return conversation_id.split("_")[-1]
        return conversation_id

    def _check_missing_indexes(
        self, index_dir: Path, num_conv: int, index_type: str = "bm25"
    ) -> List[int]:
        """
        Check for missing index files.

        Args:
            index_dir: Index directory
            num_conv: Total number of conversations
            index_type: Index type ("bm25" or "embedding")

        Returns:
            List of conversation indices with missing indexes
        """
        missing_indexes = []

        for i in range(num_conv):
            if index_type == "bm25":
                index_file = index_dir / f"bm25_index_conv_{i}.pkl"
            else:  # embedding
                index_file = index_dir / f"embedding_index_conv_{i}.pkl"

            if not index_file.exists():
                missing_indexes.append(i)

        return missing_indexes

    @staticmethod
    def _parse_question_date(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return EverCoreAdapter._normalize_datetime(value)

        text = str(value).strip()
        for candidate in (text, text.upper(), text.title()):
            for fmt in (
                "%I:%M %p on %d %B, %Y",
                "%Y/%m/%d (%a) %H:%M",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    return EverCoreAdapter._normalize_datetime(
                        datetime.strptime(candidate, fmt)
                    )
                except ValueError:
                    continue
        try:
            return EverCoreAdapter._normalize_datetime(datetime.fromisoformat(text))
        except ValueError:
            return None

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def _filter_docs_for_question_date(
        self,
        docs: List[Dict[str, Any]],
        emb_index: Optional[List[Dict[str, Any]]],
        question_date: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        metadata: Dict[str, Any] = {
            "question_date": question_date,
            "temporal_filter_enabled": False,
        }
        if not question_date:
            return docs, emb_index, metadata

        question_dt = self._parse_question_date(question_date)
        if question_dt is None:
            metadata["temporal_filter_error"] = "failed to parse question_date"
            return docs, emb_index, metadata

        metadata["temporal_filter_enabled"] = True
        doc_by_event_id = {
            str(doc["event_id"]): doc for doc in docs if doc.get("event_id") is not None
        }
        visible_event_ids = {
            event_id
            for event_id, doc in doc_by_event_id.items()
            if self._doc_is_visible_at_question_date(doc, question_dt)
        }

        filtered_docs = [
            doc
            for doc in docs
            if (
                doc.get("event_id") is None
                or str(doc["event_id"]) in visible_event_ids
            )
        ]

        filtered_emb_index = emb_index
        if emb_index is not None:
            filtered_emb_index = []
            for item in emb_index:
                doc = item.get("doc", {})
                event_id = doc.get("event_id")
                if event_id is None or str(event_id) in visible_event_ids:
                    filtered_emb_index.append(item)

        metadata.update(
            {
                "temporal_filter_granularity": "memcell_timestamp",
                "temporal_filter_date_source": "doc.timestamp",
                "original_doc_count": len(docs),
                "visible_doc_count": len(filtered_docs),
                "future_filtered_doc_count": len(docs) - len(filtered_docs),
            }
        )
        if emb_index is not None and filtered_emb_index is not None:
            metadata["original_embedding_doc_count"] = len(emb_index)
            metadata["visible_embedding_doc_count"] = len(filtered_emb_index)
            metadata["future_filtered_embedding_doc_count"] = (
                len(emb_index) - len(filtered_emb_index)
            )

        return filtered_docs, filtered_emb_index, metadata

    def _build_filtered_bm25_index(
        self, docs: List[Dict[str, Any]]
    ) -> Optional[BM25Okapi]:
        if not docs:
            return None

        stemmer = stage2_index_building.PorterStemmer()
        stop_words = set(stage2_index_building.stopwords.words("english"))
        corpus = [
            stage2_index_building.tokenize(
                stage2_index_building.build_searchable_text(doc),
                stemmer,
                stop_words,
            )
            for doc in docs
        ]
        return BM25Okapi(corpus)

    def _doc_is_visible_at_question_date(
        self, doc: Dict[str, Any], question_dt: datetime
    ) -> bool:
        for candidate in self._extract_doc_datetimes(doc):
            if candidate is not None:
                return candidate <= question_dt
        return True

    def _extract_doc_datetimes(self, doc: Dict[str, Any]) -> List[Optional[datetime]]:
        candidates: List[Optional[str]] = []

        for key in ("timestamp", "updated_at", "created_at"):
            if doc.get(key):
                candidates.append(doc.get(key))

        atomic_fact = doc.get("atomic_fact") or {}
        if isinstance(atomic_fact, dict):
            for key in ("timestamp", "time"):
                if atomic_fact.get(key):
                    candidates.append(atomic_fact.get(key))

        return [self._parse_question_date(value) for value in candidates]

    async def add(
        self,
        conversations: List[Conversation],
        output_dir: Path = None,
        checkpoint_manager=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Add stage: Extract MemCells and build indexes.

        Call flow:
        1. Stage 1: Extract MemCells (stage1_memcells_extraction.py) - concurrent processing
        2. Stage 2: Build BM25 and Embedding indexes (stage2_index_building.py)

        Returns: Index metadata (Plan A: lazy loading)
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        memcells_dir = output_dir / "memcells"
        memcells_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_dir = output_dir / "bm25_index"
        emb_index_dir = output_dir / "vectors"
        bm25_index_dir.mkdir(parents=True, exist_ok=True)
        emb_index_dir.mkdir(parents=True, exist_ok=True)

        console = Console()

        # ========== Stage 1: MemCell Extraction (concurrent processing) ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 1: MemCell Extraction", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Convert data format: evaluation framework → EverCore
        raw_data_dict = {}
        for conv in conversations:
            conv_id = conv.conversation_id
            raw_data = []

            for idx, msg in enumerate(conv.messages):
                # Handle timestamp: if None, use index-based pseudo timestamp
                if msg.timestamp is not None:
                    timestamp_str = to_iso_format(msg.timestamp)
                else:
                    # Generate pseudo timestamp using message index (maintain relative order)
                    # Base time: 2023-01-01 00:00:00, 30 seconds interval per message
                    from datetime import datetime, timedelta

                    base_time = datetime(2023, 1, 1, 0, 0, 0)
                    pseudo_time = base_time + timedelta(seconds=idx * 30)
                    timestamp_str = to_iso_format(pseudo_time)

                message_dict = {
                    "sender_id": msg.sender_id,
                    "user_name": msg.sender_name or msg.sender_id,
                    "sender_name": msg.sender_name or msg.sender_id,
                    "content": msg.content,
                    "timestamp": timestamp_str,
                }

                # Add optional fields
                for optional_field in ["img_url", "blip_caption", "query"]:
                    if (
                        optional_field in msg.metadata
                        and msg.metadata[optional_field] is not None
                    ):
                        message_dict[optional_field] = msg.metadata[optional_field]

                raw_data.append(message_dict)

            raw_data_dict[conv_id] = raw_data

        # Check completed conversations (checkpoint resume)
        # Use extracted index to check files (stage1 saves using extracted index)
        completed_convs = set()
        if checkpoint_manager:
            all_conv_indices = [
                self._extract_conv_index(conv.conversation_id) for conv in conversations
            ]
            completed_indices = checkpoint_manager.load_add_progress(
                memcells_dir, all_conv_indices
            )
            # Map completed indices back to original conversation_id
            for conv in conversations:
                if self._extract_conv_index(conv.conversation_id) in completed_indices:
                    completed_convs.add(conv.conversation_id)

        # Filter conversations to process
        pending_conversations = [
            conv
            for conv in conversations
            if conv.conversation_id not in completed_convs
        ]

        console.print(
            f"\n📊 Total conversations: {len(conversations)}", style="bold cyan"
        )
        console.print(f"✅ Completed: {len(completed_convs)}", style="bold green")
        console.print(f"⏳ Pending: {len(pending_conversations)}", style="bold yellow")

        if len(pending_conversations) == 0:
            console.print(
                f"\n🎉 All conversations completed, skipping MemCell extraction!",
                style="bold green",
            )
        else:
            total_messages = sum(
                len(raw_data_dict[c.conversation_id]) for c in pending_conversations
            )
            console.print(f"📝 Pending messages: {total_messages}", style="bold blue")
            console.print(f"🚀 Starting concurrent processing...\n", style="bold green")

            # Use Rich progress bar for concurrent processing
            start_time = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("•"),
                TextColumn("[bold blue]{task.fields[status]}"),
                console=console,
                transient=False,
                refresh_per_second=1,
            ) as progress:
                # Create main progress task
                main_task = progress.add_task(
                    "[bold cyan]🎯 Overall Progress",
                    total=len(conversations),
                    completed=len(completed_convs),
                    status="Processing",
                )

                # Create progress bars for completed conversations (show as complete)
                conversation_tasks = {}
                for conv_id in completed_convs:
                    conv_index = self._extract_conv_index(conv_id)
                    conv_task_id = progress.add_task(
                        f"[green]Conv-{conv_index}",
                        total=len(raw_data_dict.get(conv_id, [])),
                        completed=len(raw_data_dict.get(conv_id, [])),
                        status="✅ (Skipped)",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                # Create progress bars and tasks for pending conversations
                processing_tasks = []
                for conv in pending_conversations:
                    conv_id = conv.conversation_id
                    conv_index = self._extract_conv_index(
                        conv_id
                    )  # Extract numeric index
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{conv_index}",
                        total=len(raw_data_dict[conv_id]),
                        completed=0,
                        status="Waiting",
                    )
                    conversation_tasks[conv_id] = conv_task_id

                    processing_tasks.append((conv_id, conv_index, conv_task_id))

                # Define completion update function
                exp_config = self._convert_config_to_experiment_config()
                conversation_max_retries = max(
                    1, int(getattr(exp_config, "max_retries", 3))
                )

                async def run_with_completion(conv_id, conv_index, conv_task_id):
                    last_exc = None
                    for attempt in range(conversation_max_retries):
                        try:
                            progress.update(
                                conv_task_id,
                                status=(
                                    "Processing"
                                    if attempt == 0
                                    else f"Retry {attempt + 1}/{conversation_max_retries}"
                                ),
                            )
                            result = await stage1_memcells_extraction.process_single_conversation(
                                conv_id=conv_index,  # Use extracted index
                                conversation=raw_data_dict[conv_id],  # Data uses original ID
                                save_dir=str(memcells_dir),
                                llm_provider=self.llm_provider,
                                atomic_fact_extractor=self.atomic_fact_extractor,
                                progress_counter=None,
                                progress=progress,
                                task_id=conv_task_id,
                                config=exp_config,
                            )
                            break
                        except Exception as exc:
                            last_exc = exc
                            if attempt == conversation_max_retries - 1:
                                progress.update(conv_task_id, status="❌ Failed")
                                raise

                            wait_time = min(15 * (2**attempt), 120)
                            console.print(
                                f"⚠️  Conv-{conv_index} failed during MemCell extraction "
                                f"(attempt {attempt + 1}/{conversation_max_retries}); "
                                f"retrying in {wait_time}s: {exc}",
                                style="yellow",
                            )
                            await asyncio.sleep(wait_time)
                            progress.update(conv_task_id, completed=0)

                    if last_exc is not None:
                        console.print(
                            f"✅ Conv-{conv_index} recovered after retry",
                            style="green",
                        )
                    progress.update(
                        conversation_tasks[conv_id],
                        status="✅",
                        completed=progress.tasks[conversation_tasks[conv_id]].total,
                    )
                    progress.update(main_task, advance=1)
                    return result

                # Execute all pending tasks concurrently
                if processing_tasks:
                    max_concurrent_requests = max(
                        1, int(getattr(exp_config, "max_concurrent_requests", 4))
                    )
                    semaphore = asyncio.Semaphore(max_concurrent_requests)

                    async def run_with_semaphore(conv_id, conv_index, conv_task_id):
                        async with semaphore:
                            return await run_with_completion(
                                conv_id, conv_index, conv_task_id
                            )

                    results = await asyncio.gather(
                        *[
                            run_with_semaphore(conv_id, conv_index, conv_task_id)
                            for conv_id, conv_index, conv_task_id in processing_tasks
                        ]
                    )
                else:
                    results = []

                progress.update(main_task, status="✅ Complete")

            end_time = time.time()
            elapsed = end_time - start_time

            # Statistics
            successful_convs = sum(1 for _, memcell_list in results if memcell_list)
            total_memcells = sum(len(memcell_list) for _, memcell_list in results)

            console.print("\n" + "=" * 60, style="dim")
            console.print("📊 MemCell Extraction Statistics:", style="bold")
            console.print(
                f"   ✅ Successfully processed: {successful_convs}/{len(pending_conversations)}",
                style="green",
            )
            console.print(f"   📝 Total memcells: {total_memcells}", style="blue")
            console.print(f"   ⏱️  Total time: {elapsed:.2f}s", style="yellow")
            if len(pending_conversations) > 0:
                console.print(
                    f"   🚀 Average per conversation: {elapsed/len(pending_conversations):.2f}s",
                    style="cyan",
                )
            console.print("=" * 60, style="dim")

        # ========== Stage 2: Index Building ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 2: Index Building", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # Call stage2 implementation to build indexes
        exp_config = self._convert_config_to_experiment_config()
        exp_config.num_conv = len(conversations)  # Set conversation count

        # Smart skip logic: check existing index files
        bm25_need_build = self._check_missing_indexes(
            index_dir=bm25_index_dir, num_conv=len(conversations), index_type="bm25"
        )

        emb_need_build = []
        use_hybrid = self.config.get("search", {}).get("use_hybrid_search", True)
        if use_hybrid:
            emb_need_build = self._check_missing_indexes(
                index_dir=emb_index_dir,
                num_conv=len(conversations),
                index_type="embedding",
            )

        # Statistics
        total_convs = len(conversations)
        bm25_to_build = len(bm25_need_build)
        emb_to_build = len(emb_need_build) if use_hybrid else 0

        console.print(f"\n📊 Index Building Statistics:")
        console.print(f"   Total conversations: {total_convs}")
        console.print(
            f"   BM25 index: need to build {bm25_to_build}, existing {total_convs - bm25_to_build}"
        )
        if use_hybrid:
            console.print(
                f"   Embedding index: need to build {emb_to_build}, existing {total_convs - emb_to_build}"
            )

        # Build BM25 index
        if bm25_to_build > 0:
            console.print(
                f"\n🔨 Building BM25 index ({bm25_to_build} conversations)...",
                style="yellow",
            )
            stage2_index_building.build_bm25_index(
                config=exp_config, data_dir=memcells_dir, bm25_save_dir=bm25_index_dir
            )
            console.print("✅ BM25 index building completed", style="green")
        else:
            console.print("✅ All BM25 indexes exist, skipping build", style="green")

        # Build Embedding index (if enabled)
        if use_hybrid:
            if emb_to_build > 0:
                console.print(
                    f"\n🔨 Building Embedding index ({emb_to_build} conversations)...",
                    style="yellow",
                )
                await stage2_index_building.build_emb_index(
                    config=exp_config, data_dir=memcells_dir, emb_save_dir=emb_index_dir
                )
                console.print("✅ Embedding index building completed", style="green")
            else:
                console.print(
                    "✅ All Embedding indexes exist, skipping build", style="green"
                )

        # ========== Plan A: Return index metadata (lazy loading) ==========
        # Don't load indexes into memory, only return paths and metadata
        index_metadata = {
            "type": "lazy_load",  # Mark as lazy loading
            "memcells_dir": str(memcells_dir),
            "bm25_index_dir": str(bm25_index_dir),
            "emb_index_dir": str(emb_index_dir),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": use_hybrid,
            "total_conversations": len(conversations),
        }

        console.print(f"\n{'='*60}", style="dim")
        console.print(f"✅ Add stage completed", style="bold green")
        console.print(f"   📁 MemCells: {memcells_dir}", style="dim")
        console.print(f"   📁 BM25 index: {bm25_index_dir}", style="dim")
        if use_hybrid:
            console.print(f"   📁 Embedding index: {emb_index_dir}", style="dim")
        console.print(
            f"   💡 Using lazy loading strategy (memory-friendly)", style="cyan"
        )
        console.print(f"{'='*60}\n", style="dim")

        return index_metadata

    async def add_incremental_message(
        self,
        conversation: Conversation,
        *,
        sequence_index: int,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Create and append MemCells for one newly arrived message.

        The evaluation Add path writes one complete conversation file and then
        batch-builds its indexes.  This path closes the newly arrived content as
        an independent online MemCell, runs the same episode/atomic-fact memory
        extraction, appends its embeddings, and refreshes BM25's corpus-wide IDF.
        Existing MemCells and embeddings are never regenerated.
        """
        if len(conversation.messages) != 1:
            raise ValueError("EverMemOS incremental ingestion requires one message")
        if not hasattr(self, "_incremental_add_states"):
            self._incremental_add_states: Dict[str, Dict[str, Any]] = {}

        conversation_id = conversation.conversation_id
        state = self._incremental_add_states.setdefault(
            conversation_id,
            {"docs": [], "embedding_docs": [], "bm25": None},
        )
        message = conversation.messages[0]
        timestamp = (
            to_iso_format(message.timestamp)
            if message.timestamp is not None
            else to_iso_format(
                datetime(2023, 1, 1) + timedelta(seconds=sequence_index * 30)
            )
        )
        raw_message = {
            "sender_id": message.sender_id,
            "user_name": message.sender_name or message.sender_id,
            "sender_name": message.sender_name or message.sender_id,
            "content": message.content,
            "timestamp": timestamp,
        }
        for field in ("img_url", "blip_caption", "query"):
            if message.metadata.get(field) is not None:
                raw_message[field] = message.metadata[field]

        work_dir = self.output_dir / "incremental_work" / "memcells"
        work_dir.mkdir(parents=True, exist_ok=True)
        exp_config = self._convert_config_to_experiment_config()
        item_id = self._extract_conv_index(conversation_id)
        _, memcells = await stage1_memcells_extraction.process_single_conversation(
            conv_id=item_id,
            conversation=[raw_message],
            save_dir=str(work_dir),
            llm_provider=self.llm_provider,
            atomic_fact_extractor=self.atomic_fact_extractor,
            config=exp_config,
        )

        new_docs = []
        for memcell in memcells:
            doc = memcell.to_dict()
            atomic_fact = getattr(memcell, "atomic_fact", None)
            if atomic_fact:
                doc["atomic_fact"] = atomic_fact.to_dict()
            new_docs.append(doc)
        state["docs"].extend(new_docs)

        stemmer = stage2_index_building.PorterStemmer()
        stop_words = set(stage2_index_building.stopwords.words("english"))
        corpus = [
            stage2_index_building.tokenize(
                stage2_index_building.build_searchable_text(doc), stemmer, stop_words
            )
            for doc in state["docs"]
        ]
        state["bm25"] = BM25Okapi(corpus) if corpus else None

        texts = []
        field_maps = []
        for doc_offset, doc in enumerate(new_docs):
            atomic = (doc.get("atomic_fact") or {}).get("atomic_fact") or []
            for fact_index, fact in enumerate(atomic if isinstance(atomic, list) else []):
                fact_text = fact.get("fact") if isinstance(fact, dict) else fact
                if str(fact_text or "").strip():
                    texts.append(str(fact_text))
                    field_maps.append((doc_offset, f"atomic_fact_{fact_index}"))
            for field in ("subject", "summary", "episode"):
                if str(doc.get(field) or "").strip():
                    texts.append(str(doc[field]))
                    field_maps.append((doc_offset, field))

        embeddings = (
            await stage2_index_building.get_vectorize_service().get_embeddings(texts)
            if texts
            else []
        )
        new_embedding_docs = [{"doc": doc, "embeddings": {}} for doc in new_docs]
        for (doc_offset, field), embedding in zip(field_maps, embeddings):
            fields = new_embedding_docs[doc_offset]["embeddings"]
            if field.startswith("atomic_fact_"):
                fields.setdefault("atomic_facts", []).append(embedding)
            else:
                fields[field] = embedding
        state["embedding_docs"].extend(new_embedding_docs)
        update_path = self.output_dir / "evermemos_incremental_updates.jsonl"
        with update_path.open("a", encoding="utf-8") as file_object:
            file_object.write(
                json.dumps(
                    {
                        "conversation_id": conversation_id,
                        "sequence_index": sequence_index,
                        "memcells": new_docs,
                        "embedding_docs": new_embedding_docs,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )
            file_object.flush()
            os.fsync(file_object.fileno())
        self._temporal_search_cache.clear()
        return {
            "new_memcells": len(new_docs),
            "total_memcells": len(state["docs"]),
            "new_embedding_texts": len(texts),
        }

    async def finalize_incremental_conversation(
        self, conversation_id: str, **_kwargs
    ) -> None:
        """Persist accumulated MemCells and retrieval indexes once per conversation."""
        state = getattr(self, "_incremental_add_states", {}).get(conversation_id)
        if state is None:
            return
        conv_index = self._extract_conv_index(conversation_id)
        memcells_dir = self.output_dir / "memcells"
        bm25_dir = self.output_dir / "bm25_index"
        vectors_dir = self.output_dir / "vectors"
        for directory in (memcells_dir, bm25_dir, vectors_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (memcells_dir / f"memcell_list_conv_{conv_index}.json").write_text(
            json.dumps(state["docs"], ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        with (bm25_dir / f"bm25_index_conv_{conv_index}.pkl").open("wb") as file_object:
            pickle.dump({"bm25": state["bm25"], "docs": state["docs"]}, file_object)
        with (vectors_dir / f"embedding_index_conv_{conv_index}.pkl").open(
            "wb"
        ) as file_object:
            pickle.dump(state["embedding_docs"], file_object)

    async def search(
        self, query: str, conversation_id: str, index: Any, **kwargs
    ) -> SearchResult:
        """
        Search stage: Retrieve relevant MemCells.

        Lazy loading: Load indexes from files on demand (memory-friendly).
        """
        # Lazy loading - read indexes from files
        bm25_index_dir = Path(index["bm25_index_dir"])
        emb_index_dir = Path(index["emb_index_dir"])

        # Extract numeric index from conversation_id to find index files
        # Example: conversation_id = "locomo_0" -> conv_index = "0"
        conv_index = self._extract_conv_index(conversation_id)

        # Load BM25 index on demand (using numeric index)
        bm25_file = bm25_index_dir / f"bm25_index_conv_{conv_index}.pkl"
        if not bm25_file.exists():
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"BM25 index not found: {bm25_file.name}"},
            )

        with open(bm25_file, "rb") as f:
            bm25_index_data = pickle.load(f)

        bm25 = bm25_index_data.get("bm25")
        docs = bm25_index_data.get("docs")

        # Load Embedding index on demand (using numeric index)
        emb_index = None
        if index.get("use_hybrid_search"):
            emb_file = emb_index_dir / f"embedding_index_conv_{conv_index}.pkl"
            if emb_file.exists():
                with open(emb_file, "rb") as f:
                    emb_index = pickle.load(f)

        question_date = kwargs.get("question_date")
        cache_key = (conversation_id, question_date)
        cache_entry = self._temporal_search_cache.get(cache_key)
        if cache_entry is None:
            filtered_docs, filtered_emb_index, temporal_filter_metadata = (
                self._filter_docs_for_question_date(
                    docs=docs,
                    emb_index=emb_index,
                    question_date=question_date,
                )
            )
            filtered_bm25 = bm25
            if question_date and len(filtered_docs) != len(docs):
                filtered_bm25 = self._build_filtered_bm25_index(filtered_docs)

            cache_entry = {
                "docs": filtered_docs,
                "emb_index": filtered_emb_index,
                "bm25": filtered_bm25,
                "metadata": temporal_filter_metadata,
            }
            self._temporal_search_cache[cache_key] = cache_entry

        docs = cache_entry["docs"]
        emb_index = cache_entry["emb_index"]
        bm25 = cache_entry["bm25"]
        temporal_filter_metadata = dict(cache_entry["metadata"])
        if question_date and not docs:
            temporal_filter_metadata.setdefault(
                "error", "no visible memories before question_date"
            )
            temporal_filter_metadata["formatted_context"] = ""
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata=temporal_filter_metadata,
            )

        # Call stage3 retrieval implementation
        search_config = self.config.get("search", {})
        retrieval_mode = search_config.get("mode", "agentic")

        exp_config = self._convert_config_to_experiment_config()
        # Get correct format llm_config from exp_config
        llm_config = exp_config.llm_config.get(exp_config.llm_service, {})

        if retrieval_mode == "agentic":
            # Agentic retrieval
            top_results, metadata = await stage3_memory_retrivel.agentic_retrieval(
                query=query,
                config=exp_config,
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
            )
        elif retrieval_mode == "lightweight":
            # Lightweight retrieval
            top_results, metadata = await stage3_memory_retrivel.lightweight_retrieval(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                config=exp_config,
            )
        else:
            # Default to hybrid retrieval
            top_results = await stage3_memory_retrivel.hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=20,
                emb_candidates=search_config.get("hybrid_emb_candidates", 100),
                bm25_candidates=search_config.get("hybrid_bm25_candidates", 100),
                rrf_k=search_config.get("hybrid_rrf_k", 60),
            )
            metadata = {}

        # Get response_top_k from config (use early for consistency)
        response_top_k = exp_config.response_top_k

        # Convert to evaluation framework format (use response_top_k to be consistent with formatted_context)
        results = []
        for doc, score in top_results[:response_top_k]:
            results.append(
                {
                    "content": doc.get("episode", ""),
                    "score": float(score),
                    "metadata": {
                        "subject": doc.get("subject", ""),
                        "summary": doc.get("summary", ""),
                    },
                }
            )

        # Build formatted_context
        formatted_context = ""
        conversation = kwargs.get("conversation")
        if conversation and top_results:
            # Get speaker information
            speaker_a = conversation.metadata.get("speaker_a", "Speaker A")
            speaker_b = conversation.metadata.get("speaker_b", "Speaker B")

            # Build context using response_top_k
            retrieved_docs_text = []
            for doc, score in top_results[:response_top_k]:
                subject = doc.get('subject', 'N/A')
                episode = doc.get('episode', 'N/A')
                doc_text = f"{subject}: {episode}\n---"
                retrieved_docs_text.append(doc_text)

            speaker_memories = "\n\n".join(retrieved_docs_text)

            TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
            formatted_context = TEMPLATE.format(
                speaker_1=speaker_a,
                speaker_2=speaker_b,
                speaker_memories=speaker_memories,
            )

        # Add formatted_context to metadata
        metadata = {
            **temporal_filter_metadata,
            **metadata,
            "formatted_context": formatted_context,
        }

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata,
        )

    def release_conversation_search_cache(
        self, conversation_id: str, index: Any = None
    ) -> None:
        keys_to_remove = [
            cache_key
            for cache_key in self._temporal_search_cache
            if cache_key[0] == conversation_id
        ]
        for cache_key in keys_to_remove:
            self._temporal_search_cache.pop(cache_key, None)

    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Answer stage: Generate answer.

        Calls stage4_response.py implementation.
        """
        prompt_key = kwargs.get("prompt_key")
        if prompt_key and prompt_key.startswith("answer_prompt_mab_"):
            result = await self.llm_provider.generate_messages(
                messages=self._build_memoryagentbench_answer_messages(query, context),
                temperature=0,
            )
            return result.strip()

        if prompt_key:
            from evaluation.src.utils.config import load_yaml

            prompts = load_yaml(
                str(Path(__file__).resolve().parents[2] / "config" / "prompts.yaml")
            )
            prompt_template = prompts["online_api"]["default"][prompt_key]
            prompt = prompt_template.format(context=context, question=query)
            result = await self.llm_provider.generate(prompt=prompt, temperature=0)
            return result.strip()

        # Call stage4 answer generation implementation
        exp_config = self._convert_config_to_experiment_config()

        answer = await stage4_response.locomo_response(
            llm_provider=self.llm_provider,
            context=context,
            question=query,
            experiment_config=exp_config,
        )

        return answer

    def _build_memoryagentbench_answer_messages(
        self, query: str, context: str
    ) -> List[Dict[str, str]]:
        """Mirror the official MemoryAgentBench answer call shape."""
        system_prompt = (
            "You are a helpful AI. Answer the question based on query and memories.\n"
            f"{context}\n"
        )
        user_content = (
            f"{query}\n\nCurrent Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def get_system_info(self) -> Dict[str, Any]:
        """Return system info."""
        return {
            "name": "EverCore",
            "version": "1.0",
            "description": "EverCore memory system with agentic retrieval",
            "adapter": "Adapter connecting framework to EverCore implementation",
        }

    def _convert_config_to_experiment_config(self):
        """
        Convert evaluation framework config to ExperimentConfig format.
        """
        from evaluation.src.adapters.evermemos.config import ExperimentConfig
        import os

        exp_config = ExperimentConfig()

        # Map LLM configuration: convert YAML llm to ExperimentConfig llm_config format
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")

        exp_config.llm_service = provider
        exp_config.llm_config = {
            provider: {
                "llm_provider": provider,
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", ""),
                "base_url": llm_cfg.get("base_url")
                or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "temperature": llm_cfg.get("temperature", 0.3),
                "max_tokens": llm_cfg.get("max_tokens", 16384),
            }
        }

        # Map Add stage configuration (only override explicitly specified in YAML)
        add_config = self.config.get("add", {})
        if "max_retries" in add_config:
            exp_config.max_retries = max(1, int(add_config["max_retries"]))
        elif "max_retries" in self.config:
            exp_config.max_retries = max(1, int(self.config["max_retries"]))
        if "enable_foresight_extraction" in add_config:
            exp_config.enable_foresight_extraction = add_config[
                "enable_foresight_extraction"
            ]
        if "enable_clustering" in add_config:
            exp_config.enable_clustering = add_config["enable_clustering"]
        if "enable_profile_extraction" in add_config:
            exp_config.enable_profile_extraction = add_config[
                "enable_profile_extraction"
            ]
        if "max_concurrent_requests" in add_config:
            exp_config.max_concurrent_requests = max(
                1, int(add_config["max_concurrent_requests"])
            )
        if "atomic_fact_max_concurrency" in add_config:
            exp_config.atomic_fact_max_concurrency = max(
                1, int(add_config["atomic_fact_max_concurrency"])
            )

        if "sample_workers" in self.config:
            exp_config.max_concurrent_requests = max(
                1, int(self.config["sample_workers"])
            )

        # Map Search stage configuration (only override explicitly specified in YAML)
        search_config = self.config.get("search", {})
        if "response_top_k" in search_config:
            exp_config.response_top_k = max(1, int(search_config["response_top_k"]))
        elif "top_k" in search_config:
            exp_config.response_top_k = max(1, int(search_config["top_k"]))

        if "mode" in search_config:
            exp_config.retrieval_mode = search_config["mode"]
            exp_config.use_agentic_retrieval = exp_config.retrieval_mode == "agentic"

        # Map lightweight_search_mode (controls search method in lightweight mode)
        # Options: "bm25_only" | "hybrid" | "emb_only"
        if "lightweight_search_mode" in search_config:
            exp_config.lightweight_search_mode = search_config[
                "lightweight_search_mode"
            ]

        return exp_config

    def build_lazy_index(
        self, conversations: List[Conversation], output_dir: Any
    ) -> Dict[str, Any]:
        """
        Build EverCore lazy-load index metadata.

        EverCore specifics:
        - Local indexes (memcells, bm25, embeddings)
        - Lazy loading (only save metadata, don't load actual index files)

        Args:
            conversations: Conversation list
            output_dir: Output directory

        Returns:
            Index metadata dict
        """
        return {
            "type": "lazy_load",
            "memcells_dir": str(output_dir / "memcells"),
            "bm25_index_dir": str(output_dir / "bm25_index"),
            "emb_index_dir": str(output_dir / "vectors"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": True,
            "total_conversations": len(conversations),
        }
