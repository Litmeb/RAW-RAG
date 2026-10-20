import asyncio
import json
from copy import deepcopy
from pathlib import Path

from evaluation.src.core.data_models import Conversation, QAPair, SearchResult
from evaluation.src.core.stages.search_stage import run_search_stage
from evaluation.src.utils.checkpoint import CheckpointManager


class FakeAdapter:
    def __init__(self):
        self.config = {
            "search": {
                "num_workers": 2,
                "timeout_seconds": 1,
                "max_retries": 1,
            },
            "sample_workers": 1,
        }
        self.num_workers = 2
        self.calls = []

    async def search(self, query, conversation_id, index, **kwargs):
        question_id = kwargs["question_id"]
        self.calls.append(question_id)
        await asyncio.sleep(0)
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            question_id=question_id,
            results=[{"content": question_id, "score": 1.0}],
        )


class RecordingCheckpoint:
    def __init__(self, initial=None):
        self.current = deepcopy(initial or {})
        self.snapshots = []
        self.deleted = False

    def load_search_progress(self):
        return deepcopy(self.current)

    def save_search_progress(self, results):
        self.current = deepcopy(results)
        self.snapshots.append(deepcopy(results))

    def delete_search_checkpoint(self):
        self.deleted = True


def make_qa(question_id):
    return QAPair(
        question_id=question_id,
        question=f"question {question_id}",
        answer="answer",
        metadata={"conversation_id": "conversation_0"},
    )


def test_search_resumes_partial_conversation_and_checkpoints_each_question():
    qa_pairs = [make_qa("q0"), make_qa("q1"), make_qa("q2")]
    initial = {
        "conversation_0": [
            {
                "question_id": "q0",
                "query": "question q0",
                "conversation_id": "conversation_0",
                "results": [{"content": "q0", "score": 1.0}],
                "retrieval_metadata": {},
            }
        ]
    }
    checkpoint = RecordingCheckpoint(initial)
    adapter = FakeAdapter()

    results = asyncio.run(
        run_search_stage(
            adapter=adapter,
            qa_pairs=qa_pairs,
            index=None,
            conversations=[Conversation("conversation_0", [])],
            checkpoint_manager=checkpoint,
            logger=None,
        )
    )

    assert sorted(adapter.calls) == ["q1", "q2"]
    assert len(checkpoint.snapshots) == 2
    assert checkpoint.deleted is True
    assert [result.question_id for result in results] == ["q0", "q1", "q2"]


def test_search_checkpoint_write_is_atomic(tmp_path: Path):
    manager = CheckpointManager(tmp_path, "test")
    payload = {"conversation_0": [{"question_id": "q0"}]}

    manager.save_search_progress(payload)

    assert json.loads(manager.search_checkpoint.read_text(encoding="utf-8")) == payload
    assert not manager.search_checkpoint.with_suffix(".json.tmp").exists()
