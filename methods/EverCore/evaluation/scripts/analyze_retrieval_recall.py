"""Analyze retrieval recall @embedding_top_k and @rerank_top_k for a RAG run."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def cosine_similarity(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_dia_to_turn(conversation_data):
    session_keys = sorted(
        [
            k
            for k in conversation_data
            if k.startswith("session_") and not k.endswith("_date_time")
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    dia_to_turn = {}
    turn_idx = 0
    for session_key in session_keys:
        for msg in conversation_data[session_key]:
            dia_id = msg.get("dia_id")
            if dia_id:
                dia_to_turn[dia_id] = turn_idx
            turn_idx += 1
    return dia_to_turn


def chunk_covers_turn(chunk, turn_idx):
    start = chunk.get("start_turn")
    end = chunk.get("end_turn")
    if start is None or end is None:
        return False
    return start <= turn_idx < end


def chunk_hits_evidence(chunk, evidence_turns):
    return any(chunk_covers_turn(chunk, t) for t in evidence_turns)


def top_k_hit(chunks, evidence_turns):
    for chunk in chunks:
        if chunk_hits_evidence(chunk, evidence_turns):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default="evaluation/data/locomo/locomo10.json")
    parser.add_argument("--embedding-top-k", type=int, default=50)
    parser.add_argument("--rerank-top-k", type=int, default=5)
    parser.add_argument("--filter-category", default="5")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    filter_categories = {c.strip() for c in args.filter_category.split(",") if c.strip()}

    with open(args.data, encoding="utf-8") as f:
        raw_data = json.load(f)

    qa_items = []
    for idx, item in enumerate(raw_data):
        conv_id = f"locomo_{idx}"
        dia_map = build_dia_to_turn(item["conversation"])
        for qa_idx, qa in enumerate(item.get("qa", [])):
            cat = str(qa.get("category")) if qa.get("category") is not None else None
            if cat in filter_categories:
                continue
            evidence_turns = [dia_map[ev] for ev in qa.get("evidence", []) if ev in dia_map]
            qa_items.append(
                {
                    "question_id": f"{conv_id}_qa{qa_idx}",
                    "conversation_id": conv_id,
                    "question": qa["question"],
                    "category": cat,
                    "evidence": qa.get("evidence", []),
                    "evidence_turns": evidence_turns,
                }
            )

    with open(run_dir / "rag_index.json", encoding="utf-8") as f:
        index = json.load(f)
    conv_indices = index["conversations"]

    query_embeddings = {}
    with open(run_dir / "logs/rag_requests.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("call_type") == "embedding" and obj.get("stage") == "search":
                qid = obj.get("question_id")
                if qid and qid not in query_embeddings:
                    query_embeddings[qid] = obj["response"]["data"][0]["embedding"]

    with open(run_dir / "search_results.json", encoding="utf-8") as f:
        search_results = json.load(f)
    search_by_qid = {
        sr.get("retrieval_metadata", {}).get("question_id"): sr for sr in search_results
    }

    emb_any_hit = emb_all_hit = rerank_any_hit = rerank_all_hit = 0
    evidence_total = evidence_emb_hit = evidence_rerank_hit = 0
    emb_hit_rerank_miss = 0
    by_category = defaultdict(lambda: {"total": 0, "emb_any": 0, "rerank_any": 0})
    missed_emb = []
    emb_hit_rerank_miss_samples = []

    for qa in qa_items:
        if not qa["evidence_turns"]:
            continue

        conv_index = conv_indices[qa["conversation_id"]]
        chunks = conv_index["chunks"]
        embeddings = conv_index["embeddings"]
        q_emb = query_embeddings[qa["question_id"]]

        scored = sorted(
            range(len(embeddings)),
            key=lambda i: cosine_similarity(q_emb, embeddings[i]),
            reverse=True,
        )
        top50 = [chunks[i] for i in scored[: args.embedding_top_k]]

        sr = search_by_qid.get(qa["question_id"])
        rerank_chunks = []
        if sr:
            for r in sr["results"][: args.rerank_top_k]:
                ci = r["metadata"].get("chunk_index")
                rerank_chunks.append(chunks[ci] if ci is not None else r["metadata"])

        emb_any = top_k_hit(top50, qa["evidence_turns"])
        rerank_any = top_k_hit(rerank_chunks, qa["evidence_turns"]) if rerank_chunks else False
        emb_all = all(
            any(chunk_covers_turn(c, t) for c in top50) for t in qa["evidence_turns"]
        )
        rerank_all = (
            all(any(chunk_covers_turn(c, t) for c in rerank_chunks) for t in qa["evidence_turns"])
            if rerank_chunks
            else False
        )

        emb_any_hit += int(emb_any)
        emb_all_hit += int(emb_all)
        rerank_any_hit += int(rerank_any)
        rerank_all_hit += int(rerank_all)

        cat = qa["category"]
        by_category[cat]["total"] += 1
        by_category[cat]["emb_any"] += int(emb_any)
        by_category[cat]["rerank_any"] += int(rerank_any)

        for t in qa["evidence_turns"]:
            evidence_total += 1
            if any(chunk_covers_turn(c, t) for c in top50):
                evidence_emb_hit += 1
            if any(chunk_covers_turn(c, t) for c in rerank_chunks):
                evidence_rerank_hit += 1

        if emb_any and not rerank_any:
            emb_hit_rerank_miss += 1
            if len(emb_hit_rerank_miss_samples) < 5:
                emb_hit_rerank_miss_samples.append(qa)
        if not emb_any and len(missed_emb) < 5:
            missed_emb.append(qa)

    total = sum(v["total"] for v in by_category.values())
    cat_names = {"1": "Single-hop", "2": "Temporal", "3": "Multi-hop", "4": "Open-domain"}

    print("=" * 60)
    print(f"Retrieval Recall: {run_dir.name}")
    print("=" * 60)
    print(f"Questions: {total}  |  Evidence snippets: {evidence_total}")
    print()
    print("Question-level (any evidence in top-K):")
    print(f"  Embedding @{args.embedding_top_k}: {emb_any_hit}/{total} = {100 * emb_any_hit / total:.2f}%")
    print(f"  Rerank   @{args.rerank_top_k}:  {rerank_any_hit}/{total} = {100 * rerank_any_hit / total:.2f}%")
    print(f"  Rerank loss (emb hit -> rerank miss): {emb_hit_rerank_miss} ({100 * emb_hit_rerank_miss / total:.2f}%)")
    print()
    print("Question-level (all evidence in top-K):")
    print(f"  Embedding @{args.embedding_top_k}: {emb_all_hit}/{total} = {100 * emb_all_hit / total:.2f}%")
    print(f"  Rerank   @{args.rerank_top_k}:  {rerank_all_hit}/{total} = {100 * rerank_all_hit / total:.2f}%")
    print()
    print("Evidence-level:")
    print(f"  Embedding @{args.embedding_top_k}: {evidence_emb_hit}/{evidence_total} = {100 * evidence_emb_hit / evidence_total:.2f}%")
    print(f"  Rerank   @{args.rerank_top_k}:  {evidence_rerank_hit}/{evidence_total} = {100 * evidence_rerank_hit / evidence_total:.2f}%")
    print()
    print("By category (any evidence hit):")
    for cat in sorted(by_category.keys(), key=lambda x: int(x)):
        v = by_category[cat]
        name = cat_names.get(cat, cat)
        print(
            f"  Cat {cat} ({name}): "
            f"emb@{args.embedding_top_k}={100 * v['emb_any'] / v['total']:.1f}% "
            f"rerank@{args.rerank_top_k}={100 * v['rerank_any'] / v['total']:.1f}% "
            f"({v['total']} q)"
        )


if __name__ == "__main__":
    main()
