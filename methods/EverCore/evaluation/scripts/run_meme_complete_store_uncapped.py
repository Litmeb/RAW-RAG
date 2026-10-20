"""Reproduce the saved 5k protocol, then remove only its packing budget.

Stages are independently resumable. Historical results are read-only. A system
is runnable only after every historical reader context and candidate count has
been reproduced. This is a packing sensitivity test, not a write-loss estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
EVAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL.parent))

import ijson  # noqa: E402
import tiktoken  # noqa: E402
import yaml  # noqa: E402

try:
    from . import build_meme_complete_store_oracle_runs as complete
    from . import build_meme_continuity_oracle_runs as legacy
except ImportError:
    import build_meme_complete_store_oracle_runs as complete
    import build_meme_continuity_oracle_runs as legacy

TAG = "complete-store-uncapped-20260920"
SYSTEMS = ("raw", "rag", "mem0_local", "memos_local", "memu_local",
           "emem", "evermemos", "memmachine")
READER_RESPONSE_MODEL = "gpt-4o-mini-2024-07-18"


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stream(path):
    with Path(path).open("rb") as handle:
        yield from ijson.items(handle, "item", use_float=True)


def sources(manifest_path):
    if manifest_path is None:
        raise ValueError("Build requires --source-manifest; historical analysis files are not bundled")
    path = Path(manifest_path).resolve()
    manifest = load(path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("systems"), dict):
        raise ValueError("Source manifest must contain a systems mapping")

    def resolve(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Manifest paths must be nonempty strings")
        target = Path(value)
        return (target if target.is_absolute() else path.parent / target).resolve()

    data = resolve(manifest.get("data"))
    if not data.is_file():
        raise FileNotFoundError(data)
    result = {}
    for system, spec in manifest["systems"].items():
        if system not in SYSTEMS or not isinstance(spec, dict):
            raise ValueError(f"Invalid system specification: {system}")
        source = resolve(spec.get("source_run"))
        store = resolve(spec["store_run"]) if spec.get("store_run") else None
        if system in {"rag", "mem0_local", "emem", "evermemos"} and store is None:
            raise ValueError(f"{system}: store_run must point to the native persisted store")
        for filename in ("answer_results.json", "eval_results.json",
                         "meme_continuity_oracle_audit.json"):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
        if store is not None and not store.is_dir():
            raise FileNotFoundError(store)
        result[system] = {"source": source, "store": store, "data": data}
    if not result:
        raise ValueError("Source manifest must select at least one system")
    return result


def destination(system, tag):
    return EVAL / "results" / f"meme-{system}-{tag}"


def token_count(encoding, text):
    return len(encoding.encode(text, disallowed_special=()))


def prompt_template():
    with (EVAL / "config/prompts.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["online_api"]["default"]["answer_prompt_meme_official"]


def reader_messages(answer, context, template):
    query = answer["question"]
    date = answer["metadata"].get("question_date")
    if date:
        query = (
            f"Question date: {date}\n"
            "Treat this as the current time when resolving relative phrases such as "
            '"recently", "last month", "previous", "current", or "now". '
            "Only use memories dated at or before this question date.\n"
            f"Question: {query}"
        )
    return [{"role": "user", "content": template.format(context=context, question=query)}]


def memos_store(source, ids, maps):
    cubes = defaultdict(list)
    for item in stream(source / "memos_complete_store_export.json"):
        cubes[item["_cube_id"]].append(item)

    def response(_endpoint, payload):
        # The original loader consumes the same archived object order per cube.
        return {"data": cubes[payload["mem_cube_id"]]}

    with patch.object(complete, "http_json", response), patch.object(
        complete, "flatten_memos_response", lambda rows: rows
    ):
        store, audit, _ = complete.load_memos_store("archived", "fine", ids, maps)
    return store, audit


def memu_store(source):
    store = defaultdict(list)
    for index, row in enumerate(stream(source / "memu_complete_store_export.json")):
        if not row["content"]:
            continue
        metadata = row["metadata"]
        store[metadata["conversation_id"]].append(
            legacy.MemoryObject(row["content"], 0.0, metadata, index)
        )
    return dict(store), load(source / "complete_store_audit.json")


def memmachine_store(source, ids, maps, citations_path=None):
    """Recover saved provenance; never guess an unarchived citation boundary."""
    exports = load(source / "memmachine_complete_store_export.json")
    citations = load(citations_path)["citations"] if citations_path else {}
    provenance = {}
    for row in stream(source / "search_results.json"):
        for item in row["results"]:
            metadata = item["metadata"]["native_metadata"]
            provenance[(row["conversation_id"], metadata["memory_kind"],
                        metadata["memory_id"])] = metadata
    store = {}
    unresolved = []
    first_seen = 0
    for cid in ids:
        objects = []
        export = exports[cid]
        boundaries = maps[cid]
        for item in export["episodic_memory"]:
            text, _ = complete.escape_model_control_tokens(str(item.get("content") or "").strip())
            if not text:
                continue
            metadata = item.get("metadata") or {}
            session = complete.memmachine_session_index(
                metadata, item.get("created_at"), boundaries
            )
            if session is None:
                raise ValueError(f"Unmapped archived episode: {cid}/{item['uid']}")
            timestamp = str(metadata.get("source_timestamp") or item.get("created_at") or "")
            speaker = str(metadata.get("source_speaker") or item.get("producer_id") or "")
            prefix = ([f"[{timestamp}]"] if timestamp else []) + ([f"{speaker}:"] if speaker else [])
            text = f"{' '.join(prefix)} {text}".strip()
            objects.append(legacy.MemoryObject(text, 0.0, {
                "memory_id": str(item["uid"]), "memory_kind": "episodic",
                "source_session_indices": [session],
            }, first_seen))
            first_seen += 1
        for item in export["semantic_memory"]:
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            identity = str(item["metadata"]["id"])
            metadata = provenance.get((cid, "semantic", identity))
            if identity in citations:
                sessions = sorted({
                    session for citation in citations[identity]
                    if (session := complete.memmachine_session_index(
                        citation["metadata"], citation.get("created_at"), boundaries
                    )) is not None
                })
                recovered = {
                    "memory_id": identity, "memory_kind": "semantic",
                    "source_session_indices": sessions or sorted(i for _, i in boundaries),
                    "conservative_unknown_provenance": not sessions,
                }
                if metadata is not None and metadata["source_session_indices"] != recovered["source_session_indices"]:
                    raise ValueError(f"Citation archive disagrees with historical metadata: {cid}/{identity}")
                metadata = recovered
            if metadata is None:
                unresolved.append({"conversation_id": cid, "memory_id": identity})
                # Unknown objects are excluded, not exposed at a guessed cutoff.
                metadata = {"memory_id": identity, "memory_kind": "semantic",
                            "source_session_indices": [], "unresolved_provenance": True}
            text, _ = complete.escape_model_control_tokens(
                f"{item.get('category') or 'semantic'} / "
                f"{item.get('feature_name') or item.get('tag') or 'fact'}: {value}"
            )
            objects.append(legacy.MemoryObject(text, 0.0, metadata, first_seen))
            first_seen += 1
        summary = str(export.get("working_memory_summary", {}).get("summary") or "").strip()
        if summary:
            text, _ = complete.escape_model_control_tokens(summary)
            objects.append(legacy.MemoryObject(text, 0.0, {
                "memory_id": f"{export['session_key']}:summary",
                "memory_kind": "episodic_summary",
                "source_session_indices": sorted(i for _, i in boundaries),
            }, first_seen))
            first_seen += 1
        store[cid] = objects
    audit = load(source / "complete_store_audit.json")
    audit["unresolved_archived_semantic_provenance"] = unresolved
    if citations_path:
        audit["supplemental_citations_path"] = str(citations_path)
        audit["supplemental_citations_sha256"] = file_digest(citations_path)
    return store, audit


def load_store(system, source, episodes, citations_path=None, store_run=None):
    maps = complete.session_timestamp_maps(episodes)
    ids = [f"meme_{i}" for i in range(len(episodes))]
    if system == "raw":
        return {}, {}
    if system in {"rag", "mem0_local", "emem", "evermemos"} and store_run is None:
        raise ValueError(f"{system} requires an explicit native store_run")
    if system == "rag":
        return complete.load_rag_store(store_run, maps, ids)
    if system == "mem0_local":
        return complete.load_mem0_store(store_run, maps)
    if system == "memos_local":
        return memos_store(source, ids, maps)
    if system == "memu_local":
        return memu_store(source)
    if system == "emem":
        return complete.load_emem_store(store_run, ids)
    if system == "evermemos":
        return complete.load_evermemos_store(store_run, ids, maps)
    return memmachine_store(source, ids, maps, citations_path)


def prepare_question(system, store, episodes, spec, encoding):
    if system == "raw":
        def pack(budget):
            return legacy.pack_raw_context(
                episodes[spec.episode_index], spec, encoding, budget
            )[:2]
        return pack, None, None
    if system == "emem":
        pool = complete.deduplicated_pool(store[(spec.conversation_id, spec.phase)])
    elif system == "rag":
        pool = [m for m in store[spec.conversation_id]
                if m.metadata.get("source_session_indices")
                and max(m.metadata["source_session_indices"]) <= spec.cutoff]
    else:
        pool = complete.phase_safe_pool(store[spec.conversation_id], spec.cutoff)
    if system == "rag":
        locations = {int(i) for i in episodes[spec.episode_index]["evidence_session_indices"]
                     if int(i) <= spec.cutoff}
        ranked, _ = complete.rank_rag_chunks(pool, spec, locations)
        def pack(budget):
            return complete.pack_rag_chunk_context(ranked, encoding, budget)[:2]
    else:
        ranked, _ = legacy.rank_memory_objects(pool, spec)
        if system == "emem" and spec.task_type not in {"Agg", "Del"}:
            ranked = complete.focus_emem_ranked_candidates(ranked, spec)
            def pack(budget):
                return complete.pack_emem_context(ranked, encoding, budget)
        else:
            def pack(budget):
                return legacy.pack_memory_context(ranked, encoding, budget)
    return pack, pool, ranked


def build(system, tag, citations_path=None, source_manifest=None):
    spec = sources(source_manifest)[system]
    source = spec["source"]
    output = destination(system, tag)
    inputs = {
        "source": str(source), "store": str(spec["store"]), "data": str(spec["data"]),
        "data_sha256": file_digest(spec["data"]),
        "source_answer_sha256": file_digest(source / "answer_results.json"),
        "source_eval_sha256": file_digest(source / "eval_results.json"),
        "source_audit_sha256": file_digest(source / "meme_continuity_oracle_audit.json"),
        "manifest_sha256": file_digest(source_manifest),
    }
    if (output / "manifest.json").exists():
        manifest = load(output / "manifest.json")
        if manifest.get("inputs") != inputs:
            raise ValueError("Prepared source inputs changed; choose a fresh --tag")
        if file_digest(output / "search_results.json") != manifest["search_sha256"]:
            raise ValueError("Previously prepared inputs changed")
        print(f"{system}: already prepared", flush=True)
        return manifest
    episodes = load(spec["data"])
    specs = legacy.question_specs(episodes, 0, len(episodes))
    answers = {r["question_id"]: r for r in stream(source / "answer_results.json")}
    old_audit = {r["question_id"]: r for r in load(source / "meme_continuity_oracle_audit.json")}
    if len(answers) != len(specs) or set(answers) != {s.question_id for s in specs}:
        raise ValueError("Historical question alignment failed")
    store, store_audit = load_store(system, source, episodes, citations_path, spec["store"])
    # Reader text and ordering do not depend on these potentially huge fields.
    for objects in store.values():
        for memory in objects:
            memory.metadata.pop("store_metadata", None)
            memory.metadata.pop("store_payload", None)
    encoding = tiktoken.get_encoding("o200k_base")
    template = prompt_template()
    rows, audit, mismatches = [], [], []
    for index, spec in enumerate(specs):
        pack, pool, ranked = prepare_question(system, store, episodes, spec, encoding)
        capped, capped_evidence = pack(legacy.evidence_budget(encoding, 5000, spec, True))
        selected, evidence = pack(None)
        capped_context = legacy.add_readout_policy(capped_evidence, spec, True)
        context = legacy.add_readout_policy(evidence, spec, True)
        old = answers[spec.question_id]
        old_context = old["formatted_context"]
        previous = old_audit[spec.question_id]
        pool_key = ("deduplicated_candidate_pool_size" if system == "emem"
                    else "phase_visible_candidate_pool_size")
        if capped_context != old_context:
            mismatches.append({"question_id": spec.question_id, "reason": "reader_context",
                               "saved_sha256": digest(old_context),
                               "rebuilt_sha256": digest(capped_context)})
        if pool is not None and len(pool) != previous[pool_key]:
            mismatches.append({"question_id": spec.question_id, "reason": "candidate_pool",
                               "saved": previous[pool_key], "rebuilt": len(pool)})
        count_key = "focused_candidate_count" if system == "emem" else "ranked_candidate_count"
        if ranked is not None and len(ranked) != previous[count_key]:
            mismatches.append({"question_id": spec.question_id, "reason": "selected_candidates",
                               "saved": previous[count_key], "rebuilt": len(ranked)})
        if ranked is not None and len(selected) != len(ranked):
            raise AssertionError("Uncapped packing dropped an admitted object")
        full_tokens = token_count(encoding, evidence)
        kept_tokens = token_count(encoding, capped_evidence)
        messages = reader_messages(old, context, template)
        prompt_tokens = token_count(encoding, messages[0]["content"]) + 7
        completion_tokens = 8192 if system == "emem" else 16384
        # A safety margin is conservative; over-window inputs are never clipped.
        fits = prompt_tokens + completion_tokens + 256 <= 128000
        clipped = len(capped) > 0 and not capped_evidence.endswith(
            "</RAW_EVIDENCE>\n\n" if system == "raw" else
            "</RAG_CHUNK>\n\n" if system == "rag" else "</MEMORY>\n\n"
        )
        record = {
            "question_id": spec.question_id, "conversation_id": spec.conversation_id,
            "phase": spec.phase, "task_type": spec.task_type,
            "phase_visible_objects": len(pool) if pool is not None else None,
            "phase_visible_content_tokens": sum(token_count(encoding, m.content) for m in pool)
            if pool is not None else None,
            "selected_objects_before_packing": len(selected), "objects_5k": len(capped),
            "objects_skipped_5k": len(selected) - len(capped),
            "objects_clipped_5k": int(clipped),
            "evidence_tokens_before_packing": full_tokens,
            "evidence_tokens_5k": kept_tokens,
            "tokens_dropped_5k": full_tokens - kept_tokens,
            "token_drop_fraction_5k": (full_tokens - kept_tokens) / max(1, full_tokens),
            "context_tokens_uncapped": token_count(encoding, context),
            "context_tokens_5k": token_count(encoding, old_context),
            "context_changed": context != old_context,
            "prompt_tokens_estimate": prompt_tokens, "reserved_output_tokens": completion_tokens,
            "fits_reader_window": fits, "context_sha256": digest(context),
            "old_context_sha256": digest(old_context),
            "request_sha256": digest(json.dumps(messages, ensure_ascii=False, sort_keys=True)),
        }
        audit.append(record)
        rows.append(legacy.base_row(spec, [], {
            "formatted_context": context, "context_token_budget": None,
            "oracle_complete_store_candidate_access": system != "raw",
            "uncapped_audit": record,
        }))
        if (index + 1) % 200 == 0:
            print(f"{system}: audited {index + 1}/{len(specs)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    save(output / "reconstruction_audit.json", {
        "system": system, "questions": len(specs), "mismatches": mismatches,
        "store": store_audit,
    })
    if mismatches or store_audit.get("unresolved_archived_semantic_provenance"):
        raise ValueError(f"{system}: {len(mismatches)} reconstruction mismatches; no QA authorized")
    save(output / "search_results.json", rows)
    save(output / "packing_audit.json", audit)
    save(output / "reader_prompt.json", {"template": template})
    old_eval = load(source / "eval_results.json")
    manifest = {
        "system": system, "source": str(source), "questions": len(specs),
        "changed_questions": sum(r["context_changed"] for r in audit),
        "over_window_questions": sum(not r["fits_reader_window"] for r in audit),
        "reader_model": "gpt-4o-mini", "reader_temperature": 0,
        "reader_max_tokens": 8192 if system == "emem" else 16384,
        "judge_model": old_eval["metadata"]["model"],
        "judge_num_runs": old_eval["metadata"]["num_runs"],
        "context_window": 128000, "safety_margin_tokens": 256,
        "search_sha256": file_digest(output / "search_results.json"),
        "source_answer_sha256": file_digest(source / "answer_results.json"),
        "source_eval_sha256": file_digest(source / "eval_results.json"),
        "builder_sha256": file_digest(Path(__file__)),
        "prompt_sha256": digest(template),
        "protocol": "original selection/filter/order/prompt; packing budget only removed",
        "reconstruction": f"all {len(specs)} source reader inputs and candidate counts matched",
        "inputs": inputs,
    }
    save(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def detailed(evaluation):
    rows = evaluation["detailed_results"]
    return [r for group in rows.values() for r in group] if isinstance(rows, dict) else rows


def compact_answer(answer):
    return {k: v for k, v in answer.items() if k not in {"formatted_context", "search_results"}}


def refusal_judgment(old):
    return {
        **{k: v for k, v in old.items() if k not in {
            "u_pass", "u_score", "judge_runs", "u_reason", "filtered_u_pass",
            "pass_type", "u_pass_count", "u_pass_total", "u_pass_per_entity",
        }},
        "generated_answer": "", "u_pass": False, "u_score": 0.0,
        "judge_runs": [], "u_reason": "Provider content_filter: no usable answer",
        "evaluation_status": "provider_refusal_scored_incorrect",
    }


async def execute(system, tag, workers, limit=None):
    from dotenv import dotenv_values
    from openai import AsyncOpenAI
    from evaluation.src.core.data_models import AnswerResult
    from evaluation.src.evaluators.meme_judge import MEMEJudge, summarize_meme_results

    output = destination(system, tag)
    started = datetime.now(timezone.utc).isoformat()
    failure_path = output / "failures.json"
    history_path = output / "execution_history.json"
    history = load(history_path) if history_path.exists() else []
    if failure_path.exists() and load(failure_path):
        prior = load(failure_path)
        if not any(record.get("failures") == prior for record in history):
            history.append({"stage": "previous_attempt", "failures": prior})
            save(history_path, history)
    manifest = load(output / "manifest.json")
    if file_digest(output / "search_results.json") != manifest["search_sha256"]:
        raise ValueError("Prepared input hash changed")
    source = Path(manifest["source"])
    for name in ("answer", "eval"):
        if file_digest(source / f"{name}_results.json") != manifest[f"source_{name}_sha256"]:
            raise ValueError("Historical result changed since preparation")
    env = {**dotenv_values(EVAL.parent / ".env"), **os.environ}
    api_key = env.get("LLM_API_KEY", "").split(",")[0].strip()
    base_url = env.get("LLM_BASE_URL")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)
    judge = MEMEJudge({"llm": {
        "api_key": env.get("EVALUATE_API_KEY") or api_key,
        "base_url": env.get("EVALUATE_BASE_URL") or base_url,
        "model": manifest["judge_model"],
    }, "num_runs": manifest["judge_num_runs"]})
    judge_requests = ContextVar("judge_requests", default=None)
    original_judge_create = judge.client.chat.completions.create

    async def audited_judge_create(**kwargs):
        response = await original_judge_create(**kwargs)
        audit = judge_requests.get()
        if audit is None:
            raise RuntimeError("Judge call has no question audit scope")
        audit.append({
            "response_id": response.id, "response_model": response.model,
            "request_sha256": digest(json.dumps(kwargs["messages"], ensure_ascii=False, sort_keys=True)),
            "usage": response.usage.model_dump() if response.usage else None,
        })
        if response.model != manifest["judge_model"]:
            raise ValueError(f"Provider substituted judge model: {response.model}")
        return response

    judge.client.chat.completions.create = audited_judge_create
    searches = list(stream(output / "search_results.json"))
    old_answers = {r["question_id"]: r for r in stream(source / "answer_results.json")}
    old_judgments = {r["question_id"]: r for r in detailed(load(source / "eval_results.json"))}
    template = load(output / "reader_prompt.json")["template"]
    if digest(template) != manifest["prompt_sha256"]:
        raise ValueError("Frozen prompt changed")
    checkpoint = output / "qa_checkpoint.json"
    progress = load(checkpoint) if checkpoint.exists() else {}
    model_audit_path = output / "invalidated_model_responses.json"
    model_audit = load(model_audit_path) if model_audit_path.exists() else []
    judgment_archive_path = output / "unverified_judgment_archive.json"
    judgment_archive = load(judgment_archive_path) if judgment_archive_path.exists() else {}
    for qid, entry in list(progress.items()):
        request = entry.get("request")
        if request and request["response_model"] != READER_RESPONSE_MODEL:
            model_audit.append({"question_id": qid, "reason": "provider_model_substitution",
                                "invalidated_checkpoint": entry})
            del progress[qid]
        elif (entry.get("status") == "complete" and entry.get("reuse") == "none"
              and "judge_requests" not in entry):
            judgment_archive[qid] = entry["judgment"]
            entry["status"] = "answered"
            del entry["judgment"]
    if model_audit:
        save(model_audit_path, model_audit)
    if judgment_archive:
        save(judgment_archive_path, judgment_archive)
    semaphore = asyncio.Semaphore(workers)
    pending = []
    for search in searches:
        qid = search["question_id"]
        info = search["retrieval_metadata"]["uncapped_audit"]
        if qid in progress and progress[qid].get("status") == "complete":
            if progress[qid]["context_sha256"] != info["context_sha256"]:
                raise ValueError("Checkpoint input mismatch")
            continue
        if not info["fits_reader_window"]:
            progress[qid] = {"status": "over_window", "context_sha256": info["context_sha256"]}
        elif not info["context_changed"]:
            progress[qid] = {
                "status": "complete", "context_sha256": info["context_sha256"],
                "answer": compact_answer(old_answers[qid]), "judgment": old_judgments[qid],
                "reuse": "identical_reader_input_and_saved_judgment",
            }
        else:
            pending.append(search)
    save(checkpoint, progress)
    if limit is not None:
        pending = pending[:limit]
    failures = []

    async def one(search):
        qid = search["question_id"]
        async with semaphore:
            context = search["retrieval_metadata"]["formatted_context"]
            info = search["retrieval_metadata"]["uncapped_audit"]
            previous = progress.get(qid, {})
            judge_request_records = []
            audit_token = judge_requests.set(judge_request_records)
            try:
                if previous.get("answer") and previous.get("context_sha256") == info["context_sha256"]:
                    answer = previous["answer"]
                    request_audit = previous.get("request", {})
                else:
                    messages = reader_messages(old_answers[qid], context, template)
                    if digest(json.dumps(messages, ensure_ascii=False, sort_keys=True)) != info["request_sha256"]:
                        raise ValueError("Actual request differs from audited request")
                    response = await client.chat.completions.create(
                        model=manifest["reader_model"], messages=messages,
                        temperature=0, max_tokens=manifest["reader_max_tokens"],
                    )
                    if response.model != READER_RESPONSE_MODEL:
                        model_audit.append({
                            "question_id": qid, "reason": "provider_model_substitution",
                            "response_model": response.model, "response_id": response.id,
                            "request_sha256": info["request_sha256"],
                            "usage": response.usage.model_dump() if response.usage else None,
                        })
                        save(model_audit_path, model_audit)
                        raise ValueError(f"Provider substituted reader model: {response.model}")
                    choice = response.choices[0]
                    text = (choice.message.content or "").strip()
                    if choice.finish_reason not in {"stop", "content_filter"} or (
                        choice.finish_reason == "stop" and not text
                    ):
                        raise ValueError(f"Incomplete reader output: {choice.finish_reason}")
                    if response.usage is None:
                        raise ValueError("Missing usage: cannot verify full prompt delivery")
                    if response.usage.prompt_tokens != info["prompt_tokens_estimate"]:
                        raise ValueError(
                            "Server prompt-token count differs from full audited input: "
                            f"{response.usage.prompt_tokens} != {info['prompt_tokens_estimate']}"
                        )
                    text = text.split("FINAL ANSWER:")[-1].strip()
                    if choice.finish_reason == "content_filter":
                        text = ""
                    answer = {**compact_answer(old_answers[qid]), "answer": text}
                    request_audit = {
                        "request_sha256": info["request_sha256"],
                        "response_model": response.model,
                        "response_id": response.id,
                        "response_created": response.created,
                        "finish_reason": choice.finish_reason,
                        "usage": response.usage.model_dump() if response.usage else None,
                    }
                    progress[qid] = {
                        "status": "answered", "context_sha256": info["context_sha256"],
                        "answer": answer, "request": request_audit,
                    }
                    save(checkpoint, progress)
                if request_audit.get("finish_reason") == "content_filter":
                    judgment = refusal_judgment(old_judgments[qid])
                    reuse = "provider_refusal_scored_incorrect"
                elif answer["answer"] == old_answers[qid]["answer"]:
                    judgment = old_judgments[qid]
                    reuse = "identical_answer_saved_judgment"
                else:
                    allowed = AnswerResult.__dataclass_fields__
                    result = AnswerResult(**{k: v for k, v in answer.items() if k in allowed})
                    judgment = await judge._evaluate_single_answer(result)
                    reuse = "none"
                progress[qid] = {
                    "status": "complete", "context_sha256": info["context_sha256"],
                    "answer": answer, "judgment": judgment, "request": request_audit,
                    "reuse": reuse, "judge_requests": judge_request_records,
                }
                save(checkpoint, progress)
                done = sum(r.get("status") == "complete" for r in progress.values())
                if done % 25 == 0:
                    print(f"{system}: completed {done}/{len(searches)}", flush=True)
            except Exception as exc:
                failures.append({"question_id": qid, "error": str(exc)[:1000],
                                 "judge_requests": judge_request_records})
                print(f"{system}: {qid}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            finally:
                judge_requests.reset(audit_token)

    try:
        await asyncio.gather(*(one(search) for search in pending))
    finally:
        await client.close()
        await judge.client.close()
        save(output / "failures.json", failures)
        history.append({"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                        "pending_questions": len(pending), "failures": failures})
        save(history_path, history)
    complete_rows = [progress[s["question_id"]] for s in searches
                     if progress.get(s["question_id"], {}).get("status") == "complete"]
    print(f"{system}: {len(complete_rows)}/{len(searches)} complete; {len(failures)} errors", flush=True)
    if len(complete_rows) == len(searches):
        contexts = {r["question_id"]: r["retrieval_metadata"]["formatted_context"] for r in searches}
        answers = [{**r["answer"], "formatted_context": contexts[r["answer"]["question_id"]]}
                   for r in complete_rows]
        groups = defaultdict(list)
        for row in complete_rows:
            judgment = copy.deepcopy(row["judgment"])
            groups[judgment["category"]].append(judgment)
        summary = summarize_meme_results(groups)
        judgments = [row for group in groups.values() for row in group]
        before = [row for row in judgments if row["phase"] == "before"]
        after = [row for row in judgments if row["phase"] == "after"]
        before_pass = sum(bool(row["u_pass"]) for row in before)
        after_pass = sum(bool(row["u_pass"]) for row in after)
        filtered_after_pass = sum(row["filtered_u_pass"] for row in after)
        correct = before_pass + filtered_after_pass
        save(output / "answer_results.json", answers)
        save(output / "eval_results.json", {
            "total_questions": len(answers), "correct": correct,
            "accuracy": summary["overall_accuracy"],
            "detailed_results": groups, "metadata": {
                "model": manifest["judge_model"], "num_runs": manifest["judge_num_runs"],
                "official_accuracy": summary["overall_accuracy"],
                "raw_accuracy": (before_pass + after_pass) / len(answers),
                "filtered_accuracy": correct / len(answers),
                "before_pass": before_pass, "before_total": len(before),
                "before_accuracy": before_pass / len(before),
                "after_pass": after_pass, "after_total": len(after),
                "after_accuracy": after_pass / len(after),
                "after_filtered_pass": filtered_after_pass,
                "after_filtered_total": len(after),
                "after_filtered_accuracy": filtered_after_pass / len(after),
                **summary, "reuse_counts": {
                    value: sum(r["reuse"] == value for r in complete_rows)
                    for value in sorted({r["reuse"] for r in complete_rows})
                },
            },
        })
    if failures:
        raise RuntimeError(f"{system}: incomplete requests; rerun to resume")


def main():
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, "reconfigure"):
            output.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "run"))
    parser.add_argument("--system", choices=SYSTEMS, action="append")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--source-manifest", type=Path,
                        help="JSON mapping data and systems to capped runs and native stores (required for build).")
    parser.add_argument("--memmachine-citations", type=Path,
                        help="Historical complete feature-id -> citation-list archive, including empty lists.")
    args = parser.parse_args()
    if not args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.tag):
        parser.error("--tag must contain only letters, digits, hyphens, or underscores")
    if args.stage == "build" and args.source_manifest is None:
        parser.error("build requires --source-manifest")
    selected = args.system or (
        list(sources(args.source_manifest)) if args.stage == "build" else SYSTEMS
    )
    failures = []
    for system in selected:
        try:
            if args.stage == "build":
                build(system, args.tag, args.memmachine_citations, args.source_manifest)
            else:
                asyncio.run(execute(system, args.tag, args.workers, args.limit))
        except Exception as exc:
            failures.append({"system": system, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {system}: {exc}", flush=True)
    failure_path = EVAL / "analysis" / args.tag / f"{args.stage}_failures.json"
    previous = load(failure_path) if failure_path.exists() else []
    selected_systems = set(selected)
    save(failure_path, [r for r in previous if r["system"] not in selected_systems] + failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
