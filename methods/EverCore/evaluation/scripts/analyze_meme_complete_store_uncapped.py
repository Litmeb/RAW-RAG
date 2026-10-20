"""Audit and summarize paired complete-store 5k/uncapped experiments."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .run_meme_complete_store_uncapped import (
        EVAL, READER_RESPONSE_MODEL, SYSTEMS, TAG, destination, detailed, digest, file_digest,
        load, save, sources, stream,
    )
except ImportError:
    from run_meme_complete_store_uncapped import (
        EVAL, READER_RESPONSE_MODEL, SYSTEMS, TAG, destination, detailed, digest, file_digest,
        load, save, sources, stream,
    )

TASKS = ("ER", "Agg", "Tr", "Del", "Cas", "Abs")


def scores(evaluation):
    rows = detailed(evaluation)
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate judged questions")
    before = {(r["conversation_id"], r["entity_key"]): bool(r["u_pass"])
              for r in rows if r["phase"] == "before"}
    result = []
    for row in rows:
        score = bool(row["u_pass"])
        if row["phase"] == "after" and row["task_type"] in {"Del", "Cas", "Abs"}:
            score &= before.get((row["conversation_id"], row["entity_key"]), False)
        if row["phase"] == "after" and score != bool(row["filtered_u_pass"]):
            raise ValueError("Saved filtered score disagrees with recomputed before gate")
        result.append({**row, "score": int(score)})
    return result


def summary(rows):
    by_task = {}
    for task in TASKS:
        subset = [r for r in rows if r["phase"] == "after" and r["task_type"] == task]
        by_task[task] = {
            "n": len(subset), "accuracy": float(np.mean([r["score"] for r in subset]))
            if subset else None,
            "ungated_accuracy": float(np.mean([r["u_pass"] for r in subset]))
            if subset else None,
        }
    present = [v["accuracy"] for v in by_task.values() if v["n"]]
    before = [r for r in rows if r["phase"] == "before"]
    non_abs = [v["accuracy"] for k, v in by_task.items() if k != "Abs" and v["n"]]
    return {
        "questions": len(rows), "after_questions": sum(v["n"] for v in by_task.values()),
        "task_accuracy": by_task,
        "macro": float(np.mean(present)) if present else None,
        "macro_tasks": [k for k, v in by_task.items() if v["n"]],
        "non_abs_macro": float(np.mean(non_abs)) if non_abs else None,
        "before_accuracy": float(np.mean([r["score"] for r in before])) if before else None,
    }


def paired(left, right, samples=20000, seed=20260920):
    left = {r["question_id"]: r for r in left if r["phase"] == "after"}
    right = {r["question_id"]: r for r in right if r["phase"] == "after"}
    if left.keys() != right.keys():
        raise ValueError("Paired question sets differ")
    if not left:
        return None
    tasks = [t for t in TASKS if any(r["task_type"] == t for r in left.values())]
    episodes = sorted({r["conversation_id"] for r in left.values()})
    counts = np.zeros((len(episodes), len(tasks), 3))
    for qid, row in left.items():
        other = right[qid]
        if (row["conversation_id"], row["task_type"]) != (
            other["conversation_id"], other["task_type"]
        ):
            raise ValueError("Paired identity mismatch")
        i, j = episodes.index(row["conversation_id"]), tasks.index(row["task_type"])
        counts[i, j] += [row["score"], other["score"], 1]
    totals = counts.sum(axis=0)
    difference = float(((totals[:, 0] - totals[:, 1]) / totals[:, 2]).mean())
    draws = []
    rng = np.random.default_rng(seed)
    for start in range(0, samples, 1000):
        indices = rng.integers(0, len(episodes), (min(1000, samples - start), len(episodes)))
        resampled = counts[indices].sum(axis=1)
        valid = (resampled[:, :, 2] > 0).all(axis=1)
        resampled = resampled[valid]
        draws.extend(((resampled[:, :, 0] - resampled[:, :, 1]) / resampled[:, :, 2]).mean(axis=1))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "difference": difference, "ci95_low": float(low), "ci95_high": float(high),
        "episodes": len(episodes), "questions": len(left), "tasks": tasks,
        "resampling_unit": "paired episode", "samples": samples, "valid_samples": len(draws),
        "seed": seed,
    }


def describe(values):
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)), "p95": float(np.quantile(values, .95)),
        "max": int(max(values)),
    }


def packing(rows):
    before = sum(r["evidence_tokens_before_packing"] for r in rows)
    return {
        "questions": len(rows), "affected": sum(r["context_changed"] for r in rows),
        "affected_fraction": sum(r["context_changed"] for r in rows) / len(rows),
        "prepack_tokens": describe([r["evidence_tokens_before_packing"] for r in rows]),
        "capped_context_tokens": describe([r["context_tokens_5k"] for r in rows]),
        "uncapped_context_tokens": describe([r["context_tokens_uncapped"] for r in rows]),
        "skipped_objects": sum(r["objects_skipped_5k"] for r in rows),
        "clipped_objects": sum(r["objects_clipped_5k"] for r in rows),
        "micro_token_drop_fraction": sum(r["tokens_dropped_5k"] for r in rows) / max(1, before),
        "mean_question_token_drop_fraction": float(np.mean([r["token_drop_fraction_5k"] for r in rows])),
        "over_window": sum(not r["fits_reader_window"] for r in rows),
    }


def verify_requests(run, audit):
    checkpoint = load(run / "qa_checkpoint.json")
    answers = {r["question_id"]: r for r in stream(run / "answer_results.json")}
    checks = []
    models = Counter()
    usage = Counter()
    refusals = []
    judge_models = Counter()
    judge_model = load(run / "manifest.json")["judge_model"]
    for row in audit:
        qid = row["question_id"]
        entry = checkpoint[qid]
        if entry["status"] != "complete":
            raise ValueError(f"Incomplete question: {qid}")
        if digest(answers[qid]["formatted_context"]) != row["context_sha256"]:
            raise ValueError(f"Actual reader context mismatch: {qid}")
        if entry["reuse"] == "none":
            if "judge_requests" not in entry:
                raise ValueError(f"New judge response identity was not recorded: {qid}")
            expected_calls = 0 if row["task_type"] == "ER" else 1
            if len(entry["judge_requests"]) != expected_calls:
                raise ValueError(f"Unexpected judge request count: {qid}")
            for request in entry["judge_requests"]:
                if request["response_model"] != judge_model:
                    raise ValueError(f"Provider substituted judge model: {qid}")
                judge_models[request["response_model"]] += 1
        if row["context_changed"]:
            request = entry["request"]
            if request["response_model"] != READER_RESPONSE_MODEL:
                raise ValueError(f"Provider substituted reader model: {qid}")
            checks.append(request["usage"]["prompt_tokens"] - row["prompt_tokens_estimate"])
            if request["request_sha256"] != row["request_sha256"]:
                raise ValueError(f"Actual request hash mismatch: {qid}")
            if request["finish_reason"] == "content_filter":
                refusals.append(qid)
                if entry["judgment"].get("evaluation_status") != "provider_refusal_scored_incorrect":
                    raise ValueError("Filtered response incorrectly scored as normal QA")
            elif request["finish_reason"] != "stop":
                raise ValueError(f"Truncated completion: {qid}")
            models[request["response_model"]] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += request["usage"][key]
    if any(checks):
        raise ValueError(f"Server prompt usage mismatch: {Counter(checks)}")
    return {"new_reader_questions": len(checks), "server_prompt_tokens_exact_match": True,
            "response_models": dict(models), "reader_usage": dict(usage),
            "new_judge_response_models": dict(judge_models),
            "provider_refusal_question_ids": refusals,
            "invalidated_model_responses": len(load(run / "invalidated_model_responses.json"))
            if (run / "invalidated_model_responses.json").exists() else 0,
            "reuse": dict(Counter(r["reuse"] for r in checkpoint.values()))}


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}"


def effect(value):
    if value is None:
        return "-"
    return (f"{100 * value['difference']:+.2f} "
            f"[{100 * value['ci95_low']:+.2f}, {100 * value['ci95_high']:+.2f}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    source = {system: spec["source"] for system, spec in sources(args.source_manifest).items()}
    raw_run = destination("raw", args.tag)
    raw_rows = scores(load(raw_run / "eval_results.json")) if (
        "raw" in source and (raw_run / "eval_results.json").exists()
    ) else None
    report = {
        "protocol": "Frozen historical gold selection/filter/order/readout; only packing cap removed",
        "score": "Six-task after-phase macro; Del/Cas/Abs gated by matching before answer",
        "bootstrap": f"{args.samples:,} paired-episode percentile resamples, conditional on saved outputs",
        "systems": {},
    }
    all_rows = {}
    per_question = []
    for offset, system in enumerate(source):
        run = destination(system, args.tag)
        entry = {"run": str(run), "source": str(source[system])}
        report["systems"][system] = entry
        if not (run / "manifest.json").exists():
            entry["status"] = "blocked_reconstruction"
            path = run / "reconstruction_audit.json"
            if path.exists():
                failed = load(path)
                entry["mismatch_count"] = len(failed["mismatches"])
                entry["unresolved_provenance"] = len(
                    failed["store"].get("unresolved_archived_semantic_provenance", [])
                )
                entry["mismatch_reasons"] = dict(Counter(r["reason"] for r in failed["mismatches"]))
            continue
        manifest = load(run / "manifest.json")
        if file_digest(run / "search_results.json") != manifest["search_sha256"]:
            raise ValueError("Prepared input changed")
        audit = load(run / "packing_audit.json")
        prepared_audit = [r["retrieval_metadata"]["uncapped_audit"]
                          for r in stream(run / "search_results.json")]
        if audit != prepared_audit:
            raise ValueError("Packing audit differs from hashed prepared inputs")
        if file_digest(source[system] / "eval_results.json") != manifest["source_eval_sha256"]:
            raise ValueError("Historical scores changed since preparation")
        per_question.extend({"system": system, **row} for row in audit)
        entry["packing"] = packing(audit)
        entry["packing_by_phase_task"] = {}
        for phase in ("before", "after"):
            for task in TASKS:
                subset = [r for r in audit if r["phase"] == phase and r["task_type"] == task]
                if subset:
                    entry["packing_by_phase_task"][f"{phase}:{task}"] = packing(subset)
        if not (run / "eval_results.json").exists():
            entry["status"] = "qa_incomplete"
            continue
        old = scores(load(source[system] / "eval_results.json"))
        new = scores(load(run / "eval_results.json"))
        if len(new) != manifest["questions"] or {r["question_id"] for r in new} != {r["question_id"] for r in old}:
            raise ValueError("Incomplete paired result")
        entry["request_verification"] = verify_requests(run, audit)
        entry["status"] = "complete"
        entry["capped"] = summary(old)
        entry["uncapped"] = summary(new)
        for condition, path in (("capped", source[system]), ("uncapped", run)):
            if abs(entry[condition]["macro"] - load(path / "eval_results.json")["accuracy"]) > 1e-12:
                raise ValueError("Recomputed score disagrees with saved overall")
        entry["uncapped_minus_capped"] = paired(new, old, args.samples, 20260920 + offset)
        if raw_rows is not None:
            entry["raw_uncapped_minus_system"] = paired(raw_rows, new, args.samples, 20261020 + offset)
        affected = {r["question_id"] for r in audit if r["context_changed"]}
        # An unchanged after context can change score when its before gate changes.
        gate_keys = {(r["conversation_id"], r["entity_key"]) for r in old
                     if r["phase"] == "before" and r["question_id"] in affected}
        gated_affected = affected | {
            r["question_id"] for r in old
            if r["phase"] == "after" and r["task_type"] in {"Del", "Cas", "Abs"}
            and (r["conversation_id"], r["entity_key"]) in gate_keys
        }
        for name, ids in (("directly_affected", affected), ("affected_including_before_gate", gated_affected)):
            left = [r for r in new if r["question_id"] in ids]
            right = [r for r in old if r["question_id"] in ids]
            entry[name] = {"capped": summary(right), "uncapped": summary(left),
                           "effect": paired(left, right, args.samples, 20261120 + offset)}
        all_rows[system] = new
    report["all_eight_conditions_complete"] = len(all_rows) == len(SYSTEMS)
    report["all_requested_conditions_complete"] = len(all_rows) == len(source)
    # Exclude the same provider-failure questions from every completed condition,
    # including their before/after partner whenever official gating needs it.
    excluded = set()
    for name, rows in all_rows.items():
        refused = set(report["systems"][name]["request_verification"]["provider_refusal_question_ids"])
        keys = {(r["conversation_id"], r["entity_key"]) for r in rows
                if r["question_id"] in refused}
        excluded.update(refused)
        excluded.update(r["question_id"] for r in rows
                        if r["task_type"] in {"Del", "Cas", "Abs"}
                        and (r["conversation_id"], r["entity_key"]) in keys)
    report["common_successful_subset"] = {
        "conditions": list(all_rows), "excluded_ids": sorted(excluded),
        "all_requested_conditions_included": report["all_requested_conditions_complete"],
        "systems": {},
    }
    for offset, (name, rows) in enumerate(all_rows.items()):
        old = scores(load(source[name] / "eval_results.json"))
        left = [r for r in rows if r["question_id"] not in excluded]
        right = [r for r in old if r["question_id"] not in excluded]
        report["common_successful_subset"]["systems"][name] = {
            "capped": summary(right), "uncapped": summary(left),
            "effect": paired(left, right, args.samples, 20261220 + offset),
        }
    report["limitations"] = [
        "Budget removal does not isolate irreversible write-time information loss.",
        "Original system-specific selection and ordering are intentionally unchanged.",
        "Unchanged reader inputs reuse historical answers; identical answer strings reuse judgments.",
        "Before gates are recomputed on the full paired question set before subset analysis.",
        "Input-affected subset macros average only represented tasks; task counts are explicit.",
        "Historical mutable stores are not reconstructed historical snapshots except for EMem.",
        "Every retained new reader response reports gpt-4o-mini-2024-07-18; responses reporting a substituted model are archived, invalidated, and rerun with the original request.",
        "Original task-aware policy, including Abs state guidance, is retained for this intervention.",
        "Token loss measures serialized evidence, not semantic sufficiency.",
        "Provider content_filter outcomes, if any, are preserved as empty answers and scored incorrect, without a fabricated judge call or modified prompt.",
        "Usage totals count retained response records, not discarded model substitutions or earlier retry attempts; they are not billing totals.",
    ]
    lines = [
        "# Complete-Store QA: 5k vs No Packing Cap", "",
        "Paired before/after questions from the supplied source runs. Scores are percentages.",
        "Only the original packing budget changes. Raw control is also uncapped.",
        "Incomplete or non-reproducible systems are not assigned a score.", "",
        "## Results", "",
        "| Condition | Status | 5k | Uncapped | Change, pp [95% CI] | Raw minus uncapped, pp [95% CI] |",
        "|---|---|---:|---:|---|---|",
    ]
    for name, entry in report["systems"].items():
        if entry["status"] == "complete":
            lines.append(f"| {name} | complete | {percent(entry['capped']['macro'])} | "
                         f"{percent(entry['uncapped']['macro'])} | {effect(entry['uncapped_minus_capped'])} | "
                         f"{effect(entry.get('raw_uncapped_minus_system'))} |")
        else:
            lines.append(f"| {name} | {entry['status']} | - | - | - | - |")
    lines.extend([
        "", "## Packing Audit", "",
        "| Condition | Affected / total | Prepack median / P90 / max | 5k mean | Uncapped mean | Dropped tokens | Skipped / clipped objects | Window failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, entry in report["systems"].items():
        if "packing" not in entry:
            continue
        p = entry["packing"]
        length = p["prepack_tokens"]
        lines.append(
            f"| {name} | {p['affected']} / {p['questions']} | {length['median']:.0f} / {length['p90']:.0f} / {length['max']} | "
            f"{p['capped_context_tokens']['mean']:.1f} | {p['uncapped_context_tokens']['mean']:.1f} | "
            f"{percent(p['micro_token_drop_fraction'])}% | {p['skipped_objects']} / {p['clipped_objects']} | {p['over_window']} |"
        )
    lines.extend(["", "## Task Scores", "",
                  "| Condition | Budget | ER | Agg | Tr | Del | Cas | Abs | Non-Abs | Before |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, entry in report["systems"].items():
        if entry["status"] != "complete":
            continue
        for budget in ("capped", "uncapped"):
            s = entry[budget]
            cells = [percent(s["task_accuracy"][task]["accuracy"]) for task in TASKS]
            lines.append(f"| {name} | {budget} | " + " | ".join(
                cells + [percent(s["non_abs_macro"]), percent(s["before_accuracy"])]) + " |")
    lines.extend(["", "## Affected Subsets", "",
                  "Gating is recomputed globally before restricting rows. These subset macros are not full six-task scores when some tasks have no affected questions.", "",
                  "| Condition | Direct affected after N | Tasks | Change [95% CI] | Including changed before gate N | Change [95% CI] |",
                  "|---|---:|---|---|---:|---|"])
    for name, entry in report["systems"].items():
        if entry["status"] != "complete":
            continue
        direct, gated = entry["directly_affected"], entry["affected_including_before_gate"]
        lines.append(
            f"| {name} | {direct['uncapped']['after_questions']} | "
            f"{', '.join(direct['uncapped']['macro_tasks']) or '-'} | {effect(direct['effect'])} | "
            f"{gated['uncapped']['after_questions']} | {effect(gated['effect'])} |"
        )
    lines.extend(["", "## Verification And Limits", ""])
    lines.extend(f"- {value}" for value in report["limitations"])
    lines.append("- Every new completed reader request must match the prepared hash and server-reported prompt token count exactly; no input truncation is permitted.")
    for name, entry in report["systems"].items():
        if entry.get("request_verification", {}).get("provider_refusal_question_ids"):
            count = len(entry["request_verification"]["provider_refusal_question_ids"])
            lines.append(f"- {name}: {count} provider-filtered responses scored incorrect.")
        invalidated = entry.get("request_verification", {}).get("invalidated_model_responses", 0)
        if invalidated:
            lines.append(f"- {name}: {invalidated} provider model-substitution responses discarded and rerun.")
    if excluded:
        lines.extend(["", "## Common Successful Subset", "",
                      f"The same {len(excluded)} questions are excluded from every completed condition, including gated partners. This subset does not include blocked conditions.", "",
                      "| Condition | After N | 5k | Uncapped | Change [95% CI] |",
                      "|---|---:|---:|---:|---|"])
        for name, entry in report["common_successful_subset"]["systems"].items():
            lines.append(
                f"| {name} | {entry['uncapped']['after_questions']} | "
                f"{percent(entry['capped']['macro'])} | {percent(entry['uncapped']['macro'])} | "
                f"{effect(entry['effect'])} |"
            )
    for name, entry in report["systems"].items():
        if entry["status"] == "blocked_reconstruction":
            lines.append(f"- {name}: reconstruction blocked; "
                         f"{entry.get('unresolved_provenance', 'unknown')} missing archived provenance records; "
                         f"{entry.get('mismatch_count', 'unknown')} reconstruction mismatches.")
    out = EVAL / "analysis" / args.tag
    save(out / "report.json", report)
    if per_question:
        with (out / "per_question.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
            writer.writeheader()
            writer.writerows(per_question)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:17]))
    print(f"\nReport: {out / 'report.md'}")


if __name__ == "__main__":
    main()
