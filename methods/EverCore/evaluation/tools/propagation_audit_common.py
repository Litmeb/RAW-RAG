"""Shared helpers for post-hoc question-level memory propagation audits."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


SUPPORT_LABELS = {"sufficient", "partial", "none", "contradictory", "unclear"}


def truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    half = max(1, (limit - 40) // 2)
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
    raise ValueError("No valid JSON object found")


def flatten_rows(value: Any, *, container_keys: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Flatten common list/dict result envelopes without duplicating row objects."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        return []
    for key in container_keys:
        rows = value.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for child in value.values():
        if not isinstance(child, list):
            continue
        for row in child:
            if isinstance(row, dict) and id(row) not in seen:
                seen.add(id(row))
                rows.append(row)
    return rows


def conversation_index(value: Any) -> int:
    match = re.search(r"(\d+)$", str(value or ""))
    if not match:
        raise ValueError(f"Cannot parse conversation index from {value!r}")
    return int(match.group(1))


def row_correct(eval_row: dict[str, Any], answer_row: dict[str, Any]) -> bool:
    metric = eval_row.get("metric_value")
    if metric is not None:
        try:
            return float(metric) > 0
        except (TypeError, ValueError):
            pass
    for row in (eval_row, answer_row):
        for key in ("is_correct", "correct"):
            if key in row:
                return bool(row[key])
    return False


def stratified_question_sample(
    rows: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    import random

    if sample_size <= 0:
        return []
    if len(rows) <= sample_size:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[tuple[int, bool], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["conv"]), bool(row["correct"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for group in groups.values():
        quota = max(1, round(sample_size * len(group) / len(rows)))
        selected.extend(rng.sample(group, min(quota, len(group))))
    if len(selected) > sample_size:
        selected = rng.sample(selected, sample_size)
    selected_ids = {id(row) for row in selected}
    remaining = [row for row in rows if id(row) not in selected_ids]
    rng.shuffle(remaining)
    selected.extend(remaining[: sample_size - len(selected)])
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def normalize_judgment(
    parsed: dict[str, Any],
    *,
    allowed_stages: set[str],
    support_fields: Iterable[str],
    allowed_mechanisms: set[str] | None = None,
) -> dict[str, Any]:
    for key in support_fields:
        value = str(parsed.get(key) or "unclear")
        parsed[key] = value if value in SUPPORT_LABELS else "unclear"
    stage = str(parsed.get("error_stage") or "unclear")
    parsed["error_stage"] = stage if stage in allowed_stages else "unclear"
    mechanisms = parsed.get("mechanism_signals") or []
    if not isinstance(mechanisms, list):
        mechanisms = []
    if allowed_mechanisms is not None:
        mechanisms = [str(item) for item in mechanisms if str(item) in allowed_mechanisms]
    parsed["mechanism_signals"] = list(dict.fromkeys(mechanisms))
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    return parsed


async def run_judging(
    samples: list[dict[str, Any]],
    *,
    prompt_builder: Callable[[dict[str, Any]], str],
    normalizer: Callable[[dict[str, Any]], dict[str, Any]],
    output_path: Path,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    concurrency: int,
    retries: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    from memory_layer.llm.llm_provider import LLMProvider

    llm = LLMProvider(
        provider_type=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=max_tokens,
    )
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[dict[str, Any]] = []

    def normalize_for_sample(
        sample: dict[str, Any], parsed: dict[str, Any]
    ) -> dict[str, Any]:
        judgment = normalizer(parsed)
        if not sample.get("correct") and judgment.get("error_stage") == "no_error":
            judgment["error_stage"] = "unclear"
        return judgment

    async def judge(sample: dict[str, Any]) -> None:
        raw = ""
        last_error = ""
        async with semaphore:
            for _ in range(max(1, retries)):
                try:
                    raw = await llm.generate(
                        prompt=prompt_builder(sample),
                        temperature=0,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                    result = {
                        "sample_id": sample["sample_id"],
                        "category": "question_propagation",
                        "correct": bool(sample.get("correct")),
                        "ok": True,
                        "judgment": normalize_for_sample(
                            sample, extract_json_object(raw)
                        ),
                        "raw_response": raw,
                    }
                    break
                except Exception as exc:  # Preserve failures in the audit artifact.
                    last_error = f"{type(exc).__name__}: {exc}"
            else:
                result = {
                    "sample_id": sample["sample_id"],
                    "category": "question_propagation",
                    "correct": bool(sample.get("correct")),
                    "ok": False,
                    "error": last_error,
                    "judgment": normalize_for_sample(
                        sample, {"rationale": last_error}
                    ),
                    "raw_response": raw,
                }
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        results.append(result)
        if len(results) % 25 == 0 or len(results) == len(samples):
            print(f"Judged {len(results)}/{len(samples)} samples", flush=True)

    await asyncio.gather(*(judge(sample) for sample in samples))
    return sorted(results, key=lambda row: str(row["sample_id"]))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    margin = z * math.sqrt((rate * (1 - rate) + z * z / (4 * total)) / total)
    return {
        "low": (centre - margin) / denominator,
        "high": (centre + margin) / denominator,
    }


def summarize(
    results: list[dict[str, Any]], support_fields: Iterable[str]
) -> dict[str, Any]:
    stages = Counter(
        str(row.get("judgment", {}).get("error_stage", "unclear")) for row in results
    )
    by_outcome: dict[str, Any] = {}
    for outcome, correct in (("correct", True), ("wrong", False)):
        rows = [row for row in results if bool(row.get("correct")) is correct]
        by_outcome[outcome] = {
            "total": len(rows),
            "error_stage_counts": dict(
                Counter(
                    str(row.get("judgment", {}).get("error_stage", "unclear"))
                    for row in rows
                )
            ),
        }
    mechanisms = Counter(
        signal
        for row in results
        for signal in row.get("judgment", {}).get("mechanism_signals", [])
    )
    return {
        "total": len(results),
        "parse_failures": sum(not row.get("ok") for row in results),
        "error_stage_counts": dict(stages),
        "support_counts": {
            key: dict(
                Counter(
                    str(row.get("judgment", {}).get(key, "unclear")) for row in results
                )
            )
            for key in support_fields
        },
        "mechanism_signal_counts": dict(mechanisms),
        "by_final_outcome": by_outcome,
    }
