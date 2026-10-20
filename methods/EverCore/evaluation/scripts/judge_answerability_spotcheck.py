"""LLM spot-check for evidence answerability proxy rows.

The full evidence analysis is deterministic. This script samples rows from
sample_metrics.csv, reloads the corresponding retrieved formatted_context, and
asks an OpenAI-compatible model whether the context is sufficient to answer the
question with the gold answer.
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import random
import re
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
DEFAULT_ANALYSIS_DIR = EVALUATION_DIR / "analysis" / "evidence_answerability"
DEFAULT_ENV_PATH = EVALUATION_DIR.parent / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_answer_index(manifest_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for manifest in manifest_rows:
        dataset = manifest["dataset"]
        system = manifest["system"]
        split = manifest["split"]
        source = manifest["source"]
        for run_dir_str in manifest.get("run_dirs", []):
            run_dir = Path(run_dir_str)
            answer_path = run_dir / "answer_results.json"
            if not answer_path.exists():
                continue
            answers = load_json(answer_path)
            if not isinstance(answers, list):
                continue
            for item in answers:
                if not isinstance(item, dict) or not item.get("question_id"):
                    continue
                key = (dataset, system, split, source, str(item["question_id"]))
                index[key] = item
    return index


def sample_rows(
    rows: list[dict[str, str]],
    per_bucket: int,
    seed: int,
    all_rows: bool = False,
) -> list[dict[str, str]]:
    if all_rows:
        return list(rows)
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["system"], row["bucket"])].append(row)

    sampled: list[dict[str, str]] = []
    for key in sorted(groups):
        group_rows = groups[key]
        rng.shuffle(group_rows)
        sampled.extend(group_rows[:per_bucket])
    return sampled


def context_excerpt(context: str, gold_values: str, max_chars: int) -> str:
    values = [value.strip() for value in gold_values.split("|") if value.strip()]
    snippets: list[str] = []
    for value in values:
        if len(value) < 3:
            continue
        match = re.search(re.escape(value), context, flags=re.I)
        if match:
            start = max(0, match.start() - 900)
            end = min(len(context), match.end() + 900)
            snippets.append(context[start:end])
        if sum(len(snippet) for snippet in snippets) >= max_chars:
            break

    if snippets:
        excerpt = "\n\n[...]\n\n".join(snippets)
        return excerpt[:max_chars]
    return context[:max_chars]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    answerable_match = re.search(r'"?answerable"?\s*:\s*(true|false)', text, flags=re.I)
    support_match = re.search(
        r'"?support"?\s*:\s*"?(sufficient|partial|none|contradictory)',
        text,
        flags=re.I,
    )
    if answerable_match:
        return {
            "answerable": answerable_match.group(1).lower() == "true",
            "support": support_match.group(1).lower() if support_match else "partial_json",
            "notes": text[:200],
        }
    return {"answerable": None, "support": "parse_error", "notes": text[:200]}


def judge_one(
    row: dict[str, str],
    item: dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    max_context_chars: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    question = str(item.get("question") or row.get("question") or "")
    context = str(item.get("formatted_context") or "")
    excerpt = context_excerpt(context, row.get("gold_values_preview", ""), max_context_chars)
    prompt = f"""You are judging retrieval answerability.

Return JSON only with:
{{"answerable": true/false, "support": "sufficient|partial|none|contradictory", "notes": "short reason"}}

Mark answerable=true only if the retrieved context contains enough information to derive the gold answer without relying on world knowledge or the model's prior memory. Exact wording is not required.

Question:
{question}

Gold answer:
{row.get('golden_answer', '')}

Retrieved context excerpt:
{excerpt}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict evidence answerability judge."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error = None
    last_content = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
            message = body["choices"][0]["message"]
            # Some OpenAI-compatible reasoning endpoints occasionally leave
            # ``content`` empty and place their visible response in
            # ``reasoning_content``. Treat that as a fallback rather than
            # discarding an otherwise usable judge response.
            content = message.get("content") or message.get("reasoning_content") or ""
            last_content = content
            parsed = extract_json(content)
            if isinstance(parsed.get("answerable"), bool):
                return {
                    **{key: row.get(key, "") for key in row},
                    "llm_model": model,
                    "llm_answerable": parsed.get("answerable"),
                    "llm_support": parsed.get("support"),
                    "llm_notes": parsed.get("notes"),
                }
            last_error = ValueError("model response did not contain a complete answerable judgment")
            time.sleep(1.5 * (attempt + 1))
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
            socket.timeout,
            TimeoutError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    if last_content:
        return {
            **{key: row.get(key, "") for key in row},
            "llm_model": model,
            "llm_answerable": "",
            "llm_support": "parse_error",
            "llm_notes": last_content[:200],
        }
    return {
        **{key: row.get(key, "") for key in row},
        "llm_model": model,
        "llm_answerable": "",
        "llm_support": "error",
        "llm_notes": str(last_error)[:200],
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["system"], row["bucket"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, system, bucket), group_rows in sorted(groups.items()):
        valid = [row for row in group_rows if str(row.get("llm_answerable")) in {"True", "False", "true", "false"}]
        answerable = [
            str(row.get("llm_answerable")).lower() == "true"
            for row in valid
        ]
        summary.append(
            {
                "dataset": dataset,
                "system": system,
                "bucket": bucket,
                "n": len(group_rows),
                "valid_n": len(valid),
                "llm_answerable_rate": sum(answerable) / len(answerable) if answerable else 0.0,
            }
        )
    return summary


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], model: str) -> None:
    lines = [
        "# LLM Answerability Spot Check",
        "",
        f"Model: `{model}`",
        "",
        "| Dataset | System | Proxy Bucket | N | Valid | LLM Answerable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {system} | {bucket} | {n} | {valid_n} | {rate:.1f}% |".format(
                dataset=row["dataset"],
                system=row["system"],
                bucket=row["bucket"],
                n=row["n"],
                valid_n=row["valid_n"],
                rate=100 * float(row["llm_answerable_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--sample-per-bucket", type=int, default=3)
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Judge every row in sample_metrics.csv instead of sampling per bucket.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid rows from an existing spot-check CSV and retry the rest.",
    )
    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help="Process at most this many unfinished rows (0 means all); useful for resumable batches.",
    )
    args = parser.parse_args()

    load_dotenv(args.env_path)
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if not base_url or not api_key:
        raise SystemExit("DEEPSEEK_BASE_URL/DEEPSEEK_API_KEY (or LLM_BASE_URL/LLM_API_KEY) is missing")

    rows = load_csv(args.analysis_dir / "sample_metrics.csv")
    sampled = sample_rows(rows, args.sample_per_bucket, args.seed, all_rows=args.all_rows)
    manifest = load_json(args.analysis_dir / "run_manifest.json")
    answer_index = build_answer_index(manifest)

    output_path = args.analysis_dir / "llm_answerability_spotcheck.csv"
    existing_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    if args.resume and output_path.exists():
        for existing in load_csv(output_path):
            if str(existing.get("llm_answerable", "")).lower() not in {"true", "false"}:
                continue
            key = (
                existing["dataset"],
                existing["system"],
                existing["split"],
                existing["source"],
                existing["question_id"],
            )
            existing_by_key[key] = existing

    tasks: list[tuple[dict[str, str], dict[str, Any]]] = []
    retained: list[dict[str, Any]] = []
    for row in sampled:
        key = (
            row["dataset"],
            row["system"],
            row["split"],
            row["source"],
            row["question_id"],
        )
        if key in existing_by_key:
            retained.append(existing_by_key[key])
            continue
        item = answer_index.get(key)
        if item:
            tasks.append((row, item))

    if args.max_new_samples > 0:
        tasks = tasks[:args.max_new_samples]

    results: list[dict[str, Any]] = retained
    completed_new = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                judge_one,
                row,
                item,
                args.model,
                base_url,
                api_key,
                args.max_context_chars,
                args.max_tokens,
                args.timeout,
            )
            for row, item in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
            completed_new += 1
            if completed_new % 25 == 0:
                write_csv(output_path, results, list(results[0].keys()))
                print(f"Judged {completed_new}/{len(tasks)} new samples", flush=True)

    result_fields = list(results[0].keys()) if results else []
    write_csv(output_path, results, result_fields)
    summary_rows = summarize(results)
    write_csv(
        args.analysis_dir / "llm_answerability_spotcheck_summary.csv",
        summary_rows,
        ["dataset", "system", "bucket", "n", "valid_n", "llm_answerable_rate"],
    )
    write_markdown(args.analysis_dir / "llm_answerability_spotcheck.md", summary_rows, args.model)

    print(f"Judged {len(results)} samples with {args.model}")
    print(f"Summary: {args.analysis_dir / 'llm_answerability_spotcheck.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
