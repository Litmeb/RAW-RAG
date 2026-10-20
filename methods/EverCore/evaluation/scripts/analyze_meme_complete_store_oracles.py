"""Summarize MEME complete-store oracles against native and raw controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_meme_continuity_experiment import (
    EVALUATION_DIR,
    RESULTS_DIR,
    TASKS,
    detailed_rows,
    paired_episode_bootstrap,
    scored_rows,
    summarize,
)


DEFAULT_OUTPUT = EVALUATION_DIR / "analysis/meme_continuity_complete_store_full_v1_20260717"
DEFAULT_RAW = RESULTS_DIR / "meme-raw-meme-continuity-gold-fields-task-readout-full-v1-20260717"
CONDITIONS = {
    "mem0_local": {
        "native": RESULTS_DIR / "meme-mem0_local-normal",
        "complete": RESULTS_DIR
        / "meme-mem0_local-meme-continuity-complete-store-task-readout-full-v1-20260717",
    },
    "memos_local": {
        "native": RESULTS_DIR / "meme-memos_local-fine",
        "complete": RESULTS_DIR
        / "meme-memos_local-meme-continuity-complete-store-memos-fine-task-readout-full-v1-20260818",
    },
    "memu_local": {
        "native": RESULTS_DIR / "meme-memu_local-normal",
        "complete": RESULTS_DIR
        / "meme-memu_local-meme-continuity-complete-store-memu-local-all-reps-task-readout-full-v3-20260818",
    },
    "emem": {
        "native": RESULTS_DIR / "meme-emem-normal",
        "complete": RESULTS_DIR
        / "meme-emem-meme-continuity-complete-store-task-readout-full-v1-20260717",
    },
    "evermemos": {
        "native": RESULTS_DIR / "meme-evermemos-normal-3",
        "complete": RESULTS_DIR
        / "meme-evermemos-meme-continuity-complete-store-task-readout-full-v1-20260717",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def audit_summary(run_dir: Path) -> dict[str, Any]:
    rows = load_json(run_dir / "meme_continuity_oracle_audit.json")
    store = load_json(run_dir / "complete_store_audit.json")
    pool_keys = (
        "phase_visible_candidate_pool_size",
        "deduplicated_candidate_pool_size",
    )
    pool_key = next((key for key in pool_keys if key in rows[0]), None)
    return {
        "questions": len(rows),
        "any_support": sum(row["any_support_value_observed"] for row in rows),
        "all_support": sum(row["all_support_values_observed"] for row in rows),
        "mean_context_tokens": sum(row["context_tokens"] for row in rows) / len(rows),
        "mean_selected": sum(row["selected_count"] for row in rows) / len(rows),
        "mean_candidate_pool": (
            sum(row[pool_key] for row in rows) / len(rows) if pool_key else None
        ),
        "empty_contexts": sum(row["selected_count"] == 0 for row in rows),
        "store": store,
    }


def task_cells(summary: dict[str, Any]) -> list[str]:
    return [pct(summary["task_accuracy"][task]["accuracy"]) for task in TASKS]


def add_non_abs_macro(summary: dict[str, Any]) -> None:
    summary["non_abs_macro_accuracy"] = sum(
        summary["task_accuracy"][task]["accuracy"] for task in TASKS if task != "Abs"
    ) / (len(TASKS) - 1)


def add_condition_override(
    conditions: dict[str, dict[str, Path]],
    name: str,
    native: Path | None,
    complete: Path | None,
) -> None:
    if (native is None) != (complete is None):
        raise ValueError(f"--{name}-native and --{name}-complete must be provided together")
    if native is not None and complete is not None:
        conditions[name] = {"native": native, "complete": complete}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--emem-complete",
        type=Path,
        help="Override the EMem complete-store run for a follow-up packing analysis.",
    )
    parser.add_argument("--memos-native", type=Path)
    parser.add_argument("--memos-complete", type=Path)
    parser.add_argument("--memu-native", type=Path)
    parser.add_argument("--memu-complete", type=Path)
    parser.add_argument(
        "--rag-native",
        type=Path,
        help="Optional native fixed-chunk RAG run to include in the comparison.",
    )
    parser.add_argument(
        "--rag-complete",
        type=Path,
        help="Optional complete fixed-chunk RAG oracle run to include in the comparison.",
    )
    parser.add_argument(
        "--memmachine-native",
        type=Path,
        help="Optional native MemMachine run to include in the comparison.",
    )
    parser.add_argument(
        "--memmachine-complete",
        type=Path,
        help="Optional complete-store MemMachine oracle run to include in the comparison.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--only",
        action="append",
        choices=tuple(CONDITIONS),
        help="Limit the report to selected built-in systems (repeatable).",
    )
    args = parser.parse_args()

    raw_dir = args.raw.resolve()
    raw_ids = {str(row["question_id"]) for row in detailed_rows(raw_dir)}
    raw_rows = scored_rows(raw_dir, raw_ids)
    raw = summarize(raw_rows)
    add_non_abs_macro(raw)
    raw["run_dir"] = str(raw_dir)

    conditions = {
        name: dict(paths)
        for name, paths in CONDITIONS.items()
        if args.only is None or name in args.only
    }
    if args.emem_complete is not None:
        conditions["emem"]["complete"] = args.emem_complete
    add_condition_override(
        conditions, "memos_local", args.memos_native, args.memos_complete
    )
    add_condition_override(
        conditions, "memu_local", args.memu_native, args.memu_complete
    )
    if (args.rag_native is None) != (args.rag_complete is None):
        raise ValueError("--rag-native and --rag-complete must be provided together")
    if args.rag_native is not None and args.rag_complete is not None:
        conditions = {
            "rag": {"native": args.rag_native, "complete": args.rag_complete},
            **conditions,
        }
    if (args.memmachine_native is None) != (args.memmachine_complete is None):
        raise ValueError(
            "--memmachine-native and --memmachine-complete must be provided together"
        )
    if args.memmachine_native is not None and args.memmachine_complete is not None:
        conditions["memmachine"] = {
            "native": args.memmachine_native,
            "complete": args.memmachine_complete,
        }

    systems = {}
    for offset, (name, paths) in enumerate(conditions.items()):
        native_dir = paths["native"].resolve()
        complete_dir = paths["complete"].resolve()
        native_rows = scored_rows(native_dir, raw_ids)
        complete_rows = scored_rows(complete_dir, raw_ids)
        if len(native_rows) != len(raw_rows) or len(complete_rows) != len(raw_rows):
            raise ValueError(
                f"Question alignment failed for {name}: native={len(native_rows)}, "
                f"complete={len(complete_rows)}, raw={len(raw_rows)}"
            )
        native = summarize(native_rows)
        complete = summarize(complete_rows)
        add_non_abs_macro(native)
        add_non_abs_macro(complete)
        native["run_dir"] = str(native_dir)
        complete["run_dir"] = str(complete_dir)
        complete["audit"] = audit_summary(complete_dir)
        systems[name] = {
            "native": native,
            "complete_store": complete,
            "effects": {
                "complete_minus_native": paired_episode_bootstrap(
                    complete_rows,
                    native_rows,
                    samples=args.bootstrap_samples,
                    seed=args.seed + offset,
                ),
                "raw_minus_complete": paired_episode_bootstrap(
                    raw_rows,
                    complete_rows,
                    samples=args.bootstrap_samples,
                    seed=args.seed + 100 + offset,
                ),
            },
        }

    report = {
        "experiment": "MEME phase-safe complete-store representation oracle",
        "questions": len(raw_rows),
        "episodes": 100,
        "raw": raw,
        "systems": systems,
        "method": {
            "selection": "MEME gold fields with deterministic exact-value/query-overlap ranking",
            "token_budget": 5000,
            "readout": "task-aware policy without entity values or gold answer",
            "bootstrap": f"{args.bootstrap_samples} paired episode resamples",
            "important_backend_note": (
                "Mem0, Memos, and MemU use local stores and must be compared to their local native runs; "
                "they are not pure candidate-pool replacements for the earlier cloud runs."
            ),
        },
    }
    output = args.output_prefix.resolve()
    dump_json(output.with_suffix(".json"), report)

    lines = [
        "# MEME Complete-Store Continuity Oracle",
        "",
        (
            "Scope: 100 `meme_filler32k` episodes / 1188 questions. Each oracle can "
            "select from every phase-visible persisted memory representation using MEME gold "
            "fields, but the reader receives only memory text plus the value-free task policy."
        ),
        "",
        "## Main results",
        "",
        "| System | Native | Complete store | Complete−Native | Raw−Complete | Native/Complete/Raw Non-Abs | Any/all support | Mean ctx |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in systems.items():
        native_acc = values["native"]["overall_macro_accuracy"]
        complete_acc = values["complete_store"]["overall_macro_accuracy"]
        audit = values["complete_store"]["audit"]
        lines.append(
            f"| {name} | {pct(native_acc)} | **{pct(complete_acc)}** | "
            f"{pp(complete_acc - native_acc)} | "
            f"{pp(raw['overall_macro_accuracy'] - complete_acc)} | "
            f"{pct(values['native']['non_abs_macro_accuracy'])}/"
            f"{pct(values['complete_store']['non_abs_macro_accuracy'])}/"
            f"{pct(raw['non_abs_macro_accuracy'])} | "
            f"{audit['any_support']}/{audit['all_support']} of 1188 | "
            f"{audit['mean_context_tokens']:.1f} |"
        )
    lines.extend(
        [
            f"| raw gold evidence | — | **{pct(raw['overall_macro_accuracy'])}** | — | — | —/—/{pct(raw['non_abs_macro_accuracy'])} | official evidence | — |",
            "",
            "Memos and MemU use their exact local backend stores and matching local native runs. They must not be compared as if they were candidate-pool-only replacements for unrelated cloud or namespace runs.",
            "",
            "## Task accuracy",
            "",
            "| Condition | ER | Agg | Tr | Del | Cas | Abs | Overall | Non-Abs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, values in systems.items():
        lines.append(
            f"| {name} native | " + " | ".join(task_cells(values["native"]))
            + f" | {pct(values['native']['overall_macro_accuracy'])}"
            + f" | {pct(values['native']['non_abs_macro_accuracy'])} |"
        )
        lines.append(
            f"| {name} complete | "
            + " | ".join(task_cells(values["complete_store"]))
            + f" | {pct(values['complete_store']['overall_macro_accuracy'])}"
            + f" | {pct(values['complete_store']['non_abs_macro_accuracy'])} |"
        )
    lines.append(
        "| raw gold evidence | " + " | ".join(task_cells(raw))
        + f" | {pct(raw['overall_macro_accuracy'])}"
        + f" | {pct(raw['non_abs_macro_accuracy'])} |"
    )
    lines.extend(
        [
            "",
            "## Paired episode bootstrap",
            "",
            "| System | Complete−native (95% CI) | Raw−complete (95% CI) |",
            "|---|---:|---:|",
        ]
    )
    for name, values in systems.items():
        left = values["effects"]["complete_minus_native"]
        right = values["effects"]["raw_minus_complete"]
        lines.append(
            f"| {name} | {pp(left['difference'])} "
            f"[{pp(left['ci95_low'])}, {pp(left['ci95_high'])}] | "
            f"{pp(right['difference'])} "
            f"[{pp(right['ci95_low'])}, {pp(right['ci95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Complete-store access is a representation/readout upper bound under the fixed 5000-token packing policy, not an unlimited-context mathematical oracle.",
            "- Raw evidence remains the continuity-preserving control. A positive, non-zero-crossing Raw−complete interval supports residual information/relationship loss after eliminating native retrieval coverage as the candidate boundary.",
            "- Source metadata (`metadata.sources`, `original_data`, or MemMachine citation episodes) is used only for phase filtering; raw provenance content is never added to extracted-memory contexts.",
            "- Abs remains affected by the benchmark's ignorance advantage: missing evidence can produce a rewarded `uncertain`, while recovered stale values without the invalidating relation can reduce accuracy.",
            "",
            f"Machine-readable report: `{output.with_suffix('.json')}`.",
            "",
        ]
    )
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(output.with_suffix(".md"))
    print(output.with_suffix(".json"))


if __name__ == "__main__":
    main()
