"""Guard resumable artifacts against changes in inputs or experiment settings."""

import json
import uuid
from pathlib import Path

from evaluation.src.utils.fingerprint import fingerprint


ANSWER_ARTIFACTS = (
    "answer_results.json", "answer_results_checkpoint.json",
    "responses_checkpoint_*.json", "eval_results.json", "report.txt",
    "judge_progress.json", "checkpoint_*.json", "run_identity.json",
    "evaluation_identity.json",
)
RUN_ARTIFACTS = (
    *ANSWER_ARTIFACTS, "search_results.json", "search_results_checkpoint.json",
    "answer_identity.json", "stage_timings.json",
)
EVALUATION_ARTIFACTS = (
    "eval_results.json", "report.txt", "checkpoint_*.json", "run_identity.json",
)


def prompt_fingerprint() -> str:
    path = Path(__file__).resolve().parents[2] / "config" / "prompts.yaml"
    return fingerprint(path.read_text(encoding="utf-8"))


def guard_artifacts(
    output_dir: Path, name: str, inputs, patterns, *, allow_reset: bool = True,
) -> bool:
    """Archive stale outputs before allowing a new identity to write results.

    Only digest values are saved, not raw settings, credentials or source text.
    Legacy outputs without an identity cannot be implicitly accepted as current.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / f"{name}_identity.json"
    identity = {"version": 1, "fingerprint": fingerprint(inputs)}
    previous = None
    if identity_path.exists():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
    if previous == identity:
        return False

    stale = {path for pattern in patterns for path in output_dir.glob(pattern) if path.is_file()}
    if stale and not allow_reset:
        raise ValueError(
            "Run inputs/configuration changed or legacy outputs have no run identity. "
            "Run all stages (including add) to invalidate stale results, or use a new output directory."
        )
    if identity_path.exists():
        stale.add(identity_path)
    if stale:
        archive = output_dir / "stale_artifacts" / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for path in sorted(stale):
            if path.parent.resolve() != output_dir:
                raise ValueError(f"Refusing to archive a path outside the run: {path}")
            path.replace(archive / path.name)
        print(f"Archived {len(stale)} stale run artifacts to {archive}")

    temporary = identity_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    temporary.replace(identity_path)
    return True
