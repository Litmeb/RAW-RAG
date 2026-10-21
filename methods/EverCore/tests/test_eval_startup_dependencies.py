"""Keep the standalone evaluation install sufficient for CLI startup."""

from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_declares_sqlalchemy_asyncio_extra():
    requirements = [
        Requirement(line.strip())
        for line in (ROOT / "requirements-eval.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert any(req.name.lower() == "sqlalchemy" and "asyncio" in req.extras
               for req in requirements)


def test_cli_help_starts_in_a_fresh_process():
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.cli", "--help"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dataset" in result.stdout
