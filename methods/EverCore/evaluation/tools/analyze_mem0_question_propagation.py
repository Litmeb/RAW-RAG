"""Audit MEME propagation for the Mem0 Local run and persisted Qdrant store."""

from evaluation.tools.meme_question_propagation_common import run


if __name__ == "__main__":
    raise SystemExit(run("mem0_local"))
