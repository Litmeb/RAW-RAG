from pathlib import Path

from evaluation.src.utils.config import load_yaml


PROMPTS_PATH = Path(__file__).resolve().parents[4] / "config" / "prompts.yaml"
PROMPTS = load_yaml(str(PROMPTS_PATH))

# Keep the legacy symbol name for callers, but source the official MEME prompt.
ANSWER_PROMPT = PROMPTS["online_api"]["default"]["answer_prompt_meme_official"]

