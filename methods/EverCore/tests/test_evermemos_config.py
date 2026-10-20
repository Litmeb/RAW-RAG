from pathlib import Path

from evaluation.src.utils.config import load_yaml


def test_evermemos_enables_profile_updates_by_default():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "config"
        / "systems"
        / "evermemos.yaml"
    )

    config = load_yaml(str(config_path))

    assert config["adapter"] == "evermemos"
    assert config["add"]["enable_profile_extraction"] is True
