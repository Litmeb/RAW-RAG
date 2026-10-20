from pathlib import Path

from evaluation.src.utils.config import load_yaml


def test_memu_local_disables_unbounded_category_summary_retrieval():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "config"
        / "systems"
        / "memu_local.yaml"
    )

    config = load_yaml(str(config_path))
    retrieve = config["memu"]["retrieve"]
    memorize = config["memu"]["memorize"]

    assert config["adapter"] == "memu_local"
    assert retrieve["category"]["enabled"] is False
    assert retrieve["item"]["enabled"] is True
    assert memorize["update_category_summaries"] is False
    assert config["search"]["embedding_batch_size"] == 8
    assert config["search"]["gpu_similarity"] is False
    assert config["search"]["item_similarity_backend"] == "torch_gpu"
