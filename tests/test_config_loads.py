from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_config_loads() -> None:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config.yaml", overrides=[])
    assert "data" in cfg
    assert "train" in cfg
