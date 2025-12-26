from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

from jigsaw_toxic_classification_ml.data.preprocess import preprocess_data


def test_preprocess_synthetic(tmp_path: Path) -> None:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config.yaml",
            overrides=[
                f"data.raw_dir={tmp_path / 'raw'}",
                f"data.processed_dir={tmp_path / 'processed'}",
                f"data.artifacts_dir={tmp_path / 'artifacts'}",
            ],
        )
    out = preprocess_data(cfg)
    assert Path(out["train_path"]).exists()
    assert Path(out["val_path"]).exists()
    assert Path(out["vocab_path"]).exists()
