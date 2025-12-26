from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import fire
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from jigsaw_toxic_classification_ml.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def _normalize_overrides(overrides: list[str] | str | None) -> list[str]:
    if overrides is None:
        return []
    if isinstance(overrides, list):
        return [str(x) for x in overrides]
    # allow passing JSON via Fire: --overrides='["a=b","c=d"]'
    s = str(overrides).strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    # fallback: single override string
    return [s]


def _compose_cfg(overrides: list[str] | str | None = None) -> DictConfig:
    config_dir = (Path(__file__).resolve().parents[1] / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config.yaml", overrides=_normalize_overrides(overrides)
        )
    return cfg


def _pretty_cfg(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


class Commands:
    def download_data(self, overrides: list[str] | str | None = None) -> dict[str, Any]:
        cfg = _compose_cfg(overrides)
        configure_logging()
        LOGGER.info("Resolved config:\n%s", _pretty_cfg(cfg))
        from jigsaw_toxic_classification_ml.data.download import download_data

        return download_data(cfg)

    def preprocess(self, overrides: list[str] | str | None = None) -> dict[str, Any]:
        cfg = _compose_cfg(overrides)
        configure_logging()
        LOGGER.info("Resolved config:\n%s", _pretty_cfg(cfg))
        from jigsaw_toxic_classification_ml.data.preprocess import preprocess_data

        return preprocess_data(cfg)

    def train(self, overrides: list[str] | str | None = None) -> dict[str, Any]:
        cfg = _compose_cfg(overrides)
        configure_logging()
        LOGGER.info("Resolved config:\n%s", _pretty_cfg(cfg))
        from jigsaw_toxic_classification_ml.training.train import train_model

        return train_model(cfg)

    def export_onnx(self, overrides: list[str] | str | None = None) -> dict[str, Any]:
        cfg = _compose_cfg(overrides)
        configure_logging()
        LOGGER.info("Resolved config:\n%s", _pretty_cfg(cfg))
        from jigsaw_toxic_classification_ml.production.export_onnx import (
            export_onnx_model,
        )

        return export_onnx_model(cfg)

    def infer(
        self,
        text: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
        overrides: list[str] | str | None = None,
    ) -> dict[str, Any]:
        cfg = _compose_cfg(overrides)
        configure_logging()
        LOGGER.info("Resolved config:\n%s", _pretty_cfg(cfg))
        from jigsaw_toxic_classification_ml.production.infer_onnx import infer_onnx

        return infer_onnx(
            cfg=cfg,
            text=text,
            input_path=Path(input_path) if input_path else None,
            output_path=Path(output_path) if output_path else None,
        )


def main() -> None:
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
