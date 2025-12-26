from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from jigsaw_toxic_classification_ml.data.dvc_utils import dvc_pull
from jigsaw_toxic_classification_ml.data.preprocess import load_vocab
from jigsaw_toxic_classification_ml.models.toxic_classifier import (
    ToxicClassifierConfig,
    ToxicCommentClassifier,
)
from jigsaw_toxic_classification_ml.utils.paths import ensure_dir, resolve_path
from jigsaw_toxic_classification_ml.utils.seeding import seed_everything

LOGGER = logging.getLogger(__name__)


def export_onnx_model(cfg: DictConfig) -> dict[str, Any]:
    seed_everything(int(cfg.seed))

    ckpt_path = resolve_path(cfg.export.checkpoint_path)
    vocab_path = resolve_path(cfg.data.vocab_path)

    dvc_pull([ckpt_path, vocab_path])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run training first (or configure DVC remote)."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. Run preprocess first."
        )

    labels: list[str] = list(cfg.data.labels)
    vocab = load_vocab(vocab_path)

    model_cfg = ToxicClassifierConfig(
        vocab_size=len(vocab.id_to_token),
        num_labels=len(labels),
        embed_dim=int(cfg.model.encoder.embed_dim),
        hidden_dim=int(cfg.model.encoder.hidden_dim),
        num_layers=int(cfg.model.encoder.num_layers),
        dropout=float(cfg.model.encoder.dropout),
        bidirectional=bool(cfg.model.encoder.bidirectional),
        embedding_spatial_dropout=float(cfg.model.encoder.embedding_spatial_dropout),
        pooling=str(cfg.model.encoder.pooling),
        head_dropout=float(cfg.model.head.dropout),
    )
    net = ToxicCommentClassifier(model_cfg)
    _load_net_state_from_lightning_ckpt(net, ckpt_path)
    net.eval()

    onnx_path = resolve_path(cfg.export.onnx_path)
    ensure_dir(onnx_path.parent)

    max_len = int(cfg.preprocess.tokenize.max_length)
    dummy_ids = torch.zeros((1, max_len), dtype=torch.long)
    dummy_attn = torch.ones((1, max_len), dtype=torch.long)

    LOGGER.info("Exporting ONNX to %s", onnx_path)
    torch.onnx.export(
        net,
        (dummy_ids, dummy_attn),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=int(cfg.export.opset_version),
        # Use legacy exporter for maximum compatibility in offline/CI environments.
        # New dynamo exporter may require specific onnxscript versions.
        dynamo=False,
    )

    # Copy vocab next to ONNX for production use
    onnx_dir = ensure_dir(resolve_path(cfg.export.onnx_dir))
    (onnx_dir / "vocab.json").write_bytes(vocab_path.read_bytes())

    return {
        "status": "ok",
        "onnx_path": str(onnx_path),
        "vocab_path": str(onnx_dir / "vocab.json"),
    }


def _load_net_state_from_lightning_ckpt(net: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    # Our LightningModule stores under "model.*"
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model.") :]] = v
    missing, unexpected = net.load_state_dict(new_state, strict=False)
    if missing:
        LOGGER.warning("Missing keys in ONNX export load: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys in ONNX export load: %s", unexpected)
