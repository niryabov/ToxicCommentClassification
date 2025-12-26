from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from omegaconf import DictConfig

from jigsaw_toxic_classification_ml.data.dvc_utils import dvc_pull
from jigsaw_toxic_classification_ml.data.preprocess import (
    clean_text,
    encode_batch,
    load_vocab,
)
from jigsaw_toxic_classification_ml.data.schema import PredictionResponse
from jigsaw_toxic_classification_ml.utils.paths import resolve_path

LOGGER = logging.getLogger(__name__)


def infer_onnx(
    cfg: DictConfig,
    text: str | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    labels: list[str] = list(cfg.data.labels)
    model_path = resolve_path(cfg.infer.onnx.model_path)
    vocab_path = resolve_path(cfg.infer.onnx.vocab_path)

    dvc_pull([model_path, vocab_path])
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}. Run export_onnx first."
        )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Vocab not found at {vocab_path}. Run preprocess first."
        )

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    vocab = load_vocab(vocab_path)
    max_len = int(cfg.infer.max_length)

    if text is not None:
        cleaned = clean_text(text, cfg)
        ids, attn = encode_batch([cleaned], vocab, max_len)
        probs = _predict_probs(sess, ids, attn)[0]
        pred_dict = {lab: float(probs[i]) for i, lab in enumerate(labels)}
        resp = PredictionResponse(predictions=pred_dict)
        out = resp.model_dump()
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        return out

    if input_path is None:
        raise ValueError("Provide either --text or --input_path")

    df = pd.read_csv(input_path)
    if "comment_text" not in df.columns:
        raise ValueError("Expected 'comment_text' column in input CSV")
    texts = [
        clean_text(s, cfg) for s in df["comment_text"].fillna("").astype(str).tolist()
    ]

    all_probs: list[list[float]] = []
    bs = int(cfg.infer.batch_size)
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        ids, attn = encode_batch(chunk, vocab, max_len)
        probs = _predict_probs(sess, ids, attn)
        all_probs.extend(probs.tolist())

    out_df = pd.DataFrame(all_probs, columns=labels)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        return {"status": "ok", "output_path": str(output_path)}
    return {"status": "ok", "predictions": out_df.to_dict(orient="records")}


def _predict_probs(
    sess: ort.InferenceSession, input_ids: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    logits = sess.run(
        ["logits"],
        {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        },
    )[0]
    return 1.0 / (1.0 + np.exp(-logits))
