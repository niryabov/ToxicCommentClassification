from __future__ import annotations

import json
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import onnxruntime as ort
import pandas as pd

from jigsaw_toxic_classification_ml.data.preprocess import Vocab, encode_batch
from jigsaw_toxic_classification_ml.data.preprocess import load_vocab as _load_vocab


class OnnxToxicPyfuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for ONNX toxic classifier.

    Expects input DataFrame with a column "comment_text".
    Outputs DataFrame with 6 columns for toxicity labels.
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        onnx_path = Path(context.artifacts["onnx_model"]).resolve()
        vocab_path = Path(context.artifacts["vocab"]).resolve()
        labels_path = Path(context.artifacts["labels"]).resolve()
        meta_path = Path(context.artifacts["meta"]).resolve()

        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.vocab: Vocab = _load_vocab(vocab_path)
        self.labels: list[str] = json.loads(labels_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.max_length: int = int(meta.get("max_length", 200))

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> pd.DataFrame:
        if "comment_text" not in model_input.columns:
            raise ValueError("Expected column 'comment_text'")
        texts = model_input["comment_text"].fillna("").astype(str).tolist()
        ids, attn = encode_batch(texts, self.vocab, self.max_length)
        logits = self.sess.run(
            ["logits"],
            {
                "input_ids": ids.astype(np.int64),
                "attention_mask": attn.astype(np.int64),
            },
        )[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return pd.DataFrame(probs, columns=self.labels)


def log_pyfunc_model(
    artifact_path: str,
    onnx_model_path: Path,
    vocab_path: Path,
    labels: list[str],
    max_length: int,
) -> None:
    labels_path = onnx_model_path.parent / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    meta_path = onnx_model_path.parent / "meta.json"
    meta_path.write_text(
        json.dumps({"max_length": int(max_length)}, indent=2) + "\n", encoding="utf-8"
    )

    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=OnnxToxicPyfuncModel(),
        artifacts={
            "onnx_model": str(onnx_model_path),
            "vocab": str(vocab_path),
            "labels": str(labels_path),
            "meta": str(meta_path),
        },
    )
