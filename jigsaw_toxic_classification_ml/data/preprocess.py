from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from jigsaw_toxic_classification_ml.data.download import download_data
from jigsaw_toxic_classification_ml.data.io import read_csv, write_parquet
from jigsaw_toxic_classification_ml.utils.paths import ensure_dir, resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str
    unk_token: str

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]


def preprocess_data(cfg: DictConfig) -> dict[str, Any]:
    download_data(cfg)  # ensure raw exists (or synthetic)

    raw_train = resolve_path(cfg.data.train_csv)
    df = read_csv(raw_train)

    text_col = str(cfg.preprocess.text_column)
    labels: list[str] = list(cfg.data.labels)
    for col in [text_col, "id", *labels]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {raw_train}")

    df[text_col] = df[text_col].fillna("").astype(str)
    df["clean_text"] = df[text_col].map(lambda s: clean_text(s, cfg))

    train_df, val_df = train_test_split(
        df,
        test_size=float(cfg.preprocess.split.val_size),
        random_state=int(cfg.preprocess.split.random_state),
        stratify=None,
    )

    processed_dir = resolve_path(cfg.data.processed_dir)
    ensure_dir(processed_dir)

    train_out = resolve_path(cfg.data.train_processed_path)
    val_out = resolve_path(cfg.data.val_processed_path)
    vocab_out = resolve_path(cfg.data.vocab_path)

    vocab = build_vocab(
        train_df["clean_text"].tolist(),
        max_size=int(cfg.preprocess.vocab.max_size),
        min_freq=int(cfg.preprocess.vocab.min_freq),
        pad_token=str(cfg.preprocess.vocab.pad_token),
        unk_token=str(cfg.preprocess.vocab.unk_token),
    )
    save_vocab(vocab, vocab_out)

    keep_cols = ["id", "clean_text", *labels]
    write_parquet(train_df[keep_cols].reset_index(drop=True), train_out)
    write_parquet(val_df[keep_cols].reset_index(drop=True), val_out)

    # Kaggle test set may not include labels; we still preprocess it if present
    test_csv = resolve_path(cfg.data.test_csv)
    if test_csv.exists():
        test_df = read_csv(test_csv)
        if text_col not in test_df.columns:
            raise ValueError(f"Expected column '{text_col}' in {test_csv}")
        test_df[text_col] = test_df[text_col].fillna("").astype(str)
        test_df["clean_text"] = test_df[text_col].map(lambda s: clean_text(s, cfg))
        test_out = resolve_path(cfg.data.test_processed_path)
        write_parquet(test_df[["id", "clean_text"]].reset_index(drop=True), test_out)

    return {
        "status": "ok",
        "train_path": str(train_out),
        "val_path": str(val_out),
        "vocab_path": str(vocab_out),
    }


def clean_text(text: str, cfg: DictConfig) -> str:
    s = text
    if bool(cfg.preprocess.clean.lowercase):
        s = s.lower()
    if bool(cfg.preprocess.clean.remove_punctuation):
        s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    if bool(cfg.preprocess.clean.collapse_whitespace):
        s = re.sub(r"\s+", " ", s)
    if bool(cfg.preprocess.clean.strip):
        s = s.strip()
    return s


def simple_tokenize(text: str) -> list[str]:
    return [t for t in text.split(" ") if t]


def build_vocab(
    texts: list[str],
    max_size: int,
    min_freq: int,
    pad_token: str,
    unk_token: str,
) -> Vocab:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # Reserve IDs: 0 = PAD, 1 = UNK
    id_to_token = [pad_token, unk_token]
    token_to_id = {pad_token: 0, unk_token: 1}

    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in token_to_id:
            continue
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)
        if len(id_to_token) >= max_size:
            break

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_token=pad_token, unk_token=unk_token)


def save_vocab(vocab: Vocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "token_to_id": vocab.token_to_id,
        "id_to_token": vocab.id_to_token,
        "pad_token": vocab.pad_token,
        "unk_token": vocab.unk_token,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_vocab(path: Path) -> Vocab:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return Vocab(
        token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
        id_to_token=[str(x) for x in payload["id_to_token"]],
        pad_token=str(payload["pad_token"]),
        unk_token=str(payload["unk_token"]),
    )


def encode_batch(
    texts: list[str],
    vocab: Vocab,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_ids = np.full((len(texts), max_length), vocab.pad_id, dtype=np.int64)
    attn = np.zeros((len(texts), max_length), dtype=np.int64)
    for i, t in enumerate(texts):
        toks = simple_tokenize(t)
        ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in toks][:max_length]
        input_ids[i, : len(ids)] = np.asarray(ids, dtype=np.int64)
        attn[i, : len(ids)] = 1
    return input_ids, attn


