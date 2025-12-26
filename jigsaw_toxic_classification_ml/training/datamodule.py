from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from jigsaw_toxic_classification_ml.data.io import read_parquet
from jigsaw_toxic_classification_ml.data.preprocess import (
    Vocab,
    encode_batch,
    load_vocab,
)
from jigsaw_toxic_classification_ml.utils.paths import resolve_path


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ToxicDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        vocab: Vocab,
        labels: list[str],
        max_length: int,
    ) -> None:
        self.texts = df["clean_text"].astype(str).tolist()
        self.labels = labels
        self.y = df[labels].astype(np.float32).to_numpy()
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x_ids, x_attn = encode_batch([self.texts[idx]], self.vocab, self.max_length)
        return {
            "input_ids": torch.from_numpy(x_ids[0]),
            "attention_mask": torch.from_numpy(x_attn[0]),
            "labels": torch.from_numpy(self.y[idx]),
        }


class ToxicDataModule:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.labels: list[str] = list(cfg.data.labels)

        self.train_path = resolve_path(cfg.data.train_processed_path)
        self.val_path = resolve_path(cfg.data.val_processed_path)
        self.vocab_path = resolve_path(cfg.data.vocab_path)
        self.max_length = int(cfg.preprocess.tokenize.max_length)

        self.batch_size = int(cfg.train.batch_size)
        self.num_workers = int(cfg.train.num_workers)
        self.max_train_samples = cfg.train.data.max_train_samples
        self.max_val_samples = cfg.train.data.max_val_samples

        self.vocab: Vocab | None = None
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def setup(self) -> None:
        vocab = load_vocab(self.vocab_path)
        train_df = read_parquet(self.train_path)
        val_df = read_parquet(self.val_path)
        if self.max_train_samples is not None:
            train_df = train_df.head(int(self.max_train_samples))
        if self.max_val_samples is not None:
            val_df = val_df.head(int(self.max_val_samples))
        self.vocab = vocab
        self.train_ds = ToxicDataset(train_df, vocab, self.labels, self.max_length)
        self.val_ds = ToxicDataset(val_df, vocab, self.labels, self.max_length)

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
