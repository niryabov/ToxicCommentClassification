from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass(frozen=True)
class ToxicClassifierConfig:
    vocab_size: int
    num_labels: int
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = True
    embedding_spatial_dropout: float = 0.0
    pooling: Literal["masked_mean", "avgmax"] = "avgmax"
    head_dropout: float = 0.2


class ToxicCommentClassifier(nn.Module):
    """
    Lightweight text encoder (Embedding + GRU) for multi-label classification.

    Forward signature is ONNX-friendly:
      - input_ids: LongTensor [B, T]
      - attention_mask: LongTensor [B, T] (1 for tokens, 0 for padding)
    Returns:
      - logits: FloatTensor [B, num_labels]
    """

    def __init__(self, cfg: ToxicClassifierConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout2d(cfg.embedding_spatial_dropout)
        self.encoder = nn.GRU(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        enc_out = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        if cfg.pooling == "avgmax":
            enc_out = enc_out * 2
        self.head = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(enc_out, cfg.num_labels),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, T, D]
        # SpatialDropout1D analogue: drop entire embedding channels across time.
        # Dropout2d expects [N, C, *], so treat embed_dim as channels.
        x = self.embedding_dropout(x.transpose(1, 2)).transpose(1, 2)

        out, _ = self.encoder(x)  # [B, T, H]
        mask = attention_mask.to(out.dtype).unsqueeze(-1)  # [B, T, 1]

        if self.cfg.pooling == "masked_mean":
            out = out * mask
            denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
            pooled = out.sum(dim=1) / denom  # [B, H]
            return self.head(pooled)

        # Kaggle reference-style pooling: avg + max pooling then concatenate.
        out_masked = out * mask
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]
        avg_pool = out_masked.sum(dim=1) / denom  # [B, H]

        # For max-pool, set padded positions to a large negative value.
        neg_inf = torch.tensor(-1e9, dtype=out.dtype, device=out.device)
        out_for_max = torch.where(mask.bool(), out, neg_inf)
        max_pool = out_for_max.max(dim=1).values  # [B, H]

        pooled = torch.cat([avg_pool, max_pool], dim=-1)  # [B, 2H]
        return self.head(pooled)  # [B, C]
