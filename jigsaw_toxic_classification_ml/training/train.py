from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from jigsaw_toxic_classification_ml.data.dvc_utils import dvc_pull
from jigsaw_toxic_classification_ml.data.embeddings import (
    find_vec_in_dir,
    load_fasttext_subset,
)
from jigsaw_toxic_classification_ml.data.preprocess import load_vocab, preprocess_data
from jigsaw_toxic_classification_ml.models.baseline_tfidf import (
    TfidfBaseline,
    TfidfBaselineConfig,
)
from jigsaw_toxic_classification_ml.models.toxic_classifier import (
    ToxicClassifierConfig,
    ToxicCommentClassifier,
)
from jigsaw_toxic_classification_ml.training.datamodule import ToxicDataModule
from jigsaw_toxic_classification_ml.utils.logging import (
    end_run,
    log_figure,
    plot_auc_bar,
    plot_label_prevalence,
    plot_loss_curves,
    start_run,
)
from jigsaw_toxic_classification_ml.utils.paths import ensure_dir, resolve_path
from jigsaw_toxic_classification_ml.utils.seeding import seed_everything

LOGGER = logging.getLogger(__name__)


class LightningToxicModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_labels: int,
        labels: list[str],
        lr: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.labels = labels

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.auroc = MultilabelAUROC(num_labels=num_labels, average=None)
        self.f1 = MultilabelF1Score(
            num_labels=num_labels, average="macro", threshold=0.5
        )

        self._train_losses: list[float] = []
        self._val_losses: list[float] = []

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        loss = self.trainer.callback_metrics.get("train_loss_epoch")
        if loss is not None:
            self._train_losses.append(float(loss.detach().cpu().item()))

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        probs = torch.sigmoid(logits)
        self.auroc.update(probs, batch["labels"].int())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_macro_f1",
            self.f1(probs, batch["labels"].int()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self._val_losses.append(float(loss.detach().cpu().item()))

        auc = self.auroc.compute().detach().cpu().numpy()
        self.auroc.reset()
        aucs = {}
        for i, v in enumerate(auc):
            lab = self.labels[i] if i < len(self.labels) else f"label_{i}"
            val = float(v)
            aucs[lab] = val
            self.log(f"val_auc_{lab}", val, on_epoch=True, prog_bar=False)
        mean_auc = float(np.nanmean(list(aucs.values()))) if aucs else float("nan")
        # Kaggle competition metric: mean column-wise ROC-AUC
        self.log("val_mean_columnwise_auc", mean_auc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )
        return opt


def train_model(cfg: DictConfig) -> dict[str, Any]:
    seed_everything(int(cfg.seed))

    # Best-effort DVC pull for processed files / vocab; if missing, preprocess
    processed = [
        Path(str(cfg.data.train_processed_path)),
        Path(str(cfg.data.val_processed_path)),
        Path(str(cfg.data.vocab_path)),
    ]
    dvc_pull(processed)
    if (
        not resolve_path(cfg.data.vocab_path).exists()
        or not resolve_path(cfg.data.train_processed_path).exists()
    ):
        preprocess_data(cfg)

    artifacts_dir = resolve_path(cfg.data.artifacts_dir)
    ckpt_dir = ensure_dir(artifacts_dir / "checkpoints")

    labels: list[str] = list(cfg.data.labels)
    vocab = load_vocab(resolve_path(cfg.data.vocab_path))

    # --- Optional baseline (TF-IDF + linear) ---
    start_run(cfg)
    baseline_summary: dict[str, Any] | None = None
    try:
        if bool(cfg.train.baseline.enabled):
            baseline_summary = _train_baseline_tfidf(cfg, labels)
            LOGGER.info("Baseline TF-IDF summary: %s", baseline_summary)
    finally:
        # keep the run open for neural training below if it happens; end at the end
        pass

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
    _maybe_load_pretrained_embeddings(cfg, vocab, net)

    dm = ToxicDataModule(cfg)
    dm.setup()

    module = LightningToxicModule(
        model=net,
        num_labels=len(labels),
        labels=labels,
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=str(cfg.train.checkpoint.monitor),
        mode=str(cfg.train.checkpoint.mode),
        save_top_k=int(cfg.train.checkpoint.save_top_k),
        filename="best",
    )

    mlflow.log_params(
        {
            "batch_size": int(cfg.train.batch_size),
            "max_epochs": int(cfg.train.max_epochs),
            "lr": float(cfg.train.learning_rate),
            "weight_decay": float(cfg.train.weight_decay),
            **{f"model.{k}": v for k, v in asdict(model_cfg).items()},
        }
    )

    try:
        trainer_kwargs = dict(cfg.train.trainer)
        trainer = pl.Trainer(
            max_epochs=int(cfg.train.max_epochs),
            callbacks=[ckpt_cb],
            **trainer_kwargs,
            limit_train_batches=(
                int(cfg.train.smoke.limit_train_batches)
                if bool(cfg.train.smoke.enabled)
                else 1.0
            ),
            limit_val_batches=(
                int(cfg.train.smoke.limit_val_batches)
                if bool(cfg.train.smoke.enabled)
                else 1.0
            ),
        )
        trainer.fit(
            module,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

        best_path = ckpt_cb.best_model_path
        if best_path:
            (ckpt_dir / "best.ckpt").write_bytes(Path(best_path).read_bytes())

        # --- Plots (must be >=3) ---
        local_plots = Path(str(cfg.logging.local_plots_dir))
        # label prevalence (from train)
        import pandas as pd

        train_df = pd.read_parquet(resolve_path(cfg.data.train_processed_path))
        fig1 = plot_label_prevalence(labels, train_df[labels].to_numpy())
        log_figure(fig1, "label_prevalence.png", local_plots)

        # AUC bar (from last epoch metrics)
        aucs = {}
        for lab in labels:
            v = trainer.callback_metrics.get(f"val_auc_{lab}")
            if v is not None:
                aucs[lab] = float(v.detach().cpu().item())
        fig2 = plot_auc_bar(labels, aucs)
        log_figure(fig2, "val_auc_bar.png", local_plots)

        # loss curves
        fig3 = plot_loss_curves(module._train_losses, module._val_losses)
        log_figure(fig3, "loss_curves.png", local_plots)

        # log key metrics explicitly
        mlflow.log_metrics(
            {
                "val_macro_f1": float(
                    trainer.callback_metrics["val_macro_f1"].detach().cpu().item()
                ),
                "val_loss": float(
                    trainer.callback_metrics["val_loss"].detach().cpu().item()
                ),
                "val_mean_columnwise_auc": (
                    float(np.nanmean(list(aucs.values()))) if aucs else float("nan")
                ),
            }
        )

        return {
            "status": "ok",
            "best_checkpoint": str(ckpt_dir / "best.ckpt"),
            "baseline": baseline_summary,
        }
    finally:
        end_run()


def _maybe_load_pretrained_embeddings(cfg: DictConfig, vocab: Any, net: nn.Module) -> None:
    """
    If enabled in config, initialize embedding weights from a fastText .vec file.
    """
    try:
        enabled = bool(cfg.model.encoder.pretrained.enabled)
    except Exception:
        enabled = False
    if not enabled:
        return

    embed_dim = int(cfg.model.encoder.embed_dim)
    vec_path_cfg = cfg.model.encoder.pretrained.path

    candidates: list[Path] = []
    if vec_path_cfg not in (None, "null"):
        candidates.append(resolve_path(str(vec_path_cfg)))

    # Auto-detect: repo root + data/raw
    project_root = resolve_path(str(cfg.data.project_root))
    candidates.append(project_root / "crawl-300d-2M.vec")
    if (p := find_vec_in_dir(resolve_path(cfg.data.raw_dir))) is not None:
        candidates.append(p)

    vec_path = next((p for p in candidates if p.exists()), None)
    if vec_path is None:
        raise FileNotFoundError(
            "Pretrained embeddings enabled, but no .vec found. "
            "Set model.encoder.pretrained.path or place a *.vec in repo root or data/raw."
        )

    matrix = load_fasttext_subset(
        vec_path=vec_path,
        token_to_id=vocab.token_to_id,
        embed_dim=embed_dim,
        pad_id=vocab.pad_id,
        seed=int(cfg.seed),
    )

    emb: nn.Embedding = getattr(net, "embedding")
    if emb.weight.shape != torch.Size(matrix.shape):
        raise ValueError(
            f"Embedding shape mismatch: model has {tuple(emb.weight.shape)}, "
            f"but fastText matrix is {matrix.shape}. "
            "Check model.encoder.embed_dim and vocab size."
        )

    emb.weight.data.copy_(torch.from_numpy(matrix))
    freeze = bool(cfg.model.encoder.pretrained.freeze)
    emb.weight.requires_grad_(not freeze)

    mlflow.log_params(
        {
            "pretrained.enabled": True,
            "pretrained.path": str(vec_path),
            "pretrained.freeze": freeze,
        }
    )
    LOGGER.info("Initialized embedding from fastText: %s (freeze=%s)", vec_path, freeze)


def _train_baseline_tfidf(cfg: DictConfig, labels: list[str]) -> dict[str, Any]:
    import pandas as pd

    train_df = pd.read_parquet(resolve_path(cfg.data.train_processed_path))
    val_df = pd.read_parquet(resolve_path(cfg.data.val_processed_path))

    max_tr = cfg.train.baseline.max_train_samples
    max_va = cfg.train.baseline.max_val_samples
    if max_tr is not None:
        train_df = train_df.head(int(max_tr))
    if max_va is not None:
        val_df = val_df.head(int(max_va))

    x_train = train_df["clean_text"].astype(str).tolist()
    y_train = train_df[labels].astype(np.int32).to_numpy()
    x_val = val_df["clean_text"].astype(str).tolist()
    y_val = val_df[labels].astype(np.int32).to_numpy()

    baseline_cfg = TfidfBaselineConfig(
        max_features=int(cfg.train.baseline.max_features),
        ngram_range=(
            int(cfg.train.baseline.ngram_min),
            int(cfg.train.baseline.ngram_max),
        ),
        C=float(cfg.train.baseline.C),
    )
    baseline = TfidfBaseline(baseline_cfg)

    mlflow.log_params(
        {
            "baseline.enabled": True,
            "baseline.max_features": baseline_cfg.max_features,
            "baseline.ngram_min": baseline_cfg.ngram_range[0],
            "baseline.ngram_max": baseline_cfg.ngram_range[1],
            "baseline.C": baseline_cfg.C,
            "baseline.max_train_samples": None if max_tr is None else int(max_tr),
            "baseline.max_val_samples": None if max_va is None else int(max_va),
        }
    )

    baseline.fit(x_train, y_train)
    prob = baseline.predict_proba(x_val)  # [N, C]

    # AUC per label (handle degenerate single-class columns)
    aucs: dict[str, float] = {}
    for j, lab in enumerate(labels):
        try:
            aucs[lab] = float(roc_auc_score(y_val[:, j], prob[:, j]))
        except ValueError:
            aucs[lab] = float("nan")

    pred = (prob >= 0.5).astype(np.int32)
    macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))

    mlflow.log_metrics(
        {
            "baseline_val_macro_f1": macro_f1,
            "baseline_val_mean_columnwise_auc": float(np.nanmean(list(aucs.values()))),
        }
    )
    for lab, v in aucs.items():
        mlflow.log_metric(f"baseline_val_auc_{lab}", v)

    return {
        "val_macro_f1": macro_f1,
        "val_mean_columnwise_auc": float(np.nanmean(list(aucs.values()))),
        "val_auc_per_label": aucs,
    }
