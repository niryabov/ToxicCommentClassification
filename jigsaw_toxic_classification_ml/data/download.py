from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from jigsaw_toxic_classification_ml.data.io import write_csv
from jigsaw_toxic_classification_ml.utils.paths import ensure_dir, resolve_path

LOGGER = logging.getLogger(__name__)


def download_data(cfg: DictConfig) -> dict[str, Any]:
    """
    Kaggle downloads typically require credentials; this function is intentionally
    "best-effort". If raw files already exist, we keep them. Otherwise, we create
    a small synthetic dataset for smoke tests.
    """
    raw_dir = resolve_path(cfg.data.raw_dir)
    ensure_dir(raw_dir)

    train_csv = resolve_path(cfg.data.train_csv)
    test_csv = resolve_path(cfg.data.test_csv)
    sample_sub_csv = resolve_path(cfg.data.sample_submission_csv)

    # Never overwrite user-provided real data. If at least train.csv exists,
    # we consider the raw dataset "present" and only create missing optional files.
    if train_csv.exists():
        if test_csv.exists() and sample_sub_csv.exists():
            LOGGER.info("Raw data already present at %s", raw_dir)
            return {"status": "ok", "raw_dir": str(raw_dir), "source": "existing"}

        LOGGER.warning(
            "Found existing train.csv but missing test/sample_submission. "
            "Generating missing files for convenience (without overwriting train.csv)."
        )
        if not test_csv.exists() or not sample_sub_csv.exists():
            _generate_synthetic_test_and_submission(
                test_csv=test_csv,
                sample_sub_csv=sample_sub_csv,
                labels=list(cfg.data.labels),
                n_test=int(cfg.preprocess.synthetic.n_test),
                seed=int(cfg.seed),
            )
        return {"status": "ok", "raw_dir": str(raw_dir), "source": "partial_existing"}

    LOGGER.warning(
        "Raw Kaggle files not found. Generating synthetic dataset at %s for smoke tests.",
        raw_dir,
    )
    _generate_synthetic_kaggle_like(
        train_csv=train_csv,
        test_csv=test_csv,
        sample_sub_csv=sample_sub_csv,
        labels=list(cfg.data.labels),
        n_train=int(cfg.preprocess.synthetic.n_train),
        n_test=int(cfg.preprocess.synthetic.n_test),
        seed=int(cfg.seed),
    )
    return {"status": "ok", "raw_dir": str(raw_dir), "source": "synthetic"}


def _generate_synthetic_test_and_submission(
    test_csv: Path,
    sample_sub_csv: Path,
    labels: list[str],
    n_test: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    toxic_words = ["idiot", "stupid", "hate", "kill", "moron", "trash"]
    neutral_words = ["please", "thanks", "discussion", "wikipedia", "source", "edit"]

    def make_text() -> str:
        n = int(rng.integers(5, 20))
        toks = []
        for _ in range(n):
            if rng.random() < 0.12:
                toks.append(rng.choice(toxic_words))
            else:
                toks.append(rng.choice(neutral_words))
        return " ".join(toks)

    if not test_csv.exists():
        test = pd.DataFrame(
            {
                "id": [f"test_{i}" for i in range(n_test)],
                "comment_text": [make_text() for _ in range(n_test)],
            }
        )
        write_csv(test, test_csv)

    if not sample_sub_csv.exists():
        test = pd.read_csv(test_csv)
        sample = pd.DataFrame({"id": test["id"]})
        for lab in labels:
            sample[lab] = 0.5
        write_csv(sample, sample_sub_csv)


def _generate_synthetic_kaggle_like(
    train_csv: Path,
    test_csv: Path,
    sample_sub_csv: Path,
    labels: list[str],
    n_train: int,
    n_test: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    toxic_words = ["idiot", "stupid", "hate", "kill", "moron", "trash"]
    neutral_words = ["please", "thanks", "discussion", "wikipedia", "source", "edit"]

    def make_text() -> str:
        n = int(rng.integers(5, 20))
        toks = []
        for _ in range(n):
            if rng.random() < 0.12:
                toks.append(rng.choice(toxic_words))
            else:
                toks.append(rng.choice(neutral_words))
        return " ".join(toks)

    train = pd.DataFrame(
        {
            "id": [f"train_{i}" for i in range(n_train)],
            "comment_text": [make_text() for _ in range(n_train)],
        }
    )
    # Create correlated multi-label targets (not perfect, but deterministic-ish)
    base = rng.random((n_train, len(labels)))
    # Increase probability if toxic words present
    toxic_signal = train["comment_text"].str.contains("|".join(toxic_words)).astype(float)
    base = 0.15 * base + 0.85 * toxic_signal.to_numpy()[:, None] * rng.random(
        (n_train, len(labels))
    )
    y = (base > 0.55).astype(int)
    for j, lab in enumerate(labels):
        train[lab] = y[:, j]

    test = pd.DataFrame(
        {
            "id": [f"test_{i}" for i in range(n_test)],
            "comment_text": [make_text() for _ in range(n_test)],
        }
    )
    sample = pd.DataFrame({"id": test["id"]})
    for lab in labels:
        sample[lab] = 0.5

    write_csv(train, train_csv)
    write_csv(test, test_csv)
    write_csv(sample, sample_sub_csv)


