from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


@dataclass(frozen=True)
class TfidfBaselineConfig:
    max_features: int = 50000
    ngram_range: tuple[int, int] = (1, 2)
    C: float = 2.0


class TfidfBaseline:
    """Optional baseline: TF-IDF + OneVsRest Logistic Regression."""

    def __init__(self, cfg: TfidfBaselineConfig) -> None:
        self.cfg = cfg
        self.vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            ngram_range=cfg.ngram_range,
            strip_accents="unicode",
            lowercase=True,
        )
        base = LogisticRegression(max_iter=200, C=cfg.C, solver="liblinear")
        self.clf = OneVsRestClassifier(base)

    def fit(self, texts: list[str], y: np.ndarray) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, y)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)
