from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def find_vec_in_dir(raw_dir: Path) -> Path | None:
    """
    Best-effort auto-detection of a fastText .vec file in a directory.
    """
    if not raw_dir.exists():
        return None
    vecs = sorted(raw_dir.glob("*.vec"))
    return vecs[0] if vecs else None


def load_fasttext_subset(
    vec_path: Path,
    token_to_id: dict[str, int],
    embed_dim: int,
    pad_id: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """
    Load only embeddings for words present in token_to_id.

    Supports fastText text .vec with optional header line: "<n> <dim>".
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=0.0, scale=0.02, size=(len(token_to_id), embed_dim)).astype(
        np.float32
    )
    if 0 <= pad_id < matrix.shape[0]:
        matrix[pad_id] = 0.0

    found = 0
    remaining = set(token_to_id.keys())
    # Never try to match special tokens
    remaining.discard("[PAD]")
    remaining.discard("[UNK]")

    with vec_path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        parts = first.rstrip().split(" ")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            # header line, continue
            pass
        else:
            _maybe_assign(parts, token_to_id, embed_dim, matrix, remaining)

        for line in f:
            parts = line.rstrip().split(" ")
            if _maybe_assign(parts, token_to_id, embed_dim, matrix, remaining):
                found += 1
                if not remaining:
                    break

    LOGGER.info(
        "Loaded fastText embeddings from %s: matched %d/%d tokens",
        vec_path,
        found,
        len(token_to_id),
    )
    return matrix


def _maybe_assign(
    parts: list[str],
    token_to_id: dict[str, int],
    embed_dim: int,
    matrix: np.ndarray,
    remaining: set[str],
) -> bool:
    if len(parts) < embed_dim + 1:
        return False
    word = parts[0]
    idx = token_to_id.get(word)
    if idx is None:
        return False
    try:
        vec = np.asarray(parts[1 : embed_dim + 1], dtype=np.float32)
    except ValueError:
        return False
    if vec.shape[0] != embed_dim:
        return False
    matrix[idx] = vec
    remaining.discard(word)
    return True
