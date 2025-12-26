from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def cfg_path(cfg: DictConfig, dotted: str) -> Path:
    cur = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return resolve_path(str(cur))
