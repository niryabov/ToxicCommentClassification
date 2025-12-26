from __future__ import annotations

import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def dvc_pull(paths: list[Path]) -> bool:
    """
    Best-effort DVC pull for required artifacts.

    Returns True if dvc pull succeeded, False otherwise (caller should fallback).
    """
    tracked_paths = _filter_dvc_tracked_paths(paths)
    if not tracked_paths:
        LOGGER.info("No DVC-tracked paths requested; skipping dvc pull.")
        return False

    try:
        args = ["dvc", "pull"] + [str(p) for p in tracked_paths]
        LOGGER.info("Running: %s", " ".join(args))
        subprocess.check_call(args)
        return True
    except FileNotFoundError:
        LOGGER.warning("DVC is not installed in this environment.")
        return False
    except subprocess.CalledProcessError as e:
        LOGGER.warning(
            "DVC pull failed (exit %s). Falling back. Details: %s", e.returncode, e
        )
        return False


def _filter_dvc_tracked_paths(paths: list[Path]) -> list[Path]:
    """
    Only attempt `dvc pull` for paths that are actually tracked by DVC.

    - If a repo has a dvc.yaml pipeline, paths may be referenced there.
      (We don't parse dvc.yaml here; we just allow pull if it exists.)
    - For `dvc add`-tracked files, a `<file>.dvc` metafile exists next to the file.
    """
    repo_root = Path.cwd()
    has_dvc_yaml = (repo_root / "dvc.yaml").exists()

    tracked: list[Path] = []
    for p in paths:
        p = Path(p)
        # `dvc add data/raw/train.csv` creates `data/raw/train.csv.dvc`
        meta = Path(str(p) + ".dvc")
        if meta.exists() or has_dvc_yaml:
            tracked.append(p)
    return tracked
