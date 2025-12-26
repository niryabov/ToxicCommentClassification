from __future__ import annotations

import logging
import os
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
        return _dvc_pull_python_api(tracked_paths)
    except FileNotFoundError:
        LOGGER.warning("DVC is not installed in this environment.")
        return False
    except ImportError as e:
        LOGGER.warning("DVC Python API unavailable (%s). Falling back to CLI.", e)
        return _dvc_pull_cli(tracked_paths)
    except subprocess.CalledProcessError as e:
        LOGGER.warning(
            "DVC pull failed (exit %s). Falling back. Details: %s", e.returncode, e
        )
        return False
    except Exception as e:
        LOGGER.warning("DVC pull failed (%s). Falling back to CLI.", e)
        try:
            return _dvc_pull_cli(tracked_paths)
        except Exception:
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


def _dvc_pull_python_api(paths: list[Path]) -> bool:
    """
    Prefer DVC Python API to avoid shelling out.
    """
    from dvc.repo import Repo  # type: ignore

    repo = Repo(os.getcwd())
    targets = [str(p) for p in paths]
    LOGGER.info("DVC (python api) pull targets: %s", targets)
    repo.pull(targets=targets)
    return True


def _dvc_pull_cli(paths: list[Path]) -> bool:
    args = ["dvc", "pull"] + [str(p) for p in paths]
    LOGGER.info("Running: %s", " ".join(args))
    subprocess.check_call(args)
    return True
