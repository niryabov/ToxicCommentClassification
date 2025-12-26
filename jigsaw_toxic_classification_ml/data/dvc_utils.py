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
    try:
        args = ["dvc", "pull"] + [str(p) for p in paths]
        LOGGER.info("Running: %s", " ".join(args))
        subprocess.check_call(args)
        return True
    except FileNotFoundError:
        LOGGER.warning("DVC is not installed in this environment.")
        return False
    except subprocess.CalledProcessError as e:
        LOGGER.warning("DVC pull failed (exit %s). Falling back. Details: %s", e.returncode, e)
        return False


