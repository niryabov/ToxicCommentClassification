#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "[self_check] Repo root: ${REPO_ROOT}"

if ! command -v python >/dev/null 2>&1; then
  echo "python is required"
  exit 1
fi

if ! command -v poetry >/dev/null 2>&1; then
  echo "poetry is required. Install from https://python-poetry.org/docs/#installation"
  exit 1
fi

poetry --version

echo "[self_check] Installing deps (with train+dev groups)"
poetry install --with train,dev --no-interaction

echo "[self_check] Running pre-commit"
poetry run pre-commit install
poetry run pre-commit run -a

echo "[self_check] Running pytest"
poetry run pytest

echo "[self_check] Running preprocess + train (smoke mode)"
poetry run python -m jigsaw_toxic_classification_ml.commands preprocess
poetry run python -m jigsaw_toxic_classification_ml.commands train --overrides='["train.smoke.enabled=true","train.max_epochs=1"]'

echo "[self_check] Exporting ONNX"
poetry run python -m jigsaw_toxic_classification_ml.commands export_onnx

echo "[self_check] Running ONNX inference"
poetry run python -m jigsaw_toxic_classification_ml.commands infer --text "you are so stupid and I hate this"

echo "[self_check] OK"
