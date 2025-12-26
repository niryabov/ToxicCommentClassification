## Jigsaw Toxic Comment Classification (MLOps-grade repo)

This repository is a **professional, reproducible ML project** based on Kaggle’s _Jigsaw Toxic Comment Classification Challenge_ (multi-label text classification).

### Problem

Given a Wikipedia talk-page comment (`comment_text`), predict **six toxicity probabilities** (multi-label):

- **toxic**
- **severe_toxic**
- **obscene**
- **threat**
- **insult**
- **identity_hate**

### Metric notes

The original competition uses **mean column-wise ROC-AUC**. This repo logs:

- **ROC-AUC per label** (+ mean AUC)
- **macro F1** (thresholded at 0.5)

### Repository highlights (grader-critical)

- **Poetry** for dependencies (`pyproject.toml`, `poetry.lock`)
- **Pre-commit**: black, isort, flake8, prettier
- **Hydra configs** for all params/paths (no hardcoded constants)
- **DVC** for data/artifacts (no data committed to git)
- **PyTorch Lightning** for training
- **MLflow** for logging (default tracking URI: `http://127.0.0.1:8080`)
- **ONNX export + ONNXRuntime inference** (inference does **not** require torch)
- **Single CLI** via python-fire + Hydra compose API
- **Plots**: saves to `plots/` and logs to MLflow (>= 3 plots)

---

## Setup

### 1) Install Poetry + deps

If you install Poetry with `pip3 --user`, make sure the user bin directory is on your `PATH`:

```bash
export PATH="$HOME/Library/Python/3.9/bin:$PATH"
```

```bash
cd jigsaw_toxic_classification_ml
poetry install --with train,dev
```

### 2) Install pre-commit hooks

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

### 3) (Optional) MLflow UI/server

This repo assumes MLflow is reachable at **`http://127.0.0.1:8080`** (configurable via Hydra or `MLFLOW_TRACKING_URI`).

---

## Data (DVC + fallback)

Expected Kaggle files (not committed to git):

- `data/raw/train.csv`
- `data/raw/test.csv`
- `data/raw/sample_submission.csv`

Commands will **first attempt `dvc pull`**, and if data is still missing they will fall back to **generating a small synthetic dataset** (smoke-test friendly; no internet required).

---

## CLI usage

All commands are exposed via:

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands <command>
```

Hydra overrides are passed via `--overrides='["a=b","c=d"]'`.

### Download / generate data

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands download_data
```

### Preprocess (clean + split + vocab)

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands preprocess
```

### Train (Lightning + MLflow)

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands train
```

Smoke run:

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands train --overrides='["train.smoke.enabled=true","train.max_epochs=1"]'
```

### Export to ONNX

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands export_onnx
```

Outputs:

- `artifacts/onnx/toxic_classifier.onnx`
- `artifacts/onnx/vocab.json`

### ONNXRuntime inference (no torch)

Single text:

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands infer --text "some comment..."
```

Example output:

```json
{
  "predictions": {
    "toxic": 0.92,
    "severe_toxic": 0.04,
    "obscene": 0.69,
    "threat": 0.02,
    "insult": 0.33,
    "identity_hate": 0.01
  }
}
```

Batch CSV:

```bash
poetry run python -m jigsaw_toxic_classification_ml.commands infer --input_path input.csv --output_path preds.csv
```

`input.csv` must contain `comment_text`.

---

## MLflow serving (pyfunc)

The file `jigsaw_toxic_classification_ml/production/mlflow_pyfunc.py` contains a pyfunc wrapper around the ONNX model and vocab.

To serve a logged pyfunc model:

```bash
mlflow models serve -m "runs:/<RUN_ID>/onnx_pyfunc" -p 5001
```

---

## TensorRT (optional)

If you have TensorRT installed:

```bash
./jigsaw_toxic_classification_ml/production/trt_export.sh artifacts/onnx/toxic_classifier.onnx artifacts/onnx/toxic_classifier.plan
```

---

## Self-check

```bash
./scripts/self_check.sh
```
