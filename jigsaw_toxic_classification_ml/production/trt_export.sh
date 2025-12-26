#!/usr/bin/env bash
set -euo pipefail

# Optional: TensorRT export from ONNX using trtexec.
# Requires TensorRT installed and `trtexec` available on PATH.
#
# Usage:
#   ./jigsaw_toxic_classification_ml/production/trt_export.sh artifacts/onnx/toxic_classifier.onnx artifacts/onnx/toxic_classifier.plan
#

ONNX_PATH="${1:-}"
PLAN_PATH="${2:-}"

if [[ -z "${ONNX_PATH}" || -z "${PLAN_PATH}" ]]; then
  echo "Usage: $0 <onnx_path> <plan_path>"
  exit 1
fi

trtexec --onnx="${ONNX_PATH}" --saveEngine="${PLAN_PATH}" --fp16
