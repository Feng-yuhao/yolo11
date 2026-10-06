#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${E40S3_E47S1_PAIR_PYTHON:-${EXPERIMENT_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}}"
[[ -x "$PYTHON" ]] || { echo "Missing existing yolo11 Python: $PYTHON" >&2; exit 3; }
export EXPERIMENT_PYTHON="$PYTHON"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec "$PYTHON" -u "$PACKAGE/run_pair.py" "$@"
