#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "$0")" && pwd)"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /root/miniconda3/envs/yolo11/bin/python -u "$PACKAGE/run_pair.py" "$@"
