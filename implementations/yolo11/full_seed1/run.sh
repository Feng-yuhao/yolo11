#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${E60_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="$PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u "$PACKAGE/controller.py" "$@"
