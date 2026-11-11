#!/usr/bin/env bash
set -euo pipefail
tood_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export YOLO_AUTOINSTALL=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset PYTHONPATH PYTHONHOME
exec /root/miniconda3/envs/yolo11/bin/python -u "$tood_pkg/controller.py" "$@"
