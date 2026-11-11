#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec /root/miniconda3/envs/yolo11/bin/python -u "$script_dir/controller.py" "$@"
