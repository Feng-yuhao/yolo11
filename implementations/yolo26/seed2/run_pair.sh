#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export YOLO_AUTOINSTALL=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
exec /root/miniconda3/envs/yolo26/bin/python -u "$script_dir/controller.py" "$@"
