#!/usr/bin/env bash
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
unset PYTHONPATH PYTHONHOME
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export YOLO_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec /root/miniconda3/envs/yolo11/bin/python -I -u \
  "$package_dir/controller.py" all --enable-shutdown "$@"
