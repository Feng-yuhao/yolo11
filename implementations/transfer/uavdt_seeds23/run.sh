#!/usr/bin/env bash
set -euo pipefail
pkg_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec env -u PYTHONPATH -u PYTHONHOME PYTHONUTF8=1 \
  /root/miniconda3/envs/yolo11/bin/python -u "$pkg_dir/controller.py" "$@"
