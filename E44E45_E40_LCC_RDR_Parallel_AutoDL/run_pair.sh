#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${EXPERIMENT_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u "$PACKAGE/run_pair.py" "$@"
