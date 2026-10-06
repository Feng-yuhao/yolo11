#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${TESTDEV_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, verify-gates, run, or score}"
shift
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
# Dedicated evaluation lock only. Never acquires or changes a training lock.
exec 9>"$PACKAGE/.testdev_evaluation.lock"
flock -n 9 || { echo 'Another test-dev evaluator is active; no duplicate launched' >&2; exit 3; }
case "$MODE" in
  check|smoke|verify-gates|run|score)
    exec "$PYTHON" -u "$PACKAGE/testdev.py" --mode "$MODE" "$@" ;;
  *) echo 'Use check, smoke, verify-gates, run, or score' >&2; exit 2 ;;
esac
