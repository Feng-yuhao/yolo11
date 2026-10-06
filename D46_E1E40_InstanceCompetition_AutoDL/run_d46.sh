#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${D46_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, verify-gates, run, or all}"
shift
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec 9>"$PACKAGE/.audit.lock"
flock -n 9 || { echo 'D46 already running; no duplicate launched'; exit 3; }
case "$MODE" in
  all)
    "$PYTHON" -u "$PACKAGE/audit.py" check "$@"
    "$PYTHON" -u "$PACKAGE/audit.py" smoke "$@"
    "$PYTHON" -u "$PACKAGE/audit.py" run "$@" ;;
  check|smoke|verify-gates|run|self-test)
    "$PYTHON" -u "$PACKAGE/audit.py" "$MODE" "$@" ;;
  *) echo 'Use check, smoke, verify-gates, run, or all'; exit 2 ;;
esac
