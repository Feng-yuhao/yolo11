#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${D47_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, run, all, cache, fit, eval or verify-gates}"
shift
export PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec 9>"$PACKAGE/.d47.lock"
flock -n 9 || { echo 'D47 already running; duplicate blocked'; exit 3; }
case "$MODE" in
  all)
    "$PYTHON" -u "$PACKAGE/run_d47.py" check "$@"
    "$PYTHON" -u "$PACKAGE/run_d47.py" smoke "$@"
    "$PYTHON" -u "$PACKAGE/run_d47.py" run "$@" ;;
  check|smoke|run|cache|fit|eval|verify-gates|self-test)
    "$PYTHON" -u "$PACKAGE/run_d47.py" "$MODE" "$@" ;;
  *) echo 'Unknown mode'; exit 2 ;;
esac
