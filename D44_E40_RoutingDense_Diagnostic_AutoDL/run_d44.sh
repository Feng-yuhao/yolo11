#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${D44_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, or run}"
shift
case "$MODE" in check|smoke|run) ;; *) echo 'Use check, smoke, or run' >&2; exit 2 ;; esac
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec 9>"$PACKAGE/.d44_execution.lock"
flock -n 9 || { echo 'Another D44 controller is active; no duplicate diagnostic launched' >&2; exit 3; }
exec "$PYTHON" -u "$PACKAGE/diagnose_d44.py" --mode "$MODE" "$@"
