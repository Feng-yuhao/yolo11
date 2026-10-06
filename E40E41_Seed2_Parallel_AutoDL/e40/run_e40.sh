#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${E40_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, run, eval, or verify-gates}"
shift
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [[ "$MODE" == queue || "$MODE" == inspect ]]; then
  echo 'Seed2 uses the run_pair.sh controller; old seed1 queue is disabled' >&2
  exit 2
fi
exec 9>"$PACKAGE/.e40_execution.lock"
flock -n 9 || { echo 'Another E40 check/smoke/run is active; no second job launched' >&2; exit 3; }
case "$MODE" in
  cpu-check) exec "$PYTHON" -u "$PACKAGE/run_e40.py" --mode check --device cpu "$@" ;;
  check|smoke|run|eval|verify-gates|local-integration) exec "$PYTHON" -u "$PACKAGE/run_e40.py" --mode "$MODE" "$@" ;;
  *) echo 'Use check, smoke, run, eval, or verify-gates' >&2; exit 2 ;;
esac
