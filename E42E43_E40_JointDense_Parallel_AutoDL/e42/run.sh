#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${EXPERIMENT_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, run, eval, verify-gates, cpu-check, local-integration}"
shift
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec 9>"$PACKAGE/.e42_execution.lock"
flock -n 9 || { echo 'This variant is already active; refusing second launch' >&2; exit 3; }
if [[ "$MODE" == cpu-check ]]; then
  exec "$PYTHON" -u "$PACKAGE/run_e40.py" --mode check --device cpu "$@"
fi
exec "$PYTHON" -u "$PACKAGE/run_e40.py" --mode "$MODE" "$@"
