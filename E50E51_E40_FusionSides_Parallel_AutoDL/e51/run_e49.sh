#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${EXPERIMENT_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:?Use check, smoke, verify-gates, run, eval, all, cpu-check, local-integration}"
shift
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
PROJECT="${EXPERIMENT_PROJECT:-/root/autodl-tmp/yolo11_project}"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[i]}" == --project ]]; then PROJECT="${ARGS[i+1]:?Missing project}"; fi
  if [[ "${ARGS[i]}" == --project=* ]]; then PROJECT="${ARGS[i]#--project=}"; fi
done
[[ -d "$PROJECT" ]] || { echo "Missing project: $PROJECT" >&2; exit 3; }
[[ -x "$PYTHON" ]] || { echo "Missing yolo11 Python: $PYTHON" >&2; exit 3; }
exec 9>"$PROJECT/.e51_execution.lock"
flock -n 9 || { echo 'E51 already active; refusing duplicate launch' >&2; exit 3; }
if [[ "$MODE" == all ]]; then
  for STAGE in check smoke run; do
    echo "E51 stage=$STAGE START $(date --iso-8601=seconds)"
    "$PYTHON" -u "$PACKAGE/run_e49.py" --mode "$STAGE" "$@"
  done
  echo "E51 ALL COMPLETED $(date --iso-8601=seconds); auto shutdown disabled"
elif [[ "$MODE" == cpu-check ]]; then
  exec "$PYTHON" -u "$PACKAGE/run_e49.py" --mode check --device cpu "$@"
else
  exec "$PYTHON" -u "$PACKAGE/run_e49.py" --mode "$MODE" "$@"
fi
