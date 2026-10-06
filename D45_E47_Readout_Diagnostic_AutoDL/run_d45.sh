#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${D45_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
  echo "Usage: bash run_d45.sh {offline|check|smoke|run|all} [--seed 1|2 ...]" >&2
  exit 2
fi
shift
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec "$PYTHON" -u "$HERE/diagnose_d45.py" --mode "$MODE" "$@"
