#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/root/miniconda3/envs/yolo11/bin/python
MODE="${1:-eval}"
if (( $# > 0 )); then shift; fi
REPORT=/root/autodl-tmp/yolo11_project/comparison_reports/E39_RUN_20260929_125730_777892
if [[ "$MODE" != eval && "$MODE" != all && "$MODE" != inspect ]]; then
  echo 'Usage: bash run_recovery.sh {inspect|eval|all} [recovery arguments]' >&2
  exit 2
fi
if [[ "$MODE" == all ]]; then
  # Preserve all prior CUDA gates: no edits to E40 code or gate receipts.
  "$PYTHON" -u /root/autodl-tmp/E40a_SpatialTaskHead_AutoDL/run_e40.py --mode verify-gates
fi
if [[ "$MODE" == inspect ]]; then
  exec "$PYTHON" -u "$HERE/recover_e39.py" --run-report "$REPORT" --inspect "$@"
fi
"$PYTHON" -u "$HERE/recover_e39.py" --run-report "$REPORT" "$@"
if [[ "$MODE" == all ]]; then
  echo 'E39 recovery succeeded. Checking GPU release and original E40 gates before continuation.'
  exec "$PYTHON" -u "$HERE/queue_after_e39.py" --e39-report "$REPORT"
fi
