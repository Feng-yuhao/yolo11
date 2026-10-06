#!/usr/bin/env bash
set -euo pipefail
PYTHON="${EXPERIMENT_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
PROJECT="${EXPERIMENT_PROJECT:-/root/autodl-tmp/yolo11_project}"
NAME="${1:?Use e44 or e45}"
if [[ "$NAME" != e44 && "$NAME" != e45 ]]; then echo 'Use e44 or e45' >&2; exit 2; fi
LOG="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["logs"][sys.argv[2]])' "$PROJECT/logs/e44e45_latest_run.json" "$NAME")"
echo "Viewing $LOG; Ctrl+C stops ONLY this viewer, not nohup training."
exec tail -n 40 -f "$LOG"
