#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${E58E59_PYTHON:-/root/miniconda3/envs/yolo11/bin/python}"
NAME="${1:?Use e58 or e59}"
MODE="${2:-run}"
PROJECT="${3:-/root/autodl-tmp/yolo11_project}"
LOG="$("$PYTHON" -c 'import json,sys,pathlib; n,m,p=sys.argv[1:]; assert n in ("e58","e59") and m in ("run","smoke"); print(json.loads((pathlib.Path(p)/"logs"/("latest_e58_e59_"+m+".json")).read_text())["logs"][n])' "$NAME" "$MODE" "$PROJECT")"
echo "$LOG"
exec tail -n 40 -f "$LOG"
