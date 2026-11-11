#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf "%s\\n" "YOLO26 SEED3: check -> both smoke -> baseline200 -> E60200 -> four official scores -> save -> AUTO SHUTDOWN"
exec bash "$script_dir/run_pair.sh" all --auto-shutdown "$@"
