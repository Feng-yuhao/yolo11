#!/usr/bin/env bash
set -euo pipefail
src="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target=/root/autodl-tmp/yolo11_project/experiment_code/E0_Seeds23_Parallel_E60_Testdev
[[ "$(realpath /root/autodl-tmp/yolo11_project)" == /root/autodl-tmp/yolo11_project ]] || exit 1
if [[ "$src" == "$target" ]]; then
    printf 'Already installed: %s\n' "$target"
    exit 0
fi
[[ ! -e "$target" ]] || { printf 'Refusing to overwrite existing package: %s\n' "$target"; exit 1; }
mkdir -p -- "$(dirname -- "$target")"
cp -a -- "$src" "$target"
printf 'Installed real code files: %s\n' "$target"
printf 'No training started; old reports/logs/weights unchanged.\n'
