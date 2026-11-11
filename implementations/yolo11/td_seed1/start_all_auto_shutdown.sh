#!/usr/bin/env bash
set -euo pipefail
tood_pkg="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
printf '%s\n' 'TOOD-TD: check -> real smoke -> verify -> 200e -> official val -> official test-dev -> save -> idle shutdown'
exec bash "$tood_pkg/run.sh" all --auto-shutdown "$@"
