#!/usr/bin/env bash
set -euo pipefail
pkg_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bash "$pkg_dir/prepare_dataset.sh"
exec bash "$pkg_dir/run.sh" all "$@"
