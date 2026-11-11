#!/usr/bin/env bash
set -euo pipefail
pkg_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$pkg_dir/run.sh" all --auto-shutdown "$@"
