#!/usr/bin/env bash
set -euo pipefail
target=/root/autodl-tmp/project/UAVDT_TestOnly_YOLO_v1
archive=/root/autodl-tmp/project/uavdt_test_upload_i8aJKC/UAVDT_TestOnly_YOLO_v1.zip
if [[ -e "$target" ]]; then
  test -d "$target"
  test ! -L "$target"
  test -f "$target/manifest.json"
  printf 'Existing test dataset found; controller will verify all payload hashes.\n'
else
  printf '%s  %s\n' '6172bbd24ef8eeab1d4e2b2005694c5a6357017794f4b2ef6f50f113dbc529c9' "$archive" | sha256sum -c -
  # Only our byte-verified archive; never overwrite an existing data directory.
  unzip -q "$archive" -d /root/autodl-tmp/project
fi
