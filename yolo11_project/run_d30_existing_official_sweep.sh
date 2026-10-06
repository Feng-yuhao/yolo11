#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/yolo11_project

stamp="$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p logs
log="logs/d30_existing_official_sweep_${stamp}.log"

echo "[D30] preflight/check-only" | tee "$log"
python diagnose_d30_existing_official_sweep.py --check-only 2>&1 | tee -a "$log"

echo "[D30] full zero-training official sweep" | tee -a "$log"
python diagnose_d30_existing_official_sweep.py 2>&1 | tee -a "$log"

echo "[D30] done; log=$log"
