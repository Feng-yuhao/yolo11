#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/yolo11_project

stamp="$(date +%Y%m%d_%H%M%S)_$$"
log="/root/autodl-tmp/yolo11_project/logs/d27_official_visdrone_${stamp}.log"
mkdir -p /root/autodl-tmp/yolo11_project/logs

echo "D27 log: ${log}"
echo "Zero training: reusing existing area_predictions.json files."
echo "Automatic shutdown is disabled."

python -u diagnose_d27_official_visdrone.py 2>&1 | tee "${log}"
