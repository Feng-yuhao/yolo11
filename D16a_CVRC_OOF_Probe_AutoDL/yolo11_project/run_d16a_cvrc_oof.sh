#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/yolo11_project
mkdir -p logs

export OMP_NUM_THREADS="${D16A_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${D16A_MKL_NUM_THREADS:-8}"
export PYTHONPATH="/root/autodl-tmp/yolo11_src${PYTHONPATH:+:${PYTHONPATH}}"

stamp="$(date +%Y%m%d_%H%M%S)"
log="/root/autodl-tmp/yolo11_project/logs/d16a_cvrc_oof_${stamp}_$$.log"
echo "D16a diagnostic log: ${log}"
echo "E1 and D15a stay frozen. Automatic shutdown is disabled."

set +e
/root/miniconda3/envs/yolo11/bin/python -u diagnose_d16a_cvrc_oof.py "$@" 2>&1 | tee "${log}"
status=${PIPESTATUS[0]}
set -e

echo "Python exited: ${status} at $(date --iso-8601=seconds)"
echo "Automatic shutdown is disabled by user policy; instance remains running."
exit "${status}"
