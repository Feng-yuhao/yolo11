#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/autodl-tmp/yolo11_project
mkdir -p logs

export OMP_NUM_THREADS="${D15B_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${D15B_MKL_NUM_THREADS:-8}"
export PYTHONPATH="/root/autodl-tmp/yolo11_src${PYTHONPATH:+:${PYTHONPATH}}"

stamp="$(date +%Y%m%d_%H%M%S)"
log="/root/autodl-tmp/yolo11_project/logs/d15b_correction_fusion_headroom_${stamp}_$$.log"
echo "D15b diagnostic log: ${log}"
echo "This run performs no training and never shuts down the instance."

set +e
/root/miniconda3/envs/yolo11/bin/python -u \
  diagnose_d15b_correction_fusion_headroom.py "$@" 2>&1 | tee "${log}"
status=${PIPESTATUS[0]}
set -e

echo "Python exited: ${status} at $(date --iso-8601=seconds)"
echo "Automatic shutdown is disabled by user policy; instance remains running."
exit "${status}"
