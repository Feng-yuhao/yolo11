#!/usr/bin/env bash
set -Eeuo pipefail
cd /root/autodl-tmp/yolo11_project
mkdir -p logs
export OMP_NUM_THREADS="${D17A_OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${D17A_MKL_NUM_THREADS:-8}"
export PYTHONPATH="/root/autodl-tmp/yolo11_src${PYTHONPATH:+:${PYTHONPATH}}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="/root/autodl-tmp/yolo11_project/logs/d17a_local_expert_probe_${stamp}_$$.log"
echo "D17a log: ${log}"
echo "Only a separate local expert is trained. Automatic shutdown is disabled."
set +e
/root/miniconda3/envs/yolo11/bin/python -u yolo11s_d17a_dynamic_local_expert_probe.py "$@" 2>&1 | tee "${log}"
status=${PIPESTATUS[0]}
set -e
echo "Python exited: ${status} at $(date --iso-8601=seconds)"
echo "Automatic shutdown is disabled by user policy; instance remains running."
exit "${status}"
