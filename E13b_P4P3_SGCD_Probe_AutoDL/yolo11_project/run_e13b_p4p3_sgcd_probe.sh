#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/yolo11_project
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec /root/miniconda3/envs/yolo11/bin/python -u yolo11s_p2_p4p3_sgcd_probe_img1024_seed1.py --no-shutdown "$@"
