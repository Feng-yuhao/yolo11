#!/bin/bash
# FORMAL 200 epochs only. Completion OR Python failure requests AutoDL shutdown.
# First run --check-only and --smoke2 interactively; this wrapper is NOT for smoke.
set -u
if [ "$#" -ne 0 ]; then
    echo "No arguments allowed. For smoke use: python -u yolo11s_p2_img1024_seed1.py --smoke2"
    exit 2
fi
PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/yolo11s_p2_img1024_seed1.py"
cd "$PROJECT" || exit 1
if [ ! -x "$PYTHON" ] || [ ! -f "$SCRIPT" ] || [ ! -f "$PROJECT/yolo11s-p2-add.yaml" ]; then
    echo "Missing environment, P2 script, or companion YAML. Nothing started."
    exit 1
fi
if [ ! -f "$PROJECT/comparison_reports/e1_p2_smoke_passed.json" ]; then
    echo "Run --smoke2 first. No formal training/shutdown started."
    exit 1
fi
mkdir -p "$PROJECT/logs" || exit 1
LOG="$PROJECT/logs/p2_formal_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG"
echo "Started: $(date -Is)" >> "$LOG"
"$PYTHON" -u "$SCRIPT" >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "Python exited: $TRAIN_EXIT at $(date -Is)" >> "$LOG"
# Handles many process-exit failures, NOT a hung process or killed wrapper.
# Keep an AutoDL scheduled shutdown as an independent hard deadline.
if [ -f /usr/bin/shutdown ]; then
    echo "Shell fallback: requesting AutoDL shutdown" >> "$LOG"
    sync
    /bin/bash -c /usr/bin/shutdown >> "$LOG" 2>&1
    SHUTDOWN_EXIT=$?
    echo "Shutdown command returned $SHUTDOWN_EXIT; verify console power state." >> "$LOG"
else
    echo "WARNING: /usr/bin/shutdown missing; shut down in AutoDL console." >> "$LOG"
fi
exit "$TRAIN_EXIT"
