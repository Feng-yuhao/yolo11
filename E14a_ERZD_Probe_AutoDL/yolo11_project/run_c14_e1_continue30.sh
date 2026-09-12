#!/usr/bin/env bash
set -uo pipefail
if [ "$#" -ne 0 ]; then
    echo "No arguments allowed. Run the Python script directly for check/smoke."
    exit 2
fi
PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/c14_control_e1_continue30_seed1.py"
GATE="$PROJECT/comparison_reports/c14_e1_continue30_smoke_passed.json"
export OMP_NUM_THREADS=8
cd "$PROJECT" || exit 1
for REQUIRED in "$SCRIPT" "$PROJECT/e14_erzd_common.py" "$PROJECT/yolo11s-p2-add.yaml" \
    "$PROJECT/p2_tal_chunked_vgpu32.py"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "Missing required file: $REQUIRED"
        exit 1
    fi
done
if [ ! -x "$PYTHON" ]; then
    echo "Missing yolo11 Python: $PYTHON"
    exit 1
fi
if [ ! -f "$GATE" ]; then
    echo "Run C14 --smoke2 first. Control training was not started."
    exit 1
fi
mkdir -p "$PROJECT/logs" || exit 1
LOG="$PROJECT/logs/c14_e1_continue30_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG"
echo "Started: $(date -Is)" | tee -a "$LOG"
"$PYTHON" -u "$SCRIPT" --no-shutdown 2>&1 | tee -a "$LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "Python exited: $TRAIN_EXIT at $(date -Is)" | tee -a "$LOG"
echo "Automatic shutdown disabled by user policy; instance remains running." | tee -a "$LOG"
exit "$TRAIN_EXIT"
