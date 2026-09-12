#!/bin/bash
# E8b-clean formal 200-epoch launcher. Automatic shutdown is disabled everywhere.
set -u
if [ "$#" -ne 0 ]; then
    echo "No arguments allowed. Use the Python script directly for check/smoke."
    exit 2
fi
PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/yolo11s_p2_p3sde_img1024_seed1.py"
export OMP_NUM_THREADS=8
cd "$PROJECT" || exit 1
for REQUIRED in \
    "$SCRIPT" \
    "$PROJECT/yolo11s-p2-p3sde.yaml" \
    "$PROJECT/yolo11s-p2-add.yaml" \
    "$PROJECT/p2_tal_chunked.py" \
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
if [ ! -f "$PROJECT/comparison_reports/e8b_p3sde_clean_tal2_smoke_passed.json" ]; then
    echo "Run E8b-clean --smoke2 first. Formal training was not started."
    exit 1
fi
mkdir -p "$PROJECT/logs" || exit 1
LOG="$PROJECT/logs/e8b_p3sde_clean_formal_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG"
echo "Started: $(date -Is)" >> "$LOG"
"$PYTHON" -u "$SCRIPT" --no-shutdown >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "Python exited: $TRAIN_EXIT at $(date -Is)" >> "$LOG"
echo "Automatic shutdown disabled by user policy; instance remains running." >> "$LOG"
exit "$TRAIN_EXIT"
