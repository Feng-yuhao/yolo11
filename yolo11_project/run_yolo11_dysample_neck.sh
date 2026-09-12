#!/bin/bash
# E5a formal 200-epoch launcher. This is not the smoke launcher.
set -u
if [ "$#" -ne 0 ]; then
    echo "No arguments allowed. Use the Python script directly for check/smoke."
    exit 2
fi
PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/yolo11s_p2_dysample_neck_img1024_seed1.py"
export OMP_NUM_THREADS=8
cd "$PROJECT" || exit 1
for REQUIRED in "$SCRIPT" "$PROJECT/yolo11s-p2-dysample-neck.yaml" "$PROJECT/yolo11s-p2-add.yaml" "$PROJECT/p2_tal_chunked.py"; do
    if [ ! -f "$REQUIRED" ]; then
        echo "Missing required file: $REQUIRED"
        exit 1
    fi
done
if [ ! -x "$PYTHON" ]; then
    echo "Missing yolo11 Python: $PYTHON"
    exit 1
fi
if [ ! -f "$PROJECT/comparison_reports/e5a_dysample_smoke_passed.json" ]; then
    echo "Run --smoke2 first. Formal training and shutdown were not started."
    exit 1
fi
mkdir -p "$PROJECT/logs" || exit 1
LOG="$PROJECT/logs/e5a_dysample_formal_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG"
echo "Started: $(date -Is)" >> "$LOG"
"$PYTHON" -u "$SCRIPT" >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "Python exited: $TRAIN_EXIT at $(date -Is)" >> "$LOG"
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
