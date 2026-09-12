#!/bin/bash
# Frozen E1/E5a full validation slicing diagnostic with shutdown fallback.
set -u

PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/evaluate_e1_e5a_slicing.py"
LOG_DIR="$PROJECT/logs"

export OMP_NUM_THREADS=8
cd "$PROJECT" || exit 1
mkdir -p "$LOG_DIR" || exit 1

for REQUIRED in \
    "$SCRIPT" \
    "$PROJECT/runs/e1_yolo11s_p2add_img1024_seed1/weights/best.pt" \
    "$PROJECT/runs/e5a_yolo11s_p2_dysample_neck_img1024_seed1/weights/best.pt"
do
    if [ ! -f "$REQUIRED" ]; then
        echo "Missing required file: $REQUIRED"
        exit 1
    fi
done

if [ ! -x "$PYTHON" ]; then
    echo "Missing yolo11 Python: $PYTHON"
    exit 1
fi

LOG="$LOG_DIR/sliced_diagnostic_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Diagnostic log: $LOG"
echo "Started: $(date -Is)" >> "$LOG"
"$PYTHON" -u "$SCRIPT" --models both --shutdown >> "$LOG" 2>&1
DIAG_EXIT=$?
echo "Python exited: $DIAG_EXIT at $(date -Is)" >> "$LOG"

if [ -f /usr/bin/shutdown ]; then
    echo "Shell fallback: requesting AutoDL shutdown" >> "$LOG"
    sync
    /bin/bash -c /usr/bin/shutdown >> "$LOG" 2>&1
    SHUTDOWN_EXIT=$?
    echo "Shutdown command returned $SHUTDOWN_EXIT; verify console power state." >> "$LOG"
else
    echo "WARNING: /usr/bin/shutdown missing; shut down in AutoDL console." >> "$LOG"
fi
exit "$DIAG_EXIT"
