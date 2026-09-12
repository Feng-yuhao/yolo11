#!/bin/bash
# Upload beside yolo11s_img1024_seed1.py. Linux LF line endings required.
# Launch: nohup bash run_yolo11_baseline.sh > launcher.log 2>&1 &
# WARNING: training completion OR failure requests AutoDL shutdown.
set -u
PROJECT=/root/autodl-tmp/yolo11_project
PYTHON=/root/miniconda3/envs/yolo11/bin/python
SCRIPT="$PROJECT/yolo11s_img1024_seed1.py"
cd "$PROJECT" || exit 1
if [ ! -x "$PYTHON" ] || [ ! -f "$SCRIPT" ]; then
    echo "Missing yolo11 Python or training script. Nothing started."
    exit 1
fi
mkdir -p "$PROJECT/logs" || exit 1
LOG="$PROJECT/logs/baseline_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "Training log: $LOG"
echo "Started: $(date -Is)" >> "$LOG"
"$PYTHON" -u "$SCRIPT" >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "Python exited: $TRAIN_EXIT at $(date -Is)" >> "$LOG"
# The Python script normally already requested shutdown. This fallback covers
# import/launch failures and many process-exit errors, but not a hanging process
# or a terminated wrapper. Keep an AutoDL scheduled shutdown as a hard deadline.
if [ -f /usr/bin/shutdown ]; then
    echo "Shell fallback: requesting AutoDL shutdown" >> "$LOG"
    sync
    /bin/bash -c /usr/bin/shutdown >> "$LOG" 2>&1
    SHUTDOWN_EXIT=$?
    echo "Shutdown command returned $SHUTDOWN_EXIT; verify console power state." >> "$LOG"
else
    echo "WARNING: /usr/bin/shutdown missing; shut down in console." >> "$LOG"
fi
exit "$TRAIN_EXIT"
