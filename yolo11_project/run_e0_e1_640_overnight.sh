#!/bin/bash
set -u

cd /root/autodl-tmp/yolo11_project
PY=/root/miniconda3/envs/yolo11/bin/python

echo "=================================================="
echo "[640] E0 formal 200e START"
date
echo "=================================================="

$PY yolo11s_img640_seed1.py --no-shutdown
S=$?

if [ $S -ne 0 ]; then
    echo "[640] E0 FAILED, status=$S"
    date
    /bin/bash -c /usr/bin/shutdown || true
    exit $S
fi

echo "=================================================="
echo "[640] E0 COMPLETE"
echo "[640] E1 smoke2 START"
date
echo "=================================================="

$PY yolo11s_p2_img640_seed1.py --smoke2
S=$?

if [ $S -ne 0 ]; then
    echo "[640] E1 smoke2 FAILED, status=$S"
    date
    /bin/bash -c /usr/bin/shutdown || true
    exit $S
fi

echo "=================================================="
echo "[640] E1 smoke2 COMPLETE"
echo "[640] E1 formal 200e START"
date
echo "=================================================="

# 不加 --no-shutdown：
# E1 正式实验完成并保存全部结果后，由原 runner 自动请求关机。
$PY yolo11s_p2_img640_seed1.py
S=$?

echo "=================================================="
echo "[640] ALL FINISHED, final status=$S"
date
echo "=================================================="

exit $S
