# YOLO26s-P2 / E60 / seed1 / VisDrone test-dev

Only evaluate the two COMPLETED YOLO26 seed1 val-selected best checkpoints.
No training entry, optimizer, model edits, automatic dependency installation,
YOLO11 environment/source import, threshold search, or automatic shutdown.

## Fixed evaluation

- Independent `/root/miniconda3/envs/yolo26/bin/python`, source
  `/root/autodl-tmp/yolo26_src`, Ultralytics8.4.177, torch2.5.1+cu121.
- FP32, 640, batch4, one2many+NMS, conf=.001, IoU=.7, max_det500.
- Same original1610 test-dev images/annotations, class mapping and frozen
  D27 VisDrone Matlab1.0.4 ignore scorer as existing main results.
- Original seed1 best.pt hashes are frozen from completed training/val reports.
  Not the pretrained COCO checkpoint or last.pt; no epochs selected on test-dev.
- Inference custom module, audited Predictor, NMS and inverse coordinate oracle
  are copied byte-for-byte from the executed YOLO26 seed1 package.
- First reproduce first8 original validation predictions for BOTH candidates,
  then8 test-dev image smoke and CPU scoring, before full serial evaluation.
- GPU memory allocator6GiB, GPU inference+CPU official scoring. Do not use
  no-GPU2GiB mode. Failure is recorded; no bypass/retry/batch reduction.

## Server

Unzip this package into `/root/autodl-tmp/yolo26_project/experiment_code`.

```bash
bash /root/autodl-tmp/yolo26_project/experiment_code/YOLO26_P2_E60_Testdev_Seed1_AutoDL/run.sh all
```

`all` includes check -> baseline smoke -> E60 smoke -> baseline full test-dev
-> E60 full test-dev -> comparison. `check` does not use GPU; `smoke` does not
produce the full test-dev score. All calls use new timestamped output directories.
No modifications to existing training/evaluation directories/checkpoints.

Background start:

```bash
cd /root/autodl-tmp/yolo26_project
nohup bash experiment_code/YOLO26_P2_E60_Testdev_Seed1_AutoDL/run.sh all \
  > launcher_yolo26_testdev_seed1.log 2>&1 &
tail -n 30 -F launcher_yolo26_testdev_seed1.log
```

CPU official scoring may be quiet between class messages. Final marker:
`COMPLETED REPORT=...; automatic shutdown DISABLED`.

Download the ONE full new `comparison_reports/TESTDEV_YOLO26_P2_E60_S1_*`
directory: report, official metrics/per-class AP, predictions/image mappings,
source snapshot, preflight/smoke, copied child logs and controller state.
No new `runs` directory. Original logs also stay in project/logs and latest
pointer is `logs/latest_yolo26_testdev_seed1.json` (`report`, `session`).

The primary delta is E60 minus the matched YOLO26s-P2 baseline, NOT old YOLO11
E60 minus YOLO26. One seed is not cross-seed stability. Retain negative results;
do not use test-dev to resume architecture/threshold/checkpoint selection.

## Local validation boundary

See LOCAL_VALIDATION.md for actual executed tests. Synthetic local forward,
fresh checkpoint reload, coordinate guard and scorer tests are not a claim
that real server640/1610 evaluation has executed or that it will improve AP.
