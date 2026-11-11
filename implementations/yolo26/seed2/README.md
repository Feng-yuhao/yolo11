# YOLO26s-P2 baseline / +E60 / seed2 / SERIAL

Only TWO new200-epoch trainings: baseline seed2, then E60 seed2.
No seed3, no new architecture, no fine-tuning old best.pt, no parallel training.
Each completed candidate is automatically scored on full548-val and1610-test-dev.

## Frozen numerical protocol

- Independent yolo26 environment/source/project; Ultralytics8.4.177,
  torch2.5.1+cu121, torchvision0.20.1+cu121. No YOLO11 imports/upgrades/edits.
- Same actual seed1 model implementation, explicit standard26->P2 pretrained
  mapping, W96/E60 FP32 sidecars, native progressive dual-head loss and STAL.
- Same SGD/640/train16/val32/workers8/cosine200/close_mosaic10/augmentations.
- Only GLOBAL TRAINING seed becomes2 and output names change. The frozen model
  constructor's common initialization seed1 and isolated40001/57001/60001
  E60 initializations are unchanged. This repeats training randomness, not
  an extra experiment varying every random initialization independently.
- Serial28GiB allocator cap as executed seed1; no automatic smaller batch,
  optimizer replacement, shorter schedule or OOM retry. Serial avoids competing
  model memory but is not a guarantee that arbitrary dense inputs cannot OOM.
- Both original26s pretraining and26n AMP-check checkpoints bundled offline.
  No waiting for an implicit GitHub weight download.
- Original offline Arial plotting font is also bundled and deployed into each
  new settings directory, so native plots do not need a first-use font download.
- Fixed FP32/640/batch4/one2many+NMS/conf.001/IoU.7/max500 official scoring.
  Same frozen D27 VisDrone Matlab1.0.4 ignore handling, original data byte audit.
- Native training mAP and official AP are separate. Test-dev never chooses
  epoch, thresholds, architecture or seeds. Keep every positive/negative result.

## Server execution

Unzip under `/root/autodl-tmp/yolo26_project/experiment_code`.

```bash
cd /root/autodl-tmp/yolo26_project
nohup bash experiment_code/YOLO26_P2_E60_Seed2_Serial_AutoDL/run_pair.sh all \
  > launcher_yolo26_seed2_serial.log 2>&1 &
```

`all` automatically runs:

1. Data/package/source/weights/seed1 protocol audit.
2. CUDA engineering check, baseline real-b16 smoke, E60 real-b16 smoke.
3. At least8 actual finite optimizer updates and E60 all52 gradient/update
   groups, native full548-image b32 validation and checkpoint export smoke.
4. Verify newly bound seed2 serial gates, then baseline200 epochs.
5. After baseline exits and releases GPU, E60 independent200 epochs.
6. After both training attempts finish, serial official baseline val/test-dev,
   E60 val/test-dev; aggregate frozen seed1 + fresh seed2 results.

No seed1 weights/results are overwritten. Existing seed2 runs cause refusal,
not silent resume/restart. Check/smoke failure stops before formal training.
A formal failure is retained; the other independent model may still finish.
Only successfully trained candidates are scored. Partial output is not falsely
reported as a complete pair; no missing weight or score is invented.

`check` only checks. `smoke` runs check+both real smokes without formal training.
`run` requires valid previously produced seed2 gates; `all` is the simplest entry.

Compact monitor (Ctrl+C only stops the monitor):

```bash
/root/miniconda3/envs/yolo26/bin/python -u \
  experiment_code/YOLO26_P2_E60_Seed2_Serial_AutoDL/watch_progress.py --follow
```

Raw live output:

```bash
tail -n 30 -F /root/autodl-tmp/yolo26_project/launcher_yolo26_seed2_serial.log
```

## Optional shutdown, NOT enabled by default

Add `--auto-shutdown` after `all` ONLY if you want power-off after the whole
training+four-scoring pipeline reaches its terminal state. Not after baseline.
Helper waits for controller exit, original lock release, saved complete/partial
outputs, no other Python/GPU task, then60s countdown and rechecks.
AutoDL's observed text helper without shebang is called through `/bin/bash`.
No other process is killed. Unknown tasks/kernel block shutdown. Request return
does not establish platform billing has stopped. Local shutdown tests use mocks.

To cancel an enabled shutdown:

```bash
touch /root/autodl-tmp/yolo26_project/KEEP_RUNNING_YOLO26_SEED2
```

Do not enable shutdown while knowingly running another training/evaluation.

## Outputs and scoring-only recovery

- New runs: `yolo26s_p2_baseline_img640_seed2_200e` and
  `yolo26s_p2_e60_img640_seed2_200e`, each with best/last/results/args.
- ONE main report root: `comparison_reports/YOLO26_P2_E60_S2_ALL_*`.
  Includes actual source, checks, smokes, both training receipts, all four
  official evaluations/predictions, paired seed1/2 aggregate JSON/report, logs.
- ONE controller session under logs. Pointer:
  `logs/latest_yolo26_seed2_serial.json` (`session`, `report_root`).
- Auto-shutdown (if enabled) preservation/status remain in that logs session.

If training completed but scoring failed, do NOT relaunch `all`:

```bash
bash experiment_code/YOLO26_P2_E60_Seed2_Serial_AutoDL/run_pair.sh eval \
  --training-report /root/autodl-tmp/yolo26_project/comparison_reports/YOLO26_P2_E60_S2_ALL_ORIGINAL_TIME
```

This creates new score outputs from the verified original seed2 bests; it
does not retrain or overwrite the failed evaluation. Two seeds provide limited
repeatability evidence; they do not establish significance or guarantee publication.
