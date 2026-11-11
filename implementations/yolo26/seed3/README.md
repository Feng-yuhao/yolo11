# YOLO26s-P2 baseline / +E60 / seed3 / SERIAL / auto-shutdown entry

Exactly TWO new200-epoch trainings: baseline seed3, then E60 seed3. Each uses
the original official YOLO26s checkpoint, not seed1/2 best weights. Old results,
the YOLO11 environment, and existing seed2 training are never modified.

## Frozen protocol

- Independent `/root/miniconda3/envs/yolo26`, pinned Ultralytics8.4.177 and
  original yolo26_src; torch2.5.1+cu121, torchvision0.20.1+cu121.
- Same original YOLO26s->P2 operator mapping, both native heads, progressive
  loss/STAL, four W96 banks, P2/P3 FP32 group-residual sidecars and original towers.
- Same shared constructor initialization seed1 and isolated40001/57001/60001.
  Only GLOBAL TRAINING seed becomes3. This does not independently resample all
  model-construction random seeds; the definition matches seed1/2 exactly.
- Same SGD,640/train16/val32/workers8/cosine200/close_mosaic10/augmentations.
- Serial28GiB allocator cap, no silent batch reduction or old-run overwrite.
- Same548 validation and1610 test-dev images, fixed validation-best, frozen D27
  VisDrone Matlab1.0.4 ignore scorer. FP32/640/b4/one2many+NMS/conf.001/IoU.7/max500.
- Both pretrained26s and26n AMP-check weights and the Arial font are offline.

## Server start (only when the previous job/monitor has finished)

Extract into `/root/autodl-tmp/yolo26_project/experiment_code/`.
The named start script explicitly enables shutdown so the flag cannot be omitted:

```bash
cd /root/autodl-tmp/yolo26_project
nohup bash experiment_code/YOLO26_P2_E60_Seed3_Serial_AutoDL/start_all_auto_shutdown.sh \
  > launcher_yolo26_seed3_serial.log 2>&1 < /dev/null &
```

It runs check -> BOTH real-b16 smokes and full548-image b32 validation ->
bound fresh seed3 gates -> baseline200 -> E60200 -> baseline official val/test-dev
-> E60 official val/test-dev -> comparison/report/log preservation -> idle checks
->60-second auto-shutdown countdown. E60 smoke requires all52 groups to have
actual finite gradients and actual parameter updates. Check/smoke failures do
not authorize bypass or silent changes to the experiment.

Do NOT start while seed2 or another GPU/Python job is running. Auto-shutdown
admission refuses competing jobs; it does not kill them or automatically queue.
The old seed2 shutdown monitor should also finish before starting this job.

## Observe

```bash
/root/miniconda3/envs/yolo26/bin/python -u \
  experiment_code/YOLO26_P2_E60_Seed3_Serial_AutoDL/watch_progress.py --follow
```

Raw live log: `tail -n 30 -F launcher_yolo26_seed3_serial.log`.
Ctrl+C stops the viewer, not the nohup training controller.

## Outputs

- TWO runs: `runs/yolo26s_p2_baseline_img640_seed3_200e` and
  `runs/yolo26s_p2_e60_img640_seed3_200e`: best/last,results.csv,args and plots.
  Native smoke also creates two separately named `yolo26_smoke_*` run folders;
  these are engineering outputs, not extra200-epoch experiments. Their key
  receipts/checkpoints are already inside the main report root.
- ONE report root: `comparison_reports/YOLO26_P2_E60_S3_ALL_*`: actual source,
  audits, fresh check/smoke receipts, training records, FOUR official scores and
  per-image predictions, REPORT.md/comparison.json and stage log copies.
- ONE logs session: `logs/yolo26_seed3_all_*`: complete stage logs, state and
  shutdown-status/preservation manifest. Pointer: `logs/latest_yolo26_seed3_serial.json`.
- Original frozen seed1 results are bundled. Seed2 is included in the mean only
  if its exact server index, completed200 receipts, protocol, checkpoint SHA and
  FOUR score/prediction files validate. It is copied as a reference, never
  retrained. Otherwise missing/incompatible seed2 is explicitly marked; seed3
  remains valid and no fabricated3-seed mean is produced. All signs retained.

## Shutdown and failure boundary

`start_all_auto_shutdown.sh` explicitly passes `all --auto-shutdown`.
Read-only `check`, `smoke` and the progress viewer do not power off.
Success requires saved training and all four completed scores before shutdown.
Failure retains available outputs and logs, marks missing scores/weights, and
does not invent unsaved artifacts. The independent helper waits for controller
exit, flock release and no other GPU/unknown Python/CPU scoring tasks, then60s.
Known Jupyter/TensorBoard frontends do not block; notebook kernels do block.
AutoDL's no-shebang text helper is verified and called via bash, no host-poweroff
fallback. A returned request is not proof of actual power-off/billing termination.

Cancel only shutdown (training continues):
`touch /root/autodl-tmp/yolo26_project/KEEP_RUNNING_YOLO26_SEED3`.

If scoring alone failed, do NOT rerun all/retrain. After inspecting the error,
`run_pair.sh eval --training-report ORIGINAL_S3_REPORT --auto-shutdown` scores
the original verified seed3 best weights into a new result directory.

## Local validation boundary

See LOCAL_VALIDATION.md for actually executed tests. Private small/synthetic
checks are never server640/b16 memory receipts, 200-epoch evidence or real AP.
