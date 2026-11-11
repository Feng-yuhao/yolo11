# E58 seed2 — independent 200 epochs + official val + opt-in AutoDL shutdown

## What changes

Only global training seed=2 and output/settings/receipt directories change.
E58 four-scale W96 model, original pretrained E1 transfer, fixed isolated
40001/57001 branch initialization, 640/batch16/workers8/SGD/cosine200/close_mosaic10
are unchanged. No fine-tuning, early stop, new modules, E59 or test-dev selection.
The existing E1 seed2 and E40 seed2 are reused, NOT retrained.

Training smoke uses at least 8 successful real-data AMP updates (overflow skips
do not count), all 28 branch groups, and EMA-half batch32 full548-image validation
with the training model/optimizer still resident. Same 15GiB allocator cap as
E58 seed1 in both smoke and formal. Single-job only; no automatic batch reduction.

## Upload/extract

Upload E58_Seed2_AutoShutdown_AutoDL.zip to /root/autodl-tmp, then:

```bash
cd /root/autodl-tmp
unzip E58_Seed2_AutoShutdown_AutoDL.zip
```

## Recommended unattended run

Do not start if another training/evaluation/queue job is active. This command
performs CUDA check -> real smoke -> verify gates -> 200 epochs -> official val,
then saves status/logs and requests AutoDL shutdown after 60 seconds of safety
countdown. Failure at any stage also saves error state and requests shutdown
if the instance is idle. Duplicate launch, admission failure or intentional
controller cancellation never shut down an active job.

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh all --auto-shutdown > launcher_e58_seed2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e58_seed2.log
```

Ctrl+C in `tail -f` only closes log viewing; the nohup controller keeps running.
Closing VS Code/SSH also does not terminate this nohup job. Powering off AutoDL
manually DOES interrupt training.

## Optional manual gates before background formal run

```bash
bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh check
bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh smoke
bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh verify-gates
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh run --auto-shutdown > launcher_e58_seed2.log 2>&1 &
```

Manual check/smoke/verify/eval never shut down. `run` verifies independent seed2
receipts again. Seed1/pair receipts cannot bypass seed2 gates. Omitting
`--auto-shutdown` disables shutdown; no program auto-retries failed training.

## Cancel shutdown while keeping training

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_E58_SEED2
```

An existing marker intentionally suppresses shutdown; only remove it manually
when you want future auto-shutdown. This package never kills unrelated jobs.
If another GPU process / recognizable training/evaluation job is active, GPU
state is unavailable, or /usr/bin/shutdown is missing, shutdown is skipped and
logged. Hourly billing may continue; verify the instance is OFF in AutoDL.

Shutdown command follows AutoDL documentation:
https://www.autodl.com/docs/save_money/ — `/usr/bin/shutdown`, no API credentials.
Shutdown is not deleting/releasing the instance, and is not downloading files
to your PC. Save local backups separately. Platform storage charges may differ.

## Results and recovery

Weights and training CSV:
`/root/autodl-tmp/yolo11_project/runs/e58a_e57_allscale_w96_img640_seed2_200e/`

Formal report:
`/root/autodl-tmp/yolo11_project/comparison_reports/E58S2_RUN_*/`
contains training.json, transfer.json, actual_source.zip, training_completion.json,
official_metrics.json, comparison.json, area_predictions.json, gt_rows.json,
REPORT.md and completion.json. Training completion and official-scoring completion
are separate: a failure does not fabricate AP or remove best.pt/last.pt.

Controller state and shutdown_status.json:
`/root/autodl-tmp/yolo11_project/logs/e58_seed2_all_*/`
Pointer: `logs/latest_e58_seed2_controller.json`. Each stage has its own log,
also streamed into the launcher log.

If training completes but scoring fails, after restarting the instance:

```bash
bash /root/autodl-tmp/E58_Seed2_AutoShutdown_AutoDL/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e58a_e57_allscale_w96_img640_seed2_200e/weights/best.pt
```

This scores without retraining and without automatic shutdown. Existing run
directories are never overwritten; rerunning `all` is not a resume command.

Required existing reports: E1 seed2 `e1_yolo11s_p2add_img640_seed2_20260926_172907_911324`,
E40 seed2 `E40S2_RUN_20260930_130808_314014`, and E58 seed1
`E58_RUN_20261008_001519_559728/training.json`; original yolo11s.pt and paired
E1/E40 seed2 best.pt must still exist. Missing/changed inputs fail before training.

Local engineering verification is documented separately in LOCAL_VALIDATION.md;
it cannot replace the server's CUDA check and real-data memory smoke.
