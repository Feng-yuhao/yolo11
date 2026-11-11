# YOLO26s-P2 + E60 / paired seed1

Independent `/root/miniconda3/envs/yolo26`, fixed Ultralytics 8.4.177, source
`/root/autodl-tmp/yolo26_src`, outputs only `/root/autodl-tmp/yolo26_project`.
Do not install this into the YOLO11 source or use the old YOLO11 CLI/Python.

## Prerequisites and staged use

The completed server CUDA engineering check in the original
`YOLO26_P2_E60_Preflight_AutoDL` package is required. Its four model/check/source
files are byte-identical here. It is not a real-data smoke gate.

Install this directory as `yolo26_project/experiment_code/YOLO26_P2_E60_Seed1_AutoDL`.
`bash run_pair.sh smoke` runs two independent real smoke jobs SERIALly, no 200-epoch training.
Each uses 640/b16 training, native initialization/loss/optimizer/warmup, at least
eight actual finite updates, and for E60 all 52 nonzero-gradient/actual-update
groups. Up to 128 batches are allowed to expose cold-start zero-output branches.
It then runs a worst-case square b32 forward, complete 548-image native validation
(b32/rectangular), and saved-checkpoint prediction/export on eight real images.
The training model, optimizer and actual EMA remain resident through validation.
Validation uses the **8.4.177 native FP32 EMA with autocast**, not the old 8.3.0
EMA-half workaround. FP16 checkpoint serialization preserves E60 refinements in FP32.

`bash run_pair.sh verify-gates` checks code/environment/weights/complete data/device
signatures and the completed reports. Local fixture runs cannot issue server gates.
`bash run_pair.sh run` requires valid real smoke and fresh formal output paths.
It trains baseline for 200 epochs, scores its fixed val-best, then trains E60 for
200 epochs and scores its val-best, then writes one paired REPORT/comparison.
It is SERIAL, not simultaneous; no parallel-layout smoke is provided.
No default/implicit auto-shutdown, no test-dev, no automatic retry, no smaller batch.
Smoke/check failures do not start training. Formal runs never reuse smoke weights.

## Fixed model/protocol

Both are official YOLO26s-P2 (`64/128/256/512`, strides4/8/16/32, reg_max1).
Only E60 has the original four W96 banks, task/position routers and six FP32
P2/P3 group residual refinements, adapted first/last P2 projections to cin64.
Shared preparation is before native one2one detach; no detach is removed.
Original one2many/one2one towers, original E2ELoss/ProgLoss and STAL remain.
Inference is explicitly **one2many + class-aware NMS**, conf.001/IoU.7/max500.
This is not a claim to be the stock default NMS-free YOLO26 benchmark.
The native progressive schedule still trains both heads; inference selection
does not disable the one2one loss. No old YOLO11 TAL monkey-patch is imported.

The exact explicit pretraining mapping and zero-init audits are inherited from
the passed preflight. Both candidates share initialization; enhanced uses isolated
40001/57001/60001 branch RNG. No E60/Y11 best checkpoint is loaded.
Pretraining includes Objects365/COCO, not identical provenance to YOLO11.
SGD .01/.01 cosine200/momentum.937/decay.0005, b16/nbs64, seed1/workers8,
warmup3/close_mosaic10, default listed augmentation. Memory format is explicitly
contiguous on both OSes. Native YOLO26 scheduler/warmup/dual loss differ from old11.
This pair tests module portability **within YOLO26**, not a pure architecture-only
comparison between YOLO11 and YOLO26.

## Scoring / resources / outputs

Primary scoring is the unchanged frozen `d27_official.py` port of VisDrone2019
DET Matlab1.0.4 ignore semantics used for E1/E60; unit tests run before scoring.
YOLO's native validation mAP is separately retained; do not subtract one protocol
from the other. FP32/b4 predictions keep continuous original-image coordinates,
inverse coordinates are independently checked; NMS never mutates raw predictions.
Per-image predictions, class metrics, ignore audit, checkpoint SHA, common mapping,
actual source, steps, CSV, best/last, report, state and logs are retained.

Fixed serial 28GiB CUDA allocator budget; requires >=28GiB GPU capacity and idle
headroom. Never silently reduce batch. Native STAL may retry assignment image-wise
on OOM without changing model batch; that original behavior is retained and logged.
Finite loss/EMA/serialization guards stop before the upstream saver can sanitize
nonfinite tensors. There is no silent NaN recovery from last.pt.
Official YOLO26n (AMP-only, not initialization) and YOLO26s weights are bundled with
verified GitHub release digests, avoiding AMP's implicit network download.

Server: `comparison_reports/YOLO26_P2_E60_SMOKE_*` or `...RUN_*`;
`runs/yolo26s_p2_baseline_img640_seed1_200e` and
`runs/yolo26s_p2_e60_img640_seed1_200e`; `logs/yolo26_p2_e60_*`.
`logs/latest_yolo26_p2_e60_run.json` identifies the formal controller/session/report.
Do not delete packages, weights, locks, datasets, source or results while it runs.

Local validation is synthetic128/b16, 35 validation images and two epochs only;
not server640/b16 smoke, real VisDrone resource proof, accuracy or 200-epoch results.
Both favorable and unfavorable seed1 results must be retained.
