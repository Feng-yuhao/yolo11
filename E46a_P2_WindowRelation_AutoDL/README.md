# E46a: E40 + P2 parallel shifted-window relation context

Only one formal experiment: seed1,640,batch16,200 epochs; single GPU job.
No remote source edits, pretrained E40 checkpoint finetuning, auxiliary loss,
new TAL rule, early stopping, automatic shutdown, crop or extra image read.

## Fixed structure

The unchanged E40 P2/P3 bank and routers still read the original Detect input.
Only P2 additionally reads that same input through:

128->64 projection -> window8,shift0 -> window8,shift4 -> two separate
64->128 zero-weight task projections. Each block has pre-LayerNorm, four-head
attention, relative position bias, attention residual, pre-LayerNorm and
GELU FFN (ratio2) residual. There is no downsampling or stochastic dropout.
Add each new task residual to the corresponding ORIGINAL E40 task output.

The E40 blocks keep seed40001 and the extra branch uses isolated CPU seed46001.
At initialization predictions equal E40 (and E1 because E40 outputs start zero).
Extra output gradients and shared clipping can change training trajectories:
this is NOT a claim that the first real optimizer update equals E40.

Context computations are FP32 in AMP and EMA-half inference, without mutating
parameter dtypes; all original E40/tower precision is preserved. Padding keys
and cyclic wrap are masked, and outputs are cropped back to the original shape.
The shared context receives BOTH tasks' gradients: separate task output
projections do NOT constitute complete gradient isolation.

New context parameters:93320; total detector parameters:9947728 before fuse.
Estimated640 MAC-equivalent GFLOPs:45.3791, vs E40 39.9265. QK/AV tensor
matmuls are explicitly counted. LN/GELU/softmax/bias scalar operations are
not included in this estimate. Default THOP alone omits attention matmuls.

Borrowed window attention design: Swin Transformer, ICCV2021,
https://arxiv.org/abs/2103.14030 and https://github.com/microsoft/Swin-Transformer.
This is not a Swin backbone reproduction or a claim to invent window attention.
It tests extra content-dependent cross-position evidence alongside E40, not
an established root cause of E1 errors or guaranteed orthogonality/improvement.

## Upload/unzip

Upload E46a_P2_WindowRelation_AutoDL.zip to /root/autodl-tmp, then:

```bash
cd /root/autodl-tmp
unzip E46a_P2_WindowRelation_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E46a_P2_WindowRelation_AutoDL/run.sh check
bash /root/autodl-tmp/E46a_P2_WindowRelation_AutoDL/run.sh smoke
```

Wait for CHECK PASSED and SMOKE PASSED with exit0. Smoke is an engineering
stress test, NOT a20/90-epoch accuracy probe:8 successful real CUDA updates
(max32 attempts), full548-image batch32 validation and square batch32 stress.
Loss, gradients, actual updates, raw/decoded finiteness and memory are checked.
Formal config remains unchanged:200-epoch cosine schedule, close_mosaic10.
GPU must be idle, at least30GiB free; smoke AND formal use28GiB allocator cap.
No auto batch reduction. If OOM/failure, DO NOT start formal or bypass gates.

Then:

```bash
nohup bash /root/autodl-tmp/E46a_P2_WindowRelation_AutoDL/run.sh run > launcher_e46.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e46.log
```

Ctrl+C in tail only stops viewing. Epoch seconds, elapsed/ETA, timestamps and
six output-weight L1 statistics are printed. No automatic shutdown.
Run-name collision is refused; do not delete previous runs to force rerun.
Shell flock rejects duplicate E46 launches. This package does not kill jobs.

## Prerequisites and outputs

Pinned unmodified Ultralytics8.3.0 source at /root/autodl-tmp/yolo11_src;
original yolo11s.pt at yolo11_project/weights/yolo11s.pt; E1 seed1 baseline
report e1_yolo11s_p2add_img640_seed1_20260926_102750_718906; original dataset.
For final comparison also retain E40_RUN_20260929_205658_507620 and its best.pt.

Weights: yolo11_project/runs/e46a_p2_window_relation_img640_seed1_200e/weights.
Reports: yolo11_project/comparison_reports/E46_RUN_timestamp/:
training.json,transfer.json,area_predictions.json,gt_rows.json,
official_metrics.json,comparison.json,REPORT.md,completion.json.
Only completion.json proves training AND official scoring completed.
Training status is evaluating until scoring succeeds; failure.json preserves
the traceback and checkpoints are not deleted on evaluation failure.

Independent evaluation retry (no retraining, retains original checkpoint):

```bash
bash /root/autodl-tmp/E46a_P2_WindowRelation_AutoDL/run.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e46a_p2_window_relation_img640_seed1_200e/weights/best.pt
```

Official AP/AP50/AP75 reuse exactly the D27/D38/E40 evaluator and prediction
options. E1 cached predictions are rescored and must reproduce25.63124AP.
Raw coverage/size bands are supplementary, not official AP-small or recall.
Resource screen: gain>=0.25pp vs E40 and AP50/AP75 nondecreasing; goal AP26.63.
This is not a significance, publication or success guarantee. Paired additional
seeds are still required if the candidate passes.

## Local versus server verification

LOCAL_TESTS.md records tests actually executed. CPU synthetic Trainer/EMA and
reload/official fixtures do NOT establish CUDA memory safety or real accuracy.
Server CUDA check and real smoke receipts are code/config/environment-bound;
CPU gates cannot authorize formal training. No remote training is launched here.
The utility filename run_e40.py is inherited internally; public entry is run.sh.
Ultralytics may display the YAML stem e1_reference as a model label; this does
not mean the branch is absent. Check.json records the actual parameter counts
and19 registered/updated branch groups. The built-in FLOPs display omits QK/AV;
use the documented count convention, not that display for a paper table.
