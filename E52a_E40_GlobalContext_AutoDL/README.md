# E52a: complete E40 + stride-8 global-context residual

One seed1/640/batch16/200epoch experiment, single GPU, no automatic shutdown.
It is a literature-driven structural hypothesis, NOT a diagnosed causal cure
and NOT a promised AP improvement. Borrowed Linear-SRA must be cited as PVTv2.

## Exact change

Original model.4 C3k2 and its weights stay intact. Its C256 stride8 output gets
one additional residual: hidden64, four heads, full-grid queries, 7x7 pooled
K/V, local depthwise position term and ratio2 convolutional FFN. Output 64->256
projection starts exactly zero; other weights do not. No extra zero gain.
Original feature has a direct value path. No detach or in-place input changes.
The complete original E40 P2/P3 task head is unchanged. Model indices/save
references, Neck, detection towers, assignment, decoder, losses and NMS stay
unchanged. The new branch affects later backbone and multiple Neck paths; it
is NOT causally independent of E40.

Common E40 parameters retain isolated seed40001 initialization. Added branch
uses isolated seed52001. Training RNG is seed1. Formal initialization is the
same original yolo11s pretraining plus E1 migration, never E40 best fine-tuning.
E40 best is used only by disposable serialization checks/reference SHA audit.

All new branch arithmetic is FP32; output returns input dtype. Original network
AMP settings stay unchanged. This costs real runtime and cannot restore pixels
lost during image resizing. Existing P5 C2PSA already models context.

## Upload and extract (AutoDL)

Upload E52a_E40_GlobalContext_AutoDL.zip to /root/autodl-tmp, then:

```bash
cd /root/autodl-tmp
unzip -o E52a_E40_GlobalContext_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/E52a_E40_GlobalContext_AutoDL/run_e52.sh check
bash /root/autodl-tmp/E52a_E40_GlobalContext_AutoDL/run_e52.sh smoke
bash /root/autodl-tmp/E52a_E40_GlobalContext_AutoDL/run_e52.sh verify-gates
```

Wait until other GPU jobs finish. Real smoke and formal use the same 28GiB
allocator cap, unchanged train batch16, full validation batch32. Real smoke
requires eight successful optimizer updates, checks all 27 branch groups,
then full 548-image EMA-half validation. Initial AMP overflow skips are logged
and are NOT successful updates; they do not alone mean the model is broken.
Any check/smoke failure prevents formal training. No automatic batch reduction.

After all three commands succeed:

```bash
nohup bash /root/autodl-tmp/E52a_E40_GlobalContext_AutoDL/run_e52.sh run > launcher_e52.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e52.log
```

Ctrl+C exits this tail only; it does not stop nohup training. Shutting down the
server DOES stop training. Only one experiment is launched, no queue/parallel.
`run_e52.sh all` alternatively runs check->smoke->formal sequentially, failing
immediately if any stage fails. Per-epoch logs show loss, seconds and ETA.

## Outputs / recovery

Weights and results.csv:
`yolo11_project/runs/e52a_e40_global_context_img640_seed1_200e/`

Reports:
`yolo11_project/comparison_reports/E52_RUN_*/`
including training.json, transfer.json, official_metrics.json, comparison.json,
area_predictions.json, gt_rows.json, REPORT.md, completion.json after success.
Training and evaluation status are distinct; a missing official result is not
automatically failed training. If evaluation alone fails, retain weights and:

```bash
bash /root/autodl-tmp/E52a_E40_GlobalContext_AutoDL/run_e52.sh eval \
  --weights /root/autodl-tmp/yolo11_project/runs/e52a_e40_global_context_img640_seed1_200e/weights/best.pt
```

Same official VisDrone val evaluator/ignore rules as E40; no test-dev tuning.
Single-seed continuation screen: >=E40+0.25pp AP with AP50/AP75 nondecreasing.
This is a compute-budget rule, not significance/publication proof.

## Checks / profiling boundary

Independent attention/pooling and input-gradient oracle, remote-token influence,
native zero-start two-step updates, initial predictions/loss/common-gradient
equivalence, actual stride8 graph reach, untouched E40 routing, empty GT,
fresh-process reload/fuse, standard Predictor, EMA-half value/AMP update, original
TAL equivalence and official evaluator units are checked.

New params: 71,744. Reported FLOPs use actual640 graph shape and include QK/AV,
functional convolution and linear projections; expected added major operations
0.837607424 GFLOPs (2 FLOPs/MAC). Norm/softmax/pooling/GELU/copies are not counted
as major operations. YOLO's ordinary summary may OMIT the functional branch;
use check.json profile, not the console summary, for this model's complexity.
Do not infer latency from FLOPs. Local tests are synthetic/CPU and cannot issue
the CUDA check/smoke receipts required on the server.

## Related work / scientific boundary

- PVTv2 Linear-SRA: https://arxiv.org/abs/2106.13797
- MaxViT local/global interaction: https://arxiv.org/abs/2204.01697
- YOLOv12 attention detector: https://arxiv.org/abs/2502.12524

Not equivalent to E35 multi-kernel leaf replacement, E46 head shifted windows,
or E49-E51 local fusion reassembly. No proof yet that context deficiency causes
the observed crowding/raw-IoU gaps. Seven-by-seven pooling can dilute tiny
instances; Mosaic may mix unrelated scenes; added context may be redundant
with C2PSA. Success requires AP evidence, not merely nonzero branch parameters.
