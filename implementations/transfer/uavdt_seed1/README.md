# VisDrone -> UAVDT / seed1 E1 vs E60 source-only

Only two existing VisDrone source-val best checkpoints. No target training,
fine-tuning, BN updates, epoch/threshold selection, YOLO26, or shutdown command.
The same fixed source checkpoints used by the completed Songdo direct-transfer
experiment are byte-bound in references/checkpoints.json.

Run in /root/miniconda3/envs/yolo11, with original /root/autodl-tmp/yolo11_src
Ultralytics8.3.0/Torch2.5.1+cu121. Source ten-class heads are retained; car+van
scores map to target car by per-anchor max, source truck->truck, bus->bus.
This explicit taxonomy harmonization is the same for both models, fixed before
target AP. It is not learned adaptation or novel-class detection.

FP32, maximum image side640, batch4, native8.3.0 LetterBox, conf.001,
class-aware NMS .7, best-class-only, max500. Native 1024x540 input is resized
and stride-padded by the original predictor (actual BCHW shapes recorded).

## Start

Upload ZIP and checksum to /root/autodl-tmp/yolo11_project; verify and unzip
under experiment_code with a nonexisting destination. Then nohup bash
experiment_code/VisDrone_to_UAVDT_Seed1_SourceOnly_AutoDL/start_all.sh.
The entry first extracts the byte-bound test-only data archive that was already
uploaded to project/uavdt_test_upload_i8aJKC, unless an existing target directory
is present. Existing data is NEVER silently replaced; full manifest checks follow.

All -> CPU data/weight/code verification -> synthetic fixture check -> two real
CUDA 8-image smoke passes (eight different sequences) -> full E1 test -> full
E60 test -> paired report and all artifact hashes. It is SERIAL, not training.
Do not start while other training/evaluation/unknown Python kernels are active.
Idle admission refuses busy GPU; no existing process is killed.

## Scoring: two distinct outputs, not interchangeable

Original author *_gt_whole.txt xywh and all375884 boxes are used, without
clipping, center-minus-one, or deriving GT from rounded YOLO labels.
Original IDs are deterministic sequence/frame IDs described in generated GT.
For 18 sequences named in the author's CalculateDetectionPR_overall.m, a
prediction is removed only if its four edges lie STRICTLY inside a same-frame
*_gt_ignore.txt region. GT boxes themselves are not removed by this filter.
Raw and filtered predictions, counts and original source files are preserved.

1. COCO-style three-class AP/AP50/AP75, maxDets[1,10,100], with that explicit
   prediction prefilter. This is NOT an official UAVDT COCO AP release/score.
   AP-small uses ORIGINAL native image GT areas, not resized image areas.
2. Supplement: author overall class-agnostic greedy AP at IoU>=.7 + continuous
   VOC integral AP, ported to Python. It uses exactly the same fixed predictions.
   This is a math port checked by small independent scalar fixtures, not a claim
   that MATLAB/the entire official submission service was executed. It is not
   three-class COCO AP70, and not mixed with COCO AP in one gain.

No target AP is used to pick the taxonomy mapping, seed, hyperparameters or
checkpoint. Both positive and negative gains are retained. This one source-seed
pair does not by itself establish multi-seed stability.

## Outputs and monitoring

One result directory: yolo11_project/comparison_reports/
VISDRONE_TO_UAVDT_S1_SOURCE_ONLY_<timestamp>. Contains actual code, source/data
bindings, original-geometry GT, check/smoke proof, per-model predictions,
COCO/native metrics, logs, protocol, summary.json, REPORT.md and artifact hashes.
No new runs directory or training weights are created.
Index: yolo11_project/logs/latest_visdrone_to_uavdt_seed1_source_only.json.
watch_progress.py --follow prints only changes; Ctrl+C stops viewer only.
Inference ETA does not include subsequent CPU scoring or remaining model.

No automatic shutdown. Completion requires both evaluations AND report hashes,
not merely 16592/16592 predicted images. Failure keeps partial data and logs.
Source model assets are unchanged, adapted from the previous source-only
delivery; respect the bundled Ultralytics AGPL-3.0 license. Original UAVDT
toolkit/licensing/citation notices are in the separate verified data package.
