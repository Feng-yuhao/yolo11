# VisDrone -> UAVDT seed2/3 paired source-only evaluation

This is EVALUATION, not training. Four existing source-val-selected VisDrone
checkpoints: E1 s2, E60 s2, E1 s3, E60 s3. Full checks and four real 8-image
smoke passes precede serial full16592-image evaluations. No target adaptation,
BN update, checkpoint/threshold tuning or model changes. Seed1 is not rerun.

FP32/640maxside/b4/conf.001/NMS.7/top500 and original ten-class source heads
are unchanged from the completed seed1 package. Same car=max(car,van), truck,
bus mapping before NMS; same original DET GT, strict ignored-region filter,
COCO-style AP and separate author-math overall AP70. Neither is silently
relabelled as a new official COCO release. All positive/negative results kept.

Large predictions use compact JSON (whitespace only). Numeric values and all
raw/filtered predictions are retained. Need >=6GiB free AFTER dataset extraction.
No deletion, migration to paid storage, or old-results overwrite. The previously
uploaded test-only data is reused and byte-verified. Only one GPU worker at a time.

Entry start_all_auto_shutdown.sh explicitly passes --auto-shutdown. Default
run.sh all is non-shutdown. Success requires all four evaluations, bound weights,
original GT, result hashes, summary/report saved. Independent helper waits for
controller identity disappearance and lock release, rereads result hashes,
checks GPU/unknown Python jobs and countdown60seconds before calling the
recognized AutoDL TEXT helper through bash. No kill, generic poweroff fallback,
or failure shutdown. KEEP_RUNNING_UAVDT_S23 in yolo11_project cancels shutdown.
If another task stays active, helper waits up to30minutes then leaves server on.
Helper return zero does NOT prove billing stopped: check the AutoDL console.

Four evaluation results plus frozen seed1 metrics are combined into an honest
three-seed summary. The seed1 reference is byte-bound to the already verified
completed report; no need to duplicate/download seed1 predictions again.

One report directory: comparison_reports/VISDRONE_TO_UAVDT_S23_SOURCE_ONLY_*.
Index: logs/latest_visdrone_to_uavdt_seeds23_source_only.json.
watch_progress.py --follow prints changes (inference ETA excludes CPU scoring).
All actual source, protocol, GT, raw/filtered predictions, COCO/native scores,
unit/smoke/weight/BN/coordinate proofs, logs and hashes are saved there.
Shutdown logs/status are additional files written after the artifact index.

Run ONLY with /root/miniconda3/envs/yolo11/bin/python and original yolo11_src.
Never yolo26/base. Do not start while other training/evaluation is active.
After completion download this one new report folder; original seed1 is intact.
Model source assets retain bundled Ultralytics AGPL3.0 and prior attributions.
