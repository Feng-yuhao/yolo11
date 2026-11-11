# Local verification — 2026-10-08 (Asia/Shanghai)

Executed on Windows / Python3.9.25 / torch2.6.0+cu124 / RTX4060 Laptop.
Server remains Python3.10 / torch2.5.1+cu121 / Ultralytics8.3.0 / vGPU32GB.

Passed:

- 12 unit tests: frozen-source identity, exact original E58 architecture/counts,
  fixed branch initialization/RNG preservation, paired seed2 metadata/weight
  SHA checks, training-protocol equality, memory byte/GiB units and gate rejection.
- Shutdown/controller tests use fake GPU snapshots and mocked command runners:
  success, child error, admission error, duplicate controller, unknown GPU state,
  other compute/training job, cancellation marker and command failure.
  NO real shutdown or remote command was executed.
- Full CPU engineering check: exact initial E1 predictions/main loss/common
  gradients, all 28 groups' real gradients/updates, independent route oracle,
  empty GT, save/reload/fuse/fresh process, standard Predictor, official scorer
  unit tests and TAL FP32/FP16 equivalence fixtures.
- Local small CUDA engineering check: same tests plus FP16 inference/value
  agreement, actual AMP finite update after GradScaler calibration and actual
  CUDA TAL updates. Local peak reserved 2.0762GiB on the small fixture;
  NOT a real batch16/640 memory guarantee and NOT a server smoke receipt.
- Standard Trainer two-epoch synthetic CPU integration: seed2 in Trainer args,
  all eight output projections learned, optimizer coverage, EMA best/last save
  and reload, rectangular raw/final official prediction path for four images.
- bash syntax and Python compilation checks.

Model parameters: 11,023,640 total; 1,344,672 branch, exactly as E58 seed1.
Model/transfer/TAL/scorer/reference YAML files are normalized-SHA identical to
the completed E58 seed1 package. Only orchestration, seed2 reference/evaluation,
receipt naming and shutdown safety changed.

Reports (not shipped as fake gates):
`yolo11_project/comparison_reports/E58S2_CHECK_20261008_154915_858342/`
`tmp/e58_seed2_local_cuda_20261008_155059_967759/`
`yolo11_project/comparison_reports/E58S2_LOCAL-INTEGRATION_20261008_155115_105930/`

Not executed locally: VisDrone6471 real training batches, full548-image
EMA-half batch32 server memory smoke, 200-epoch seed2 training, full seed2
official validation scoring or actual AutoDL shutdown. The delivered `all`
command performs the required real CUDA gates on the server before training.
