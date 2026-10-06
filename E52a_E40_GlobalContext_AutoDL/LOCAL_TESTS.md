# Local engineering evidence (2026-10-05)

Environment: Windows, Python3.9.25, torch2.6.0+cu124, pinned source Ultralytics8.3.0.
Tests below executed on CPU. Server is a different environment; CUDA check and
real batch16/640 + full batch32 val smoke have NOT been executed here.
No remote access, no formal training launched.

## Executed

1. `test_package.py`: all five tests passed. Independent per-head attention
   reference, manually enumerated adaptive pool boundaries, independent input
   gradients, remote-query influence, zero-start identity, two-step loss-only
   updates, BF16 stored-module exact values vs quantized FP32 oracle, source
   preservation, RNG isolation, odd/rectangular full model and profile arithmetic.
2. Full `run_e52.py --mode check --device cpu`: passed.
   Report: `yolo11_project/comparison_reports/E52_CHECK_20261005_165336_778587/`.
   Initial original loss/common gradients, all 27 new branch groups updated,
   original E40 routes, actual model4 stride8 path and four-scale reach, empty GT,
   trained E40 bypass identity fixture, fresh-process reload/fuse, standard
   Predictor, original TAL FP32/FP16 equivalence and real FP32 loss/update tests,
   official evaluator units and actual640 profiler all passed.
3. Two-epoch/four-image synthetic standard Trainer integration: passed.
   Report: `yolo11_project/comparison_reports/E52_LOCAL-INTEGRATION_20261005_165648_055298/`.
   Includes real Trainer optimizer/EMA/save/final validation, both best/last reload,
   post-save rectangular raw/final predictions and four-image official scoring.
   This is only an engineering integration fixture, NOT a performance probe.

Initial integration attempt could not access the installed pandas dependency
inside the local sandbox; the identical code rerun outside the sandbox passed.
No fake pandas module, loss change, altered server configuration or checks bypass.

## Resource accounting

- Raw E40: 9,854,408 parameters; E52: 9,926,152; difference 71,744.
- E40 major GFLOPs640: 39.926528.
- E52 major GFLOPs640: 40.764135424; difference 0.837607424, about +2.10%.
- Full query grid80x80=6400; pooled K/V49; actual original backbone site4C256.
- THOP branch accounting independently agrees with PyTorch conv/mm/bmm FLOPs.
- Normal YOLO console summary omits functional new-branch work. Norm, softmax,
  pool, GELU, traffic are outside this MAC-style measure. No latency claim.

The unit objective is amplified solely so norm weight updates exceed FP32 ULP;
formal original loss is never amplified. Whole-graph check uses a disposable
active fixture after proving native zero-start update. Neither fixture is used
to initialize formal training; fresh model gets exact zero output projection.
