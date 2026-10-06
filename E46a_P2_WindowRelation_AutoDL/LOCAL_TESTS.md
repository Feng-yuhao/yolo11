# Executed local checks (2026-10-02)

Host: Windows, Python3.9.25, torch2.6.0, CPU execution, pinned Ultralytics8.3.0.
This does NOT claim CUDA, AutoDL memory or full200-epoch accuracy validation.

Passed:

- Independent partition/reversal, padding and cyclic-wrap mask oracle for
  16x24,13x19,1x1,5x7,8x8; both shift0 andshift4.
- Independent per-window/per-head QKV, relative bias, key softmax and value
  aggregation oracle; nonfinite forward/backward rejected.
- Cross-position value response; separate cls/reg output paths; both paths
  read the original P2 input, without modifying it.
- Zero context outputs preserve NONZERO E40 outputs exactly, not just E1 zeros.
- All context parameters have finite nonzero loss gradients and real SGD updates
  in an active-branch unit, with NO weight decay or momentum.
- Pure-bfloat16 parameters retain dtype and yield exact FP32-context values
  compared with an independently promoted copy. CUDA float16 test is in check.
- Exact inherited E1 and E40 initialization, preserved CPU RNG; initial square
  and rectangular raw/decoded equality; unchanged original detection criterion.
- Finite loss, numeric base gradient equivalence,19 staged branch gradient and
  actual-update groups; empty labels; actual cv2/cv3 routing and P4/P5 bypass.
- Active-branch save/reload/fuse, separate fresh process, standard YOLO Predictor.
- TAL assignment equivalence across random/dense/empty/ties cases; real P2
  loss/gradient/two-update equivalence; official evaluator and geometry units.
- Native DetectionTrainer two-epoch FOUR-IMAGE SYNTHETIC integration, optimizer
  coverage, EMA best/last checkpoints, half-storage->float reload, fused
  inference, rectangular predictions, official evaluator fixture.

Evidence:

- comparison_reports/E46_CHECK_20261002_135328_110939/check.json
- comparison_reports/E46_LOCAL-INTEGRATION_20261002_135147_388662/integration.json

Initial integration attempts were blocked by sandbox access to local pandas
test dependencies, before Trainer CSV serialization. Real pandas2.2.3/pytz
were downloaded into tmp/e46_test_deps, then the unchanged Trainer was tested
with access permission. No server environment or global packages were changed.
Those failed test receipts are retained, not passed off as successful tests.

Server still REQUIRED: CUDA check plus real8-successful-update batch16/640 and
full548-image batch32/FP16 validation smoke. Local unit tests are not GPU gates.
