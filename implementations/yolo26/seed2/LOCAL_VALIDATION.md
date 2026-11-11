# Local engineering validation / 2026-10-16

No remote connection, server training, real VisDrone evaluation, checkpoint
modification, or actual shutdown was executed by the assistant.

## Executed private native integration

Windows / Python3.9.25 / torch2.6.0+cu124 / RTX4060 Laptop8GiB, pinned actual
Ultralytics8.4.177 source. Private128px/b16, 48train/35val/9test synthetic images.
Both models initialized from the same original bundled YOLO26s pretrained SHA.

`audit/yolo26_seed2_serial/local_cuda_v1/PRIVATE_LOCAL_VALIDATION.json` passed:

- CUDA original dual-head loss/common-gradient equivalence, detach audit,
  FP32+AMP all52 gradient/update groups, half sidecar precision, fresh reload/fuse.
- Both native smoke loops, eight actual finite updates (overflow skips not
  counted); E60 all52 groups changed; full35-image native b32 validation;
  square b32 and saved smoke checkpoint standalone8-image prediction passed.
- Baseline TWO COMPLETE native epochs, best/last/results/args and final-best
  validation; then E60 TWO COMPLETE native epochs with corresponding outputs.
- Both trained seed2 checkpoints subsequently loaded in FOUR separate scoring
  processes: baseline val/test and E60 val/test. Independent inverse coordinate
  checks, original NMS and raw non-mutation guard, full private data counts,
  frozen D27 unit checks and actual scoring executed.
- Bound source/package/data/seed/serial-layout gates enforced between stages.
  PRIVATE receipts cannot authorize server runs. No repeated formal seed2
  directory can silently overwrite or retrain old output.
- Whole controller finished in the requested serial order. No training input,
  loss, optimizer, FP32 sidecar implementation or numerical tolerance was changed.

Private synthetic AP0 is NOT a research result or a prediction of real gain.
The private final-rebuild rerun and recovery checks are retained alongside v1.

## Executed unit regressions

Fourteen unittest methods passed, with additional parameterized scenarios:

- Seed1 options exactly equal after removing ONLY seed1 vs seed2; output names
  cannot overwrite seed1; frozen actual implementation source SHA unchanged.
- Original image count restored after errors; stale/local/parallel gates rejected;
  receipt edits rejected; all52 real branch updates required; changed best rejected.
- Negative seed2 retained in both paired means/sample standard deviations.
- Mocked check/smoke/all serial stages, and baseline formal failure followed by
  independent E60+scoring; partial results marked failed, never complete.
- Shutdown mocks: no opt-in -> no command; no-shebang platform helper -> bash;
  normal idle request, cancel marker, busy GPU, changed session. No actual shutdown.
- New isolated settings directory receives the bundled original Arial font;
  frozen native check_font finds that file locally, with networking forbidden.
  This final plotting-asset addition does not change numerical training code.

An initial Windows-only unit run hit a transient `os.replace` WinError5 during
rapid state-file writes. An unchanged complete rerun passed all13 then-existing tests. The
frozen Linux atomic-write implementation and model/training code were not
weakened to hide that event; full native integration passed independently.

## Server still required

User must execute new all/check/smoke with server Python3.10/torch2.5.1+cu121,
real640/b16, full548-image b32 validation, original6471/1610 data and32GiB card.
Only that establishes true memory gates. Local two-epoch tests are not formal
200-epoch training, real AP evidence, or a promise that OOM cannot occur.

Serial28GiB does not change batch16/val32 to fit. Training retains original
native OOM guard and refuses silent batch reduction. Auto-shutdown is optional;
if opted in it verifies saved complete/partial artifacts after parent exit and
idle checks, invokes the observed AutoDL helper via bash, and records request
status. A successful request is not confirmation that platform billing stopped.
