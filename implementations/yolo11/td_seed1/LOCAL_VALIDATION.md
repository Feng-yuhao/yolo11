# Local validation boundary

Executed on Windows, Python3.9.25 / torch2.6.0+cu124 (RTX4060 Laptop),
with the frozen local Ultralytics8.3.0 source. This is NOT the server environment.

- CPU full engineering check: passed, audit/TOOD_TD_LOCAL_cpu_20261017_162405_586896.
- CUDA full engineering check including cold AMP, all24 groups, original TAL
  FP32/FP16 loss/gradient/update equivalence, save/reload/fuse/fresh process,
  actual cv2/cv3 task inputs: passed,
  audit/TOOD_TD_LOCAL_cuda0_20261017_162826_668735.
- Double precision forward/backward equivalence with the author's extracted TD
  class and an independent gated-feature/dense-convolution oracle: passed.
- Two native Trainer epochs, actual optimizer coverage, checkpoint/EMA-half,
  reload, standard predictor, inverse letterbox and official toy export: passed,
  audit/TOOD_TD_LOCAL_integration_20261017_162559_714362.
  The first local integration attempt exposed missing local pandas; rerun used
  the existing real pandas2.2.3 dependency directory. No Trainer mock or altered
  loss was substituted. Server checks now fail early if pandas is absent.
-30 stdlib control/shutdown tests: passed (test logs in workspace audit).
  Includes all7 stages, every failed stage, no-shebang bash invocation, parent
  exit/lock waits, incomplete results, other GPU/Python jobs, cancellation,
  countdown reset and duplicate request protection. ALL poweroff runners fake.
- Original E1/E60 same-seed cached val/test SHA/protocol evidence: validated.

Unfused parameters: E1 9,678,968; TD-adapt 10,153,716; new474,748.
Executed Conv/Linear + TD bmm: E1 16.9381632 GMAC, TD-adapt 27.03400896,
added10.09584576 GMAC at640, batch1. Not full-operator FLOPs.
No performance result is claimed. No remote job or real shutdown was executed.
Only a NEW server CUDA check and real640/b16 + full548/b32 smoke authorize200e.
