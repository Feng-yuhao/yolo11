# Local validation / 2026-10-17 / Asia/Shanghai

No remote connection, server launch, formal200-epoch training, real VisDrone
evaluation or actual shutdown was performed by the assistant.

## Executed native integration

Windows/Python3.9.25/torch2.6.0+cu124/RTX4060 Laptop8GiB; exact pinned
Ultralytics8.4.177 source, private128px/b16,48train/35val/9test synthetic images.
`audit/yolo26_seed3_serial/local_cuda_v1/PRIVATE_LOCAL_VALIDATION.json` passed.

- Actual FP32 and AMP engineering updates, both native heads/original loss and
  common-gradient equivalence, one2one-detach boundary, FP32 sidecar values,
  all52 gradient/parameter-update groups and fresh reload/fuse checks.
- Both native smoke loops completed at least8 actual finite optimizer updates;
  overflow skips were not counted. E60 updated all52 groups. Each native b32
  validation processed all35 fixture images while model/optimizer/EMA stayed
  resident; checkpoint save and independent8-image prediction passed.
- Both candidates completed TWO native epochs from the original official
  pretrained checkpoint, with seed3 args, unique best/last/results/args and final
  best validation. Original model construction and initializations stayed frozen.
- FOUR separate processes loaded the resulting best checkpoints and performed
  original prediction and D27 scoring: baseline/E60 x validation/test-dev.
  Inverse coordinate and raw-output non-mutation checks were retained.
- The whole controller completed in serial order with gates tied to source,
  data, seed3 and layout. It was explicitly PRIVATE, without shutdown. AP0 on
  synthetic fixtures is not a research result or an expected real-data gain.

## CPU / mocked regressions

`python -m unittest -q test_local test_seed3` passed all25 test methods,
including additional parameterized failure/shutdown scenarios. It checks source-byte freezing,
seed-only option differences, old-output isolation, offline font deployment,
stale/private/wrong-layout gate rejection, all52 update requirements, changed
best/prediction rejection, and serial success/partial failure.

Added checks ensure the named server entry explicitly includes --auto-shutdown;
the helper is dispatched only AFTER all four scoring attempts and copied logs;
both normal and failed termination preserve existing outputs; complete status
with missing scoring cannot power off; artifact changes abort countdown;
cancel/new session/busy GPU prevent unsafe requests; the AutoDL no-shebang helper
is invoked through bash. All shutdown commands, process/GPU state and Linux
locking in these tests are MOCKED. No real shutdown was invoked.

Available fixed seed2 results are included regardless of sign only when index,
training protocol, checkpoint, four scores and prediction SHA validate. Missing
or incompatible seed2 is explicitly flagged; no fabricated3-seed mean. Tests
include a negative seed2 and missing/mismatched references.

## Still required on server

The user must execute the packaged all entry in the real yolo26 environment.
Only its fresh CUDA check and true640/b16/full548-image b32 smoke establish the
server memory/engineering gates. Private tests are not formal200 results or
proof of accuracy/stability. Local documentation/tests added after the native
run do not change the frozen numerical/model/worker protocol. The final upload
manifest/ZIP and all unit tests are revalidated after packaging.

AutoDL command return is not proof of actual power-off or stopped billing.
