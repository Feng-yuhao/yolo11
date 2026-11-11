# Local engineering validation / 2026-10-16

Scope: new read-only evaluation package. No server connection, training, model
edits, real test-dev scoring, or shutdown was performed by the assistant.

Executed with Windows/Python3.9.25/torch2.6.0+cu124/RTX4060 Laptop, using the
frozen Ultralytics8.4.177 source and both actual downloaded YOLO26 seed1 best.pt:

- Nine unit regressions passed: fixed protocol; missing/wrong checkpoint;
  incomplete original training refusal; validation prediction mismatch refusal;
  expected dataset count restored after an exception; package manifests;
  rejection of a training command; mocked check/smoke/all success and worker
  failure, serial order and complete/failed state with no automatic shutdown.
- Both exact completed checkpoint SHA256 values matched original200-epoch
  training and official validation records before local model loading.
- Synthetic eight-image fixture, square and odd-sized rectangular images,
 160px input, original custom head loading and topology/FP32 sidecar checks.
- Both actual trained models used the frozen audited Predictor, original
  one2many+NMS/max500, independent inverse coordinate check, and raw tensor
  immutability check; all eight images exported and scored with frozen D27.
- New model reload + fused prediction repeated for BOTH checkpoints, original
  image/class/count/order and box/score values reproduced within strict FP32
  .002px/2e-6 tolerances. No checkpoint was modified or written back.

Private outputs: audit/yolo26_testdev_s1/local_cuda_v1 and subsequent v2;
PRIVATE_LOCAL_VALIDATION.json is explicitly NOT a server receipt or AP result.
Synthetic scores are not research results. Controller tests are mocks only.

V1 found a local settings directory did not yet exist, causing Ultralytics to
create a workspace settings fallback. The directory is now explicitly created
before import in the local fixture and server runtime. No inference/model or
numerical tolerance changed. V2 is the final regression run.

Server still must execute all: exact yolo26/Python3.10/torch2.5.1 environment,
actual source/file hashes, original548-val/1610-test image/annotation audits,
first8 REAL validation prediction reproduction for each model, real640/b4
smoke, and complete serial test-dev scoring. Local160px tests do not establish
server memory behavior, real-data score, or accuracy gains.

Inference/scoring/custom module source files are byte-for-byte copies of the
already executed seed1 training/evaluation package. New wrapper code is local;
existing YOLO11/26 code, training outputs and model state remain unchanged.
