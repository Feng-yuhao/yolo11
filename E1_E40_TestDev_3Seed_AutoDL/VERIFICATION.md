# Local engineering verification — 2026-10-04

Executed, not inferred:

- 9 unit tests passed: pinned files, original scorer, dataset count/mapping/content binding,
  equivalent class-0 filtering, fail-closed cache, interrupted writer, NMS timeout rejection,
  immutable manifest/protocol, paired mean/sample-standard-deviation.
- All six real downloaded checkpoints matched their recorded SHA256 and loaded with the
  intended class, original four scales/strides, ten class names, and E40 branch parameter count.
- E1/E40 seed1 predictions on 8 nonempty saved-mosaic fixtures were exactly equal to the
  unchanged standard DetectionPredictor, including rectangular and mixed-shape batch4:
  CPU 2428/1868 final boxes; local CUDA 2427/1871 final boxes.
  CPU and CUDA are not asserted bit-identical to each other; each matches its same-device reference.
- Local CUDA fixture used a 2GiB allocator cap; peaks were about 0.75GiB (E1) and 0.81GiB (E40).
  This is NOT the AutoDL CUDA coexistence smoke and is NOT a worst-case test-dev memory bound.
- Synthetic 1610-image six-model CPU-score CLI and repeated recovery passed, including
  expected 90/100 synthetic AP and paired +10. No actual test-dev AP was calculated locally.
- Python compilation passed. The shell script is LF/UTF-8, without CRLF; no local Bash
  runtime was available for executing the Linux wrapper. Python subprocess/data paths were tested.

Local environment: Windows, Python in yolov8 env, PyTorch 2.6.0+cu124, NumPy 2.0.2,
project Ultralytics 8.3.0. AutoDL's actual PyTorch/CUDA version and source hash are recorded
by server check and bound to its own smoke receipt.

Before running the full test-dev set, execute server check and six-weight CUDA smoke.
No remote process or existing training job was launched, killed, or changed during this work.
