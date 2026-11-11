# Local engineering validation — 2026-10-24

Executed locally, not on AutoDL. No training, full target evaluation, remote changes or real shutdown was executed for this code delivery.

- Frozen E1/E60 source model files are byte-identical to the previously completed source-only package; all six genuine source val-best checkpoint SHA256 values match.
- Both complete test payloads passed full SHA256, JPEG decode and supervision/YOLO inverse-coordinate checks (DIOR11738, TinyPerson786). Original images and annotations were not changed.
- The final lightweight metadata scanner was independently required to reproduce exactly the earlier full check's per-file size/mtime digest before reusing that full-check receipt. It changes scan cost, not integrity criteria or data.
- 17 engineering unit checks passed: all12 keys, independent NumPy semantic max at every anchor, nonmutation, merge-before-NMS, unmapped class exclusion, nonfinite rejection, landscape/portrait inverse coordinates, perfect/empty scoring, multiple ignored-region detections, background false positives, no-ignore agreement with standard COCO math, lossless exact-float gzip, storage guards, BN fingerprint and negative-seed retention.
- All12 target/checkpoint combinations executed the real bound worker on8 fixed target images at FP32/native640/b4. Inference, mapping/coordinate oracles, unchanged model/BN fingerprints, gzip readback and corresponding scoring passed. TinyPerson exercised both ignore-aware COCO-style and native tiny scoring paths.
- 15 independent shutdown tests passed using injected mocked runners only: complete output proof; missing12th score; failed check/smoke/full job; corrupted predictions; live parent; active lock; replacement session; cancellation; changed helper; helper nonzero; occupied GPU; unknown kernel; wrong population.
- Five controller simulations passed with mocked child processes: exact check→12smokes→12full-tests→helper ordering, early check failure, last scoring failure, saved-file integrity failure, and no opt-in. The Windows-only interpreter-path gate was bypassed only inside these private orchestration mocks, not in deliverable code.
- AST parsing, all frozen model/checkpoint bindings, LF-only Bash text and ZIP CRC verified by build_release.py. No Bash runtime was available locally; actual Linux shell/platform probes and all CUDA smokes are rerun by the server entry point.

Local runtime: Windows / Python3.9 / Torch2.6.0+cu124 / RTX4060 Laptop8GiB. pycocotools2.0.11 was supplied from a private audit dependency directory, without changing the existing environment. This differs from the pinned AutoDL yolo11 runtime and is not a server-memory guarantee.

The author TinyBenchmark v1 matcher is shipped unchanged with its MIT/BSD notices and byte binding. Runtime compatibility substitutions are limited to the deprecated NumPy float alias and linspace integer sample counts. Ignore/IOD thresholds, sorting and matching were not changed.

Private receipts and test harnesses:
`PAPER_EXPERIMENTS/audit/prepare_extra_testsets_20261024/s123_engineering_v2/VALIDATION.json`
`PAPER_EXPERIMENTS/audit/prepare_extra_testsets_20261024/SHUTDOWN_TESTS.json`
`PAPER_EXPERIMENTS/audit/prepare_extra_testsets_20261024/CONTROLLER_TESTS.json`

Initial local dependency and metadata-scan debugging records are preserved separately; failed preflight runs are not target performance evidence. The final executable Python hashes match the12 successful smoke receipts. Only documentation/license additions were made when sealing the final ZIP.
