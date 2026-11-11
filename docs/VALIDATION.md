# Release validation

Preparation checks are recorded separately from historical experiment results.

- All 27 original selected checkpoints were found locally, SHA256-bound to the release manifest, and loaded in fresh processes on CPU with the appropriate frozen implementation. Saved training-seed metadata and expected parameter counts were checked.
- All nine configurations produced finite outputs for a synthetic 1×3×128×128 CPU input. These are engineering forwards, not detection-quality measurements.
- The 21 YOLO26 source pins match the original training-source record. Preserved implementation files have a separate byte-level source manifest.
- All 27 test-dev metric triples match the unrounded result index, and the paper's four tables and paired transfer statistics can be regenerated without PyTorch.
- Forty-five original test/transfer prediction files were packed without changing their bytes. Seven ZIPs passed CRC verification; each payload and archive has a SHA256 manifest. Remote Release asset digests/sizes are checked during upload.
- Nine fresh model constructions from the original initialization produce the same parameter counts as the trained architectures. Fresh unfused 640×640 profiling exactly reproduces all three Table 3 parameter and MAC counts, including the separate TD matrix-recombination term. See [construction and complexity checks](construction_and_complexity_validation.json).
- Conversion of all 548 VisDrone validation annotations produces the same 38,759 positive labels as the independent frozen upstream converter, using Linux label line endings from the original training environment. DIOR GT and all label files were rebuilt for 11,738 images and match the original bytes. UAVDT GT and ignore definitions match the original bytes, all 16,592 frame IDs are in the original order, and strict ignore-boundary behavior was checked. These annotation fixtures did not repeat transport of the entire image populations. See [data checks](data_preparation_validation.json).
- All twelve public command-line entry points accept `--help`; numerical construction/profile checks are separate from those interface checks. See [entry-point checks](public_entrypoint_validation.json) and [checkpoint checks](model_load_validation.json).

The release preparation does not claim a new-machine full 200-epoch training run, a new full 27-model AP benchmark, or a fresh reconstruction of every VisDrone training image/label. The original image files must be obtained from their official providers; the published training-label fingerprint detects data reconstruction drift.
