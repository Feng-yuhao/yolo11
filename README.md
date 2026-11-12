# ST-MDRF for object detection in UAV imagery

Code and experiment artifacts for **Object detection in UAV imagery with YOLO11s-P2 using multi-depth readout and task-adaptive fusion**.

Authors: Yuhao Feng and Xuyang Wang. School of Computer Science and Artificial Intelligence, Lanzhou University of Technology. Corresponding author: Xuyang Wang, wangxy@lut.edu.cn.

ST-MDRF retains four encoding-depth readouts at each pyramid level, enhances intermediate readouts at P2/P3 with grouped residual units, and fuses them with spatially varying classification/regression weights. The original prediction towers remain in use.

## Supplementary placement experiment (S1)

The additional archived **seed-1 validation** comparison of grouped enhancement at P2/P3 versus P2/P3/P4/P5 is documented in [Supplementary Material S1](docs/SUPPLEMENTARY_S1.md). [View the unrounded results](results/supplementary/s1_placement_validation_seed1.csv) or [download its complete code, checkpoint, predictions, and original records](https://github.com/Feng-yuhao/yolo11/releases/download/ivc-supplement-s1-v1.0.0/Supplementary_Material_S1_Placement.zip). S1 is separate from the main three-seed test-dev experiments and does not change the main release's stated counts.

## Scope and results

The main reproduction release contains the preserved implementations and original results for **nine configurations and three training seeds (27 source training runs)**, plus **18 frozen-weight target evaluations** on Songdo Vision v2, the DIOR vehicle task, and UAVDT. The target evaluations reuse the six YOLO11s-P2/ST-MDRF source checkpoints; they do not retrain target models.

| Configuration | VisDrone2019 test-dev AP (%) |
|---|---:|
| YOLO11s | 21.23 ± 0.31 |
| YOLO11s-P2 | 23.46 ± 0.23 |
| YOLO11s-P2 + ST-MDRF | 24.40 ± 0.14 |
| YOLO26s-P2 | 22.76 ± 0.08 |
| YOLO26s-P2 + ST-MDRF | 23.62 ± 0.23 |

Values are mean ± sample SD across training seeds 1, 2, and 3. The full configuration gains 0.94 percentage points over YOLO11s-P2, 0.20 over uniform fusion, and 0.01 over task-shared routing. The latter difference changes sign across seeds. The grouped branch adds capacity, so its ablation does not isolate hierarchy from capacity. The reported test-dev split was examined during method development. See [protocols and interpretation](docs/PROTOCOLS.md).

## Start with the recorded results

The table reproduction step only needs standard Python; it does not download datasets or invoke PyTorch:

```bash
git clone --depth 1 https://github.com/Feng-yuhao/yolo11.git
cd yolo11
python tools/reproduce_tables.py
python tools/verify_release.py
```

`outputs/paper_tables/tables.md` and `statistics.json` are regenerated from unrounded records. Original per-run JSON is under `results/raw`, and training arguments/native logs are under `results/training`. The checkpoint and prediction manifests bind artifact names, sizes and SHA256 hashes. Native YOLO monitoring metrics and the final VisDrone evaluator are distinct.

## Install the model/evaluation environment

Use a separate Python 3.10 environment and the original PyTorch/CUDA family:

```bash
python -m venv .venv
# Activate this environment using the command appropriate for your shell.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environment/requirements-yolo11.txt
# In a separate YOLO26 environment, use requirements-yolo26.txt instead.
```

The repository supplies frozen Ultralytics 8.3.0 and 8.4.177 sources in separate directories. The tools select the matching version in a fresh process; do not replace both with one current Ultralytics installation. Historical environment evidence is in `environment/*.json`; the requirements file is a portable dependency specification, not a claim that a complete new-machine 200-epoch run has been executed.

## Models, checkpoints and predictions

| CLI model ID | Role |
|---|---|
| `yolo11s` | Original three-scale baseline |
| `yolo11s-p2` | Direct four-scale baseline |
| `st-mdrf` | Complete proposed module |
| `no-group` | Remove grouped readout enhancement |
| `uniform` | Uniform depth fusion |
| `shared` | Task-shared spatial depth routing |
| `tood-td` | TOOD task-decomposition adaptation, not a full TOOD detector |
| `yolo26s-p2` | YOLO26 P2 baseline |
| `yolo26s-p2-st-mdrf` | YOLO26 P2 with ST-MDRF |

Each ID has seeds 1, 2, and 3. All 27 original validation-selected checkpoints and seven lossless prediction archives are available from the [versioned release](https://github.com/Feng-yuhao/yolo11/releases/tag/ivc-repro-v1.0.0). Initialization weights and native AMP-check auxiliary weights are supplied separately and clearly labeled.

```bash
python tools/download.py --group weights --model st-mdrf --seed 1
python tools/download.py --group predictions --list
python tools/download.py --group predictions --asset predictions_songdo.zip --output outputs/artifacts
python tools/download.py --group pretrained
```

The prediction archives contain the original JSON or gzip bytes, including all reported seeds and negative differences. Their manifests preserve the original file hashes. They contain no benchmark images or raw benchmark GT. Extract them with a ZIP tool and use the matching official/prepared GT for scoring.

## Prepare the four datasets

Download the exact official versions and follow [data instructions](data/README.md). Images are obtained from the dataset providers. This repository supplies split membership, annotation fingerprints, target image IDs/dimensions, conversion/verification code and scoring rules. The original dataset files are not relicensed as this project's code.

```bash
python tools/prepare_data.py visdrone --raw-root /data/VisDrone2019 --output /data/VisDrone_prepared
python tools/prepare_data.py songdo --raw-root /data/Songdo_v2/test --output /data/Songdo_prepared
python tools/prepare_data.py dior --images-root /data/DIOR/JPEGImages-test --xml-root "/data/DIOR/Annotations/Horizontal Bounding Boxes" --split-file /data/DIOR/ImageSets/Main/test.txt --output /data/DIOR_prepared
python tools/prepare_data.py uavdt --images-root /data/UAVDT_frames --toolkit /data/UAV-benchmark-MOTD_v1.0 --output /data/UAVDT_prepared
```

All preparation commands write to a new destination and preserve image bytes. `--image-mode hardlink` can be used when input/output are on the same filesystem; the default copies images. VisDrone training labels must match the published original combined fingerprint before training is admitted. Dataset preparation for the complete VisDrone training image set has not been rerun as part of this release preparation; the original fingerprint is included to detect reconstruction differences.

Verify a prepared target against the released image population, dimensions, available image hashes and canonical GT:

```bash
python tools/check_data.py --target songdo --dataset-root /data/Songdo_prepared
```

Use `--target dior` or `--target uavdt` with the corresponding prepared root.

## Re-evaluate a trained source model

```bash
python tools/predict_visdrone.py --model st-mdrf --seed 1 --dataset-root /data/VisDrone2019/VisDrone2019-DET-test-dev --split test-dev --device 0 --output outputs/stmdrf_s1_testdev.json
python tools/score_visdrone.py --dataset-root /data/VisDrone2019/VisDrone2019-DET-test-dev --split test-dev --predictions outputs/stmdrf_s1_testdev.json --output outputs/stmdrf_s1_metrics.json
```

Prediction uses FP32, maximum image side 640, batch 4, confidence 0.001, NMS IoU 0.7 and up to 500 predictions per image. YOLO26 uses the trained one-to-many branch followed by NMS, while its native dual-branch training is retained.

For target transfer, keep the ten-class source checkpoint and map scores **before** NMS:

```bash
python tools/predict_transfer.py --target songdo --model st-mdrf --seed 1 --dataset-root /data/Songdo_prepared --gt /data/Songdo_prepared/annotations_coco/test.json --device 0 --output outputs/songdo_stmdrf_s1.json
python tools/score_transfer.py --target songdo --gt /data/Songdo_prepared/annotations_coco/test.json --predictions outputs/songdo_stmdrf_s1.json --output outputs/songdo_stmdrf_s1_scores
```

Use `--target dior` with `annotations_coco/test_vehicle.json`. UAVDT additionally requires `--ignore .../annotations/ignore_by_image.json` and reports both the custom COCO-style AP and the native class-agnostic AP70 supplement. Full UAVDT prediction/COCO evaluation requires substantial disk/RAM; recorded predictions and original metrics can be inspected independently.

## Train from the original initialization

Training is explicit. Omitting `--execute` prints the intended run without starting training:

```bash
python tools/train.py --model st-mdrf --seed 1 --data /data/VisDrone_prepared/data.yaml --pretrained weights/pretrained/yolo11s.pt --output outputs/train_stmdrf_s1
# Add --execute to run the archived 200-epoch recipe on CUDA.
```

Use `weights/pretrained/yolo26s.pt` for the YOLO26 configurations. Each seed/configuration has its own archived recipe and preserved implementation. The source files under `implementations/` retain original names for correct checkpoint deserialization; their historical AutoDL paths/guards are provenance, while `tools/` provides the portable entry points. No automatic shutdown is invoked by the public tools.

## Rebuild figures and check computation

```bash
python tools/plot_per_class.py
python tools/profile_model.py --model st-mdrf --seed 1 --output outputs/stmdrf_complexity.json
```

Figure sources are in `assets/figures`. Complexity counts are unfused Conv/Linear MACs plus a separately reported TD matrix-recombination term for a 1×3×640×640 input. They exclude other operations and are not latency measurements.

## Validation and provenance

The release preparation verified all 27 checkpoint hashes and fresh-process CPU loads, nine finite synthetic 128×128 forwards, nine fresh model constructions, exact 640×640 complexity counts for the three Table 3 models, original source-file hashes, and table reconstruction. VisDrone validation-label conversion was compared against the frozen upstream converter; DIOR GT/labels and UAVDT GT/ignore definitions were rebuilt and compared with the original artifacts. [Validation details](docs/VALIDATION.md) describe the checks and their limits. Preserved 200-epoch experiment outputs are historical results, not new training performed during packaging.

See [third-party notices](THIRD_PARTY_NOTICES.md) and [citation metadata](CITATION.cff). The code is distributed under AGPL-3.0; separately licensed upstream material retains its own notices, and dataset licenses remain with their providers. The public repository's previous contents remain recoverable from Git history and the pre-release archive branch.
