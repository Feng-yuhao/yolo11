# ST-MDRF for object detection in UAV imagery

Code and original experiment records for **Object detection in UAV imagery with YOLO11s-P2 using multi-depth readout and task-adaptive fusion**.

Yuhao Feng and Xuyang Wang, Lanzhou University of Technology. Corresponding author: Xuyang Wang, wangxy@lut.edu.cn.

## Paper materials

- **Main experiments:** [fixed release ivc-repro-v1.0.0](https://github.com/Feng-yuhao/yolo11/releases/tag/ivc-repro-v1.0.0). It contains 27 validation-selected source checkpoints (nine configurations × three seeds) and 45 saved prediction payloads, including 18 frozen-weight target evaluations.
- **Placement-validation record (Section 4.3):** [configuration and unrounded results](docs/SUPPLEMENTARY_S1.md), [original-record archive](https://github.com/Feng-yuhao/yolo11/releases/download/ivc-supplement-s1-v1.0.0/Supplementary_Material_S1_Placement.zip). The manuscript reports this seed-1 validation comparison directly; the archive supplies its underlying code and experiment records, separately from the three-seed test-dev means.
- **Datasets:** follow [data/README.md](data/README.md). Images and raw benchmark annotations are obtained from their official providers; the repository supplies preparation code, split metadata, fingerprints, and evaluation rules.
- **Evaluation details:** [protocols](docs/PROTOCOLS.md) and [validation scope](docs/VALIDATION.md). Original scores and training records are in `results/`; artifact sizes and checksums are in `manifests/`.

| Configuration | VisDrone2019 test-dev AP (%) |
|---|---:|
| YOLO11s | 21.23 ± 0.31 |
| YOLO11s-P2 | 23.46 ± 0.23 |
| YOLO11s-P2 + ST-MDRF | 24.40 ± 0.14 |
| YOLO26s-P2 | 22.76 ± 0.08 |
| YOLO26s-P2 + ST-MDRF | 23.62 ± 0.23 |

Values are mean ± sample SD for seeds 1, 2, and 3. Task-specific and task-shared routing have comparable mean AP, with paired differences of both signs. The grouped-branch ablation also changes capacity. The test-dev split was examined during method development; see the protocols for interpretation.

## Installation

Use Python 3.10 and a separate environment for each Ultralytics family.

```bash
git clone --depth 1 https://github.com/Feng-yuhao/yolo11.git
cd yolo11
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environment/requirements-yolo11.txt
# For YOLO26, use requirements-yolo26.txt in a separate environment.
```

`vendor/` contains the frozen Ultralytics-based sources used in the original experiments: 8.3.0 for YOLO11 and 8.4.177 for YOLO26. The runtime selects these local sources; do not install a replacement Ultralytics version. The packages include general upstream functionality beyond the paper's models. These are third-party dependencies, not additional proposed methods. Attribution and source details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The paper-specific implementations are in `implementations/`, model-loading adapters in `st_mdrf/`, and runnable commands in `tools/`.

## Reproduction

Reconstruct the reported tables and check repository files without datasets or PyTorch:

```bash
python tools/reproduce_tables.py
python tools/verify_release.py
```

The tables are written to `outputs/paper_tables/`. Prepare datasets using [data/README.md](data/README.md), then download the required artifacts and evaluate:

```bash
python tools/download.py --group weights --model st-mdrf --seed 1
python tools/download.py --group predictions --list
python tools/predict_visdrone.py --model st-mdrf --seed 1 --dataset-root /data/VisDrone2019/VisDrone2019-DET-test-dev --split test-dev --device 0 --output outputs/stmdrf_s1_testdev.json
python tools/score_visdrone.py --dataset-root /data/VisDrone2019/VisDrone2019-DET-test-dev --split test-dev --predictions outputs/stmdrf_s1_testdev.json --output outputs/stmdrf_s1_metrics.json
```

Model IDs: `yolo11s`, `yolo11s-p2`, `st-mdrf`, `no-group`, `uniform`, `shared`, `tood-td`, `yolo26s-p2`, and `yolo26s-p2-st-mdrf`. Each has seeds 1–3. `tood-td` is the task-decomposition adaptation, not a full TOOD detector.

For frozen-weight transfer, use `tools/predict_transfer.py` and `tools/score_transfer.py` with the prepared target annotations. Class-score mapping is performed before NMS; target-specific rules and commands are documented in [data instructions](data/README.md) and [protocols](docs/PROTOCOLS.md). Each command supports `--help`.

Training uses the preserved 200-epoch recipe. Download initialization weights first:

```bash
python tools/download.py --group pretrained
python tools/train.py --model st-mdrf --seed 1 --data /data/VisDrone_prepared/data.yaml --pretrained weights/pretrained/yolo11s.pt --output outputs/train_stmdrf_s1
# Add --execute to start training. Without it, the command only prints the recipe.
```

Use `yolo26s.pt` for YOLO26. Historical module names and source snapshots are retained for checkpoint compatibility; public commands do not invoke automatic server shutdown.

Rebuild the class plot and inspect computation:

```bash
python tools/plot_per_class.py
python tools/profile_model.py --model st-mdrf --seed 1 --output outputs/stmdrf_complexity.json
```

Figure sources are in `assets/figures/`. Reported computation counts exclude some operations and are not latency measurements. Packaging checks did not rerun complete training or the full AP benchmark; see [validation details](docs/VALIDATION.md).

## Citation and license

See [CITATION.cff](CITATION.cff), [LICENSE](LICENSE), and [third-party notices](THIRD_PARTY_NOTICES.md). Dataset terms remain with their providers. Older archive/work branches and commit history preserve earlier exploration; the versioned main release and placement-validation archive define this manuscript's shared materials.
