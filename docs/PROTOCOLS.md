# Evaluation protocols and interpretation

All paper comparisons are paired by source training seed 1, 2, and 3. Report means and sample SD. Paired differences use unrounded scores and are expressed in percentage points. Three runs provide a descriptive variability estimate, not a significance test.

The original native validation fitness selects checkpoints. Test-dev was examined during development, so it is not described as a completely untouched blind test. Native training mAP is distinct from the fixed VisDrone v1.0.4-port final score.

The main full-model gain is +0.94 pp on VisDrone test-dev. Against uniform fusion the gain is +0.20 pp. Against task-shared routing it is about +0.01 pp, with differences of mixed sign. The data do not isolate a consistently beneficial task-specific weighting effect. Removing grouped units also removes parameters/computation, so that comparison evaluates the branch as a whole.

The TD reference is a six-layer, width-64 TOOD task-decomposition adaptation at prediction-tower inputs. It shares encoders/TD parameters across levels and has level-specific input/output adapters. It is not a reproduction of the complete TOOD detector, prediction alignment, or original losses. The original author commit and Apache-2.0 source are preserved.

Frozen transfer uses all official target test samples and the six source-val-selected E1/E60 checkpoints without target fitting, BN updates, threshold selection or checkpoint reselection. Map scores per anchor before best-class selection and class-aware NMS:

- Songdo: car=max(source car,van), bus=bus, truck=truck, motorcycle=motor.
- DIOR vehicle: max(source car,van,truck,bus).
- UAVDT: car=max(source car,van), truck=truck, bus=bus.

Target scores are interpreted separately. Songdo mean AP gains 1.08 pp with one negative seed; DIOR vehicle AP gains 0.72 pp; UAVDT COCO-style AP gains 0.15 pp while native AP70 falls 0.14 pp. Do not average these targets into one benchmark or describe category-mapped transfer as novel-class/open-vocabulary detection.

Unfused complexity counts describe Conv/Linear operations and the separately counted TD torch.bmm. They exclude normalization, activation, element-wise fusion, decoding and other uncounted operations. Different parameter/computation budgets and these counts do not establish deployment latency or memory superiority.
