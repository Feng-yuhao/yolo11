# Supplementary Material S1: grouped-enhancement placement

[Download the complete S1 archive](https://github.com/Feng-yuhao/yolo11/releases/download/ivc-supplement-s1-v1.0.0/Supplementary_Material_S1_Placement.zip) · [Supplementary release](https://github.com/Feng-yuhao/yolo11/releases/tag/ivc-supplement-s1-v1.0.0) · [Unrounded comparison CSV](../results/supplementary/s1_placement_validation_seed1.csv) · [Archive manifest](../manifests/supplementary_s1.json)

## Scope

S1 supports the placement discussion in Section 4.3 of the revised manuscript. It preserves the additional archived E61 **seed-1 validation** comparison, rather than a new three-seed test-dev ablation. The main reproduction release, `ivc-repro-v1.0.0`, retains its original scope of 27 source checkpoints and 45 prediction payloads.

| Grouped enhancement | Split | Seed | AP (%) | AP50 (%) | AP75 (%) | Parameters |
|---|---|---:|---:|---:|---:|---:|
| P2/P3 | VisDrone2019 validation | 1 | 26.969156 | 46.395205 | 26.567837 | 11,173,112 |
| P2/P3/P4/P5 | VisDrone2019 validation | 1 | 26.674089 | 46.036132 | 26.199821 | 11,322,584 |

The all-level extension changes AP by **−0.29506763455268725 percentage points** and adds **149,472 parameters**. The shared encoding width remains 96. The P2/P3 units, encoding chain, spatial router, original prediction towers, losses, 640 input setting, batch size 16, and 200-epoch recipe are retained. The original E60 model is initialized first; the extra P4/P5 units use an isolated initialization stream. Both configurations train from the original pretrained initialization rather than fine-tuning an E60 checkpoint.

This exploratory comparison motivates retaining the reported P2/P3 placement. It does **not** establish a general relationship between object size and grouped-enhancement benefit, or a multi-seed ranking. P4/P5 still retain multi-depth fusion in the final model. No training, inference, or AP scoring was rerun during this supplementary release preparation.

## What is in the ZIP?

- `README_EN.md`: provenance, result interpretation, archive map, and reproduction limits.
- `placement_validation_seed1.csv`: both configurations with unrounded scores.
- `comparison.json`, `official_metrics.json`, and `REPORT.md`: original scores and comparison records.
- `training.json`, `training_completion.json`, `completion.json`, and `args.yaml`: original training settings and completion records.
- `actual_source.zip`: the 31-file archived implementation snapshot, including model definition, evaluation code, original launch documentation, and license.
- `E61_seed1_best.pt`: the selected all-level candidate checkpoint.
- `area_predictions.json`: the original saved validation predictions.
- `SHA256SUMS.txt`: checksums for the archive payloads.

Unzip the outer archive, then `actual_source.zip` to inspect the original implementation. The historical launcher retains reference-report prerequisites and absolute server paths; read its archived documentation and map paths to your own environment. The checkpoint uses the original Python module names. The main E60 seed-1 reference checkpoint is available from the [main reproduction release](https://github.com/Feng-yuhao/yolo11/releases/tag/ivc-repro-v1.0.0). Obtain benchmark images and ground-truth annotations separately from the official provider.

Archive size: **32,242,552 bytes**. Archive SHA-256:

```text
8f7c0785f1d81ef345b31d37c6df9e703d8c6506b369462a953e3b172595a95f
```

Candidate checkpoint SHA-256, verified against the original comparison report:

```text
5560523f4e1b88213e8f8e90503ecb1be58f5926424cb45d8fb8dae65e548c9e
```

## 中文说明

这份 S1 对应修订稿第4.3节“为什么只在 P2/P3 使用分组残差增强”的补充记录。它来自原有 E61 实验备份，**不是新跑的实验，也不是表2中的三种子 test-dev 消融**。

把同样的分组单元扩展到 P4/P5 后，单种子验证集 AP 从26.97变为26.67，参数增加149,472。该结果用于解释当前配置选择，不能据此断言“小目标普遍更需要分组残差”。最终模型的 P4/P5 仍保留多深度融合。

完整包包括原始代码快照、配置、指标报告、选定权重、保存的预测和校验值。上方 CSV 可直接在线查看；下载 ZIP 后可查看全部原始材料。主实验材料继续放在 `ivc-repro-v1.0.0`，S1 使用单独的补充版本，不改变主实验的数量统计。投稿时，仍需把同一份 S1 ZIP 作为 Supplementary material 随正文提交。
