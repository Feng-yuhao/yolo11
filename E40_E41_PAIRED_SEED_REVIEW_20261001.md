# E40 / E41：640 正式两种子配对结果复核

日期：2026-10-01。只复核已同步的结果，没有启动训练或修改远程系统。

## 原始证据

- E40 seed1：`yolo11_project/comparison_reports/E40_RUN_20260929_205658_507620/`
- E41 seed1：`yolo11_project/comparison_reports/E41_RUN_20260930_034452_376576/`
- E40 seed2：`yolo11_project/comparison_reports/E40S2_RUN_20260930_130808_314014/`
- E41 seed2：`yolo11_project/comparison_reports/E41S2_RUN_20260930_130808_373575/`
- E1 seed2：`yolo11_project/comparison_reports/e1_yolo11s_p2add_img640_seed2_20260926_172907_911324/`

两份 seed2 的 training.json 均 completed / 200 epochs，comparison.json 与 official_metrics.json 均 completed。
每份七个结果文件 REPORT.md、training.json、transfer.json、official_metrics.json、comparison.json、gt_rows.json、area_predictions.json 都已同步。
两份运行的 best.pt、last.pt 均已同步；best.pt 实际 SHA256 与官方评分报告一致。
E1 seed2 的权重 SHA256 与配对报告中的 baseline_checkpoint_sha256 一致。
640、batch16、seed2、200轮、优化器、增强、loss 和学习率配置与 E1 seed2 对齐。
设备写法是 E1 的 0 与 E40/E41 的 cuda:0，指同一设备编号。
两份 seed2 模块源码分别与自己的 seed1 模块逐字节相同。

## 官方 VisDrone 指标

所有增益以同 seed 的 E1 为参照，单位 pp。

| 模型 | Seed1 AP | Seed2 AP | Seed1 AP增益 | Seed2 AP增益 | 两seed平均AP增益 |
|---|---:|---:|---:|---:|---:|
| E1 | 25.6312 | 25.4107 | — | — | — |
| E40 | 26.2225 | 26.0639 | +0.5912 | +0.6532 | +0.6222 |
| E41 | 26.2159 | 25.8652 | +0.5847 | +0.4546 | +0.5196 |

| 模型 | Seed1 AP50增益 | Seed2 AP50增益 | Seed1 AP75增益 | Seed2 AP75增益 |
|---|---:|---:|---:|---:|
| E40 | +0.6359 | +0.9018 | +0.8146 | +0.9651 |
| E41 | +0.5803 | +0.5510 | +0.6495 | +0.7535 |

E40 seed2 有8/10类AP提高，下降为van -0.2318、tricycle -0.6640 pp。
E41 seed2 有7/10类AP提高，下降为van -0.2965、tricycle -0.8693、bus -1.4450 pp。
总体增益均为正，不能描述为所有类别均改善。

## 解释与候选状态

E40两个种子都维持约0.6 pp净收益；E41两个种子也为正，但seed2 AP净收益有所收缩。
这是重复的正结果，证据强于单种子排名；两个种子仍不能证明统计显著或保证第三种子/跨数据集复现。
当前优先候选为E40，E41保留为有正结果的另一种路由对照，不因seed2增益缩小而判为失败。
E40/E41属于相同多深度编码框架的替代路由，现有结果不证明两者正交或可直接叠加。
AP75改善不能单独证明raw high-IoU几何缺口已解决，更不能确诊历史唯一根因。
seed2报告没有相对E1 seed2的raw几何差值；不得拿D38的E1 seed1与seed2模型冒充配对几何对照。

下一步建议是优先补E40 seed3，再补与论文贡献对应的训练消融、复杂度/速度和泛化验证。
这是后续建议，没有在本次复核中启动训练，也没有宣布模型已达到录用标准。

## 时间

报告记录的服务器本地训练完成时间（北京时间）：
- E40 seed2：2026-10-01 01:24:14，训练计时约12.26小时。
- E41 seed2：2026-10-01 01:23:27，训练计时约12.24小时。

上述是训练完成时间；官方评分在其后进行，不能将同步文件时间当作评分完成时间。
