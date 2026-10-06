# D47：冻结 E40 的定位质量排序可学习性实验

这是一个缓存候选上的固定终点实验，**不是 200 轮全网训练，不是已证明有效的新论文模块**。
目的：检验 D46 观察到的同实例高 IoU 候选未被优先保留，能否转化成实际 Official AP 收益。

## 固定设计

- 依次使用 E40 seed1 / seed2 现有 val-best，严格校验 checkpoint SHA；原权重全部 eval/frozen。
- 输入：原回归塔末端的 64 通道、分类塔末端的 128 通道、16 个 DFL 分布统计、框尺度、检测层/预测类别及原分数，共 225 维。
- 对照：原 E40；DFL+相同元数据的 33 维排序器；完整 225 维排序器。对照 MLP 宽度均为 64→32→1。
- 原训练集按视频前缀 SHA 固定约 90/10 fit/holdout；不读取 val 来拟合、早停、选 checkpoint 或调阈值。
- 训练集只缓存原分数大于 0.00013545 左右的候选（精确值在 ranking.SUPPORT）；每图最多 512 行。
- 每图最多 256 个 IoU≥.5 候选，优先包含每个 GT 的高 IoU 和高分候选；负样本最多 256 个，含高分及随机负样本。
- 标签：与预测类别相同的非忽略 GT 的连续 IoU；错误类别/无匹配目标的质量为 0。class-0 区域与同类别 score-zero ignored GT 的候选不作负监督。
- 同一 GT 内质量差≥.05的候选对，至多四对/GT，辅助 pairwise 排序损失；不存在跨实例 GT 排序。
- quality BCE 在正/负两组等权平均，pairwise 权重固定为 1；AdamW lr=.001 / wd=.0001，FP32，固定 10 轮，无 AMP，无 best 选择。
- 改分数而不改框：有界 score-logit 残差 `2*tanh(z)`；零输出严格恢复原预测；同一因子作用于十类分数，保留原 argmax 类别。
- 这个统一分数同时用于 conf=.001、原 hard-NMS iou=.7、top500 和官方 AP，不只用在 NMS 内部。
- 四尺度候选均参与，不新增 tiny gate。原 E40 的 P2/P3 结构不变。

DFL-only 不是 E20 的精确复现：没有 NWD 混合标签、不只监督 TAL 正样本、也不联合更新分类塔。
只代表一个受控的输入信息对照。拟合结果不证明完整端到端训练一定有效，也不证明分类是所有错误的根因。

## 安装

上传 ZIP 到 `/root/autodl-tmp` 后：

```bash
cd /root/autodl-tmp
unzip -o D47_E40_FrozenQualityRanking_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
```

无需改动 yolo11_src、原训练文件或 conda 环境。脚本明确使用 `/root/miniconda3/envs/yolo11/bin/python`。
必须存在原始 VisDrone train/val `annotations/`，不能用 converted YOLO labels 替代。

训练集由独立适配器读取：零宽/零高的 class-0 ignore 行保留原始数值，并沿用官方 raster-ignore 的 `max(1,round)` 查询语义；数量与行号记入 train metadata。不修改原始标注文件，不放宽前景框检查。负尺寸、非整数/非法类别、零尺寸非 class-0 框仍拒绝。验证集继续使用未修改的冻结 D27 解析和评分代码。

## 先独立门禁

```bash
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh check
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh smoke
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh verify-gates
```

check 包含真实权重加载/fuse、独立坐标标签、候选索引、梯度更新、状态重载、分数变换及 NMS/ignore 单元检查。
smoke 对**每个 seed**执行 8 张真实 train 图缓存、1 轮小 MLP 拟合、8 张真实 val 图标准推理/三路评分。
smoke 不是原 YOLO batch16 训练：基础检测器不训练，batch4 推理；缓存行 minibatch 最多 4096。
默认单开，每进程 6 GiB allocator 上限，另需至少 2 GiB 余量，不静默降配置。
正在运行别的任务时，确认至少 8 GiB 空闲后，可在以上每条和下面 run 命令中一致加 `--allow-busy-gpu`。
门禁凭据绑定代码、数据、原权重、设备、cap、seeds；中断的一次 smoke 不会删除之前有效凭据。

## 后台完整实验

```bash
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh run > launcher_d47_frozen_quality.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_d47_frozen_quality.log
```

这里 Ctrl+C 只退出 tail。不要在前台 check/smoke 中 Ctrl+C。
默认顺序：seed1 缓存→固定 10 轮拟合两种输入→官方评分，然后 seed2 同样流程。
报告会打印起止时间、图像缓存 ETA、拟合轮次和 loss/训练集 holdout 排序准确率。
缓存阶段会占主要时间；不能在实测速度之前保证小时数。自动关机关闭，没有远程同步和自动新实验。

结果在 `comparison_reports/D47_RUN_时间/REPORT.md` 和 `all_models.json`。
缓存在 `d47_candidate_cache/签名/`；拟合权重和 history 在 `runs/d47_frozen_quality/签名/`。
需要至少 8 GiB 空闲硬盘（两个 seed 的 FP32 训练候选缓存）。只下载报告和 fit 权重即可，不必下载候选缓存。

## 中断恢复/单独评分

```bash
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh cache
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh fit
bash /root/autodl-tmp/D47_E40_FrozenQualityRanking_AutoDL/run_d47.sh eval
```

完整缓存通过逐文件 SHA 验证后复用；不完整缓存重新产生。fit 从头拟合这 10 轮，不从 last 自动续轮；已完成基础模型无需再推理缓存。
每轮保存 last endpoint/optimizer，但 eval 只接受完整第 10 轮，拒绝 smoke/中途 checkpoint。
如果评分失败，eval 可直接补评分。不要改代码/配置后强行复用旧签名。

## 如何解读

优先看 full 对原 E40 的 Official AP/AP50/AP75，再看 full 对 DFL-only。
两 seed 各增加约 .20 pp 是研究投入筛选线，不是显著性标准/录用门槛；AP75 应同向，AP50不应明显牺牲。
只提高训练 holdout 排序准确率但不提高 AP，不批准新的 200 轮训练。
与 full 差不多的 DFL-only 收益不能宣称来自 E40 的双任务特征。
原始基线必须先逐图复现参考预测、完整 val 官方指标，否则实验失败而不是解释模型收益。
**严禁用 test-dev 调参或 GT-oracle 分数作为可实现收益。**

## 文献边界

质量估计/质量排序不是首次提出：GFLV2 (CVPR 2021)、Rank & Sort (ICCV 2021) 已有相关机制。
本实验不复制其损失实现，不宣称训练期根因、结构原创性、保证提升或论文一定录用。
成功后才讨论与 E40 的具体组合和正式训练协议；不得把现有 P2 收益冒充新增贡献。

- https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html
- https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html
