# D46 E1/E40 实例竞争定向审计

只读审计，不训练、不改共享源码/权重、不关机、不读取 test-dev。
默认检查 E1/E40 的 seed1、seed2 四个已有 val-best，640/FP32/batch4/conf0.001/NMS0.7/max_det500。
所有模型依次运行，不是并行训练；原始 checkpoint SHA 固定。

## 上传与运行

将整个目录放在 `/root/autodl-tmp/D46_E1E40_InstanceCompetition_AutoDL/`，目录不能少一层或多一层。

```bash
cd /root/autodl-tmp/yolo11_project
bash /root/autodl-tmp/D46_E1E40_InstanceCompetition_AutoDL/run_d46.sh check
bash /root/autodl-tmp/D46_E1E40_InstanceCompetition_AutoDL/run_d46.sh smoke
nohup bash /root/autodl-tmp/D46_E1E40_InstanceCompetition_AutoDL/run_d46.sh run > launcher_d46_instance_audit.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_d46_instance_audit.log
```

`Ctrl+C` 只退出后台任务的 `tail`。不要在前台 check/smoke 时按 Ctrl+C，那里会中断检查。
也可 `nohup bash .../run_d46.sh all > launcher_d46_instance_audit.log 2>&1 &` 自动串联门禁，失败即停。
每四张图打印模型、进度、耗时与 ETA。无自动关机。完成结果在 `comparison_reports/D46_RUN_时间/REPORT.md`。

默认 allocator 上限4 GiB；GPU忙时默认拒绝。明确允许共享GPU，可在 **check、smoke、run三次命令都** 追加 `--allow-busy-gpu`；需要至少 cap+2 GiB 空闲。
上限只覆盖PyTorch分配器，不能保证不影响其他任务速度或绝无OOM。不会自动降batch/精度。
要加seed3，三个阶段都追加 `--seeds 1 2 3`；不要只在正式命令改seed。

## 审计内容

1. 每张图保留raw候选ID；比较逐GT独立覆盖与**精确最大基数一对一匹配**。匹配不使用置信度优化，因此是几何上界，不是可实现AP。
2. raw所有框都参加几何检查；另外分别分析conf/max_nms/NMS/top500。不能用置信度截断假装raw框不存在。
3. 分类别、尺寸、密度记录同类最佳候选的中心偏差、宽高偏差及朝最近同类GT的投影。候选按几何选择，存在选择偏差，不是训练期因果证明。
   同时提供同图/同类/2像素尺寸段/密度段内部100次方向置换的描述性对照；单样本层无法置换，必须报告覆盖率，不能强行解释。
4. NMS调用仍是项目原torchvision算法；候选ID作为mask列携带，明确`nc=10`。必须与普通Predictor结果逐项相等。
5. 为每个被NMS去掉的高IoU候选重建**第一个保留重叠见证框**，记录同实例/不同实例/无归属，以及目标是否仍有其他好框。
6. 分数相同、阈值边界和无法重建的见证标记为不确定，不能宣称唯一真实suppressor。
7. 先逐图复现原始预测，再用固定D27官方ignore评分复现原正式AP/AP50/AP75/AR500。任何漂移会失败，不生成“成功审计”。

## 输出

- `manifest.json`：代码/权重/参考预测/图像/标注指纹与协议。
- `<模型>/gt_rows.json`：中心与尺度误差、近邻方向、各阶段覆盖。
- `<模型>/nms_events.json`：被抑制好候选及见证框，是否有存活替代候选。
- `<模型>/image_stats.json`：逐图精确一对一匹配与候选独立覆盖。
- `<模型>/summary.json`：官方复现、分层统计及明确边界。
- `all_models.json / REPORT.md / decision.json / status.json`：整体结果。

“跨实例抑制事件”不是净可追回GT，更不是AP增益。一对一匹配的具体GT身份在多解时不唯一；不要把某次匹配未选中的GT当确定失败样本。
当前代码不据正负号/任意相关阈值自动批准训练。需要结合配对seed、类别/尺寸分层和错误对证据再决定。

## 本地测试

```bash
python audit.py self-test --device cpu --source /path/to/yolo11_src --project /path/to/yolo11_project
```

包括全部512个3×3二分图与独立穷举oracle、重复GT、错误类别、中心方向、NMS身份、跨类别、分数平局与max_nms/top500预算。
本地CPU/小CUDA测试不是服务器548图真实复现或显存门禁；正式结论必须等服务器四模型全量通过。
官方评分及 E40 原模块为项目已使用版本原样复制，不重新实现评分。
