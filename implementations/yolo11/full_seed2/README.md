# E60 seed2 收尾复验

冻结已完成的 E60 seed1 设计，只改全局训练 seed=2；不是 E61，也不微调 E60 best.pt。
P2/P3 保留三层读出副本分组残差编码，P4/P5 保留 E58 四尺度 W96 模块而不增加该编码。
新增编码的 FP32 参数、内部计算及 EMA/权重保存保护逐字保留。
新增支路的隔离初始化种子保持不变；E1 中非预训练迁移参数和数据增强按全局 seed2 初始化。

## 固定配置

- 原始 yolo11s.pt 初始化；640 / batch16 / workers8 / seed2 / 200 epochs / patience0。
- 原 SGD、余弦 200 轮日程、close_mosaic10、增强、Detect tower、DFL、损失、TAL、NMS 不变。
- 独占 32GiB GPU，固定 28GiB PyTorch allocator 上限，不自动缩 batch，不并行。
- CUDA check → 真实 smoke → verify-gates → 正式 200 轮 → 官方 val 评分。
- smoke 必须至少 8 次成功 AMP 更新且全部 52 组新增参数实际变化，并覆盖 EMA-half batch32 全 548 图验证。
- 预计正式训练约 9–11 小时，检查/评分另需时间；以当前 GPU 的实际日志为准，不保证截止时间。

## 服务器需要保留的文件

以下已有对照只读复用，不重训；缺少时先恢复本地备份，不能跳过协议门禁。
除各自 best.pt 外，需要下列 comparison_reports 目录中的原文件：

| 对照 | 报告目录 | 必需文件 |
|---|---|---|
| E1 seed2 | e1_yolo11s_p2add_img640_seed2_20260926_172907_911324 | metrics.json、area_ground_truth.json、area_predictions.json |
| E40 seed2 | E40S2_RUN_20260930_130808_314014 | comparison.json、training.json、area_predictions.json |
| E58 seed2 | E58S2_RUN_20261008_162925_565240 | comparison.json、training.json、area_predictions.json |
| E60 seed1 | E60_RUN_20261010_133029_103161 | comparison.json、training.json、actual_source.zip |

best.pt 路径（均在 yolo11_project/runs 下）：

```text
e1_yolo11s_p2add_img640_seed2/weights/best.pt
e40a_spatial_task_head_img640_seed2_200e/weights/best.pt
e58a_e57_allscale_w96_img640_seed2_200e/weights/best.pt
e60a_e58_group_residual_encoding_img640_seed1_200e/weights/best.pt
```

还需要原 yolo11_src、weights/yolo11s.pt、原 VisDrone2019.yaml、训练集及官方 val 原标注。
训练输入及数据 YAML 与 E1 seed2 对齐，预训练和对照权重 SHA 校验；E60 seed1 actual_source.zip 中
14 个冻结文件与本包逐字核对。没有复制服务器数据或权重进 ZIP。

## 上传解压并运行

把 ZIP 上传到 /root/autodl-tmp，然后执行：

```bash
cd /root/autodl-tmp
unzip -n E60_Seed2_AutoShutdown_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh all --auto-shutdown > launcher_e60_seed2.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e60_seed2.log
```

这里只退出 tail 的 Ctrl+C 不影响 nohup 后台训练。不要直接关闭正在计费的 AutoDL 实例。
使用固定 /root/miniconda3/envs/yolo11/bin/python，无需当前终端激活 conda。
该命令包含 check 和真实 smoke，不能跳过。若需要先单独查看，可执行：

```bash
bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh check
bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh smoke
bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh verify-gates
nohup bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh run --auto-shutdown > launcher_e60_seed2.log 2>&1 &
```

all 会重新做门禁；run 复核已保存门禁再训练。代码/配置/环境或对照变更会使旧门禁失效。
已有 seed2 输出目录时拒绝覆盖，不能用重复启动补救故障。

## 输出和评分恢复

权重及曲线：runs/e60a_e58_group_residual_encoding_img640_seed2_200e/。
正式报告：comparison_reports/E60S2_RUN_时间/。

- training.json / training_completion.json / completion.json：区分训练结束与评分结束。
- comparison.json / official_metrics.json / REPORT.md：配对 E1/E40/E58 seed2 官方 AP/AP50/AP75/AR500。
- paired_seeds12_summary.json：E60/E1/E40/E58 seeds1/2 数值、均值、样本标准差与配对提升，保留不利种子。
- area_predictions.json / gt_rows.json：完整 val 预测和候选覆盖诊断；覆盖不等于尺度 AP 或一对一召回。
- actual_source.zip：此次实际训练源码。

默认不关机，上述 --auto-shutdown 明确启用。训练及评分成功或失败均先保存状态，确认没有其他任务、
GPU 已释放后等待 60 秒请求关机；取消任务、重复启动、准入失败或状态不可验证不执行关机。
兼容 AutoDL 无 shebang 的 ASCII /usr/bin/shutdown，但返回 0 不代表已确认平台停止计费。
务必检查 AutoDL 控制台；关机失败记录在控制器目录 shutdown_status.json。
控制器索引：logs/latest_e60_seed2_controller.json。

取消自动关机但保留训练：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_E60_SEED2
```

若只有评分失败，不重训，服务器空闲时执行：

```bash
nohup bash /root/autodl-tmp/E60_Seed2_AutoShutdown_AutoDL/run.sh eval \
  --weights /root/autodl-tmp/yolo11_project/runs/e60a_e58_group_residual_encoding_img640_seed2_200e/weights/best.pt \
  > /root/autodl-tmp/yolo11_project/launcher_e60_seed2_eval.log 2>&1 &
```

eval 不自动关机。两 seed 只构成初步重复性证据，不宣称统计显著或保证发表。
本包不自动添加新结构、test-dev 筛选、seed3 或第二数据集训练。
