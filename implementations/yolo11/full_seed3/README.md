# E60 seed3 / 正式200轮 / 可选自动关机

仅将已完成 E60 seed1/2 的全局训练 seed 改为3及隔离输出路径。
模型实现14个冻结文件与两次 actual_source.zip 对齐：四尺度W96，只有P2/P3增加分组残差编码，
新增编码FP32计算/参数/EMA保护不变。不是E61，不从best微调，不训练E1/E40对照。

## 配置与流程

- 原始 weights/yolo11s.pt 迁移；640/batch16/workers8/seed3/200轮/patience0。
- 原SGD、cosine200、close_mosaic10、增强、TAL、损失、Detect塔及官方评分不变。
- 单开32GiB卡，固定28GiB allocator；不自动降batch，不复用seed1/2门禁。
- all：CUDA check → 真实batch16至少8次成功AMP更新、52组实际变化和EMA-half batch32全548图验证 → verify-gates → 正式训练 → 官方val评分及三seed汇总。
- 模块隔离初始化40001/57001/60001保持不变；全局seed3控制E1未迁移参数/训练随机过程。
- 时间参考：seed2正式训练约8小时51分；服务器门禁/评分另计，实际速度以当前日志为准。

## 必需的服务器原有文件

请保留下列正式目录及文件；若清理过先从本地恢复，不能跳过门禁。包中不含权重、数据或旧结果。

|comparison_reports 下目录|必需文件|
|---|---|
|e1_yolo11s_p2add_img640_seed3_20260926_172932_190728|metrics.json、area_ground_truth.json、area_predictions.json|
|E40S3_RUN_20261002_223525_596764|comparison.json、training.json、area_predictions.json|
|E60_RUN_20261010_133029_103161|comparison.json、training.json、actual_source.zip|
|E60S2_RUN_20261011_120724_723053|comparison.json、training.json、actual_source.zip|

runs 下对应 best.pt 也须保留：

```text
e1_yolo11s_p2add_img640_seed3/weights/best.pt
e40a_spatial_task_head_img640_seed3_200e/weights/best.pt
e60a_e58_group_residual_encoding_img640_seed1_200e/weights/best.pt
e60a_e58_group_residual_encoding_img640_seed2_200e/weights/best.pt
```

还需原 yolo11_src、weights/yolo11s.pt、VisDrone2019.yaml、原训练数据及官方val原标注。
同seed对照/原始预训练权重SHA、训练协议、官方图像映射、实际源码均检查。
E58仅有seed1/2，seed3不要求不存在的E58 seed3，也不拿其他seed冒充。

## 上传、安装及后台运行

上传ZIP到 /root/autodl-tmp。代码实际放进项目内 experiment_code，不是临时目录链接：

```bash
mkdir -p /root/autodl-tmp/yolo11_project/experiment_code
unzip -n /root/autodl-tmp/E60_Seed3_AutoShutdown_AutoDL.zip \
  -d /root/autodl-tmp/yolo11_project/experiment_code
cd /root/autodl-tmp/yolo11_project
nohup bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh \
  all --auto-shutdown > launcher_e60_seed3.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e60_seed3.log
```

只退出tail的Ctrl+C不影响后台任务。启动器固定使用yolo11环境Python，不需激活conda。
已有seed3 run目录时拒绝覆盖；不要重复启动。manifest改变将使门禁失效，不要边训练边改源码。

可先分别执行（单独门禁不关机），全部通过后再执行 run --auto-shutdown：

```bash
bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh check
bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh smoke
bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh verify-gates
nohup bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh \
  run --auto-shutdown > /root/autodl-tmp/yolo11_project/launcher_e60_seed3.log 2>&1 &
```

## 自动关机

只由 all/run --auto-shutdown 启用；成功或失败均先保存状态、确认无其他训练/评分进程且GPU释放，
再倒计时60秒、重新确认空闲并调用平台 /usr/bin/shutdown。
兼容无shebang的ASCII平台脚本，用 /bin/bash 执行；不杀其他任务、不用其他破坏性关机替代命令。
重复启动、准入失败、手动中断、KEEP_RUNNING标记或状态无法确认时不关机。
命令返回0只表示请求已返回，不保证平台已停止计费；请检查AutoDL控制台。
本地关机测试全部mock，不执行真实关机。

取消关机但保留训练：

```bash
touch /root/autodl-tmp/yolo11_project/KEEP_RUNNING_E60_SEED3
```

控制器索引 logs/latest_e60_seed3_controller.json 指向本次会话，含 state.json、shutdown_status.json及各阶段日志。

## 结果与评分恢复

- runs/e60a_e58_group_residual_encoding_img640_seed3_200e：best.pt、last.pt、results.csv。
- comparison_reports/E60S3_RUN_时间：training.json、training_completion.json、completion.json、actual_source.zip。
- comparison.json、official_metrics.json、REPORT.md：同seed E1/E40 官方AP/AP50/AP75/AR500。
- paired_seeds123_summary.json：E1/E40/E60三个seed均值、样本标准差、配对差值，保留负收益seed。
- E60相对E58仍明确只汇总两个配对seed，绝不比较不等数量的均值。
- area_predictions.json、gt_rows.json：预测与补充几何诊断，不是尺度AP/一对一召回。

若训练已完成仅评分失败，空闲时补评分，不重训：

```bash
nohup bash /root/autodl-tmp/yolo11_project/experiment_code/E60_Seed3_AutoShutdown_AutoDL/run.sh eval \
  --weights /root/autodl-tmp/yolo11_project/runs/e60a_e58_group_residual_encoding_img640_seed3_200e/weights/best.pt \
  > /root/autodl-tmp/yolo11_project/launcher_e60_seed3_eval.log 2>&1 &
```

eval不自动关机。本包不启动test-dev预测、不调参、不扩展新结构或第二数据集；三seed结果不等于显著性或发表保证。
