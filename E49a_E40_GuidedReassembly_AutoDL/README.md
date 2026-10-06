# E49a: E40 + Guided P3→P2 Reassembly

仅一项 seed1/640/batch16/200轮实验，单开。保留完整E40，新增重建在原P2 Concat之前。
设计、来源、边界和对照见 PLAN.md。不是E47/E48，也不是训练E40 best微调。

V2修复仅涉及FLOPs门禁和回归测试，说明见FIX_V2.md。若已解压旧包，
上传`E49a_E40_GuidedReassembly_FixV2_AutoDL.zip`后执行：

```bash
unzip -o /root/autodl-tmp/E49a_E40_GuidedReassembly_FixV2_AutoDL.zip -d /root/autodl-tmp
```

核心前向/损失/训练代码不变；当前代码签名变化，仍需重跑以下独立门禁。

## 上传并解压

将 `E49a_E40_GuidedReassembly_AutoDL.zip` 上传 `/root/autodl-tmp/`，在服务器执行：

```bash
cd /root/autodl-tmp
unzip E49a_E40_GuidedReassembly_AutoDL.zip
cd /root/autodl-tmp/yolo11_project
```

先等当前训练正常结束、释放GPU。不要杀现有任务。
Python绝对路径固定到yolo11环境；无需先conda activate。
不改共享Ultralytics源码，不自动关机，不下载新的权重或数据。

## 独立门禁

```bash
bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh check
bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh smoke
bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh verify-gates
```

分别看到 CHECK PASSED、CUDA SMOKE PASSED、RECEIPTS VALID 后再运行。
smoke是最多32真实batch/至少8次成功更新＋完整验证，不是20/90轮精度probe。
进度逐batch打印；AMP开始几次overflow_skip属于记录的缩放尝试，不自动判通过。
check/smoke失败立即停，不绕门禁；代码/数据/环境签名变化须重跑门禁。

## 正式200轮

```bash
nohup bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh run > /root/autodl-tmp/yolo11_project/launcher_e49.log 2>&1 &
tail -f /root/autodl-tmp/yolo11_project/launcher_e49.log
```

仅对上述 `tail -f` 按Ctrl+C是退出日志查看；直接前台check/smoke按Ctrl+C会终止门禁。
nohup后台正式任务不依赖VSCode保持开启。服务器关机仍会终止任务。
训练日志原生逐batch显示，额外每轮打印起止/epoch_seconds/ETA和重建gain。
同时记录每轮首个真实batch的输入/实际输出变化RMS比例，不用“参数非零”冒充路径起效。
总耗时以运行初期实际速度估计，不保证完成时刻。

可选完整链条（尚未check/smoke时使用；任一失败即停）：

```bash
nohup bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh all > /root/autodl-tmp/yolo11_project/launcher_e49_all.log 2>&1 &
```

不是自动接续E48的监控队列。当前任务未释放GPU时smoke/run会明确拒绝启动。

## 输出与补评分

权重和曲线：`yolo11_project/runs/e49a_e40_guided_reassembly_img640_seed1_200e/`。
报告：`yolo11_project/comparison_reports/E49_RUN_时间/`。
包括 training.json、transfer.json、completion.json、official_metrics.json、
comparison.json、area_predictions.json、gt_rows.json、REPORT.md。
官方AP/AP50/AP75/AR500、逐类别结果、尺寸几何，与E1和E40分别比较。
只有200轮和官方评分都完成才写completion.json；训练完成≠评分完成。

若训练完成但评分异常，权重保留，不重训，执行：

```bash
bash /root/autodl-tmp/E49a_E40_GuidedReassembly_AutoDL/run_e49.sh eval --weights /root/autodl-tmp/yolo11_project/runs/e49a_e40_guided_reassembly_img640_seed1_200e/weights/best.pt
```

自定义模型的预测/重载请使用本包入口；不要将包外同名spatial_task_head.py混入PYTHONPATH。
不允许同一路径重复正式训练或静默覆盖已有run。
